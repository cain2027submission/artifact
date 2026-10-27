#!/usr/bin/env python3
"""RQ3 fourth arm: answer the same questions from a DIALOGUE RECORD.

Three arms existed before this one. The rollup recovers 68.5%, the full raw
transcript 95.5%, and the output-stripped transcript (tool inputs kept, results dropped)
87.6% over the same 89 questions. The dialogue record sits between rollup and
output-stripped: it is generated during rollup creation (level 2), carries the turn
structure with tool calls inlined, keeps reasoning blocks, and drops tool results
the same way the output-stripped arm does. Unlike the output-stripped arm, it is a durable
artifact that is already written and stored rather than one computed on demand.

This script measures whether the dialogue record form retains what the output-stripped form
does. It holds the questions, the ground truth, the judge model, the grading
prompt and the grade parser fixed, and changes exactly one thing: the context.

Construction: rollup_verbatim.py output. Keeps human prompts, the assistant's LAST prose
answer per exchange, and a tool-call bullet index `- \`Tool\` <arg>` truncated to
TOOL_ARG_CHARS = 120. Drops all reasoning and ALL tool results. Subagent prompts/answers
only behind a `--subagents` flag, off by default.

The answering prompt differs from the output-stripped arm's only in the clause that
names the context. Calling a dialogue record "a transcript with tool outputs
removed" would be a false statement to the answerer, which is a worse confound
than an honest one-clause difference.

Dialogue records (the file on disk is verbatim.md) come from the trace archive,
resolved via the 8-hex prefix in the session slug. All 25 RQ3 graded sessions
resolve with zero collisions, verified by `verbatim_arm_coverage.py`.

Usage:
    rq3_verbatim_arm.py --dry-run        # resolve + size report only
    rq3_verbatim_arm.py --limit 3        # smoke test
    rq3_verbatim_arm.py                  # full run (resumable)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import TRACES_DIR  # noqa: E402
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, call_llm, extract_confidence, grade_answer,
    index_transcripts, load_records, rates, record_id, resolve,
)

OUT = DATA / "rq3_verbatim_arm.jsonl"
REPORT = DATA / "rq3_verbatim_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

TRACES = TRACES_DIR

# The raw arm's gate, reused so "fits" means the same thing.
CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 2.45


def resolve_verbatim(prefix: str) -> Optional[Path]:
    """Find verbatim.md for the 8-hex session prefix, matching verbatim_arm_coverage.py."""
    if not TRACES.is_dir():
        return None
    for d in TRACES.iterdir():
        if not d.is_dir():
            continue
        # Match the 8-hex prefix anywhere in the directory name
        if prefix.lower() in d.name.lower():
            verbatim = d / "verbatim.md"
            if verbatim.exists():
                return verbatim
    return None


def answer_from_verbatim(question: str, verbatim: str) -> Tuple[str, str, dict]:
    """The stripped arm's prompt, differing only in the clause naming the context."""
    prompt = f"""Someone asks: "{question}"

Below is a verbatim session record. Answer the question based ONLY on this record. If the record does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

SESSION RECORD:
{verbatim}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        # Never NOT_ANSWERABLE: that grades as `absent`, which is the outcome
        # under measurement, so an API failure would masquerade as a finding.
        return "", "API_ERROR", usage

    answer, confidence = extract_confidence(response)
    return answer, confidence, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    records = [r for r in load_records(QUESTIONS) if record_id(r) in graded]

    eligible, unresolved = [], []
    for r in records:
        m = SLUG.match(r.get("session_id", ""))
        if not m:
            unresolved.append(r.get("session_id", "")[:8])
            continue
        prefix = m.group(1)
        p = resolve_verbatim(prefix)
        if p is None:
            unresolved.append(prefix)
            continue
        eligible.append(dict(r, _verbatim=str(p), _prefix=prefix,
                             _bytes=p.stat().st_size))

    print(f"graded {len(records)}  resolved {len(eligible)}  "
          f"unresolved {len(set(unresolved))}")

    if args.dry_run:
        seen, rows = set(), []
        for r in eligible:
            if r["_verbatim"] in seen:
                continue
            seen.add(r["_verbatim"])
            text = Path(r["_verbatim"]).read_text(errors="replace")
            fits = len(text) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
            rows.append((r["_prefix"], r["_bytes"], len(text), fits))
        rows.sort(key=lambda x: -x[1])
        print(f"\n{'sess':10s} {'bytes':>10s} {'chars':>10s} {'ratio':>6s} {'fits':>6s}")
        for pre, b, c, f in rows:
            print(f"{pre:10s} {b:10d} {c:10d} {c/max(b,1):5.2f}x "
                  f"{'yes' if f else 'NO':>6s}")
        print(f"\ntotal verbatim records: {len(rows)}")
        print("\n--dry-run: no API calls made")
        return

    done = {}
    if OUT.exists():
        for r in load_records(OUT):
            done[r.get("record_id")] = r
        print(f"resuming: {len(done)} already answered")

    todo = [r for r in eligible if record_id(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    # Group by session so one verbatim read serves consecutive calls and
    # the gateway's prompt cache can hit.
    todo.sort(key=lambda r: (r["_prefix"], r.get("class", "")))
    print(f"to answer: {len(todo)}\n")

    cache: Tuple[Optional[str], str] = (None, "")
    errors: List[dict] = []
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        if cache[0] != r["_verbatim"]:
            text = Path(r["_verbatim"]).read_text(errors="replace")
            cache = (r["_verbatim"], text)
        verbatim = cache[1]

        if len(verbatim) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP verbatim too big "
                  f"({len(verbatim)/1e6:.2f} MB)")
            continue

        answer, confidence, usage = answer_from_verbatim(r["question"], verbatim)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class'):15s} "
              f"{len(verbatim)/1e6:5.2f}MB {confidence:18s} {grade:18s} "
              f"(rollup: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "verbatim_path": r["_verbatim"], "verbatim_bytes": r["_bytes"],
               "verbatim_chars": len(verbatim),
               "answer": answer, "rating": confidence, "grade": grade,
               "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    print(f"\nanswered {len(todo)-len(errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors)")

    rows = list(done.values())
    report = {
        "n_questions": len(rows),
        "n_sessions": len({r["session_id"] for r in rows}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "source": "dialogue record (verbatim.md) from the trace archive",
        "construction": ("rollup_verbatim.py output: human prompts, assistant's last prose "
                         "answer per exchange, tool-call bullet index truncated to 120 chars; "
                         "drops all reasoning and tool results"),
        "prompt_delta_vs_output_stripped_arm": ("one clause names the context as a "
                                                "dialogue record; nothing else differs"),
        "n_api_errors_this_run": len(errors),
        "verbatim_arm": rates(rows),
        "median_verbatim_chars": sorted(r["verbatim_chars"] for r in rows)[len(rows)//2] if rows else None,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    d = rates(rows)
    h = d["hallucination_of_attempted"]
    print(f"\ndialogue record arm n={d['n']} correct {d['correct']['n']} "
          f"partial {d['partially_correct']['n']} wrong {d['wrong']['n']} "
          f"absent {d['absent']['n']}")
    print(f"  recoverable {d['recoverable']['pct']}% {d['recoverable']['ci95']}  "
          f"wrong-of-attempted {h['pct']}% {h['ci95']}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
