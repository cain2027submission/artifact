#!/usr/bin/env python3
"""RQ3 third arm: answer the same questions from an OUTPUT-STRIPPED TRANSCRIPT.

Two arms existed before this one. The rollup recovers 68.5% of ground truth and
the full raw transcript 95.5% over the same 89 questions, and that gap invites a
pivot: if the transcript is the better memory, keep the transcript. But a raw
`.jsonl` is not a storable artifact -- one minified line per content block, tool
payloads inline, 0.7 MB median, and `git log -p` unreadable. The obvious middle
is to strip the mechanical I/O and keep the rest, which `rq6_stripped_size.py`
prices at 12.3x a rollup and 17x smaller than the raw log.

What nobody has measured is whether the output-stripped form RETAINS anything. The
95.5% belongs to the full transcript. Stripping removes tool results, which are
73.98% of transcript volume, and the `artifact_detail` and `outcome` question
classes plausibly have their ground truth sitting inside exactly that. So the
output-stripped arm's true retention is somewhere in [68.5, 95.5] and citing the
top of that range would be an unforced error of the same shape as discounting one
arm's refusals but not the other's.

This script measures it. It holds the questions, the ground truth, the judge
model, the grading prompt and the grade parser fixed, and changes exactly one
thing: the context. Two consequences worth stating up front:

  * The answering prompt differs from the raw arm's in the clause that names the
    context, and in nothing else. Calling a tool-stripped rendering "the raw
    transcript" would be a false statement to the answerer, which is a worse
    confound than an honest one-clause difference. This mirrors how
    `rq3_uniform_rollup_arm.py` handles the same problem.
  * Output-stripping fits sessions the raw arm had to drop. The raw arm excluded 5
    sessions for overflowing a 1.18M-token window; all 5 fit output-stripped, the
    largest going 21.2 MB -> 953k tokens. So this arm reports on TWO sets: the
    89 shared with the other two arms, for the three-way paired contrast, and
    its own wider reach, which is a result about coverage rather than accuracy.

Construction: strip_transcript() output rendered as markdown, not JSON. Keeps authored
user prose (harness injections removed via split_meta), assistant prose, non-empty
reasoning text under ### REASONING headings, and tool calls under ### TOOL CALL: <name>
with inputs capped at MAX_TOOL_INPUT_CHARS = 2000 and truncation marked inline. Drops
all tool_result content (counted, not rendered), empty thinking blocks (via continue at
line 121, so base64 signatures cannot appear), and all JSON scaffolding. Emits a header
stating message count, tool-call count and how many results were withheld. This is
output-stripped, not raw-minus-results -- thinking signatures are 0.00% of this artifact
because empty-text blocks are skipped, measured across 3,659 of 3,666 thinking blocks.

Usage:
    rq3_stripped_transcript_arm.py --dry-run        # resolve + size report only
    rq3_stripped_transcript_arm.py --limit 3        # smoke test
    rq3_stripped_transcript_arm.py                  # full run (resumable)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, call_llm, extract_confidence, grade_answer,
    index_transcripts, load_records, rates, record_id, resolve,
)
from rq1_signal_composition_corrected import split_meta  # noqa: E402

OUT = DATA / "rq3_questions_stripped_arm.jsonl"
REPORT = DATA / "rq3_stripped_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

# The raw arm's own gate, reused so "fits" means the same thing in both arms.
CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 2.45

# A tool input can itself be a whole file. Capping it keeps one Write call from
# reintroducing the payload volume stripping is meant to remove. Charged, not
# silent: the truncation is marked in the rendering the answerer sees.
MAX_TOOL_INPUT_CHARS = 2_000


def strip_transcript(raw: str) -> Tuple[str, dict]:
    """Render a transcript as L2: prose + tool intents, no tool results.

    Returns (markdown, stats). The block-walking mirrors rq1's `scan()` so the
    categories here mean what they mean in RQ1 rather than being a second,
    slightly different definition of the same words.
    """
    out: List[str] = []
    stats = {"entries": 0, "tool_calls": 0, "dropped_results": 0,
             "thinking_blocks": 0, "thinking_empty": 0, "truncated_inputs": 0}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        stats["entries"] += 1
        role = msg.get("role")
        is_meta = bool(rec.get("isMeta"))
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [content]

        for b in blocks:
            if isinstance(b, str):
                btype, text = "text", b
            elif isinstance(b, dict):
                btype = b.get("type", "other")
            else:
                continue

            if btype in ("thinking", "redacted_thinking"):
                text = (b.get("thinking") or b.get("text") or "")
                stats["thinking_blocks"] += 1
                if not text.strip():
                    # Kept as a counted absence rather than rendered. The empty
                    # block is RQ1's finding, but pasting `[thinking]` with no
                    # text into the answerer's context would spend tokens on
                    # nothing -- which is the bug rollup_summarise.py has.
                    stats["thinking_empty"] += 1
                    continue
                out.append(f"\n### REASONING\n{text.strip()}\n")

            elif btype == "tool_use":
                stats["tool_calls"] += 1
                name = b.get("name") or "tool"
                inp = b.get("input")
                s = inp if isinstance(inp, str) else json.dumps(inp, indent=None)
                if s and len(s) > MAX_TOOL_INPUT_CHARS:
                    s = s[:MAX_TOOL_INPUT_CHARS] + f"\n[... {len(s)-MAX_TOOL_INPUT_CHARS} chars of tool input truncated]"
                    stats["truncated_inputs"] += 1
                out.append(f"\n### TOOL CALL: {name}\n```\n{s}\n```\n")

            elif btype == "tool_result":
                # The whole point of the strip level. Counted so the rendering
                # can tell the answerer how much was withheld, rather than
                # letting it assume the session simply ran no tools.
                stats["dropped_results"] += 1

            elif btype == "text":
                if isinstance(b, dict):
                    text = b.get("text") or ""
                if not text.strip():
                    continue
                if role == "assistant":
                    out.append(f"\n### ASSISTANT\n{text.strip()}\n")
                elif role == "user":
                    if is_meta:
                        continue
                    authored, _inj = split_meta(text)
                    if authored.strip():
                        out.append(f"\n### USER\n{authored.strip()}\n")

    header = (f"[This is an agent session transcript with tool OUTPUTS removed. "
              f"{stats['entries']} messages, {stats['tool_calls']} tool calls "
              f"shown with their inputs, {stats['dropped_results']} tool results "
              f"withheld.]\n")
    return header + "".join(out), stats


def answer_from_stripped(question: str, stripped: str, return_raw: bool = False):
    """The raw arm's prompt, differing only in the clause naming the context.

    Returns (answer, confidence, usage), or (answer, confidence, usage, reply)
    when `return_raw` is set. The kwarg defaults off so every existing caller and
    test_confidence_extraction.py, which unpacks a 3-tuple, is untouched. See the
    matching docstring in rq3_action_record_arm.py for why the reply is kept.
    """
    prompt = f"""Someone asks: "{question}"

Below is the transcript of an agent session, with tool outputs removed. Answer the question based ONLY on this transcript. If the transcript does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

TRANSCRIPT:
{stripped}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        # Never NOT_ANSWERABLE: that grades as `absent`, which is the outcome
        # under measurement, so an API failure would masquerade as a finding.
        return ("", "API_ERROR", usage, response) if return_raw \
            else ("", "API_ERROR", usage)

    answer, confidence = extract_confidence(response)
    return (answer, confidence, usage, response) if return_raw \
        else (answer, confidence, usage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    records = [r for r in load_records(QUESTIONS) if record_id(r) in graded]
    by_prefix = index_transcripts()

    eligible, unresolved = [], []
    for r in records:
        m = SLUG.match(r.get("session_id", ""))
        if not m:
            unresolved.append(r.get("session_id", "")[:8])
            continue
        prefix = m.group(1)
        p = resolve(prefix, by_prefix)
        if p is None:
            unresolved.append(prefix)
            continue
        eligible.append(dict(r, _transcript=str(p), _prefix=prefix,
                             _bytes=p.stat().st_size))

    print(f"graded {len(records)}  resolved {len(eligible)}  "
          f"unresolved {len(set(unresolved))}")

    if args.dry_run:
        # Strip every distinct transcript once and report what it costs. This is
        # the check that the renderer works before any money is spent, and it is
        # also the honest place to see whether stripping really does fit the
        # sessions the raw arm dropped.
        seen, rows = set(), []
        for r in eligible:
            if r["_transcript"] in seen:
                continue
            seen.add(r["_transcript"])
            text, st = strip_transcript(
                Path(r["_transcript"]).read_text(errors="replace"))
            raw_fits = r["_bytes"] / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
            str_fits = len(text) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
            rows.append((r["_prefix"], r["_bytes"], len(text), st, raw_fits, str_fits))
        rows.sort(key=lambda x: -x[1])
        print(f"\n{'sess':10s} {'raw MB':>8s} {'strip MB':>9s} {'ratio':>6s} "
              f"{'calls':>6s} {'drop':>6s} {'think':>6s} {'raw fit':>8s} {'str fit':>8s}")
        for pre, rb, sb, st, rf, sf in rows:
            print(f"{pre:10s} {rb/1e6:8.2f} {sb/1e6:9.3f} {rb/max(sb,1):5.1f}x "
                  f"{st['tool_calls']:6d} {st['dropped_results']:6d} "
                  f"{st['thinking_blocks']-st['thinking_empty']:6d} "
                  f"{'yes' if rf else 'NO':>8s} {'yes' if sf else 'NO':>8s}")
        gained = [r for r in rows if not r[4] and r[5]]
        print(f"\nsessions the raw arm could not fit but stripped can: {len(gained)}"
              + (f" ({', '.join(g[0] for g in gained)})" if gained else ""))
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
    # Group by session so one stripped rendering serves consecutive calls and
    # the gateway's prompt cache can hit.
    todo.sort(key=lambda r: (r["_prefix"], r.get("class", "")))
    print(f"to answer: {len(todo)}\n")

    cache: Tuple[Optional[str], str, dict] = (None, "", {})
    errors: List[dict] = []
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        if cache[0] != r["_transcript"]:
            text, st = strip_transcript(
                Path(r["_transcript"]).read_text(errors="replace"))
            cache = (r["_transcript"], text, st)
        stripped, st = cache[1], cache[2]

        if len(stripped) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP stripped still too big "
                  f"({len(stripped)/1e6:.2f} MB)")
            continue

        answer, confidence, usage, raw_reply = answer_from_stripped(
            r["question"], stripped, return_raw=True)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class'):15s} "
              f"{len(stripped)/1e6:5.2f}MB {confidence:18s} {grade:18s} "
              f"(rollup: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "transcript": r["_transcript"], "raw_bytes": r["_bytes"],
               "stripped_chars": len(stripped), "strip_stats": st,
               "answer": answer, "rating": confidence, "grade": grade,
               # Additive: `answer` is the post-parse extraction and is unchanged.
               "raw_reply": raw_reply,
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
        "strip_level": ("output-stripped transcript: authored user prose, assistant prose, "
                        "non-empty reasoning text, and every tool call's input capped at "
                        f"{MAX_TOOL_INPUT_CHARS} chars; tool RESULTS dropped, empty thinking "
                        "blocks skipped so base64 signatures cannot appear (0.00% measured)"),
        "prompt_delta_vs_raw_arm": ("one clause names the context as a transcript "
                                    "with tool outputs removed; nothing else differs"),
        "n_api_errors_this_run": len(errors),
        "stripped_arm": rates(rows),
        "median_stripped_chars": sorted(r["stripped_chars"] for r in rows)[len(rows)//2] if rows else None,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    d = rates(rows)
    h = d["hallucination_of_attempted"]
    print(f"\noutput-stripped transcript arm n={d['n']} correct {d['correct']['n']} "
          f"partial {d['partially_correct']['n']} wrong {d['wrong']['n']} "
          f"absent {d['absent']['n']}")
    print(f"  recoverable {d['recoverable']['pct']}% {d['recoverable']['ci95']}  "
          f"wrong-of-attempted {h['pct']}% {h['ci95']}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
