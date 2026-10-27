#!/usr/bin/env python3
"""SPIKE: a rollup written from the OUTPUT-STRIPPED transcript ("rollup with tool calls").

Question: does giving the production summary prompt the tool-call inputs (which
the paper's stripped arm keeps and the production rollup mostly drops) produce a
rollup that retains more of the 124 RQ3 questions' ground truth?

This file builds step 1 and 2 only:
  1. for each of the 25 graded sessions, build the output-stripped transcript
     with strip_transcript() from rq3_stripped_transcript_arm.py (prose, reasoning,
     tool inputs capped at 2000 chars, tool results removed);
  2. turn it into an Epoch and build the production summary prompt with the real
     build_summary_prompt() from agent-session-rollup.
Step 3 (grade with the existing grader on the same 124 questions and compare to
the production rollup) is deliberately NOT implemented here.

Modes:
  --dry-run   (default ON) no model calls. The only network traffic is one GET to
              the gateway /health. Prints a per-session table and cost estimates.
  --generate  the only mode that calls the model. Writes
              data/paper-rollups/rq3_rollup_from_stripped_spike.jsonl keyed on
              session id. Implemented, never run as part of the spike's authoring.
  --limit N   first N sessions only.

Token estimate: the budget ladder does not tokenize at all (it sizes by chars),
so this uses len(chars) / 2.7 chars-per-token, as the brief specified. The raw
arm's measured figure for dense JSONL is 2.45 bytes/token; prose runs higher.
"""

import argparse
import json
import re
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROLLUP_SRC_DIR))

from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, GATEWAY, QUESTIONS, SUMMARY, SLUG, index_transcripts, load_records,
    record_id, resolve,
)
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402
from rollup_types import Epoch, Turn  # noqa: E402
import rollup_summarise as rs  # noqa: E402

OUT = DATA / "rq3_rollup_from_stripped_spike.jsonl"

CHARS_PER_TOKEN = 2.7
PRICE_IN = 3.00 / 1_000_000    # rq5_generation_cost.py PRICE_INPUT
PRICE_OUT = 15.00 / 1_000_000  # rq5_generation_cost.py PRICE_OUTPUT
GEN_OUT_TOKENS = 1_000
GRADE_OVERHEAD_TOKENS = 440    # ladder's measured prompt overhead beyond the artifact
GRADE_OUT_TOKENS = 150
N_QUESTIONS = 124

SECTION = re.compile(r"^### (USER|ASSISTANT|REASONING|TOOL CALL: .*)$", re.M)


def est_tokens(chars: int) -> int:
    return round(chars / CHARS_PER_TOKEN)


def stripped_to_epoch(stripped: str) -> Epoch:
    """Split strip_transcript()'s markdown back into Turns, in order.

    Roles map onto rollup_types.ROLES: USER->user, ASSISTANT->assistant,
    REASONING->thinking, TOOL CALL->tool. A tool turn's text is the tool name
    plus its (capped) input, which is the one thing the production reader omits.
    Line numbers are the section index, enough for the head/tail trimmer.
    """
    parts = SECTION.split(stripped)
    turns = []
    for i in range(1, len(parts) - 1, 2):
        head, body = parts[i], parts[i + 1].strip()
        if head.startswith("TOOL CALL: "):
            role, text = "tool", head[len("TOOL CALL: "):] + "\n" + body
        else:
            role = {"USER": "user", "ASSISTANT": "assistant",
                    "REASONING": "thinking"}[head]
            text = body
        turns.append(Turn(line=len(turns) + 1, role=role, text=text))
    return Epoch("spike", turns)


DOUBLE = False       # set by --double-length
MAX_TOKENS = 1024    # set by --max-tokens
VARIANT = "stripped"  # set by --variant: stripped | prod

_LEN_RE = re.compile(r"\((\d)-(\d) sentences\)")


def double_lengths(prompt: str) -> str:
    """Double ONLY the length numbers in the instruction part (before 'Transcript:').
    '(1-2 sentences)' -> '(2-4 sentences)', '(2-4 sentences)' -> '(4-8 sentences)'."""
    head, sep, tail = prompt.partition("\n\nTranscript:\n")
    head = _LEN_RE.sub(lambda m: f"({int(m.group(1)) * 2}-{int(m.group(2)) * 2} sentences)", head)
    return head + sep + tail


def build_prompts(epoch: Epoch) -> list:
    """The prompt(s) the production summarise() would send: one, or one per chunk
    (plus a reduce call, not built here) when the epoch exceeds the budget."""
    budget = rs.HAIKU_WINDOW - 2000
    chunks = rs.chunk_epoch(epoch, budget)
    prompts = [rs.build_summary_prompt(c, budget) for c in chunks]
    return [double_lengths(p) for p in prompts] if DOUBLE else prompts


def prod_epoch(path: Path) -> Epoch:
    """The production summariser's input: rollup_read.read_turns (parent prose,
    compaction summaries, tool-name+path stubs, no tool results), empty turns
    dropped as the drain does, as ONE epoch (no day/gap split)."""
    import rollup_read  # noqa: E402
    turns = [t for t in rollup_read.read_turns(path) if t.text.strip()]
    return Epoch("spike", turns)


def health() -> str:
    try:
        with urllib.request.urlopen(f"{GATEWAY}/health", timeout=5) as resp:
            return str(resp.status)
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return f"unreachable ({getattr(e, 'reason', e)})"


def load_sessions(limit: int):
    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    records = [r for r in load_records(QUESTIONS) if record_id(r) in graded]
    by_prefix = index_transcripts()
    sessions = {}
    for r in records:
        sid = r.get("session_id", "")
        if sid in sessions:
            sessions[sid]["n_questions"] += 1
            continue
        m = SLUG.match(sid)
        path = resolve(m.group(1), by_prefix) if m else None
        sessions[sid] = {"session_id": sid, "path": path, "n_questions": 1}
    out = list(sessions.values())
    return out[:limit] if limit else out, len(records)


GRADED = DATA / "rq3_rollup_from_stripped_spike_graded.jsonl"
JUDGE_MODEL = "claude-sonnet-5"


def answer_from_rollup(question: str, rollup: str):
    """Prompt of rq3_uniform_rollup_arm.answer_from_rollup (the arm that produced the
    56.5% baseline): the ladder's answer prompt with 'compressed note' naming the
    context. Same model, no temperature, same extract_confidence."""
    from rq3_raw_transcript_arm import call_llm, extract_confidence
    prompt = f"""Someone asks: "{question}"

Below is a compressed note summarizing an agent session. Answer the question based ONLY on this note. If the note does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

NOTE:
{rollup}

Provide your answer and confidence now:"""
    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=180)
    if not response:
        return "", "API_ERROR", usage
    answer, confidence = extract_confidence(response)
    return answer, confidence, usage


def grade_mode(limit: int) -> int:
    from rq3_raw_transcript_arm import grade_answer
    rolls = {}
    for l in OUT.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            rolls[d["session_id"]] = d["rollup_raw"]
    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    records = [r for r in load_records(QUESTIONS) if record_id(r) in graded]
    done = set()
    if GRADED.exists():
        done = {json.loads(l)["record_id"] for l in GRADED.read_text().splitlines() if l.strip()}
    todo = [r for r in records if record_id(r) not in done and r.get("session_id") in rolls]
    if limit:
        todo = todo[:limit]
    print(f"records {len(records)}  to answer {len(todo)}  rollups {len(rolls)}")
    tin = tout = 0
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        answer, conf, usage = answer_from_rollup(r["question"], rolls[r["session_id"]])
        tin += usage.get("prompt_tokens", 0)
        tout += usage.get("completion_tokens", 0)
        if conf == "API_ERROR":
            print(f"[{i}/{len(todo)}] {rid} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), conf)
        print(f"[{i}/{len(todo)}] {rid} {conf:18s} {grade}")
        row = {"record_id": rid, "session_id": r.get("session_id"), "class": r.get("class"),
               "question": r.get("question"), "ground_truth": r.get("ground_truth"),
               "rollup_chars": len(rolls[r["session_id"]]), "answer": answer,
               "confidence": conf, "grade": grade, "usage": usage, "model": JUDGE_MODEL}
        # string-match confidence correction, identical to the ladder's
        up = answer.upper()
        if conf != "NOT_ANSWERABLE" and ("NOT ANSWERABLE" in up or "NOT_ANSWERABLE" in up):
            row["corrected_confidence"], row["corrected_grade"] = "NOT_ANSWERABLE", "absent"
        else:
            row["corrected_confidence"], row["corrected_grade"] = conf, grade
        with GRADED.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    print(f"answer tokens: prompt={tin} completion={tout} (grading calls not counted)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    ap.add_argument("--generate", action="store_true",
                    help="call the model (off; never run during authoring)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grade", action="store_true",
                    help="answer+grade the 124 questions from the generated rollups "
                         "(--limit N then means N questions)")
    ap.add_argument("--variant", choices=["stripped", "prod"], default="stripped",
                    help="input: stripped = output-stripped transcript (first spike); "
                         "prod = rollup_read.read_turns, the production summariser input")
    ap.add_argument("--arm-name", default="",
                    help="outputs go to rq3_spike_<arm-name>.jsonl and ..._graded.jsonl")
    ap.add_argument("--double-length", action="store_true",
                    help="double the '(N-M sentences)' length numbers in the prompt")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()
    global OUT, GRADED, DOUBLE, MAX_TOKENS, VARIANT
    DOUBLE, MAX_TOKENS, VARIANT = args.double_length, args.max_tokens, args.variant
    if args.arm_name:
        OUT = DATA / f"rq3_spike_{args.arm_name}.jsonl"
        GRADED = DATA / f"rq3_spike_{args.arm_name}_graded.jsonl"
    if args.grade:
        return grade_mode(args.limit)
    if args.generate:
        args.dry_run = False

    sessions, n_records = load_sessions(0)
    print(f"graded records {n_records}  sessions {len(sessions)}  "
          f"model(production) {rs.model_for('cli')}  "
          f"summary max_tokens 1024 (summarise(), client path)")
    print(f"{'session id':58s} found  strip_chars  est_in_tok  chunks")

    rows = []
    for s in sessions:
        found = s["path"] is not None
        row = dict(s, found=found, stripped_chars=0, prompts=[], stripped="")
        if found and VARIANT == "prod":
            ep = prod_epoch(s["path"])
            row["stripped_chars"] = sum(len(t.text) for t in ep.turns)
            row["prompts"] = build_prompts(ep)
        elif found:
            stripped, _ = strip_transcript(s["path"].read_text(errors="replace"))
            row["stripped"] = stripped
            row["stripped_chars"] = len(stripped)
            row["prompts"] = build_prompts(stripped_to_epoch(stripped))
        row["est_in"] = est_tokens(sum(len(p) for p in row["prompts"]))
        rows.append(row)
        print(f"{s['session_id'][:58]:58s} {'y' if found else 'n':5s} "
              f"{row['stripped_chars']:12,d} {row['est_in']:11,d} {len(row['prompts']):6d}")

    found_rows = [r for r in rows if r["found"]]
    if args.limit:  # smallest sessions first, so a smoke test is cheap
        found_rows = sorted(found_rows, key=lambda r: r["stripped_chars"])[:args.limit]
    in_toks = [r["est_in"] for r in found_rows]
    strip_toks = [est_tokens(r["stripped_chars"]) for r in found_rows]
    n_calls = sum(len(r["prompts"]) for r in found_rows)
    print(f"\ntranscripts found: {len(found_rows)} of {len(rows)}")
    if found_rows:
        print(f"stripped tokens: total {sum(strip_toks):,}  min {min(strip_toks):,}  "
              f"median {int(statistics.median(strip_toks)):,}  max {max(strip_toks):,}")
        print(f"prompt tokens after production trimming/chunking: total {sum(in_toks):,}  "
              f"min {min(in_toks):,}  median {int(statistics.median(in_toks)):,}  "
              f"max {max(in_toks):,}  ({n_calls} summary calls, reduce calls not counted)")
    gen = sum(in_toks) * PRICE_IN + n_calls * GEN_OUT_TOKENS * PRICE_OUT
    grade_in = N_QUESTIONS * (GEN_OUT_TOKENS + GRADE_OVERHEAD_TOKENS)
    grade = grade_in * PRICE_IN + N_QUESTIONS * GRADE_OUT_TOKENS * PRICE_OUT
    print(f"est generation, one draw @ $3/M in $15/M out, {GEN_OUT_TOKENS} out tok/call: ${gen:.2f}")
    print(f"est grading, {N_QUESTIONS} q x ({GEN_OUT_TOKENS} artifact + {GRADE_OVERHEAD_TOKENS} overhead "
          f"in, {GRADE_OUT_TOKENS} out): ${grade:.2f}")
    print(f"token basis: chars/{CHARS_PER_TOKEN} (the ladder has no tokenizer); "
          f"the artifact graded is the ~{GEN_OUT_TOKENS}-token rollup, not the transcript")

    if not args.generate:
        print(f"\ngateway {GATEWAY}/health: {health()}")
        print("--dry-run: no model calls made")
        return 0

    # ---- generate: the only path that calls the model ----
    import os
    from rq3_raw_transcript_arm import call_llm  # noqa: E402
    model = os.environ.get("PKM_ROLLUP_MODEL") or rs.model_for("cli")
    done = set()
    if OUT.exists():
        done = {json.loads(l)["session_id"] for l in OUT.read_text().splitlines() if l.strip()}
    for r in found_rows:
        if r["session_id"] in done:
            continue
        replies, usage = [], []
        for p in r["prompts"]:
            text, u = call_llm(p, model=model, max_tokens=MAX_TOKENS)
            replies.append(text)
            usage.append(u)
        if len(replies) > 1 and all(replies):
            parts = [json.loads(rs.extract_json(t)) for t in replies]
            rp = rs.build_reduce_prompt(parts)
            if DOUBLE:
                rp = _LEN_RE.sub(lambda m: f"({int(m.group(1)) * 2}-{int(m.group(2)) * 2} sentences)", rp)
            text, u = call_llm(rp, model=model, max_tokens=MAX_TOKENS)
            replies.append(text)
            usage.append(u)
        rollup = replies[-1]
        with OUT.open("a") as fh:
            fh.write(json.dumps({"session_id": r["session_id"], "model": model,
                                 "stripped_chars": r["stripped_chars"],
                                 "rollup_raw": rollup, "usage": usage}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
