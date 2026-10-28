#!/usr/bin/env python3
"""RQ3 dense encoding arms: is 500 characters a hard information ceiling?

The published read-budget arms at 500 characters scored 32.6% (rollup snippet)
and 37.1% (tool-stripped chunk). Both use naive excerpt formats: verbatim prose
with stopwords, function words, and markdown preserved. This script tests whether
encoding efficiency can beat that ceiling at the same character budget.

Three arms, all at 500 characters, same questions, same judge, same grading:

  blind-dense       Model compresses session WITHOUT seeing questions (the fair arm)
  oracle-dense      Model compresses WITH questions visible (upper-bound control)
  mechanical-dense  Deterministic algorithm: drop stopwords, abbreviate, densest-first

If blind-dense matches baseline, the budget is a hard ceiling. If oracle beats it
but blind doesn't, the effect is question-targeting not encoding. If blind beats
baseline, encoding efficiency is real. If mechanical matches blind, the ceiling
is free (no model needed).

Design fidelity (must match rq3_read_budget_arms.py exactly):
  * Same 89-record subset (build_items() logic)
  * Same source (tool-stripped transcript, the 37.1% baseline)
  * Same answering prompt template, only context varies
  * Same grading (grade_answer, PARTIALLY_CORRECT before CORRECT)
  * Same character budget (500 chars, byte-matched to rollup_hookshape)
  * Same shape (pointer + intro + body, matching stripped_chunk format)

Usage:
    rq3_dense_encoding_arms.py --dry-run           # build contexts, no API calls
    rq3_dense_encoding_arms.py --degenerate        # encoder smoke test
    rq3_dense_encoding_arms.py --limit 3           # smoke test
    rq3_dense_encoding_arms.py                     # full run (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    call_llm, grade_answer, index_transcripts, load_records, record_id, resolve, SLUG,
)
from rq3_read_budget_arms import build_items, JUDGE_MODEL  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402
from paths import REPO_ROOT  # noqa: E402

DATA = REPO_ROOT / "data" / "paper-rollups"
OUT = DATA / "rq3_dense_encoding_arms.jsonl"
REPORT = DATA / "rq3_dense_encoding_summary.json"

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")

# Match the existing arms exactly
MAX_BODY_CHARS = 500
RECOVERABLE = ("correct", "partially_correct")
ARMS = ("blind_dense", "oracle_dense", "mechanical_dense")

# Stopwords for mechanical_dense
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "what", "which", "who", "whom", "why", "how", "when", "where", "did", "do",
    "does", "done", "had", "has", "have", "not", "no", "yes", "he", "she", "they",
    "them", "his", "her", "their", "you", "your", "i", "we", "our", "us", "me",
    "so", "there", "here", "about", "into", "over", "under", "after", "before",
    "during", "while", "also", "only", "very", "more", "most", "some", "any",
    "all", "both", "each", "other", "such", "own", "same", "too", "can", "will",
    "just", "should", "would", "could", "may", "might", "must",
}


def encode_blind_dense(stripped: str, budget: int) -> str:
    """Model compresses session to ≤budget chars WITHOUT seeing questions.

    Returns the compressed text. Caller enforces the budget via assertion.
    """
    prompt = f"""Compress the following agent session transcript into AT MOST {budget} characters, maximizing recoverable detail.

Preserve verbatim: identifiers, file paths, numbers, decisions, outcomes, error strings.
Drop: function words, prose connectives, redundant context.
Use abbreviations where unambiguous.

Your compressed output will be used to answer questions about this session, but you do NOT see the questions. Maximize information density.

TRANSCRIPT:
{stripped}

Provide your compressed version (max {budget} characters):"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=180)
    if not response:
        return ""
    return response.strip()


def encode_oracle_dense(stripped: str, questions: list, budget: int) -> str:
    """Model compresses session to ≤budget chars WITH questions visible.

    This is the upper-bound control: deliberately NOT a fair arm.
    """
    q_list = "\n".join(f"- {q}" for q in questions)
    prompt = f"""Compress the following agent session transcript into AT MOST {budget} characters, maximizing recoverable detail FOR THE QUESTIONS BELOW.

QUESTIONS YOU MUST ANSWER FROM THIS:
{q_list}

Preserve verbatim: identifiers, file paths, numbers, decisions, outcomes, error strings relevant to these questions.
Drop: function words, prose connectives, anything not relevant to answering these questions.
Use abbreviations where unambiguous.

TRANSCRIPT:
{stripped}

Provide your compressed version (max {budget} characters):"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=180)
    if not response:
        return ""
    return response.strip()


def encode_mechanical_dense(stripped: str, budget: int) -> str:
    """Deterministic compression: drop stopwords, abbreviate, densest-first.

    No model calls. This tests whether the ceiling is free.
    """
    # Extract tokens, keep identifiers/paths/numbers, drop stopwords
    tokens = []
    for word in re.findall(r'[A-Za-z0-9_./\\-]+', stripped):
        low = word.lower()
        # Keep if it's a path, identifier, number, or not a stopword
        if any(c in word for c in './_\\') or word.isdigit() or low not in STOP:
            # Simple abbreviations
            if len(word) > 8 and low in ("error", "failed", "success", "command"):
                word = word[:4]
            tokens.append(word)

    # Join densely, truncate at budget
    dense = " ".join(tokens)
    if len(dense) <= budget:
        return dense
    # Hard truncate at word boundary
    return dense[:budget].rsplit(" ", 1)[0] + "…"


def build_dense_context(tpath: str, encoding: str, budget_chars: int) -> str:
    """Match the shape of stripped_chunk: pointer + intro + body.

    The body is the encoded text, capped at the budget remaining after
    the pointer and intro overhead.
    """
    head = f"- {tpath}"
    intro = f"\n\nFrom {tpath}:\n"
    room = budget_chars - len(head) - len(intro)
    if room <= 0:
        return head[:budget_chars]

    # Truncate encoding if needed
    body = encoding if len(encoding) <= room else encoding[:room].rsplit(" ", 1)[0] + "…"
    return head + intro + body


ANSWER_PROMPT = """Someone asks: "{question}"

Below is the context block that was automatically retrieved and injected alongside that question. Answer the question based ONLY on this context. If the context does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

CONTEXT:
{context}

Provide your answer and confidence now:"""


def answer(question: str, context: str):
    """Match rq3_read_budget_arms.answer() exactly."""
    response, usage = call_llm(ANSWER_PROMPT.format(question=question,
                                                    context=context),
                               model=JUDGE_MODEL, timeout=180)
    if not response:
        return "", "API_ERROR", usage
    lines = response.strip().split("\n")
    confidence = "NOT_ANSWERABLE"
    for keyword in ["ANSWERED", "PARTIALLY_ANSWERED", "NOT_ANSWERABLE"]:
        if keyword in lines[-1].upper():
            confidence = keyword
            break
    return "\n".join(lines[:-1]).strip(), confidence, usage


def rates(records: list) -> dict:
    """Compute rates, CI, hallucination. Verbatim from rq3_read_budget_arms."""
    from math import sqrt

    n = len(records)
    counts = Counter(r["grade"] for r in records)

    def wilson95(k, n):
        if n == 0:
            return [0.0, 0.0]
        p = k / n
        z = 1.96
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        spread = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        return [max(0.0, center - spread) * 100, min(1.0, center + spread) * 100]

    out = {"n": n}
    for grade in ["correct", "partially_correct", "wrong", "absent"]:
        k = counts.get(grade, 0)
        out[grade] = {"n": k, "pct": 100.0 * k / max(1, n), "ci95": wilson95(k, n)}

    recoverable_n = counts.get("correct", 0) + counts.get("partially_correct", 0)
    out["recoverable"] = {
        "n": recoverable_n,
        "pct": 100.0 * recoverable_n / max(1, n),
        "ci95": wilson95(recoverable_n, n),
    }

    attempted = n - counts.get("absent", 0)
    wrong = counts.get("wrong", 0)
    out["hallucination_of_attempted"] = {
        "n_wrong": wrong,
        "n_attempted": attempted,
        "pct": 100.0 * wrong / max(1, attempted) if attempted else None,
        "ci95": wilson95(wrong, attempted),
    }

    return out


def build_encodings(items: list, limit: int, dry_run: bool) -> dict:
    """Pre-encode all sessions for blind_dense and oracle_dense.

    Returns a dict: {session_id: {arm: encoding}}
    Skips encoding if dry_run=True.
    """
    # Group items by session to encode each session once
    by_session = {}
    for item in items[:limit] if limit else items:
        sid = item["session_id"]
        if sid not in by_session:
            by_session[sid] = {
                # build_items() hands back the transcript path and
                # stripped_chars, never the stripped text, so strip it here
                # once per session rather than once per question.
                "stripped": strip_transcript(
                    Path(item["transcript"]).read_text(errors="replace"))[0],
                "transcript": item["transcript"],
                "questions": [],
            }
        by_session[sid]["questions"].append(item["question"])

    encodings = {}
    if dry_run:
        print("--dry-run: skipping encoding")
        return encodings

    print(f"\nEncoding {len(by_session)} unique sessions...")
    for n, (sid, data) in enumerate(sorted(by_session.items()), 1):
        stripped = data["stripped"]
        questions = data["questions"]

        # Calculate budget: match the stripped_chunk budget from the first item
        # In practice, all items from same session should have same budget
        budget = MAX_BODY_CHARS

        print(f"[{n}/{len(by_session)}] Encoding {sid}...")

        blind = encode_blind_dense(stripped, budget)
        oracle = encode_oracle_dense(stripped, questions, budget)
        mechanical = encode_mechanical_dense(stripped, budget)

        # Enforce budget with assertions
        if len(blind) > budget:
            print(f"  WARNING: blind_dense exceeded budget: {len(blind)} > {budget}, truncating")
            blind = blind[:budget].rsplit(" ", 1)[0] + "…"
        if len(oracle) > budget:
            print(f"  WARNING: oracle_dense exceeded budget: {len(oracle)} > {budget}, truncating")
            oracle = oracle[:budget].rsplit(" ", 1)[0] + "…"
        if len(mechanical) > budget:
            print(f"  WARNING: mechanical_dense exceeded budget: {len(mechanical)} > {budget}, truncating")
            mechanical = mechanical[:budget].rsplit(" ", 1)[0] + "…"

        assert len(blind) <= budget, f"blind_dense {len(blind)} > {budget}"
        assert len(oracle) <= budget, f"oracle_dense {len(oracle)} > {budget}"
        assert len(mechanical) <= budget, f"mechanical_dense {len(mechanical)} > {budget}"

        encodings[sid] = {
            "blind_dense": blind,
            "oracle_dense": oracle,
            "mechanical_dense": mechanical,
        }

        print(f"  blind: {len(blind)} chars, oracle: {len(oracle)} chars, mechanical: {len(mechanical)} chars")

    return encodings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--degenerate", action="store_true",
                    help="smoke the encoders on empty/garbage inputs")
    args = ap.parse_args()

    if args.degenerate:
        print("=== Degenerate input check ===")
        for stripped, budget in [("", 500), ("abc", 500), ("x" * 10, 500), ("y" * 5000, 500)]:
            print(f"\nInput: {len(stripped)} chars, budget: {budget}")
            blind = encode_blind_dense(stripped, budget)
            oracle = encode_oracle_dense(stripped, ["what happened?"], budget)
            mech = encode_mechanical_dense(stripped, budget)
            print(f"  blind: {len(blind)} chars: {repr(blind[:60])}")
            print(f"  oracle: {len(oracle)} chars: {repr(oracle[:60])}")
            print(f"  mechanical: {len(mech)} chars: {repr(mech[:60])}")
            assert len(blind) <= budget, f"blind exceeded: {len(blind)}"
            assert len(oracle) <= budget, f"oracle exceeded: {len(oracle)}"
            assert len(mech) <= budget, f"mechanical exceeded: {len(mech)}"
        print("\nDegenerate check passed")
        return

    items, skipped = build_items(args.limit)
    if not items:
        print("No items to process")
        return

    # Cache stripped transcripts per item (build_items already does this)
    print(f"\nBuilding encodings for {len(items)} items...")

    # Build encodings
    encodings = build_encodings(items, args.limit, args.dry_run)

    # Build contexts
    for item in items:
        sid = item["session_id"]
        tpath = item["transcript"]
        budget = MAX_BODY_CHARS  # Match the existing arms

        if args.dry_run:
            # Use dummy encodings
            item["encodings"] = {
                "blind_dense": "(not encoded in dry-run)",
                "oracle_dense": "(not encoded in dry-run)",
                "mechanical_dense": "(not encoded in dry-run)",
            }
        else:
            item["encodings"] = encodings.get(sid, {
                "blind_dense": "",
                "oracle_dense": "",
                "mechanical_dense": "",
            })

        item["contexts"] = {}
        for arm in ARMS:
            encoding = item["encodings"].get(arm, "")
            ctx = build_dense_context(tpath, encoding, budget)
            item["contexts"][arm] = ctx

    # Report context sizes
    sizes = {a: [len(i["contexts"][a]) for i in items] for a in ARMS}
    print("\ncontext sizes (chars), n=%d" % len(items))
    for a in ARMS:
        v = sorted(sizes[a])
        print("  %-17s median %4d  min %4d  max %4d"
              % (a, v[len(v) // 2] if v else 0, v[0] if v else 0, v[-1] if v else 0))

    if args.dry_run:
        print("\n--dry-run: no API calls made")
        print("\n=== sample contexts (first item) ===")
        for a in ARMS:
            print(f"\n--- {a} ({len(items[0]['contexts'][a])} chars) ---\n{items[0]['contexts'][a][:200]}")
        return

    # Resume from existing records
    done = set()
    if OUT.exists():
        for r in load_records(OUT):
            done.add((r.get("record_id"), r.get("arm")))
        print(f"resuming: {len(done)} (record, arm) cells already graded")

    todo = [(i, a) for i in items for a in ARMS if (i["record_id"], a) not in done]
    print(f"to grade: {len(todo)} cells\n")

    t0 = time.time()
    tokens_in = tokens_out = 0
    errors = 0
    for n, (item, arm) in enumerate(todo, 1):
        ctx = item["contexts"][arm]
        ans, conf, usage = answer(item["question"], ctx)
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("completion_tokens") or 0
        if conf == "API_ERROR":
            errors += 1
            print("[%d/%d] %s %s -> API_ERROR (not recorded)"
                  % (n, len(todo), item["record_id"], arm))
            continue
        grade = grade_answer(ans, item["ground_truth"], conf)
        rec = {"record_id": item["record_id"], "arm": arm,
               "session_id": item["session_id"], "class": item["class"],
               "context_chars": len(ctx), "confidence": conf, "grade": grade,
               "answer": ans, "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print("[%d/%d] %-16s %-16s %-10s %s  (%.0fs, in=%d out=%d)"
              % (n, len(todo), item["record_id"], arm, conf, grade,
                 time.time() - t0, tokens_in, tokens_out))

    print("\nAPI errors (unrecorded, retryable): %d" % errors)
    print("tokens: prompt=%d completion=%d" % (tokens_in, tokens_out))
    summarise(items, skipped, {"prompt_tokens": tokens_in,
                               "completion_tokens": tokens_out,
                               "api_errors": errors})


def summarise(items: list, skipped: list, spend: dict) -> None:
    rows = load_records(OUT)
    by_arm = {a: {} for a in ARMS}
    for r in rows:
        if r.get("arm") in by_arm:
            by_arm[r["arm"]][r["record_id"]] = r
    attempted = {i["record_id"] for i in items}
    complete = sorted(set.intersection(*[set(by_arm[a]) & attempted for a in ARMS]))
    print("\nattempted %d records x %d arms = %d cells; %d records have all 3 arms"
          % (len(attempted), len(ARMS), len(attempted) * len(ARMS), len(complete)))
    for a in ARMS:
        miss = sorted(attempted - set(by_arm[a]))
        if miss:
            print("  %s missing %d: %s" % (a, len(miss), miss[:6]))

    out = {"n_attempted_records": len(attempted),
           "n_complete_records": len(complete),
           "skipped_records": skipped, "spend": spend, "arms": {}}
    hdr = ("%-17s %5s  %3s %3s %3s %3s  %-8s %-14s %s"
           % ("arm", "n", "cor", "par", "wrg", "abs", "recover", "wilson95",
              "median chars"))
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for a in ARMS:
        rs = [by_arm[a][i] for i in complete]
        d = rates(rs)
        chars = sorted(r["context_chars"] for r in rs)
        d["median_context_chars"] = chars[len(chars) // 2] if chars else 0
        out["arms"][a] = d
        print("%-17s %5d  %3d %3d %3d %3d  %7.1f%% %-14s %d"
              % (a, d["n"], d["correct"]["n"], d["partially_correct"]["n"],
                 d["wrong"]["n"], d["absent"]["n"], d["recoverable"]["pct"],
                 str(d["recoverable"]["ci95"]), d["median_context_chars"]))

    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote %s" % REPORT)


if __name__ == "__main__":
    main()
