#!/usr/bin/env python3
"""Build the blind human-grading sheet for the RQ3 retention experiment.

Zero model calls. Reads the frozen per-arm graded jsonl files, draws 50
(question, arm) items with a fixed seed, and writes:

  data/paper-rollups/human_grading_sheet.csv   blind: no arm, judge grade or session
  data/paper-rollups/human_grading_key.json    item_id -> arm, session, judge grade

Why: one model family generated, answered and graded every record, so a human
grade on a subset measures judge agreement with someone outside that family.
The judge `grade` field is the single-protocol grade the paper's rates use.

Sampling: rollup, output-stripped and raw get 12 each; dialogue record,
prose-only and call-only share the remaining 14 (5/5/4). At most 3 items per
session, spread across as many sessions as possible, at least 8 judged wrong
and 8 judged absent, one item per question where possible, then shuffled.

    python3 scripts/paper-rollups/human_grading_sample.py
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
SEED = 20261008
N_TOTAL = 50
MAX_PER_SESSION = 3
MIN_WRONG = 8
MIN_ABSENT = 8
ANSWER_CLIP = 2000

# arm label -> (file, answer field, quota)
ARMS = {
    "rollup": ("rq3_rollup_uniform.jsonl", "answer", 12),
    "output-stripped": ("rq3_questions_stripped_arm.jsonl", "answer", 12),
    "raw": ("rq3_questions_raw_arm.jsonl", "raw_answer", 12),
    "dialogue record": ("rq3_verbatim_arm.jsonl", "answer", 5),
    "prose-only": ("rq3_prose_only_arm.jsonl", "answer", 5),
    "call-only": ("rq3_tool_inputs_only_arm.jsonl", "answer", 4),
}

SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
]
redactions = Counter()


def redact(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text, n = pat.subn("[REDACTED]", text)
        redactions[pat.pattern] += n
    return text


def load(name: str):
    out = []
    for line in (DATA / name).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().replace("\x00", "")
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # NUL-filled partial write, as in the other loaders
    return out


def build_pool():
    pool = []
    for arm, (fname, afield, _) in ARMS.items():
        for r in load(fname):
            if r.get("grade") not in ("correct", "partially_correct", "wrong", "absent"):
                continue
            pool.append({
                "arm": arm,
                "session": r["session_id"],
                "record_id": r["record_id"],
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "answer": r.get(afield) or "",
                "judge": r["grade"],
            })
    return pool


def attempt(pool, rng):
    quota = {a: v[2] for a, v in ARMS.items()}
    chosen, sess, used_q = [], Counter(), set()

    def pick(cands):
        cands = [c for c in cands
                 if quota[c["arm"]] > 0 and sess[c["session"]] < MAX_PER_SESSION
                 and c["record_id"] not in used_q]
        if not cands:
            return False
        low = min(sess[c["session"]] for c in cands)
        c = rng.choice([c for c in cands if sess[c["session"]] == low])
        chosen.append(c)
        quota[c["arm"]] -= 1
        sess[c["session"]] += 1
        used_q.add(c["record_id"])
        return True

    for grade, need in (("wrong", MIN_WRONG), ("absent", MIN_ABSENT)):
        for _ in range(need):
            if not pick([c for c in pool if c["judge"] == grade]):
                return None
    while any(quota.values()):
        if not pick(pool):
            return None
    return chosen


def main():
    rng = random.Random(SEED)
    pool = build_pool()
    best = None
    for _ in range(2000):
        got = attempt(pool, rng)
        if got is None:
            continue
        n_sess = len({c["session"] for c in got})
        if best is None or n_sess > best[0]:
            best = (n_sess, got)
    if best is None:
        raise SystemExit("no sample satisfies the constraints")
    chosen = best[1]
    assert len(chosen) == N_TOTAL
    rng.shuffle(chosen)

    rows, key = [], {}
    for i, c in enumerate(chosen, 1):
        item_id = f"H{i:02d}"
        ans = c["answer"].strip() or "[no answer given]"
        if len(ans) > ANSWER_CLIP:
            ans = ans[:ANSWER_CLIP] + f" [... clipped, {len(c['answer']) - ANSWER_CLIP} more characters]"
        rows.append({
            "item_id": item_id,
            "question": redact(c["question"]),
            "ground_truth": redact(c["ground_truth"]),
            "answer": redact(ans),
            "human_grade": "",
            "notes": "",
        })
        key[item_id] = {"arm": c["arm"], "session": c["session"], "judge_grade": c["judge"]}

    with open(DATA / "human_grading_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "question", "ground_truth", "answer", "human_grade", "notes"])
        w.writeheader()
        w.writerows(rows)
    (DATA / "human_grading_key.json").write_text(
        json.dumps({"seed": SEED, "items": key}, indent=2) + "\n", encoding="utf-8")

    (DATA / "HUMAN_GRADING.md").write_text(INSTRUCTIONS, encoding="utf-8")

    print(f"wrote {len(rows)} items over {best[0]} sessions")
    print("by arm:", dict(Counter(k["arm"] for k in key.values())))
    print("by judge grade:", dict(Counter(k["judge_grade"] for k in key.values())))
    print("redactions:", dict(redactions))


INSTRUCTIONS = """# Human grading instructions

Grade each answer in `human_grading_sheet.csv` against its ground truth, using only the question, ground truth and answer shown. Write one value in the `human_grade` column: `correct`, `partial`, `wrong` or `absent`. Use `notes` for anything unclear.

- `correct`: matches the ground truth.
- `partial`: partially matches or covers some aspects.
- `wrong`: contradicts the ground truth or provides incorrect information.
- `absent`: no real answer provided.

These are the judge's definitions verbatim (the judge prompt in `scripts/paper-rollups/rq3_raw_transcript_arm.py`, `grade_answer`, where partial is spelled PARTIALLY_CORRECT). Grade the answer, not how you think it was produced; do not open `human_grading_key.json` until every row is graded. Estimated time: 50 items at about 1 minute each, so 45 to 60 minutes. Then run `python3 scripts/paper-rollups/human_grading_score.py`.
"""

if __name__ == "__main__":
    main()
