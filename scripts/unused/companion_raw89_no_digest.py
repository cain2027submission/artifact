#!/usr/bin/env python3
"""Companion test (a): no-digest action record vs raw on the 89 questions raw fits. No model calls.

Pre-registration: tasks/2026-10-13 a fresh-session replication decides whether the no-digest
action record can be the primary recommendation, pre-registered before any run.md

Reuses existing answers and `grade` (3-draw majority, grade_protocol v2) from
  rq3_questions_raw_arm.jsonl (the 89), rq3_questions_stripped_arm.jsonl,
  rq3_action_record_no_digest_arm.jsonl (covers all 89, so spend is zero).
Recall = correct + partially_correct. Session-clustered paired bootstrap, B=20000,
seed 20261004, 90% CI (same method and constants as rq3_heldout_equivalence_rebuild.py).
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper-rollups"
B, SEED, OK = 20000, 20261004, ("correct", "partially_correct")


def load(n):
    return {json.loads(l)["record_id"]: json.loads(l) for l in (DATA / n).open() if l.strip()}


def boot(a, b, sess):
    rng = random.Random(SEED)
    by = {}
    for i, s in enumerate(sess):
        by.setdefault(s, []).append(i)
    keys = list(by)
    d = []
    for _ in range(B):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in by[k]]
        d.append(sum(a[i] for i in idx) / len(idx) - sum(b[i] for i in idx) / len(idx))
    d.sort()

    def pct(p):
        k = p * (B - 1); lo = int(k)
        return d[lo] * (1 - (k - lo)) + d[lo + 1] * (k - lo) if lo + 1 < B else d[lo]
    return [100 * pct(.05), 100 * pct(.95)], [100 * pct(.025), 100 * pct(.975)]


def cmp(name, a, b, ids, sess):
    ao = [a[i]["grade"] in OK for i in ids]
    bo = [b[i]["grade"] in OK for i in ids]
    c90, c95 = boot(ao, bo, sess)
    eq = lambda ci, m: -m <= ci[0] and ci[1] <= m
    return {"comparison": name, "n": len(ids), "n_sessions": len(set(sess)),
            "recall_first_pct": 100 * sum(ao) / len(ids), "recall_second_pct": 100 * sum(bo) / len(ids),
            "diff_points": 100 * (sum(ao) - sum(bo)) / len(ids),
            "ci90_points": c90, "ci95_points": c95,
            "equivalent_10pts": eq(c90, 10), "equivalent_5pts": eq(c90, 5)}


raw, st, nd = load("rq3_questions_raw_arm.jsonl"), load("rq3_questions_stripped_arm.jsonl"), load("rq3_action_record_no_digest_arm.jsonl")
ids = sorted(raw)
assert all(i in st and i in nd for i in ids)
sess = [raw[i]["session_id"] for i in ids]
out = {
    "pre_registration": "tasks/2026-10-13 a fresh-session replication decides whether the no-digest action record can be the primary recommendation, pre-registered before any run",
    "grade_field": "grade", "bootstrap": {"B": B, "seed": SEED, "kind": "session-clustered paired", "ci": 90},
    "questions_covered_by_existing_no_digest_answers": len([i for i in ids if i in nd]), "new_paid_calls": 0, "usd_spent": 0.0,
    "primary_no_digest_minus_raw": cmp("no_digest_minus_raw", nd, raw, ids, sess),
    "reference_output_stripped_minus_raw": cmp("output_stripped_minus_raw", st, raw, ids, sess),
    "no_digest_minus_output_stripped_same_89": cmp("no_digest_minus_output_stripped", nd, st, ids, sess),
}
rate = lambda d, ids: {"absent": sum(d[i]["grade"] == "absent" for i in ids), "wrong": sum(d[i]["grade"] == "wrong" for i in ids)}
out["counts"] = {"raw": rate(raw, ids), "output_stripped": rate(st, ids), "no_digest": rate(nd, ids)}
(DATA / "companion").mkdir(exist_ok=True)
(DATA / "companion" / "raw89_no_digest.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
