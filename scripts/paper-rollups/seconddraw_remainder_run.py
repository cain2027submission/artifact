#!/usr/bin/env python3
"""Second answer draw (cap 46, order seed 20261015) for the two Replication 1 arms on the 136 remainder question ids
(data/paper-rollups/seconddraw-remainder/remainder_ids.json). Reuses replication_run.py unchanged
(substrates, prompts, 3-draw majority grading, blind judge) and only redirects its output files,
call ledger and cap. usage: seconddraw_run.py LO HI  (indices into the sample file order)."""
import json, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replication_common as rc
import replication_run as rr

SD = rc.DATA / "seconddraw-remainder"
rc.CALLS = SD / "calls.jsonl"; rc.COST_LOG = SD / "cost_log.jsonl"; rc.CAP_USD = 46.0; rr.ORDER_SEED = 20261015
rr.ANS = SD / "answers.jsonl"; rr.FAIL = SD / "failures.jsonl"


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    ids = json.loads((SD / "remainder_ids.json").read_text())
    qmap = {q["question_id"]: q for q in rr.verify_frozen()}
    qs = [qmap[i] for i in ids]
    done = {(json.loads(l)["question_id"], json.loads(l)["arm"]) for l in rr.ANS.open()} if rr.ANS.exists() else set()
    rng = random.Random(rr.ORDER_SEED)
    jobs = []
    for k, q in enumerate(qs):
        order = ["nodigest", "stripped"] if rng.random() < .5 else ["stripped", "nodigest"]
        if lo <= k < hi:
            for arm in order:
                if (q["question_id"], arm) not in done:
                    jobs.append((q, arm, order[0]))
    est = sum(rr.estimate([j[0]], [j[1]]) for j in jobs)
    stage = f"remainder_{lo}_{hi}"
    rc.gate(stage, est)
    before = rc.cumulative_usd(); t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(rr.do, jobs))
    print(Counter(r[0] for r in res))
    rc.log_batch(stage, est, note=f"jobs={len(jobs)} outcomes={dict(Counter(r[0] for r in res))} batch_usd={rc.cumulative_usd()-before:.3f}", since=t0)


if __name__ == "__main__":
    main()
