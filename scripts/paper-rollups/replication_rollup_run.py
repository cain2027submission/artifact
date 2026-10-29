#!/usr/bin/env python3
"""Production-rollup arm on the Replication 1 questions, per
data/paper-rollups/replication-rollup/analysis_plan.md. Reuses
heldout_rollup_run.one() unchanged (answer_from_rollup + 3-draw majority grade);
only the question source, note path resolution, output and cost log differ.
Usage: replication_rollup_run.py --dry-run | --limit N | (full) | --sensitivity
"""
import argparse, glob, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_rollup_run as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "paper-rollups"
RR = D / "replication-rollup"
MAIN_SESSIONS = SESSIONS_DIR  # read-only
OUT, OUT_SENS = RR / "rollup_arm.jsonl", RR / "rollup_arm_sensitivity.jsonl"
H.COST = RR / "cost_log.jsonl"
H.CAP_USD = 5.0
ALT = "2026-09-25_claude-code_2ce8e87b_example-claim-one.md"
PRIMARY_0328 = "2026-09-25_claude-code_2ce8e87b_example-claim-two.md"


def note_for(prefix):
    hits = sorted(glob.glob(str(MAIN_SESSIONS / f"*_{prefix}_*.md")))
    if prefix == "2ce8e87b":
        return str(MAIN_SESSIONS / PRIMARY_0328)
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def load(sens):
    ans = {json.loads(l)["question_id"] for l in (D / "replication/answers.jsonl").open()
           if json.loads(l)["arm"] == "stripped"}
    qs = [json.loads(l) for l in (D / "replication/questions.jsonl").open() if l.strip()]
    qs = [q for q in qs if q["question_id"] in ans]
    if sens:
        qs = [q for q in qs if q["session_prefix"] == "2ce8e87b"]
    for q in qs:
        q["_rollup"] = str(MAIN_SESSIONS / ALT) if sens else note_for(q["session_prefix"])
        q["_rid"] = q["question_id"]
        assert Path(q["_rollup"]).exists()
    return qs


def est(qs):
    pin = sum(Path(q["_rollup"]).stat().st_size / 3.2 + 150 + 3 * (len(q["question"]) + len(q["ground_truth"]) + 700) / 3.2 * 1.15 for q in qs)
    return (pin * H.PRICE_IN + len(qs) * (200 + 15) * H.PRICE_OUT) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args()
    out = OUT_SENS if a.sensitivity else OUT
    qs = load(a.sensitivity)
    done = {json.loads(l)["record_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [q for q in qs if q["_rid"] not in done]
    if a.limit:
        todo = sorted(todo, key=lambda q: Path(q["_rollup"]).stat().st_size)
        n = len(todo)
        todo = [todo[i] for i in dict.fromkeys([0, n - 1, n // 2, n // 4, 3 * n // 4])][:a.limit]
    print(f"questions {len(qs)} done {len(done)} todo {len(todo)} estimate ${est(todo):.3f} running ${H.total():.3f} cap ${H.CAP_USD}")
    if a.dry_run:
        return
    errs = []
    for b in range(0, len(todo), H.BATCH):
        batch = todo[b:b + H.BATCH]
        e = est(batch)
        print(f"batch {b//H.BATCH+1}: estimate ${e:.3f}, running total ${H.total():.3f}, cap ${H.CAP_USD}", flush=True)
        if H.total() + e > H.CAP_USD:
            print("STOP: batch would exceed cap"); sys.exit(2)
        t0 = time.time()
        with ThreadPoolExecutor(5) as ex:
            res = list(ex.map(H.one, batch))
        paths = {q["_rid"]: q["_rollup"] for q in batch}
        calls = ptok = ctok = 0
        for row, (c, p, k), err in res:
            calls += c; ptok += p; ctok += k
            if row:
                row["rollup"] = "sessions/" + Path(paths[row["record_id"]]).name
                with out.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                errs.append(err)
        spent = (ptok * H.PRICE_IN + ctok * H.PRICE_OUT) / 1e6
        H.log("sensitivity" if a.sensitivity else "main", calls, ptok, ctok, spent)
        print(f"  done {time.time()-t0:.0f}s calls {calls} tokens {ptok}+{ctok} est ${spent:.3f}; total ${H.total():.3f}; failed {len(errs)}", flush=True)
    if errs:
        print("FAILED:", errs)


if __name__ == "__main__":
    main()
