#!/usr/bin/env python3
"""Answer the 125 held-out questions from the Haiku-written rollups and compare, per
data/paper-rollups/heldout-rollup-haiku/analysis_plan.md (committed before any answer).
Reuses heldout_rollup_run.one (same answer_from_rollup, same 3-draw `grade` protocol) and the
statistics of heldout_rollup_error_test.py (loaded from its source, since it runs on import).
Usage: heldout_rollup_haiku_test.py --dry-run | (run + analyse) | --analyse
"""
import argparse, json, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_rollup_run as hr  # noqa: E402
from rq3_raw_transcript_arm import load_records  # noqa: E402
import tiktoken  # noqa: E402

DATA = hr.DATA
HH = DATA / "heldout-rollup-haiku"
hr.COST = HH / "cost_log.jsonl"
hr.CAP_USD = 4.0   # the whole task's cap: generation estimate is already in this log
OUT = HH / "rollup_arm.jsonl"
RES = HH / "results.json"
ns = {"__file__": str(Path(__file__).parent / "heldout_rollup_error_test.py")}
exec((Path(__file__).parent / "heldout_rollup_error_test.py").read_text().split("def main():")[0], ns)
ENC = tiktoken.get_encoding("cl100k_base")
PROV = {}
for l in (DATA / "rollup-provenance.jsonl").read_text().splitlines():
    r = json.loads(l)
    PROV[(r["arm"], r["session_id"])] = r


def load(p):
    return [json.loads(l) for l in (DATA / p).open() if l.strip()]


def run(dry):
    qs = load_records(hr.QUESTIONS)
    for q in qs:
        q["_rid"] = hr.rid(q)
        q["_rollup"] = str(HH / "rollups" / f"{q['session_id']}.md")
    qs = [q for q in qs if Path(q["_rollup"]).exists()]
    done = {json.loads(l)["record_id"] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
    todo = [q for q in qs if q["_rid"] not in done]
    chars = lambda q: Path(q["_rollup"]).stat().st_size
    pin = sum(chars(q) / 3.2 + 150 + 3 * (len(q["question"]) + len(q["ground_truth"]) + 700) / 3.2 * 1.15 for q in todo)
    e = (pin * hr.PRICE_IN + len(todo) * (200 + 15) * hr.PRICE_OUT) / 1e6
    print(f"questions {len(qs)} todo {len(todo)} estimate ${e:.3f} running total ${hr.total():.3f} cap ${hr.CAP_USD}")
    if dry:
        return
    if hr.total() + e > hr.CAP_USD:
        print("STOP: would exceed cap"); sys.exit(2)
    t0 = time.time()
    with ThreadPoolExecutor(5) as ex:
        res = list(ex.map(hr.one, todo))
    calls = ptok = ctok = 0
    errs = []
    for row, (c, p, k), err in res:
        calls += c; ptok += p; ctok += k
        if row:
            pv = PROV[("heldout-haiku-regen", row["session_id"])]
            row.update(arm="rollup_haiku_regen", rollup=pv["path"],
                       rollup_provenance_status=pv["provenance_status"],
                       rollup_generator_model_returned=pv["generator_model_returned"],
                       rollup_cl100k_tokens=pv["cl100k_tokens"])
            with OUT.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            errs.append(err)
    spent = (ptok * hr.PRICE_IN + ctok * hr.PRICE_OUT) / 1e6
    hr.log("answer+grade", calls, ptok, ctok, spent)
    print(f"done {time.time()-t0:.0f}s calls {calls} est ${spent:.3f} total ${hr.total():.3f} failed {errs}")


def med(arm):
    return statistics.median(r["cl100k_tokens"] for (a, _), r in PROV.items() if a == arm)


def analyse():
    hk = load("heldout-rollup-haiku/rollup_arm.jsonl")
    so = load("heldout-rollup/rollup_arm.jsonl")
    st = load("rq3_heldout_stripped_arm_extended.jsonl")
    summ, compare, pair = ns["summ"], ns["compare"], ns["pair"]
    ids = {r["record_id"] for r in hk}
    so = [r for r in so if r["record_id"] in ids]; st = [r for r in st if r["record_id"] in ids]
    def extra(a, b):
        out = {}
        for name, f in (("absent_pct", lambda rs: 100 * sum(r["grade"] == "absent" for r in rs) / len(rs)),
                        ("recall_pct", lambda rs: 100 * sum(r["grade"] in ("correct", "partially_correct") for r in rs) / len(rs))):
            import random
            rng = random.Random(ns["SEED"]); sess = {}
            for i, x in enumerate(a): sess.setdefault(x["session_id"], []).append(i)
            keys = list(sess); d = []
            for _ in range(ns["B"]):
                idx = [i for k in (rng.choice(keys) for _ in keys) for i in sess[k]]
                d.append(f([a[i] for i in idx]) - f([b[i] for i in idx]))
            d.sort()
            out[name + "_diff_pts"] = {"diff": round(f(a) - f(b), 1), "ci95": [round(d[int(.025 * (len(d) - 1))], 1), round(d[int(.975 * (len(d) - 1))], 1)]}
        return out
    res = {"n_questions": len(hk), "field": "grade", "plan": "heldout-rollup-haiku/analysis_plan.md",
           "rows": {"haiku_written_rollup": summ(hk), "sonnet_written_production_note": summ(so), "output_stripped": summ(st)},
           "rollup_median_cl100k_tokens": {"haiku_regen": med("heldout-haiku-regen"), "heldout_production_note": med("heldout-production"), "design_set_note": med("design-uniform")},
           "rollup_provenance": {"haiku_regen": "logged, model returned " + str(sorted({str(r["rollup_generator_model_returned"]) for r in hk})),
                                 "heldout_production": "inferred (not logged)"},
           "haiku_minus_output_stripped": dict(compare(*pair(hk, st)), **extra(*pair(hk, st))),
           "haiku_minus_sonnet_production_note": dict(compare(*pair(hk, so)), **extra(*pair(hk, so))),
           "protocol": hr.PROTOCOL, "bootstrap": {"B": ns["B"], "seed": ns["SEED"]}}
    RES.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if not a.analyse:
        run(a.dry_run)
    if not a.dry_run:
        analyse()
