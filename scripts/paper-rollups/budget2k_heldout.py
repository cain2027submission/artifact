#!/usr/bin/env python3
"""2,000-token Sonnet rollup + equal-length rule cut on the 125 held-out questions.
Plan: data/paper-rollups/budget2k-heldout/analysis_plan.md (committed before any call, binding).
Reuses sonnet1k_heldout.py unchanged by re-pointing its module paths and the length target
(R.MATCHED_TARGET 932 -> 2000, R.MATCHED_MAX_TOKENS 3000 -> 5000). Internal arm names stay
'sonnet1k'/'cut' inside the reused code; rows are relabelled sonnet2k/cut2k on disk after each run.

Usage: budget2k_heldout.py --generate [--smoke] | --answer {sonnet2k,cut2k} [--smoke] | --provenance | --analyse
       add --dry-run for the estimate only.
"""
import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sonnet1k_heldout as S  # noqa: E402

R, L = S.R, S.L
OD = S.DATA / "budget2k-heldout"
OLD = S.DATA / "sonnet1k-heldout"
R.MATCHED_TARGET = 2000
R.MATCHED_MAX_TOKENS = 5000
S.OD, S.ROLLUPS, S.GENLOG, S.COST, S.RESULTS = OD, OD / "rollups", OD / "generations.jsonl", OD / "cost_log.jsonl", OD / "results.json"
S.CAP_USD = 14.0
S.B, S.SEED = 10000, 20261014
FILES = {"sonnet2k": ("sonnet1k", OD / "arm_sonnet2k.jsonl"), "cut2k": ("cut", OD / "arm_cut2k.jsonl")}
S.ARMS = {"sonnet1k": FILES["sonnet2k"][1], "cut": FILES["cut2k"][1]}


def relabel():
    for new, (old, p) in FILES.items():
        rs = S.rows(p)
        if any(r["arm"] != new for r in rs):
            for r in rs:
                r["arm"] = new
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))


def provenance():
    have = {(r["session_id"], r["arm"]) for r in S.rows(S.PROV)}
    gens = {}
    for r in S.rows(S.GENLOG):
        gens.setdefault(r["session_id"], []).append(r)
    n = 0
    with S.PROV.open("a") as fh:
        for sid, g in sorted(gens.items()):
            ok = [x for x in g if x.get("ok")]
            if not ok or (sid, "sonnet2k-heldout-regen") in have:
                continue
            p = S.ROLLUPS / f"{sid}.md"
            b = p.read_bytes()
            m = ok[-1]
            fh.write(json.dumps({
                "session_id": sid, "arm": "sonnet2k-heldout-regen", "path": str(p.relative_to(S.ROOT)),
                "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "cl100k_tokens": L.ntok(b.decode()),
                "note_date": sid[:10], "generator_model_requested": m["model_requested"],
                "generator_model_returned": m["model_returned"], "provenance_status": "logged",
                "evidence": f"budget2k-heldout/generations.jsonl: {len(g)} call row(s), finish_reason {m['finish_reason']}, "
                            f"prompt {m['usage'].get('prompt_tokens')} completion {m['usage'].get('completion_tokens')}"}) + "\n")
            n += 1
    print(f"provenance rows appended: {n}")


def analyse():
    a2, c2 = S.rows(FILES["sonnet2k"][1]), S.rows(FILES["cut2k"][1])
    ref = {"sonnet1k": S.rows(OLD / "arm_sonnet1k.jsonl"), "cut1k": S.rows(OLD / "arm_cut.jsonl"),
           "cut_prod": S.rows(OLD / "arm_cut_prod.jsonl"),
           "output_stripped": S.rows(S.DATA / "rq3_heldout_stripped_arm_extended.jsonl"),
           "production_note": S.rows(S.DATA / "heldout-rollup" / "rollup_arm.jsonl")}
    gens = [r for r in S.rows(S.GENLOG) if r.get("ok")]
    toks = sorted(r["rollup_cl100k_tokens"] for r in gens)
    med = (toks[len(toks) // 2] if len(toks) % 2 else (toks[len(toks) // 2 - 1] + toks[len(toks) // 2]) / 2) if toks else None
    cutt = sorted(r["artifact_cl100k_tokens"] for r in c2)
    arms = {"sonnet2k": a2, "cut2k": c2, **ref}
    res = {"arms": {k: S.summ(v) for k, v in arms.items() if v}, "cap_usd": S.CAP_USD, "spend_usd_assumed_3_15": round(S.total(), 3),
           "generation": {"sessions": len(gens), "rollup_tokens_median": med, "min": toks[0] if toks else None,
                          "max": toks[-1] if toks else None, "per_session": {r["session_id"][:30]: r["rollup_cl100k_tokens"] for r in gens},
                          "models_returned": sorted({str(r.get("model_returned")) for r in gens}),
                          "finish_reasons": dict(S.Counter(r["finish_reason"] for r in gens))},
           "cut2k_artifact_tokens": {"median": cutt[len(cutt) // 2] if cutt else None, "min": cutt[0] if cutt else None, "max": cutt[-1] if cutt else None},
           "paired": {}}
    for name, (x, y) in {"sonnet2k_minus_cut2k": ("sonnet2k", "cut2k"), "sonnet2k_minus_sonnet1k": ("sonnet2k", "sonnet1k"),
                         "cut2k_minus_cut1k": ("cut2k", "cut1k"), "sonnet2k_minus_output_stripped": ("sonnet2k", "output_stripped"),
                         "sonnet2k_minus_production_note": ("sonnet2k", "production_note")}.items():
        if arms.get(x) and arms.get(y):
            res["paired"][name] = S.compare(arms[x], arms[y])
    if "sonnet2k_minus_cut2k" in res["paired"]:
        res["verdict_primary_recall"] = S.verdict(res["paired"]["sonnet2k_minus_cut2k"])
    r = res["arms"]
    if "sonnet2k" in r and "cut2k" in r:
        d = res["paired"]["sonnet2k_minus_cut2k"]["recall"]["diff_pts"]
        res["predictions"] = {
            "1_recall_rises_both": {"sonnet2k": r["sonnet2k"]["recall_pct"], "ref_1k": 48.8, "cut2k": r["cut2k"]["recall_pct"], "ref_cut_1k": 42.4,
                                    "held": r["sonnet2k"]["recall_pct"] > 48.8 and r["cut2k"]["recall_pct"] > 42.4},
            "2_below_output_stripped": {"sonnet2k": r["sonnet2k"]["recall_pct"], "ref": 82.4, "held": r["sonnet2k"]["recall_pct"] < 82.4},
            "3_within_10pts": {"diff_pts": d, "held": abs(d) <= 10}}
    S.RESULTS.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--answer", choices=list(FILES))
    ap.add_argument("--provenance", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    OD.mkdir(parents=True, exist_ok=True)
    qs = S.questions(a.smoke)
    if a.generate:
        S.generate(qs, a.dry_run)
    if a.answer:
        S.answer(FILES[a.answer][0], qs, a.dry_run)
        relabel()
    if a.provenance:
        provenance()
    if a.analyse:
        analyse()


if __name__ == "__main__":
    main()
