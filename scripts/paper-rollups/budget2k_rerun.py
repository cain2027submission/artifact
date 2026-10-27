#!/usr/bin/env python3
"""Calibrated 2,000-token rollup rerun. Plan: data/paper-rollups/budget2k-heldout/rerun_plan.md (binding).
Stages: --setup | --calibrate N | --freeze N | --generate [--smoke] | --answer {sonnet2kc,cut2kc} [--smoke] | --provenance | --analyse
"""
import argparse, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sonnet1k_heldout as S  # noqa: E402

R, L = S.R, S.L
OD = S.DATA / "budget2k-heldout"
OLD = S.DATA / "sonnet1k-heldout"
S.OD, S.ROLLUPS, S.GENLOG = OD, OD / "rollups_calibrated", OD / "generations_rerun.jsonl"
S.COST, S.RESULTS = OD / "cost_log_rerun.jsonl", OD / "results_rerun.json"
S.CAP_USD = 14.0  # raised from 9 by rerun_amendment_1.md
S.B, S.SEED = 10000, 20261014
R.MATCHED_TARGET, R.MATCHED_MAX_TOKENS = 2000, 6000  # held-out stages set 12000 after the freeze
FILES = {"sonnet2kc": ("sonnet1k", OD / "arm_sonnet2kc.jsonl"), "cut2kc": ("cut", OD / "arm_cut2kc.jsonl")}
S.ARMS = {"sonnet1k": FILES["sonnet2kc"][1], "cut": FILES["cut2kc"][1]}
CALSESS, CALLOG = OD / "calibration_sessions.json", OD / "calibration_log.md"
FROZEN, FROZEN_SHA = OD / "frozen_length_instruction.txt", OD / "frozen_length_instruction.sha256"
CALGEN = OD / "calibration_generations.jsonl"
CANDS = [
    "\n\nLength: write a note of at least 1,800 and at most 2,300 cl100k tokens (roughly 1,350 to 1,725 words) in total across the intent, thinking and outcome fields combined. Keep writing until that length is reached; do not stop early. Each field should be several substantial paragraphs. Be selective about what is decisive, but include the decisive facts, numbers, identifiers, files and dead ends in specific detail.",
    "\n\nLength: write a note of at least 1,800 and at most 2,300 cl100k tokens (roughly 1,350 to 1,725 words) in total across the intent, thinking and outcome fields combined. Keep writing until that length is reached; output shorter than 1,800 tokens is a failure. Expand every subsection of every field with specifics: exact numbers, identifiers, file names, commands, the alternatives that were rejected and why, and dead ends. Each field should be several substantial paragraphs.",
    "\n\nLength: min_tokens: 2000. Write a note of at least 2,000 and at most 2,300 cl100k tokens (roughly 1,500 to 1,725 words) in total across the intent, thinking and outcome fields combined; output shorter than 2,000 tokens is a failure and you must keep writing until the length is reached. Use exactly three fields, intent, thinking and outcome, and write at least five paragraphs in each field, roughly 130 to 150 words per paragraph. Fill every paragraph with specifics: exact numbers, identifiers, file names, commands, the alternatives that were rejected and why, and dead ends.",
]


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def setup():
    ids = sorted({json.loads(l)["session_id"] for l in (S.DATA / "rq3_questions.jsonl").read_text().splitlines() if l.strip()})[:6]
    CALSESS.write_text(json.dumps({"rule": "6 lexicographically lowest session ids of the design set (rq3_questions.jsonl)", "session_ids": ids}, indent=1) + "\n")
    md = ["# calibration log, written before any call", "", "Sessions: calibration_sessions.json. Rule: first candidate with median realised length 1,800 to 2,300 cl100k tokens on the 6 sessions; stop if none.", ""]
    for i, c in enumerate(CANDS, 1):
        md += [f"## candidate {i}", "", "```", c.strip(), "```", ""]
    CALLOG.write_text("\n".join(md) + "\n")
    print(ids)


def calibrate(n):
    ids = json.loads(CALSESS.read_text())["session_ids"]
    R.LENGTH_INSTR = CANDS[min(n, 3) - 1]  # candidate 4 = candidate 3 text unchanged, max_tokens 12000 (amendment 1)
    R.MATCHED_MAX_TOKENS = 12000 if n == 4 else 6000
    infos = {}
    est_in = 0
    for s in ids:
        raw = S.transcript(s).read_text(errors="replace")
        stripped, _ = L.strip_transcript(raw)
        infos[s] = stripped
        est_in += L.ntok(stripped) * 1.15 + 1500
    est = (est_in * 3 + 6 * 2800 * 15) / 1e6
    print(f"candidate {n}: estimate ${est:.2f}, running ${S.total():.2f}, cap {S.CAP_USD}")
    if S.total() + est > S.CAP_USD:
        sys.exit("STOP cap")

    def one(s):
        prompt = R.build_prompt_matched(infos[s], R.MATCHED_TARGET)
        d = S.post(f"{S.GATEWAY}/v1/chat/completions", {"model": L.GEN_MODEL, "max_tokens": R.MATCHED_MAX_TOKENS,
                   "messages": [{"role": "user", "content": prompt}]}, timeout=900)
        ch = d["choices"][0]; u = d.get("usage", {}); t = ch["message"]["content"] or ""
        return {"candidate": n, "session_id": s, "tokens": L.ntok(t), "finish_reason": ch.get("finish_reason"), "model_returned": d.get("model"), "usage": u}
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(one, ids))
    with CALGEN.open("a") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    S.logcost(f"calibrate{n}", len(res), sum(r["usage"].get("prompt_tokens", 0) for r in res), sum(r["usage"].get("completion_tokens", 0) for r in res))
    t = sorted(r["tokens"] for r in res)
    med = (t[2] + t[3]) / 2
    ok = 1800 <= med <= 2300
    print(f"candidate {n} lengths {t} median {med} in_range {ok} total ${S.total():.2f}")
    with CALLOG.open("a") as fh:
        fh.write(f"\n## result candidate {n}\n\nlengths {t}, median {med}, in range 1,800 to 2,300: {ok}. models {sorted({r['model_returned'] for r in res})}. finish {[r['finish_reason'] for r in res]}\n")


def freeze(n):
    FROZEN.write_text(CANDS[n - 1]); FROZEN_SHA.write_text(sha(CANDS[n - 1]) + "\n")
    print("frozen", n, sha(CANDS[n - 1]))


def generate(qs, dry):
    t = FROZEN.read_text()
    assert sha(t) == FROZEN_SHA.read_text().strip()
    R.LENGTH_INSTR = t
    S.generate(qs, dry)


def provenance():
    have = {(r["session_id"], r["arm"]) for r in S.rows(S.PROV)}
    gens = {}
    for r in S.rows(S.GENLOG):
        gens.setdefault(r["session_id"], []).append(r)
    n = 0
    with S.PROV.open("a") as fh:
        for sid, g in sorted(gens.items()):
            ok = [x for x in g if x.get("ok")]
            if not ok or (sid, "sonnet2k-calibrated-heldout-regen") in have:
                continue
            p = S.ROLLUPS / f"{sid}.md"; b = p.read_bytes(); m = ok[-1]
            fh.write(json.dumps({"session_id": sid, "arm": "sonnet2k-calibrated-heldout-regen", "path": str(p.relative_to(S.ROOT)),
                "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "cl100k_tokens": L.ntok(b.decode()), "note_date": sid[:10],
                "generator_model_requested": m["model_requested"], "generator_model_returned": m["model_returned"], "provenance_status": "logged",
                "evidence": f"budget2k-heldout/generations_rerun.jsonl: {len(g)} call row(s), finish_reason {m['finish_reason']}, frozen instruction sha256 {FROZEN_SHA.read_text().strip()[:16]}"}) + "\n")
            n += 1
    print("provenance rows appended:", n)


def relabel():
    for new, (old, p) in FILES.items():
        rs = S.rows(p)
        if any(r["arm"] != new for r in rs):
            for r in rs:
                r["arm"] = new
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))


def analyse():
    ra = lambda f: S.rows(f)
    a, c = ra(FILES["sonnet2kc"][1]), ra(FILES["cut2kc"][1])
    ref = {"sonnet_1.3k": ra(OD / "arm_sonnet2k.jsonl"), "cut_1.3k": ra(OD / "arm_cut2k.jsonl"), "sonnet1k": ra(OLD / "arm_sonnet1k.jsonl"),
           "cut1k": ra(OLD / "arm_cut.jsonl"), "output_stripped": ra(S.DATA / "rq3_heldout_stripped_arm_extended.jsonl"),
           "production_note": ra(S.DATA / "heldout-rollup" / "rollup_arm.jsonl")}
    gens = [r for r in S.rows(S.GENLOG) if r.get("ok")]
    toks = sorted(r["rollup_cl100k_tokens"] for r in gens)
    med = (toks[len(toks) // 2] if len(toks) % 2 else (toks[len(toks) // 2 - 1] + toks[len(toks) // 2]) / 2)
    cutt = sorted(r["artifact_cl100k_tokens"] for r in c)
    arms = {"sonnet2kc": a, "cut2kc": c, **ref}
    res = {"arms": {k: S.summ(v) for k, v in arms.items() if v}, "cap_usd": S.CAP_USD, "spend_usd_assumed_3_15": round(S.total(), 3),
           "frozen_instruction_sha256": FROZEN_SHA.read_text().strip(),
           "generation": {"sessions": len(gens), "median": med, "min": toks[0], "max": toks[-1], "tokens": toks,
                          "models_returned": sorted({str(r.get("model_returned")) for r in gens}), "finish_reasons": dict(S.Counter(r["finish_reason"] for r in gens))},
           "cut2kc_artifact_tokens": {"median": cutt[len(cutt) // 2], "min": cutt[0], "max": cutt[-1]}, "paired": {}}
    for name, (x, y) in {"sonnet2kc_minus_cut2kc": ("sonnet2kc", "cut2kc"), "sonnet2kc_minus_sonnet_1.3k": ("sonnet2kc", "sonnet_1.3k"),
                         "sonnet2kc_minus_sonnet1k": ("sonnet2kc", "sonnet1k"), "cut2kc_minus_cut_1.3k": ("cut2kc", "cut_1.3k"),
                         "cut2kc_minus_cut1k": ("cut2kc", "cut1k"), "sonnet2kc_minus_output_stripped": ("sonnet2kc", "output_stripped"),
                         "sonnet2kc_minus_production_note": ("sonnet2kc", "production_note")}.items():
        if arms.get(x) and arms.get(y):
            res["paired"][name] = S.compare(arms[x], arms[y])
    if "sonnet2kc_minus_cut2kc" in res["paired"]:
        res["verdict_primary_recall"] = S.verdict(res["paired"]["sonnet2kc_minus_cut2kc"])
        r = res["arms"]; d = res["paired"]["sonnet2kc_minus_cut2kc"]["recall"]["diff_pts"]
        res["predictions"] = {"1_median_in_1800_2300": {"median": med, "held": 1800 <= med <= 2300},
            "2_recall_above_53.6": {"v": r["sonnet2kc"]["recall_pct"], "held": r["sonnet2kc"]["recall_pct"] > 53.6},
            "3_below_82.4": {"v": r["sonnet2kc"]["recall_pct"], "held": r["sonnet2kc"]["recall_pct"] < 82.4},
            "4_within_10": {"diff_pts": d, "held": abs(d) <= 10}}
    S.RESULTS.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true"); ap.add_argument("--calibrate", type=int); ap.add_argument("--freeze", type=int)
    ap.add_argument("--generate", action="store_true"); ap.add_argument("--answer", choices=list(FILES))
    ap.add_argument("--provenance", action="store_true"); ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.setup: setup()
    if a.calibrate: calibrate(a.calibrate)
    if a.freeze: freeze(a.freeze)
    if a.generate: generate(S.questions(a.smoke), a.dry_run)
    if a.answer:
        t = FROZEN.read_text(); R.LENGTH_INSTR = t
        S.answer(FILES[a.answer][0], S.questions(a.smoke), a.dry_run); relabel()
    if a.provenance: provenance()
    if a.analyse: analyse()


if __name__ == "__main__":
    main()
