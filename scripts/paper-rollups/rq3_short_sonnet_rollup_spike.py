#!/usr/bin/env python3
"""SPIKE: length-control arm for rq3_long_rollup_spike.py.

Identical to the long spike (same 8 sessions, same output-stripped input, same GEN_MODEL,
same answerer, same judge + majority rule, same frozen questions) except the generation
prompt is the UNCHANGED production rollup_summarise.build_summary_prompt() (production
per-section sentence counts, no length override). Isolates length from generator and input:
long Sonnet vs short Sonnet = length effect; short Sonnet vs Haiku-stripped-matchlen =
generator effect; short Sonnet vs production rollup = input/generator effect at matched length.

All logic is imported from rq3_long_rollup_spike; only prompt construction and output
paths differ.

Modes:
  --generate [--only SID...]   generate short rollups (resumable)
  --grade    [--only SID...]   answer + 3-draw judge (resumable)
  --analyse                    no model calls: table + paired tests + .md + summary json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import rq3_long_rollup_spike as L  # noqa: E402
from rq3_long_rollup_spike import (  # noqa: E402
    DATA, REC, mcnemar_exact, cluster_boot, ntok, rows, src, wilson, call_llm,
    strip_transcript, stripped_to_epoch, rs, CHUNK_BUDGET, PRICE_IN, PRICE_OUT, SEED,
)

GEN_OUT = DATA / "rq3_short_sonnet_rollup_spike.jsonl"
GRADED = DATA / "rq3_short_sonnet_rollup_spike_graded.jsonl"
SUMMARY_OUT = DATA / "rq3_short_sonnet_rollup_spike_summary.json"
MD_OUT = DATA / "rq3_short_sonnet_rollup_spike.md"
LONG_GEN = DATA / "rq3_long_rollup_spike.jsonl"
LONG_GRADED = DATA / "rq3_long_rollup_spike_graded.jsonl"
MATCHLEN = "rq3_spike_stripped_input_matchlen_graded.jsonl"
GEN_MAX_TOKENS = 4096

# route the imported grade() at this arm's files
L.GEN_OUT = GEN_OUT
L.GRADED = GRADED


def build_prompt(stripped: str) -> str:
    """Production build_summary_prompt() on the stripped input, no length edit."""
    chunks = rs.chunk_epoch(stripped_to_epoch(stripped), CHUNK_BUDGET)
    assert len(chunks) == 1, f"{len(chunks)} chunks"
    return rs.build_summary_prompt(chunks[0], CHUNK_BUDGET)


def sessions_from_long():
    """The same 8 sessions as the long spike, recovered from its output."""
    tr = {}
    for r in rows(src("rq3_questions_stripped_arm.jsonl")):
        tr.setdefault(r["session_id"], r["transcript"])
    out = []
    for g in rows(LONG_GEN):
        out.append({"session_id": g["session_id"], "tier": g.get("tier"), "raw_tokens": g["raw_tokens"],
                    "stripped_tokens": g["stripped_tokens"], "transcript": tr[g["session_id"]]})
    return sorted(out, key=lambda s: s["raw_tokens"])


def generate(sessions):
    done = {r["session_id"] for r in rows(GEN_OUT)} if GEN_OUT.exists() else set()
    for s in sessions:
        if s["session_id"] in done:
            continue
        raw = Path(s["transcript"]).read_text(errors="replace")
        stripped, _ = strip_transcript(raw)
        prompt = build_prompt(stripped)
        text, usage = "", {}
        for _try in range(3):   # no temperature: gateway rejects it on Claude 5
            text, usage = call_llm(prompt, model=L.GEN_MODEL, timeout=900, max_tokens=GEN_MAX_TOKENS)
            if text and usage.get("finish_reason") == "stop":
                break
            print(f"  attempt {_try + 1} finish_reason={usage.get('finish_reason')} empty={not text}")
            text = ""
        if not text:
            print(f"GENERATION FAILED {s['session_id']} {usage}")
            continue
        rec = {"session_id": s["session_id"], "tier": s.get("tier"), "model": L.GEN_MODEL,
               "prompt": "unchanged production build_summary_prompt()",
               "raw_tokens": s["raw_tokens"], "stripped_tokens": s["stripped_tokens"],
               "rollup_cl100k_tokens": ntok(text), "rollup_chars": len(text),
               "pct_of_raw": 100 * ntok(text) / s["raw_tokens"],
               "completion_tokens": usage.get("completion_tokens"), "usage": usage, "rollup_raw": text}
        with GEN_OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{s['session_id'][:50]} actual {rec['rollup_cl100k_tokens']} tok "
              f"({rec['pct_of_raw']:.2f}% of raw) usage {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}")


# ------------------------------------------------------------------ analysis
def load_matchlen(ids):
    out = {}
    for r in rows(src(MATCHLEN)):
        if r["record_id"] in ids:
            g = r.get("corrected_grade") or r.get("grade")
            out[r["record_id"]] = (g, (r.get("usage") or {}).get("prompt_tokens"))
    return out


def wilson_fmt(k, n):
    lo, hi = wilson(k, n)
    f = lambda x: f"{100 * x:.1f}" if x <= 1 else f"{x:.1f}"
    return f"{f(lo)}-{f(hi)}"


def analyse():
    gen = rows(GEN_OUT)
    lgen = rows(LONG_GEN)
    short = {r["record_id"]: r for r in rows(GRADED)}
    long_ = {r["record_id"]: r for r in rows(LONG_GRADED)}
    ids = sorted(long_)                      # the frozen 40
    sess = {rid: long_[rid]["session_id"] for rid in ids}

    def tup(d, rid):
        return (d[rid]["grade"] or "absent", d[rid]["usage"].get("prompt_tokens"))

    grades = {
        "short Sonnet rollup (new)": {r: tup(short, r) for r in ids if r in short},
        "long Sonnet rollup": {r: tup(long_, r) for r in ids},
        "production rollup": L.load_arm("production rollup", ids),
        "Haiku-from-stripped matched-length": load_matchlen(set(ids)),
        "output-stripped": L.load_arm("output-stripped", ids),
        "prose-only": L.load_arm("prose-only", ids),
    }

    def stats(g):
        n = len(g)
        k = sum(v[0] in REC for v in g.values())
        toks = [v[1] for v in g.values() if v[1]]
        return {"n": n, "recoverable": k, "recoverable_pct": round(100 * k / n, 1) if n else None,
                "wilson95": wilson(k, n) if n else None, "wilson_str": wilson_fmt(k, n) if n else "-",
                "median_read_tokens": statistics.median(toks) if toks else None}

    arms = {lab: stats(g) for lab, g in grades.items()}

    S, LG = "short Sonnet rollup (new)", "long Sonnet rollup"
    pairs_def = [("long vs short Sonnet (length effect)", LG, S),
                 ("short Sonnet vs Haiku-stripped-matchlen (generator effect)", S, "Haiku-from-stripped matched-length"),
                 ("short Sonnet vs production rollup", S, "production rollup")]
    paired = {}
    for name, a_lab, b_lab in pairs_def:
        common = [r for r in ids if r in grades[a_lab] and r in grades[b_lab]]
        missing = len(ids) - len(common)
        a = {r: grades[a_lab][r][0] in REC for r in common}
        b = {r: grades[b_lab][r][0] in REC for r in common}
        a_only = sum(a[r] and not b[r] for r in common)
        b_only = sum(b[r] and not a[r] for r in common)
        lo, hi = cluster_boot({r: (a[r], b[r]) for r in common}, sess) if common else (0, 0)
        paired[name] = {"a": a_lab, "b": b_lab, "n": len(common), "missing_dropped": missing,
                        "both": sum(a[r] and b[r] for r in common), "a_only": a_only, "b_only": b_only,
                        "neither": sum(not a[r] and not b[r] for r in common),
                        "diff_pp": round(100 * (sum(a.values()) - sum(b.values())) / len(common), 1) if common else None,
                        "mcnemar_exact_p": round(mcnemar_exact(a_only, b_only), 4),
                        "session_cluster_bootstrap95_pp": [round(lo, 1), round(hi, 1)]}

    gm = lambda rs_, k: statistics.median(g[k] for g in rs_)
    generation = {"short_median_tokens": gm(gen, "rollup_cl100k_tokens"), "long_median_tokens": gm(lgen, "rollup_cl100k_tokens"),
                  "short_median_pct_of_raw": round(gm(gen, "pct_of_raw"), 3), "long_median_pct_of_raw": round(gm(lgen, "pct_of_raw"), 3),
                  "short_median_completion_tokens": gm(gen, "completion_tokens"),
                  "long_median_completion_tokens": gm(lgen, "completion_tokens"),
                  "per_session": [{"session_id": g["session_id"], "raw_tokens": g["raw_tokens"],
                                   "short_tokens": g["rollup_cl100k_tokens"],
                                   "long_tokens": next(x["rollup_cl100k_tokens"] for x in lgen if x["session_id"] == g["session_id"])}
                                  for g in gen]}

    u_in = sum(g["usage"].get("prompt_tokens", 0) for g in gen)
    u_out = sum(g["usage"].get("completion_tokens", 0) for g in gen)
    a_in = sum(r["usage"].get("prompt_tokens", 0) for r in short.values())
    a_out = sum(r["usage"].get("completion_tokens", 0) for r in short.values())
    n_judge = sum(len(r["draws"]) for r in short.values())
    spend = {"generation": u_in * PRICE_IN + u_out * PRICE_OUT, "answering": a_in * PRICE_IN + a_out * PRICE_OUT,
             "judging_estimated": n_judge * 450 * PRICE_IN + n_judge * 5 * PRICE_OUT, "judge_calls": n_judge}
    spend["total"] = round(spend["generation"] + spend["answering"] + spend["judging_estimated"], 2)

    summary = {"n_questions": len(ids), "n_sessions": len(set(sess.values())), "arms": arms, "paired": paired,
               "generation": generation, "spend_usd_estimated_at_3_15_per_M": spend,
               "single_draw_sensitivity_short_pct": round(100 * sum(short[r]["single_draw_grade"] in REC for r in ids if r in short) / len(short), 1)}
    SUMMARY_OUT.write_text(json.dumps(summary, indent=1, default=list))

    # ---- markdown
    def rowf(lab, s):
        return f"| {lab} | {s['n']} | {s['recoverable_pct']}% | {s['wilson_str']} | {s['median_read_tokens']:,.0f} |"

    order = [S, LG, "production rollup", "Haiku-from-stripped matched-length", "output-stripped", "prose-only"]
    md = [f"# RQ3 spike: short Sonnet rollup, length control for the long spike (8 sessions, {len(ids)} questions, direction only)", "",
          f"Generator claude-sonnet-5, input the output-stripped transcript, prompt the UNCHANGED production build_summary_prompt() (production per-section sentence counts, no length override, no temperature). Everything else is the long spike: same 8 sessions, same answerer, same judge with 3-draw majority, same {len(ids)} frozen questions. Denominator for every row is those {len(ids)} questions on 8 sessions; arms with fewer rows are listed with their n.", "",
          f"Median generated tokens (cl100k): short {generation['short_median_tokens']:,.0f} ({generation['short_median_pct_of_raw']}% of raw) vs long {generation['long_median_tokens']:,.0f} ({generation['long_median_pct_of_raw']}% of raw), a {generation['long_median_tokens'] / generation['short_median_tokens']:.1f}x ratio. Median completion tokens: short {generation['short_median_completion_tokens']:,.0f}, long {generation['long_median_completion_tokens']:,.0f}.", "",
          "| arm | n | recoverable | 95% CI (Wilson) | median read tok |", "|---|---|---|---|---|"]
    md += [rowf(l, arms[l]) for l in order]
    md += ["", "Haiku-from-stripped matched-length is `rq3_spike_stripped_input_matchlen_graded.jsonl` restricted to the 40 questions, using its `corrected_grade` (single answer, single judge). Production rollup and output-stripped use their three-draw majority files; prose-only is single-draw graded.", "",
           "Paired exact McNemar (recoverable = correct or partially correct):", "",
           "| comparison | n | A only | B only | both | neither | A minus B | p |", "|---|---|---|---|---|---|---|---|"]
    for name, p in paired.items():
        md.append(f"| {name} | {p['n']} | {p['a_only']} | {p['b_only']} | {p['both']} | {p['neither']} | {p['diff_pp']:+.1f} pp | {p['mcnemar_exact_p']} |")
    md += ["", "A is the first arm named in each comparison. Questions missing from a comparison arm: " +
           ", ".join(f"{n.split(' (')[0]}: {p['missing_dropped']}" for n, p in paired.items()) + ".", "",
           f"Session-cluster bootstrap 95% (pp, 10,000 resamples, seed 42): " +
           "; ".join(f"{n.split(' (')[0]} {p['session_cluster_bootstrap95_pp'][0]} to {p['session_cluster_bootstrap95_pp'][1]}" for n, p in paired.items()) + ".", "",
           f"Caveats: 8 sessions, direction only. Short-arm single-draw sensitivity {summary['single_draw_sensitivity_short_pct']}% recoverable. Estimated spend ${spend['total']} at 3/15 per M tokens.", ""]
    MD_OUT.write_text("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    for f in ("generate", "grade", "analyse"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    ss = sessions_from_long()
    if a.only:
        ss = [s for s in ss if any(o in s["session_id"] for o in a.only)]
    if a.generate:
        generate(ss)
    if a.grade:
        L.grade(ss)
    if a.analyse:
        analyse()


if __name__ == "__main__":
    main()
