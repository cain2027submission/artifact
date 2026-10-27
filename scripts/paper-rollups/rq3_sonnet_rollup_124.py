#!/usr/bin/env python3
"""Sonnet-written rollup arms on ALL 124 RQ3 questions / 25 sessions.

Extends rq3_long_rollup_spike.py (5%-of-raw prompt, "long") and
rq3_short_sonnet_rollup_spike.py (unchanged production prompt, "short") from the
8-session spike to the full frozen set (rq3_questions_stripped_arm.jsonl's 25 sessions;
questions from rq3_rollup_uniform.jsonl). Generator, input (output-stripped transcript),
answerer, judge (3-draw majority) and parsers are the spikes' own, imported not copied.
The 8 spike sessions' already-written rollups and grades are seeded into the new output
files (identical protocol), so only the other 17 sessions cost model calls.

Third variant "matched": the long prompt with an explicit fixed length target (MATCHED_TARGET
tokens, the production Haiku rollup's cl100k mean of 932) so a Sonnet rollup lands at the
production rollup's size. Same generator, input, answerer, judge (3-draw majority, 2 more on a
1-1-1 split) as long/short and as the published production-rollup arm (rq3_majority_uniform).

Modes: --estimate --generate --grade --analyse   [--variant long|short|matched] [--only SID..]
"""
import argparse, json, re, shutil, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import rq3_long_rollup_spike as L          # noqa: E402
import rq3_short_sonnet_rollup_spike as S  # noqa: E402
import cluster_bootstrap as cb             # noqa: E402
import numpy as np                         # noqa: E402

DATA = L.DATA
REC = L.REC
REFUSAL = re.compile(r"^\s*[*_`#>\s]*NOT[ _]ANSWERABLE\b", re.IGNORECASE)
FILES = {v: (DATA / f"rq3_sonnet_rollup_124_{v}.jsonl", DATA / f"rq3_sonnet_rollup_124_{v}_graded.jsonl")
         for v in ("long", "short", "matched", "matched_haiku", "matched_haiku_v2")}
SEEDS = {"long": (DATA / "rq3_long_rollup_spike.jsonl", DATA / "rq3_long_rollup_spike_graded.jsonl"),
         "short": (DATA / "rq3_short_sonnet_rollup_spike.jsonl", DATA / "rq3_short_sonnet_rollup_spike_graded.jsonl")}
CACHE = DATA / "rq3_sonnet_rollup_124_sessions.json"   # holds transcript paths: do not commit
L.SESSIONS_CACHE = CACHE


MATCHED_TARGET = 932   # production Haiku rollup mean cl100k tokens (spinoff_frontier_bootstrap.py)
MATCHED_MAX_TOKENS = 3000
# Generator option for the matched variant: --generator haiku writes the same prompt/input with the
# production rollup model (rs.model_for("cli")); outputs go to the "matched_haiku" files.
HAIKU_MODEL = "claude-haiku-4-5@20251001"
HAIKU_PRICE_IN, HAIKU_PRICE_OUT = 1.00 / 1e6, 5.00 / 1e6


# Haiku v2 (the single retune): Haiku under-shot the plain instruction (mean 689 vs 932 target), so the
# model is shown a longer figure and told short answers are too short. Nothing else differs.
LENGTH_INSTR_HAIKU_V2 = ("\n\nLength: this note must be {n} tokens (roughly {w} words) in total across the intent, thinking and "
                         "outcome fields combined: never fewer than {lo} tokens and no more than {hi}. Aim for the middle of that "
                         "range. Haiku-length answers of a few short sentences are too short: write each field as two to four full "
                         "paragraphs. Be selective: keep the decisive facts, numbers, identifiers, files and dead ends, and drop the rest.")
HAIKU_V2_PROMPTED_TARGET = 1700


def build_prompt_matched(stripped, n, instr=None):
    """Long variant's prompt with the sentence counts removed and an explicit, tighter length target."""
    epoch = L.stripped_to_epoch(stripped)
    chunks = L.rs.chunk_epoch(epoch, L.CHUNK_BUDGET)
    assert len(chunks) == 1, f"{len(chunks)} chunks"
    p = L.rs.build_summary_prompt(chunks[0], L.CHUNK_BUDGET)
    head, sep, tail = p.partition("\n\nTranscript:\n")
    head = re.sub(r" \(\d-\d sentences\)", "", head)
    head += (instr or LENGTH_INSTR).format(n=n, w=int(n * 0.75), lo=int(n * 0.9), hi=int(n * 1.1))
    return head + sep + tail


LENGTH_INSTR = ("\n\nLength: write about {n} tokens (roughly {w} words) in total across the intent, thinking and "
                "outcome fields combined: never fewer than {lo} tokens and no more than {hi}. Each field should "
                "be a substantial paragraph or two. Be selective: keep the decisive facts, numbers, "
                "identifiers, files and dead ends, and drop the rest.")


def generate_matched(sessions, key="matched", model=None, instr=None, prompted=None):
    model = model or L.GEN_MODEL
    done = {r["session_id"] for r in L.rows(FILES[key][0])} if FILES[key][0].exists() else set()
    for s in sessions:
        if s["session_id"] in done:
            continue
        raw = Path(s["transcript"]).read_text(errors="replace")
        stripped, _ = L.strip_transcript(raw)
        prompt = build_prompt_matched(stripped, prompted or MATCHED_TARGET, instr)
        text, usage = "", {}
        for _try in range(3):
            text, usage = L.call_llm(prompt, model=model, timeout=900, max_tokens=MATCHED_MAX_TOKENS)
            if text and usage.get("finish_reason") == "stop":
                break
            text = ""
        if not text:
            print(f"GENERATION FAILED {s['session_id']} {usage}")
            continue
        rec = {"session_id": s["session_id"], "model": model, "raw_tokens": s["raw_tokens"],
               "stripped_tokens": s["stripped_tokens"], "target_tokens": MATCHED_TARGET,
               "rollup_cl100k_tokens": L.ntok(text), "rollup_chars": len(text),
               "pct_of_raw": 100 * L.ntok(text) / s["raw_tokens"], "completion_tokens": usage.get("completion_tokens"),
               "usage": usage, "rollup_raw": text}
        with FILES[key][0].open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{s['session_id'][:50]} target {MATCHED_TARGET} actual {rec['rollup_cl100k_tokens']} usage {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}")


def seed(v):
    if v not in SEEDS:
        return
    for dst, srcp in zip(FILES[v], SEEDS[v]):
        if not dst.exists() and srcp.exists():
            shutil.copy(srcp, dst)


def route(v):
    L.GEN_OUT, L.GRADED = FILES[v]
    S.GEN_OUT = FILES[v][0]


def all_sessions(only=None):
    t = L.session_table()
    ss = sorted(t.values(), key=lambda s: s["raw_tokens"])
    if only:
        ss = [s for s in ss if any(o in s["session_id"] for o in only)]
    return ss


def estimate(ss):
    gi = go = 0
    for s in ss:
        n = L.target_for(s)[0]
        gi += s["stripped_tokens"] * 1.15 + 1500
        go += n * 1.1
    qs = sum(s["n_questions"] for s in ss)
    ai = sum(s["n_questions"] * (min(L.target_for(s)[0], s["raw_tokens"]) * 1.2 + 300) for s in ss)
    est = gi * L.PRICE_IN + go * L.PRICE_OUT + ai * L.PRICE_IN + qs * 4 * 450 * L.PRICE_IN + qs * 400 * L.PRICE_OUT
    print(f"LONG: sessions {len(ss)} gen in {gi:,.0f} out {go:,.0f}; answer in {ai:,.0f}; q {qs}; est ${est:.2f} (before seeding discount)")


def spend_of(gen, graded, gen_prices=None):
    pi, po = L.PRICE_IN, L.PRICE_OUT
    gpi, gpo = gen_prices or (pi, po)
    g = sum(x["usage"].get("prompt_tokens", 0) * gpi + x["usage"].get("completion_tokens", 0) * gpo for x in gen)
    a = sum(x["usage"].get("prompt_tokens", 0) * pi + x["usage"].get("completion_tokens", 0) * po for x in graded)
    nj = sum(len(x["draws"]) for x in graded)
    j = nj * 450 * pi + nj * 5 * po
    return {"generation": round(g, 2), "answering": round(a, 2), "judging_est": round(j, 2), "total": round(g + a + j, 2)}


def series_new(v):
    return {r["record_id"]: (r["session_id"], int((r["grade"] or "absent") in REC))
            for r in L.rows(FILES[v][1])}


def counts(v):
    """(n, recoverable, wrong, attempted) of the revised-grader 3-draw grades.
    No separate corrected stage: the grader applies the refusal rule itself."""
    gr = [r["grade"] or "absent" for r in L.rows(FILES[v][1])]
    return (len(gr), sum(g in REC for g in gr), sum(g == "wrong" for g in gr),
            len(gr) - sum(g == "absent" for g in gr))


VARIANTS = ("long", "short", "matched", "matched_haiku", "matched_haiku_v2")


def vlabel(v):
    return {"matched_haiku": "matched haiku", "matched_haiku_v2": "matched haiku v2"}.get(v) or f"{v} sonnet"


def analyse():
    Sx = cb.build()
    for v in VARIANTS:
        if FILES[v][1].exists():
            Sx[vlabel(v)] = series_new(v)
    # original pooled production-rollup arm restricted to the same 124 records
    import hashlib   # rq3_questions.jsonl has no record_id: regrade_s4.qid()
    pooled = {hashlib.sha256(f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}".encode()).hexdigest()[:16]: r
              for r in L.rows(DATA / "rq3_questions.jsonl")}
    Sx["rollup (original pooled)"] = {rid: (pooled[rid]["session_id"], int(pooled[rid]["grade"] in REC)) for rid in Sx["rollup"]}
    sessions = sorted({s for ser in Sx.values() for s, _ in ser.values()})
    bt = cb.Boot(Sx, sessions)
    res = {"arms": {}, "paired": {}}
    # comparator wrong-of-attempted from arm files
    def comp(f, corrected_key):
        rr = L.rows(DATA / f)
        n = len(rr)
        gs = [r["grade"] for r in rr]
        return n, sum(g in REC for g in gs), sum(g == "wrong" for g in gs), n - sum(g == "absent" for g in gs)
    cmp_ = {"production rollup": comp("rq3_rollup_uniform.jsonl", "uniform"),
            "output-stripped": comp("rq3_questions_stripped_arm.jsonl", "stripped")}
    tok = {"production rollup": statistics.mean(r["usage"]["prompt_tokens"] for r in L.rows(DATA / "rq3_rollup_uniform.jsonl")),
           "output-stripped": statistics.mean(r["usage"]["prompt_tokens"] for r in L.rows(DATA / "rq3_questions_stripped_arm.jsonl"))}
    for lab, (n, rec, wr, att) in cmp_.items():
        res["arms"][lab] = {"n": n, "recoverable": rec, "pct": round(100 * rec / n, 1), "wrong": wr, "attempted": att,
                            "wrong_of_attempted_pct": round(100 * wr / att, 1), "mean_answerer_prompt_tokens": round(tok[lab])}
    for v in VARIANTS:
        if not FILES[v][1].exists():
            continue
        gprices = (HAIKU_PRICE_IN, HAIKU_PRICE_OUT) if v.startswith("matched_haiku") else None
        gen = L.rows(FILES[v][0])
        graded = L.rows(FILES[v][1])
        n, rec, wr, att = counts(v)
        sess_gen = {g["session_id"] for g in gen}
        res["arms"][f"{vlabel(v)} rollup"] = {
            "n": n, "n_sessions": len({r["session_id"] for r in graded}), "recoverable": rec, "pct": round(100 * rec / n, 1),
            "wrong": wr, "attempted": att, "wrong_of_attempted_pct": round(100 * wr / att, 1) if att else None,
            "single_draw_pct": round(100 * sum(r["single_draw_grade"] in REC for r in graded) / n, 1),
            "unresolved": sum(bool(r["unresolved"]) for r in graded),
            "mean_rollup_tokens": round(statistics.mean(g["rollup_cl100k_tokens"] for g in gen)),
            "sd_rollup_tokens": round(statistics.pstdev(g["rollup_cl100k_tokens"] for g in gen)),
            "min_max_rollup_tokens": [min(g["rollup_cl100k_tokens"] for g in gen), max(g["rollup_cl100k_tokens"] for g in gen)],
            "median_rollup_tokens": statistics.median(g["rollup_cl100k_tokens"] for g in gen),
            "mean_pct_of_raw": round(statistics.mean(g["pct_of_raw"] for g in gen), 2),
            "mean_answerer_prompt_tokens": round(statistics.mean(r["usage"]["prompt_tokens"] for r in graded)),
            "gen_cost_per_session_usd": round(sum(g["usage"].get("prompt_tokens", 0) * (gprices or (L.PRICE_IN, L.PRICE_OUT))[0] + g["usage"].get("completion_tokens", 0) * (gprices or (L.PRICE_IN, L.PRICE_OUT))[1] for g in gen) / len(gen), 3),
            "spend_usd_est_for_stored_records": spend_of(gen, graded, gprices)}
        # 8-session subset check against the spike
        sub = {r["record_id"] for r in L.rows(SEEDS[v][1])} if v in SEEDS else set()
        sg = [r for r in graded if r["record_id"] in sub]
        res["arms"][f"{vlabel(v)} rollup"]["spike_subset_n"] = len(sg)
        res["arms"][f"{vlabel(v)} rollup"]["spike_subset_pct"] = round(100 * sum((r["grade"] or "absent") in REC for r in sg) / max(1, len(sg)), 1)
        rest = [r for r in graded if r["record_id"] not in sub]
        res["arms"][f"{vlabel(v)} rollup"]["other_sessions_n"] = len(rest)
        res["arms"][f"{vlabel(v)} rollup"]["other_sessions_pct"] = round(100 * sum((r["grade"] or "absent") in REC for r in rest) / max(1, len(rest)), 1)
        comps = [("rollup", "production rollup"), ("output-stripped", "output-stripped"),
                 ("prose-only", "prose-only")]
        if v.startswith("matched_haiku"):
            comps += [("rollup (original pooled)", "production rollup, original pooled"), ("prose matched", "prose cut to 933 tok"),
                      ("matched sonnet", "matched sonnet")]
        if v == "matched":
            comps += [("prose matched", "prose cut to 933 tok"), ("long sonnet", "long sonnet"),
                      ("short sonnet", "short sonnet")]
        for base, bl in comps:
            if base not in Sx:
                continue
            shared, point, d, disc = bt.diff(vlabel(v), base)
            res["paired"][f"{vlabel(v)} - {bl}"] = {
                "n": len(shared), "diff_pts": round(100 * point, 3), "cluster95_pts": cb.ci(d),
                "cluster90_pts": cb.ci(d, 5, 95),
                "boot_p": round(cb.boot_p(d), 4), "discordant_new_only_other_only": list(disc)}
    out = DATA / "rq3_sonnet_rollup_124_summary.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for f in ("estimate", "generate", "grade", "analyse"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--variant", default="long", choices=list(VARIANTS))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--generator", default="sonnet", choices=["sonnet", "haiku", "haiku_v2"])
    a = ap.parse_args()
    if a.analyse:
        analyse()
        return
    v = a.variant
    if a.generator in ("haiku", "haiku_v2"):
        assert v == "matched", "--generator haiku* only for --variant matched"
        v = "matched_haiku" if a.generator == "haiku" else "matched_haiku_v2"
    if not a.no_seed:
        seed(v)
    route(v)
    ss = all_sessions(a.only)
    if a.estimate and v.startswith("matched"):
        gi = sum(x["stripped_tokens"] * 1.15 + 1500 for x in ss)
        qs = sum(x["n_questions"] for x in ss)
        gpi, gpo = (HAIKU_PRICE_IN, HAIKU_PRICE_OUT) if v.startswith("matched_haiku") else (L.PRICE_IN, L.PRICE_OUT)
        est = (gi * gpi + len(ss) * 1000 * gpo + qs * (MATCHED_TARGET * 1.2 + 300) * L.PRICE_IN
               + qs * 4 * 450 * L.PRICE_IN + qs * 400 * L.PRICE_OUT)
        print(f"MATCHED: sessions {len(ss)} questions {qs} gen in {gi:,.0f}; est ${est:.2f}")
    elif a.estimate:
        estimate(ss)
    if a.generate:
        if v == "matched_haiku_v2":
            generate_matched(ss, v, HAIKU_MODEL, LENGTH_INSTR_HAIKU_V2, HAIKU_V2_PROMPTED_TARGET)
        elif v == "matched_haiku":
            generate_matched(ss, "matched_haiku", HAIKU_MODEL)
        else:
            {"long": L.generate, "short": S.generate, "matched": generate_matched}[v](ss)
    if a.grade:
        L.grade(ss)


if __name__ == "__main__":
    main()
