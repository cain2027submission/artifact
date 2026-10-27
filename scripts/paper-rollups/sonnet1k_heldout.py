#!/usr/bin/env python3
"""Held-out replication of the 1,000-token Sonnet rollup result.
Plan: data/paper-rollups/sonnet1k-heldout/analysis_plan.md (committed before any call).

Arms on the 125 held-out questions:
  sonnet1k   Sonnet rollup, design-set 'matched' variant (rq3_sonnet_rollup_124.build_prompt_matched, target 932)
  cut        prose-only cut to the session's own sonnet1k rollup length (primary control)
  cut_prod   prose-only cut to the session's existing held-out production note length (design-set construction)
Answer + 3-draw majority `grade` as heldout_rollup_run.py. Paid; resumable; cap USD 7.

Usage: sonnet1k_heldout.py --generate [--smoke] | --answer ARM [--smoke] | --provenance | --analyse
       add --dry-run to print the estimate only.
"""
import argparse, glob, hashlib, json, os, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROLLUP_SRC_DIR))
import grading_common as gc  # noqa: E402
import rq3_long_rollup_spike as L  # noqa: E402
import rq3_sonnet_rollup_124 as R  # noqa: E402
from rq3_raw_transcript_arm import post, GATEWAY, call_llm, extract_confidence, load_records  # noqa: E402
from rq3_uniform_rollup_arm import answer_from_rollup, JUDGE_MODEL  # noqa: E402
from rq3_prose_only_arm import prose_only_context  # noqa: E402
from rq3_control_prose_arms import truncate_tokens, answer_prose, ntok  # noqa: E402

ROOT = HERE.parents[1]
DATA = ROOT / "data" / "paper-rollups"
OD = DATA / "sonnet1k-heldout"
ROLLUPS = OD / "rollups"
GENLOG = OD / "generations.jsonl"
COST = OD / "cost_log.jsonl"
RESULTS = OD / "results.json"
QUESTIONS = DATA / "rq3_heldout_questions.jsonl"
PROV = DATA / "rollup-provenance.jsonl"
CAP_USD = 10.0  # raised from 7.0 after the first 55 cut-arm questions; see results.json deviation
PRICE_IN, PRICE_OUT = 3.0, 15.0
B, SEED = 10000, 20261013
PROTOCOL = f"{gc.PROMPT_VERSION}+{gc.PARSER_VERSION}+majority3+{JUDGE_MODEL}"
ARMS = {"sonnet1k": OD / "arm_sonnet1k.jsonl", "cut": OD / "arm_cut.jsonl", "cut_prod": OD / "arm_cut_prod.jsonl"}
BATCH = 25


def rid(r):
    return hashlib.sha256(f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}".encode()).hexdigest()[:16]


def rows(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def total():
    return sum(r["usd"] for r in rows(COST))


def usd(u):
    return (u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT) / 1e6


def logcost(stage, calls, p, c):
    OD.mkdir(parents=True, exist_ok=True)
    with COST.open("a") as fh:
        fh.write(json.dumps({"stage": stage, "calls": calls, "prompt_tokens": p, "completion_tokens": c,
                             "usd": round((p * PRICE_IN + c * PRICE_OUT) / 1e6, 5)}) + "\n")


def questions(smoke=False):
    qs = load_records(QUESTIONS)
    for q in qs:
        q["_rid"] = rid(q)
    if smoke:
        seen, out = set(), []
        for q in qs:
            if q["session_id"] not in seen:
                seen.add(q["session_id"]); out.append(q)
            if len(out) == 5:
                break
        return out
    return qs


def transcript(sid):
    pre = sid.split("_")[2]
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{pre}*.jsonl"))
    assert hits, sid
    return Path(hits[0])


# ---------------------------------------------------------------- generation
def generate(qs, dry):
    sids = sorted({q["session_id"] for q in qs})
    done = {r["session_id"] for r in rows(GENLOG) if r.get("ok")}
    todo = [s for s in sids if s not in done]
    info = {}
    est_in = est_out = 0
    for s in todo:
        raw = transcript(s).read_text(errors="replace")
        stripped, _ = L.strip_transcript(raw)
        info[s] = (raw, stripped)
        est_in += L.ntok(stripped) * 1.15 + 1500
        est_out += 1300
    est = (est_in * PRICE_IN + est_out * PRICE_OUT) / 1e6
    print(f"generate: sessions {len(sids)} todo {len(todo)} estimate ${est:.3f}, running total ${total():.3f}, cap ${CAP_USD}")
    if dry or not todo:
        return
    if total() + est > CAP_USD:
        print("STOP: would exceed cap"); sys.exit(2)
    ROLLUPS.mkdir(parents=True, exist_ok=True)

    def one(s):
        raw, stripped = info[s]
        prompt = R.build_prompt_matched(stripped, R.MATCHED_TARGET)
        p = c = 0
        for attempt in range(2):  # one retry at most
            body = {"model": L.GEN_MODEL, "max_tokens": R.MATCHED_MAX_TOKENS,
                    "messages": [{"role": "user", "content": prompt}]}
            try:
                d = post(f"{GATEWAY}/v1/chat/completions", body, timeout=900)
            except Exception as e:  # noqa: BLE001
                rec = {"session_id": s, "ok": False, "error": str(e)[:200], "attempt": attempt + 1}
                with GENLOG.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue
            ch = d["choices"][0]
            u = d.get("usage", {})
            p += u.get("prompt_tokens", 0); c += u.get("completion_tokens", 0)
            text = ch["message"]["content"] or ""
            ok = bool(text) and ch.get("finish_reason") == "stop"
            rec = {"session_id": s, "ok": ok, "attempt": attempt + 1, "model_requested": L.GEN_MODEL,
                   "model_returned": d.get("model"), "finish_reason": ch.get("finish_reason"), "usage": u,
                   "target_tokens": R.MATCHED_TARGET, "rollup_cl100k_tokens": L.ntok(text) if text else 0,
                   "raw_tokens": L.ntok(raw), "stripped_tokens": L.ntok(stripped)}
            with GENLOG.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if ok:
                (ROLLUPS / f"{s}.md").write_text(text, encoding="utf-8")
                return s, p, c, True
        return s, p, c, False

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, todo))
    logcost("generate", len(res), sum(r[1] for r in res), sum(r[2] for r in res))
    bad = [r[0] for r in res if not r[3]]
    print(f"generated {len(res) - len(bad)}/{len(res)}; failed {bad}; total ${total():.3f}")


def rollup_text(sid):
    return (ROLLUPS / f"{sid}.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- answering + grading
_prose = {}


def prose(sid):
    if sid not in _prose:
        _prose[sid] = prose_only_context(transcript(sid).read_text(errors="replace"), keep_continuation=False)[0]
    return _prose[sid]


def artifact(arm, sid):
    if arm == "sonnet1k":
        return rollup_text(sid)
    n = ntok(rollup_text(sid)) if arm == "cut" else ntok((ROOT / "sessions" / f"{sid}.md").read_text(errors="replace"))
    return truncate_tokens(prose(sid), n)


def grade_one(arm, q, art):
    fn = answer_from_rollup if arm == "sonnet1k" else answer_prose
    calls = pt = ct = 0
    answer = conf = None
    for _ in range(2):
        answer, conf, u = fn(q["question"], art)
        calls += 1; pt += u.get("prompt_tokens", 0); ct += u.get("completion_tokens", 0)
        if conf != "API_ERROR":
            break
    if conf == "API_ERROR":
        return None, (calls, pt, ct), q["_rid"]

    def draw():
        nonlocal calls, pt, ct

        def call(p):
            nonlocal calls, pt, ct
            r, u = call_llm(p, timeout=120)
            calls += 1; pt += u.get("prompt_tokens", 0); ct += u.get("completion_tokens", 0)
            return r
        for _ in range(2):
            try:
                return gc.grade_with(call, q["question"], q["ground_truth"], answer, conf)["grade"]
            except gc.GradeError:
                continue
        return gc.GRADE_ERROR

    def maj(g):
        c = Counter(g).most_common()
        return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None
    draws = [draw() for _ in range(3)]
    valid = [g for g in draws if g != gc.GRADE_ERROR]
    m = maj(valid) if len(valid) >= 2 else None
    if m is None and len(valid) >= 2:
        draws += [draw() for _ in range(2)]
        valid = [g for g in draws if g != gc.GRADE_ERROR]
        m = maj(valid)
    if m is None:
        return None, (calls, pt, ct), q["_rid"]
    row = {"record_id": q["_rid"], "session_id": q["session_id"], "class": q["class"], "question": q["question"],
           "ground_truth": q["ground_truth"], "arm": arm, "artifact_cl100k_tokens": ntok(art), "answer": answer,
           "rating": conf, "grade": m, "grade_draws": draws[:3], "grade_protocol": PROTOCOL, "model": JUDGE_MODEL}
    if len(draws) > 3:
        row["grade_tiebreak_draws"] = draws[3:]
    return row, (calls, pt, ct), None


def answer(arm, qs, dry):
    out = ARMS[arm]
    done = {r["record_id"] for r in rows(out)}
    todo = [q for q in qs if q["_rid"] not in done and (ROLLUPS / f"{q['session_id']}.md").exists()]
    for q in todo:
        q["_art"] = artifact(arm, q["session_id"])

    def est(b):
        pin = sum(ntok(q["_art"]) + 300 + 3 * (len(q["question"]) + len(q["ground_truth"]) + 700) / 3.2 * 1.15 for q in b)
        return (pin * PRICE_IN + len(b) * 230 * PRICE_OUT) / 1e6
    print(f"{arm}: questions {len(qs)} done {len(done)} todo {len(todo)} estimate ${est(todo):.3f}, running total ${total():.3f}, cap ${CAP_USD}")
    if dry:
        return
    errs = []
    for i in range(0, len(todo), BATCH):
        b = todo[i:i + BATCH]
        e = est(b)
        print(f" batch {i // BATCH + 1}: estimate ${e:.3f}, running total ${total():.3f}")
        if total() + e > CAP_USD:
            print("STOP: batch would exceed cap; arm not completed"); sys.exit(2)
        with ThreadPoolExecutor(5) as ex:
            res = list(ex.map(lambda q: grade_one(arm, q, q["_art"]), b))
        c = p = k = 0
        for row, (cc, pp, kk), err in res:
            c += cc; p += pp; k += kk
            if row:
                with out.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                errs.append(err)
        logcost(f"answer:{arm}", c, p, k)
        print(f"  done calls {c} tokens {p}+{k}; total ${total():.3f}; failed {len(errs)}")
    if errs:
        print("FAILED (excluded):", errs)


# ---------------------------------------------------------------- provenance
def provenance():
    have = {(r["session_id"], r["arm"]) for r in rows(PROV)}
    gens = {}
    for r in rows(GENLOG):
        gens.setdefault(r["session_id"], []).append(r)
    n = 0
    with PROV.open("a") as fh:
        for sid, g in sorted(gens.items()):
            ok = [x for x in g if x.get("ok")]
            if not ok or (sid, "sonnet1k-heldout-regen") in have:
                continue
            p = ROLLUPS / f"{sid}.md"
            b = p.read_bytes()
            m = ok[-1]
            fh.write(json.dumps({
                "session_id": sid, "arm": "sonnet1k-heldout-regen", "path": str(p.relative_to(ROOT)),
                "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "cl100k_tokens": L.ntok(b.decode()),
                "note_date": sid[:10], "generator_model_requested": m["model_requested"],
                "generator_model_returned": m["model_returned"], "provenance_status": "logged",
                "evidence": f"sonnet1k-heldout/generations.jsonl: {len(g)} call row(s), finish_reason {m['finish_reason']}, "
                            f"prompt {m['usage'].get('prompt_tokens')} completion {m['usage'].get('completion_tokens')}"}) + "\n")
            n += 1
    print(f"provenance rows appended: {n}")


# ---------------------------------------------------------------- analysis
REC = ("correct", "partially_correct")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def summ(rs):
    c = Counter(r["grade"] for r in rs)
    n, w, a = len(rs), c["wrong"], c["absent"]
    rec = c["correct"] + c["partially_correct"]
    return {"n": n, "recall_pct": round(100 * rec / n, 1), "recall_ci95": wilson(rec, n), "attempted": n - a, "wrong": w,
            "wrong_given_attempted_pct": round(100 * w / (n - a), 1) if n - a else None,
            "wrong_given_attempted_ci95": wilson(w, n - a), "absent": a, "absent_pct": round(100 * a / n, 1)}


METRICS = {"recall": lambda rs: sum(r["grade"] in REC for r in rs) / len(rs),
           "wrong_given_attempted": lambda rs: sum(r["grade"] == "wrong" for r in rs) / sum(r["grade"] != "absent" for r in rs),
           "absent": lambda rs: sum(r["grade"] == "absent" for r in rs) / len(rs)}


def fisher(a, b, c, d):
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    pr = lambda x: comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(A, Bm):
    bi = {r["record_id"]: r for r in Bm}
    ids = [r["record_id"] for r in A if r["record_id"] in bi]
    a = [r for r in A if r["record_id"] in bi]
    b = [bi[i] for i in ids]
    out = {"n_paired": len(a)}
    sess = {}
    for i, r in enumerate(a):
        sess.setdefault(r["session_id"], []).append(i)
    keys = list(sess)
    for name, f in METRICS.items():
        rng = random.Random(SEED)
        ds = []
        for _ in range(B):
            idx = [i for k in (rng.choice(keys) for _ in keys) for i in sess[k]]
            try:
                ds.append(f([a[i] for i in idx]) - f([b[i] for i in idx]))
            except ZeroDivisionError:
                continue
        ds.sort()
        q = lambda x: round(100 * ds[int(x * (len(ds) - 1))], 1)
        out[name] = {"diff_pts": round(100 * (f(a) - f(b)), 1), "clustered_bootstrap_ci95_pts": [q(.025), q(.975)],
                     "resamples": len(ds)}
    ra = [r["grade"] in REC for r in a]; rb = [r["grade"] in REC for r in b]
    ao = sum(x and not y for x, y in zip(ra, rb)); bo = sum(y and not x for x, y in zip(ra, rb))
    out["recall_mcnemar_exact_p"] = round(mcnemar(ao, bo), 4)
    out["recall_discordant_a_only_b_only"] = [ao, bo]
    out["recall_fisher_p"] = round(fisher(sum(ra), len(ra) - sum(ra), sum(rb), len(rb) - sum(rb)), 4)
    out.update({"n_sessions": len(keys), "B": B, "seed": SEED})
    return out


def verdict(c):
    lo, hi = c["recall"]["clustered_bootstrap_ci95_pts"]
    return "REPLICATES (CI entirely above 0)" if lo > 0 else ("CONTRADICTS (CI entirely below 0)" if hi < 0 else "NO DETECTED DIFFERENCE (CI contains 0)")


def analyse():
    arms = {k: rows(v) for k, v in ARMS.items()}
    arms["output_stripped"] = rows(DATA / "rq3_heldout_stripped_arm_extended.jsonl")
    arms["production_note"] = rows(DATA / "heldout-rollup" / "rollup_arm.jsonl")
    gens = [r for r in rows(GENLOG) if r.get("ok")]
    toks = sorted(r["rollup_cl100k_tokens"] for r in gens)
    res = {"arms": {k: summ(v) for k, v in arms.items() if v}, "cap_usd": CAP_USD, "spend_usd_assumed_3_15": round(total(), 3),
           "generation": {"sessions": len(gens), "rollup_tokens_median": (toks[len(toks) // 2] if len(toks) % 2 else (toks[len(toks) // 2 - 1] + toks[len(toks) // 2]) / 2) if toks else None,
                          "min": toks[0] if toks else None, "max": toks[-1] if toks else None,
                          "models_returned": sorted({str(r.get("model_returned")) for r in gens}),
                          "finish_reasons": dict(Counter(r["finish_reason"] for r in gens))},
           "paired": {}}
    for other in ("cut", "cut_prod", "output_stripped", "production_note"):
        if arms.get(other) and arms["sonnet1k"]:
            res["paired"][f"sonnet1k_minus_{other}"] = compare(arms["sonnet1k"], arms[other])
    if "sonnet1k_minus_cut" in res["paired"]:
        res["verdict_primary_recall_sonnet1k_minus_cut"] = verdict(res["paired"]["sonnet1k_minus_cut"])
    pre = arms["cut"][:55]  # file order: the first 55 rows were written under the USD 7 cap
    if len(arms["cut"]) > 55:
        res["prefix_check_first55_cut"] = compare(arms["sonnet1k"], pre)
    res["deviation"] = ("the cap was raised from 7 to 10 after the first 55 cut-arm questions were inspected: the cap, not the result, "
                        "stopped the arm; the 55-question partial primary CI contained 0 (+5.5 [-14.5, +23.6]) and was seen before the decision; "
                        "the continuation covers the remaining questions in file order with no selection; the final numbers use all 125 and the "
                        "partial 55-question result is reported alongside as a prefix check (prefix_check_first55_cut).")
    res["incomplete_arms"] = {k: f"{len(arms[k])} of {len(arms['sonnet1k'])} questions, stopped by the {CAP_USD} USD cap; first questions in file order, not a random subset"
                              for k in ("cut", "cut_prod") if arms.get(k) and len(arms[k]) < len(arms["sonnet1k"])}
    res["not_run"] = [k for k in ("cut_prod",) if not arms.get(k)]
    RESULTS.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--answer", choices=list(ARMS))
    ap.add_argument("--provenance", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    OD.mkdir(parents=True, exist_ok=True)
    qs = questions(a.smoke)
    if a.generate:
        generate(qs, a.dry_run)
    if a.answer:
        answer(a.answer, qs, a.dry_run)
    if a.provenance:
        provenance()
    if a.analyse:
        analyse()


if __name__ == "__main__":
    main()
