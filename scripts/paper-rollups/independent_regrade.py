#!/usr/bin/env python3
"""Independent-judge re-grade of stored answers. Plan: data/paper-rollups/independent-regrade/analysis_plan.md.
No answers are generated; only grading calls, by JUDGE (not claude-sonnet-5), via grading_common.grade_with unchanged.
Usage: independent_regrade.py smoke | run | analyse
 smoke   grade 20 model-graded answers, print projected total
 run     grade sets in plan order until the cap; resumable (grades.jsonl)
 analyse no model calls: agreement, kappa, confusion, paired differences -> results.json"""
import hashlib, json, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grading_common as gc  # noqa: E402
from rq3_raw_transcript_arm import call_llm  # noqa: E402

JUDGE = "claude-opus-5-5"
DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
OUT = DATA / "independent-regrade"
GRADES, COST, RESULTS = OUT / "grades.jsonl", OUT / "cost_log.jsonl", OUT / "results.json"
CAP = 20.0
P_IN, P_OUT = 15.0, 75.0  # USD per 1M tokens, assumed, not billed
B, SEED = 10000, 20261013
PROTOCOL = f"{gc.PROMPT_VERSION}+{gc.PARSER_VERSION}+single-draw+{JUDGE}"
SET_ORDER = ["a_rep1_nodigest", "a_rep1_stripped", "b_held_nodigest", "b_held_stripped",
             "c_rep_rollup", "b_held_rollup"]


def load(p):
    return [json.loads(l) for l in (DATA / p).open() if l.strip()]


def units():
    q = {r["question_id"]: r for r in load("replication/questions.jsonl")}
    out = []
    for r in load("replication/answers.jsonl"):
        if r["arm"] in ("nodigest", "stripped"):
            out.append(dict(set="a_rep1_" + r["arm"], arm=r["arm"], key=r["question_id"], r=r,
                            question=q[r["question_id"]]["question"], gt=q[r["question_id"]]["ground_truth"]))
    for s, p, a in (("b_held_nodigest", "rq3_heldout_no_digest_arm.jsonl", "nodigest"),
                    ("b_held_stripped", "rq3_heldout_stripped_arm_extended.jsonl", "stripped"),
                    ("b_held_rollup", "heldout-rollup/rollup_arm.jsonl", "rollup"),
                    ("c_rep_rollup", "replication-rollup/rollup_arm.jsonl", "rollup")):
        for r in load(p):
            out.append(dict(set=s, arm=a, key=r["record_id"], r=r, question=r["question"], gt=r["ground_truth"]))
    for u in out:
        u["id"] = f'{u["set"]}|{u["key"]}'
        u["rule_absent"] = gc.is_rule_absent(u["r"]["answer"], u["r"]["rating"])
    return out


def done():
    return {json.loads(l)["id"]: json.loads(l) for l in GRADES.open() if l.strip()} if GRADES.exists() else {}


def spent():
    return sum(json.loads(l)["usd"] for l in COST.open() if l.strip()) if COST.exists() else 0.0


def usd(pt, ct):
    return (pt * P_IN + ct * P_OUT) / 1e6


def grade_one(u):
    r = u["r"]
    use = {"p": 0, "c": 0, "calls": 0}
    models = []

    def call(prompt):
        text, usage = call_llm(prompt, model=JUDGE, timeout=120, max_tokens=1024)  # opus spends hidden tokens; 64 truncated 9/20 in the smoke test
        use["calls"] += 1; use["p"] += usage.get("prompt_tokens", 0); use["c"] += usage.get("completion_tokens", 0)
        return text
    row = {"id": u["id"], "set": u["set"], "arm": u["arm"], "key": u["key"], "session_id": r["session_id"],
           "stored": r["grade"], "rule_absent": u["rule_absent"], "judge": JUDGE, "protocol": PROTOCOL}
    err = None
    for _ in range(2):  # one retry at most
        try:
            frag = gc.grade_with(call, u["question"], u["gt"], r["answer"], confidence=r["rating"],
                                 raw_reply=r.get("raw_reply"))
            row.update(indep=frag["grade"], grader_response=frag["grader_response"])
            err = None
            break
        except gc.GradeError as e:
            err = str(e)
        except Exception as e:  # gateway/HTTP error, incl. 403 access denial: reported verbatim
            err = f"{type(e).__name__}: {e}"
            if "403" in err:
                break
    if err:
        row.update(indep=None, error=err)
    return row, use


def batch(us, label):
    todo = [u for u in us if u["id"] not in done()]
    if not todo:
        return True
    todo.sort(key=lambda u: hashlib.sha256(f"{SEED}{u['id']}".encode()).hexdigest())  # shuffled, arm-blind
    model = [u for u in todo if not u["rule_absent"]]
    model_total = len(model)
    per = usd(1000, 300)  # measured on re-probe: ~800-1350 in, 100-400 out (hidden reasoning tokens bill as output)
    room = int(max(CAP - spent(), 0) / per * 0.95)
    trimmed = len(model) > room
    if trimmed:  # trim to what the cap allows; rule-absent rows are free so keep them
        keep = {u["id"] for u in model[:room]}
        todo = [u for u in todo if u["rule_absent"] or u["id"] in keep]
        model = model[:room]
    est = len(model) * per
    print(f"[{label}] {len(todo)} rows, {len(model)} model calls, est ${est:.2f}, running ${spent():.2f}, cap ${CAP}", flush=True)
    if room == 0 and model_total > 0:
        return False
    pt = ct = calls = 0
    with ThreadPoolExecutor(8) as ex, GRADES.open("a") as fh:
        for row, use in ex.map(grade_one, todo):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            pt += use["p"]; ct += use["c"]; calls += use["calls"]
    with COST.open("a") as fh:
        fh.write(json.dumps({"stage": label, "judge": JUDGE, "calls": calls, "prompt_tokens": pt,
                             "completion_tokens": ct, "usd": round(usd(pt, ct), 5)}) + "\n")
    print(f"[{label}] spent ${usd(pt, ct):.2f}, running ${spent():.2f}", flush=True)
    return not trimmed


def main_run(smoke):
    OUT.mkdir(parents=True, exist_ok=True)
    us = units()
    if smoke:
        pool = [u for u in us if not u["rule_absent"] and u["id"] not in done()]
        pool.sort(key=lambda u: hashlib.sha256(f"smoke{u['id']}".encode()).hexdigest())
        batch(pool[:20], "smoke")
        n = sum(not u["rule_absent"] for u in us)
        sp = spent()
        proj = sp / 20 * n
        print(f"projection: {n} model-graded answers x ${sp/20:.4f} = ${proj:.2f}")
        return
    for s in SET_ORDER:
        chunk = [u for u in us if u["set"] == s]
        for i in range(0, len(chunk), 100):
            if not batch(chunk[i:i + 100], f"{s}[{i}]"):
                print("STOP at cap; sets covered so far reported by analyse"); return
        bad = [g for g in done().values() if g["set"] == s and g["indep"] is None]
        if any("403" in (g.get("error") or "") for g in bad):
            print("403 access denial; stopping:", bad[0]["error"]); return


# ---------- analysis ----------
REC = ("correct", "partially_correct")
CLS = list(gc.GRADES)


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def agree(rows):
    four = [(g["stored"], g["indep"]) for g in rows]
    two = [(a in REC, b in REC) for a, b in four]
    return {"n": len(rows),
            "raw_agreement_4class": round(sum(a == b for a, b in four) / len(four), 3) if four else None,
            "kappa_4class": kappa(four),
            "raw_agreement_recovered": round(sum(a == b for a, b in two) / len(two), 3) if two else None,
            "kappa_recovered": kappa(two),
            "confusion_stored_rows_by_indep_cols": {"classes": CLS, "matrix": [[sum(1 for a, b in four if a == s and b == i) for i in CLS] for s in CLS]}}


def m_rec(rows, f): return sum(r[f] in REC for r in rows) / len(rows)
def m_abs(rows, f): return sum(r[f] == "absent" for r in rows) / len(rows)
def m_wga(rows, f): return sum(r[f] == "wrong" for r in rows) / sum(r[f] != "absent" for r in rows)


def paired(a, b, f, fn):
    """a, b: dicts key->row (with session_id). bootstrap over sessions on keys present in both."""
    keys = [k for k in a if k in b]
    sess = {}
    for k in keys:
        sess.setdefault(a[k]["session_id"], []).append(k)
    sk = list(sess)
    rng = random.Random(SEED)
    d = []
    for _ in range(B):
        ks = [k for s in (rng.choice(sk) for _ in sk) for k in sess[s]]
        try:
            d.append(fn([a[k] for k in ks], f) - fn([b[k] for k in ks], f))
        except ZeroDivisionError:
            pass
    d.sort()
    q = lambda x: round(100 * d[int(x * (len(d) - 1))], 1)
    pt = round(100 * (fn([a[k] for k in keys], f) - fn([b[k] for k in keys], f)), 1)
    return {"n_pairs": len(keys), "n_sessions": len(sk), "diff_pts": pt, "ci95_pts": [q(.025), q(.975)]}


def main_analyse():
    G = done()
    us = units()
    tot = {s: sum(u["set"] == s for u in us) for s in SET_ORDER}
    ok = {s: [g for g in G.values() if g["set"] == s and g["indep"]] for s in SET_ORDER}
    excl = [{"id": g["id"], "error": g.get("error")} for g in G.values() if not g["indep"]]
    res = {"plan": "analysis_plan.md", "judge_requested": JUDGE, "protocol": PROTOCOL, "single_draw": True,
           "assumed_price_usd_per_M": [P_IN, P_OUT], "spend_usd": round(spent(), 4), "cap_usd": CAP,
           "coverage": {s: {"rows": tot[s], "graded": len(ok[s])} for s in SET_ORDER},
           "excluded_rows": excl}
    res["agreement_overall"] = agree([g for s in SET_ORDER for g in ok[s]])
    res["agreement_model_graded_only"] = agree([g for s in SET_ORDER for g in ok[s] if not g["rule_absent"]])
    res["agreement_per_set"] = {s: agree(ok[s]) for s in SET_ORDER if ok[s]}

    def side(s):  # key -> row with both stored and indep as fields
        return {g["key"]: {"session_id": g["session_id"], "stored": g["stored"], "indep": g["indep"]} for g in ok[s]}
    full = lambda s: len(ok[s]) == tot[s]
    cmp = {}
    def add(name, sa, sb, metrics):
        if not (ok[sa] and ok[sb]):
            cmp[name] = "not computed: set not covered"; return
        a, b = side(sa), side(sb)
        e = {"complete_coverage": full(sa) and full(sb)}
        for mname, fn in metrics:
            e[mname] = {f: paired(a, b, f, fn) for f in ("stored", "indep")}
            e[mname]["change_in_point_estimate_pts"] = round(e[mname]["indep"]["diff_pts"] - e[mname]["stored"]["diff_pts"], 1)
            s0, s1 = e[mname]["stored"]["diff_pts"], e[mname]["indep"]["diff_pts"]
            e[mname]["sign_flip"] = bool(s0 * s1 < 0)
        cmp[name] = e
    add("nodigest_minus_stripped_recall__replication1", "a_rep1_nodigest", "a_rep1_stripped", [("recall", m_rec)])
    add("nodigest_minus_stripped_recall__heldout", "b_held_nodigest", "b_held_stripped", [("recall", m_rec)])
    rw = [("wrong_given_attempted", m_wga), ("absent", m_abs), ("recall", m_rec)]
    add("rollup_minus_stripped__heldout", "b_held_rollup", "b_held_stripped", rw)
    # replication questions: rollup vs the stripped arm of replication 1
    add("rollup_minus_stripped__replication_questions", "c_rep_rollup", "a_rep1_stripped", rw)
    for n, e in cmp.items():
        if isinstance(e, dict):
            for m in ("recall", "wrong_given_attempted", "absent"):
                if m in e:
                    x = e[m]; margin = 10 if m == "recall" and n.startswith("nodigest") else 0
                    x["margin_pts"] = margin
                    c = lambda f: (x[f]["ci95_pts"][0] <= 0 <= x[f]["ci95_pts"][1]) if margin == 0 else (x[f]["ci95_pts"][0] < -margin or x[f]["ci95_pts"][1] > margin)
                    x["ci_crosses_margin"] = {f: bool(c(f)) for f in ("stored", "indep")}  # margin 0: CI contains 0; margin 10: CI extends outside +-10
                    x["judge_dependence_flag"] = bool(x["sign_flip"] or (m == "recall" and abs(x["change_in_point_estimate_pts"]) > 3))
    res["paired_differences"] = cmp
    res["judge_ids_returned"] = "gateway chat/completions does not echo a model field through call_llm; requested id is " + JUDGE + " (probe returned model claude-opus-5-5)"
    RESULTS.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("coverage", "spend_usd", "agreement_overall", "paired_differences")}, indent=1))


if __name__ == "__main__":
    {"smoke": lambda: main_run(True), "run": lambda: main_run(False), "analyse": main_analyse}[sys.argv[1]]()
