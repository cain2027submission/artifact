#!/usr/bin/env python3
"""Every RQ3 number the paper states that is not a plain arm rate, recomputed from
the per-record files after the 2026-10-09 regrade. No model calls.

Run order: regrade_s4.py apply  ->  cluster_bootstrap.py  ->  rq3_sonnet_rollup_124.py
--analyse  ->  this script. It writes data/paper-rollups/regrade-2026-10-09/
renumber.json, which check_paper_invariants.py reads, and prints the numbers.

Everything here reads the revised per-record `grade` (3-draw majority of the
fixed grader, question in the prompt) and, for the before column, the stored
`grade_published`. Sentences in main.tex that cite these are gated against the
json, so a number cannot be edited without the data moving with it.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_rates as ar  # noqa: E402

DATA = ar.DATA
OUT = DATA / "regrade-2026-10-09" / "renumber.json"
REC = ("correct", "partially_correct")
NO_TOOL_SESSIONS = ("9d4c165a", "a596be9f", "d7aa4f50")   # tool_calls == 0 in the call-only arm


def rows(name, **flt):
    out = []
    for l in (DATA / name).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if all(r.get(k) == v for k, v in flt.items()):
                out.append(r)
    return out


def rate_of(grades):
    c = Counter(grades)
    n = len(grades)
    k = c["correct"] + c["partially_correct"]
    att = k + c["wrong"]
    lo, hi = ar.wilson(k, n)
    return {"n": n, "recoverable": k, "recall": ar.r1(100 * k / n), "ci": [lo, hi],
            "wrong": c["wrong"], "attempted": att,
            "wrong_rate": ar.r1(100 * c["wrong"] / att) if att else 0.0,
            "absent": c["absent"]}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[g] * cb[g] for g in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_boot(a_rows, b_rows, B=20000, seed=20261009):
    """Session-clustered paired bootstrap of the recoverable difference a - b."""
    shared = sorted(set(a_rows) & set(b_rows))
    by = defaultdict(list)
    for i in shared:
        by[a_rows[i]["session_id"]].append((a_rows[i]["grade"] in REC, b_rows[i]["grade"] in REC))
    keys = sorted(by)
    rng = random.Random(seed)
    d = []
    for _ in range(B):
        flat = [p for k in (rng.choice(keys) for _ in keys) for p in by[k]]
        d.append(100 * (sum(x for x, _ in flat) - sum(y for _, y in flat)) / len(flat))
    d.sort()
    pt = 100 * sum((a_rows[i]["grade"] in REC) - (b_rows[i]["grade"] in REC) for i in shared) / len(shared)
    q = lambda p: round(d[int(p * (len(d) - 1))], 1)
    ao = sum(1 for i in shared if a_rows[i]["grade"] in REC and b_rows[i]["grade"] not in REC)
    bo = sum(1 for i in shared if a_rows[i]["grade"] not in REC and b_rows[i]["grade"] in REC)
    return {"n": len(shared), "diff": round(pt, 1), "ci95": [q(.025), q(.975)], "ci90": [q(.05), q(.95)],
            "a_only": ao, "b_only": bo, "mcnemar_p": round(mcnemar_exact(ao, bo), 4)}


def by_id(name, **flt):
    return {r["record_id"]: r for r in rows(name, **flt)}


def compute():
    """Every number below, recomputed from the per-record files (no file is read
    that this script or arm_rates did not derive from them, except the stored
    judge agreement and probe overlap reports)."""
    res = {}
    # ---- A. every arm, published protocol against revised grader
    labels = list(ar.UNCORRECTED_SOURCES) + list(ar.RETENTION_ONLY_SOURCES)
    res["arms"] = {}
    for lab in labels:
        new, old = ar.any_rate(lab), ar.published(lab)
        res["arms"][lab] = {"new": rate_of_rate(new), "published": rate_of_rate(old)}
    # ---- B. the 89 questions raw can be judged on
    raw_ids = set(by_id("rq3_questions_raw_arm.jsonl"))
    c89 = {}
    for lab, (f, bud) in {**ar.UNCORRECTED_SOURCES, **ar.RETENTION_ONLY_SOURCES}.items():
        rs = [r for r in rows(f) if r["record_id"] in raw_ids and (bud is None or r.get("budget") == bud)]
        if rs:
            c89[lab] = {"new": rate_of([r["grade"] for r in rs]),
                        "published": rate_of([r["grade_published"] for r in rs])}
    res["common89"] = c89
    strip = by_id("rq3_questions_stripped_arm.jsonl")
    res["stripped_on_other_35"] = rate_of([r["grade"] for i, r in strip.items() if i not in raw_ids])
    raw = by_id("rq3_questions_raw_arm.jsonl")
    res["raw_failures"] = [{"grade": r["grade"], "first_line": (r["raw_answer"] or "").strip().splitlines()[0][:60]
                            if (r["raw_answer"] or "").strip() else "", "session": r["session_id"][-30:]}
                           for r in raw.values() if r["grade"] not in REC]
    res["raw_wrong"] = sum(r["grade"] == "wrong" for r in raw.values())
    # ---- C. by class, rollup arm
    roll = by_id("rq3_rollup_uniform.jsonl")
    cls = defaultdict(list)
    for r in roll.values():
        cls[r["class"]].append(r["grade"])
    res["rollup_by_class"] = {k: rate_of(v) for k, v in sorted(cls.items())}
    # ---- D. sessions with no tool calls
    call = by_id("rq3_tool_inputs_only_arm.jsonl")
    keep = lambda d: {i: r for i, r in d.items() if not any(s in r["session_id"] for s in NO_TOOL_SESSIONS)}
    rk, ck, sk = keep(roll), keep(call), keep(strip)
    res["no_tool_exclusion"] = {
        "rollup": rate_of([r["grade"] for r in rk.values()]),
        "call_only": rate_of([r["grade"] for r in ck.values()]),
        "output_stripped": rate_of([r["grade"] for r in sk.values()]),
        "gap_stripped_minus_rollup_full": round(ar.uncorrected("output-stripped").recall - ar.uncorrected("rollup").recall, 1),
        "gap_stripped_minus_rollup_excl": round(rate_of([r["grade"] for r in sk.values()])["recall"]
                                                - rate_of([r["grade"] for r in rk.values()])["recall"], 1)}
    # ---- E. the 349-question rollup arm: frozen 124 against held-back 225
    allq = rows("rq3_questions.jsonl")
    frozen = set(roll)
    import hashlib
    rid = lambda r: hashlib.sha256(f"{r['session_id']}|{r['class']}|{r['question']}".encode()).hexdigest()[:16]
    res["rollup_349"] = {"frozen": rate_of([r["grade"] for r in allq if rid(r) in frozen]),
                         "held_back": rate_of([r["grade"] for r in allq if rid(r) not in frozen]),
                         "published_held_back": rate_of([r["grade_published"] for r in allq if rid(r) not in frozen])}
    hb = rows("rq3_arm_prose_only_heldback225_graded.jsonl")
    res["prose_only_held_back"] = {"new": rate_of([r["grade"] for r in hb]),
                                   "published": rate_of([r["grade_published"] for r in hb])}
    # ---- F. grader draws on fixed answers (rollup arm)
    dr = [r["grade_draws"] for r in roll.values()]
    res["rollup_draws"] = {"per_draw_recall": [ar.r1(100 * sum(d[i] in REC for d in dr) / len(dr)) for i in range(3)],
                           "kappa_pairs": [round(kappa([d[i] for d in dr], [d[j] for d in dr]), 2)
                                           for i, j in ((0, 1), (0, 2), (1, 2))],
                           "unanimous": sum(len(set(d)) == 1 for d in dr), "n": len(dr)}
    # ---- G. recap (compaction summary) rows
    cont = by_id("rq3_continuation_only_arm.jsonl")
    prose = by_id("rq3_prose_only_arm.jsonl")
    withc = by_id("rq3_prose_only_arm.with-continuation.jsonl")
    ids = set(cont)
    res["recap"] = {"n_records": len(ids),
                    "continuation_only": rate_of([cont[i]["grade"] for i in ids]),
                    "prose_only_same": rate_of([prose[i]["grade"] for i in ids if i in prose]),
                    "prose_only_with_recap_all": rate_of([r["grade"] for r in withc.values()])}
    # ---- H. Sonnet / Haiku regeneration family
    fam = {}
    for lab, f in (("haiku_from_stripped", "rq3_rollup_from_stripped_spike_graded.jsonl"),
                   ("haiku_prod_input_2x", "rq3_spike_prod_input_2x_graded.jsonl"),
                   ("haiku_stripped_matchlen", "rq3_spike_stripped_input_matchlen_graded.jsonl"),
                   ("renderer_control", "rq3_arm_prod_pipeline_control_graded.jsonl"),
                   ("prose_933_median", "rq3_arm_prose_matched_median_graded.jsonl"),
                   ("prose_933_per_record", "rq3_arm_prose_matched_per_record_graded.jsonl")):
        rs = rows(f)
        fam[lab] = {"new": rate_of([r["grade"] for r in rs]), "published": rate_of([r["grade_published"] for r in rs])}
    res["regenerations"] = fam
    # spike (8 sessions, 40 questions): short sonnet against matched-length Haiku
    sp_short = by_id("rq3_sonnet_rollup_124_short_graded.jsonl")
    spike_ids = set(by_id("rq3_short_sonnet_rollup_spike_graded.jsonl"))
    mh = by_id("rq3_spike_stripped_input_matchlen_graded.jsonl")
    sp = {i: sp_short[i] for i in spike_ids if i in sp_short}
    a_only = sum(sp[i]["grade"] in REC and mh[i]["grade"] not in REC for i in sp)
    b_only = sum(sp[i]["grade"] not in REC and mh[i]["grade"] in REC for i in sp)
    res["spike_generator_effect"] = {
        "n": len(sp), "sessions": len({r["session_id"] for r in sp.values()}),
        "diff": round(100 * (a_only - b_only) / len(sp), 1), "a_only": a_only, "b_only": b_only,
        "p": round(mcnemar_exact(a_only, b_only), 4)}
    for lab, f in (("spike_long", "rq3_long_rollup_spike_graded.jsonl"),
                   ("spike_short", "rq3_short_sonnet_rollup_spike_graded.jsonl")):
        rs = rows(f)
        res[lab] = {"new": rate_of([r["grade"] for r in rs]), "published": rate_of([r["grade_published"] for r in rs])}
    res["spike_production_rollup_on_40"] = rate_of([roll[i]["grade"] for i in spike_ids if i in roll])
    # ---- I. held-out validation
    ho = {}
    A, S68 = by_id("rq3_heldout_action_record_arm.jsonl"), by_id("rq3_heldout_stripped_arm.jsonl")
    sh = sorted(set(A) & set(S68))
    ao = sum(A[i]["grade"] in REC and S68[i]["grade"] not in REC for i in sh)
    bo = sum(A[i]["grade"] not in REC and S68[i]["grade"] in REC for i in sh)
    ho["action_vs_stripped_68"] = {"n": len(sh), "action": rate_of([A[i]["grade"] for i in sh]),
                                   "stripped": rate_of([S68[i]["grade"] for i in sh]),
                                   "action_only": ao, "stripped_only": bo,
                                   "p": round(mcnemar_exact(ao, bo), 4),
                                   **bootstrap_pairs([A[i]["grade"] in REC for i in sh], [S68[i]["grade"] in REC for i in sh], 20261004, 20000)}
    ND, SX = by_id("rq3_heldout_no_digest_arm.jsonl"), by_id("rq3_heldout_stripped_arm_extended.jsonl")
    sh2 = sorted(set(ND) & set(SX))
    ho["no_digest_vs_stripped_125"] = {"n": len(sh2), "no_digest": rate_of([ND[i]["grade"] for i in sh2]),
                                       "stripped": rate_of([SX[i]["grade"] for i in sh2]),
                                       **bootstrap_pairs([ND[i]["grade"] in REC for i in sh2], [SX[i]["grade"] in REC for i in sh2], 20261004, 20000)}
    res["heldout"] = ho
    # design-set action record vs stripped discordance (power paragraph)
    a_, s_ = by_id("rq3_action_record_arm.jsonl"), strip
    d_ao = sum(a_[i]["grade"] in REC and s_[i]["grade"] not in REC for i in s_)
    d_so = sum(a_[i]["grade"] not in REC and s_[i]["grade"] in REC for i in s_)
    diff = ar.uncorrected("action record").recall - ar.uncorrected("output-stripped").recall
    res["action_vs_stripped_design"] = {"diff": round(diff, 1), "action_only": d_ao, "stripped_only": d_so,
                                        "power_at_124": power_mcnemar(d_ao, d_so, 124),
                                        "n_for_80pct_power": n_for_power(d_ao, d_so, 124)}
    # ---- J. judge leniency: Opus 5.5 / Opus 4.6 on the rollup arm (s2 / s5)
    res["judge_leniency"] = leniency()
    res["tied_probes"] = tied_probes()
    return res


def main():
    res = compute()
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def rate_of_rate(r):
    lo, hi = r.recall_ci
    return {"n": r.n, "recoverable": r.recoverable_n, "recall": r.recall, "ci": [lo, hi],
            "wrong": r.n_wrong, "attempted": r.attempted_n, "wrong_rate": r.wrong_rate,
            "absent": r.n_absent}


def bootstrap_pairs(a_ok, b_ok, seed, B):
    """Record-level paired percentile bootstrap, the method of rq3_heldout_equivalence_test.py."""
    n = len(a_ok)
    rng = random.Random(seed)
    d = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a_ok[i] for i in idx) / n - sum(b_ok[i] for i in idx) / n)
    d.sort()
    q = lambda p: round(100 * d[int(p * (B - 1))], 1)
    ci90 = [q(.05), q(.95)]
    return {"diff": round(100 * (sum(a_ok) - sum(b_ok)) / n, 1), "ci90": ci90, "ci95": [q(.025), q(.975)],
            "equiv_pm5": ci90[0] > -5 and ci90[1] < 5, "equiv_pm10": ci90[0] > -10 and ci90[1] < 10}


def power_mcnemar(b, c, n_now, alpha_z=1.959964):
    """Normal-approximation power of the McNemar test at n_now questions, discordant rates b/n, c/n."""
    pb, pc = b / n_now, c / n_now
    if pb + pc == 0:
        return 0.0
    z = (abs(pb - pc) * math.sqrt(n_now) - alpha_z * math.sqrt(pb + pc)) / math.sqrt(pb + pc - (pb - pc) ** 2)
    return round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)


def n_for_power(b, c, n_now, target=0.8, zb=0.841621, za=1.959964):
    pb, pc = b / n_now, c / n_now
    if pb == pc:
        return None
    return math.ceil(((za * math.sqrt(pb + pc) + zb * math.sqrt(pb + pc - (pb - pc) ** 2)) / abs(pb - pc)) ** 2)


def leniency(_roll=None):
    """Second and third judge on the original rollup arm's 124 answers (the
    fixed grader's prompt, one draw each), from rq3_judge_agreement.json. The 34
    records the grader assigns `absent` by rule carry no model call, so each
    judge's recoverable count over the 90 judged records is also its count over
    all 124."""
    d = json.loads((DATA / "rq3_judge_agreement.json").read_text())
    n_all = d["n_records_in_sample"]
    out = {"n_all": n_all, "n_judged": d["n_agreement_sample"]}
    for j, v in d["per_judge"].items():
        k = v["recoverable"]["n"]
        out[j] = {"recoverable": k, "recall_all": ar.r1(100 * k / n_all),
                  "wrong_rate": v["wrong_of_attempted"]["pct"]}
    out["kappa_range"] = [min(a["cohens_kappa_4way"] for a in d["agreement"].values()),
                          max(a["cohens_kappa_4way"] for a in d["agreement"].values())]
    out["wrong_vs_not_range"] = [min(a["exact_wrong_vs_not"]["pct"] for a in d["agreement"].values()),
                                 max(a["exact_wrong_vs_not"]["pct"] for a in d["agreement"].values())]
    return out


def tied_probes():
    """Prose-only against call-only on the probes whose two overlaps are equal."""
    d = json.loads((DATA / "rq3_arm_probe_overlap.json").read_text())
    tied = [x for x in d["per_probe"] if abs(x["prose_overlap"] - x["call_overlap"]) < 1e-12]
    p = sum(x["prose_grade"] in REC for x in tied)
    c = sum(x["call_grade"] in REC for x in tied)
    return {"n": len(tied), "prose": p, "call": c, "gap": ar.r1(100 * (p - c) / len(tied)),
            "mean_overlap_gap": d["gap"]["mean"] if "mean" in d["gap"] else None}


if __name__ == "__main__":
    main()
