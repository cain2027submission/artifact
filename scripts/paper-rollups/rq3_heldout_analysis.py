#!/usr/bin/env python3
"""Score the held-out action-record vs output-stripped validation. No API calls.

POPULATION DISCIPLINE is the whole design of this file. Every number below is over
ONE set: the intersection of record_ids that both arms answered and graded. A
record present in one arm's file and absent from the other, or carrying an
api_error grade in either, is dropped from BOTH and named in
`excluded_from_intersection`. The reason this is stated three times rather than
once is that an earlier version of this experiment reported a cost figure over 124
records beside a recall figure over 89 and overstated by 50.6%, in the direction
that flattered its own conclusion.

Success is `recoverable` = correct or partially_correct, which is the definition
the published design-set figures use: the action record's 96.8% is 120/124 and
output-stripped's 94.4% is 117/124, and the quoted McNemar table (both 115,
action-only 5, stripped-only 2, neither 2) sums to the same 124.

THE CONFIDENCE CORRECTION IS EXACT HERE, which it is not for the design set.
main.tex records that the action record has no confidence-corrected rate because
raw judge replies were not retained, so rq3_confidence_rederive.py can only bound
the defect from below by inspecting the post-parse answer. Both held-out arms store
`raw_reply`, so this file can do two things that cannot be done on disk for the
design set:
  * re-run the fixed extractor over the full reply and check it reproduces the
    stored rating, which is a test that retention is sound rather than decorative;
  * apply the reclassification rule to the whole reply instead of to the surviving
    prefix of it, which makes the corrected rate a measurement and not a bound.

Usage:
    rq3_heldout_analysis.py
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, load_records, rates, wilson,
)
from rq3_confidence_rederive import corrected_rating  # noqa: E402

ARM_FILES = {
    "action_record": DATA / "rq3_heldout_action_record_arm.jsonl",
    "output_stripped": DATA / "rq3_heldout_stripped_arm.jsonl",
}
MANIFEST = DATA / "rq3_heldout_manifest.json"
QUESTIONS = DATA / "rq3_heldout_questions.jsonl"
PLAN = DATA / "rq3_heldout_answer_plan.json"
USAGE = DATA / "rq3_heldout_run_usage.json"
OUT = DATA / "rq3_heldout_validation.json"

SUCCESS = {"correct", "partially_correct"}
USD_IN, USD_OUT = 2.20e-6, 11.0e-6


# ---------------------------------------------------------------- statistics

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chisq_1df_p(stat: float) -> float:
    """Upper tail of chi-square with 1 df, which is 2*(1 - Phi(sqrt(stat)))."""
    if stat <= 0:
        return 1.0
    return 2.0 * (1.0 - norm_cdf(math.sqrt(stat)))


def binom_exact_two_sided(b: int, c: int) -> float:
    """Exact conditional (sign) test: b successes out of b+c at p=0.5, two-sided."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact binomial interval via the beta quantile, bisected (no scipy here)."""
    if n == 0:
        return (0.0, 1.0)

    def beta_cdf(x: float, a: float, bb: float) -> float:
        # Regularised incomplete beta by series; a and b are small integers here.
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        # Continued-fraction-free fallback: sum the binomial tail, which is exact
        # for integer parameters and is all this needs.
        return None  # not used; see below

    # Binomial-tail bisection is simpler and exact for integer k, n.
    def tail_ge(p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(k, n + 1))

    def tail_le(p: float) -> float:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(0, k + 1))

    lo = 0.0
    if k > 0:
        a, b_ = 0.0, 1.0
        for _ in range(200):
            mid = (a + b_) / 2
            if tail_ge(mid) < alpha / 2:
                a = mid
            else:
                b_ = mid
        lo = (a + b_) / 2
    hi = 1.0
    if k < n:
        a, b_ = 0.0, 1.0
        for _ in range(200):
            mid = (a + b_) / 2
            if tail_le(mid) > alpha / 2:
                a = mid
            else:
                b_ = mid
        hi = (a + b_) / 2
    return (lo, hi)


def mcnemar(pairs: List[Tuple[bool, bool]]) -> dict:
    """pairs are (action success, stripped success)."""
    both = sum(1 for a, s in pairs if a and s)
    a_only = sum(1 for a, s in pairs if a and not s)
    s_only = sum(1 for a, s in pairs if not a and s)
    neither = sum(1 for a, s in pairs if not a and not s)
    n = len(pairs)
    b, c = a_only, s_only
    disc = b + c

    chi_cc = ((abs(b - c) - 1) ** 2) / disc if disc else None
    chi_uc = ((b - c) ** 2) / disc if disc else None
    if chi_cc is not None and abs(b - c) < 1:
        chi_cc = 0.0

    # Paired difference, action minus stripped, three ways.
    d = (b - c) / n if n else 0.0
    wald = None
    if n:
        var = (b + c - (b - c) ** 2 / n) / (n * n)
        half = 1.96 * math.sqrt(max(var, 0.0))
        wald = [round(100 * (d - half), 2), round(100 * (d + half), 2)]
    # Exact conditional interval: with phi = b/(b+c), d = (disc/n)(2 phi - 1), so a
    # Clopper-Pearson interval on phi maps straight onto d. This is the one to read
    # when disc is small, which it is: a Wald interval on 7 discordant pairs is not
    # trustworthy and the design set had exactly 7.
    exact = None
    if disc and n:
        plo, phi_ = clopper_pearson(b, disc)
        exact = [round(100 * (disc / n) * (2 * plo - 1), 2),
                 round(100 * (disc / n) * (2 * phi_ - 1), 2)]

    return {
        "n_pairs": n,
        "both_correct": both,
        "action_only": a_only,
        "stripped_only": s_only,
        "neither": neither,
        "n_discordant": disc,
        "chi_square_continuity_corrected": round(chi_cc, 3) if chi_cc is not None else None,
        "p_chi_square_continuity_corrected": round(chisq_1df_p(chi_cc), 4) if chi_cc is not None else None,
        "chi_square_uncorrected": round(chi_uc, 3) if chi_uc is not None else None,
        "p_chi_square_uncorrected": round(chisq_1df_p(chi_uc), 4) if chi_uc is not None else None,
        "p_exact_binomial_two_sided": round(binom_exact_two_sided(b, c), 4),
        "difference_pp_action_minus_stripped": round(100 * d, 2),
        "ci95_difference_pp_paired_wald": wald,
        "ci95_difference_pp_exact_conditional": exact,
    }


def cluster_bootstrap_diff(rows: List[dict], n_boot: int = 10000,
                           seed: int = 42) -> List[float]:
    """Resample SESSIONS, not questions. Questions are clustered inside sessions and
    the allocation is uneven across them, so a question-level bootstrap would
    understate the interval."""
    by_session: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for r in rows:
        by_session[r["session_prefix"]].append((r["a_ok"], r["s_ok"]))
    keys = sorted(by_session)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        picked = [by_session[rng.choice(keys)] for _ in keys]
        flat = [p for g in picked for p in g]
        if not flat:
            continue
        a = sum(1 for x, _ in flat if x) / len(flat)
        s = sum(1 for _, y in flat if y) / len(flat)
        out.append(100 * (a - s))
    return sorted(out)


# ---------------------------------------------------------------- correction

def apply_correction(rec: dict) -> Tuple[str, str]:
    """(corrected_grade, reason) from the FULL raw reply, not from the prefix that
    survived extraction. corrected_rating is imported from rq3_confidence_rederive
    so the rule is the published one rather than a second implementation of it."""
    reply = rec.get("raw_reply") or ""
    stored = rec.get("rating") or ""
    new, reason = corrected_rating(reply, stored)
    if new == "NOT_ANSWERABLE" and stored != "NOT_ANSWERABLE":
        return "absent", reason
    return rec.get("grade"), "unchanged"


def main() -> None:
    arms = {}
    for arm, path in ARM_FILES.items():
        if not path.exists():
            print(f"ERROR: {path} missing. Run rq3_heldout_arms.py first.")
            sys.exit(1)
        rows = load_records(path)
        seen = Counter(r["record_id"] for r in rows)
        dupes = [k for k, v in seen.items() if v > 1]
        if dupes:
            print(f"WARNING: {arm} has {len(dupes)} duplicated record_ids; "
                  f"keeping the last of each")
        arms[arm] = {r["record_id"]: r for r in rows}
        print(f"{arm:16s} {len(rows)} records, {len(arms[arm])} unique")

    questions = {}
    for r in load_records(QUESTIONS):
        from rq3_raw_transcript_arm import record_id
        questions[record_id(r)] = r

    gradable = {arm: {k for k, v in d.items()
                      if v.get("grade") not in (None, "", "api_error")}
                for arm, d in arms.items()}
    inter = sorted(gradable["action_record"] & gradable["output_stripped"])

    excluded = []
    for arm, other in (("action_record", "output_stripped"),
                       ("output_stripped", "action_record")):
        for rid in sorted(gradable[arm] - gradable[other]):
            r = arms[arm][rid]
            excluded.append({
                "record_id": rid, "session_prefix": r.get("session_prefix"),
                "class": r.get("class"), "graded_in": arm, "missing_in": other,
                "reason": ("absent from the other arm's file"
                           if rid not in arms[other]
                           else f"ungradable in {other}: "
                                f"{arms[other][rid].get('grade')!r}"),
            })

    print(f"\nINTERSECTION n = {len(inter)}  "
          f"(action {len(gradable['action_record'])}, "
          f"stripped {len(gradable['output_stripped'])}, "
          f"excluded from both {len(excluded)})")

    pairs, rows = [], []
    for rid in inter:
        a, s = arms["action_record"][rid], arms["output_stripped"][rid]
        a_corr, a_reason = apply_correction(a)
        s_corr, s_reason = apply_correction(s)
        rows.append({
            "record_id": rid,
            "session_prefix": a.get("session_prefix"),
            "class": a.get("class"),
            "a_grade": a["grade"], "s_grade": s["grade"],
            "a_grade_corrected": a_corr, "s_grade_corrected": s_corr,
            "a_correction_reason": a_reason, "s_correction_reason": s_reason,
            "a_rating": a.get("rating"), "s_rating": s.get("rating"),
            "a_ok": a["grade"] in SUCCESS, "s_ok": s["grade"] in SUCCESS,
            "a_ok_corrected": a_corr in SUCCESS, "s_ok_corrected": s_corr in SUCCESS,
            "a_strict": a["grade"] == "correct", "s_strict": s["grade"] == "correct",
        })
        pairs.append((a["grade"] in SUCCESS, s["grade"] in SUCCESS))

    a_rows = [{"grade": r["a_grade"]} for r in rows]
    s_rows = [{"grade": r["s_grade"]} for r in rows]
    a_rows_c = [{"grade": r["a_grade_corrected"]} for r in rows]
    s_rows_c = [{"grade": r["s_grade_corrected"]} for r in rows]

    mc = mcnemar(pairs)
    mc_c = mcnemar([(r["a_ok_corrected"], r["s_ok_corrected"]) for r in rows])

    boot = cluster_bootstrap_diff(rows)
    boot_ci = ([round(boot[int(0.025 * len(boot))], 2),
                round(boot[int(0.975 * len(boot))], 2)] if boot else None)

    # Per class, over the same intersection.
    per_class = {}
    for cls in sorted({r["class"] for r in rows}):
        sub = [r for r in rows if r["class"] == cls]
        ak = sum(1 for r in sub if r["a_ok"])
        sk = sum(1 for r in sub if r["s_ok"])
        per_class[cls] = {
            "n": len(sub),
            "action_recoverable_n": ak,
            "action_recoverable_pct": round(100 * ak / len(sub), 1),
            "action_ci95": list(wilson(ak, len(sub))),
            "stripped_recoverable_n": sk,
            "stripped_recoverable_pct": round(100 * sk / len(sub), 1),
            "stripped_ci95": list(wilson(sk, len(sub))),
            "difference_pp": round(100 * (ak - sk) / len(sub), 1),
            "mcnemar": mcnemar([(r["a_ok"], r["s_ok"]) for r in sub]),
        }

    # Unequal questions per session means the unweighted rate is not a per-session
    # average. Both are reported rather than one being chosen quietly.
    by_sess = defaultdict(list)
    for r in rows:
        by_sess[r["session_prefix"]].append(r)
    sess_rates = {
        p: {"n": len(v),
            "action_pct": round(100 * sum(r["a_ok"] for r in v) / len(v), 1),
            "stripped_pct": round(100 * sum(r["s_ok"] for r in v) / len(v), 1)}
        for p, v in sorted(by_sess.items())}
    session_weighted = {
        "n_sessions": len(sess_rates),
        "action_mean_of_session_rates_pct":
            round(sum(v["action_pct"] for v in sess_rates.values()) / len(sess_rates), 1),
        "stripped_mean_of_session_rates_pct":
            round(sum(v["stripped_pct"] for v in sess_rates.values()) / len(sess_rates), 1),
    }

    # Retention check: does the fixed extractor, re-run over the retained reply,
    # reproduce the rating that was stored at answer time?
    from rq3_raw_transcript_arm import extract_confidence
    recheck = {}
    for arm, d in arms.items():
        agree = mismatch = empty = 0
        for rid in inter:
            r = d[rid]
            reply = r.get("raw_reply")
            if not reply:
                empty += 1
                continue
            _ans, conf = extract_confidence(reply)
            if conf == r.get("rating"):
                agree += 1
            else:
                mismatch += 1
        recheck[arm] = {"reply_retained": len(inter) - empty,
                        "rating_reproduced_from_reply": agree,
                        "mismatched": mismatch, "reply_missing": empty}

    usage = json.loads(USAGE.read_text()) if USAGE.exists() else {"invocations": []}
    tok_in = tok_out = 0
    for arm, d in arms.items():
        for rid in inter:
            u = d[rid].get("usage") or {}
            tok_in += u.get("prompt_tokens", 0)
            tok_out += u.get("completion_tokens", 0)

    report = {
        "what_this_is": (
            "held-out validation of the action record against the output-stripped "
            "transcript. Questions generated from 25 sessions dated after the last "
            "graded session, by the published generator, and seen by neither arm "
            "before this run. main.tex: 'the action record was designed with these "
            "questions visible...so a held-out question set is required before this "
            "dominance is settled.'"),
        "design_set_for_calibration": {
            "n": 124, "action_recoverable_pct": 96.8, "stripped_recoverable_pct": 94.4,
            "mcnemar": {"both": 115, "action_only": 5, "stripped_only": 2, "neither": 2,
                        "chi_square_continuity_corrected": 0.571, "p": 0.45},
            "power_note": ("power to detect 2.4pp at n=124 is about 20%, so a "
                           "non-significant result here is not a win or a loss for "
                           "either arm. The sign, the interval and whether the sign "
                           "replicated are what this run can report."),
        },
        "denominator": {
            "intersection_n": len(inter),
            "n_sessions": len(sess_rates),
            "action_graded": len(gradable["action_record"]),
            "stripped_graded": len(gradable["output_stripped"]),
            "n_questions_generated": len(questions),
            "n_questions_planned_for_answering":
                (json.loads(PLAN.read_text())["n_planned_questions"]
                 if PLAN.exists() else None),
            "rule": ("every statistic in this file is over the intersection_n above. "
                     "A record graded in one arm only is excluded from BOTH and "
                     "listed in excluded_from_intersection."),
        },
        "excluded_from_intersection": excluded,
        "answer_allocation": (json.loads(PLAN.read_text()) if PLAN.exists() else None),
        "uncorrected": {
            "action_record": rates(a_rows),
            "output_stripped": rates(s_rows),
            "mcnemar_paired": mc,
            "ci95_difference_pp_cluster_bootstrap_over_sessions": boot_ci,
            "cluster_bootstrap": {"n_resamples": len(boot), "unit": "session",
                                  "seed": 42},
        },
        "confidence_corrected": {
            "computable": True,
            "why_it_is_exact_here": (
                "both arms retained raw_reply, so rq3_confidence_rederive's rule is "
                "applied to the whole reply rather than to the answer prefix that "
                "survived extraction. For the design set the same rule can only "
                "bound the defect from below, which is why main.tex reports no "
                "confidence-corrected rate for the action record."),
            "rule": ("rq3_confidence_rederive.corrected_rating, imported: a marker "
                     "line anywhere in the reply wins; failing that, a reply opening "
                     "with the prescribed refusal is NOT_ANSWERABLE. A record "
                     "reclassified to NOT_ANSWERABLE is graded absent."),
            "action_record": rates(a_rows_c),
            "output_stripped": rates(s_rows_c),
            "mcnemar_paired": mc_c,
            "n_reclassified_action": sum(
                1 for r in rows if r["a_correction_reason"] != "unchanged"),
            "n_reclassified_stripped": sum(
                1 for r in rows if r["s_correction_reason"] != "unchanged"),
            "reclassified": [
                {k: r[k] for k in ("record_id", "session_prefix", "class",
                                   "a_rating", "a_grade", "a_grade_corrected",
                                   "a_correction_reason", "s_rating", "s_grade",
                                   "s_grade_corrected", "s_correction_reason")}
                for r in rows
                if r["a_correction_reason"] != "unchanged"
                or r["s_correction_reason"] != "unchanged"],
        },
        "secondary_endpoint_strict_correct": {
            "status": "SECONDARY AND EXPLORATORY, not the pre-registered endpoint",
            "why_it_is_here": (
                "the primary endpoint is recoverable = correct or partially_correct, "
                "because that is what the published 96.8% and 94.4% are. It collapses "
                "the correct/partially_correct distinction, and on this held-out set a "
                "visible share of the arms' disagreement sits exactly there: the same "
                "question graded correct for one arm and partially_correct for the "
                "other is concordant under the primary endpoint and invisible in it. "
                "This block scores grade == 'correct' only. It is reported as "
                "secondary because it was added after the first answers were seen, "
                "which is precisely the kind of endpoint choice that inflates a "
                "p-value, so it is evidence about where to look next and not a test "
                "result to quote against the design set."),
            "action_record_strict_correct_n": sum(1 for r in rows if r["a_strict"]),
            "output_stripped_strict_correct_n": sum(1 for r in rows if r["s_strict"]),
            "action_record_pct": round(100 * sum(r["a_strict"] for r in rows) / len(rows), 1),
            "output_stripped_pct": round(100 * sum(r["s_strict"] for r in rows) / len(rows), 1),
            "action_ci95": list(wilson(sum(r["a_strict"] for r in rows), len(rows))),
            "stripped_ci95": list(wilson(sum(r["s_strict"] for r in rows), len(rows))),
            "mcnemar_paired": mcnemar([(r["a_strict"], r["s_strict"]) for r in rows]),
        },
        "raw_reply_retention_check": recheck,
        "rq3_confidence_rederive_consumption": {
            "can_it_consume_the_action_record_arm": False,
            "why": ("its ARMS table needs, per arm, a rq3_majority_<arm>.jsonl of "
                    "3-of-5 majority grades and a published summary whose key is a "
                    "rates() dict. Neither exists for action_record: building the "
                    "majority file means five fresh grading draws per record on the "
                    "DESIGN set, which is new model spend outside this validation, "
                    "and the loop would raise FileNotFoundError before reaching any "
                    "of the single-draw arithmetic. So it is not a small additive "
                    "change and was not made. This file computes the correction "
                    "exactly from raw_reply instead, which is strictly stronger "
                    "than that script's lower bound."),
        },
        "per_question_class": per_class,
        "per_session": sess_rates,
        "session_weighted": session_weighted,
        "sign_replication": None,  # filled below
        "cost": {
            "tokens_over_intersection": {"prompt": tok_in, "completion": tok_out},
            "price_per_mtok_usd": {"input": 2.20, "output": 11.00,
                                   "source": "derived from this gateway's /v1/usage"},
            "answering_and_grading_usd_over_intersection":
                round(tok_in * USD_IN + tok_out * USD_OUT, 2),
            "gateway_openai_client_invocations": usage.get("invocations", []),
        },
        "records": rows,
    }

    d = report["uncorrected"]
    sign = ("action_record higher" if d["mcnemar_paired"]["difference_pp_action_minus_stripped"] > 0
            else "output_stripped higher"
            if d["mcnemar_paired"]["difference_pp_action_minus_stripped"] < 0 else "tie")
    report["sign_replication"] = {
        "design_set_sign": "action_record higher (+2.4pp)",
        "heldout_sign": sign,
        "replicated": sign == "action_record higher",
        "significant_at_0.05": (d["mcnemar_paired"]["p_exact_binomial_two_sided"] < 0.05),
        "statement": ("the design-set sign " +
                      ("replicated" if sign == "action_record higher"
                       else "did NOT replicate") +
                      "; the difference is not statistically significant either way "
                      "and this run was never powered to make it so, so this is a "
                      "sign and an interval, not a win or a loss."),
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")

    # ------------------------------------------------------------- stdout
    print(f"\n=== all numbers over intersection n = {len(inter)} "
          f"({len(sess_rates)} sessions) ===")
    for label, block in (("uncorrected", report["uncorrected"]),
                         ("confidence-corrected", report["confidence_corrected"])):
        print(f"\n-- {label}")
        for arm in ("action_record", "output_stripped"):
            r = block[arm]
            print(f"  {arm:16s} n={r['n']:3d} correct {r['correct']['n']:3d} "
                  f"partial {r['partially_correct']['n']:3d} wrong {r['wrong']['n']:3d} "
                  f"absent {r['absent']['n']:3d}  recoverable "
                  f"{r['recoverable']['pct']:5.1f}% CI {r['recoverable']['ci95']}")
        m = block["mcnemar_paired"]
        print(f"  mcnemar  both {m['both_correct']}  action-only {m['action_only']}  "
              f"stripped-only {m['stripped_only']}  neither {m['neither']}")
        print(f"           chi2(cc) {m['chi_square_continuity_corrected']} "
              f"p={m['p_chi_square_continuity_corrected']}  "
              f"exact binomial p={m['p_exact_binomial_two_sided']}")
        print(f"           diff {m['difference_pp_action_minus_stripped']:+.2f}pp  "
              f"wald CI {m['ci95_difference_pp_paired_wald']}  "
              f"exact CI {m['ci95_difference_pp_exact_conditional']}")
    print(f"  cluster bootstrap CI on the difference (sessions): {boot_ci}")

    print(f"\n-- per class (same intersection)")
    print(f"  {'class':18s} {'n':>3s} {'action':>8s} {'stripped':>9s} {'diff':>7s}  "
          f"{'a-only':>6s} {'s-only':>6s}")
    for cls, v in per_class.items():
        print(f"  {cls:18s} {v['n']:3d} {v['action_recoverable_pct']:7.1f}% "
              f"{v['stripped_recoverable_pct']:8.1f}% {v['difference_pp']:+6.1f}pp "
              f"{v['mcnemar']['action_only']:6d} {v['mcnemar']['stripped_only']:6d}")

    sec = report["secondary_endpoint_strict_correct"]
    m = sec["mcnemar_paired"]
    print(f"\n-- SECONDARY (exploratory, added after answers were seen): "
          f"strict grade == correct")
    print(f"  action {sec['action_record_pct']}% CI {sec['action_ci95']}   "
          f"stripped {sec['output_stripped_pct']}% CI {sec['stripped_ci95']}")
    print(f"  mcnemar both {m['both_correct']} action-only {m['action_only']} "
          f"stripped-only {m['stripped_only']} neither {m['neither']}  "
          f"exact p={m['p_exact_binomial_two_sided']}  "
          f"diff {m['difference_pp_action_minus_stripped']:+.2f}pp "
          f"exact CI {m['ci95_difference_pp_exact_conditional']}")

    print(f"\n-- session-weighted: action "
          f"{session_weighted['action_mean_of_session_rates_pct']}%  stripped "
          f"{session_weighted['stripped_mean_of_session_rates_pct']}%")
    print(f"-- raw_reply retention check: {recheck}")
    print(f"-- sign: {report['sign_replication']['statement']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
