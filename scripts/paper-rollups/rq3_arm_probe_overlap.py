#!/usr/bin/env python3
"""Probe-provenance overlap for the prose-only and call-only RQ3 arms.

The paper discloses mean probe-token overlap of 0.902 against the raw transcript
and 0.705 against the rollup, a 0.197 gap that trips its own 0.15 fairness
threshold (``rq4_known_item.py`` prints that WARNING; the stored values live in
``data/paper-rollups/rq4_known_item_n300.json`` under ``fairness``). That
comparison is whole-transcript vs rollup. It says nothing about the pair the new
headline rests on: prose-only (89.5% at 3,082 tokens) against call-only (47.2% at
6,329 tokens), where the 42.3-point gap is attributed to content class.

The claim under test is that probe-provenance bias "is shared between them and
largely cancels". Probes are natural-language questions a model wrote while
reading the transcript, so they plausibly share far more tokens with prose than
with JSON tool-call arguments. This script measures it.

Method, reused rather than reimplemented:
  * overlap  -- ``rq4_known_item.compute_fairness_stats``, imported verbatim.
    Containment, not Jaccard: |probe_tokens & doc_tokens| / |probe_tokens|, over
    casefolded sets from ``TOKEN_RE = r"\\w+|[^\\w\\s]"``, no stopword removal.
  * prose-only context  -- ``rq3_prose_only_arm.prose_only_context`` (default
    keep_continuation=False, the setting behind the published 89.5%).
  * call-only context   -- ``rq3_tool_inputs_only_arm.tool_inputs_only_context``.

Denominator: records present in BOTH arm artifacts
(``rq3_prose_only_arm.jsonl``, 124; ``rq3_tool_inputs_only_arm.jsonl``, 123).
A record missing from either arm is excluded from both and counted, never scored
as zero overlap.

Usage:
    rq3_arm_probe_overlap.py            # writes the JSON report
    rq3_arm_probe_overlap.py --no-write # print only
"""

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402
from rq4_known_item import compute_fairness_stats  # noqa: E402
from rq3_prose_only_arm import prose_only_context  # noqa: E402
from rq3_tool_inputs_only_arm import tool_inputs_only_context  # noqa: E402

PROSE_ARM = DATA_DIR / "rq3_prose_only_arm.jsonl"
CALL_ARM = DATA_DIR / "rq3_tool_inputs_only_arm.jsonl"
OUT = DATA_DIR / "rq3_arm_probe_overlap.json"

FAIRNESS_THRESHOLD = 0.15
RECOVERABLE = ("correct", "partially_correct")


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["record_id"]] = r
    return rows


def rate(grades):
    n = len(grades)
    if n == 0:
        return None, 0, 0
    k = sum(1 for g in grades if g in RECOVERABLE)
    return 100.0 * k / n, k, n


def summarise(values):
    return {
        "n": len(values),
        "mean": st.mean(values),
        "median": st.median(values),
        "p25": st.quantiles(values, n=4)[0] if len(values) > 3 else None,
        "p75": st.quantiles(values, n=4)[2] if len(values) > 3 else None,
        "min": min(values),
        "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    prose_rows = load(PROSE_ARM)
    call_rows = load(CALL_ARM)

    only_prose = sorted(set(prose_rows) - set(call_rows))
    only_call = sorted(set(call_rows) - set(prose_rows))
    both = sorted(set(prose_rows) & set(call_rows))

    print(f"prose-only arm records: {len(prose_rows)}")
    print(f"call-only  arm records: {len(call_rows)}")
    print(f"in both (denominator):  {len(both)}")
    print(f"excluded, missing from call-only: {len(only_prose)} {only_prose}")
    print(f"excluded, missing from prose-only: {len(only_call)} {only_call}")

    # Context strings are not persisted by either arm, so rebuild them from the
    # transcript path each arm recorded, with that arm's own builder. Cache per
    # transcript: 25 sessions carry 123 questions.
    cache = {}

    def contexts(path):
        if path not in cache:
            raw = Path(path).read_text(errors="replace")
            prose, _ = prose_only_context(raw)
            call, cstats = tool_inputs_only_context(raw)
            cache[path] = (prose, call, cstats)
        return cache[path]

    rows = []
    for rid in both:
        p, c = prose_rows[rid], call_rows[rid]
        assert p["question"] == c["question"], rid
        assert p["transcript"] == c["transcript"], rid
        prose_ctx, call_ctx, cstats = contexts(p["transcript"])
        # One call, both documents, so the same function produces both numbers.
        prose_ov, call_ov = compute_fairness_stats(p["question"], prose_ctx, call_ctx)
        rows.append({
            "record_id": rid,
            "session_id": p["session_id"],
            "class": p.get("class"),
            "question_chars": len(p["question"]),
            "prose_overlap": prose_ov,
            "call_overlap": call_ov,
            "prose_grade": p["grade"],
            "call_grade": c["grade"],
            "prose_ctx_chars": len(prose_ctx),
            "call_ctx_chars": len(call_ctx),
            "session_tool_calls": cstats.get("tool_calls", 0),
        })

    prose_ov = [r["prose_overlap"] for r in rows]
    call_ov = [r["call_overlap"] for r in rows]
    gap = st.mean(prose_ov) - st.mean(call_ov)
    median_gap = st.median(prose_ov) - st.median(call_ov)
    paired = [r["prose_overlap"] - r["call_overlap"] for r in rows]

    print(f"\n## overlap, n = {len(rows)} probes over "
          f"{len({r['session_id'] for r in rows})} sessions")
    print(f"prose-only: mean {st.mean(prose_ov):.3f}  median {st.median(prose_ov):.3f}")
    print(f"call-only : mean {st.mean(call_ov):.3f}  median {st.median(call_ov):.3f}")
    print(f"gap: mean {gap:+.3f}  median-of-medians {median_gap:+.3f}  "
          f"mean paired difference {st.mean(paired):+.3f}")
    print(f"threshold {FAIRNESS_THRESHOLD}: "
          f"{'TRIPPED' if abs(gap) > FAIRNESS_THRESHOLD else 'not tripped'} "
          f"({abs(gap) / FAIRNESS_THRESHOLD:.2f}x)")
    print(f"probes where prose overlap > call overlap: "
          f"{sum(1 for d in paired if d > 0)} of {len(paired)}  "
          f"(call higher on {sum(1 for d in paired if d < 0)}, "
          f"tied on {sum(1 for d in paired if d == 0)})")

    rnd = random.Random(42)
    boot = sorted(st.mean(rnd.choices(paired, k=len(paired))) for _ in range(2000))
    ci = [boot[50], boot[1949]]
    print(f"bootstrap 95% CI on the mean paired overlap gap "
          f"(2000 resamples, seed 42): [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Sessions with no tool calls hand call-only a header and nothing else. They
    # are a known composition problem for this arm, so report the set both ways.
    no_tool = [r for r in rows if r["session_tool_calls"] == 0]
    with_tool = [r for r in rows if r["session_tool_calls"] > 0]
    print(f"\nprobes from sessions with zero tool calls: {len(no_tool)} "
          f"(sessions {sorted({r['session_id'][:30] for r in no_tool})})")
    if with_tool:
        wp = [r["prose_overlap"] for r in with_tool]
        wc = [r["call_overlap"] for r in with_tool]
        print(f"excluding them, n = {len(with_tool)}: prose mean {st.mean(wp):.3f} "
              f"median {st.median(wp):.3f}; call mean {st.mean(wc):.3f} "
              f"median {st.median(wc):.3f}; gap {st.mean(wp) - st.mean(wc):+.3f}")

    # Recall by tercile of prose-only overlap. Grades are already on disk, so
    # this is a regrouping of the published uncorrected grades.
    def tercile_table(key, label):
        ordered = sorted(rows, key=lambda r: (key(r), r["record_id"]))
        n = len(ordered)
        cuts = [0, n // 3, 2 * n // 3, n]
        out = []
        for i in range(3):
            chunk = ordered[cuts[i]:cuts[i + 1]]
            p_rate, p_k, p_n = rate([r["prose_grade"] for r in chunk])
            c_rate, c_k, c_n = rate([r["call_grade"] for r in chunk])
            out.append({
                "grouping": label,
                "tercile": i + 1,
                "n": len(chunk),
                "key_range": [key(chunk[0]), key(chunk[-1])],
                "prose_overlap_mean": st.mean(r["prose_overlap"] for r in chunk),
                "call_overlap_mean": st.mean(r["call_overlap"] for r in chunk),
                "prose_recoverable_pct": p_rate,
                "prose_recoverable": [p_k, p_n],
                "call_recoverable_pct": c_rate,
                "call_recoverable": [c_k, c_n],
                "recall_gap_pp": p_rate - c_rate,
            })
        return out

    terciles = tercile_table(lambda r: r["prose_overlap"], "prose_only_overlap")
    adv_terciles = tercile_table(
        lambda r: r["prose_overlap"] - r["call_overlap"], "prose_minus_call_overlap")

    overall_p, pk, pn = rate([r["prose_grade"] for r in rows])
    overall_c, ck, cn = rate([r["call_grade"] for r in rows])
    print(f"\n## uncorrected recoverable recall, regrouped by prose-only overlap tercile")
    print(f"{'tercile':8s} {'n':>4s} {'prose ov':>9s} {'call ov':>8s} "
          f"{'prose rec':>10s} {'call rec':>9s} {'gap':>8s}")
    for t in terciles:
        print(f"{t['tercile']:<8d} {t['n']:4d} {t['prose_overlap_mean']:9.3f} "
              f"{t['call_overlap_mean']:8.3f} "
              f"{t['prose_recoverable_pct']:9.1f}% {t['call_recoverable_pct']:8.1f}% "
              f"{t['recall_gap_pp']:+7.1f}pp")
    print(f"{'ALL':8s} {len(rows):4d} {st.mean(prose_ov):9.3f} {st.mean(call_ov):8.3f} "
          f"{overall_p:9.1f}% {overall_c:8.1f}% {overall_p - overall_c:+7.1f}pp")
    print(f"\nprose recoverable {pk}/{pn}, call recoverable {ck}/{cn}")

    print(f"\n## same recall, regrouped by the per-probe prose-minus-call overlap "
          f"advantage (tercile 1 = probes that favour prose-only least)")
    print(f"{'tercile':8s} {'n':>4s} {'prose ov':>9s} {'call ov':>8s} "
          f"{'prose rec':>10s} {'call rec':>9s} {'gap':>8s}  key range")
    for t in adv_terciles:
        print(f"{t['tercile']:<8d} {t['n']:4d} {t['prose_overlap_mean']:9.3f} "
              f"{t['call_overlap_mean']:8.3f} "
              f"{t['prose_recoverable_pct']:9.1f}% {t['call_recoverable_pct']:8.1f}% "
              f"{t['recall_gap_pp']:+7.1f}pp  "
              f"[{t['key_range'][0]:+.3f}, {t['key_range'][1]:+.3f}]")

    report = {
        "question": ("does probe-provenance bias cancel between the prose-only "
                     "and call-only RQ3 arms, or does it favour prose-only?"),
        "overlap_metric": {
            "source": "scripts/paper-rollups/rq4_known_item.py::compute_fairness_stats",
            "definition": "containment: |probe_tokens & doc_tokens| / |probe_tokens|",
            "tokeniser": r"TOKEN_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)",
            "casefolding": "probe and document both .lower()",
            "stopwords": "none removed",
            "set_based": True,
            "is_jaccard": False,
            "reused_verbatim": True,
            "paper_reference": ("same function produced mean 0.902 transcript / "
                                "0.705 rollup over 600 RQ4 probes, stored in "
                                "data/paper-rollups/rq4_known_item_n300.json"),
        },
        "context_builders": {
            "prose_only": "scripts/paper-rollups/rq3_prose_only_arm.py::prose_only_context (keep_continuation=False)",
            "call_only": "scripts/paper-rollups/rq3_tool_inputs_only_arm.py::tool_inputs_only_context",
            "persisted": False,
            "note": "neither arm stores its context string, so both were rebuilt from the transcript path each arm recorded",
        },
        "denominator": {
            "prose_arm_records": len(prose_rows),
            "call_arm_records": len(call_rows),
            "scored_in_both": len(rows),
            "sessions": len({r["session_id"] for r in rows}),
            "excluded_missing_from_call_arm": only_prose,
            "excluded_missing_from_prose_arm": only_call,
            "exclusion_rule": "a record absent from either arm is excluded from both, never scored as zero overlap",
        },
        "prose_only_overlap": summarise(prose_ov),
        "call_only_overlap": summarise(call_ov),
        "gap": {
            "mean_difference": gap,
            "median_difference": median_gap,
            "mean_paired_difference": st.mean(paired),
            "mean_paired_difference_ci95_bootstrap": ci,
            "bootstrap": {"resamples": 2000, "seed": 42},
            "fairness_threshold": FAIRNESS_THRESHOLD,
            "trips_threshold": abs(gap) > FAIRNESS_THRESHOLD,
            "multiples_of_threshold": abs(gap) / FAIRNESS_THRESHOLD,
            "probes_favouring_prose": sum(1 for d in paired if d > 0),
            "probes_favouring_call": sum(1 for d in paired if d < 0),
            "probes_tied": sum(1 for d in paired if d == 0),
        },
        "zero_tool_call_sessions": {
            "n_probes": len(no_tool),
            "sessions": sorted({r["session_id"] for r in no_tool}),
            "prose_overlap_mean_excluding": st.mean(r["prose_overlap"] for r in with_tool) if with_tool else None,
            "call_overlap_mean_excluding": st.mean(r["call_overlap"] for r in with_tool) if with_tool else None,
            "gap_excluding": (st.mean(r["prose_overlap"] for r in with_tool)
                              - st.mean(r["call_overlap"] for r in with_tool)) if with_tool else None,
        },
        "recall_by_prose_overlap_tercile": terciles,
        "recall_by_overlap_advantage_tercile": adv_terciles,
        "recall_overall": {
            "prose_recoverable_pct": overall_p,
            "prose_recoverable": [pk, pn],
            "call_recoverable_pct": overall_c,
            "call_recoverable": [ck, cn],
            "gap_pp": overall_p - overall_c,
            "grade_stage": "uncorrected, the stage the paper's 89.5% / 47.2% headline uses",
            "published_prose_pct_on_124": 89.5,
            "published_call_pct_on_123": 47.2,
        },
        "class_distribution": dict(Counter(r["class"] for r in rows)),
        "per_probe": rows,
    }

    if not args.no_write:
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
