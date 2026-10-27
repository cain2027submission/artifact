#!/usr/bin/env python3
"""Two sensitivities on the held-out result, both no-API, both appended to
rq3_heldout_validation.json rather than replacing anything in it.

1. REFUSAL-ANYWHERE. rq3_confidence_rederive's rule gives a marker line precedence
   over a refusal, so a reply that opens "NOT ANSWERABLE -- ..." and then signs off
   "ANSWERED: PARTIALLY_ANSWERED" keeps the marker and stays graded. That is the
   published rule and the primary number uses it unchanged. This counts how many
   replies look like that and reports what the rates would be if a refusal opener
   beat the marker instead, which is the stricter reading of the same evidence.

2. THE OUTCOME-CLASS TRUNCATION CONFOUND, which the per-class table surfaced: both
   arms score 46.2% on `outcome` against 100% on `sequence`. The generator sees
   transcript_text[:30000] while both arms see the whole artifact, so on a large
   session a question asking what the state was "by the final message" is asking
   about the final message of the first 30 KB, and an arm that correctly describes
   the real end of a 1 MB session is graded wrong. This is a property of the
   published generator, it hits both arms through the same artifact, and it is
   therefore unbiased for the paired contrast -- but it caps both absolute rates
   and it inflates the concordant-wrong cell, which is where power goes. This
   re-runs the primary test with `outcome` excluded to show the paired result does
   not depend on it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import DATA, rates, wilson  # noqa: E402
from rq3_confidence_rederive import REFUSAL_OPENER, MARKER_LINE  # noqa: E402
from rq3_heldout_analysis import mcnemar, cluster_bootstrap_diff  # noqa: E402

VALID = DATA / "rq3_heldout_validation.json"
FILES = {"action_record": DATA / "rq3_heldout_action_record_arm.jsonl",
         "output_stripped": DATA / "rq3_heldout_stripped_arm.jsonl"}
SUCCESS = {"correct", "partially_correct"}


def main() -> None:
    report = json.loads(VALID.read_text())
    rows = report["records"]
    arms = {a: {r["record_id"]: r for r in
                (json.loads(l) for l in p.read_text().splitlines() if l.strip())}
            for a, p in FILES.items()}

    # ---- 1. refusal-anywhere
    flagged = {"action_record": [], "output_stripped": []}
    for arm, d in arms.items():
        for r in rows:
            rec = d[r["record_id"]]
            reply = rec.get("raw_reply") or ""
            if rec.get("rating") == "NOT_ANSWERABLE":
                continue
            if REFUSAL_OPENER.match(reply):
                flagged[arm].append({"record_id": r["record_id"],
                                     "session_prefix": r["session_prefix"],
                                     "class": r["class"],
                                     "stored_rating": rec.get("rating"),
                                     "grade": rec.get("grade"),
                                     "reply_opens": reply.strip()[:120]})
    fa = {a: {x["record_id"] for x in v} for a, v in flagged.items()}
    strict_pairs, strict_rows = [], {"action_record": [], "output_stripped": []}
    for r in rows:
        ok = {}
        for arm, key in (("action_record", "a_grade"), ("output_stripped", "s_grade")):
            g = "absent" if r["record_id"] in fa[arm] else r[key]
            strict_rows[arm].append({"grade": g})
            ok[arm] = g in SUCCESS
        strict_pairs.append((ok["action_record"], ok["output_stripped"]))

    sens1 = {
        "rule_applied": ("a reply whose FIRST line matches the refusal opener is "
                         "NOT_ANSWERABLE even when a later marker line says "
                         "otherwise. This is stricter than the published rule and is "
                         "reported as a sensitivity, not as the result."),
        "n_reclassified": {a: len(v) for a, v in flagged.items()},
        "reclassified": flagged,
        "action_record": rates(strict_rows["action_record"]),
        "output_stripped": rates(strict_rows["output_stripped"]),
        "mcnemar_paired": mcnemar(strict_pairs),
    }

    # ---- 2. outcome excluded
    keep = [r for r in rows if r["class"] != "outcome"]
    pairs = [(r["a_ok"], r["s_ok"]) for r in keep]
    boot = cluster_bootstrap_diff(keep)
    sens2 = {
        "why": ("the generator sees transcript_text[:30000] and both arms see the "
                "whole artifact, so an `outcome` question about the final state of a "
                "1 MB session is about the final state of its first 30 KB. Both arms "
                "answer from the same artifact, so this is unbiased for the paired "
                "contrast, but it caps both absolute rates and fills the "
                "concordant-wrong cell. Shown excluded to confirm the paired result "
                "does not hinge on it."),
        "n": len(keep),
        "action_record": rates([{"grade": r["a_grade"]} for r in keep]),
        "output_stripped": rates([{"grade": r["s_grade"]} for r in keep]),
        "mcnemar_paired": mcnemar(pairs),
        "ci95_difference_pp_cluster_bootstrap_over_sessions":
            [round(boot[int(0.025 * len(boot))], 2),
             round(boot[int(0.975 * len(boot))], 2)] if boot else None,
    }

    report["sensitivity_refusal_anywhere"] = sens1
    report["sensitivity_outcome_class_excluded"] = sens2
    VALID.write_text(json.dumps(report, indent=2) + "\n")

    for name, s in (("refusal-anywhere (stricter than published)", sens1),
                    ("outcome class excluded", sens2)):
        m = s["mcnemar_paired"]
        print(f"\n-- sensitivity: {name}")
        if "n_reclassified" in s:
            print(f"   reclassified to absent: {s['n_reclassified']}")
        for arm in ("action_record", "output_stripped"):
            print(f"   {arm:16s} n={s[arm]['n']:3d} recoverable "
                  f"{s[arm]['recoverable']['pct']:5.1f}% CI "
                  f"{s[arm]['recoverable']['ci95']}")
        print(f"   mcnemar both {m['both_correct']} action-only {m['action_only']} "
              f"stripped-only {m['stripped_only']} neither {m['neither']}  "
              f"exact p={m['p_exact_binomial_two_sided']}  "
              f"diff {m['difference_pp_action_minus_stripped']:+.2f}pp  "
              f"exact CI {m['ci95_difference_pp_exact_conditional']}")
    print(f"\nupdated {VALID}")


if __name__ == "__main__":
    main()
