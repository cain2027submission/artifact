#!/usr/bin/env python3
"""Three-way paired RQ3 comparison: rollup vs stripped transcript vs raw transcript.

The three arms hold everything fixed except the context the judge is shown --
same questions, same ground truth, same judge model, same prompt modulo one
clause naming the context, same parser, same majority-vote rule. So a difference
between arms is attributable to the context source and nothing else.

The arms do not cover the same records, which is the whole reason this script
exists rather than a table of three headline percentages:

  rollup     124 records   the generated summary, always fits
  stripped   124 records   prose + tool-call inputs, results dropped
  raw         89 records   the full transcript, minus 35 that overflow 1M tokens

Comparing 94.4% (stripped, n=124) against 95.5% (raw, n=89) is comparing two
different question sets, and the 35 missing records are not missing at random --
they are the largest sessions, which are the hardest ones. So every comparison
below is computed on an explicit intersection and the n is printed with it.

Two views are reported:

  pairwise    each arm pair on the records they share, with paired discordance
              (b = first arm right / second wrong, c = the reverse) and a
              two-sided exact binomial on the discordant pairs. This is the
              McNemar-style test the paper uses elsewhere.
  triple      all three arms on the 89 records all three cover, which is the
              only place a three-way statement is licensed.

Stratification by delegation is included because RQ3's sample oversamples
delegating sessions ~4x against the corpus, and delegating sessions are the ones
the rollup handles worst. The flag is built by globbing for subagent transcripts
on disk, NOT from the `has_subagents` column in rq1_corrected_per_session.csv --
that column is set from `isSidechain` and so is 0 for every parent row.

Caveat that travels with the strata: the glob sees subagent files present on
disk TODAY. A pruned subagent file makes a delegating session look solo, which
biases the strata by an unmeasured amount in an unknown direction.

Usage:
    rq3_three_way_paired.py
"""

import csv
import glob
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper-rollups"
PROJECTS = PROJECTS_DIR
REPORT = DATA / "rq3_three_way_paired.json"

# The majority-vote grade files, one per arm. Keys are the arm names used in the
# paper; the rollup arm's file is named "uniform" because it is the arm that
# established the uniform protocol the other two inherited.
ARM_FILES = {
    "rollup": "rq3_majority_uniform.jsonl",
    "stripped": "rq3_majority_stripped.jsonl",
    "raw": "rq3_majority_raw.jsonl",
}

# recoverable = the judge produced a usable answer. A partially-correct answer
# still carries the fact being asked about, so it counts; this matches the
# definition used for every RQ3 number in the paper.
RECOVERABLE = {"correct", "partially_correct"}
# wrong-of-attempted excludes `absent` from the denominator: a refusal is not a
# hallucination, and counting it as one would reward arms that answer less.
ATTEMPTED_EXCLUDES = {"absent"}

# Session slugs look like 2026-09-18_claude-code_230b823e_example-task...
# so the third underscore-delimited field is the first 8 hex of the session uuid.
SLUG_UUID = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z-]+_([0-9a-f]{8})_")


def load_arm(name: str) -> dict:
    """record_id -> row, for records this arm actually resolved to a grade."""
    rows = {}
    path = DATA / ARM_FILES[name]
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # An unresolved record is a 3-way judge tie that survived 5 draws.
            # It has no grade, so it cannot be scored in either direction and is
            # dropped rather than silently counted as a failure.
            if r.get("unresolved") or not r.get("grade"):
                continue
            rows[r["record_id"]] = r
    return rows


def wilson(k: int, n: int) -> tuple:
    """95% Wilson score interval. Used everywhere in the paper instead of the
    normal approximation, which goes out of [0,1] at these small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1))


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p on the discordant pairs under p=0.5.

    This is the exact form of McNemar's test, correct at the small discordant
    counts here where the chi-square approximation is not.
    """
    n = b + c
    if n == 0:
        return 1.0
    obs = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, obs + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def score(rows, ids) -> dict:
    """recoverable and wrong-of-attempted over a fixed id set."""
    grades = [rows[i]["grade"] for i in ids if i in rows]
    n = len(grades)
    rec = sum(1 for g in grades if g in RECOVERABLE)
    att = [g for g in grades if g not in ATTEMPTED_EXCLUDES]
    wrong = sum(1 for g in att if g == "wrong")
    return {
        "n": n,
        "recoverable_n": rec,
        "recoverable_pct": round(100 * rec / n, 1) if n else None,
        "recoverable_ci95": wilson(rec, n),
        "wrong_n": wrong,
        "attempted_n": len(att),
        "wrong_of_attempted_pct": round(100 * wrong / len(att), 1) if att else None,
        "wrong_of_attempted_ci95": wilson(wrong, len(att)),
    }


def delegation_flags() -> dict:
    """uuid8 -> True if a subagent transcript for that session exists on disk.

    The real delegation signal. rq1's `has_subagents` column cannot be used: it
    is populated from `rec.get("isSidechain")`, which is only ever true inside a
    subagent file, so it duplicates `is_subagent` and is 0 for all parent rows.
    """
    flags = {}
    for p in glob.glob(str(PROJECTS / "*" / "*" / "subagents" / "agent-*.jsonl")):
        # .../projects/<project>/<session-uuid>/subagents/agent-<id>.jsonl
        session_uuid = Path(p).parent.parent.name
        flags[session_uuid[:8]] = True
    return flags


def uuid8(session_id: str):
    m = SLUG_UUID.match(session_id or "")
    return m.group(1) if m else None


def corpus_delegation_rate() -> dict:
    """What share of substantive parent sessions delegate, corpus-wide?

    RQ3's sample rate is compared against this. Substantive is defined the same
    way rq1 defines it for its own reporting, so the two are comparable.
    """
    csv_path = DATA / "rq1_corrected_per_session.csv"
    if not csv_path.exists():
        return {}
    flags = delegation_flags()
    n = deleg = 0
    sizes = {True: [], False: []}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("is_subagent") not in ("0", "", None):
                continue
            sid = r.get("session_id") or ""
            d = sid[:8] in flags
            n += 1
            deleg += 1 if d else 0
            try:
                sizes[d].append(float(r.get("serialized_bytes") or 0))
            except ValueError:
                pass
    med = {k: (statistics.median(v) if v else None) for k, v in sizes.items()}
    return {
        "n_parent_sessions": n,
        "n_delegating": deleg,
        "pct_delegating": round(100 * deleg / n, 1) if n else None,
        "median_bytes_delegating": med[True],
        "median_bytes_solo": med[False],
    }


def main() -> None:
    arms = {name: load_arm(name) for name in ARM_FILES}
    print("records with a resolved grade, per arm:")
    for name, rows in arms.items():
        print(f"  {name:10s} n={len(rows)}")

    flags = delegation_flags()
    print(f"\nsessions with a subagent transcript on disk: {len(flags)}")

    out = {"arm_sizes": {k: len(v) for k, v in arms.items()},
           "corpus_delegation": corpus_delegation_rate()}

    cd = out["corpus_delegation"]
    if cd:
        print(f"corpus: {cd['n_delegating']}/{cd['n_parent_sessions']} parent "
              f"sessions delegate ({cd['pct_delegating']}%)")
        if cd["median_bytes_delegating"] and cd["median_bytes_solo"]:
            print(f"  median size delegating {cd['median_bytes_delegating']:,.0f} B "
                  f"vs solo {cd['median_bytes_solo']:,.0f} B "
                  f"({cd['median_bytes_delegating']/cd['median_bytes_solo']:.1f}x)")

    # ---- each arm on its own full record set, for the headline numbers -------
    out["per_arm_full"] = {}
    print("\nper arm, on all records that arm covers (NOT comparable across arms):")
    print(f"  {'arm':10s} {'n':>4s} {'recoverable':>14s} {'wrong-of-att':>14s}")
    for name, rows in arms.items():
        s = score(rows, list(rows))
        out["per_arm_full"][name] = s
        print(f"  {name:10s} {s['n']:4d} {s['recoverable_pct']:11.1f}%  "
              f"{s['wrong_of_attempted_pct']:12.1f}%")

    # ---- pairwise, on shared records only -----------------------------------
    out["pairwise"] = {}
    print("\npairwise, on the records both arms cover:")
    hdr = (f"  {'pair':22s} {'n':>4s} {'A rec':>8s} {'B rec':>8s} {'gap':>8s} "
           f"{'b':>4s} {'c':>4s} {'p':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for a, b in (("rollup", "stripped"), ("rollup", "raw"), ("stripped", "raw")):
        shared = sorted(set(arms[a]) & set(arms[b]))
        sa, sb = score(arms[a], shared), score(arms[b], shared)
        # discordance: nb = a recoverable and b not, nc = the reverse
        nb = sum(1 for i in shared
                 if arms[a][i]["grade"] in RECOVERABLE
                 and arms[b][i]["grade"] not in RECOVERABLE)
        nc = sum(1 for i in shared
                 if arms[b][i]["grade"] in RECOVERABLE
                 and arms[a][i]["grade"] not in RECOVERABLE)
        p = binom_two_sided(nb, nc)
        key = f"{a}_vs_{b}"
        out["pairwise"][key] = {
            "n_shared": len(shared), a: sa, b: sb,
            "gap_pp": round(sb["recoverable_pct"] - sa["recoverable_pct"], 1),
            "discordant_a_only": nb, "discordant_b_only": nc,
            "exact_binomial_p": round(p, 6),
        }
        print(f"  {key:22s} {len(shared):4d} {sa['recoverable_pct']:7.1f}% "
              f"{sb['recoverable_pct']:7.1f}% "
              f"{sb['recoverable_pct']-sa['recoverable_pct']:+7.1f}pp "
              f"{nb:4d} {nc:4d} {p:8.4f}")

    # ---- all three, on the records all three cover --------------------------
    triple = sorted(set(arms["rollup"]) & set(arms["stripped"]) & set(arms["raw"]))
    out["triple"] = {"n_shared": len(triple),
                     "arms": {n: score(arms[n], triple) for n in ARM_FILES}}
    print(f"\nall three arms, on the {len(triple)} records all three cover:")
    for name in ("rollup", "stripped", "raw"):
        s = out["triple"]["arms"][name]
        print(f"  {name:10s} recoverable {s['recoverable_pct']:5.1f}% "
              f"{s['recoverable_ci95']}  wrong-of-attempted "
              f"{s['wrong_of_attempted_pct']:5.1f}%")

    # ---- per class, on the triple intersection ------------------------------
    by_class = defaultdict(list)
    for i in triple:
        by_class[arms["rollup"][i].get("class") or "unknown"].append(i)
    out["triple_by_class"] = {}
    print(f"\nper class, on those {len(triple)} records (recoverable %):")
    print(f"  {'class':18s} {'n':>4s} {'rollup':>8s} {'stripped':>9s} {'raw':>8s}")
    for cls in sorted(by_class):
        ids = by_class[cls]
        row = {n: score(arms[n], ids) for n in ARM_FILES}
        out["triple_by_class"][cls] = {"n": len(ids),
                                       **{n: row[n]["recoverable_pct"] for n in ARM_FILES}}
        print(f"  {cls:18s} {len(ids):4d} {row['rollup']['recoverable_pct']:7.1f}% "
              f"{row['stripped']['recoverable_pct']:8.1f}% "
              f"{row['raw']['recoverable_pct']:7.1f}%")

    # ---- stratified by delegation ------------------------------------------
    out["by_delegation"] = {}
    print("\nstratified by whether the session delegated to a subagent:")
    # Each arm gets its own n. On the rollup+strip set the raw arm covers only a
    # subset, so a single shared n in this table would overstate what raw was
    # scored on -- the number that matters is printed next to each percentage.
    print(f"  {'stratum':12s} {'set':12s} {'rollup':>16s} "
          f"{'stripped':>16s} {'raw':>16s}")
    for label, want in (("delegating", True), ("solo", False)):
        for setname, ids_all in (("triple", triple),
                                 ("rollup+strip", sorted(set(arms["rollup"]) & set(arms["stripped"])))):
            ids = [i for i in ids_all
                   if (uuid8(arms["rollup"][i]["session_id"]) in flags) == want]
            if not ids:
                continue
            row = {n: score(arms[n], ids) for n in ARM_FILES}
            out["by_delegation"].setdefault(label, {})[setname] = {
                "n": len(ids),
                **{n: row[n] for n in ARM_FILES},
            }
            def cell(arm):
                s = row[arm]
                if s["recoverable_pct"] is None:
                    return f"{'--':>16s}"
                return f"{s['recoverable_pct']:7.1f}% (n={s['n']:3d})"

            print(f"  {label:12s} {setname:12s} "
                  f"{cell('rollup')} {cell('stripped')} {cell('raw')}")

    # ---- reweight the full-sample headlines to the corpus delegation rate ---
    # RQ3's sample is delegation-heavy. A corpus-rate estimate is the sample's
    # two strata recombined at the corpus weight, not a new measurement.
    if cd.get("pct_delegating") is not None:
        w = cd["pct_delegating"] / 100.0
        out["reweighted_to_corpus"] = {"corpus_pct_delegating": cd["pct_delegating"]}
        print(f"\nreweighted to the corpus delegation rate ({cd['pct_delegating']}% "
              f"delegating), full record set per arm:")
        for name, rows in arms.items():
            strata = {}
            for label, want in (("delegating", True), ("solo", False)):
                ids = [i for i in rows if (uuid8(rows[i]["session_id"]) in flags) == want]
                strata[label] = score(rows, ids) if ids else None
            d, s = strata["delegating"], strata["solo"]
            if not d or not s or d["recoverable_pct"] is None or s["recoverable_pct"] is None:
                continue
            rec = w * d["recoverable_pct"] + (1 - w) * s["recoverable_pct"]
            wr = None
            if d["wrong_of_attempted_pct"] is not None and s["wrong_of_attempted_pct"] is not None:
                wr = w * d["wrong_of_attempted_pct"] + (1 - w) * s["wrong_of_attempted_pct"]
            out["reweighted_to_corpus"][name] = {
                "sample_recoverable_pct": score(rows, list(rows))["recoverable_pct"],
                "reweighted_recoverable_pct": round(rec, 1),
                "sample_wrong_of_attempted_pct": score(rows, list(rows))["wrong_of_attempted_pct"],
                "reweighted_wrong_of_attempted_pct": round(wr, 1) if wr is not None else None,
                "strata": strata,
            }
            print(f"  {name:10s} recoverable {score(rows, list(rows))['recoverable_pct']:5.1f}%"
                  f" -> {rec:5.1f}%   wrong-of-attempted "
                  f"{score(rows, list(rows))['wrong_of_attempted_pct']:5.1f}%"
                  + (f" -> {wr:5.1f}%" if wr is not None else ""))
        print("  (a stratified recombination of this sample, not a new measurement;"
              "\n   the strata are small, so treat these as indicative)")

    REPORT.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
