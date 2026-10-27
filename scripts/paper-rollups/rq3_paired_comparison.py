#!/usr/bin/env python3
"""The RQ3 paired comparison: same questions, same judge, same prompts, only the
context differs.

Three arms are on disk by the time this runs:

  published   rq3_questions.jsonl -- the paper's arm, pooled from up to three
              scripts with different answering prompts, different grading
              prompts, different grade parsers, and in one case a 15,000-char
              clip on the rollup.
  uniform     rq3_rollup_uniform.jsonl -- every question re-answered from the
              full untruncated rollup under ONE protocol.
  raw         rq3_raw_arm.jsonl -- every question answered from the full raw
              transcript, using the uniform arm's prompts verbatim apart from
              the two words naming the context.

Only `uniform` vs `raw` is a controlled contrast. `published` is reported beside
it to show whether the pooled protocols distorted the paper's numbers, and to
expose the selection effect, since the transcripts that overflow a 1M window are
the large ones and the rollup does worse on those.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, load_records, rates, record_id, wilson,
)

REPORT = DATA / "rq3_paired_comparison.json"

# Single-draw grades live in each arm's own answer file; majority-vote grades in
# the files written by rq3_majority_vote_grades.py. The answers are identical in
# both -- only the grading differs -- so this is a pure re-read, not a re-run.
SOURCES = {
    "single": {"uniform": "rq3_rollup_uniform.jsonl",
               "raw": "rq3_questions_raw_arm.jsonl"},
    "majority": {"uniform": "rq3_majority_uniform.jsonl",
                 "raw": "rq3_majority_raw.jsonl"},
}


def line(label: str, d: dict) -> str:
    h = d["hallucination_of_attempted"]
    return (f"{label:26s} n={d['n']:3d}  {d['correct']['n']:3d} {d['partially_correct']['n']:3d} "
            f"{d['wrong']['n']:3d} {d['absent']['n']:3d}   "
            f"{d['recoverable']['pct']:5.1f}% {str(d['recoverable']['ci95']):14s} "
            f"{h['pct']}% {h['ci95']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", choices=sorted(SOURCES), default="majority",
                    help="which grading pass to read (default: majority vote)")
    args = ap.parse_args()
    src = SOURCES[args.grades]

    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    published = {record_id(r): r for r in load_records(QUESTIONS)
                 if record_id(r) in graded}
    uniform = {r["record_id"]: r for r in load_records(DATA / src["uniform"])}
    raw = {r["record_id"]: r for r in load_records(DATA / src["raw"])}

    # A majority-vote record with no plurality after 5 draws has grade None. It is
    # dropped from the paired set rather than counted as a failure of either arm,
    # which would invent a grade the grader declined to give.
    unresolved = sorted({i for d in (uniform, raw) for i, r in d.items()
                         if r.get("grade") is None})
    for d in (uniform, raw):
        for i in unresolved:
            d.pop(i, None)

    shared = sorted(set(uniform) & set(raw) & set(published))
    print(f"grades={args.grades}  published {len(published)}  "
          f"uniform {len(uniform)}  raw {len(raw)}  shared {len(shared)}"
          f"  (dropped unresolved: {len(unresolved)})")

    pub_shared = [published[i] for i in shared]
    uni_shared = [uniform[i] for i in shared]
    raw_shared = [raw[i] for i in shared]

    r_pub_all = rates(list(published.values()))
    r_uni_all = rates(list(uniform.values()))
    r_pub = rates(pub_shared)
    r_uni = rates(uni_shared)
    r_raw = rates(raw_shared)

    hdr = (f"{'arm':26s} {'n':>5s}  {'cor':>3s} {'par':>3s} {'wrg':>3s} {'abs':>3s}   "
           f"{'recoverable':>6s}                {'wrong-of-attempted'}")
    print("\n" + hdr)
    print("-" * len(hdr))
    print(line("published, all 124", r_pub_all))
    print(line("uniform rollup, all 124", r_uni_all))
    print("-" * len(hdr))
    print(line("published, shared", r_pub))
    print(line("uniform rollup, shared", r_uni))
    print(line("RAW TRANSCRIPT, shared", r_raw))

    # Per-class, on the controlled contrast only.
    print("\nper class, uniform rollup -> raw transcript (recoverable %):")
    classes = sorted({r.get("class") for r in uni_shared if r.get("class")})
    per_class = {}
    for c in classes:
        u = rates([r for r in uni_shared if r.get("class") == c])
        w = rates([r for r in raw_shared if r.get("class") == c])
        per_class[c] = {"n": u["n"], "uniform": u, "raw": w}
        print(f"  {c:16s} n={u['n']:3d}  {u['recoverable']['pct']:5.1f}% -> "
              f"{w['recoverable']['pct']:5.1f}%   wrong-of-att "
              f"{u['hallucination_of_attempted']['pct']}% -> "
              f"{w['hallucination_of_attempted']['pct']}%")

    # McNemar-style paired discordance on recoverable-vs-not, the paired test the
    # unpaired Wilson intervals above cannot provide.
    uni_ok = {i for i in shared
              if uniform[i]["grade"] in ("correct", "partially_correct")}
    raw_ok = {i for i in shared
              if raw[i]["grade"] in ("correct", "partially_correct")}
    b = len(raw_ok - uni_ok)   # raw recovered, rollup did not
    c = len(uni_ok - raw_ok)   # rollup recovered, raw did not
    both = len(raw_ok & uni_ok)
    neither = len(shared) - len(raw_ok | uni_ok)
    print(f"\npaired on 'recoverable' over {len(shared)} records:")
    print(f"  both {both}   raw only {b}   rollup only {c}   neither {neither}")
    if b + c:
        # Exact binomial two-sided on the discordant pairs.
        from math import comb
        n = b + c
        k = min(b, c)
        p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
        print(f"  discordant {n}, exact binomial two-sided p = {min(p,1.0):.2e}")
    else:
        p = 1.0

    report = {
        "note": ("only 'uniform rollup, shared' vs 'RAW TRANSCRIPT, shared' is a "
                 "controlled contrast: same questions, same ground truth, same "
                 "judge model, same answering and grading prompts, same grade "
                 "parser. Only the context differs."),
        "grades": args.grades,
        "selection_effect": (
            "the shared set excludes sessions whose transcript overflows a 1M "
            "window, which are the large ones, and the rollup does better on the "
            f"survivors: recoverable {r_uni_all['recoverable']['pct']}% over all "
            f"{r_uni_all['n']} against {r_uni['recoverable']['pct']}% over the "
            "shared set. The contrast is therefore measured in the regime that "
            "favours the rollup, which makes it a conservative estimate of the "
            "gap. The raw arm's own failures also concentrate in large "
            "transcripts, so its rate must not be extrapolated to the excluded "
            "sessions either."),
        "n_shared": len(shared),
        "n_unresolved_dropped": len(unresolved),
        "published_all_124": r_pub_all,
        "uniform_all_124": r_uni_all,
        "published_shared": r_pub,
        "uniform_shared": r_uni,
        "raw_shared": r_raw,
        "per_class": per_class,
        "paired_recoverable": {"both": both, "raw_only": b, "rollup_only": c,
                               "neither": neither,
                               "exact_binomial_two_sided_p": min(p, 1.0)},
    }
    out = REPORT if args.grades == "majority" else DATA / (
        REPORT.stem + f"_{args.grades}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
