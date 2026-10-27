#!/usr/bin/env python3
"""Compute the F-banded substrate frontier from the Table substrate figures.

The paper's band sentence (Section RQ3) names which substrate wins at which
failure cost F. Those boundaries were hardcoded literals until now. They are
fully determined by the substrate table: a substrate is on the frontier if it
sits on the upper convex hull of (token cost, recall), and the boundary between
two consecutive hull points is

    F* = (cost_b - cost_a) / (recall_b - recall_a)

i.e. the failure cost at which the extra recall first pays for the extra tokens.

Run with --published to reproduce the published bands from the published
numbers; that is the control, and it must print the sentence currently in
main.tex. Run with --filtered to recompute after the continuation-summary
filter, reading the two re-run summaries from data/paper-rollups/.

Usage:
    frontier_bands.py --published
    frontier_bands.py --filtered
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import List, Tuple

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"

# The paper's prose Tokens column is the MEASURED cl100k token count (median over
# the 124 frozen records) of the artifact each arm delivers: prose-only 3,055
# (12,857 chars), prose @2k 478 (1,994 chars), prose @8k 1,861 (7,986 chars);
# answer-only 2,533 and prompts-only 202 likewise. See
# jobs/<id>/tmp/recompute/measured_costs.py. The constant below is
# only the prose-only chars/token ratio, used by --filtered as an approximation
# when re-run contexts have no token counts.
PROSE_CHARS_PER_TOKEN = 12857 / 3055

# (name, tokens, recall) exactly as printed in main.tex's substrate table, for the
# rows the F bands price: every rule cut plus the production rollup.
#
# Updated 2026-10-14 (section-by-section review): tab:frontier absorbed tab:length,
# the no-digest action record became a priced row (5,506 tokens, 93.5% on the
# design set, designed with those questions visible), and raw moved to a separate
# "not comparable (89 questions)" block, so it is no longer priced. Raw was
# dominated anyway (94.4% at 137,230 tokens, below the action record's 95.2%).
PUBLISHED: List[Tuple[str, int, float]] = [
    ("zero", 0, 0.000),
    ("prompts-only", 202, 0.315),
    ("prose-2k", 478, 0.379),
    ("rollup", 929, 0.556),
    ("prose-8k", 1861, 0.766),
    ("answer-only", 2533, 0.847),
    ("prose-only", 3055, 0.887),
    ("dialogue record", 3928, 0.742),
    ("no-digest", 5506, 0.935),
    ("call-only", 6329, 0.463),
    ("action record", 6393, 0.952),
    ("output-stripped", 8880, 0.903),
]

# The published band sentence, for the control to compare against.
#
# Updated 2026-10-09 to the regraded recalls of tab:frontier (fixed grader).
#
# Updated 2026-10-02, when the substring-guard fix in rq3_answer_only_arm.py and
# rq3_prompts_only_arm.py un-voided 34 graded records. Recall moved (prompts-only
# 27.4 -> 32.3, answer-only 66.1 -> 85.5) while both Tokens cells held, because the
# 124 contexts per arm are byte-identical before and after -- only the grades
# changed. At 85.5% answer-only stops being dominated and takes a band of its own,
# so seven configurations win where six did.
#
# Updated 2026-10-14: with the no-digest row priced and raw unpriced, eight win.
# main.tex no longer enumerates every band (the review dropped the enumeration);
# it states the zero, rollup and top two bands, which verify_tex() checks against
# this computation.
PUBLISHED_BANDS_SENTENCE = (
    "zero (F < 641), prompts-only (641-3017), rollup (3017-4438), "
    "prose-8k (4438-8296), answer-only (8296-13050), "
    "prose-only (13050-51062), no-digest (51062-52176), action record (F >= 52176)"
)


def upper_hull(points: List[Tuple[str, int, float]]) -> List[Tuple[str, int, float]]:
    """Upper-left convex hull: best recall per token, with diminishing returns.

    Walks substrates cheapest-first and keeps only those whose marginal recall
    per token is strictly less than the step before it. A substrate that cannot
    beat the straight line between its neighbours is dominated.
    """
    pts = sorted(points, key=lambda p: (p[1], -p[2]))
    hull: List[Tuple[str, int, float]] = []
    for p in pts:
        # Never keep a substrate that costs more and recovers no more.
        while hull and p[2] <= hull[-1][2]:
            if p[1] >= hull[-1][1]:
                p = None
                break
            hull.pop()
        if p is None:
            continue
        # Drop any point the new one makes non-concave.
        while len(hull) >= 2:
            (_, c1, r1), (_, c2, r2) = hull[-2], hull[-1]
            s_prev = (r2 - r1) / (c2 - c1)
            s_new = (p[2] - r2) / (p[1] - c2)
            if s_new >= s_prev:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def bands(hull: List[Tuple[str, int, float]]) -> List[Tuple[str, float, float]]:
    """Return (name, F_lo, F_hi) per hull substrate; inf for the last."""
    edges = []
    for a, b in zip(hull, hull[1:]):
        edges.append((b[1] - a[1]) / (b[2] - a[2]))
    out = []
    for i, (name, _, _) in enumerate(hull):
        lo = edges[i - 1] if i > 0 else 0.0
        hi = edges[i] if i < len(edges) else float("inf")
        out.append((name, lo, hi))
    return out


def sentence(rows: List[Tuple[str, float, float]]) -> str:
    parts = []
    for i, (name, lo, hi) in enumerate(rows):
        if i == 0:
            parts.append(f"{name} (F < {hi:.0f})")
        elif hi == float("inf"):
            parts.append(f"{name} (F >= {lo:.0f})")
        else:
            parts.append(f"{name} ({lo:.0f}-{hi:.0f})")
    return ", ".join(parts)


def load_filtered() -> List[Tuple[str, int, float]]:
    """Published table with the three prose arms replaced by re-run figures."""
    arm = json.loads((DATA / "rq3_prose_only_arm_summary.json").read_text())
    lad = json.loads((DATA / "rq3_prose_only_budget_ladder_summary.json").read_text())

    prose_only_recall = arm["prose_only_arm_uncorrected"]["recoverable"]["pct"] / 100
    prose_only_chars = arm["median_prose_chars"]
    prose_only_tokens = round(prose_only_chars / PROSE_CHARS_PER_TOKEN)

    recalls = {int(b): d["uncorrected"]["recoverable"]["pct"] / 100
               for b, d in lad["by_budget"].items()}
    # A budget arm's cost is NOT the budget: sessions whose whole prose is
    # shorter than the budget contribute their full length, so the median
    # context sits just under it. Deriving the cost from the records instead
    # of from the budget reproduces the published 478 and 1914 exactly.
    ctx = defaultdict(list)
    for line in (DATA / "rq3_prose_only_budget_ladder.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ctx[r["budget"]].append(r["context_chars"])
    costs = {b: round(median(v) / PROSE_CHARS_PER_TOKEN) for b, v in ctx.items()}

    out = []
    for name, tok, rec in PUBLISHED:
        if name == "prose-only":
            out.append((name, prose_only_tokens, prose_only_recall))
        elif name == "prose-2k":
            out.append((name, costs[2000], recalls[2000]))
        elif name == "prose-8k":
            out.append((name, costs[8000], recalls[8000]))
        else:
            out.append((name, tok, rec))
    return out


# Table substrate labels (after stripping $^{...}$ markers) vs the names used here.
TEX_ALIASES = {
    "zero context": "zero",
    "prose @2k": "prose-2k",
    "prose @8k": "prose-8k",
    "production rollup": "rollup",
    "no-digest action record": "no-digest",
    "prose, first 1{,}500 chars": "prose-1500",
    "prose cut to rollup length": "prose-cut",
}

# One row of the merged tab:frontier (which absorbed tab:length on 2026-10-14):
# Substrate & Generator & n & Tokens & Recall & Wrong
TEX_ROW = re.compile(
    r"^([^&\n]+?) & (none|Haiku|Sonnet) & (\d+) & ([\d{},]+|--) & ([\d.]+) & ([\d.]+|--) \\\\$",
    re.M)

# The band sentence that replaced the seven-band enumeration. Since the second
# section-by-section pass (2026-10-14) it states the bands in words, with no
# boundary values: zero context first, the cheapest substrate changing with F, the
# rollup's single narrow band, rule cuts everywhere else, and the two action
# records on top. Every one of those words is still recomputed in verify_tex().
# Re-anchored 2026-10-14 (round 5): "cheapest substrate" -> "cheapest artifact",
# the paper's one word for the compared things; the claim is unchanged.
TEX_SENTENCE = re.compile(
    r"Zero context is cheapest at the lowest \$F\$, and the cheapest artifact changes as \$F\$ grows\.\s+"
    r"The rollup wins a single narrow band, on size alone.*?Rule cuts win every other band\.\s+"
    r"The top two belong to the two action records, the no-digest record and the full record,"
    r" and both rest on design-set recall",
    re.S)

WORD_NUM = {"Five": 5, "Six": 6, "Seven": 7, "Eight": 8,
            "five": 5, "six": 6, "seven": 7, "eight": 8}


def _num(s: str) -> int:
    return int(s.replace("{,}", "").replace(",", ""))


def clean_name(raw: str) -> str:
    """'action record$^{\\dagger}$' -> 'action record'."""
    return re.sub(r"\$\^\{[^}]*\}\$", "", raw).strip()


def table_blocks(tex: str):
    """The merged tab:frontier split into its blocks at each \\multicolumn.

    It held four blocks (comparison, raw, capped, strict) until cain-review-1014
    round 7, when the human cut the caption to one sentence and the zero-context,
    raw, capped and strict rows left the table for the prose; since then it is one
    comparison block. Zero context is still priced, from its prose sentence (see
    zero_point()).

    Each block is a list of (name, generator, n, tokens|None, recall, wrong|None,
    marked) where `marked` is True for a daggered (designed-with-questions-visible)
    row. Returns None if the table cannot be found.
    """
    m = re.search(r"\\label\{tab:frontier\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        return None
    parts = re.split(r"\\multicolumn", m[1])
    out = []
    for part in parts:
        rows = []
        for r in TEX_ROW.finditer(part):
            rows.append((TEX_ALIASES.get(clean_name(r[1]), clean_name(r[1])), r[2], int(r[3]),
                         None if r[4] == "--" else _num(r[4]), float(r[5]),
                         None if r[6] == "--" else float(r[6]), "dagger" in r[1]))
        out.append(rows)
    return out


# Zero context left tab:frontier in round 7 (cain-review-1014); its read cost and
# recall now live in one method-section sentence, which is what gets priced.
ZERO_PROSE = re.compile(r"Zero\s+context\s+reads\s+(\d+)\s+tokens\s+and\s+recovers\s+([\d.]+)\\%")


def zero_point(tex: str):
    """('zero', tokens, recall) from the prose sentence, or None if it is missing."""
    m = ZERO_PROSE.search(tex)
    return None if not m else ("zero", int(m[1]), float(m[2]) / 100)


def priced_points(tex: str):
    """(name, tokens, recall) for every row the F bands price: the comparison
    block's rule cuts (generator none), the production rollup, and zero context,
    which is read from its prose sentence when the table no longer carries it."""
    blocks = table_blocks(tex)
    if not blocks:
        return None, None
    comp = blocks[0]
    pts = [(n, tok, rec / 100) for n, gen, _nn, tok, rec, _w, _d in comp
           if gen == "none" or n == "rollup"]
    zp = zero_point(tex)
    if zp and "zero" not in {q[0] for q in pts}:
        pts.insert(0, zp)
        comp = [("zero", "none", 124, zp[1], zp[2] * 100, None, False)] + comp
    return pts, comp


def verify_tex(tex: str) -> Tuple[bool, str]:
    """Recompute the frontier from main.tex's own merged table and compare it to
    main.tex's own band sentence. Catches a half-edit, where a table cell and
    the band sentence are each individually plausible but disagree.

    Re-anchored 2026-10-14: the table is the merged tab:frontier (generator, n,
    tokens, recall, wrong), raw sits in a separate not-comparable block and is not
    priced, and the seven-band enumeration was replaced by a sentence naming the
    zero band, the rollup's single band, the rule-cut claim and the top two bands.
    Every one of those is recomputed here; the middle bands are no longer stated.

    Returns (ok, message). Used both by --tex here and as a custom check in
    check_paper_invariants.py.
    """
    problems: List[str] = []
    pts, comp = priced_points(tex)
    if pts is None:
        return False, "merged substrate table tab:frontier not found"
    want = {p[0] for p in PUBLISHED}
    got = {p[0] for p in pts}
    if not want <= got:
        return False, f"priced rows missing from the table: {sorted(want - got)}"
    generated = [n for n, gen, *_ in comp if gen != "none"]
    if "rollup" not in [n for n, gen, *_ in comp if gen == "Haiku"]:
        problems.append("the production rollup is not marked as Haiku-generated")

    sm = TEX_SENTENCE.search(tex)
    if not sm:
        return False, "could not find the band sentence"

    hull = upper_hull(pts)
    rows = bands(hull)
    names = [h[0] for h in hull]
    gen_of = {n: gen for n, gen, *_ in comp}
    dag = {n: d for n, _g, _n, _t, _r, _w, d in comp}

    # The boundary values are no longer stated (retired 2026-10-14, see
    # check_paper_invariants.OBSOLETE_PINS); the claims in words are checked instead.
    if rows[0][0] != "zero":
        problems.append(f"zero context is not cheapest at the lowest F: the table gives {rows[0]}")
    if len(rows) < 3:
        problems.append("the cheapest substrate does not change with F")
    rb = [r for r in rows if r[0] == "rollup"]
    finite = sorted(hi - lo for _n, lo, hi in rows if hi != float("inf"))
    if len(rb) != 1:
        problems.append("the rollup does not win exactly one band")
    elif not (rb[0][2] - rb[0][1]) < finite[len(finite) // 2]:
        problems.append(f"the rollup's band ({rb[0][1]:.0f}--{rb[0][2]:.0f}) is not narrow: "
                        f"its width is not below the median finite band width {finite[len(finite) // 2]:.0f}")
    others = [n for n in names if n not in ("zero", "rollup")]
    not_cut = [n for n in others if gen_of.get(n) != "none"]
    if not_cut:
        problems.append(f"bands won by something other than a rule cut: {not_cut}")
    if names[-2:] != ["no-digest", "action record"]:
        problems.append(f"the top two bands are {names[-2:]}, not the two action records")
    else:
        if not (dag.get("no-digest") and dag.get("action record")):
            problems.append("the two action-record rows are not daggered as designed with the questions visible")

    # A spelled-out band count anywhere in the prose must equal the number of winners.
    for rx in (r"band of (\w+)", r"of\s+(\w+) operating bands",
               r"(\w+) operating bands", r"(\w+) bands:"):
        for m in re.finditer(rx, tex, re.S):
            claimed = WORD_NUM.get(m[1])
            if claimed is not None and claimed != len(hull):
                problems.append(f"prose says {m[1]} bands in \"{m[0].strip()}\"; "
                                f"the table's own numbers give {len(hull)}")

    if problems:
        return False, "; ".join(problems)
    return True, (f"{len(hull)} of {len(pts)} priced rows win a band; zero first, the rollup's single "
                  f"narrow band, the rule-cut claim and the two action records on top reproduce from the table")


def check_tex(path: Path) -> int:
    ok, msg = verify_tex(path.read_text())
    print(f"  TEX {'PASS' if ok else 'FAIL'}: {path.name}: {msg}")
    return 0 if ok else 1


def report(label: str, pts: List[Tuple[str, int, float]]) -> List[Tuple[str, float, float]]:
    print(f"=== {label} ===")
    for name, tok, rec in sorted(pts, key=lambda p: p[1]):
        print(f"  {name:18s} {tok:7,d} tokens  {rec*100:5.1f}%")
    hull = upper_hull(pts)
    rows = bands(hull)
    dominated = [p[0] for p in pts if p[0] not in {h[0] for h in hull}]
    print(f"\n  on frontier ({len(hull)}): {', '.join(h[0] for h in hull)}")
    print(f"  dominated ({len(dominated)}): {', '.join(dominated)}")
    print(f"\n  {sentence(rows)}\n")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", action="store_true")
    ap.add_argument("--filtered", action="store_true")
    ap.add_argument("--tex", metavar="PATH",
                    help="verify a main.tex against its own substrate table")
    a = ap.parse_args()

    rc = 0
    if a.tex:
        rc |= check_tex(Path(a.tex))
        if not (a.published or a.filtered):
            return rc
    if a.published or not a.filtered:
        rows = report("PUBLISHED (control)", PUBLISHED)
        got = sentence(rows)
        if got == PUBLISHED_BANDS_SENTENCE:
            print("  CONTROL PASS: reproduces the band sentence in main.tex\n")
        else:
            print("  CONTROL FAIL: does not reproduce main.tex")
            print(f"    expected: {PUBLISHED_BANDS_SENTENCE}")
            print(f"    got:      {got}\n")
            rc = 1
    if a.filtered:
        report("FILTERED (continuation summaries removed)", load_filtered())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
