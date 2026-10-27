#!/usr/bin/env python3
"""Recompute every per-arm rate the CAIN paper reports, from data/paper-rollups/.

Written because check_paper_invariants.py could not read its own source data.
It imported only re/subprocess/sys/pathlib, its only open() was the .tex given
as its argument, and all seven of its references to .json files were comments.
So "61 passed, 0 failed" meant main.tex still contains the constants the gate
hardcodes -- a figure that had drifted from the data behind it passed, which is
how a half-finished edit once passed 70/0. frontier_bands.verify_tex() was the
one honourable exception, recomputing the band boundaries from the table in the
.tex instead of pinning them; this module is the same idea carried back one
step, to the records the table is built from.

Two rules this module exists to enforce on itself:

1. **Never read a published rate.** Every arm block in data/paper-rollups/
   carries `pct` and `ci95` beside its counts. Reading those would make the
   gate agree with the summary writer rather than with the records, so this
   module reads ONLY integer counts and recomputes every percentage and every
   Wilson interval from them. A rate and its n can then not come apart without
   the gate noticing, which a pinned value can never catch.

2. **Never guess the grade vocabulary.** The per-record files spell the four
   grades `correct` / `partially_correct` / `wrong` / `absent`. An earlier
   recompute over this same corpus assumed `partial`, counted zero of them, and
   silently produced wrong rates that looked plausible. `_rate()` therefore
   raises on any label outside GRADES instead of defaulting it to zero, so a
   renamed grade is a crash and not a quiet wrong answer.

Definitions, fixed once here because the paper uses both and they differ:
  recoverable      = correct + partially_correct, over n
  wrong-of-attempted = wrong / (correct + partially_correct + wrong)
Absent answers are excluded from the wrong-of-attempted denominator and are not
excluded from n.

Run it directly to print the whole recomputed table:

    python3 arm_rates.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Dict, Optional, Tuple

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"

# Spelled exactly as the per-record files spell them. See rule 2 above.
GRADES = ("correct", "partially_correct", "wrong", "absent")

# Two-sided 95%. This exact value reproduces every ci95 printed in
# data/paper-rollups/ to the tenth of a point, which is the control that says
# the recomputation below is the same statistic the summaries recorded.
Z = 1.959963984540054


def r1(x: float) -> float:
    """Round to one decimal, half away from zero, like the paper prints.

    Not round(): round() is half-to-even, so a value landing exactly on .X5
    would round in a direction the paper's own formatting does not.
    """
    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"), ROUND_HALF_UP))


def wilson(k: int, n: int) -> Tuple[float, float]:
    """Wilson score interval, as one decimal, clamped into [0, 100]."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + Z * Z / n
    centre = p + Z * Z / (2.0 * n)
    half = Z * math.sqrt(p * (1.0 - p) / n + Z * Z / (4.0 * n * n))
    lo = max(0.0, (centre - half) / denom) * 100.0
    hi = min(1.0, (centre + half) / denom) * 100.0
    return (r1(lo), r1(hi))


@dataclass(frozen=True)
class Rate:
    """One arm's counts, with every rate derived rather than stored."""
    label: str
    source: str
    n_correct: int
    n_partially_correct: int
    n_wrong: int
    n_absent: int

    @property
    def n(self) -> int:
        return (self.n_correct + self.n_partially_correct
                + self.n_wrong + self.n_absent)

    @property
    def recoverable_n(self) -> int:
        return self.n_correct + self.n_partially_correct

    @property
    def recall(self) -> float:
        return r1(100.0 * self.recoverable_n / self.n)

    @property
    def recall_ci(self) -> Tuple[float, float]:
        return wilson(self.recoverable_n, self.n)

    @property
    def attempted_n(self) -> int:
        return self.recoverable_n + self.n_wrong

    @property
    def wrong_rate(self) -> float:
        if self.attempted_n == 0:
            return 0.0
        return r1(100.0 * self.n_wrong / self.attempted_n)

    @property
    def wrong_ci(self) -> Tuple[float, float]:
        return wilson(self.n_wrong, self.attempted_n)

    def line(self) -> str:
        lo, hi = self.recall_ci
        wlo, whi = self.wrong_ci
        return (f"{self.label:28s} n={self.n:3d}  recoverable {self.recoverable_n:3d}"
                f" = {self.recall:5.1f}% [{lo}, {hi}]   wrong {self.n_wrong:2d}"
                f"/{self.attempted_n:3d} = {self.wrong_rate:4.1f}% [{wlo}, {whi}]"
                f"   absent {self.n_absent:3d}   <- {self.source}")


def _rate(label: str, source: str, counts: Dict[str, int]) -> Rate:
    stray = set(counts) - set(GRADES)
    if stray:
        raise ValueError(
            f"{source}: grade label(s) {sorted(stray)} are not in {GRADES}. "
            f"The four grades are correct/partially_correct/wrong/absent; a "
            f"recompute that silently defaults an unknown label to zero "
            f"produces plausible wrong rates.")
    return Rate(label, source,
                counts.get("correct", 0), counts.get("partially_correct", 0),
                counts.get("wrong", 0), counts.get("absent", 0))


@lru_cache(maxsize=None)
def _records(name: str) -> str:
    return (DATA / name).read_text()


def counts_from_records(name: str, budget: Optional[int] = None,
                        field: str = "grade", arm: Optional[str] = None,
                        record_ids: Optional[set] = None) -> Dict[str, int]:
    """Count the `grade` field over a per-record .jsonl, one record per line.

    This is the strongest source available: it counts the graded records
    themselves, so no summary writer sits between the gate and the data.
    """
    c: Counter = Counter()
    for line in _records(name).splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if budget is not None and rec.get("budget") != budget:
            continue
        if arm is not None and rec.get("arm") != arm:
            continue
        if record_ids is not None and rec["record_id"] not in record_ids:
            continue
        c[rec[field]] += 1
    if not c:
        raise ValueError(f"{name}: no records matched (budget={budget})")
    return dict(c)


def counts_from_summary(name: str, *path: str) -> Dict[str, int]:
    """Read the four integer counts out of one arm block of a summary .json.

    Only `n` per grade is read. The block's own `pct` and `ci95` are ignored on
    purpose -- they are what the gate is checking the paper against, so taking
    them here would make the check circular. The block's own `n` is read back
    only as a consistency assertion on the four counts.
    """
    block = json.loads((DATA / name).read_text())
    for key in path:
        block = block[key]
    counts = {g: int(block[g]["n"]) for g in GRADES}
    if "n" in block and int(block["n"]) != sum(counts.values()):
        raise ValueError(f"{name}:{'/'.join(path)}: the four grade counts sum to "
                         f"{sum(counts.values())}, but the block says n="
                         f"{block['n']}")
    return counts


# ---------------------------------------------------------------------------
# The registries. Every arm the paper names, and where its counts come from.
#
# One grading stage. Since 2026-10-09 every per-record `grade` in these files is
# the 3-draw majority of the revised grader (grading_common.py: first-word
# verdict parser, first-line refusal rule, grading failures retried rather than
# scored absent, the question in the prompt), written by
# data/paper-rollups/regrade-2026-10-09/regrade_s4.py apply. The grade the paper
# published before that is kept beside it as `grade_published`, and
# published() reads it, only so the before/after comparison can be recomputed.
# The old uncorrected / confidence-corrected split is gone: the refusal rule is
# applied to every record by the grader, so there is no second stage to report.
# ---------------------------------------------------------------------------

UNCORRECTED_SOURCES = {
    "zero context":    ("rq3_zero_context_arm.jsonl", None),
    "prompts-only":    ("rq3_prompts_only_arm.jsonl", None),
    "prose-2k":        ("rq3_prose_only_budget_ladder.jsonl", 2000),
    "rollup":          ("rq3_rollup_uniform.jsonl", None),
    "prose-8k":        ("rq3_prose_only_budget_ladder.jsonl", 8000),
    "answer-only":     ("rq3_answer_only_arm.jsonl", None),
    "prose-only":      ("rq3_prose_only_arm.jsonl", None),
    "dialogue record": ("rq3_verbatim_arm.jsonl", None),
    "call-only":       ("rq3_tool_inputs_only_arm.jsonl", None),
    "action record":   ("rq3_action_record_arm.jsonl", None),
    "output-stripped": ("rq3_questions_stripped_arm.jsonl", None),
    "raw":             ("rq3_questions_raw_arm.jsonl", None),
}

# Arms that appear only in fig:retention and must NOT join UNCORRECTED_SOURCES:
# check_substrate_table_recalls cross-checks the RQ4 frontier table's row count
# against len(UNCORRECTED_SOURCES), and that table deliberately does not carry
# this arm yet. A shared dict would silently couple the two tables. The same
# holds for the other arms the paper tabulates outside the frontier table.
RETENTION_ONLY_SOURCES = {
    "action record (no digest)": ("rq3_action_record_no_digest_arm.jsonl", None),
    "prose cut to rollup length": ("rq3_arm_prose_matched_per_record_graded.jsonl", None),
    "sonnet long":     ("rq3_sonnet_rollup_124_long_graded.jsonl", None),
    "sonnet short":    ("rq3_sonnet_rollup_124_short_graded.jsonl", None),
    "sonnet matched":  ("rq3_sonnet_rollup_124_matched_graded.jsonl", None),
    "prose-only with recap": ("rq3_prose_only_arm.with-continuation.jsonl", None),
}


@lru_cache(maxsize=None)
def uncorrected(label: str) -> Rate:
    """The revised-grader rate of an arm in UNCORRECTED_SOURCES (name kept so
    existing callers work; there is no corrected stage any more)."""
    name, budget = UNCORRECTED_SOURCES[label]
    src = name if budget is None else f"{name} budget={budget}"
    return _rate(label, src, counts_from_records(name, budget))


rate = uncorrected


@lru_cache(maxsize=None)
def uncorrected_retention_only(label: str) -> Rate:
    """Like uncorrected(), but reads RETENTION_ONLY_SOURCES, kept separate so
    these arms cannot inflate the RQ4 frontier table's expected row count."""
    name, budget = RETENTION_ONLY_SOURCES[label]
    src = name if budget is None else f"{name} budget={budget}"
    return _rate(label, src, counts_from_records(name, budget))


def any_rate(label: str) -> Rate:
    return (uncorrected(label) if label in UNCORRECTED_SOURCES
            else uncorrected_retention_only(label))


@lru_cache(maxsize=None)
def published(label: str) -> Rate:
    """The same arm under the grader the paper used before 2026-10-09 (the
    per-record `grade_published`). For the before/after table only."""
    name, budget = (UNCORRECTED_SOURCES.get(label) or RETENTION_ONLY_SOURCES[label])
    return _rate(label + " (published protocol)", name,
                 counts_from_records(name, budget, field="grade_published"))


@lru_cache(maxsize=None)
def read_budget(arm: str) -> Rate:
    """The capped-read-path re-run (89 questions), one arm of the file."""
    return _rate("read budget " + arm, "rq3_read_budget_arms.jsonl",
                 counts_from_records("rq3_read_budget_arms.jsonl", arm=arm))


# ---------------------------------------------------------------------------
# Figures that are not grade counts but are still read from the data rather
# than pinned.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def composition() -> Dict[str, float]:
    """Pooled payload shares over the 249 substantive sessions, as percentages.

    `tool_io` is derived, not stored: the abstract's 73.98% is tool results
    plus tool invocations, and a check that pinned it could not catch the two
    components moving in opposite directions by the same amount.
    """
    d = json.loads((DATA / "rq1_corrected_summary.json").read_text())
    p = d["shares_substantive"]["_pooled"]
    out = {k: v * 100.0 for k, v in p.items()}
    out["tool_io"] = (p["tool_result"] + p["tool_use"]) * 100.0
    out["n_substantive"] = d["n_substantive"]
    out["n_trivial_excluded"] = d["n_trivial_excluded"]
    return out


# Per-arm median context size, which the paper quotes in characters beside the
# recall it bought. The key differs per arm because each arm's builder named it
# after the thing it was measuring.
_MEDIAN_CHARS = {
    "call-only":       ("rq3_tool_inputs_only_arm_summary.json", "median_context_chars"),
    "dialogue record": ("rq3_verbatim_arm_summary.json", "median_verbatim_chars"),
    "prose-only":      ("rq3_prose_only_arm_summary.json", "median_prose_chars"),
    "output-stripped": ("rq3_stripped_arm_summary.json", "median_stripped_chars"),
}


@lru_cache(maxsize=None)
def median_context_chars(label: str) -> int:
    name, key = _MEDIAN_CHARS[label]
    return int(json.loads((DATA / name).read_text())[key])


@lru_cache(maxsize=None)
def raw_arm_coverage() -> Dict[str, object]:
    """How much of the 124-question set the raw arm could be judged on.

    The denominator is the full question set as the other arms see it, read
    from an arm that runs on all of them, so the 71.8% coverage figure and the
    `n = 124` it is a fraction of cannot drift apart.
    """
    full = uncorrected("output-stripped").n
    kept = uncorrected("raw").n
    d = json.loads((DATA / "rq3_raw_arm_summary.json").read_text())
    return {
        "full": full,
        "kept": kept,
        "dropped": full - kept,
        "coverage_pct": r1(100.0 * kept / full),
        "n_over_context_sessions": len(d["dropped_over_context_sessions"]),
    }


def main() -> int:
    print("=== fig:retention arms, as the figure reports them ===")
    for label in ("call-only", "rollup", "dialogue record", "prose-only",
                  "output-stripped", "raw", "action record"):
        print("  " + uncorrected(label).line())
    print("\n=== substrate table, single-stage uncorrected ===")
    for label in UNCORRECTED_SOURCES:
        print("  " + uncorrected(label).line())
    print("\n=== composition, pooled over substantive sessions ===")
    c = composition()
    print(f"  tool I/O {c['tool_io']:.2f}%  (results {c['tool_result']:.2f}%"
          f" + invocations {c['tool_use']:.2f}%), assistant prose"
          f" {c['assistant_text']:.2f}%, reasoning {c['thinking']:.2f}%,"
          f" n={c['n_substantive']} substantive, {c['n_trivial_excluded']} excluded")
    print("\n=== raw arm coverage ===")
    print("  " + json.dumps(raw_arm_coverage()))
    print("\n=== median context characters ===")
    for label in _MEDIAN_CHARS:
        print(f"  {label:18s} {median_context_chars(label):,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
