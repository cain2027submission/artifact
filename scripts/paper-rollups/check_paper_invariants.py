#!/usr/bin/env python3
"""Gate paper/main.tex on the figures and audit trail that must survive an edit.

Written because five subagents are compressing RQ3/RQ4/RQ5/Threats/Method into
replacement fragments, and the measured pattern this session is that a delegated
writer picks a gate that passes rather than one that discriminates. Run this
yourself over the result; do not accept a report.

**What a pinned check can and cannot prove.** The MUST/MUST_NOT lists below are
regexes over the .tex against constants spelled out in this file, so a PASS from
them says only that main.tex still contains the string this file expects. It
says nothing about whether that string still matches `data/paper-rollups/`. For
most of this file's life that was the whole gate -- it imported only
re/subprocess/sys/pathlib, its single open() was the .tex path, and every one of
its references to a .json file was a comment -- so a figure that had drifted
from its source data passed, which is how a half-finished edit once passed
70/0. Two kinds of check are therefore not interchangeable, and the printed
label says which you are reading:

  PASS present / PASS absent   -- a pinned constant, checked against this file
  PASS custom                  -- recomputed from data/paper-rollups/

`frontier_bands.verify_tex()` was the first of the second kind, recomputing the
band boundaries from the substrate table in the .tex rather than pinning them.
`arm_rates` carries that one step further back, to the graded records the table
is built from, and the custom checks below read every per-arm recall, every
wrong-of-attempted rate, every Wilson interval and every sample size through
it. Prefer adding a check of the second kind; a new pinned constant is only
appropriate for something that is not in the data at all (a citation key, a
piece of required wording, a superseded value that must not return).

Usage:
  python check_paper_invariants.py <path-to-main.tex>
  python check_paper_invariants.py --origin      # reads git show origin/main:paper/main.tex
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier_bands import TEX_ALIASES, TEX_ROW, verify_tex  # noqa: E402
import frontier_bands as _fb  # noqa: E402
from paths import REPO_ROOT  # noqa: E402
import arm_rates  # noqa: E402
from arm_rates import (  # noqa: E402
    UNCORRECTED_SOURCES, any_rate, composition, median_context_chars,
    published, raw_arm_coverage, read_budget, uncorrected,
)
import rq3_renumber  # noqa: E402

VAULT = str(REPO_ROOT)
RA = " (re-anchored 2026-10-14, round 3 plain-language pass: points -> pp, arm -> condition, method names out of the prose)"
RA5 = " (re-anchored 2026-10-14, round 5: condition/substrate -> artifact, the paper's one word for the compared memory artifacts; value unchanged)"
RA4 = " (re-anchored 2026-10-14, round 4: tab:sens moved to supplementary Table S1 in paper/supplement/sensitivity-checks.tex; this grades the Threats sentence that now carries the fact in main.tex)"
R6 = " (round 6: the difference block moved to supplementary Table S4 in paper/supplement/sensitivity-checks.tex, in pp with one decimal; this grades that cell in the supplement)"
R7 = " (round 7: threats halved, fact moved to supplementary Table S1 in paper/supplement/sensitivity-checks.tex; this grades the cell that now carries it, same values)"
SUP = "supplement S1: "   # label prefix of an entry graded against paper/supplement/sensitivity-checks.tex

# (label, regex, min_count) -- must appear at least min_count times
MUST = [
    # corrected-extractor headline figures (rq3_confidence_corrected.json, corrected block)
    ("rollup recoverable 55.6%",        r"55\.6\\%", 1),
    ("rollup wrong-of-attempted 11.5%", r"11\.5\\%", 1),
    ("stripped recoverable 90.3%",      r"90\.3\\%", 1),
    ("stripped wrong 1.8%",             r"1\.8\\%", 1),
    ("raw recoverable 94.4%",           r"94\.4\\%", 1),
    # The 19.5% / 13.6% / 10.3% audit trail and its 46-absent / 78-attempted counts were
    # pinned here until 2026-10-09. They described the confidence correction, which the
    # revised grader replaced; the paragraph that held them now states the published ->
    # revised movement and is gated from the data by data_backed_prose().
    ("the word superseded",             r"supersede", 1),
    # prose rows are calibrated, not tokenised; the rate is recoverable from the
    # graders' own prompt_tokens even though the prose text itself was not kept.
    # The 4.4 chars/token calibration pin was removed 2026-10-08: since b9d707c1 the
    # prose rows are priced at measured cl100k medians of the delivered artifacts,
    # so the calibration sentence it guarded no longer exists to be guarded.
    # The RQ5 repriced total 114.48 and ledger total 104.03 pins were removed when RQ5
    # folded into RQ3 (sec:gencost): the paper keeps only the $0.578 per rollup.
    # RQ1: the bound that survives the dead parent-vs-subagent control. This is what
    # the trace arms rest on, and the docs make it unconditional -- "No `display`
    # setting returns the raw chain of thought", so even populated subagent blocks
    # hold summaries, not reasoning.
    ("RQ1 surviving bound: reasoning not recoverable",
     r"the record cannot\s+recover it", 1),
    # tab:retention became fig:retention on 2026-10-02. Everything that used to be
    # gated here by `&`-scoped regex over main.tex is now gated by
    # check_retention_figure() instead, which reads paper/fig/fig-retention.tex
    # from the same source as main.tex and pairs recall with wrong rate by ROW
    # INDEX -- the property the `&` stood in for. Eight values left main.tex
    # entirely in that move, because a Wilson interval that was a bracket in a
    # cell is a whisker coordinate in a figure: [63.3, 78.9], [4.9, 16.5],
    # [83.8, 94.4], [0.5, 6.2], [87.5, 97.6], [2.4, 10.7], [8.2, 25.0] and
    # 0.8% [0.1, 4.5]. Pinning those against main.tex would fail outright; leaving
    # the old `&` regexes in place would have passed vacuously forever, which is
    # worse, because an inert guard reports PASS.
    #
    # 71.8% is the trap in this move and the reason the recall pins went with them.
    # It still matches main.tex once -- but on "89 of 124 questions (71.8% of the
    # set)", the raw arm's COVERAGE, not the dialogue record's recall. The pin
    # would have stayed green while the figure's value was edited to anything at
    # all. Same collision class as the 14.7% below.
    ("rollup CI [46.9, 64.1] cited in prose", r"\[46\.9, 64\.1\]", 1),
    # verbatim arm figures from rq3_verbatim_arm_summary.json -- all four moved into
    # fig:retention and are gated by check_retention_figure(). Do not re-add them
    # here: 71.8% collides with the raw arm's coverage and the other three cannot
    # match main.tex at all. See the note on the retention pins above.
    # four-arm framing (verbatim was the fourth arm added)
    ("four-condition framing (re-anchored 2026-10-14 round 3: arm -> condition; round 5: condition -> artifact)", r"[Aa]ll four artifacts", 1),
    # corrected raw-arm exclusion split
    ("raw exclusion: 5 sessions exceeding context window", r"5 exceed[^.]{0,40}context window", 1),
    ("raw exclusion: 2 sessions failing API calls",        r"2 failed[^.]{0,40}API calls", 1),
    # raw artifact size
    ("raw artifact size 691 KB",         r"691\\,KB", 1),
    # 2026-10-09 swap: the two action-record text pins (the [89.8, 97.8] interval and "one
    # wrong answer in 119 attempted") are removed because the paragraph carrying them was cut.
    # The arm's rate is still graded by check_retention_figure() against its figure row;
    # the bans below on the unsourced intervals stay.
]

# (label, regex) -- must NOT appear anywhere
MUST_NOT = [
    # 95.5% is raw's UNCORRECTED rate (85/89) and 94.4% its confidence-corrected one
    # (84/89); one record carries a surviving refusal marker and separates them. This
    # ban was written when tab:retention was the only table in the paper, where the
    # corrected 94.4% is the correct cell and 95.5% was a stage-mixing error -- and
    # that row is still positively pinned by the "table row raw" MUST above, which is
    # the primary guard. Narrowed 2026-09-30 for the cost-per-correct-answer frontier
    # table, which is explicitly single-stage uncorrected throughout and where raw's
    # 95.5% is the only correct value; a global ban would force a stage-mixing error
    # into the one table built to avoid it. Scoped by the bracket: tab:retention cells
    # carry a Wilson interval after the percentage, frontier cells are bare. Same
    # reasoning as the verbatim 79.0%/75.8% narrowing below.
    # The `& 95.5\% [` and `& \ \ 4.1\%` bans that lived here until 2026-10-02 are
    # gone, not relaxed. Both required a tab:retention table cell, and when that
    # table became fig:retention there was no cell left to match -- a ban that
    # cannot fire still prints PASS, which is the failure mode this file exists to
    # avoid. Both are now enforced inside check_retention_figure(), against the
    # figure source, where the superseded values would actually have to appear.
    # Scoping them to the figure also preserves the carve-out the originals had:
    # the frontier table is explicitly single-stage uncorrected and 95.5% is the
    # correct value there, so a global ban would force an error into the one table
    # built to avoid it.
    # verbatim arm superseded values -- these MUST be table-row-scoped only, because
    # 79.0% and 75.8% legitimately appear in the threats section disclosing dedup
    # sensitivity (best-grade 79.8%, first-seen 75.8%, last-seen 79.0%, worst-grade 75.0%).
    # A global ban would break a correct disclosure.
    ("superseded verbatim 79.0% cited in a table row", r"& \\?\s*79\.0\\%"),
    ("superseded verbatim 75.8% cited in a table row", r"& \\?\s*75\.8\\%"),
    # unsourced verbatim CI that was removed
    ("superseded verbatim CI [6.0, 13.9]", r"\[6\.0, 13\.9\]"),
    # The action record row carried 96.8\% [91.8, 98.9] and 0.0\% [0.0, 3.1] from
    # 5710b938 until 465dc6f5. rq3_action_record_arm_summary.json gives Wilson
    # [92.0, 98.7] for recoverable 120/124 and 0.8\% [0.1, 4.5] for wrong-of-attempted
    # 1/121; [91.8, 98.9] is neither Wilson nor Clopper-Pearson ([91.9, 99.1]), and
    # [0.0, 3.1] is EXACTLY Wilson for 0 of 121, so the row was computed with
    # n_wrong = 0 while both the uncorrected and corrected blocks record n_wrong = 1.
    # Section 5 read it out as "zero hallucinated answers" for an arm that answered one
    # question wrongly. Nothing fired, because the percentage was right and pinned: a
    # pin on a figure says nothing about the interval printed beside it. Every other
    # row in tab:retention reproduces its source as exact Wilson, which is the only
    # reason this one was findable by hand.
    ("unsourced action record CI [91.8, 98.9]", r"\[91\.8, 98\.9\]"),
    ("action record wrong rate 0.0% (the arm has 1 wrong of 121)", r"0\.0\\% \[0\.0, 3\.1\]"),
    ("action record claimed 'zero hallucinated answers'", r"zero hallucinated answers"),
    # fabricated sample sizes that were removed
    ("fabricated sample size '325 of 325'", r"325 of 325"),
    ("fabricated sample size '325 graded sessions'", r"325 graded sessions"),
    # superseded raw artifact size
    ("superseded raw artifact size 700 KB", r"700\\,KB"),
    # The two-axis defect: asserting the 34/63/27 extraction mix underlies BOTH the
    # 11.1% and the 19.5%. Only the 11.1% comes from it; 87 = 124 - 37 absent is the
    # grade axis, and the extraction mix can only yield 90 or 63, never 87. Three
    # wordings seen so far -- the original, the filler pass's, and the compression's --
    # so match the claim shape, not one phrasing.
    ("two-axis defect, 'Both ... published label mix'",
     r"Both[^.]{0,60}(are computed on|use)[^.]{0,20}\s*\n?\s*published label mix"),
    ("two-axis defect, 'using 34 refusals ...' before both rates",
     r"using 34 refusals, 63 answers and 27 partials:[^.]*11\.1"),
    # The compression read the 6 reclassification CANDIDATES (35-29) as the number of
    # records moved. n_reclassified_answer_is_only_the_refusal = 3, and 11 wrong of 81
    # becomes 8 of 78, so the move is 3.
    ("miscount: 'moves 6 records'", r"moves 6 records"),
    # The dead parent-vs-subagent serialization control. Emptiness is determined by
    # the model, not the role: reproduced independently over the pinned manifest,
    # four models occur in both roles and each has an identical rate in each (opus-5
    # 17281/17281 parent and 492/492 subagent; sonnet-4-5 0/27 and 0/13016), with
    # zero exceptions. The mechanism is the documented `display: "omitted"` default.
    # None of this wording may come back.
    ("dead control: 'eliminates all four' explanations",
     r"eliminates all four"),
    ("dead control: loss located in client serialization",
     r"locates the loss in how this client serializes"),
    ("dead control: 'not a published interface'",
     r"not a published interface"),
    # Over-claim of corpus coverage. 968 of the 2,602 pinned transcripts are gone
    # from disk (966 of 1,477 parent files, against 2 of 1,125 subagent files), so a
    # walk today sees 35,377 blocks where the published figures were computed over
    # 35,680 (18,096 parent + 17,584 subagent). The cross-tab covers the re-readable
    # subset, not the corpus, and must not be described as "all ... in the corpus".
    ("over-claim: cross-tab covers 'all' blocks 'in the corpus'",
     r"all 35\{,\}377 blocks in the corpus"),
    # Wrong denominator for the surviving developer sessions. 226 is a single project
    # directory (the primary project under the transcript store). The non-spawn parent population is 300
    # files across 45 directories -- the other 74 are vault worktrees,
    # agent-session-rollup, an unrelated game project, public/ and job tmp dirs -- and all 74 survive,
    # so the figure is 299 of 300. Reproduced from the manifest. This also makes the
    # paragraph self-inconsistent: its "17,807 re-readable parent thinking blocks" is
    # computed over 511 files = 299 real + 212 spawn, the 300-population, so a reader
    # who divides cannot arrive at the file sentence's 226.
    ("wrong denominator: developer sessions '225 of 226'", r"225 of 226"),
]

# Additions to MUST that gate the pruning disclosure itself. The compression pass
# rewrites Threats, and this paragraph is the only place the corpus's re-readability
# is admitted; a later pass dropping it would leave a reproducibility claim the
# artifacts cannot support.
MUST += [
    (SUP + "pruning disclosure: 965 of the 966 missing parent files are pipeline spawns" + R7,
     r"965 of the 966", 1),
    (SUP + "pruning disclosure: developer sessions 299 of 300" + R7, r"299 of 300", 1),
    (SUP + "pruning disclosure: subagent files 1,123 of 1,125" + R7, r"1\{,\}123 of 1\{,\}125", 1),
    (SUP + "pruning disclosure: 35,377 re-readable against 35,680 published" + R7,
     r"35\{,\}377 thinking blocks against the 35\{,\}680", 1),
    # Two pins that stood here ("23 grew past it" and "raises the parent emptiness rate
    # rather than lowering it") are retired: see OBSOLETE_PINS.
]

# The nearest prior result, arXiv:2605.30790 (Ross, Koopman, van der Vegt, Zuccon,
# 2026-05-29). Verified against the arXiv record itself, not relayed: title "On the
# impact of retrieved content representations in RAG pipelines", four authors as
# spelled in refs.bib, baseline plus thirteen transformations = fourteen
# representations, four generators, retrieval held fixed, answer retention the primary
# determinant.
#
# The RETENTION CONDITION is the load-bearing clause and the most fragile sentence in
# the paper. Ross finds that once retention is high, wording, structure, LENGTH and
# query-dependence have limited effect. Our thesis is that delivered bytes govern, so
# taken unconditionally the two results contradict each other. They reconcile only
# because our read budget is what varies retention rather than holding it high -- a
# 500-character snippet often does not carry the answer-bearing span where a full trace
# does. A compression pass that drops the condition while keeping the citation turns
# this into a paper citing a result that refutes its own headline. Note the inverse
# error is equally available: our claim must NOT be hedged with Ross's high-retention
# condition, because under that condition their length null holds and ours does not.
MUST += [
    ("prior work: Ross et al. cited", r"\\cite\{ross2026representations\}", 1),
    ("prior work: retention condition kept on the Ross reconciliation",
     r"retention is not held high", 1),
    ("prior work: Ross named as the closest prior result",
     r"closest prior result", 1),
    # RQ5 figures migrated out of tab:breakeven when it was dropped for the Ross
    # paragraph (bd2f206b). The table was the only other place these lived.
    #
    # The p10 1.69, p90 2.37 and max 217.8 pins were removed deliberately. They were
    # refactor safety for that migration, never a claim the percentiles earn their
    # place, and the migration landed long ago. Keeping them forced :597 to carry five
    # figures for one quantity, which paper/CUTS.md:536 audited as Tier B decoration
    # and prescribed loosening "to the median and the floor, not to keep all five".
    # The distribution itself is not lost: it lives in rq5_cold_cache_correction.json.
    # The median and the floor stay pinned below, because they are the two the prose
    # still cites and dropping them would be a silent loss rather than a decision.
    ("RQ5 median 1.80 retained",        r"1\.80", 1),
    # The 1.63 floor is NOT a pricing artifact, and this is the permanent record of
    # where it comes from, since nothing else in the repo states it. From
    # rq5_cold_cache_correction.py:295-335 the break-even in retrievals is
    #
    #   be = (tt*inflation*mult*p_in + rt*p_out) / (saved*p_in)
    #      = (tt*inflation*mult + rt*(p_out/p_in)) / saved
    #
    # so the price LEVEL cancels and only the output:input ratio survives. All three
    # routes in that script's PRICES are exactly 5:1 ($2/$10, $1/$5, $3/$15), which is
    # why break_even_true is bit-identical across all three route blocks while
    # per_session_generation_cost_true_usd differs 2x -- a mathematical property, not
    # the copy-paste bug it looks like. As tt/rt grows the expression tends to
    # inflation*mult = 2.05 * 0.7915 = 1.6226, against an observed min of 1.6349: the
    # floor is the per-epoch transcript re-read times the effective cache multiplier.
    # Consequence for anyone repricing again: the $0.578 per-rollup figure moves with
    # the price level and the 1.80 median does not, so a future pricing correction
    # changes one and must leave the other alone.
    ("RQ5 floor 1.63 retained",         r"1\.63", 1),
]

# Reversed 2026-09-30: the tool_inputs_only RQ3 arm (rq3_tool_inputs_only_arm.jsonl,
# rq3_tool_inputs_only_arm_summary.json) strips ALL prose AND all tool results,
# keeping only tool-call input arguments capped at 2000 chars/tool. RQ3's questions
# are transcript-derived with ground truth living in prose, so this construction
# deletes the entire substrate where an answer could exist -- not a fair
# bytes-vs-content-type comparison against the other arms. It was excluded from the
# paper entirely as of an earlier pass today; it is now cited in Threats to Validity
# as an explicit negative/floor control (44.7% recoverable, corrected, at 19,764
# median context chars, against 0.0% for a true zero-context control), not as
# evidence for the delivered-bytes-vs-artifact-type question. These figures must now
# survive in main.tex rather than be banned outright -- assert presence instead of
# absence, so a later compression pass cannot silently delete the paragraph. See
# data/paper-rollups/rq3_tool_inputs_only_arm_summary.json for the current framing.
MUST += [
    ("tool_inputs_only floor control headline 46.3% cited as a negative control",
     r"46\.3\\%", 1),
    ("tool_inputs_only floor control median context 19,764 chars cited",
     r"19\{,\}764", 1),
]

# The frontier table's raw read cost. 206,665 is the median read cost over the full
# 124-question population, but the 95.5% recall in the same row is measured over only
# the 89 records the raw arm could be judged on -- the other 35 are transcripts that
# overflow the judge's context window. Pairing the 124-population cost with the
# 89-population recall is a cross-population artifact, and it propagated: the old
# punchline derived a "23x" span and "at most 1.1 points of recall" from it, both of
# which compared a cost on one population with a recall on another. 137,230 is the
# median over the same 89 records, so cost and recall now share one population.
# Assert the corrected figure is present and the superseded one cannot drift back.
MUST += [
    ("frontier raw read cost 137,230 on its own 89-record population",
     r"137\{,\}230", 1),
]

MUST_NOT += [
    ("superseded cross-population raw read cost 206,665", r"206\{,\}665"),
    # tab:breakeven was dropped, so any surviving \ref to it dangles. tectonic reports
    # this only as a warning and the page count is unaffected, so it will not fail a
    # build -- assert it here instead.
    ("dangling reference to the dropped tab:breakeven",
     r"\\ref\{tab:breakeven\}"),
    # The superseded cached-pricing columns went with the table. The prose keeps 0.24
    # to name it as superseded, which is correct; what must not come back is either
    # figure as a table CELL presented as a live result.
    ("superseded cached break-even 0.24 back in a table row", r"& \\?\s*0\.24"),
    ("superseded uncached break-even 1.16 back in a table row", r"& \\?\s*1\.16"),
]


MUST_NOT += [
    # 2026-10-09 regrade: the confidence-corrected / uncorrected split is gone, so the
    # words may not come back as a live stage, and the claims the new arms falsified
    # may not be restored.
    ("superseded stage words 'confidence-corrected' / 'uncorrected'",
     r"confidence-corrected|uncorrected"),
    ("superseded output-stripped-over-rollup gap 37.9 points", r"37\.9 points"),
    ("falsified: 'mostly its size and not its generator'",
     r"mostly its size and not its generator"),
    ("falsified: 'the rule cut still leads' the long Sonnet rollup",
     r"rule cut still leads by 8\.1"),
    ("falsified: matches-or-beats at EVERY length, unqualified",
     r"at every length we tested\.(?!\s*\n?\s*At equal)"),
    ("falsified: tie with a prose truncation stated for every generator",
     r"At equal length the generated summary (?:shows no detect(?:ed|able) difference|is statistically indistinguishable)"),
    ("superseded raw/output-stripped pair 98.9 against 95.5",
     r"98\.9\\% against 95\.5"),
]


# The seven whole-artifact arms of fig:retention. Until 2026-10-04 this was a
# literal table of recalls, intervals and wrong rates copied from the published
# tab:retention rows as they stood at c172175d. It is now a table of SOURCES:
# which arm, read at which grading stage, and what colour the figure draws it.
# Every number comes from data/paper-rollups/ through arm_rates, so a figure
# value that no longer matches its graded records fails here.
#
# `stage` is load-bearing and is the caption's own claim. fig:retention reports
# confidence-corrected rates "except prose-only, call-only and the action
# record", which are reported at the uncorrected stage because the correction
# cannot be derived for an arm whose raw replies were not kept (Section VI-D).
# Reading the wrong stage would silently compare the figure against a rate the
# figure does not claim: raw, for instance, is 95.5% uncorrected and 94.4%
# corrected, and the paper prints both in different places on purpose.
#
# `colour` stays a literal because it is not a measurement. It encodes the
# paper's content-class claim about the arm -- keeps the prose, replaces it with
# a generated summary, or deletes it and keeps the tool inputs -- which no file
# in data/paper-rollups/ records.
#
# (arm in arm_rates, figure's own row label, colour), in ROW-INDEX order, row 0 at
# the bottom of the axis. Re-anchored 2026-10-14 (section-by-section review): the
# figure is ordered by role, not recall -- top to bottom output-stripped,
# prose-only, raw, dialogue record, rollup, call-only, then the exploratory action
# record below a rule -- and the no-digest action record's design-set bar was
# dropped because all three of its measurements now sit in tab:nodigest, which
# check_nodigest_table() gates. Every remaining arm is still recomputed from the
# graded records; only the order rule changed, from "ascending" to this list.
RETENTION_ARMS = [
    ('action record',   'action record',   'cgreen'),
    ('call-only',       'call-only',       'corange'),
    ('rollup',          'rollup',          'cverm'),
    ('dialogue record', 'dialogue record', 'cgreen'),
    ('raw',             'raw transcript',  'cgreen'),
    ('prose-only',      'prose-only',      'cgreen'),
    ('output-stripped', 'output-stripped', 'cgreen'),
]


def retention_expected():
    """(label, recall, lo, hi, wrong_of_attempted, colour, n, source) per arm.

    Recomputed on every run from the graded records, never pinned. One grading
    stage since 2026-10-09: the revised grader's 3-draw majority.
    """
    out = []
    for arm, label, colour in RETENTION_ARMS:
        r = any_rate(arm)
        lo, hi = r.recall_ci
        out.append((label, r.recall, lo, hi, r.wrong_rate, colour, r.n, r.source))
    return out


# Values that were published under the previous grader and must not come back in
# the figure: every old recall and wrong rate that no arm's revised value equals.
RETENTION_BANNED_RECALLS = {47.2, 56.5, 71.8, 89.5, 96.8, 95.5, 14.7, 10.3, 9.2, 5.1, 2.5, 4.1}


def _read_retention_figure(arg):
    """Resolve paper/fig/fig-retention.tex from the SAME source as main.tex.

    A gate with its own default source grades the wrong file: run with --origin
    while the working copy holds the edit, or against a working-copy main.tex while
    the figure is read from origin, and the two halves of one float disagree
    without anything failing.
    """
    rel = "paper/fig/fig-retention.tex"
    if arg == "--origin":
        return subprocess.run(["git", "-C", VAULT, "show", f"origin/main:{rel}"],
                              capture_output=True, text=True).stdout, f"origin/main:{rel}"
    p = Path(arg).resolve().parent / "fig" / "fig-retention.tex"
    return (p.read_text() if p.exists() else ""), str(p)


def _err_fix():
    """data/paper-rollups/rq3_error_rate_fix.json, written by rq3_error_rate_fix.py, field `grade`."""
    import json
    return json.loads((REPO_ROOT / "data" / "paper-rollups" / "rq3_error_rate_fix.json").read_text())


def _err_gap():
    d = _err_fix()["rollup_uniform_minus_output_stripped"]["grade"]
    return [f"{d['diff_wrong_given_attempted_pts']:.1f}", f"{d['clustered_bootstrap_ci95_pts'][0]:.1f}",
            f"{d['clustered_bootstrap_ci95_pts'][1]:.1f}"]


def _err_arm(name):
    return _err_fix()["arms"][name]["grade"]


def absent_pair():
    """(rollup_uniform absent %, output_stripped absent %) as 1-decimal strings."""
    return [f"{_err_arm('rollup_uniform')['absent_pct']:.1f}", f"{_err_arm('output_stripped')['absent_pct']:.1f}"]


def _held():
    import json
    return json.loads((REPO_ROOT / "data" / "paper-rollups" / "heldout-rollup" / "results.json").read_text())["heldout"]


def provenance_medians():
    """(held-out, design) median cl100k tokens of the rollup notes, as TeX-formatted strings."""
    import json
    d = json.loads((REPO_ROOT / "data" / "paper-rollups" / "heldout-rollup" / "results.json").read_text())["rollup_provenance"]
    return [f"{d['heldout_notes']['median_cl100k_tokens']:,}".replace(",", "{,}"),
            f"{d['design_notes']['median_cl100k_tokens']:,}".replace(",", "{,}")]


def held_vals():
    """Held-out figures as strings: rollup wrong%, strip wrong%, diff, ci lo, ci hi, rollup absent%, strip absent%, rollup recall%."""
    h = _held()
    r, o, d = h["rollup"], h["output_stripped"], h["rollup_minus_output_stripped"]
    return [f"{r['wrong_given_attempted_pct']:.1f}", f"{o['wrong_given_attempted_pct']:.1f}",
            f"{d['diff_wrong_given_attempted_pts']:.1f}", f"{d['clustered_bootstrap_ci95_pts'][0]:.1f}",
            f"{d['clustered_bootstrap_ci95_pts'][1]:.1f}", f"{r['absent_pct']:.1f}", f"{o['absent_pct']:.1f}",
            f"{r['recall_pct']:.1f}"]


def W(rx):
    """A pin regex whose literal spaces tolerate a line wrap in main.tex (spaces become \\s+)."""
    return rx.replace(" ", r"\s+")


def _j(rel):
    """A JSON file under data/paper-rollups/, parsed."""
    import json
    return json.loads((REPO_ROOT / "data" / "paper-rollups" / rel).read_text())


def _f1(v):
    return f"{v:.1f}"


def rep_gap():
    """Replication 1 rollup-minus-output-stripped wrong-of-attempted: diff, ci lo, ci hi (1 decimal strings)."""
    d = _j("replication-rollup/results.json")["rollup_minus_output_stripped"]["wrong_given_attempted"]
    return [_f1(d["diff_pts"]), _f1(d["clustered_ci95_pts"][0]), _f1(d["clustered_ci95_pts"][1])]


def haiku_regen_wrong():
    """Logged Haiku 4.5 held-out rollup: wrong-of-attempted %, attempts."""
    r = _j("heldout-rollup-haiku/results.json")["rows"]["haiku_written_rollup"]
    return [_f1(r["wrong_given_attempted_pct"]), str(r["attempted"])]


def sonnet1k():
    """Held-out Sonnet 1,000-token arm: dict of 1-decimal strings."""
    d = _j("sonnet1k-heldout/results.json")
    a, pr = d["arms"], d["paired"]
    cu, pn = pr["sonnet1k_minus_cut"]["recall"], pr["sonnet1k_minus_production_note"]["recall"]
    return {"median": f"{d['generation']['rollup_tokens_median']:,}".replace(",", "{,}"),
            "s1k": _f1(a["sonnet1k"]["recall_pct"]), "prod": _f1(a["production_note"]["recall_pct"]),
            "strip": _f1(a["output_stripped"]["recall_pct"]), "cut": _f1(a["cut"]["recall_pct"]),
            "pn_d": _f1(pn["diff_pts"]), "pn_lo": _f1(pn["clustered_bootstrap_ci95_pts"][0]),
            "pn_hi": _f1(pn["clustered_bootstrap_ci95_pts"][1]),
            "cu_d": _f1(cu["diff_pts"]), "cu_lo": _f1(abs(cu["clustered_bootstrap_ci95_pts"][0])),
            "cu_hi": _f1(cu["clustered_bootstrap_ci95_pts"][1]),
            "cap_from": __import__("re").search(r"raised from (\d+) to (\d+)", d["deviation"]).group(1), "cap_to": str(int(d["cap_usd"])), "prefix_n": str(d["prefix_check_first55_cut"]["n_paired"])}



def strip_wrong_n():
    return _err_fix()["arms"]["output_stripped"]["grade"]["wrong"]


def strip_att_n():
    return _err_fix()["arms"]["output_stripped"]["grade"]["attempted"]


def strip_wrong_ci():
    return _err_fix()["arms"]["output_stripped"]["grade"]["wrong_given_attempted_ci95"]


def check_retention_figure(arg):
    """Gate fig:retention, which replaced tab:retention on 2026-10-02.

    The four MUST entries this replaces asserted that a recall and its
    wrong-of-attempted rate sat in ONE table row, so an edit could not move one
    without the other. A figure has no rows, so the equivalent property is the row
    INDEX shared by the recall node, the wrong node, the interval whiskers and the
    dot: this reads all four back and requires them to agree per arm.

    Every expected value is recomputed from the graded records by
    retention_expected(), so this now also catches the case the pinned version
    could not: the figure and its source data disagreeing, with the figure still
    self-consistent.
    """
    fig, src = _read_retention_figure(arg)
    if not fig.strip():
        return False, f"could not read {src}"
    expected = retention_expected()

    # recall column at x=117, wrong column at x=135, both keyed by row index
    rec = {int(i): float(v) for i, v in
           re.findall(r"axis cs:117,(\d+)\) \{([\d.]+)\}", fig)}
    wrong = {int(i): float(v) for i, v in
             re.findall(r"axis cs:135,(\d+)\) \{([\d.]+)\}", fig)}
    # the dot: colour carries the content-class claim, x is the point estimate
    dot = {int(i): (c, float(x)) for c, x, i in
           re.findall(r"\\fill\[(\w+)\] \(axis cs:([\d.]+),(\d+)\) circle", fig)}
    # the whisker: a horizontal rule at one row index from lo to hi
    whisk = {int(i): (float(lo), float(hi)) for lo, i, hi, i2 in
             [(a, b, c, d) for a, b, c, d in
              re.findall(r"\(axis cs:([\d.]+),(\d+)\) -- \(axis cs:([\d.]+),(\d+)\);", fig)]
             if i == i2}

    # each row's own comment line, which is where the figure records its n
    noted = {}
    for m in re.finditer(r"% ([\w -]+): ([\d.]+) \[([\d.]+), ([\d.]+)\], "
                         r"wrong ([\d.]+), n=(\d+)", fig):
        noted[m[1]] = (float(m[2]), float(m[3]), float(m[4]), float(m[5]), int(m[6]))

    n = len(expected)
    for name, dct in (("recall", rec), ("wrong", wrong), ("dot", dot),
                      ("whisker", whisk), ("annotation", noted)):
        if len(dct) != n:
            return False, f"{src}: found {len(dct)} {name} entries, expected {n}"

    for i, (name, r, lo, hi, w, colour, nn, source) in enumerate(expected):
        if rec.get(i) != r:
            return False, (f"{name}: recall column row {i} is {rec.get(i)}, "
                           f"{source} recomputes to {r}")
        if wrong.get(i) != w:
            return False, (f"{name}: wrong rate row {i} is {wrong.get(i)}, {source}"
                           f" recomputes to {w} -- recall and wrong rate have come apart")
        if dot.get(i, (None, None))[1] != r:
            return False, f"{name}: dot at row {i} is {dot.get(i)}, expected {r}"
        if dot.get(i, (None, None))[0] != colour:
            return False, (f"{name}: row {i} is drawn {dot.get(i, ('?',))[0]}, expected"
                           f" {colour} -- the colour is the content-class claim")
        if whisk.get(i) != (lo, hi):
            return False, (f"{name}: whisker row {i} is {whisk.get(i)}, Wilson over "
                           f"{source} is ({lo}, {hi})")
        if not lo <= r <= hi:
            return False, f"{name}: {r} sits outside its own interval [{lo}, {hi}]"
        if noted.get(name) != (r, lo, hi, w, nn):
            return False, (f"{name}: the row's own annotation is {noted.get(name)},"
                           f" {source} recomputes to {(r, lo, hi, w, nn)} -- the"
                           f" sample size or a rate disagrees with the data")

    # role order, not recall order (the caption says so): the per-row checks above
    # already pair row i with RETENTION_ARMS[i], so the order is pinned by that list;
    # this also requires the exploratory action record at the bottom, below a rule
    if expected[0][0] != "action record" or "(axis cs:38,0.5) -- (axis cs:102,0.5)" not in fig:
        return False, f"{src}: the exploratory action record is not last and ruled off"
    if "no digest" in fig:
        return False, f"{src}: the no-digest bar is back; its measurements live in tab:nodigest"

    banned = RETENTION_BANNED_RECALLS & set(rec.values()) | \
        RETENTION_BANNED_RECALLS & set(wrong.values())
    if banned:
        return False, f"{src}: superseded value(s) {sorted(banned)} are back in the figure"

    # the capped-budget arms belong to Section VI-E's prose, not to this figure
    for stray in ("pointer", "chunk", "rollup body"):
        if stray in fig:
            return False, f"{src}: capped-budget arm '{stray}' has leaked into the figure"

    return True, (f"{n} arms recomputed from data/paper-rollups/, recall paired with "
                  f"wrong rate by row, Wilson intervals and n agree, role order, no "
                  f"superseded value")


def check_intro_tight_budget_range(s):
    """The 400-token summary's lead over same-length rule cuts, stated as 'by A to B
    points'; A and B must be the rounded body figures for the 2,000-character cut
    and the 1,500-character head cut in sec:rq3-budget.

    Re-anchored 2026-10-14 (second section-by-section pass): the intro paragraph that
    carried the range was cut to the rule in words, and the range now sits in the
    abstract and the third contribution as well as the budget section's lead. Every
    occurrence of the range, in either wording, is checked, and the abstract must
    carry one -- stricter than the single first match this used to read."""
    # re-anchored 2026-10-14 (round 3): the abstract spells out "percentage points", the rest says "pp"
    rx = W(r"beats (?:rule cuts of the same length|same-length rule cuts) by (\d+) to (\d+) (?:percentage points|pp)")
    occ = list(re.finditer(rx, s))
    b = re.search(r"beats the first 1\{,\}500 characters of prose by ([\d.]+) pp.*?nearer its length, by ([\d.]+)", s, re.S)
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    if not occ or not b or not ab:
        return False, "range, budget-section figures or abstract not found"
    want = (str(round(float(b.group(2)))), str(round(float(b.group(1)))))
    if not re.search(rx, ab.group(1)):
        return False, "the abstract no longer states the tight-budget range"
    bad = [f"'{m.group(0)}' says {m.group(1)} to {m.group(2)}" for m in occ if (m.group(1), m.group(2)) != want]
    if bad:
        return False, "; ".join(bad) + f"; body rounds to {want[0]} to {want[1]}"
    return True, f"{len(occ)} statements (abstract included) say {want[0]} to {want[1]}, as the body rounds"


def check_strict_row(s):
    """The strict 400-token Sonnet run (19 smallest sessions) against
    cap400/control/summary.json. Added 2026-10-14 when the 322-token run got its
    absolute recall into tab:frontier; moved to the budget-section prose in round 7
    (cain-review-1014), when the human dropped the strict row from the table. Same
    values and strictness: tokens, sessions, questions, recall and wrong-answer rate,
    and the row must not come back into the table as well."""
    blocks = _fb.table_blocks(s)
    if not blocks:
        return False, "merged tab:frontier not found"
    if any("strict" in r[0] for b in blocks for r in b):
        return False, "the strict row is back in tab:frontier as well as in the prose"
    c = _j("cap400/control/summary.json")["arms"]["sonnet400trim:g1:d1"]
    want = (int(c["median_context_cl100k_per_question"]), c["n_sessions"], c["n"],
            f"{c['pct']:.1f}", f"{c['wrong_of_attempted_pct']:.1f}")
    rx = W(r"held Sonnet to a median (\d+) tokens on the (\d+) smallest sessions \((\d+) questions\), where it recovers ([\d.]+)\\% with a ([\d.]+)\\% wrong-answer rate\. These sessions are easier, so we compare it only through paired differences\.")
    hits = list(re.finditer(rx, s))
    if len(hits) != 1:
        return False, f"the budget-section strict-run sentence matches {len(hits)} times, expected 1"
    m = hits[0]
    got = (int(m[1]), int(m[2]), int(m[3]), m[4], m[5])
    if got != want:
        return False, f"prose says {got}, control summary {want}"
    return True, f"n={c['n']}, {want[0]} tokens, {want[3]}% recall, {want[4]}% wrong reproduce from cap400/control (prose)"


def check_abstract_band_winners_agree_with_frontier(s):
    """Check that substrates named as F-band winners in the abstract are
    actually winners in the frontier sentence. Returns (ok: bool, message: str).
    """
    # Re-anchored 2026-10-14: the seven-band enumeration sentence this parsed was
    # dropped by the section-by-section review, so the winners now come straight
    # from the upper hull of the merged tab:frontier, which verify_tex() also checks
    # against the band sentence that replaced the enumeration. Stricter, not looser:
    # the winner set is recomputed instead of read off prose.
    pts, _comp = _fb.priced_points(s)
    if not pts:
        return False, "merged substrate table tab:frontier not found"
    frontier_winners = {h[0].lower() for h in _fb.upper_hull(pts)}

    # Extract the abstract
    abstract_match = re.search(
        r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
        s, re.DOTALL
    )
    if not abstract_match:
        return False, "abstract not found"

    abstract_text = abstract_match.group(1)

    # Define substrate names to check (various forms)
    # Map abstract forms to canonical forms for checking
    substrate_patterns = {
        'zero context': ['zero context', 'zero'],
        'prompts-only': ['prompts-only', 'prompts only'],
        'prose-2k': ['prose-2k', 'prose @2k', 'prose 2k'],
        'prose-8k': ['prose-8k', 'prose @8k', 'prose 8k'],
        'rollup': ['rollup', 'rollups'],
        'answer-only': ['answer-only', 'answer only'],
        'prose-only': ['prose-only', 'prose only'],
        'dialogue record': ['dialogue record'],
        'call-only': ['call-only', 'call only'],
        'action record': ['action record', 'structured action record', 'structured action records'],
        'output-stripped': ['output-stripped', 'output stripped', 'output-stripped transcript', 'output-stripped transcripts'],
        'raw': ['raw', 'raw transcript', 'raw transcripts']
    }

    violations = []

    # For each substrate type, check if it's mentioned with an F-band in the abstract
    for canonical, variants in substrate_patterns.items():
        for variant in variants:
            # Look for this variant mentioned with an F-band attribution
            # Pattern: variant name within 150 chars of $F with comparison operator
            pattern = (
                r'(?:' + re.escape(variant) + r'[^.]{0,150}\$F\s*[<>=])|' +
                r'(?:\$F\s*[<>=][^.]{0,150}' + re.escape(variant) + r')'
            )
            if re.search(pattern, abstract_text, re.IGNORECASE):
                # This substrate is attributed an F-band in the abstract
                # Check if canonical form (or any variant) is in frontier_winners
                found_in_frontier = False
                for check_form in [canonical] + variants:
                    if check_form.lower() in frontier_winners:
                        found_in_frontier = True
                        break
                    # Also check partial matches (e.g., "zero" matches "zero context")
                    for fw in frontier_winners:
                        # If frontier winner is substring of canonical or vice versa
                        if fw in check_form.lower() or check_form.lower() in fw:
                            found_in_frontier = True
                            break
                    if found_in_frontier:
                        break

                if not found_in_frontier:
                    violations.append(variant)
                    break  # Only report once per canonical substrate

    if violations:
        return False, f"names attributed F bands in abstract but not frontier winners: {', '.join(violations)}"

    return True, "all abstract band attributions match frontier winners"


def check_rq5_break_even_cold_cache(s, path=None):
    """rq5_generation_cost.json must carry the cold-cache break-even the paper cites.

    rq5_generation_cost.py rewrites that file with the warm-cache figures (median
    1.19). Returns (ok, message).
    """
    import json
    path = Path(path) if path else REPO_ROOT / "data" / "paper-rollups" / "rq5_generation_cost.json"
    rq5 = json.loads(path.read_text())
    be = rq5.get("break_even_retrievals", {})
    med, lo = be.get("median"), be.get("min")
    problems = []
    if "break_even_retrievals_source" not in rq5:
        problems.append("break_even_retrievals_source is missing")
    for name, val, want in (("median", med, "1.80"), ("min", lo, "1.63")):
        got = f"{val:.2f}" if isinstance(val, (int, float)) else None
        if got != want:
            problems.append(f"{name} is {got}, expected {want}")
        if not re.search(re.escape(want), s):
            problems.append(f"{want} not in main.tex")
    if problems:
        return False, "; ".join(problems) + " -- rerun rq5_cold_cache_correction.py"
    return True, f"median {med}, min {lo}, source present, both in main.tex"


def check_savings_claim_scoped_to_read_path(s):
    """Every "compression saves nothing" claim must name the mechanism that makes
    it true: the read path's fixed character cap. Ross et al. hold retrieval fixed
    and vary representation, so an unscoped "compression saves nothing" reads as
    their result restated -- and it is a far stronger claim than we measured,
    because on a whole-artifact read compression demonstrably does save (10.0x in
    tokens against the output-stripped transcript). Returns (ok, message).

    Replaced the regex r"dose[^.]{0,40}read path" on 2026-09-30. That check
    required the literal word "dose", which belonged to the falsified
    delivered-volume thesis; removing that thesis correctly deleted the word and
    then incorrectly failed the gate, which is how a stale gate invites someone to
    edit the paper to satisfy it. Guard the over-claim directly instead of
    mandating one phrasing of the scoped version, so the check survives a rewrite.
    """
    CLAIM = (r"(saves? nothing|saves? no (?:read )?tokens|zero (?:read )?saving"
             r"|read saving is\s+zero|is void)")
    # Anchor every scope token on a leading word boundary. An earlier draft used a
    # bare r"cap\b", which matched the "cap" inside unrelated words and passed a
    # deliberately unscoped claim -- a scope check that fires on a substring is a
    # scope check that cannot fail.
    SCOPE = (r"(read path|read budget|character budget|character count"
             r"|\bcapp?(?:ed|ing|s)?\b|fixed budget|\btruncat)")
    hits = list(re.finditer(CLAIM, s, re.IGNORECASE))
    if not hits:
        return False, "no savings claim found at all; step 5 of the argument is missing"
    unscoped = []
    for m in hits:
        window = s[max(0, m.start() - 320):m.end() + 320]
        if not re.search(SCOPE, window, re.IGNORECASE):
            unscoped.append(s[max(0, m.start() - 70):m.end() + 70].replace("\n", " "))
    if unscoped:
        return False, (f"{len(unscoped)} of {len(hits)} savings claims unscoped, "
                       f"first: {unscoped[0][:110]!r}")
    return True, f"all {len(hits)} savings claims scoped to the read path"


# The substrate table's column labels, after frontier_bands normalises its two
# spellings, against the arm names arm_rates knows.
_SUBSTRATE_ARM = {"zero": "zero context"}


def check_substrate_table_recalls(s):
    """Recompute every priced-substrate cell of the merged tab:frontier from its records.

    frontier_bands.verify_tex() already recomputes the band sentence from this
    table, so the table was the one link in that chain still taken on trust:
    edit a recall cell and the bands move with it, consistently, and nothing
    fired. This closes it at the other end -- the cells themselves now have to
    reproduce from data/paper-rollups/.

    Re-anchored 2026-10-14, when tab:frontier absorbed tab:length and gained n and
    wrong columns: every arm of UNCORRECTED_SOURCES must appear exactly once (raw
    in its not-comparable block), plus the no-digest action record, and each row's
    n, recall and wrong-of-attempted must reproduce ("--" exactly when nothing was
    attempted). One grading stage since 2026-10-09, so uncorrected() is that stage.
    """
    # Round 7 (cain-review-1014): zero context and raw left the table for the
    # prose. Zero context is graded by the "method: zero context" pin and priced
    # from that sentence by frontier_bands.zero_point(); raw by the RQ3 raw pins
    # (89 questions, 94.4%, zero wrong) and the 137,230 MUST pin. Neither may
    # come back into the table as well.
    blocks = _fb.table_blocks(s)
    if not blocks or len(blocks) != 1:
        return False, "merged tab:frontier not found, or not one comparison block"
    comp, rawb = blocks[0], []
    in_prose = {"raw", "zero context"}
    back = [r[0] for r in comp if {"zero": "zero context"}.get(r[0], r[0]) in in_prose]
    seen = {}
    for name, gen, n, tok, rec, wr, _d in comp + rawb:
        arm = {"zero": "zero context", "rollup": "rollup", "no-digest": "action record (no digest)"}.get(name, name)
        if arm in UNCORRECTED_SOURCES or arm == "action record (no digest)":
            seen.setdefault(arm, []).append((n, rec, wr, gen, tok))
    want = (set(UNCORRECTED_SOURCES) - in_prose) | {"action record (no digest)"}
    problems = [f"'{a}' appears {len(seen.get(a, []))} times" for a in sorted(want) if len(seen.get(a, [])) != 1]
    if back:
        problems.append(f"{back} back in tab:frontier as well as the prose")
    for arm, [(n, rec, wr, gen, tok)] in ((k, v) for k, v in seen.items() if len(v) == 1):
        r = any_rate(arm) if arm == "action record (no digest)" else uncorrected(arm)
        if (n, f"{rec:.1f}") != (r.n, f"{r.recall:.1f}"):
            problems.append(f"'{arm}' row says n={n}, {rec:.1f}%; {r.source} gives {r.n}, "
                            f"{r.recoverable_n}/{r.n} = {r.recall:.1f}%")
        want_w = None if r.attempted_n == 0 else round(r.wrong_rate, 1)
        if wr != want_w:
            problems.append(f"'{arm}' wrong cell {wr}, records give {want_w}")
        if arm != "rollup" and gen != "none":
            problems.append(f"'{arm}' is a rule cut but its generator cell says {gen}")
    pub = {n: t for n, t, _r in _fb.PUBLISHED}
    pub["raw"] = 137230
    nds = _j("action_record_no_digests_cl100k.json")["no_result_digests_variant"]["median_cl100k_tokens"]
    pub["no-digest"] = int(nds)
    for name, _g, _n, tok, *_ in comp + rawb:
        if name in pub and tok != pub[name]:
            problems.append(f"'{name}' tokens {tok}, measured {pub[name]}")
    if problems:
        return False, "; ".join(problems)
    return True, (f"all {len(want)} table rows recompute n, recall and "
                  f"wrong from the graded records; raw and zero context are graded in the prose")


def check_capped_rows(s):
    """The capped read path against read_budget().

    Added 2026-10-14, when the three capped rows moved into tab:frontier; moved
    back to the prose in round 7 (cain-review-1014), when the human dropped the
    capped block. The recalls and intervals are graded by the two "RQ3 read
    budget" pins; this checks what the table rows added: the three artifacts share
    the 89-record population the prose states, and the rows are not back in the table.
    """
    blocks = _fb.table_blocks(s)
    if not blocks:
        return False, "merged tab:frontier not found"
    src = {"rollup pointer": "rollup_pointer", "rollup summary snippet": "rollup_hookshape",
           "output-stripped chunk": "stripped_chunk"}
    back = [r[0] for b in blocks for r in b if r[0] in src]
    if back:
        return False, f"capped rows back in tab:frontier as well as the prose: {back}"
    ns = {k: read_budget(v).n for k, v in src.items()}
    bad = []
    m1 = re.search(W(r"We re-ran answering at that cap over the (\d+) complete records"), s)
    m2 = re.search(W(r"The capped path is measured on (\d+) records"), s)
    for label, m in (("re-run sentence", m1), ("section lead", m2)):
        if not m:
            bad.append(f"the {label} no longer states the capped population")
        elif any(int(m[1]) != n for n in ns.values()):
            bad.append(f"the {label} says {m[1]} records; read_budget() gives {ns}")
    return (False, "; ".join(bad)) if bad else (True, f"3 capped artifacts on {set(ns.values())} records, as both prose sentences state; not in the table")


RQ4_ARTIFACTS = ("Rollup", "Raw transcript", "Output-stripped", "No-digest record")


def _rq4_cells(s, label="tab:rq4", ncols=4):
    """The recall block of tab:rq4 (or tab:rq4-r1): one row per artifact, four cells,
    transcript probes BM25 and dense, then rollup-text probes BM25 and dense.

    Re-anchored 2026-10-14 (round 5): rows used to mix artifact and method
    ("Rollup (dense)"); they are now artifacts only, and the method is a column.
    """
    m = re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{tabular\}", s, re.S)
    if not m:
        return None
    cells = r"\s*&\s*([\d.]+)" * ncols
    names = "|".join(re.escape(a) for a in RQ4_ARTIFACTS)
    return {r[0].strip(): r[1:] for r in
            re.findall(r"^(" + names + r")" + cells + r"\s*\\\\\s*$", m[1], re.M)}


def _float_note(s, label):
    """The note under a table's tabular, inside its float: the brace-balanced body of the
    first \\parbox{...}{...} after \\label{label} and before the float's \\end{table}.

    Added 2026-10-15 (cain-review-1014, table notes): every table now carries a one-sentence
    caption and puts what a reader needs to read it in such a note, in normal type, as
    tab:frontier already did. Pins that graded caption text grade the note through this.
    """
    i = s.find("\\label{" + label + "}")
    if i < 0:
        return None
    end = s.find("\\end{table", i)
    j = s.find("\\parbox{", i, end if end >= 0 else None)
    if j < 0:
        return None
    k = s.index("}{", j) + 2          # skip the width argument
    depth, q = 1, k
    while depth:
        c = s[q]
        depth += (c == "{") - (c == "}")
        q += 1
    body = s[k:q - 1]
    for pre in ("\\scriptsize\\raggedright ", "\\scriptsize\\raggedright"):
        if body.startswith(pre):
            body = body[len(pre):]
            break
    return body


def _rq4_caption(s):
    # re-anchored 2026-10-15 (table notes): the figures left the tab:rq4 caption verbatim for its note
    return _float_note(s, "tab:rq4")


_RQ4_DCELL = re.compile(r"\$([+-]\d+\.\d)\$(?: \[\$([+-]\d+\.\d)\$, \$([+-]\d+\.\d)\$\]|(\$\^\{\*\}\$))")


def _rq4_s4_caption(sup):
    # re-anchored 2026-10-15 (table notes): the S4 caption's reading notes moved verbatim to the
    # note under the tabular, and its test figures to the 'Tests behind Table S4' paragraph after
    # the float; this returns the two joined, so every regex below grades the same text it did
    note = _float_note(sup, "tab:rq4-diff")
    m = re.search(r"^\\noindent\\textbf\{Tests behind Table~\\ref\{tab:rq4-diff\}\.\} (.*?)$", sup, re.M)
    if note is None or not m:
        return None
    return note + " " + m[1]


def _rq4_diff_rows(sup):
    """Supplementary Table S4 (tab:rq4-diff), keyed (probe set, label).

    Each cell is (point, lo, hi) as printed in percentage points with one decimal,
    or (point, None, None) for a cell flagged $^{*}$ (no interval computed).
    Added 2026-10-14 (round 5), when the paired differences and their intervals
    left the RQ4 prose for tab:rq4's difference block. Re-anchored 2026-10-14
    (round 6): the block moved verbatim to supplementary Table S4 in
    paper/supplement/sensitivity-checks.tex, converted from proportions to pp
    (-0.070 -> -7.0), as two columns (BM25, dense) instead of multicolumn pairs.
    """
    m = re.search(r"\\label\{tab:rq4-diff\}(.*?)\\end\{tabular\}", sup, re.S)
    if not m or "Difference in recall@10" not in m[1]:
        return None
    body = m[1].split("Difference in recall@10", 1)[1]
    out, sect = {}, None
    for line in body.splitlines():
        if re.match(r"\\multicolumn\{3\}\{l\}\{Transcript probes\}", line):
            sect = "transcript"; continue
        if re.match(r"\\multicolumn\{3\}\{l\}\{Rollup-text probes\}", line):
            sect = "rollup-text"; continue
        mm = re.match(r"^([A-Za-z-]+ \$-\$ [A-Za-z-]+) & (.*)\\\\\s*$", line)
        if not mm or sect is None:
            continue
        cs = _RQ4_DCELL.findall(mm[2])
        if len(cs) != 2:
            out[(sect, mm[1])] = "unparsed"
            continue
        out[(sect, mm[1])] = tuple((c[0], c[1] or None, c[2] or None) for c in cs)
    return out


def _pp(v):
    """A proportion difference printed as signless pp with one decimal (0.134 -> '13.4')."""
    return f"{100 * v:.1f}"


def check_rq4_table(s, sup=""):
    """Every cell of tab:rq4 -- recall block, difference block, caption figures -- and
    the recall@1 cells of supplementary Table S3, against the retrieval result files.

    Added 2026-10-14, when the review added the no-digest rows and the rollup-text
    probe columns, moving recall values out of RQ4's prose into the table.

    Re-anchored 2026-10-14 (round 4): tab:rq4 in main.tex keeps only recall@10;
    recall@1 moved to supplementary Table S3 (tab:rq4-r1), graded here cell for cell.

    Re-anchored 2026-10-14 (round 5): rows are artifacts only and the retrieval
    method is a column, in both tables (16 recall@10 + 16 recall@1 cells, values
    unchanged). The RQ4 prose lost every paired difference, interval, the
    equivalence test, the McNemar p, the overlaps and the method gaps; they are in
    the table's difference block and caption now and are graded here: 12 difference
    cells (point and interval, or the flagged no-interval cell, whose absence of an
    interval in the data is itself checked), and 15 caption figures.

    Re-anchored 2026-10-14 (round 6): tab:rq4 prints recall@10 as percentages with
    one decimal (0.708 -> 70.8), graded against 100 x the data. The difference
    block moved to supplementary Table S4 (tab:rq4-diff), in pp, graded cell for
    cell in the supplement. Recall-difference figures in the captions are in pp
    (method gap 13.4, margin +-5, 90% CI [-9.8, -4.3], gaps 7.5 to 12.9, about
    20); the method gap, the no-interval note and 'Stripped is' moved to the S4
    caption. Word overlaps and the McNemar p stay as they were. Same 16 + 12 +
    16 cells and 15 caption figures, plus the prose's 13.4.
    """
    cells = _rq4_cells(s, "tab:rq4", 4)
    if not cells:
        return False, "tab:rq4 recall block not found"
    r1 = _rq4_cells(sup, "tab:rq4-r1", 4)
    if not r1:
        return False, "supplementary Table S3 (tab:rq4-r1) not found in the supplement file"
    orig = _j("rq4_no_digest_probe.json")["arms"]
    pf = _j("probes-from-rollups/results.json")
    newb = pf["recall_arms_new_probes"]
    newd = pf["dense"]["recall_new_probes"]
    spec = {"Rollup": (orig["D_bm25_rollup"], orig["A_rollup"], newb["D_bm25_rollup"], newd["A_dense_rollup"]),
            "Raw transcript": (orig["E_bm25_raw"], orig["B_raw"], newb["E_bm25_raw"], newd["B_dense_raw"]),
            "Output-stripped": (orig["F_bm25_no_tool"], orig["C_raw_no_tool"], newb["F_bm25_no_tool"], newd["C_dense_no_tool"]),
            "No-digest record": (orig["G_bm25_no_digest"], orig["G_dense_no_digest"], newb["G_bm25_no_digest"], newd["G_dense_no_digest"])}
    if set(cells) != set(spec):
        return False, f"tab:rq4 rows {sorted(cells)} do not match {sorted(spec)}"
    if set(r1) != set(spec):
        return False, f"supplementary Table S3 rows {sorted(r1)} do not match {sorted(spec)}"
    bad = []
    for name, arms in spec.items():
        want10 = tuple(f"{100 * a['recall@10']:.1f}" for a in arms)
        want1 = tuple(f"{a['recall@1']:.3f}" for a in arms)
        if cells[name] != want10:
            bad.append(f"{name}: tab:rq4 R@10 {cells[name]}, data {want10}")
        if r1[name] != want1:
            bad.append(f"{name}: supplementary Table S3 R@1 {r1[name]}, data {want1}")

    # ---- difference block
    ki = _j("rq4_known_item_rq4_2026-10-06_with_F.json")
    sig = ki["significance_tests"]
    nd = _j("rq4_no_digest_probe.json")["paired_bootstrap_no_digest_minus_other"]
    pdn = pf["paired_differences_new_probes"]
    pdd = pf["dense"]["paired_substrate_differences_new_probes_dense"]

    def bs(d):
        return (d["mean_diff"], d["ci_lower"], d["ci_upper"])

    def neg(t):
        return (-t[0], -t[2], -t[1])

    if any(k in sig for k in ("C_raw_no_tool_vs_B_raw", "B_raw_vs_C_raw_no_tool")):
        bad.append("the data now has an interval for dense output-stripped minus raw; print it instead of the no-interval flag")
    dspec = {
        ("transcript", "Rollup $-$ raw"): (bs(sig["D_bm25_rollup_vs_E_bm25_raw"]["bootstrap_ci"]), bs(sig["A_rollup_vs_B_raw"]["bootstrap_ci"])),
        ("transcript", "Stripped $-$ rollup"): (bs(sig["F_bm25_no_tool_vs_D_bm25_rollup"]["bootstrap_ci"]), neg(bs(sig["A_rollup_vs_C_raw_no_tool"]["bootstrap_ci"]))),
        ("transcript", "Stripped $-$ raw"): (bs(sig["F_bm25_no_tool_vs_E_bm25_raw"]["bootstrap_ci"]),
                                             (orig["C_raw_no_tool"]["recall@10"] - orig["B_raw"]["recall@10"], None, None)),
        ("transcript", "No-digest $-$ stripped"): (bs(nd["G_bm25_no_digest_minus_F_bm25_no_tool"]), bs(nd["G_dense_no_digest_minus_C_raw_no_tool"])),
        ("rollup-text", "Rollup $-$ raw"): (bs(pdn["D_bm25_rollup_minus_E_bm25_raw"]), bs(pdd["A_dense_rollup_minus_B_dense_raw"])),
        ("rollup-text", "Stripped $-$ no-digest"): (bs(pdn["F_bm25_no_tool_minus_G_bm25_no_digest"]), bs(pdd["C_dense_no_tool_minus_G_dense_no_digest"])),
    }
    drows = _rq4_diff_rows(sup)
    if drows is None:
        return False, "supplementary Table S4 (tab:rq4-diff) has no 'Difference in recall@10' block"
    if set(drows) != set(dspec):
        return False, f"supplementary Table S4 difference rows {sorted(drows)} do not match {sorted(dspec)}"
    fmt = lambda v: None if v is None else f"{100 * v:+.1f}".replace("-0.0", "+0.0")
    for key, want in dspec.items():
        got = drows[key]
        if got == "unparsed":
            bad.append(f"{key}: difference cells do not parse"); continue
        for meth, g, w in zip(("BM25", "dense"), got, want):
            wt = tuple(fmt(x) for x in w)
            if g != wt:
                bad.append(f"{key} {meth}: supplementary Table S4 {g}, data {wt}")

    # ---- caption figures
    cap = _rq4_caption(s)
    cap4 = _rq4_s4_caption(sup)
    if cap4 is None:
        bad.append("supplementary Table S4 caption not found")
    if cap is None:
        bad.append("tab:rq4 caption not found")
    else:
        eq = ki["equivalence_tests"]["A_rollup_vs_B_raw"]
        if eq["equivalent"] or abs(eq["margin"] - 0.05) > 1e-9:
            bad.append("the caption says the +-0.05 equivalence test rejects equivalence; the data disagrees")
        gaps = [v["mean_diff"] for v in pf["dense"]["bm25_minus_dense_new_probes"].values()]
        ov = ki["arm_f_overlap"]
        ovn = pf["overlap_new_probes"]
        mgap = -sig["A_rollup_vs_D_bm25_rollup"]["effect_size"]
        # the prose's 'a larger gap than any between artifacts on these probes'
        for col in (0, 1):
            vals = [spec[a][col]["recall@10"] for a in spec]
            if not max(vals) - min(vals) < mgap:
                bad.append("the 13.4 pp method gap is not larger than every artifact gap on the transcript probes")
        if not max(gaps) <= min(pdn["D_bm25_rollup_minus_E_bm25_raw"]["mean_diff"], pdd["A_dense_rollup_minus_B_dense_raw"]["mean_diff"]):
            bad.append("the caption says the rollup-text method gaps are no larger than rollup - raw; the data disagrees")
        want_cap = [
            ("550 sessions", rf"over the ({ki['num_sessions']}) sessions that have a rollup"),
            ("1,087 transcript probes", r"from the transcripts \((" + f"{ki['num_probes']:,}".replace(",", r"\{,\}") + r"), about 2 per session\)"),
            ("1,100 rollup-text probes", r"from the rollup text \((" + f"{pf['n_probes']:,}".replace(",", r"\{,\}") + r")\)"),
            ("pointers to supplementary Tables S4 and S3", r"(Differences with 95\\% intervals: supplementary Table S4); (recall@1: Table S3)\.$"),
        ]
        for what, rx in want_cap:
            if not re.search(rx, cap):
                bad.append(f"tab:rq4 caption: {what} does not match the data (expected /{rx}/)")
        want_cap4 = [
            ("method gap 13.4 pp", rf"BM25 over rollups beats dense retrieval over rollups by ({_pp(mgap)}) pp;"),
            ("550 sessions and both probe counts", rf"({ki['num_sessions']}) sessions, (" + f"{ki['num_probes']:,}".replace(",", r"\{,\}") + r") transcript-written probes and (" + f"{pf['n_probes']:,}".replace(",", r"\{,\}") + r") rollup-text probes"),
            ("bootstrap intervals and the no-interval flag", r"Intervals are session-clustered paired bootstrap 95\\% CIs; \$\^\{\*\}\$no interval computed"),
            # round 7 (2026-10-14): the test figures below left the tab:rq4 caption verbatim for this one; same regexes, same data
            ("equivalence margin and 90% CI, pp", rf"specified in advance at \$\\pm ({100 * eq['margin']:.0f})\$ pp, rejects equivalence \(90\\% CI \[\$({100 * eq['ci_lower']:.1f})\$, \$({100 * eq['ci_upper']:.1f})\$\]\)"),
            ("McNemar p", rf"gives \$p = ({sig['D_bm25_rollup_vs_E_bm25_raw']['p_value']:.2f})\$"),
            ("overlaps and spread", rf"overlap is ({ov['mean_transcript_overlap']:.3f}) with the transcript, ({ov['mean_no_tool_overlap']:.3f}) with output-stripped and ({ov['mean_rollup_overlap']:.3f}) with the rollup, a spread of ({ki['fairness']['difference']:.3f})"),
            ("rollup-text method gaps, pp", rf"run ({_pp(min(gaps))}) to ({_pp(max(gaps))}) pp, no larger than"),
            ("rollup-text overlap spread and recall move, pp", rf"overlap spread is only ({max(ovn.values()) - min(ovn.values()):.3f}) while recall moves by about ({100 * pf['observed']['D_minus_E']:.0f}) pp"),
        ]
        for what, rx in want_cap4:
            if cap4 is not None and not re.search(rx, cap4):
                bad.append(f"supplementary Table S4 caption: {what} does not match the data (expected /{rx}/)")
    # ---- contribution 4 in the introduction (round 7, 2026-10-14: was '0.134 recall@10', now the same gap in pp, derived like the RQ4 prose)
    if not re.search(r"over rollups beats dense retrieval over the same\s+index by\s+" + re.escape(_pp(mgap)) + r"\s+percentage\s+points\s+of\s+recall@10", s):
        bad.append(f"introduction: the retrieval contribution's method gap {_pp(mgap)} percentage points is missing or does not match the data")
    # ---- the one number left in the RQ4 prose
    mp = re.search(r"\\end\{table\}\n\nRetrieval decides which artifact(.*?)\n\n", s, re.S)
    if not mp or not re.search(r"finds\s+" + re.escape(_pp(mgap)) + r"\s+percentage\s+points\s+more\s+of\s+the\s+probes\s+than\s+dense", mp[1]):
        bad.append(f"RQ4 prose: the method gap {_pp(mgap)} pp is missing or does not match the data")
    if not mp or "supplementary Table S4" not in mp[1]:
        bad.append("RQ4 prose: the pointer to supplementary Table S4 for the intervals is missing")
    return (False, "; ".join(bad)) if bad else (True, "16 R@10 cells (%) and the tab:rq4 caption figures, 12 difference cells (pp) and the caption figures of supplementary Table S4, 16 R@1 cells in supplementary Table S3, and the 13.4 pp method gap in the RQ4 prose and the introduction recompute from the retrieval result files")


def check_position_table(s):
    """tab:position, recall by question window, against late-window/control/results.json
    (the 27-session early/middle/late control) and late-window/results.json (the paper's
    25 sessions). Added 2026-10-14 (round 4): the table had no pin.

    Grades every cell: the three windows' rollup and prose-only recall and their
    prose-only minus rollup gap with its clustered 95% CI, the early-minus-late row
    (rollup, prose-only, and the gap's change with its CI), the output-stripped late
    row, and the reference row on the paper's 25 sessions.
    """
    m = re.search(r"\\label\{tab:position\}(.*?)\\end\{tabular\}", s, re.S)
    if not m:
        return False, "tab:position not found"
    body = m[1]
    num = r"\$?([+-]?[\d.]+)\$?"
    ci = r"\[\$?([+-]?[\d.]+)\$?, \$?([+-]?[\d.]+)\$?\]"
    def row(label, cols):
        rx = r"^" + re.escape(label) + r" & " + cols + r" \\\\$"
        f = re.findall(rx, body, re.M)
        return f[0] if len(f) == 1 else None
    c = _j("late-window/control/results.json")
    d, pr, bw = c["descriptive"], c["paired"], c["between"]
    paper = _j("late-window/results.json")
    def r1(v):
        return f"{round(v + 1e-9, 1):.1f}"
    def sg(v):
        return f"{round(v + 1e-9, 1):+.1f}"
    want, got = {}, {}
    for lab, w in (("Early (first 30{,}000 chars)", "early"), ("Middle (30--60\\%)", "mid"), ("Late (60--100\\%)", "late")):
        g = pr[w]["prose-rollup"]
        want[lab] = (r1(d[w]["rollup"]["recall_pct"]), r1(d[w]["prose"]["recall_pct"]), sg(g["diff_pts"]),
                     r1(g["ci95"][0]), r1(g["ci95"][1]))
        got[lab] = row(lab, num + " & " + num + " & " + num + " " + ci)
    el = bw["prose-rollup_early_minus_late"]
    want["Early $-$ late"] = (sg(bw["rollup_early_minus_late"]["b_minus_a_pts"]), sg(bw["prose_early_minus_late"]["b_minus_a_pts"]),
                             sg(el["b_minus_a_pts"]), sg(el["ci95"][0]), sg(el["ci95"][1]))
    got["Early $-$ late"] = row("Early $-$ late", num + " & " + num + " & " + num + " " + ci)
    sl = pr["late"]["stripped-rollup"]
    want["Late, output-stripped"] = (sg(sl["diff_pts"]), r1(sl["ci95"][0]), r1(sl["ci95"][1]))
    got["Late, output-stripped"] = row("Late, output-stripped", r"& & " + num + " " + ci)
    pe = paper["paired"]["early"]["prose-rollup"]
    want["Paper's 25 sessions, early"] = (r1(paper["descriptive"]["early"]["rollup"]["recall_pct"]),
                                          r1(paper["descriptive"]["early"]["prose"]["recall_pct"]),
                                          sg(pe["diff_pts"]), r1(pe["ci95"][0]), r1(pe["ci95"][1]))
    got["Paper's 25 sessions, early"] = row("Paper's 25 sessions, early", num + " & " + num + " & " + num + " " + ci)
    bad = []
    for k, w in want.items():
        g = got[k]
        if g is None:
            bad.append(f"row '{k}' not found exactly once")
            continue
        # unsigned cells in the table read as positive: normalise both sides to signed floats
        if tuple(float(x) for x in g) != tuple(float(x) for x in w):
            bad.append(f"{k}: table {g}, data {w}")
    # the caption's sample sizes and the claim that every window's gap is at the bootstrap floor
    if not all(pr[w]["prose-rollup"]["p"] <= 0.00021 for w in ("early", "mid", "late")):
        bad.append("the caption says every window's gap has p at the bootstrap floor; the data disagrees")
    if not (c["n_sessions"] == 27 and all(d[w]["rollup"]["n"] == 135 for w in ("early", "mid", "late"))
            and sl["sessions"] == 17):
        bad.append("the caption's 27 sessions, 135 questions per window or 17 output-stripped sessions disagree with the data")
    # re-anchored 2026-10-15 (table notes): the caption keeps "the same 27 longer sessions";
    # 135 per window, the bootstrap floor and the 17 sessions moved verbatim to the note under the
    # tabular, and "Zero context recovers 0 of 135 in each window" to the RQ3 position prose
    i = s.index("\\label{tab:position}")
    cap = re.sub(r"\s+", " ", s[i - 300:i])
    note = re.sub(r"\s+", " ", _float_note(s, "tab:position") or "")
    for where, txt, frag in (("caption", cap, "on the same 27 longer sessions"),
                             ("note", note, "135 questions per window"), ("note", note, "for 17 of the sessions"),
                             ("note", note, "Every window's gap has $p$ at the bootstrap floor")):
        if frag not in txt:
            bad.append(f"tab:position {where} no longer says '{frag}'")
    zc = {(d[w]["noop"]["correct"] + d[w]["noop"]["partial"], d[w]["noop"]["n"]) for w in ("early", "mid", "late")}
    if len(zc) != 1 or f"Zero context recovers {next(iter(zc))[0]} of {next(iter(zc))[1]} in each window." not in s:
        bad.append(f"RQ3 position prose: 'Zero context recovers X of N in each window' missing or not {sorted(zc)} in the data")
    return (False, "; ".join(bad)) if bad else (True, "6 rows (3 windows, early - late, output-stripped late, the paper's 25) and the caption's n recompute from late-window/control and late-window results")


def check_failmode_table(s):
    """tab:failmode, the three-set wrong-rate and decline table, against its sources.

    Added 2026-10-14. The review moved the design, held-out and Replication 1
    wrong-of-attempted rates, counts, clustered CIs and decline rates out of five
    prose sites (abstract, introduction, contribution 2, RQ3 retention, guidance)
    into this table; the pins that graded those sentences are retired and this
    check grades the same numbers in their one remaining place.
    """
    m = re.search(r"\\label\{tab:failmode\}(.*?)\\end\{tabular\}", s, re.S)
    if not m:
        return False, "tab:failmode not found"
    rows = {r[0]: r[1:] for r in re.findall(
        r"^(Design|Held-out|Replication 1) & (\d+) & ([\d.]+) \((\d+)/(\d+)\) & ([\d.]+) \((\d+)/(\d+)\) & "
        r"\$\+([\d.]+)\$ \[([\d.]+), ([\d.]+)\] & ([\d.]+) & ([\d.]+) \\\\$", m[1], re.M)}
    ro, so = _err_arm("rollup_uniform"), _err_arm("output_stripped")
    h = _held()
    rr = _j("replication-rollup/results.json")
    ra, rs = rr["arms_on_covered"]["rollup"], rr["arms_on_covered"]["output_stripped"]
    want = {
        "Design": (str(ro["n"]), _f1(ro["wrong_given_attempted_pct"]), str(ro["wrong"]), str(ro["attempted"]),
                   _f1(so["wrong_given_attempted_pct"]), str(so["wrong"]), str(so["attempted"]),
                   *_err_gap(), *absent_pair()),
        "Held-out": (str(h["rollup"]["n"]), held_vals()[0], str(h["rollup"]["wrong"]),
                     str(h["rollup"]["attempted"]), held_vals()[1], str(h["output_stripped"]["wrong"]),
                     str(h["output_stripped"]["attempted"]), *held_vals()[2:7]),
        "Replication 1": (str(ra["n"]), _f1(ra["wrong_given_attempted_pct"]), str(ra["wrong"]), str(ra["attempted"]),
                          _f1(rs["wrong_given_attempted_pct"]), str(rs["wrong"]), str(rs["attempted"]),
                          *rep_gap(), _f1(ra["absent_pct"]), _f1(rs["absent_pct"])),
    }
    if set(rows) != set(want):
        return False, f"parsed rows {sorted(rows)}, expected {sorted(want)}"
    bad = [f"{k}: table {rows[k]}, data {want[k]}" for k in want if tuple(rows[k]) != tuple(want[k])]
    ir = _j("independent-regrade/results.json")["paired_differences"]["rollup_minus_stripped__heldout"]["wrong_given_attempted"]["indep"]
    cap = f"the held-out difference is $+{_f1(ir['diff_pts'])}$ [{_f1(ir['ci95_pts'][0])}, {_f1(ir['ci95_pts'][1])}] (supplementary Table S1, row 11)"
    # re-anchored 2026-10-15 (table notes): the Opus regrade sentence left the tab:failmode caption
    # verbatim for the RQ3 'two fail in different ways' paragraph that cites the table
    mp = re.search(r"\\textbf\{The two fail in different ways\.\}(.*?)\n\n", s, re.S)
    if not mp or "Under an exploratory single-draw Opus 5.5 regrade " + cap not in re.sub(r"\s+", " ", mp[1]):
        bad.append(f"RQ3 failure-mode prose does not carry the Opus regrade as '{cap}'")
    note = re.sub(r"\s+", " ", _float_note(s, "tab:failmode") or "")
    if f"Replication 1: the {ra['n']} of its {_j('replication/results.json')['n_questions']} questions that have a rollup" not in note:
        bad.append("tab:failmode note no longer says Replication 1 is the questions that have a rollup")
    return (False, "; ".join(bad)) if bad else (True, "3 rows, the Replication 1 n in the note and the Opus regrade in the RQ3 prose recompute from the error-rate, held-out, replication and regrade files")


def check_nodigest_table(s):
    """tab:nodigest against fullar-vs-nodigest (design), pooled-estimate (held-out)
    and replication (Replication 1). Added 2026-10-14, when the review replaced the
    no-digest prose with this table."""
    m = re.search(r"\\label\{tab:nodigest\}(.*?)\\end\{tabular\}", s, re.S)
    if not m:
        return False, "tab:nodigest not found"
    rows = {r[0]: r[1:] for r in re.findall(
        r"^(Design\$\^\{\\dagger\}\$|Held-out|Replication 1 \(pre-registered\)) & (\d+) & ([\d.]+) & ([\d.]+) & "
        r"\$([+-][\d.]+)\$ & \[\$([+-][\d.]+)\$, \$([+-][\d.]+)\$\] \\\\$", m[1], re.M)}
    def sg(v):
        return f"{v:+.1f}"
    fa = _j("fullar-vs-nodigest/results.json")["part1_design"]
    fd = fa["pairs"]["nodigest_minus_stripped"]
    nd, st = any_rate("action record (no digest)"), any_rate("output-stripped")
    ph = _j("pooled-estimate/results.json")["per_set"]["heldout"]
    rp = _j("replication/results.json")
    want = {
        "Design$^{\\dagger}$": (str(fa["n"]), _f1(nd.recall), _f1(st.recall), sg(fd["recall_diff_pts"]),
                                sg(fd["ci90_pts"][0]), sg(fd["ci90_pts"][1])),
        "Held-out": ("125", _f1(ph["recall_nodigest_pct"]), _f1(ph["recall_stripped_pct"]), sg(ph["recall_diff_pts"]),
                     sg(ph["clustered_bootstrap"]["recall"]["ci90"][0]), sg(ph["clustered_bootstrap"]["recall"]["ci90"][1])),
        "Replication 1 (pre-registered)": (str(rp["n_questions"]), _f1(rp["point"]["recall_nodigest_pct"]),
                                           _f1(rp["point"]["recall_stripped_pct"]), sg(rp["point"]["recall_diff_pts"]),
                                           sg(rp["primary"]["recall_diff_ci90_session_clustered"][0]),
                                           sg(rp["primary"]["recall_diff_ci90_session_clustered"][1])),
    }
    if set(rows) != set(want):
        return False, f"parsed rows {sorted(rows)}, expected {sorted(want)}"
    bad = [f"{k}: table {rows[k]}, data {want[k]}" for k in want if tuple(rows[k]) != tuple(want[k])]
    if fd["ci90_inside_pm10"] is not True or not rp["primary"]["equivalent_pm10"] or rp["primary"]["equivalent_pm5"]:
        bad.append("the equivalence verdicts in the data no longer match the table's margin and the prose")
    return (False, "; ".join(bad)) if bad else (True, "3 rows recompute from the design, held-out and Replication 1 files")


def _tex_int(n):
    """123456 -> '123{,}456', the paper's thousands separator."""
    return f"{int(n):,}".replace(",", "{,}")


_WORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _signed(v):
    """-1.6 -> '-1.6'; the .tex wraps signed numbers in $...$, regexes capture the bare number."""
    return f"{v:.1f}"


def _cb():
    import json as _j
    return _j.loads((arm_rates.DATA / "cluster_bootstrap.json").read_text())


def _sonnet():
    import json as _j
    return _j.loads((arm_rates.DATA / "rq3_sonnet_rollup_124_summary.json").read_text())


def _haiku_v2():
    """The Haiku length-matched arms (v2 is the reported one, v1 the first, shorter run),
    counted from their graded records on the 120 questions both cover."""
    out = {}
    for k, fn in (("v1", "rq3_sonnet_rollup_124_matched_haiku_graded.jsonl"),
                  ("v2", "rq3_sonnet_rollup_124_matched_haiku_v2_graded.jsonl")):
        out[k] = arm_rates._rate("haiku matched " + k, fn, arm_rates.counts_from_records(fn))
    return out


def check_stored_reports_fresh():
    """cluster_bootstrap.json and rq3_sonnet_rollup_124_summary.json are stored outputs of a
    bootstrap. Every arm in them must carry the recoverable count the graded records give
    now, so a regrade that did not rerun them cannot pass."""
    cb, sr = _cb(), _sonnet()
    names = {"rollup": "rollup", "output-stripped": "output-stripped", "raw": "raw",
             "action record": "action record", "prose-only": "prose-only",
             "call-only": "call-only", "dialogue record": "dialogue record",
             "answer-only": "answer-only", "prompts-only": "prompts-only",
             "zero context": "zero context", "prose matched": "prose cut to rollup length",
             "long sonnet": "sonnet long", "short sonnet": "sonnet short",
             "matched sonnet": "sonnet matched"}
    bad = []
    for k, lab in names.items():
        a, r = cb["arms"][k], any_rate(lab)
        if (a["n"], a["recoverable"]) != (r.n, r.recoverable_n):
            bad.append(f"cluster_bootstrap {k}: {a['recoverable']}/{a['n']} vs records {r.recoverable_n}/{r.n}")
    for k, lab in (("long sonnet rollup", "sonnet long"), ("short sonnet rollup", "sonnet short"),
                   ("matched sonnet rollup", "sonnet matched")):
        a, r = sr["arms"][k], any_rate(lab)
        if (a["n"], a["recoverable"]) != (r.n, r.recoverable_n):
            bad.append(f"sonnet summary {k}: {a['recoverable']}/{a['n']} vs records {r.recoverable_n}/{r.n}")
    for k, key in (("v1", "matched haiku rollup"), ("v2", "matched haiku v2 rollup")):
        a, r = sr["arms"][key], _haiku_v2()[k]
        if (a["n"], a["recoverable"], a["wrong"]) != (r.n, r.recoverable_n, r.n_wrong):
            bad.append(f"sonnet summary {key}: {a['recoverable']}/{a['n']} vs records {r.recoverable_n}/{r.n}")
    if bad:
        return False, "; ".join(bad) + " -- rerun cluster_bootstrap.py and rq3_sonnet_rollup_124.py --analyse"
    return True, "cluster_bootstrap.json and the Sonnet summary agree with the graded records"


def _cell_num(x):
    return float(x.replace("$", "").replace("\\,", "").replace("{,}", "").strip())


def check_paired_table(s, sup=""):
    """Every row of tab:paired against the stored bootstrap, which is itself checked
    against the records by check_stored_reports_fresh().

    Re-anchored 2026-10-14: the discordant counts moved from the caption into a
    column (A-only/B-only, now graded for every row including 2, 3, 6 and 11, which
    the caption never carried), and rows 14 and 15 add the held-out Sonnet repeat
    from sonnet1k-heldout/results.json.

    Re-anchored 2026-10-14 (round 4): the $p$ and Disc. columns left tab:paired for
    supplementary Table S2 (tab:paired-disc in paper/supplement/sensitivity-checks.tex).
    n, difference and CI are graded in main.tex; p and the discordant counts are
    graded, row for row and with the same rules, in the supplement.
    """
    cb = _cb()["paired_differences"]
    spec = {1: "output-stripped - rollup (124)", 2: "prose-only - call-only (123 shared)",
            3: "prose matched - rollup (124)", 4: "output-stripped - raw (89)",
            5: "action record - output-stripped (124)", 6: "read-budget chunk - rollup summary (89)",
            7: "long sonnet - rollup (124)", 8: "long sonnet - output-stripped (124)",
            9: "matched sonnet - rollup (124)", 10: "matched sonnet - prose matched (124)",
            11: "long sonnet - prose-only (124)"}
    m = re.search(r"\\label\{tab:paired\}(.*?)\\end\{tabular\}", s, re.S)
    if not m:
        return False, "tab:paired not found"
    # rows 12-13 are on the 120 questions the Haiku arm covers; their source is the Sonnet/Haiku
    # summary, not cluster_bootstrap.json, and they carry the same fields under other names
    sp = _sonnet()["paired"]
    extra = {12: "matched haiku v2 - production rollup", 13: "matched haiku v2 - matched sonnet"}
    ho = _j("sonnet1k-heldout/results.json")["paired"]
    held = {14: "sonnet1k_minus_cut", 15: "sonnet1k_minus_production_note"}
    main_rows = re.findall(r"^(\d+) & [^&]+ & (\d+) & (.+?) \[(.+?), (.+?)\] \\\\$", m[1], re.M)
    if len(main_rows) != len(spec) + len(extra) + len(held):
        return False, f"parsed {len(main_rows)} rows of tab:paired, expected {len(spec) + len(extra) + len(held)}"
    ms = re.search(r"\\label\{tab:paired-disc\}(.*?)\\end\{tabular\}", sup, re.S)
    if not ms:
        return False, "supplementary Table S2 (tab:paired-disc) not found in the supplement file"
    sup_rows = {int(r[0]): r[1:] for r in re.findall(r"^(\d+) & [^&]+ & (.+?) & (\d+)/(\d+) \\\\$", ms[1], re.M)}
    if sorted(sup_rows) != sorted(int(r[0]) for r in main_rows):
        return False, f"supplementary Table S2 rows {sorted(sup_rows)} do not match tab:paired rows {[int(r[0]) for r in main_rows]}"
    rows = [(i, n, d, lo, hi, *sup_rows[int(i)]) for i, n, d, lo, hi in main_rows]
    bad = []
    for i, n, d, lo, hi, p, da, db in rows:
        i = int(i)
        if i in held:
            e = ho[held[i]]
            v = {"n_shared": e["n_paired"], "diff_pts": e["recall"]["diff_pts"],
                 "cluster95_pts": e["recall"]["clustered_bootstrap_ci95_pts"], "cluster_boot_p_two_sided": None,
                 "discordant_a_only_b_only": e["recall_discordant_a_only_b_only"]}
        elif i in extra:
            e = sp[extra[i]]
            v = {"n_shared": e["n"], "diff_pts": e["diff_pts"], "cluster95_pts": e["cluster95_pts"],
                 "cluster_boot_p_two_sided": e["boot_p"], "discordant_a_only_b_only": e["discordant_new_only_other_only"]}
        else:
            v = cb[spec[i]]
        got = (int(n), _cell_num(d), _cell_num(lo), _cell_num(hi))
        want = (v["n_shared"], round(v["diff_pts"] + 1e-9, 1), round(v["cluster95_pts"][0] + 1e-9, 1),
                round(v["cluster95_pts"][1] + 1e-9, 1))
        if got != want:
            bad.append(f"row {i}: table {got}, data {want}")
        if [int(da), int(db)] != list(v["discordant_a_only_b_only"]):
            bad.append(f"row {i}: discordant {da}/{db}, data {v['discordant_a_only_b_only']}")
        pb = v["cluster_boot_p_two_sided"]
        if pb is None:
            if p.strip() != "--":
                bad.append(f"row {i}: no clustered p exists for it, table says {p.strip()}")
        elif "approx" in p:
            if pb > 0.00021:
                bad.append(f"row {i}: table says p ~ floor, data gives {pb}")
        elif p.strip() != "--":
            if p.strip() not in (f"{pb:.2f}", f"{pb:.3f}", f"{pb:.4f}"):
                bad.append(f"row {i}: table p {p.strip()}, data {pb}")
    t90 = _cb()["tost_output_stripped_vs_raw"]["cluster90_pts"]
    if f"[{t90[0]:.1f}, +{t90[1]:.1f}]" not in s.replace("$", ""):
        bad.append(f"caption TOST CI is not [{t90[0]:.1f}, +{t90[1]:.1f}]")
    return (False, "; ".join(bad)) if bad else (True, f"{len(rows)} rows of tab:paired, their p and discordant counts in supplementary Table S2, and the TOST CI recompute from the stored bootstraps")


def check_length_table(s):
    """The length and generator rows of the merged tab:frontier (which absorbed
    tab:length on 2026-10-14) against the graded records.

    Same sources and the same strictness as the tab:length check it replaces:
    recall, wrong and tokens per row, the model-call flag now read from the
    Generator column, and tokens ascending across the comparison block. The
    production rollup is now printed at 929 (the 124-question median, as the
    frontier always printed it); 932, the 25-note median that set the length
    target, is stated in the RQ3 length prose (the caption until round 7) and pinned below.
    """
    blocks = _fb.table_blocks(s)
    if not blocks:
        return False, "merged tab:frontier not found"
    comp = blocks[0]
    sr = _sonnet()["arms"]
    src = {("production prompt", "Sonnet"): ("sonnet short", sr["short sonnet rollup"]["median_rollup_tokens"]),
           ("rollup", "Haiku"): ("rollup", 929),
           ("prose-cut", "none"): ("prose cut to rollup length", 933),
           ("length-matched prompt", "Sonnet"): ("sonnet matched", sr["matched sonnet rollup"]["median_rollup_tokens"]),
           ("prose-only", "none"): ("prose-only", 3055),
           ("long rollup", "Sonnet"): ("sonnet long", sr["long sonnet rollup"]["median_rollup_tokens"]),
           ("output-stripped", "none"): ("output-stripped", 8880),
           ("length-matched prompt", "Haiku"): ("haiku matched v2",
                                                int(sr["matched haiku v2 rollup"]["median_rollup_tokens"] + 0.5))}
    cap = _j("cap400/summary.json")["arms"]
    cap_src = {("prose-1500", "none"): "head1500:g0:d1", ("400-token prompt", "Haiku"): "haiku400:g1:d1",
               ("400-token prompt", "Sonnet"): "sonnet400:g1:d1"}
    byk = {}
    for row in comp:
        byk.setdefault((row[0], row[1]), []).append(row)
    bad = []
    for k in list(src) + list(cap_src):
        if len(byk.get(k, [])) != 1:
            bad.append(f"{k} appears {len(byk.get(k, []))} times")
    if bad:
        return False, "; ".join(bad)
    for k, c_ in cap_src.items():
        (_n, _g, n, tok, rec, wr, _d), = byk[k]
        c = cap[c_]
        if (f"{rec:.1f}", f"{wr:.1f}") != (f"{c['pct']:.1f}", f"{c['wrong_of_attempted_pct']:.1f}"):
            bad.append(f"{k}: table {rec}/{wr}, cap400 summary {c['pct']:.1f}/{c['wrong_of_attempted_pct']:.1f}")
        if tok != int(c["median_context_cl100k_per_question"]):
            bad.append(f"{k}: tokens {tok}, data {c['median_context_cl100k_per_question']}")
        if n != c.get("n", n):
            bad.append(f"{k}: n {n}, cap400 {c.get('n')}")
    for k, (arm, want_tok) in src.items():
        (_n, gen, n, tok, rec, wr, _d), = byk[k]
        r = _haiku_v2()["v2"] if arm == "haiku matched v2" else any_rate(arm)
        if (f"{rec:.1f}", f"{wr:.1f}", n) != (f"{r.recall:.1f}", f"{r.wrong_rate:.1f}", r.n):
            bad.append(f"{k}: table {rec}/{wr} n={n}, records {r.recall:.1f}/{r.wrong_rate:.1f} n={r.n}")
        if tok != int(want_tok):
            bad.append(f"{k}: tokens {tok}, data {want_tok}")
    order = [row[3] for row in comp]
    if order != sorted(order):
        bad.append("comparison rows are not ascending by tokens")
    # Round 7 (cain-review-1014): the 929/932 reconciliation left the caption for the
    # RQ3 length prose; 929 there must equal the table's rollup token cell.
    m = re.search(W(r"932, the median of the 25 design-set notes, set the length-matched target, while ([\d{},]+) is the production rollup's median over the (\d+) questions"), s)
    if not m:
        bad.append("the RQ3 length prose no longer reconciles 929 with the 932 length target")
    else:
        (_n, _g, rn, rtok, *_r), = byk[("rollup", "Haiku")]
        if (int(m[1].replace("{,}", "")), int(m[2])) != (rtok, rn):
            bad.append(f"the prose says {m[1]} over {m[2]} questions; the table's rollup row says {rtok} over {rn}")
    if provenance_medians()[1] != "932":
        bad.append(f"the design-note median is {provenance_medians()[1]}, not the 932 the caption states")
    return (False, "; ".join(bad)) if bad else (True, f"{len(src) + len(cap_src)} length/generator rows recompute from the records, tokens ascending")


SENS_CHECKS = None   # sensitivity rows are checked in data_backed_prose(): their text is prose


def data_backed_prose():
    """(label, regex, expected) per sentence in main.tex that states a measurement.

    Each regex must match main.tex EXACTLY ONCE -- zero matches means the
    sentence was deleted or reworded past its anchor, two means the figure now
    lives in two places and only one of them is being graded -- and every number
    it captures must equal the value recomputed from data/paper-rollups/.

    Anchored on the surrounding words rather than on the digits, deliberately:
    a rewrite that keeps the claim keeps the check, and a rewrite that drops the
    claim fails it.

    Rewritten 2026-10-09 for the regrade: one grading stage, so the corrected /
    uncorrected pairs and the correction-lineage entries are gone, and every
    sentence that cites a regraded number is anchored here. Token costs (3{,}082,
    6{,}329, 8{,}880, 929, 6{,}393) stay pinned by the MUST list.
    """
    comp = composition()
    cov = raw_arm_coverage()
    R = rq3_renumber.compute()
    cb = _cb()
    sr = _sonnet()
    pd_ = cb["paired_differences"]
    prose, call = uncorrected("prose-only"), uncorrected("call-only")
    strip, roll = uncorrected("output-stripped"), uncorrected("rollup")
    dial, action = uncorrected("dialogue record"), uncorrected("action record")
    raw = uncorrected("raw")
    zero = uncorrected("zero context")
    cut = any_rate("prose cut to rollup length")
    s_long, s_short, s_match = any_rate("sonnet long"), any_rate("sonnet short"), any_rate("sonnet matched")
    srl, srs, srm = (sr["arms"][k] for k in ("long sonnet rollup", "short sonnet rollup", "matched sonnet rollup"))
    spd = sr["paired"]
    c89 = R["common89"]
    orig = R["rollup_349"]
    ho = R["heldout"]
    jl = R["judge_leniency"]

    def p(v):
        return f"{v:.1f}"

    def d1(key):
        return p(round(pd_[key]["diff_pts"] + 1e-9, 1))

    def ci1(key, i):
        return p(round(pd_[key]["cluster95_pts"][i] + 1e-9, 1))

    # movement of the headline arms against the grades first published
    moved = [abs(uncorrected(a).recall - published(a).recall) for a in UNCORRECTED_SOURCES]
    moved = [m_ for m_ in moved if m_ > 0.05]
    old_corr_wrong = _earlier_correction_wrong_rate()

    E = []
    # S: entries graded against the supplement file (paper/supplement/sensitivity-checks.tex), not main.tex.
    # Added 2026-10-14 (round 4) when tab:sens left main.tex: every row the table carried is still graded there.
    S = []
    E.append((
        "RQ1 heading: tool I/O share and prose share",
        W(r"Tool input and output make up ([\d.]+)\\% of a transcript's text and assistant prose ([\d.]+)\\%"),
        [f"{comp['tool_io']:.2f}", p(comp['assistant_text'])]))
    E.append((
        "contribution 3: composition pooled over the substantive sessions",
        W(r"Tool input and output make up ([\d.]+)\\% of a transcript's text and assistant prose ([\d.]+)\\%\}.*?separately over the ([\d{},]+) substantive parent sessions.*?The ([\d.]+)\\% reasoning share is therefore"),
        [f"{comp['tool_io']:.2f}", p(comp['assistant_text']), _tex_int(comp['n_substantive']), f"{comp['thinking']:.2f}"]))
    E.append((
        "method: substantive subset size and its exclusions",
        W(r"Composition claims cover the ([\d{},]+) substantive sessions, and the other ([\d{},]+) are excluded"),
        [_tex_int(comp['n_substantive']), _tex_int(comp['n_trivial_excluded'])]))
    # ---- abstract (rewritten 2026-10-10 to the human's higher-level abstract, which
    # carries only the wrong-rate pair and the Sonnet-over-cut lift at ~1,000 tokens)
    s1k = sonnet1k()
    # ---- abstract (2026-10-14, second pass: the human asked for one number per claim)
    _rq1 = _j("rq1_corrected_summary.json")
    E.append((
        "abstract: transcripts studied (added 2026-10-14)",
        W(r"We study ([\d{},]+) Claude Code transcripts from one developer"),
        [_tex_int(_rq1["n_transcripts_parsed"])]))
    E.append((
        "abstract: the rule cut against the summary, recall (added 2026-10-14)",
        W(r"keeps the conversation and the tool calls recovers ([\d.]+)\\% of our questions against ([\d.]+)\\% for the summary"),
        [p(strip.recall), p(roll.recall)]))
    _rp_ab = _j("replication/results.json")
    if not _rp_ab["primary"]["equivalent_pm10"]:
        raise ValueError("the abstract says the smaller rule matched in the replication; the data says it did not")
    E.append((
        "abstract: the pre-registered replication's answered questions (added 2026-10-14)",
        W(r"matched it in a pre-registered replication on (\d+) fresh questions"),
        [str(_rp_ab["n_questions"])]))
    E.append((
        "corpus: transcripts parsed, moved out of tab:corpus into the corpus paragraph (added 2026-10-14)",
        W(r"one machine, ([\d{},]+) parsed transcripts"),
        [_tex_int(_rq1["n_transcripts_parsed"])]))
    # ---- introduction and contributions
    E.append((
        "contribution 1: the gap (re-anchored 2026-10-14; the long Sonnet arm is graded by the RQ3 length pins)" + RA,
        W(r"deleting tool output beats our generated rollup by ([\d.]+) percentage points \(pp\) of recall"),
        [d1("output-stripped - rollup (124)")]))
    E.append((
        "RQ3 retention: output-stripped's design-set wrong rate, its count and interval, against held-out (re-anchored 2026-10-14; the rates and gaps are graded in tab:failmode)",
        W(r"The output-stripped transcript's design-set ([\d.]+)\\% is (\d+) of (\d+) attempts, with 95\\% interval \[([\d.]+), ([\d.]+)\], against ([\d.]+)\\% held-out"),
        [p(strip.wrong_rate), str(strip_wrong_n()), str(strip_att_n())] + [p(x) for x in strip_wrong_ci()] + [held_vals()[1]]))
    # ---- method
    E.append((
        "method: the grader revision's published -> revised movement, and the earlier correction it supersedes" + RA + RA5,
        W(r"recall moves by ([\d.]+) to ([\d.]+) pp for the artifacts that moved: output-stripped from ([\d.]+)\\% to ([\d.]+)\\%, raw from ([\d.]+)\\% to ([\d.]+)\\%, and rollup from ([\d.]+)\\% to ([\d.]+)\\%\. The rollup's wrong-answer rate moves from ([\d.]+)\\% to ([\d.]+)\\%\. This supersedes the ([\d.]+)\\%"),
        [p(min(moved)), p(max(moved)), p(published("output-stripped").recall), p(strip.recall),
         p(published("raw").recall), p(raw.recall), p(published("rollup").recall), p(roll.recall),
         p(published("rollup").wrong_rate), p(roll.wrong_rate), p(old_corr_wrong)]))
    E.append((
        "method: the uniform-rollup variant, and the original pooled rollup arm regraded",
        W(r"It recovers ([\d.]+)\\% on the same (\d+) questions, where prose-only recovers.*?the uniform-rollup variant planned in advance, which recovers ([\d.]+)\\% \[([\d.]+), ([\d.]+)\]\..*?which recovers ([\d.]+)\\% \[([\d.]+), ([\d.]+)\] on the same questions"),
        [p(dial.recall), str(roll.n), p(roll.recall), p(roll.recall_ci[0]), p(roll.recall_ci[1]),
         p(orig["frozen"]["recall"]), p(orig["frozen"]["ci"][0]), p(orig["frozen"]["ci"][1])]))
    E.append((
        "method: call-only with its context size",
        W(r"Call-only recovers ([\d.]+)\\% at ([\d{},]+) median context characters, more than the dialogue record's ([\d{},]+)\. A true zero-context control recovers ([\d.]+)\\%, with zero hallucinations"),
        [p(call.recall), _tex_int(median_context_chars('call-only')),
         _tex_int(median_context_chars('dialogue record')), p(zero.recall)]))
    E.append((
        "method: the prose-only against dialogue record gap, recomputed" + RA,
        W(r"the ([\d.]+)-pp gap of Section~\\ref\{sec:rq3-retention\} \(([\d.]+)\\% against ([\d.]+)\\%\)"),
        [p(arm_rates.r1(prose.recall - dial.recall)), p(prose.recall), p(dial.recall)]))
    E.append((
        "method: raw's exclusions, and what it is left covering",
        W(r"Raw excludes (\d+) of (\d+) questions from 7 sessions\. Of these, (\d+) exceed the judge's context window and 2 failed their API calls\. Raw therefore covers (\d+) questions"),
        [str(cov['dropped']), str(cov['full']),
         str(cov['n_over_context_sessions']), str(cov['kept'])]))
    deffs = [v["design"]["design_effect"] for v in cb["arms"].values() if v["design"]["design_effect"]]
    E.append((
        "method: the range of design effects and the effective sample it implies" + RA,
        W(r"Design effects, the factor by which this nesting inflates the variance, run ([\d.]+) to ([\d.]+), so the effective sample is (\d+) to (\d+)\\% of the question count"),
        [p(min(deffs)), p(max(deffs)), str(round(100 / max(deffs))), str(round(100 / min(deffs)))]))
    # ---- RQ3 retention
    E.append((
        "fig:retention caption: the reduced n (re-anchored 2026-10-14, round 4: the caption was shortened and no longer"
        " restates the 90.3% / 46.3% contrast; 90.3% is graded by the abstract and conclusion pins and 46.3% by"
        " 'method: call-only with its context size')",
        W(r"\$\^\{\*\}\$Reduced \$n\$: call-only (\d+), raw (\d+)\."),
        [str(call.n), str(raw.n)]))
    _m = cb["tost_prose_matched_vs_rollup"]
    if _m["equivalent_pm10"]:
        raise ValueError("matched-length TOST now passes at +-10; main.tex says it fails")
    E.append((
        "RQ3: matched-length prose cut against the rollup, its 90% CI and the equivalence test" + RA + "; the 95% CI [-16.8, 13.0] left the prose and is graded in tab:paired row 3 by check_paired_table()",
        W(r"the cut recovers ([\d.]+)\\% \[([\d.]+), ([\d.]+)\] and the rollup ([\d.]+)\\%\. The paired difference is \$(-[\d.]+)\$ pp \(Table~\\ref\{tab:paired\}, row 3\), statistically indistinguishable from zero but not equivalent: the 90\\% CI \$\[(-[\d.]+), ([\d.]+)\]\$ fails the equivalence test"),
        [p(cut.recall), p(cut.recall_ci[0]), p(cut.recall_ci[1]), p(roll.recall),
         p(round(_m["diff_pts"] + 1e-9, 1)),
         p(_m["cluster90_pts"][0]), p(_m["cluster90_pts"][1])]))
    if not (_m["cluster95_pts"][0] < 0 < _m["cluster95_pts"][1]):
        raise ValueError("the paper calls the matched-length cut statistically indistinguishable from the rollup; its 95% CI excludes zero")
    E.append((
        "RQ3: the dialogue record against prose-only on the same questions",
        W(r"The dialogue record keeps only the last answer per exchange.*?It recovers ([\d.]+)\\% on the same (\d+) questions, where prose-only recovers ([\d.]+)\\% at lower cost"),
        [p(dial.recall), str(prose.n), p(prose.recall)]))
    E.append((
        "RQ3: prose-only against call-only, and output-stripped against the rollup" + RA,
        W(r"The output-stripped transcript beats the rollup by ([\d.]+) pp.*?Prose-only beats call-only by ([\d.]+) pp on the (\d+) questions both"),
        [d1("output-stripped - rollup (124)"), d1("prose-only - call-only (123 shared)"), str(call.n)]))
    t90 = cb["tost_output_stripped_vs_raw"]["cluster90_pts"]
    E.append((
        "RQ3: output-stripped against raw on the 89, its TOST, and the rollup on the same 89" + RA + " (re-anchored 2026-10-15, positive-statement pass: the 'not +-5' / 'not significant' wording left the sentence; values unchanged)",
        W(r"the output-stripped transcript recovers ([\d.]+)\\% and raw ([\d.]+)\\%, a difference of \$\+([\d.]+)\$ pp.*?\(90\\% CI \$\[(-[\d.]+), \+([\d.]+)\]\$\)\..*?The rollup recovers ([\d.]+)\\% on the same 89"),
        [p(c89["output-stripped"]["new"]["recall"]), p(raw.recall), d1("output-stripped - raw (89)"),
         p(t90[0]), p(t90[1]), p(c89["rollup"]["new"]["recall"])]))
    nt = R["no_tool_exclusion"]
    rf = R["raw_failures"]
    r89 = c89["rollup"]["new"]
    E.append((
        "RQ3: raw's failures are all refusals, the rollup's wrong answers on the same 89",
        W(r"All (\d+) questions raw fails to recover are refusals: (\d+) come from its largest inputs.*?On the same (\d+) questions the rollup answers (\d+) of its (\d+) attempts wrongly \(([\d.]+)\\%\)"),
        [str(len(rf)), str(sum(f["first_line"] == "NOT ANSWERABLE" for f in rf)), str(r89["n"]), str(r89["wrong"]),
         str(r89["attempted"]), p(r89["wrong_rate"])]))
    E.append((
        "RQ3: raw's hallucination count is zero because none of its failures is graded wrong",
        r"None is graded \\emph\{wrong\}, so raw's hallucination count is (\w+)",
        ["zero" if R["raw_wrong"] == 0 else _WORD[R["raw_wrong"]]]))
    E.append((
        "RQ3: raw's coverage of the question set",
        W(r"The excluded sessions are the largest, and ([\d.]+)\\% of the question set survives"),
        [p(cov['coverage_pct'])]))
    E.append((
        "RQ3: output-stripped on the 89 against on the other 35",
        r"scores ([\d.]+)\\% on the 89 against ([\d.]+)\\% on the 35",
        [p(c89["output-stripped"]["new"]["recall"]), p(R["stripped_on_other_35"]["recall"])]))
    # ---- method: zero context (round 7, cain-review-1014: left tab:frontier for the prose;
    # frontier_bands.zero_point() prices this same sentence)
    E.append((
        "method: zero context reads nothing, recovers nothing and attempts nothing (moved from tab:frontier, round 7)",
        W(r"Zero context reads (\d+) tokens and recovers ([\d.]+)\\% of the (\d+) questions, attempting (none)\."),
        ["0", p(zero.recall), str(zero.n), "none" if zero.attempted_n == 0 else str(zero.attempted_n)]))
    # ---- RQ3 capped read path
    E.append((
        "RQ3 read budget: the rollup summary and the output-stripped chunk",
        W(r"The rollup summary snippet recovers ([\d.]+)\\% \[([\d.]+), ([\d.]+)\] and the output-stripped chunk ([\d.]+)\\% \[([\d.]+), ([\d.]+)\]"),
        [p(read_budget("rollup_hookshape").recall), p(read_budget("rollup_hookshape").recall_ci[0]),
         p(read_budget("rollup_hookshape").recall_ci[1]), p(read_budget("stripped_chunk").recall),
         p(read_budget("stripped_chunk").recall_ci[0]), p(read_budget("stripped_chunk").recall_ci[1])]))
    E.append((
        "RQ3 read budget: the pointer recovers nothing and the controls attempt nothing",
        W(r"The pointer answers nothing\. It recovers ([\d.]+)\\% \[([\d.]+), ([\d.]+)\].*?none of the three attempts an answer"),
        [p(read_budget("rollup_pointer").recall), p(read_budget("rollup_pointer").recall_ci[0]),
         p(read_budget("rollup_pointer").recall_ci[1])]))
    # ---- RQ3 length subsection
    sp_l, sp_s = R["spike_long"]["new"], R["spike_short"]["new"]
    E.append((
        "RQ3 length: the long rollup against the rollup, the rule cut and prose-only (re-anchored 2026-10-14; its 86.3% and 2.7% left the prose and are graded in tab:frontier by check_length_table)" + RA,
        W(r"generated from the output-stripped transcript\. It is \$\+([\d.]+)\$ pp over the production rollup \(Table~\\ref\{tab:paired\}, row 7\), and the rule cuts are nominally ahead of it but statistically indistinguishable: the output-stripped transcript leads by ([\d.]+) pp \(row 8\) and prose-only by ([\d.]+) \(row 11\)"),
        [d1("long sonnet - rollup (124)"),
         p(-round(pd_["long sonnet - output-stripped (124)"]["diff_pts"] + 1e-9, 1)),
         p(-round(pd_["long sonnet - prose-only (124)"]["diff_pts"] + 1e-9, 1))]))
    sh = spd["short sonnet - production rollup"]
    E.append((
        "RQ3 length: the short Sonnet arm on the production prompt" + RA,
        r"writes (\d+) tokens and recovers ([\d.]+)\\%, \$-([\d.]+)\$ pp \[\$-([\d.]+)\$, \$([\d.]+)\$\] from the"
        r" production rollup",
        [str(srs["median_rollup_tokens"]), p(s_short.recall), p(-round(sh["diff_pts"] + 1e-9, 1)),
         p(-sh["cluster95_pts"][0]), p(sh["cluster95_pts"][1])]))
    mp = spd["matched sonnet - prose cut to 933 tok"]
    mo = spd["matched sonnet - output-stripped"]
    E.append((
        "RQ3 length: the length-matched Sonnet arm, mean tokens and its paired differences (re-anchored 2026-10-14; median 1,051, 75.0% and 5.1% left the prose and are graded in tab:frontier by check_length_table)" + RA + "; row 10's CI [4.9, 37.6] left the prose and is graded in tab:paired by check_paired_table()",
        W(r"It writes a mean ([\d{},]+) tokens against the cut's (\d+) and is \$\+([\d.]+)\$ pp over the production rollup and \$\+([\d.]+)\$ pp over the prose cut, both clearly above zero \(Table~\\ref\{tab:paired\}, rows 9 and 10\)\. It still trails the output-stripped transcript by ([\d.]+) pp"),
        [_tex_int(srm["mean_rollup_tokens"]), "933", d1("matched sonnet - rollup (124)"),
         d1("matched sonnet - prose matched (124)"), p(-round(mo["diff_pts"] + 1e-9, 1))]))
    for _k in ("matched sonnet - rollup (124)", "matched sonnet - prose matched (124)"):
        if not pd_[_k]["cluster95_pts"][0] > 0:
            raise ValueError(f"the paper says {_k} is clearly above zero; its 95% CI does not exclude zero")
    hv, hv1 = _haiku_v2()["v2"], _haiku_v2()["v1"]
    hrl = sr["arms"]["matched haiku v2 rollup"]
    hp, hs = spd["matched haiku v2 - production rollup"], spd["matched haiku v2 - matched sonnet"]
    hc = spd["matched haiku v2 - prose cut to 933 tok"]


    E.append((
        "RQ3 length: Haiku on the length-targeted prompt, as the section now words it (tokens, recall, wrong, rows 12 and 13)" + RA + RA5 + "; the CIs of rows 12 and 13 left the prose and are graded in tab:paired by check_paired_table()",
        W(r"Haiku writes a mean (\d+) tokens against the (\d+) target and recovers ([\d.]+)\\% \((\d+) of (\d+)\), with ([\d.]+)\\% of attempts wrong\."
          r" Against the Sonnet matched artifact the difference is \$-([\d.]+)\$ pp, clearly below zero \(Table~\\ref\{tab:paired\}, row 13\)\."
          r" Against the production rollup it gains \$\+([\d.]+)\$ pp \(row 12\), statistically indistinguishable\."),
        [str(round(hrl["mean_rollup_tokens"])), "932", p(hv.recall), str(hv.recoverable_n), str(hv.n), p(hv.wrong_rate),
         p(-round(hs["diff_pts"] + 1e-9, 1)),
         p(round(hp["diff_pts"] + 1e-9, 1))]))
    if not (hs["cluster95_pts"][1] < 0 and hp["cluster95_pts"][0] < 0 < hp["cluster95_pts"][1]):
        raise ValueError("the paper says row 13 is clearly below zero and row 12 indistinguishable; the CIs disagree")
    E.append((
        "RQ3 length: the roughly 21-point Sonnet-matched against production-Haiku gap, and the prompt's share of it" + RA + RA5 + " (re-anchored 2026-10-15, positive-statement pass: the 'not +-5' / 'not significant' wording left the sentence; values unchanged)",
        W(r"the prompt accounts for ([\d.]+) pp of the Sonnet matched artifact's lead over the production Haiku rollup, and the generator for the other (\d+\.\d)"),
        [p(round(hp["diff_pts"] + 1e-9, 1)), p(-round(hs["diff_pts"] + 1e-9, 1))]))
    E.append((
        "method/results: the headline gap against the original pooled rollup arm, and how much smaller it is" + RA + RA5,
        W(r"which recovers ([\d.]+)\\% \[([\d.]+), ([\d.]+)\] on the same questions\. There it is ([\d.]+) pp, ([\d.]+) smaller, and the output-stripped transcript's 95\\% interval \[([\d.]+), ([\d.]+)\] still lies above the pooled rollup's"),
        [p(orig["frozen"]["recall"]), p(orig["frozen"]["ci"][0]), p(orig["frozen"]["ci"][1]),
         p(round(strip.recall - orig["frozen"]["recall"] + 1e-9, 1)),
         p(round((strip.recall - roll.recall) - (strip.recall - orig["frozen"]["recall"]) + 1e-9, 1)),
         p(strip.recall_ci[0]), p(strip.recall_ci[1])]))
    E.append((
        "method: the headline gap against the original pooled rollup arm, stated beside the uniform baseline" + RA,
        W(r"The output-stripped transcript beats the rollup by ([\d.]+) pp,.*?There it is ([\d.]+) pp"),
        [p(round(strip.recall - roll.recall + 1e-9, 1)), p(round(strip.recall - orig["frozen"]["recall"] + 1e-9, 1))]))
    p8 = uncorrected("prose-8k")
    if abs(s_match.recall - p8.recall) > 2.0:
        raise ValueError(f"the paper says the matched Sonnet arm matches the 8,000-character cut; {s_match.recall:.1f} against {p8.recall:.1f}")
    E.append((
        "RQ3 length: the matched Sonnet arm against the 8,000-character prose cut, and its price (re-anchored 2026-10-14; the cut's 1,861 tokens and 76.6% left the prose and are graded in tab:frontier, and 'matches' is checked to within 2 points)",
        W(r"it matches the prose cut to 8\{,\}000 characters, so generation buys roughly a \$([\d.]+)\\times\$ compression of the prose at this size, at \\\$([\d.]+) per session"),
        [p(round(1861 / srm["median_rollup_tokens"] + 1e-9, 1)),
         f"{srm['gen_cost_per_session_usd']:.2f}"]))
    # ---- frontier discussion
    E.append((
        "frontier: the rollup band is won on size, and the arms the Sonnet summary would displace (re-anchored 2026-10-14; its F band 1,952--14,628 is retired, the displaced arms are recomputed from the hull)",
        W(r"the cut recovers ([\d.]+)\\% \[[\d.]+, [\d.]+\] and the rollup ([\d.]+)\\%\..*?the Sonnet summary written to the rollup's length would take the middle of the frontier from the ([\w-]+), ([\w-]+) and ([\w-]+), at a cost of \\\$([\d.]+) per session"),
        [p(cut.recall), p(roll.recall)] + _frontier_displaced_by_sonnet() + [f"{srm['gen_cost_per_session_usd']:.2f}"]))
    # ---- held-out and fresh rollup gap, the Haiku regeneration, the held-out 1,000-token repeat, retrieval
    rr = _j("replication-rollup/results.json")
    rra = rr["arms_on_covered"]
    ir = _j("independent-regrade/results.json")["paired_differences"]["rollup_minus_stripped__heldout"]["wrong_given_attempted"]["indep"]
    _three = [float(_err_gap()[0]), float(rep_gap()[0]), float(held_vals()[2])]
    if (round(min(_three)), round(max(_three))) != (10, 28):
        raise ValueError(f"the paper says the rollup wrong-rate gap varies from about 10 to 28 points; the data gives {_three}")
    E.append((
        "RQ3: the rollup wrong-rate gap varies from about 10 to 28 points over three sets (re-anchored 2026-10-14; the per-set gaps, counts and the Opus-judged gap are graded in tab:failmode)" + RA,
        W(r"The gap's size varies from about 10 to 28 pp across the three sets"),
        []))
    hh = _j("heldout-rollup-haiku/results.json")
    E.append((
        "method: the logged Haiku 4.5 held-out rollup against the production note and output-stripped",
        r"That rollup, a median (\d+) tokens, is wrong on ([\d.]+)\\% of (\d+) attempts, against ([\d.]+)\\% of (\d+) \((\d+) wrong\) for the production note"
        r" and ([\d.]+)\\% for the output-stripped transcript",
        [str(hh["rollup_median_cl100k_tokens"]["haiku_regen"])] + haiku_regen_wrong()
        + [held_vals()[0], str(_held()["rollup"]["attempted"]), str(_held()["rollup"]["wrong"]), held_vals()[1]]))
    E.append((
        "RQ3 length: the held-out repeat of the 1,000-token Sonnet arm, its cut, and the cap extension" + RA + RA5 + "; the CIs of rows 14 and 15 left the prose and are graded in tab:paired by check_paired_table()",
        W(r"\(median ([\d{},]+) tokens\) recovers ([\d.]+)\\% against ([\d.]+)\\% for the production note \(\$\+([\d.]+)\$ pp, clearly above zero; Table~\\ref\{tab:paired\}, row 15\)"
          r" and ([\d.]+)\\% for output-stripped\. Against a prose cut of its own length \(([\d.]+)\\%\) it is \$\+([\d.]+)\$ pp"
          r" \(row 14\), statistically indistinguishable\. The cost cap was raised from (\d+) to (\d+) USD after the first (\d+) questions of the cut artifact"),
        [s1k["median"], s1k["s1k"], s1k["prod"], s1k["pn_d"], s1k["strip"], s1k["cut"],
         s1k["cu_d"], s1k["cap_from"], s1k["cap_to"], s1k["prefix_n"]]))
    _ho1k = _j("sonnet1k-heldout/results.json")["paired"]
    _pn = _ho1k["sonnet1k_minus_production_note"]["recall"]["clustered_bootstrap_ci95_pts"]
    _cu = _ho1k["sonnet1k_minus_cut"]["recall"]["clustered_bootstrap_ci95_pts"]
    if not (_pn[0] > 0 and _cu[0] < 0 < _cu[1]):
        raise ValueError("the paper says row 15 is clearly above zero and row 14 indistinguishable; the CIs disagree")
    nd = _j("rq4_no_digest_probe.json")
    ndg, ndf = nd["arms"]["G_bm25_no_digest"]["recall@10"], nd["arms"]["F_bm25_no_tool"]["recall@10"]
    ndd = nd["paired_bootstrap_no_digest_minus_other"]["G_bm25_no_digest_minus_F_bm25_no_tool"]
    S.append((
        "supplement S4: RQ4: BM25 over the no-digest record against output-stripped (re-anchored 2026-10-14; the recalls moved to tab:rq4)" + RA
        + " (round 5: the difference and its CI left the prose for the difference block of tab:rq4; this grades that BM25 cell)"
        + " (round 6: the difference block moved to supplementary Table S4 in paper/supplement/sensitivity-checks.tex, in pp with one decimal; this grades that cell in the supplement)",
        r"No-digest \$-\$ stripped & \$\+([\d.]+)\$ \[\$-([\d.]+)\$, \$\+([\d.]+)\$\] &",
        [f"{100 * ndd['mean_diff']:.1f}", f"{100 * abs(ndd['ci_lower']):.1f}", f"{100 * ndd['ci_upper']:.1f}"]))
    pf = _j("probes-from-rollups/results.json")
    ra, pdn = pf["recall_arms_new_probes"], pf["paired_differences_new_probes"]
    ov = pf["overlap_new_probes"]
    fg = pdn["F_bm25_no_tool_minus_G_bm25_no_digest"]
    E.append((
        "RQ4: probes written from rollup text, the counts and the overlap spread (re-anchored 2026-10-14; the four BM25 recalls moved to tab:rq4)" + RA
        + " (round 5: the counts and overlap spread moved to the tab:rq4 caption and the two BM25 differences to its difference block; same values, regex now spans caption and table, group order caption first)"
        + " (round 6: split; the two BM25 differences moved with the difference block to supplementary Table S4 and are graded there by the entry that follows; the recall move is in pp, about 20)"
        + " (round 7: split again; the overlap spread and the recall move left the tab:rq4 caption verbatim for the caption of supplementary Table S4 and are graded there by the entry that follows; this keeps the counts in tab:rq4)"
        + " (2026-10-15, table notes: the sentence left the tab:rq4 caption verbatim for the note under its tabular, opening 'Recall@10 (%):'; same values)",
        W(r"\\label\{tab:rq4\}.*?\\parbox\{\\columnwidth\}\{\\scriptsize\\raggedright Recall@10 \(\\%\): the share of probes whose target session is among the top 10 results, over the (\d+) sessions that have a rollup\..*?or, post hoc and exploratory, from the rollup text \(([\d{},]+)\)\."),
        [str(pf["n_sessions"]), f"{pf['n_probes']:,}".replace(",", "{,}")]))
    S.append((
        "supplement S4: RQ4: probes written from rollup text, the overlap spread and the recall move (round 7, 2026-10-14: split from the counts entry above; moved verbatim from the tab:rq4 caption to the caption of supplementary Table S4, same values)"
        " (2026-10-15, table notes: moved verbatim from the S4 caption to the 'Tests behind Table S4' paragraph after the float; same values)",
        W(r"\\noindent\\textbf\{Tests behind Table~\\ref\{tab:rq4-diff\}\.\} Transcript probes:.*?the overlap spread is only ([\d.]+) while recall moves by about ([\d.]+) pp\.$"),
        [f"{max(ov.values()) - min(ov.values()):.3f}",
         f"{100 * pf['observed']['D_minus_E']:.0f}"]))
    S.append((
        "supplement S4: RQ4: probes written from rollup text, the two BM25 paired differences (split from the counts-and-overlap entry above)" + R6,
        W(r"\{Rollup-text probes\} \\\\\s*Rollup \$-\$ raw & \$\+([\d.]+)\$ \[\$\+([\d.]+)\$, \$\+([\d.]+)\$\] &"
          r".*?Stripped \$-\$ no-digest & \$-([\d.]+)\$ \[\$-([\d.]+)\$, \$\+([\d.]+)\$\] &"),
        [f"{100 * pdn['D_bm25_rollup_minus_E_bm25_raw']['mean_diff']:.1f}", f"{100 * pdn['D_bm25_rollup_minus_E_bm25_raw']['ci_lower']:.1f}",
         f"{100 * pdn['D_bm25_rollup_minus_E_bm25_raw']['ci_upper']:.1f}", f"{100 * abs(fg['mean_diff']):.1f}",
         f"{100 * abs(fg['ci_lower']):.1f}", f"{100 * fg['ci_upper']:.1f}"]))
    # ---- no-digest record: size, Replication 1, held-out and pooled estimate, full record against output-stripped
    nds = _j("action_record_no_digests_cl100k.json")
    E.append((
        "method: the no-digest action record's median size against output-stripped",
        r"has a median of ([\d{},]+)\s+cl100k tokens against ([\d{},]+) for output-stripped",
        [_tex_int(nds["no_result_digests_variant"]["median_cl100k_tokens"]),
         _tex_int(nds["output_stripped_baseline_median_cl100k_tokens"])]))
    rp = _j("replication/results.json")
    qn = [__import__("json").loads(l) for l in (REPO_ROOT / "data" / "paper-rollups" / "replication" / "questions.jsonl").read_text().splitlines() if l.strip()]
    if not (rp["primary"]["equivalent_pm10"] and not rp["primary"]["equivalent_pm5"] and rp["secondary_gating"]["holds"]):
        raise ValueError("Replication 1 equivalence verdicts no longer match the paper's text")
    zc = rp["zero_context"]
    E.append((
        "RQ3 no-digest: Replication 1, frozen and answered counts, zero-context control, TOST verdicts, gating secondary, absent rate (re-anchored 2026-10-14; recall, difference and CI moved to tab:nodigest)" + RA + " (re-anchored 2026-10-15, positive-statement pass: the 'not +-5' / 'not significant' wording left the sentence; values unchanged)",
        W(r"We froze (\d+) questions over (\d+) sessions\. The cost cap stopped the last (\w+), so (\d+) questions over (\d+) sessions were answered,"
          r" and a zero-context control recovers (\d+) of (\d+)\. Its recall difference \(Table~\\ref\{tab:nodigest\}\) is"
          r" equivalent within \$\\pm 10\$ pp\. The gating secondary, the wrong-answer rate, is ([\d.]+)\\% against ([\d.]+)\\%"
          r" \(upper one-sided 95\\% bound \$\+([\d.]+)\$ pp, inside its 5-pp margin\)\. The absent rate, reported and not gating, is ([\d.]+)\\% against ([\d.]+)\\%\."),
        [str(rp["frozen_n"]), str(len({q["session_id"] for q in qn})), _WORD[rp["frozen_n"] - rp["n_questions"]], str(rp["n_questions"]),
         str(rp["n_sessions"]), str(round(zc["n"] * zc["recall_pct"] / 100)), str(zc["n"]),
         _f1(rp["point"]["wga_nodigest_pct"]), _f1(rp["point"]["wga_stripped_pct"]), _f1(rp["secondary_gating"]["wga_diff_one_sided_95_upper"]),
         _f1(rp["point"]["absent_nodigest_pct"]), _f1(rp["point"]["absent_stripped_pct"])]))
    pe = _j("pooled-estimate/results.json")
    ph, pp = pe["per_set"]["heldout"], pe["pooled"]
    if "post hoc" not in pe["status"]:
        raise ValueError("the pooled estimate is no longer labelled post hoc in its data file")
    E.append((
        "RQ3 no-digest: the post hoc pooled estimate over the two non-design sets (re-anchored 2026-10-14; the held-out 125 row moved to tab:nodigest)" + RA,
        W(r"Pooling the held-out and Replication 1 sets, which share no question or session, gives \$-([\d.]+)\$ pp \$\[-([\d.]+), \+([\d.]+)\]\$ over (\d+) questions\."),
        [_f1(abs(pp["recall_diff_pts"])), _f1(abs(pp["pooled_clustered_bootstrap"]["recall"]["ci90"][0])),
         _f1(pp["pooled_clustered_bootstrap"]["recall"]["ci90"][1]), str(pp["n_questions"])]))
    fa = _j("fullar-vs-nodigest/results.json")
    E.append((
        "frontier: held-out full record against output-stripped, and what the result digests add" + RA + RA5,
        W(r"On the (\d+) held-out questions that all three artifacts answer, the full record minus output-stripped is \$-([\d.]+)\$ pp,"
          r" and the result digests add ([\d.]+) \(design set\) to ([\d.]+) \(held-out\) pp over the no-digest record"),
        [str(fa["part2_heldout"]["n"]), _f1(abs(fa["part2_heldout"]["pairs"]["full_minus_stripped"]["recall_diff_pts"])),
         _f1(fa["part1_design"]["pairs"]["full_minus_nodigest"]["recall_diff_pts"]),
         _f1(fa["part2_heldout"]["pairs"]["full_minus_nodigest"]["recall_diff_pts"])]))
    # ---- threats: the independent judge, and the power limit of a second replication
    ig = _j("independent-regrade/results.json")
    ao = ig["agreement_overall"]
    pdg = ig["paired_differences"]
    _moves = [abs(v["recall"]["change_in_point_estimate_pts"]) for v in pdg.values() if "recall" in v]
    _ho = pdg["rollup_minus_stripped__heldout"]["wrong_given_attempted"]
    S.append((
        SUP + "threats: the independent Opus 5.5 regrade, its answer count, agreement, largest recall movement, and the held-out wrong-rate gap" + RA + R7 + "; its CI [9.8, 27.6] is graded in tab:sens row 11 and in the tab:failmode caption by check_failmode_table()",
        W(r"single-draw Claude Opus 5\.5 regrade of (\d+) answers \(Replication 1, held-out, replication rollup\) & Four-class agreement ([\d.]+) \(\$\\kappa\$ ([\d.]+)\); recovered versus not ([\d.]+) \(\$\\kappa\$ ([\d.]+)\)\. Recall differences move [\d.]+ to ([\d.]+) pp\."
          r" Held-out rollup minus output-stripped wrong rate: \$\+([\d.]+)\$ stored, \$\+([\d.]+)\$ \["),
        [str(ao["n"]), f"{ao['raw_agreement_4class']:.3f}", f"{ao['kappa_4class']:.3f}", f"{ao['raw_agreement_recovered']:.3f}",
         f"{ao['kappa_recovered']:.3f}", _f1(max(_moves)), _f1(_ho["stored"]["diff_pts"]), _f1(_ho["indep"]["diff_pts"])]))
    S.append((
        SUP + "sens row 11: the Opus 5.5 regrade, agreement, recall movement and the held-out wrong-rate gap" + RA,
        r"Four-class agreement ([\d.]+) \(\$\\kappa\$ ([\d.]+)\); recovered versus not ([\d.]+) \(\$\\kappa\$ ([\d.]+)\)\. Recall differences move ([\d.]+) to ([\d.]+) pp\."
        r" Held-out rollup minus output-stripped wrong rate: \$\+([\d.]+)\$ stored, \$\+([\d.]+)\$ \[([\d.]+), ([\d.]+)\] regraded",
        [f"{ao['raw_agreement_4class']:.3f}", f"{ao['kappa_4class']:.3f}", f"{ao['raw_agreement_recovered']:.3f}", f"{ao['kappa_recovered']:.3f}",
         _f1(min(_moves)), _f1(max(_moves)), _f1(_ho["stored"]["diff_pts"]), _f1(_ho["indep"]["diff_pts"]),
         _f1(_ho["indep"]["ci95_pts"][0]), _f1(_ho["indep"]["ci95_pts"][1])]))
    r2 = _j("replication2/power_sim.json")
    r2s = _j("replication2/supply.json")
    _cell = [c for c in r2["cells"] if c["per_session"] == 3 and c["N"] == r2s["max_questions_at_3_per_session"]
             and abs(c["true_diff_pts"] - round(r2["observed_pooled_diff_pts"], 2)) < 0.01]
    E.append((
        "threats: a second replication aimed at plus or minus 5 is a power limit, eligible sessions, questions and stand-alone power",
        r"(\d+) sessions are eligible, capping it at (\d+) questions, and its stand-alone power at the pooled estimate is about (\d+)\\%",
        [str(r2s["n_eligible"]), str(r2s["max_questions_at_3_per_session"]), str(round(100 * _cell[0]["power_alone"]))]))
    # ---- batch 2 (2026-10-14): second answer draw, 2k length run, position check, dense retrieval
    sd = _j("seconddraw-remainder/results.json")
    sdn, sds = sd["per_arm"]["nodigest"], sd["per_arm"]["stripped"]
    sda = sd["arm_difference_nodigest_minus_stripped"]
    if "post hoc" not in (arm_rates.DATA / "seconddraw-remainder" / "analysis_plan.md").read_text().lower():
        raise ValueError("the second-draw plan is no longer labelled post hoc")
    if sd["n_questions"] != rp["n_questions"]:
        raise ValueError("the second draw does not cover every answered Replication 1 question")
    if not (sd["equivalence_descriptive_not_preregistered"]["inside_pm10"] and sd["equivalence_descriptive_not_preregistered"]["inside_pm5"]):
        raise ValueError("the paper says the averaged 90% CI lies inside +-10 and +-5; the data disagrees")
    S.append((
        SUP + "sens row 12: the second answer draw on all 196 Replication 1 questions, draw-to-draw change, status agreement, per-draw and averaged arm difference" + RA + RA5,
        W(r"second draw for both Replication 1 artifacts on all (\d+) questions moves recall by \$\+([\d.]+)\$ \(no-digest\) and \$\+([\d.]+)\$ pp \(output-stripped\),"
          r" and recovered status agrees on ([\d.]+)\\% and ([\d.]+)\\%\. The difference between artifacts is \$\+([\d.]+)\$ and \$\+([\d.]+)\$ pp in the two draws"
          r" and \$\+([\d.]+)\$ averaged \(90\\% CI \$\[-([\d.]+), \+([\d.]+)\]\$\), inside"),
        [str(sd["n_questions"]), _f1(sdn["change_draw2_minus_draw1"]["point_pts"]), _f1(sds["change_draw2_minus_draw1"]["point_pts"]),
         _f1(sdn["same_recovered_status"]["point_pts"]), _f1(sds["same_recovered_status"]["point_pts"]),
         _f1(sda["armdiff_d1"]["point_pts"]), _f1(sda["armdiff_d2"]["point_pts"]), _f1(sda["armdiff_avg"]["point_pts"]),
         _f1(abs(sda["armdiff_avg"]["ci90"][0])), _f1(sda["armdiff_avg"]["ci90"][1])]))
    E.append((
        "RQ3 no-digest: the draw-averaged difference stated in the subsection, with the sens-row pointer (added 2026-10-14; the per-draw figures moved to tab:sens row 12)" + RA + " (round 4: the pointer now reads 'supplementary Table S1, row 12')" + "; its 90% CI [-2.3, +4.6] left the prose and is graded in tab:sens row 12 by the second-answer-draw pin",
        W(r"averaged over the two draws on all (\d+) questions the difference is \$\+([\d.]+)\$ pp \(supplementary Table S1, row (\d+)\), and the \$\\pm 10\$ conclusion rests on draw 1"),
        [str(sd["n_questions"]), _f1(sda["armdiff_avg"]["point_pts"]), "12"]))
    b2 = _j("budget2k-heldout/results.json")
    b2p, b2c = b2["paired"]["sonnet2k_minus_cut2k"]["recall"], b2["paired"]["cut2k_minus_cut1k"]["recall"]
    S.append((
        SUP + "sens row 13: the 2,000-token request, realised median, recall against the equal-length cut, and the cut's own gain from length" + RA,
        W(r"asked for 2\{,\}000 tokens and got a median ([\d{},]+); it recovers ([\d.]+)\\% against ([\d.]+)\\% for a cut of that length"
          r" \(\$\+([\d.]+)\$ pp \[\$-([\d.]+)\$, \$\+([\d.]+)\$\], statistically indistinguishable\), and the cut gains ([\d.]+) pp from 1\{,\}000 to 2\{,\}000 tokens"
          r" \[\$-([\d.]+)\$, \$\+([\d.]+)\$\]"),
        [_tex_int(b2["generation"]["rollup_tokens_median"]), _f1(b2["arms"]["sonnet2k"]["recall_pct"]), _f1(b2["arms"]["cut2k"]["recall_pct"]),
         _f1(b2p["diff_pts"]), _f1(abs(b2p["clustered_bootstrap_ci95_pts"][0])), _f1(b2p["clustered_bootstrap_ci95_pts"][1]),
         _f1(b2c["diff_pts"]), _f1(abs(b2c["clustered_bootstrap_ci95_pts"][0])), _f1(b2c["clustered_bootstrap_ci95_pts"][1])]))
    import re as _re
    # candidate 4 and its "amendment 1" line report the same six lengths, so count distinct result lines
    _log = (arm_rates.DATA / "budget2k-heldout" / "calibration_log.md").read_text()
    _cal = [int(float(x) + 0.5) for x in dict.fromkeys(_re.findall(r"## result candidate \d+\n\nlengths \[[^\]]*\], median ([\d.]+),", _log))]
    S.append((
        SUP + "sens row 13: the 2,000-token request, realised median, and the calibration attempts' highest median" + R7,
        W(r"A note requested at 2\{,\}000 tokens realised a median ([\d{},]+); (\w+) calibration attempts did not push the generator past a median of ([\d{},]+) tokens"),
        [_tex_int(b2["generation"]["rollup_tokens_median"]), {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}[len(_cal)],
         _tex_int(max(_cal))]))
    ps = _j("position-heldout-vs-design/results.json")["sets"]
    ph_, pd2 = ps["held"], ps["design"]
    _hb, _db = ph_["recall_by_position_bucket"], pd2["recall_by_position_bucket"]
    S.append((
        SUP + "sens row 14: question position, held-out against design, median relative position, first quarter, and rollup recall in every bucket" + R7,
        W(r"fall at the same median relative position \(([\d.]+), ([\d.]+); first quarter ([\d.]+)\\%, ([\d.]+)\\%\), yet the rollup recalls less in every"
          r" position bucket on held-out \(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\\% against ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\\%\)\. Held-out sessions are longer"
          r" \(median ([\d.]+)k against ([\d.]+)k characters\)"),
        [f"{ph_['relative_position']['median']:.3f}", f"{pd2['relative_position']['median']:.3f}",
         _f1(ph_["share_first_25pct"]["pct"]), _f1(pd2["share_first_25pct"]["pct"])]
        + [_f1(b["recall_pct"]) for b in _hb] + [_f1(b["recall_pct"]) for b in _db]
        + [_f1(ph_["session_length_chars"]["median"] / 1000), _f1(pd2["session_length_chars"]["median"] / 1000)]))
    if not all(h["recall_pct"] < d["recall_pct"] for h, d in zip(_hb, _db)):
        raise ValueError("the paper says held-out rollup recall is lower in every position bucket; the data disagrees")
    if ps["rep"]["sessions_located"] != 98:
        raise ValueError("the paper says Replication 1 spans 98 sessions; the position file gives another number")
    S.append((
        SUP + "sens row 14: Replication 1 spans 98 other sessions" + R7,
        r"\(Replication 1 spans (\d+) other sessions\)",
        [str(ps["rep"]["sessions_located"])]))
    nd2 = _j("rq4_no_digest_probe.json")
    ndp = nd2["paired_bootstrap_no_digest_minus_other"]["G_dense_no_digest_minus_C_raw_no_tool"]
    S.append((
        "supplement S4: RQ4: dense retrieval over the no-digest record against output-stripped (re-anchored 2026-10-14; the recalls moved to tab:rq4)"
        + " (round 5: the difference and its CI left the prose for the difference block of tab:rq4; this grades that dense cell)"
        + " (round 6: the difference block moved to supplementary Table S4 in paper/supplement/sensitivity-checks.tex, in pp with one decimal; this grades that cell in the supplement)",
        r"No-digest \$-\$ stripped & \$[^$]*\$ \[[^]]*\] & \$\+([\d.]+)\$ \[\$-([\d.]+)\$, \$\+([\d.]+)\$\] \\\\",
        [f"{100 * ndp['mean_diff']:.1f}", f"{100 * abs(ndp['ci_lower']):.1f}", f"{100 * ndp['ci_upper']:.1f}"]))
    pr = _j("probes-from-rollups/results.json")["dense"]
    prn, prm = pr["recall_new_probes"], pr["bm25_minus_dense_new_probes"]
    _gaps = [v["mean_diff"] for v in prm.values()]
    S.append((
        "supplement S4: RQ4: the method gaps on the rollup-derived probes, against the substrate gaps of supplementary Table S4 (re-anchored 2026-10-14; the four dense recalls moved to tab:rq4)"
        + " (round 5: the method gaps moved to the tab:rq4 caption, 'no larger than rollup - raw under either method', and the two artifact gaps to the rollup-text Rollup - raw row; same values, group order caption first, BM25 before dense)"
        + " (round 6: split; the gaps are in pp and the caption now points at Table S4, where the two artifact gaps are graded by the entry that follows)"
        + " (round 7, 2026-10-14: the sentence moved verbatim from the tab:rq4 caption to the caption of supplementary Table S4; graded there, same regex)",
        W(r"the BM25-minus-dense gaps per artifact run ([\d.]+) to ([\d.]+) pp, no larger than rollup \$-\$ raw under either method, and"),
        [f"{100 * min(_gaps):.1f}", f"{100 * max(_gaps):.1f}"]))
    S.append((
        "supplement S4: RQ4: the two rollup-text substrate gaps, BM25 and dense rollup - raw (split from the method-gap entry above)" + R6,
        W(r"\{Rollup-text probes\} \\\\\s*Rollup \$-\$ raw & \$\+([\d.]+)\$ \[[^]]*\] & \$\+([\d.]+)\$ \["),
        [f"{100 * pr['reading_rule_same_probe_set']['largest_bm25_substrate_gap_new_probes']:.1f}",
         f"{100 * pr['reading_rule_same_probe_set']['largest_dense_substrate_gap']['mean_diff']:.1f}"]))
    if pr["reading_rule_same_probe_set"]["method_gap_larger_than_largest_dense_substrate_gap"]:
        raise ValueError("the paper says substrate gaps are at least as large as the method gaps; the data says otherwise")
    # ---- future work: the three pins that stood here are retired, see OBSOLETE_PINS
    # ---- sensitivity table rows (prose). Since 2026-10-14 (round 4) the table lives in
    # paper/supplement/sensitivity-checks.tex as supplementary Table S1, so these original
    # anchors grade the supplement file; the Threats sentences that now restate rows 2-10
    # in main.tex are graded by the RA4 entries that follow, against the same data.
    reg = R["regenerations"]
    regs = [reg[k]["new"]["recall"] for k in ("haiku_from_stripped", "haiku_prod_input_2x", "haiku_stripped_matchlen")]
    sp = R["spike_generator_effect"]
    S.append((
        SUP + "sens row 2: generator effect on the 8-session spike" + RA,
        W(r"one at matched stripped input \((\d+) sessions\).*?Matched input: \$\+(\d+)\$ pp, \$p = ([\d.]+)\$"),
        [str(sp["sessions"]), str(round(sp["diff"])), f"{sp['p']:.2f}"]))
    S.append((
        SUP + "sens row 3: the three Haiku regenerations against the production rollup",
        W(r"The three: ([\d.]+)--([\d.]+)\\%; current renderer: ([\d.]+)\\%; production rollup ([\d.]+)\\%"),
        [p(min(regs)), p(max(regs)), p(reg["renderer_control"]["new"]["recall"]), p(roll.recall)]))
    S.append((
        SUP + "sens row 4: the current renderer's regeneration against the production rollup",
        W(r"current renderer: ([\d.]+)\\%; production rollup ([\d.]+)\\%"),
        [p(reg["renderer_control"]["new"]["recall"]), p(roll.recall)]))
    S.append((
        SUP + "sens row 5: agreement between judges",
        r"Four-way Cohen's \$\\kappa\$ ([\d.]+)--([\d.]+) across judge pairs; ([\d.]+)--([\d.]+)\\% agreement on wrong",
        [f"{jl['kappa_range'][0]:.2f}", f"{jl['kappa_range'][1]:.2f}", p(jl["wrong_vs_not_range"][0]),
         p(jl["wrong_vs_not_range"][1])]))
    dr = R["rollup_draws"]
    S.append((
        SUP + "sens row 6: grader draws on fixed answers",
        r"four-grade \$\\kappa\$ ([\d.]+)--([\d.]+); rollup ([\d.]+), ([\d.]+) and ([\d.]+)\\%",
        [f"{min(dr['kappa_pairs']):.2f}", f"{max(dr['kappa_pairs']):.2f}"] + [p(x) for x in dr["per_draw_recall"]]))
    S.append((
        SUP + "sens row 7: prose-only on the 225 held-back questions against the original rollup arm" + RA + RA5,
        r"([\d.]+)\\% against ([\d.]+)\\% for the original rollup artifact's answers",
        [p(R["prose_only_held_back"]["new"]["recall"]), p(orig["held_back"]["recall"])]))
    S.append((
        SUP + "sens row 8: rollup against call-only without the sessions that carry no tool calls",
        r"Rollup ([\d.]+)\\% \[([\d.]+), ([\d.]+)\] \(\$n = (\d+)\$\) against call-only ([\d.]+)\\%"
        r" \[([\d.]+), ([\d.]+)\] \(\$n = (\d+)\$\)",
        [p(nt["rollup"]["recall"]), p(nt["rollup"]["ci"][0]), p(nt["rollup"]["ci"][1]), str(nt["rollup"]["n"]),
         p(nt["call_only"]["recall"]), p(nt["call_only"]["ci"][0]), p(nt["call_only"]["ci"][1]), str(nt["call_only"]["n"])]))
    rc = R["recap"]
    S.append((
        SUP + "sens row 9: the compaction recap, recall with it left in, and on the 54 records the recap covers",
        r"recall is unchanged at ([\d.]+)\\%\. On those (\d+) records the recap alone recovers ([\d.]+)\\% against"
        r" prose-only's ([\d.]+)\\%",
        [p(rc["prose_only_with_recap_all"]["recall"]), str(rc["n_records"]), p(rc["continuation_only"]["recall"]),
         p(rc["prose_only_same"]["recall"])]))
    tp = R["tied_probes"]
    S.append((
        SUP + "sens row 10: prose-only against call-only on the probes with equal overlap" + RA,
        r"On the (\d+) probes with equal overlap, prose-only still beats call-only by ([\d.]+) pp",
        [str(tp["n"]), p(tp["gap"])]))
    # ---- the same rows, as the Threats section of main.tex stated them (round 4); round 7 halved Threats and
    # the facts live only in supplementary Table S1, so these re-anchored pins grade the supplement cells
    S.append((
        SUP + "sens row 2: generator effect on the 8-session spike" + RA4 + R7,
        W(r"one at matched stripped input \((\d+) sessions\).*?Matched input: \$\+(\d+)\$ pp, \$p = ([\d.]+)\$, statistically indistinguishable"),
        [str(sp["sessions"]), str(round(sp["diff"])), f"{sp['p']:.2f}"]))
    S.append((
        SUP + "sens row 3: the three Haiku regenerations against the production rollup" + RA4 + R7,
        W(r"The three: ([\d.]+)--([\d.]+)\\%; current renderer: ([\d.]+)\\%; production rollup ([\d.]+)\\%\. None reproduces the rollup"),
        [p(min(regs)), p(max(regs)), p(reg["renderer_control"]["new"]["recall"]), p(roll.recall)]))
    S.append((
        SUP + "sens row 4: the current renderer's regeneration against the production rollup" + RA4 + R7,
        W(r"current renderer: ([\d.]+)\\%; production rollup ([\d.]+)\\%\. None reproduces"),
        [p(reg["renderer_control"]["new"]["recall"]), p(roll.recall)]))
    S.append((
        SUP + "sens row 5: agreement between judges" + RA4 + RA5 + R7,
        W(r"Agreement between earlier judges, original rollup artifact & Four-way Cohen's \$\\kappa\$ ([\d.]+)--([\d.]+) across judge pairs; ([\d.]+)--([\d.]+)\\% agreement on wrong versus not\."),
        [f"{jl['kappa_range'][0]:.2f}", f"{jl['kappa_range'][1]:.2f}", p(jl["wrong_vs_not_range"][0]),
         p(jl["wrong_vs_not_range"][1])]))
    S.append((
        SUP + "sens row 6: grader draws on fixed answers" + RA4 + R7,
        W(r"Grader draws on fixed answers: four-grade \$\\kappa\$ ([\d.]+)--([\d.]+); rollup ([\d.]+), ([\d.]+) and ([\d.]+)\\%\. \\\\"),
        [f"{min(dr['kappa_pairs']):.2f}", f"{max(dr['kappa_pairs']):.2f}"] + [p(x) for x in dr["per_draw_recall"]]))
    S.append((
        SUP + "sens row 7: prose-only on the 225 held-back questions against the original rollup arm" + RA4 + RA5 + R7,
        W(r"Prose-only on the 225 held-back questions & ([\d.]+)\\% against ([\d.]+)\\% for the original rollup artifact's answers, both regraded by the revised grader\. Direction only\."),
        [p(R["prose_only_held_back"]["new"]["recall"]), p(orig["held_back"]["recall"])]))
    E.append((
        "sens row 8: call-only against the rollup without the sessions that carry no tool calls, as RQ3 retention states it"
        " (added 2026-10-14, round 4: the body sentence pointing at the row was ungraded; the CIs and n are graded in the supplement)",
        W(r"without the (\d+) sessions that carry no tool calls, call-only's ([\d.]+)\\% nominally passes the rollup's ([\d.]+)\\%"),
        [str(len(rq3_renumber.NO_TOOL_SESSIONS)),
         p(nt["call_only"]["recall"]), p(nt["rollup"]["recall"])]))
    S.append((
        SUP + "sens row 9: the compaction recap, recall with it left in, and on the 54 records the recap covers" + RA4 + R7,
        W(r"\((\d+) of \d+ records\); recall is unchanged at ([\d.]+)\\%\. On those (\d+) records the recap alone recovers ([\d.]+)\\% against prose-only's ([\d.]+)\\%\."),
        [str(rc["n_records"]), p(rc["prose_only_with_recap_all"]["recall"]), str(rc["n_records"]), p(rc["continuation_only"]["recall"]),
         p(rc["prose_only_same"]["recall"])]))
    S.append((
        SUP + "sens row 10: prose-only against call-only on the probes with equal overlap" + RA4 + R7,
        W(r"Probe wording & Mean overlap [\d.]+ higher for prose than for tool calls\. On the (\d+) probes with equal overlap, prose-only still beats call-only by ([\d.]+) pp\."),
        [str(tp["n"]), p(tp["gap"])]))
    # ---- threats and conclusion
    S.append((
        SUP + "sens row 5: the stronger judges against the original rollup arm" + RA + RA5 + R7,
        W(r"Stronger judges move the original rollup artifact's recall over the same (\d+) records from ([\d.]+)\\% to"
          r" ([\d.]+)\\% \(Claude Opus 5\.5\) and ([\d.]+)\\% \(Claude Opus 4\.6\); answers fixed, comparison artifacts not regraded"),
        [str(jl["n_all"]), p(jl["original"]["recall_all"]), p(jl["claude-opus-5-5"]["recall_all"]),
         p(jl["claude-opus-4-6-20250617"]["recall_all"])]))
    S.append((
        SUP + "sens row 7: the prefix, original rollup arm on the frozen 124 and the held-back 225" + RA + R7,
        W(r"recall is ([\d.]+)\\% \[([\d.]+), ([\d.]+)\] on the frozen (\d+) and ([\d.]+)\\% \[([\d.]+), ([\d.]+)\] on the held-back (\d+)\."),
        [p(orig["frozen"]["recall"]), p(orig["frozen"]["ci"][0]), p(orig["frozen"]["ci"][1]), str(orig["frozen"]["n"]),
         p(orig["held_back"]["recall"]), p(orig["held_back"]["ci"][0]), p(orig["held_back"]["ci"][1]),
         str(orig["held_back"]["n"])]))
    S.append((
        SUP + "sens row 9 (repinned from the cut 'we report drops it' pin): recap left in the prose-only context, recall unchanged",
        r"recall is unchanged at ([\d.]+)\\%\. On those",
        [p(prose.recall)]))
    S.append((
        SUP + "sens row 9 (repinned from the cut 'we report drops it' pin): recap left in the prose-only context, recall unchanged" + RA4 + R7,
        W(r"Compaction recap left in the prose-only context & [^&]*?recall is unchanged at ([\d.]+)\\%\. On those"),
        [p(prose.recall)]))
    E.append((
        "conclusion: the recommended substrate against the rollup summary (re-anchored 2026-10-14; the conclusion now carries recall only, the wrong and decline rates are graded in tab:failmode)",
        W(r"It recovers ([\d.]+)\\% of the design questions against the rollup's ([\d.]+)\\%"),
        [p(strip.recall), p(roll.recall)]))
    E.append((
        "held-out: the production note's recall falls from the design set to the held-out set (added 2026-10-14; this was graded only inside the retired abstract pin)",
        W(r"The production note's recall falls from ([\d.]+)\\% on the design set to ([\d.]+)\\% held-out"),
        [p(roll.recall), held_vals()[7]]))
    return E, S


def _earlier_correction_wrong_rate():
    """The rollup wrong-of-attempted of the earlier refusal correction (10.3%), from the
    archived rq3_confidence_corrected.json. Read only so the paper's sentence that the
    revised grader supersedes it is checked against the number it names."""
    import json as _j
    d = _j.loads((arm_rates.DATA / "rq3_confidence_corrected.json").read_text())
    b = d["arms"]["uniform"]["majority_vote"]["corrected"]
    w = int(b["wrong"]["n"])
    att = int(b["correct"]["n"]) + int(b["partially_correct"]["n"]) + w
    return arm_rates.r1(100.0 * w / att)


def _frontier_with_sonnet_band():
    """The F band the length-matched Sonnet arm would take in the substrate table if it were a row."""
    import frontier_bands as fb
    tok = {"zero": 0, "prompts-only": 202, "prose-2k": 478, "rollup": 929, "prose-8k": 1861,
           "answer-only": 2533, "prose-only": 3055, "dialogue record": 3928, "call-only": 6329,
           "action record": 6393, "output-stripped": 8880, "raw": 137230}
    pts = [(k, t, uncorrected({"zero": "zero context"}.get(k, k)).recall / 100) for k, t in tok.items()]
    sm = _sonnet()["arms"]["matched sonnet rollup"]
    pts.append(("sonnet-matched", int(sm["median_rollup_tokens"]), any_rate("sonnet matched").recall / 100))
    rows = {n: (lo, hi) for n, lo, hi in fb.bands(fb.upper_hull(pts))}
    lo, hi = rows["sonnet-matched"]
    return [f"{lo:,.0f}", f"{hi:,.0f}"]


def _frontier_displaced_by_sonnet():
    """The hull arms the length-matched Sonnet summary would push off the frontier if it
    were priced, in hull order. Replaced the F-band boundaries of the same claim on
    2026-10-14, when the band numbers left the frontier paragraph."""
    import frontier_bands as fb
    tok = {"zero": 0, "prompts-only": 202, "prose-2k": 478, "rollup": 929, "prose-8k": 1861,
           "answer-only": 2533, "prose-only": 3055, "dialogue record": 3928, "call-only": 6329,
           "action record": 6393, "output-stripped": 8880}
    pts = [(k, t, uncorrected({"zero": "zero context"}.get(k, k)).recall / 100) for k, t in tok.items()]
    pts.append(("no-digest", 5506, any_rate("action record (no digest)").recall / 100))
    sm = _sonnet()["arms"]["matched sonnet rollup"]
    before = [h[0] for h in fb.upper_hull(pts)]
    after = [h[0] for h in fb.upper_hull(pts + [("sonnet-matched", int(sm["median_rollup_tokens"]),
                                                    any_rate("sonnet matched").recall / 100)])]
    return [n for n in before if n not in after]


def check_prose_entry(s, label, rx, expected):
    ms = list(re.finditer(rx, s, re.S))
    if len(ms) != 1:
        return False, (f"the sentence matched {len(ms)} times, expected exactly 1"
                       f" -- it was deleted, reworded past its anchor, or duplicated")
    got = list(ms[0].groups())
    bad = [f"main.tex says {g!r} where the data gives {e!r}"
           for g, e in zip(got, expected) if g != e]
    if bad:
        return False, "; ".join(bad)
    return True, f"{', '.join(expected)} reproduce from data/paper-rollups/"


# Pins retired because the sentence they checked was deliberately cut from the paper.
# (name, the sentence it checked, the commit that cut it per `git log -S`, reason).
# A retired pin is listed, never silently deleted, and is not counted as a failure.
# The sentence is still true or false in the data; it is just no longer claimed in
# main.tex. If a cut sentence returns, restore its pin from the commit named here.
OBSOLETE_PINS = [
    ("pruning disclosure: 23 files grew past their pinned length",
     "23 grew past it", "ad3d7aff4",
     "the pruning paragraph was shortened in the 400-token revision; the 965/966, 299/300 and 1,123/1,125 counts it sat beside are still pinned"),
    ("pruning disclosure: exclusion raises rather than lowers parent emptiness",
     "raises the parent emptiness rate rather than lowering it", "ad3d7aff4",
     "the 98.88% -> 99.48% direction claim is no longer made anywhere in the paper (no 'emptiness' sentence remains)"),
    ("RQ3 length: the two spikes, sessions and questions",
     "two spikes on N sessions and M questions; the K-question run is the result", "706e0044c",
     "the spike sentence was cut when the length section was rewritten around the matched and Haiku arms; the 8-session matched-input spike survives only as sens row 2-4, which keeps its own pin"),
    ("RQ3 length: Haiku on the length-targeted prompt, mean tokens, recall, wrong, and the paired differences",
     "on 120 of the 124 questions (24 of 25 sessions; ... A first, shorter run (N tokens) gave X%.", "706e0044c",
     "superseded by the reworded Haiku sentence, already pinned as 'Haiku on the length-targeted prompt, as the section now words it'; the first-shorter-run and prose-cut comparison are no longer stated"),
    ("conclusion: the same prompt on Haiku",
     "the same prompt on Haiku gains a non-significant X points (Y%, N questions)", "ad3d7aff4",
     "the Conclusion no longer mentions the Haiku run; the figure is still pinned in the abstract-level, intro and contribution pins"),
    ("frontier: prose-only against output-stripped, cost multiple and recall gap",
     "Prose-only is 2.9x cheaper for 1.6 points less recall", "28decaba7",
     "cut in the read-natural pass; the claim is no longer made, and the two arms' recalls and tokens are pinned elsewhere"),
    ("future work: the action record's lead, its power and the sample that would fix it",
     "The X-point lead over the output-stripped transcript has N% power at n = 124", "0b5553dd4",
     "the future-work power paragraph was cut with the action-record material"),
    ("future work: held-out disagreements and the 68-question equivalence test",
     "disagreements go from A against B in its favour to C against D ... 90% CI [..] points: equivalent within +-10, not within +-5", "17c6c4b0a",
     "the 68-question action-record test was moved out of the paper; the held-out no-digest result is now reported on 125 questions and pinned by the pooled-estimate pin"),
    ("future work: the no-digest ablation's equivalence on all 125",
     "(90% CI [..]), though not within +-5", "17c6c4b0a",
     "the sentence was replaced by the Replication 1 section's own wording; the 125-question CI [-9.6, +0.8] is pinned from pooled-estimate/results.json"),
    ("sens row 1: the three Sonnet arms against the production rollup and output-stripped",
     "Long X% at a median N tokens, length-matched Y% at M, production prompt Z% at K, against the production rollup's and output-stripped's", "ad3d7aff4",
     "row 1 of the sensitivity table is now the second 400-token Sonnet generation; the three-arm comparison is stated, and pinned, in the RQ3 length section"),
    ("threats: the recap's effect on the 88.7% prose-only figure",
     "The 88.7% we report drops it", "706e0044c",
     "the sentence is gone; the row-9 wording 'recall is unchanged at 88.7%' makes the same claim and is repinned"),
    ("abstract: wrong-of-attempted, output-stripped against the rollup",
     "The summary declines X% of the design questions against Y% ... so its size varies from about 10 to 28 points across three sets. The summary's recall falls from A% to B%", "cain-review-1014 (section-by-section review)",
     "the review cut the abstract to one headline number per claim; every rate, count and CI it carried is graded in tab:failmode by check_failmode_table(), the 27.2% held-out recall by the new held-out pin, and the 39.1% Haiku regeneration by the method pin"),
    ("abstract: the Sonnet summary at about 1,000 tokens against the equally long rule cut",
     "a Sonnet summary beat an equally long filtered cut by X points ... (+Y points [-L, +H])", "cain-review-1014 (number-free abstract)",
     "the human asked for an abstract that states what the paper is about without numbers; the 21.0 lift is pinned five times in the body and the held-out +6.4 [-8.8, +20.8] in the RQ3 length pin"),
    ("intro: the Sonnet summary against the prose cut at about 1,000 tokens",
     "the intro sentence restating the 1,000-token result", "cain-review-1014 (abstract and intro state one claim)",
     "the human asked the abstract, intro and conclusion to make one claim without retracting it; the 21.0 is pinned five times in the body and the Haiku 5.8 by the tab:paired row-12 check and the RQ3 length paragraph"),
    ("RQ3 length: lead paragraph, Haiku on the same prompt",
     "the intro sentence restating the 1,000-token result", "cain-review-1014 (abstract and intro state one claim)",
     "the human asked the abstract, intro and conclusion to make one claim without retracting it; the 21.0 is pinned five times in the body and the Haiku 5.8 by the tab:paired row-12 check and the RQ3 length paragraph"),
    ("intro: the length-targeted prompt on our Haiku, rounded lift",
     "the intro sentence restating the 1,000-token result", "cain-review-1014 (abstract and intro state one claim)",
     "the human asked the abstract, intro and conclusion to make one claim without retracting it; the 21.0 is pinned five times in the body and the Haiku 5.8 by the tab:paired row-12 check and the RQ3 length paragraph"),
    ("contribution 1: keeping the prose against keeping the tool inputs",
     "discarding all tool I/O recovers X% of ground truth for 3,055 tokens read. Keeping the tool inputs and discarding the prose recovers Y% for 6,329", "cain-review-1014 (section-by-section review)",
     "contributions cut to one claim and one number; 88.7% at 3,055 is pinned by 'RQ3: the prose-only gap to the cut', and 46.3% / 6,329 by the MUST pin and the merged tab:frontier check"),
    ("contribution 2: generated memory fails silently, difference in points, its clustered CI, both rates",
     "the rollup declines X% against Y% ... states a wrong answer as fact on N points more ... The gap holds on the 125 held-out questions", "cain-review-1014 (section-by-section review)",
     "contribution 2 no longer restates the wrong-rate gaps; all of them are graded in tab:failmode"),
    ("contribution 2: the length-targeted prompt alone on Haiku",
     "alone lifts Haiku by a non-significant X points (Y%, N questions)", "cain-review-1014 (section-by-section review)",
     "contribution 2 keeps one number; the Haiku arm's 62.5% (75 of 120) and +5.8 are pinned by the RQ3 length Haiku pins"),
    ("conclusion: the read-budget exception, Sonnet against the cut",
     "recovers X% against Y% for a rule cut", "cain-review-1014 (section-by-section review)",
     "the conclusion now states the rule with at most three numbers; 75.0% and 54.0% are pinned in RQ3 length"),
    ("conclusion: the long Sonnet rollup against the rule cut",
     "recovers more than a rule cut (best X%, no detected difference)", "cain-review-1014 (section-by-section review)",
     "the conclusion no longer names the best generated arm; 86.3% is pinned by the RQ3 length long-rollup pins"),
    ("frontier: the seven-band enumeration (part of the band check, not a counted pin)",
     "Seven configurations win at different $F$ bands: zero (F < 641), prompts-only (641--3,017), ... Five are dominated", "cain-review-1014 (section-by-section review)",
     "the review dropped the enumeration; verify_tex() now recomputes and checks the zero band, the rollup's single band, the rule-cut claim and the top two (action-record) bands that the replacement sentence states; the middle boundaries are no longer claimed"),
    ("frontier: the remaining band boundaries, zero below F = 641, rollup 3,017--4,438, no-digest 51,062--52,176, full record F >= 52,176 (part of the band check, not a counted pin)",
     "Zero context is cheapest below $F = 641$. The rollup wins a single band, $F$ 3{,}017--4{,}438 ... ($F$ 51{,}062--52{,}176) and the full record ($F \\ge 52{,}176$)", "cain-review-1014 (round 2)",
     "the human asked for the frontier as one paragraph without the band boundary enumeration; verify_tex() still recomputes every claim the sentence now makes in words -- zero first, the cheapest substrate changing with F, the rollup's single band and that it is narrower than the median band, rule cuts everywhere else, and the two daggered action records on top"),
    ("frontier: the F band the length-matched Sonnet summary would take, F 1,952--14,628",
     "it would take the middle of the frontier ($F$ 1,952--14,628) from the rollup, prose-8k and answer-only", "cain-review-1014 (round 2)",
     "a band boundary, dropped with the others; the displaced arms (rollup, prose-8k, answer-only) are now recomputed from the hull by _frontier_displaced_by_sonnet()"),
    ("intro: free baseline against the generated summary, recall",
     "It recovers X% of N questions against Y% for the generated summary", "cain-review-1014 (round 2)",
     "the intro paragraph before the contributions now states the read-budget rule in words; 90.3% and 55.6% are pinned in the abstract, the conclusion and fig:retention, and n = 124 by the reporting convention"),
    ("intro: free baseline against the generated summary, declines",
     "It declines X% of design questions against the summary's Y%", "cain-review-1014 (round 2)",
     "same cut; the 8.1% and 37.1% decline rates are graded in tab:failmode by check_failmode_table()"),
    ("contribution 2: the Sonnet rollup at about 1,000 tokens against the cut of that length",
     "a Sonnet-written rollup recovers X%, Y points above a rule cut of that length ... did not replicate", "cain-review-1014 (round 2)",
     "the contribution made a claim and then retracted it; it now states the 400-token result, pinned by the tight-budget range check; 75.0% is graded in tab:frontier and +21.0 [4.9, 37.6] by the RQ3 length pin and tab:paired row 10"),
    ("RQ3: the prose-only gap to the cut",
     "The gap to full prose-only (3,055 tokens, X%) comes from size", "cain-review-1014 (round 2)",
     "restated table cells removed from the RQ3 length prose; prose-only's 3,055 tokens and 88.7% are graded in tab:frontier by check_substrate_table_recalls() and check_length_table()"),
    ("RQ3 length: lead sentence, the matched Sonnet arm against the rollup and the cut, and the long arm",
     "It recovers X%, with W% of attempts wrong. That is +A over the production rollup and +B [..] over the prose cut ... It recovers Y%", "cain-review-1014 (round 2)",
     "the restated 75.0% and 86.3% left the prose; both are graded in tab:frontier by check_length_table(), and +19.4 / +21.0 by the re-anchored length-matched pin"),
]


def main():
    # No default source, deliberately. This used to fall back to --origin when called
    # with no argument, which grades what is already pushed and silently ignores the
    # edit you are about to push. On 2026-10-01 a session ran it three times before
    # pushing, read 65 passed / 0 failed each time, and landed 31849848 with two
    # invariants broken: all three runs had graded the unmodified paper. The source
    # line was printed every time and skipped every time, so printing it is not
    # sufficient -- the caller has to say which file it means.
    if len(sys.argv) < 2:
        print(__doc__.strip())
        print("\nFAIL: no source given, and there is no default.")
        print("  to gate an edit you have not pushed:  check_paper_invariants.py paper/main.tex")
        print("  to gate what is already on main:      check_paper_invariants.py --origin")
        return 2
    arg = sys.argv[1]
    if arg == "--origin":
        s = subprocess.run(["git", "-C", VAULT, "show", "origin/main:paper/main.tex"],
                           capture_output=True, text=True).stdout
        src = "origin/main:paper/main.tex"
    else:
        s = open(arg).read()
        src = arg
    if not s.strip():
        print(f"FAIL: could not read {src}")
        return 1
    # The supplement (supplementary Tables S1-S3, moved out of main.tex on 2026-10-14,
    # round 4) is read from the SAME source as main.tex, never from a default.
    sup_rel = "paper/supplement/sensitivity-checks.tex"
    if arg == "--origin":
        sup = subprocess.run(["git", "-C", VAULT, "show", f"origin/main:{sup_rel}"],
                             capture_output=True, text=True).stdout
        sup_src = f"origin/main:{sup_rel}"
    else:
        _sp = Path(arg).resolve().parent / "supplement" / "sensitivity-checks.tex"
        sup = _sp.read_text() if _sp.exists() else ""
        sup_src = str(_sp)

    if src.startswith("origin/main"):
        print("NOTE: grading origin/main, NOT your working copy. Local edits are invisible\n"
              "      to this run. Pass a path instead to gate an edit before pushing it.")
    print(f"checking {src}  ({len(s)} bytes)")
    print(f"     and {sup_src}  ({len(sup)} bytes)\n")
    bad = 0
    total = 0

    def report(kind, label, ok, msg):
        nonlocal bad, total
        total += 1
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {kind:8s} {label}  ({msg})")

    for label, rx, lo in MUST:
        n = len(re.findall(rx, sup if label.startswith(SUP) else s))
        report("present", label, n >= lo, f"found {n}, need >={lo}")
    print()
    for label, rx in MUST_NOT:
        m = re.findall(rx, s)
        report("absent", label, not m, f"found {len(m)}")

    # Custom check for abstract band winners vs frontier
    print()
    report("custom", "tight-budget range (abstract, contribution, budget lead) matches the budget section",
           *check_intro_tight_budget_range(s))
    report("custom", "abstract band winners agree with frontier",
           *check_abstract_band_winners_agree_with_frontier(s))
    report("custom", "savings claim scoped to the read path",
           *check_savings_claim_scoped_to_read_path(s))
    report("data", "rq5 break-even is the cold-cache figure (1.80 median, 1.63 min)",
           *check_rq5_break_even_cold_cache(s))

    # The band sentence is derivable from the substrate table, so it must not be
    # edited independently of it. The winner-name check above does not catch a
    # changed cost or recall, which silently moves the boundaries.
    report("custom", "frontier bands recompute from the substrate table",
           *verify_tex(s))

    # fig:retention, which replaced tab:retention. Read from the same source as
    # main.tex above, never from a default.
    report("custom", "retention figure pairs recall with wrong rate",
           *check_retention_figure(arg))

    report("data", "stored bootstrap and Sonnet summary agree with the graded records",
           *check_stored_reports_fresh())
    report("custom", "supplement file is readable", bool(sup.strip()), sup_src)
    report("data", "tab:paired rows recompute from the stored bootstrap", *check_paired_table(s, sup))
    report("data", "tab:length rows recompute from the graded records", *check_length_table(s))
    report("data", "capped read path (prose since round 7) recomputes from read_budget()", *check_capped_rows(s))
    report("data", "strict 400-token run (prose since round 7) recomputes from cap400/control", *check_strict_row(s))
    report("data", "tab:failmode rows recompute from their sources", *check_failmode_table(s))
    report("data", "tab:nodigest rows recompute from their sources", *check_nodigest_table(s))
    report("data", "tab:rq4 cells recompute from the retrieval results", *check_rq4_table(s, sup))
    report("data", "tab:position cells recompute from the late-window results", *check_position_table(s))

    # Everything below reads data/paper-rollups/ and recomputes. A FAIL here
    # means main.tex and its source data disagree, which no amount of editing
    # main.tex to satisfy the gate will fix: check which of the two is wrong.
    print()
    report("data", "substrate table recall cells recompute from the records",
           *check_substrate_table_recalls(s))
    try:
        entries, sup_entries = data_backed_prose()
    except Exception as exc:                                    # noqa: BLE001
        report("data", "data/paper-rollups/ is internally consistent", False,
               f"{type(exc).__name__}: {exc}")
        entries, sup_entries = [], []
    for label, rx, expected in entries:
        report("data", label, *check_prose_entry(s, label, rx, expected))
    # supplementary Table S1 rows, graded in the supplement file (round 4)
    for label, rx, expected in sup_entries:
        report("data", label, *check_prose_entry(sup, label, rx, expected))

    print()
    for name, sentence, sha, why in OBSOLETE_PINS:
        print(f"RETIRED {name}  [checked: {sentence!r}; cut in {sha}; {why}]")
    print(f"\n{total - bad} passed, {bad} failed, {len(OBSOLETE_PINS)} retired")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
