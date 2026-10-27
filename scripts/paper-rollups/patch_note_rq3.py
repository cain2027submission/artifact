#!/usr/bin/env python3
"""Apply the RQ3 corrections to the plain-text Obsidian mirror of the paper.

The mirror is the human's live editing surface and paper/main.tex is edited on a
branch, with no sync script in either direction. So the same corrections have to
land in both by hand. Doing it with exact string replacement rather than by
re-describing the edits to a model removes any chance of a transcription slip
between the two copies.

Every replacement is anchored on a full sentence or paragraph and asserted to
match exactly once. A miss is a hard failure rather than a silent no-op, because
a partially applied patch on a file the human is editing is worse than none:
`git add -- <path>` stages nothing if the edit never happened, so a no-op would
otherwise commit cleanly and look like success.

Usage:
    patch_note_rq3.py --check <file>    # report which edits would apply
    patch_note_rq3.py --apply <file>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import REPO_ROOT

NOTE = REPO_ROOT / "2026-09-29 rollup paper plain-text draft for editing in obsidian.md"

# (label, old, new). Order is irrelevant; each is matched independently.
EDITS = []

EDITS.append((
    "abstract",
    "of the questions a rollup attempts to answer, 19.5% are answered wrongly rather than omitted.",
    "of the questions a rollup attempts to answer, 13.6% are answered wrongly "
    "rather than omitted. Against a raw-transcript baseline on the 89 of those "
    "questions whose transcript fits a 1M-token window, the rollup recovers "
    "68.5% of the ground truth where the raw transcript recovers 95.5% "
    "($p = 8.1\\times10^{-7}$ paired), and the two fail differently in a way "
    "the rates hide: every raw-transcript failure is a refusal, while the "
    "rollup's are confident errors.",
))

EDITS.append((
    "methods-30k-window",
    "and only the latter is dangerous when a rollup is injected as context.\n",
    "and only the latter is dangerous when a rollup is injected as context. "
    "Each grade is the majority of three independent grading calls; the reason, "
    "and the grader agreement rate, are given below.\n"
    "\n"
    "One property of the question generator constrains everything in this "
    "section and is easy to miss: questions were generated from the first "
    "30,000 characters of each transcript, not from the whole of it. For the "
    "multi-megabyte sessions in the sample that is a small opening fraction, so "
    "these questions probe *early-session* content disproportionately. This "
    "does not bias the rollup arm against the raw arm, since both are asked the "
    "same questions, but it does mean the absolute retention rates describe the "
    "part of a session a rollup has seen most recently relative to its "
    "beginning, and a chronologically uniform probe set might order the classes "
    "differently.\n",
))

EDITS.append((
    "headline-numbers",
    "Over 124 questions across 25 sessions, 35.5% [27.6, 44.2] were answered "
    "correctly, 21.0% partially, 13.7% wrongly, and 29.8% were absent. "
    "Combining the first two, 56.5% [47.7, 64.9] of what we asked about was "
    "recoverable from the rollup. Of the 87 questions the rollup did attempt, "
    "19.5% [12.6, 29.1] were answered wrongly.",
    "Over 124 questions across 25 sessions, 36.3% [28.4, 45.0] were answered "
    "correctly, 20.2% partially, 8.9% wrongly, and 34.7% were absent. "
    "Combining the first two, 56.5% [47.7, 64.9] of what we asked about was "
    "recoverable from the rollup. Of the 81 questions the rollup did attempt, "
    "13.6% [7.8, 22.7] were answered wrongly.\n"
    "\n"
    "**One protocol, not three.** These figures come from a re-run, and we "
    "report the re-run rather than our first pass for a reason worth stating. "
    "The original arm was not one experiment: its 124 records were produced by "
    "three scripts that differed in the answering prompt, in whether the "
    "grading prompt defined the four grades or merely named them, in the order "
    "the grade parser tested those names, and---in one script---in clipping the "
    "rollup to 15,000 characters. Nothing in the data recorded which script "
    "reached which record, and both revisions were committed together, so "
    "provenance could not be recovered after the fact either. We therefore "
    "re-answered and re-graded all 124 questions from the full, unclipped "
    "rollup under a single protocol. Two of those defects turned out to be "
    "harmless. The clip could only have touched 4 questions, since exactly one "
    "of the 25 sessions has a rollup over 15,000 characters (median 3,528, "
    "maximum 54,619). The parser tested `CORRECT` before `PARTIALLY_CORRECT` "
    "and so recorded a partial as correct, but that substring error can only "
    "move records between the two buckets whose sum is *recoverable*, leaving "
    "*wrong* and *absent*---and hence both *recoverable* and "
    "wrong-of-attempted---arithmetically immune. We confirmed this rather than "
    "asserting it, by calling the grader once per record and parsing the "
    "identical reply bytes with both orders: the two parses give identical "
    "wrong, absent and recoverable counts, and the published data contains 26 "
    "partials, which the buggy order cannot emit. The mixed grading prompt is "
    "what moved: wrong-of-attempted was 19.5% [12.6, 29.1] pooled and is 13.6% "
    "[7.8, 22.7] under one protocol. Reassuringly, *recoverable* is unchanged "
    "at 56.5%, and the class ordering in Table 6---the result we rest "
    "on---is identical under both.\n"
    "\n"
    "Every grade behind these numbers is a majority of three independent "
    "grading calls, not a single call. We adopted voting because this same "
    "experiment measured the grader changing 18 of 124 grades when simply asked "
    "again, and a paper reporting judge instability cannot rest its own "
    "headline on one draw. The graders were unanimous on 91.7% of the 84 "
    "model-graded records, nothing was left unresolved after five draws, and "
    "voting moved wrong-of-attempted from 15.7% on a single draw to "
    "13.6%---the direction one expects if single draws occasionally overstate "
    "severity.",
))

EDITS.append((
    "artifact-detail-pct",
    "decisions (56.0%) are only modestly better than artifact details (48.0%)",
    "decisions (56.0%) are only modestly better than artifact details (52.0%)",
))

EDITS.append((
    "consequential-figure",
    "The 19.5% wrong-of-attempted rate is the most consequential figure in this section.",
    "The 13.6% wrong-of-attempted rate is the most consequential figure in this section.",
))

EDITS.append((
    "judge-scoping",
    "**Second and third judge.** Because one judge produced that figure, we "
    "re-graded with two stronger models. The 34 records that the pipeline "
    "assigns *absent* by rule, without a model call, cannot disagree and are "
    "excluded, leaving 90 model-graded records.",
    "**Second and third judge.** Because a single judge produced these figures, "
    "we re-graded with two stronger models. This check predates the protocol "
    "correction above and was run against the pooled answers, so the baseline "
    "it moves from is the pooled 19.5%, not the 13.6% we now report; we give it "
    "unchanged rather than re-running it, because what it establishes is a "
    "property of graders that the correction does not affect. Of those pooled "
    "records, the 34 the pipeline assigns *absent* by rule, without a model "
    "call, cannot disagree and are excluded, leaving 90 model-graded.",
))

EDITS.append((
    "judge-conclusion",
    "The judges differ on severity over the same items, not on which items are "
    "bad. We therefore keep 19.5% as the harshest of three readings, and treat "
    "it as an upper bound rather than adopting the more flattering numbers.",
    "The judges differ on severity over the same items, not on which items are "
    "bad.\n"
    "\n"
    "Two conclusions follow, and they pull in opposite directions, which is why "
    "we state both. The re-graders' leniency is a property of the grading "
    "model, so we decline to adopt 6.7% or 4.4%: a more permissive judge is not "
    "evidence that rollups are more accurate. The protocol correction above is "
    "different in kind, because it holds the judge fixed and removes an "
    "inconsistency in how that judge was asked, so 13.6% supersedes the pooled "
    "19.5% on the merits rather than on flattery. Read together, "
    "wrong-of-attempted is somewhere between roughly 4% and 20% depending on "
    "how severe a grader one accepts, with 13.6% the value under our documented "
    "protocol and 19.5% the harshest reading we obtained. We treat the "
    "qualitative claim as the finding---that a non-trivial minority of "
    "attempted questions are answered wrongly rather than omitted---and treat "
    "the point estimate as grader-dependent.",
))

RAW_SECTION = """### What the raw transcript would have answered

A wrong-of-attempted rate invites an obvious question that our design, as originally run, could not answer: is the alternative perfect? If a rollup answers 13.6% of attempted questions wrongly, a reader is entitled to ask what fraction the raw transcript would get wrong, and nothing in a rollup-only arm establishes that the answer is zero. Every retention number above is an absolute against an unmeasured baseline. So we added the missing arm.

**Design.** The questions, the ground-truth answers, the judge model, the answering prompt, the grading prompt, the grade parser and the majority-of-three voting are all held fixed from the rollup arm. The answering prompt differs in exactly the two words that name the context---a transcript rather than a compressed note---and the transcript is passed whole, with no clipping. Only the context varies, which makes this a paired comparison on identical items rather than two separately calibrated measurements. We deliberately kept the original judge model rather than upgrading to a larger-context one, because a bigger model would have varied model capability and context source simultaneously and left the contrast unattributable.

**Coverage.** 89 of the 124 questions, spanning 18 of the 25 sessions, have a transcript that fits the model's context window. The 35 excluded questions belong to the largest sessions; the practical ceiling we measured is around 2.2 MB of transcript JSONL, above which the request is refused outright. We report the paired contrast only on the 89, and we never mix in the excluded items.

**Result.** On the same 89 questions, the raw transcript makes 95.5% [89.0, 98.2] of the ground truth recoverable against the rollup's 68.5% [58.3, 77.2], and its wrong-of-attempted rate is 1.2% [0.2, 6.3] against 7.6% [3.3, 16.5]. Paired, the raw transcript recovers 25 questions the rollup loses while losing 1 the rollup recovers, with 60 recovered by both and 3 by neither (exact binomial on the 26 discordant pairs, two-sided $p = 8.1\\times10^{-7}$). The gap is widest exactly where the rollup is weakest: *sequence* rises from 52.9% to 94.1% and *artifact_detail* from 61.1% to 100.0%, while *outcome*, which the schema prompts for directly, was already at 88.9% and reaches 100.0%. So the answer to the question this subsection opens with is no---but it is much closer to yes than to the rollup, and the ordering of what rollups lose is a real compression artifact rather than an artifact of the probe.

**The failure modes are not the same kind.** This is the part we consider most consequential, and it is not visible in the rates. All 4 questions the raw transcript fails to recover come from its largest inputs (median 1.48 MB against 0.43 MB for those it recovers), and every one of them is a *refusal*: the model answers "NOT ANSWERABLE" or states that the transcript contains no such thing. Even the single record graded *wrong* is a confident denial that the material exists, which the grader scored as contradicting the ground truth. On inspection, then, the raw transcript's hallucination count on this set is zero, and the 1.2% is an artifact of scoring a refusal as a contradiction. The two context sources fail in opposite ways. A rollup's characteristic failure is confident wrongness, which a calling agent cannot detect. A raw transcript's characteristic failure is refusal on long input, which it can. For a system that must act on what it retrieves, that asymmetry matters more than the 27-point gap in recoverable.

**Two limitations, both of which cut against the raw arm's 95.5%.** First, this is an oracle condition and its ceiling is an instrument artifact. The ground-truth answers were extracted *from* the same transcripts the raw arm is shown, and from only the first 30,000 characters of each, so the questions systematically probe early-session content that is guaranteed to be present in the context. 95.5% is therefore close to the highest score the instrument can return, and it is not a claim about answering arbitrary questions from a transcript. Second, the 89-question set is selected in the rollup's favour: it excludes the largest sessions, and the rollup scores better on the survivors than overall (68.5% against 56.5% on all 124). The measured gap is thus conservative as an estimate of the full-corpus gap. But the same selection cuts the other way for extrapolation---the raw arm's failures concentrate in large transcripts---so 95.5% must not be read forward onto the multi-megabyte sessions it excludes.

**Cost.** The raw arm consumed 25.4M input tokens to answer 89 questions, roughly $65 at the judge model's uncached input price, against a negligible amount for the same 89 questions answered from rollups. That ratio is the paper's central economic claim restated as a retrieval experiment: the raw transcript is the more accurate context and it is not close to affordable at this frequency.

## RQ4: Retrieval"""

EDITS.append(("raw-arm-section", "## RQ4: Retrieval", RAW_SECTION))

EDITS.append((
    "where-not-to-use",
    "**Where it should not be used.** A rollup answered 19.5% of attempted "
    "questions wrongly. That rules out unsupervised use in any loop where a "
    "false premise is expensive",
    "**Where it should not be used.** A rollup answered 13.6% of attempted "
    "questions wrongly, and the raw transcript it replaces answered 1.2% "
    "wrongly on the subset where both fit---where, on inspection, even that "
    "1.2% is a refusal rather than a fabrication. The difference is not merely "
    "one of degree: substituting a rollup for a transcript trades a failure "
    "mode the calling agent can detect for one it cannot. That rules out "
    "unsupervised use in any loop where a false premise is expensive",
))

EDITS.append((
    "self-consistency",
    "re-grading RQ3 with two stronger models makes rollups look *better*, not "
    "worse, so the figure we report is the least rollup-flattering of the three "
    "available readings. That is the right choice under this threat, but it "
    "does not settle it---all three judges come from the same family, and a "
    "more capable same-family judge being more forgiving is exactly what "
    "family-level self-consistency would predict.",
    "re-grading RQ3 with two stronger models makes rollups look *better*, not "
    "worse. We decline to adopt those readings, but we also cannot claim the "
    "figure we report is the least rollup-flattering available, since the "
    "pooled 19.5% is harsher than the 13.6% we report; what we claim is that "
    "13.6% is the value under a single documented protocol, and that the spread "
    "across judges is reported rather than selected from. All the judges come "
    "from the same family, and a more capable same-family judge being more "
    "forgiving is exactly what family-level self-consistency would predict. The "
    "raw-transcript arm above is a partial check in the useful direction: there "
    "the same model family, asked the same questions by the same grader, judges "
    "the *uncompressed* context far better than the rollup, which is the "
    "opposite of what a pro-rollup self-consistency bias would produce.",
))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("path", nargs="?", default=NOTE)
    args = ap.parse_args()

    p = Path(args.path)
    text = p.read_text()
    original = text

    failures = []
    for label, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            failures.append(f"  {label}: matched {n} times, expected 1")
            continue
        text = text.replace(old, new, 1)
        print(f"  ok  {label}")

    if failures:
        print("\nFAILED -- no edit applied:")
        for f in failures:
            print(f)
        sys.exit(1)

    # Postcondition: assert the claim disappeared where it had to, rather than
    # trusting that the replacements ran. A stale 19.5% left in the abstract or
    # the discussion is the exact defect this patch exists to remove.
    for banned, where in [("19.5% are answered wrongly", "abstract"),
                          ("19.5% wrong-of-attempted rate is the most", "RQ3 body"),
                          ("A rollup answered 19.5%", "discussion")]:
        if banned in text:
            print(f"FATAL: '{banned}' still present in {where}")
            sys.exit(1)
    if "### What the raw transcript would have answered" not in text:
        print("FATAL: new subsection missing")
        sys.exit(1)

    print(f"\n{len(EDITS)} edits apply cleanly; "
          f"{len(original)} -> {len(text)} chars")
    if args.apply:
        p.write_text(text)
        print(f"wrote {p}")
    else:
        print("--check only, file untouched (pass --apply to write)")


if __name__ == "__main__":
    main()
