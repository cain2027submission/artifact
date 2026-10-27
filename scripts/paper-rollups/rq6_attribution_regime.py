#!/usr/bin/env python3
"""RQ6: does snippet-sized memory fail at everything, or only at DETAIL?

`rq3_read_budget_arms.py` answered a narrow question and answered it decisively.
At the budget the live retrieval hook actually injects, a path plus one heading
answered 0 of 89 detail questions -- indistinguishable from the `noop` and
`noop_opaque` controls -- and once a 500-char body was included the rollup note
(32.6% recoverable) and a byte-matched chunk of the tool-stripped transcript
(37.1%) were statistically tied at paired p=0.60. Those are detail-recovery
questions: a path, a number, an ordering, a dead end.

That leaves one regime untested, and it is the only one where a rollup could
still beat a transcript on anything: ATTRIBUTION. A rollup note's claim-shaped
`##` heading IS the session's finding stated in one line, where a 500-char slice
of a stripped transcript is an arbitrary mid-session chunk. So the hypothesis is
that at snippet budget the rollup wins attribution by a wide margin even though
it ties on detail -- a two-regime result. The alternative is that it ties here
too, which would mean rollups have no surviving advantage at any read budget.
Both outcomes are reportable and NOTHING here is tuned toward the first.

Two tasks per session, both graded WITHOUT a model judge:

  identification  Given the snippet and 8 candidate session-note paths, name the
                  session it came from. Graded by normalised note-path match:
                  correct iff exactly one candidate basename appears in the reply
                  and it is the right one. Chance is 1/8 = 12.5%.
  finding_recall  In one sentence, what did that session conclude? Graded by
                  content-word overlap against that note's own claim heading,
                  pre-registered threshold 0.5 of the heading's content words.

Five arms, one shared budget of MAX_BODY_CHARS = 500 chars (verified against
line 227 of ~/.claude/scripts/prompt-context.py, not assumed), one draw per cell:

  rollup_snippet      claim heading + that section's body, <=500 chars
  stripped_chunk      500-char window centred on the stripped transcript
  wrong_note_snippet  a DIFFERENT session's rollup_snippet, graded against the
                      true session. A GATE, not a datapoint.
  noop                the prior experiment's `noop`, scrubbed
  noop_opaque         a true empty control

Validity machinery, because every one of these is a place this could have faked
a positive:

  * THE GATE IS ENFORCED IN CODE. `gate_verdict()` decides, and `summarise()`
    refuses to emit an interpretation when the gate fails. If
    `wrong_note_snippet` identification exceeds the best no-op by more than
    GATE_MARGIN_PP, the question set is guessable and the other arms are not
    presented as meaningful. It is reported first in the summary either way.
  * THE PATH IS THE LABEL, so no arm may carry it. `scrub()` removes session
    uuids, 8-hex prefixes, `sessions/<stem>.md` filenames, `.jsonl` transcript
    paths and `.claude/worktrees/<name>/` prefixes from EVERY arm identically,
    including the controls. This is why the prior experiment's two controls
    could not be reused verbatim: both were built as `- <note path>`, which for
    an attribution task is the answer key. That deviation is recorded in the
    output JSON under `deviations` rather than left for a reader to notice.
  * THE FROZEN MANIFEST defines the session set. `sessions/` grew from 499 notes
    at snapshot time to ~1,114, so globbing it is a known reproducibility defect
    in this paper. Every drop is recorded by reason, including the one pruned
    transcript (bb26f10d).
  * FINDING RECALL IS CONTAMINATED FOR ONE ARM BY CONSTRUCTION. The rollup
    snippet opens with the very heading the grader scores against, so its score
    is a copy task, not recall. That is measured rather than hidden:
    `heading_in_snippet` is reported per arm, and the honest reading of the
    finding-recall row is "the rollup snippet CONTAINS the finding at 500 chars
    and the stripped chunk does not", which is a structural fact about the
    artifacts, not a judgement about the model.

Usage:
    rq6_attribution_regime.py --prereg              # write the prediction first
    rq6_attribution_regime.py --dry-run             # build contexts, no calls
    rq6_attribution_regime.py --smoke               # 3 sessions + degenerates
    rq6_attribution_regime.py --limit 40            # resumable batch
    rq6_attribution_regime.py --summarise-only      # recompute from the jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from paths import fold_repo_aliases  # noqa: E402
from rq3_raw_transcript_arm import call_llm, load_records  # noqa: E402
from rq3_read_budget_arms import boring_heading, clean, clean_body  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402

REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
MANIFEST = DATA / "corpus_snapshot.json"
OUT = DATA / "rq6_attribution_arms.jsonl"
REPORT = DATA / "rq6_attribution_regime.json"

JUDGE_MODEL = "claude-sonnet-5"

# Verified 2026-09-29 against the live prompt-context hook.
# Taken from the live hook and from rq3_read_budget_arms.py so the numbers here
# are directly comparable to that experiment's 32.6%/37.1%.
MAX_BODY_CHARS = 500
BUDGET = MAX_BODY_CHARS

ARMS = ("wrong_note_snippet", "rollup_snippet", "stripped_chunk",
        "noop", "noop_opaque")
TASKS = ("identification", "finding_recall")

N_CANDIDATES = 8
CHANCE_PCT = 100.0 / N_CANDIDATES

# Pre-registered BEFORE the batch, in the same commit as the prereg stub.
OVERLAP_THRESHOLD = 0.5
OVERLAP_REPORTED = (0.3, 0.5, 0.7)
# The gate: wrong_note_snippet may not beat the better no-op by more than this
# many percentage points on identification, or the question set is guessable.
GATE_MARGIN_PP = 10.0

PREREGISTRATION = {
    "recorded": "before the full batch, committed ahead of any result rows",
    "prediction": ("rollup_snippet >> stripped_chunk on BOTH identification and "
                   "finding_recall, and both >> noop / noop_opaque / "
                   "wrong_note_snippet"),
    "identification_graded_by": ("deterministic normalised note-path match, no "
                                 "judge; chance = 12.5% at 8 candidates"),
    "finding_recall_graded_by": ("deterministic content-word overlap against the "
                                 "note's own claim heading, threshold 0.5"),
    "budget_chars": BUDGET,
    "gate": ("wrong_note_snippet identification must not exceed the better no-op "
             "by more than %.1f pp, else no arm is presented as meaningful"
             % GATE_MARGIN_PP),
    "falsifier": ("a tie between rollup_snippet and stripped_chunk means rollups "
                  "have no surviving advantage at any read budget, and is to be "
                  "reported as the finding"),
}

STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "what", "which", "who", "whom", "why", "how", "when", "where", "did", "do",
    "does", "done", "had", "has", "have", "not", "no", "yes", "he", "she",
    "they", "them", "his", "her", "their", "you", "your", "i", "we", "our",
    "us", "me", "so", "there", "here", "about", "into", "over", "under",
    "after", "before", "during", "while", "also", "only", "very", "more",
    "most", "some", "any", "all", "both", "each", "other", "such", "own",
    "same", "too", "can", "will", "just", "should", "would", "could", "may",
    "might", "must", "one", "two", "session", "sessions", "conclude",
    "concluded", "conclusion", "snippet", "answer",
}

DATE_ONLY = re.compile(r"^[\d\s\-/.:]+$")

# ---------------------------------------------------------------- scrubbing

UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                  re.I)
HEX8 = re.compile(r"\b[0-9a-f]{8}\b", re.I)
JSONL = re.compile(r"[^\s\"'`]*\.jsonl\b")
SESSION_MD = re.compile(r"(?:sessions?[/\\])[^\s\"'`)\]]*\.md\b", re.I)
WORKTREE = re.compile(r"(?:\.claude[/\\]worktrees[/\\][^/\\\s]+[/\\])")
LINESUF = re.compile(r"(\.(?:md|py|tex|json|jsonl|sh|txt)):\d+\b", re.I)
FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def normalise_vault(text: str) -> str:
    """Fold the repo's two spellings and drop worktree prefixes and :LINE tails.

    This corpus spells the notes repo in two casings and 82.8% of mentions use
    the second, so a comparison that does not fold them undercounts matches.
    """
    text = WORKTREE.sub("", text or "")
    text = LINESUF.sub(r"\1", text)
    return fold_repo_aliases(text or "")


def scrub(text: str, note_stem: str = "", session_id: str = "") -> str:
    """Remove every direct identifier of the source session from a context.

    Applied identically to all five arms. Without it the experiment is void:
    every one of these tokens names the answer to the identification task
    outright, and stripped transcripts carry `cwd`, the session uuid and the
    `.jsonl` path in plain text.
    """
    text = normalise_vault(text or "")
    if note_stem:
        text = re.sub(re.escape(note_stem), "<REDACTED>", text, flags=re.I)
        loose = note_stem.replace("_", "[ _-]").replace("-", "[ _-]")
        try:
            text = re.sub(loose, "<REDACTED>", text, flags=re.I)
        except re.error:
            pass
    if session_id:
        text = re.sub(re.escape(session_id), "<REDACTED>", text, flags=re.I)
        text = re.sub(re.escape(session_id[:8]), "<REDACTED>", text, flags=re.I)
    text = UUID.sub("<REDACTED>", text)
    text = JSONL.sub("<REDACTED>", text)
    text = SESSION_MD.sub("sessions/<REDACTED>.md", text)
    text = HEX8.sub("<REDACTED>", text)
    return text


def leak_scan(context: str, note_stem: str, session_id: str) -> list:
    """Names of identifiers still present in a built context. Must come back []."""
    low = (context or "").lower()
    found = []
    if note_stem and note_stem.lower() in low:
        found.append("note_stem")
    if session_id and session_id.lower() in low:
        found.append("session_uuid")
    if session_id and session_id[:8].lower() in low:
        found.append("session_hex8")
    if ".jsonl" in low:
        found.append("jsonl_path")
    if UUID.search(context or ""):
        found.append("bare_uuid")
    return found


# -------------------------------------------------------- candidate labelling

SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
COMMAND_STUB = re.compile(r"<(command-name|command-message|local-command|"
                          r"user-prompt-submit-hook)[^>]*>", re.I)
MIN_INTENT_CHARS = 30
MAX_INTENT_CHARS = 150
# A candidate label has to distinguish its session. These thresholds are fixed
# before the batch and are computed from the labels only, never from an arm.
MIN_INTENT_TOKENS = 5          # content words in the opening request
MIN_INTENT_RARE_TOKENS = 3     # of which this many must be corpus-rare
INTENT_DF_CEILING = 0.10       # "rare" = present in <=10% of candidate labels
INTENT_DUP_JACCARD = 0.8       # two labels this similar are indistinguishable
DISTRACTOR_MAX_JACCARD = 0.6   # a distractor may not near-duplicate the true one


def first_user_prompt(raw: str) -> str:
    """The request that started the session, read from the transcript.

    This is the label a candidate session is identified BY, and it has to come
    from a source independent of the rollup note. The obvious label -- the note's
    own path -- is unusable: a rollup filename is mechanically derived from the
    note's claim heading, so `sessions/..._rrf-treats-correlated-evidence-as-
    independent.md` beside a snippet opening "rrf treats correlated evidence as
    independent votes" is not an attribution task, it is a string match against
    the answer key. The date prefix leaks the same way, because a rollup body
    carries a `**session:** <date>` line. Both are removed from the label.
    """
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("isMeta") or rec.get("isCompactSummary"):
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        parts = []
        for b in (content if isinstance(content, list) else [content]):
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "tool_result":
                    parts = []
                    break
                if b.get("type") == "text":
                    parts.append(b.get("text") or "")
        text = " ".join(p for p in parts if p)
        text = SYSTEM_REMINDER.sub(" ", text)
        if COMMAND_STUB.search(text):
            continue
        if text.lstrip().startswith("Caveat:"):
            continue
        text = clean(text)
        if len(text) >= MIN_INTENT_CHARS:
            return truncate(text, MAX_INTENT_CHARS)
    return ""


def candidate_label(session_id: str, intent: str) -> str:
    """`sessions/<8-hex>.md` plus the session's opening request.

    No date and no claim slug, so nothing in this label can be matched to a
    snippet without actually understanding what the session was about.
    """
    # Capped here as well as in the extractor: one very long opening request must
    # not crowd the other seven candidates out of the prompt.
    return 'sessions/%s.md - started from: "%s"' % (
        session_id[:8], truncate(intent, MAX_INTENT_CHARS))


# ------------------------------------------------------------ note handling

def claim_headings(note_text: str) -> list:
    """The `##` headings that state a finding rather than label a section.

    `boring_heading()` is the live hook's own filter and is reused rather than
    reimplemented. Two extra rejections are needed on this corpus: a bare-date
    heading (`## 2026-09-18`) passes boring_heading because it contains no
    structural word, and a two-word heading carries no recoverable claim.
    """
    out = []
    for h in re.findall(r"^##\s+(.+?)\s*$", note_text or "", re.M):
        h = clean(h)
        if not h or boring_heading(h) or DATE_ONLY.match(h):
            continue
        if len(content_tokens(h)) < 3:
            continue
        out.append(h)
    return out


def section_body(note_text: str, heading: str) -> str:
    """The prose under one `##` heading, cleaned the way the hook cleans a body."""
    pat = re.compile(r"^##\s+" + re.escape(heading) + r"\s*$(.*?)(?=^##\s|\Z)",
                     re.M | re.DOTALL)
    m = pat.search(note_text or "")
    return clean_body(m.group(1)) if m else ""


def truncate(text: str, budget: int = BUDGET) -> str:
    """Word-boundary truncation, matching rq3_read_budget_arms."""
    text = text or ""
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    cut = text[:budget]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def centre_window(text: str, budget: int = BUDGET) -> str:
    """The `budget`-char window centred on the text, snapped to word boundaries.

    Deliberately arbitrary and deliberately NOT oracle-selected. The prior
    experiment could pick the window by keyword overlap with a per-item question;
    there is no such question here, and picking the tail (where a conclusion
    lives) would hand the stripped arm a selection the production system cannot
    perform. The centre is the honest choice and the brief's own framing of what
    a transcript chunk is. A tail window was NOT run; see `limitations`.
    """
    text = text or ""
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text
    start = (len(text) - budget) // 2
    window = text[start:start + budget]
    if " " in window:
        head, _, _ = window.partition(" ")
        if len(head) < budget:
            window = window[len(head) + 1:]
        window = window.rsplit(" ", 1)[0] if " " in window else window
    return window


# ----------------------------------------------------------------- grading

def content_tokens(text: str) -> set:
    """Lowercase content words, >=3 chars, stopped, trailing plural folded."""
    out = set()
    for w in re.split(r"[^a-z0-9]+", (text or "").lower()):
        if len(w) < 3 or w in STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap(reference: str, answer: str) -> float:
    """Fraction of the reference's content words present in the answer."""
    ref = content_tokens(reference)
    if not ref:
        return 0.0
    return len(ref & content_tokens(answer)) / len(ref)


def norm_path_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalise_vault(text or "").lower())


def grade_identification(reply: str, true_id: str, candidate_ids: list) -> dict:
    """Deterministic. Correct iff EXACTLY ONE candidate is named and it is right.

    Matching is on the 8-hex session id, which is what the brief specifies and
    which is immune to the ways a model reformats a path: an alphanumeric fold
    absorbs the repo name's two casings, a `.claude/worktrees/<name>/` prefix,
    a `:LINE` suffix and any punctuation.
    Requiring exactly one match stops a reply that hedges across candidates from
    being credited -- naming all eight is not an identification.
    """
    folded = norm_path_token(reply)
    named = [c for c in candidate_ids
             if norm_path_token(c[:8]) and norm_path_token(c[:8]) in folded]
    named = sorted({c[:8] for c in named})
    correct = len(named) == 1 and named[0] == true_id[:8]
    return {"named": named, "n_named": len(named), "correct": bool(correct)}


def grade_finding(reply: str, reference: str) -> dict:
    """Deterministic overlap against the note's own claim heading."""
    body = (reply or "").strip()
    refused = bool(re.search(r"\bNOT[ _]ANSWERABLE\b", body, re.I))
    score = 0.0 if refused else overlap(reference, body)
    return {"overlap": round(score, 4), "refused": refused,
            "recalled": bool(score >= OVERLAP_THRESHOLD)}


# --------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def exact_binomial(b: int, c: int) -> float:
    """Two-sided exact binomial over discordant pairs, verbatim from the sibling."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(p, 1.0)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "pct": round(100.0 * k / n, 1) if n else 0.0,
            "ci95": [lo, hi]}


# ------------------------------------------------------------ item building

def eligible_sessions(manifest: dict) -> tuple:
    """The frozen session set, with every drop recorded by reason.

    Never globs `sessions/`. That folder grew from 499 notes at snapshot time to
    roughly 1,114 afterwards, so a glob silently widens the corpus and is a known
    reproducibility defect in this paper.
    """
    rollups = manifest["rollup_note_paths"]
    by_stem = {}
    for t in manifest["transcripts"]:
        by_stem.setdefault(Path(t).stem, []).append(t)

    kept, drops = [], {}

    def drop(reason, sid):
        drops.setdefault(reason, []).append(sid)

    for sid in sorted(manifest["paired_session_ids"]):
        note = rollups.get(sid)
        if not note or not Path(note).exists():
            drop("no_rollup_note_on_disk", sid)
            continue
        live = [Path(p) for p in by_stem.get(sid, []) if Path(p).exists()]
        if not live:
            drop("transcript_pruned", sid)
            continue
        note_text = Path(note).read_text(errors="replace")
        heads = claim_headings(note_text)
        if not heads:
            drop("no_claim_shaped_heading", sid)
            continue
        tpath = sorted(live, key=lambda q: -q.stat().st_size)[0]
        raw = tpath.read_text(errors="replace")
        stripped, _ = strip_transcript(raw)
        if len(stripped) < BUDGET:
            drop("stripped_transcript_under_budget", sid)
            continue
        intent = first_user_prompt(raw)
        if not intent:
            drop("no_usable_opening_request", sid)
            continue
        kept.append({"session_id": sid, "note": note, "note_stem": Path(note).stem,
                     "heading": heads[0], "n_claim_headings": len(heads),
                     "transcript": str(tpath), "stripped_chars": len(stripped),
                     "intent": intent,
                     "_note_text": note_text, "_stripped": stripped})

    # Second pass: an item is only answerable if its candidate label actually
    # distinguishes it. Two failure modes showed up in the dry run and both make
    # an item unanswerable FOR EVERY ARM, so excluding them is not tuning:
    #
    #   * a generic opening request. "PULL LATEST. START ON THE NEW TASK U PULLD."
    #     and "get latest, then start on new task" describe no session in
    #     particular; they are a known prompt class in this corpus.
    #   * two sessions opening with the SAME request, which are then
    #     indistinguishable candidates no matter what the snippet says.
    #
    # Both criteria are computed from the labels alone. They never look at any
    # arm's context, so neither can favour one arm over another.
    df = {}
    for k in kept:
        for w in content_tokens(k["intent"]):
            df[w] = df.get(w, 0) + 1
    n_pool = len(kept) or 1
    surviving, seen = [], []
    for k in kept:
        toks = content_tokens(k["intent"])
        # max(1, ...) keeps the criterion meaningful on a small pool, where a
        # proportional ceiling below 1 would reject every word including the
        # ones unique to a single session.
        ceiling = max(1.0, INTENT_DF_CEILING * n_pool)
        rare = [w for w in toks if df[w] <= ceiling]
        if len(toks) < MIN_INTENT_TOKENS or len(rare) < MIN_INTENT_RARE_TOKENS:
            drop("opening_request_not_discriminative", k["session_id"])
            continue
        dup = any(jaccard(toks, p) >= INTENT_DUP_JACCARD for p in seen)
        if dup:
            drop("duplicate_opening_request", k["session_id"])
            continue
        seen.append(toks)
        k["_intent_tokens"] = toks
        surviving.append(k)
    return surviving, drops


def build_items(manifest: dict, limit: int = 0) -> tuple:
    kept, drops = eligible_sessions(manifest)
    by_id = {k["session_id"]: k for k in kept}
    n = len(kept)
    items = []
    for i, k in enumerate(kept):
        # Deterministic derangement: the next session in sorted order donates the
        # wrong snippet. No session ever donates to itself.
        donor = kept[(i + 1) % n] if n > 1 else None

        heading, body = k["heading"], section_body(k["_note_text"], k["heading"])
        rollup_raw = heading + ". " + body if body else heading
        rollup = scrub(truncate(rollup_raw), k["note_stem"], k["session_id"])

        if donor is not None:
            d_body = section_body(donor["_note_text"], donor["heading"])
            d_raw = donor["heading"] + ". " + d_body if d_body else donor["heading"]
            # Scrubbed against BOTH sessions: the donor's identifiers must go too,
            # or the gate arm leaks the donor and reads as a confident wrong.
            wrong = scrub(scrub(truncate(d_raw), donor["note_stem"],
                                donor["session_id"]),
                          k["note_stem"], k["session_id"])
        else:
            wrong = ""

        stripped = scrub(centre_window(k["_stripped"]), k["note_stem"],
                         k["session_id"])

        contexts = {
            "rollup_snippet": rollup,
            "stripped_chunk": stripped,
            "wrong_note_snippet": wrong,
            # The prior experiment's `noop` was `- <note path>`. For an
            # attribution task that string IS the label, so it is carried through
            # the same scrubber every other arm gets, which is what reduces it to
            # this. Recorded under `deviations`.
            "noop": "- sessions/<REDACTED>.md",
            "noop_opaque": "(no context was retrieved for this prompt)",
        }

        # The donor is excluded from the distractors so that the
        # wrong_note_snippet arm measures pure guessing: its snippet points at a
        # session that is not on the list, so anything above chance there is the
        # question wording, not the context.
        rng = random.Random("rq6|" + k["session_id"])
        pool = sorted(s for s in by_id
                      if s != k["session_id"]
                      and (donor is None or s != donor["session_id"])
                      and jaccard(by_id[s]["_intent_tokens"],
                                  k["_intent_tokens"]) < DISTRACTOR_MAX_JACCARD)
        # Order is a per-item seeded shuffle, NOT a sort by id. Sorting by id
        # looked position-neutral but is not: items are built in sorted id order,
        # so the true session of an early item is smaller than most of its own
        # distractors and lands first. The smoke run caught this -- a zero-
        # information `noop` scored 33.3% by picking the first candidate. A
        # shuffle makes the true position uniform and independent of the id.
        cands = rng.sample(pool, min(N_CANDIDATES - 1, len(pool))) \
            + [k["session_id"]]
        rng.shuffle(cands)
        labels = [candidate_label(s, by_id[s]["intent"]) for s in cands]

        items.append({
            "session_id": k["session_id"], "note_stem": k["note_stem"],
            "note": k["note"], "transcript": k["transcript"],
            "heading": heading, "n_claim_headings": k["n_claim_headings"],
            "stripped_chars": k["stripped_chars"], "intent": k["intent"],
            "donor_session_id": donor["session_id"] if donor else None,
            "candidates": cands, "candidate_labels": labels,
            # 1-based position of the true session in the shown list. Recorded so
            # positional guessing is measurable rather than assumed absent.
            "true_position": cands.index(k["session_id"]) + 1,
            "contexts": contexts,
            "leaks": {a: leak_scan(contexts[a], k["note_stem"], k["session_id"])
                      for a in ARMS},
            "heading_in_snippet": {
                a: round(overlap(heading, contexts[a]), 4) for a in ARMS},
            # How much of the TRUE candidate's label the snippet already contains.
            # This is the mechanism behind any identification score, so it is
            # measured per arm rather than asserted.
            "intent_in_snippet": {
                a: round(overlap(k["intent"], contexts[a]), 4) for a in ARMS},
        })
    if limit:
        items = items[:limit]
    return items, drops


# ------------------------------------------------------------------ prompts

IDENT_PROMPT = """A snippet of stored memory from ONE past agent session was retrieved and injected below. Each candidate below is a different past session, listed with its note path and the request that started it.

CANDIDATE SESSIONS:
{candidates}

RETRIEVED SNIPPET:
{context}

Which candidate session did the snippet come from? Reply with your reasoning in at most two sentences, then on the FINAL line write only that candidate's note path, exactly as listed. If the snippet gives you nothing to go on, still pick the single best candidate."""

FINDING_PROMPT = """A snippet of stored memory from ONE past agent session was retrieved and injected below.

RETRIEVED SNIPPET:
{context}

In ONE sentence, what did that session conclude? State the conclusion itself, not what the session was about. If the snippet does not say what was concluded, reply with exactly: NOT ANSWERABLE"""


def ask(task: str, item: dict, arm: str):
    ctx = item["contexts"][arm]
    if task == "identification":
        prompt = IDENT_PROMPT.format(
            candidates="\n".join("- " + l for l in item["candidate_labels"]),
            context=ctx)
    else:
        prompt = FINDING_PROMPT.format(context=ctx)
    reply, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=180)
    return reply, usage


# ------------------------------------------------------------------- gating

def gate_verdict(ident: dict) -> dict:
    """Decide in CODE whether the arms may be presented as meaningful.

    Checked here rather than by eye, because "I read the questions and they
    looked fine" is exactly the assurance that has failed in this paper before.

    TWO independent conditions, because the first alone has a hole that the test
    suite found: comparing wrong_note_snippet against the no-ops says nothing if
    the no-ops are themselves at ceiling. A context carrying no information about
    the session must land near the 12.5% floor; if it does not, the candidate set
    or the prompt is leaking and the margin against it is meaningless.
    """
    gate = ident.get("wrong_note_snippet", {}).get("pct", 0.0)
    noops = [ident.get(a, {}).get("pct", 0.0) for a in ("noop", "noop_opaque")]
    best_noop = max(noops) if noops else 0.0
    margin = round(gate - best_noop, 1)
    noop_excess = round(best_noop - CHANCE_PCT, 1)
    wrong_ok = margin <= GATE_MARGIN_PP
    noop_ok = noop_excess <= GATE_MARGIN_PP
    passed = wrong_ok and noop_ok
    if passed:
        meaning = ("wrong_note_snippet sits within %.1f pp of the better no-op and "
                   "the no-ops sit within %.1f pp of the %.1f%% chance floor, so "
                   "identification is not recoverable from the prompt wording and "
                   "the other arms may be read as measuring attribution"
                   % (GATE_MARGIN_PP, GATE_MARGIN_PP, CHANCE_PCT))
    elif not noop_ok:
        meaning = ("GATE FAILED: a zero-information no-op scored %.1f%% against a "
                   "%.1f%% chance floor, so the candidate set or the prompt leaks "
                   "the answer and NO arm in this table is meaningful"
                   % (best_noop, CHANCE_PCT))
    else:
        meaning = ("GATE FAILED: wrong_note_snippet beats the better no-op by "
                   "%.1f pp on identification, so the question set is guessable "
                   "and NO arm in this table is meaningful" % margin)
    return {
        "wrong_note_snippet_pct": gate,
        "best_noop_pct": best_noop,
        "chance_pct": round(CHANCE_PCT, 1),
        "margin_over_noop_pp": margin,
        "noop_excess_over_chance_pp": noop_excess,
        "margin_allowed_pp": GATE_MARGIN_PP,
        "wrong_note_within_margin": bool(wrong_ok),
        "noops_near_chance": bool(noop_ok),
        "passed": bool(passed),
        "meaning": meaning,
    }


# ---------------------------------------------------------------- reporting

def write_json_atomic(path: Path, payload: dict) -> None:
    """Never open a published result file in mode 'w'. Temp file, then replace."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_prereg() -> dict:
    """Read the pre-registration back rather than regenerating it.

    The point of a pre-registration is that it predates the data, so the stub on
    disk wins over the constant in this file if the two ever disagree.
    """
    if REPORT.exists():
        try:
            prior = json.loads(REPORT.read_text())
            if prior.get("preregistration"):
                return prior["preregistration"]
        except (json.JSONDecodeError, OSError):
            pass
    return PREREGISTRATION


def summarise(items: list, drops: dict, spend: dict, verbose: bool = True) -> dict:
    rows = load_records(OUT) if OUT.exists() else []
    cells = {}
    for r in rows:
        cells[(r.get("task"), r.get("arm"), r.get("session_id"))] = r

    attempted = [i["session_id"] for i in items]
    by_item = {i["session_id"]: i for i in items}

    complete = {}
    for task in TASKS:
        complete[task] = sorted(
            s for s in attempted
            if all((task, a, s) in cells for a in ARMS))

    out = {
        "experiment": "rq6_attribution_regime",
        "question": ("does snippet-sized memory fail at everything, or only at "
                     "detail? tests ATTRIBUTION at the same 500-char budget at "
                     "which rq3_read_budget_arms found rollup 32.6% and stripped "
                     "37.1% on detail recovery at paired p=0.60"),
        "preregistration": load_prereg(),
        "budget_chars": BUDGET,
        "budget_source": ("MAX_BODY_CHARS, verified at line 227 of "
                          "~/.claude/scripts/prompt-context.py"),
        "judge_model": JUDGE_MODEL,
        "grading": "deterministic, no model judge on either task",
        "n_candidates": N_CANDIDATES,
        "chance_identification_pct": round(CHANCE_PCT, 1),
        "overlap_threshold": OVERLAP_THRESHOLD,
        "n_eligible_sessions": len(items),
        "n_complete": {t: len(complete[t]) for t in TASKS},
        "dropped_sessions": {k: sorted(v) for k, v in sorted(drops.items())},
        "n_dropped": {k: len(v) for k, v in sorted(drops.items())},
        "spend": spend,
        "arms": {},
        "paired": {},
        "structural": {},
        "leaks": {},
    }

    ident, finding = {}, {}
    for arm in ARMS:
        ids = complete["identification"]
        k = sum(1 for s in ids if cells[("identification", arm, s)]["correct"])
        ident[arm] = rate(k, len(ids))
        multi = sum(1 for s in ids
                    if cells[("identification", arm, s)]["n_named"] != 1)
        ident[arm]["n_not_exactly_one_candidate_named"] = multi

        fs = complete["finding_recall"]
        scores = [cells[("finding_recall", arm, s)]["overlap"] for s in fs]
        kk = sum(1 for v in scores if v >= OVERLAP_THRESHOLD)
        finding[arm] = rate(kk, len(fs))
        finding[arm]["mean_overlap"] = round(sum(scores) / len(scores), 4) if scores else 0.0
        finding[arm]["at_threshold"] = {
            str(t): rate(sum(1 for v in scores if v >= t), len(scores))["pct"]
            for t in OVERLAP_REPORTED}
        finding[arm]["n_refused"] = sum(
            1 for s in fs if cells[("finding_recall", arm, s)]["refused"])
        idl = complete["identification"]
        out["structural"][arm] = {
            "mean_heading_overlap_with_injected_snippet": round(
                sum(by_item[s]["heading_in_snippet"][arm] for s in fs) / len(fs), 4)
            if fs else 0.0,
            "pct_items_snippet_contains_90pct_of_heading": round(
                100.0 * sum(1 for s in fs
                            if by_item[s]["heading_in_snippet"][arm] >= 0.9)
                / len(fs), 1) if fs else 0.0,
            # The identification mechanism, measured: how much of the true
            # candidate's label (its opening request) the snippet already carries.
            "mean_opening_request_overlap_with_injected_snippet": round(
                sum(by_item[s]["intent_in_snippet"][arm] for s in idl) / len(idl), 4)
            if idl else 0.0,
        }
        out["leaks"][arm] = sorted({l for i in items for l in i["leaks"][arm]})

    out["arms"]["identification"] = ident
    out["arms"]["finding_recall"] = finding
    out["gate"] = gate_verdict(ident)

    # Positional guessing, measured rather than assumed absent. The smoke run
    # showed a no-op scoring 33.3% purely because the candidate list had been
    # SORTED by id while items are built in id order, putting the true session
    # first in early items. Candidates are now shuffled per item; this block is
    # the standing check that the leak stayed closed.
    idl = complete["identification"]
    pos_expected = round(100.0 / N_CANDIDATES, 1)
    out["positional"] = {
        "pct_items_true_at_position_1": rate(
            sum(1 for s in idl if by_item[s]["true_position"] == 1),
            len(idl))["pct"],
        "expected_pct_if_position_is_uniform": pos_expected,
        "per_arm_pct_of_named_answers_at_position_1": {
            arm: (round(100.0 * sum(
                1 for s in idl
                if 1 in (cells[("identification", arm, s)].get(
                    "named_positions") or []))
                / len(idl), 1) if idl else 0.0) for arm in ARMS},
    }

    for task, key in (("identification", "correct"), ("finding_recall", "recalled")):
        ids = complete[task]
        for la, lb in (("rollup_snippet", "stripped_chunk"),
                       ("rollup_snippet", "noop"),
                       ("stripped_chunk", "noop"),
                       ("wrong_note_snippet", "noop"),
                       ("noop", "noop_opaque")):
            a_ok = {s for s in ids if cells[(task, la, s)][key]}
            b_ok = {s for s in ids if cells[(task, lb, s)][key]}
            b, c = len(b_ok - a_ok), len(a_ok - b_ok)
            out["paired"]["%s|%s_vs_%s" % (task, la, lb)] = {
                "n": len(ids), "both": len(a_ok & b_ok),
                "%s_only" % la: c, "%s_only" % lb: b,
                "neither": len(ids) - len(a_ok | b_ok), "discordant": b + c,
                "exact_binomial_two_sided_p": exact_binomial(b, c)}

    out["deviations"] = [
        ("the prior experiment's `noop` and `noop_opaque` were both built as "
         "`- <note path>`; for an attribution task the note path is the label, so "
         "they were carried through the same scrubber every arm gets. `noop` is "
         "the scrubbed path shape, `noop_opaque` a true empty context. Reusing "
         "them verbatim would have scored 100% by construction."),
        ("the session set is the frozen manifest's paired set restricted to notes "
         "with a claim-shaped `##` heading, because the hypothesis under test is "
         "about that heading; a note whose headings are all `Intent`/`Outcome` "
         "labels has no finding line and cannot test it. 159 of 275 paired "
         "sessions were dropped for that reason and the number is reported."),
        ("the stripped chunk is the centre window, not an oracle-selected one. "
         "The prior experiment picked its window by keyword overlap with a "
         "per-item question; there is no per-item question here."),
        ("identification is 8-way forced choice, so its floor is 12.5% and not 0. "
         "The prior experiment's tasks were open-ended."),
        ("candidates are labelled `sessions/<8-hex>.md` plus the session's OPENING "
         "REQUEST from the transcript, not by their note filename. The first dry "
         "run showed why: a rollup filename is mechanically derived from the "
         "note's claim heading, so the true candidate's path was a near-verbatim "
         "copy of the rollup snippet's own first line, and a rollup body's "
         "`**session:** <date>` line matched the date prefix too. Identification "
         "would have scored near ceiling for the rollup arm as a string match "
         "against the answer key, and the wrong_note_snippet gate could not have "
         "caught it because the donor's heading points at a candidate that is "
         "usually absent from the list. The label now comes from a source the "
         "rollup pipeline did not write."),
        ("the donor of the wrong snippet is excluded from that item's candidate "
         "list, so the gate arm measures guessing rather than confident-wrong."),
        ("candidate order is a per-item seeded shuffle. It was a sort by opaque "
         "id, which looked position-neutral and was not: items are built in "
         "sorted id order, so an early item's true session sorts ahead of most "
         "of its own randomly drawn distractors and lands first. The 3-session "
         "smoke run caught it -- a zero-information `noop` scored 33.3% by "
         "answering first-in-list, and the gate correctly refused to interpret "
         "the table. `positional` reports the residual per arm, and the 30 smoke "
         "rows taken under the sorted order were discarded to "
         "rq6_attribution_arms_smoke_discarded.jsonl rather than reused."),
    ]
    out["limitations"] = [
        ("finding_recall is contaminated for rollup_snippet BY CONSTRUCTION: the "
         "snippet opens with the heading the grade is computed against, so that "
         "cell measures copying, not recall. `structural` reports the "
         "contamination per arm and the honest reading is that the rollup snippet "
         "CONTAINS the finding at 500 chars while the transcript chunk does not."),
        ("a tail window of the stripped transcript was not run; a conclusion is "
         "likelier at the end of a session than at its centre, so the stripped "
         "arm's finding_recall number is a floor, not an estimate."),
        ("one draw per cell, no majority vote."),
        ("the candidate label is drawn from the transcript, so it shares an origin "
         "with the stripped arm's own document. A centre window rarely contains "
         "the opening request, but where it does the stripped arm is favoured. "
         "`structural.mean_opening_request_overlap_with_injected_snippet` "
         "quantifies this per arm; the bias runs AGAINST the pre-registered "
         "prediction, which is the safe direction for it to run."),
    ]

    i_roll = ident["rollup_snippet"]["pct"]
    i_strip = ident["stripped_chunk"]["pct"]
    p_ident = out["paired"]["identification|rollup_snippet_vs_stripped_chunk"][
        "exact_binomial_two_sided_p"]
    if not out["gate"]["passed"]:
        out["interpretation"] = (
            "NONE: the wrong_note_snippet gate failed, so the question set is "
            "guessable and no arm here is meaningful.")
    elif p_ident < 0.05 and i_roll > i_strip:
        out["interpretation"] = (
            "Two-regime finding: snippet-sized memory is sufficient for "
            "attribution (rollup %.1f%% vs stripped %.1f%%, paired p=%.3g) and "
            "useless for detail (32.6%% vs 37.1%%, p=0.60 in "
            "rq3_read_budget_summary.json)." % (i_roll, i_strip, p_ident))
    elif p_ident < 0.05 and i_strip > i_roll:
        out["interpretation"] = (
            "The stripped chunk BEATS the rollup snippet on attribution "
            "(%.1f%% vs %.1f%%, paired p=%.3g), which is the opposite of the "
            "pre-registered prediction: rollups have no surviving advantage at "
            "any read budget." % (i_strip, i_roll, p_ident))
    else:
        out["interpretation"] = (
            "Rollup and stripped tie on attribution (%.1f%% vs %.1f%%, paired "
            "p=%.3g), so rollups have no surviving advantage at any read budget."
            % (i_roll, i_strip, p_ident))

    f_roll = finding["rollup_snippet"]["pct"]
    f_strip = finding["stripped_chunk"]["pct"]
    p_find = out["paired"]["finding_recall|rollup_snippet_vs_stripped_chunk"][
        "exact_binomial_two_sided_p"]
    ctrl = max(ident["noop"]["pct"], ident["noop_opaque"]["pct"],
               ident["wrong_note_snippet"]["pct"])
    out["prediction_vs_outcome"] = {
        "predicted": PREREGISTRATION["prediction"],
        "identification_rollup_vs_stripped": {
            "predicted": "rollup >> stripped",
            "observed": "rollup %.1f%% vs stripped %.1f%%, paired p=%.3g"
                        % (i_roll, i_strip, p_ident),
            "verdict": ("SUPPORTED" if p_ident < 0.05 and i_roll > i_strip
                        else "REFUTED: no significant difference"),
        },
        "finding_recall_rollup_vs_stripped": {
            "predicted": "rollup >> stripped",
            "observed": "rollup %.1f%% vs stripped %.1f%%, paired p=%.3g"
                        % (f_roll, f_strip, p_find),
            "verdict": ("SUPPORTED BUT TAUTOLOGICAL: the rollup snippet contains "
                        "the graded heading on %.0f%% of items, so this cell "
                        "measures copying, not recall"
                        % out["structural"]["rollup_snippet"]
                        ["pct_items_snippet_contains_90pct_of_heading"]
                        if p_find < 0.05 and f_roll > f_strip
                        else "REFUTED: no significant difference"),
        },
        "both_memory_arms_beat_the_controls_on_identification": {
            "predicted": "rollup and stripped >> noop / noop_opaque / wrong_note",
            "observed": "best control %.1f%% vs rollup %.1f%% (p=%.3g) and "
                        "stripped %.1f%% (p=%.3g)"
                        % (ctrl, i_roll,
                           out["paired"]["identification|rollup_snippet_vs_noop"]
                           ["exact_binomial_two_sided_p"], i_strip,
                           out["paired"]["identification|stripped_chunk_vs_noop"]
                           ["exact_binomial_two_sided_p"]),
            "verdict": ("SUPPORTED" if min(i_roll, i_strip) > ctrl + 10
                        else "REFUTED"),
        },
    }

    if verbose:
        print("\n=== GATE (reported first, as required) ===")
        print(json.dumps(out["gate"], indent=2))
        print("\n%-20s %28s %28s" % ("arm", "identification", "finding_recall"))
        print("-" * 78)
        for a in ARMS:
            d, f = ident[a], finding[a]
            print("%-20s %6.1f%% %-18s %6.1f%% %-18s"
                  % (a, d["pct"], str(d["ci95"]), f["pct"], str(f["ci95"])))
        print("\nmean heading-overlap of the INJECTED snippet (contamination):")
        for a in ARMS:
            print("  %-20s %.3f  (>=0.9 on %.1f%% of items)"
                  % (a, out["structural"][a]["mean_heading_overlap_with_injected_snippet"],
                     out["structural"][a]["pct_items_snippet_contains_90pct_of_heading"]))
        print("\npositional guessing check (true at position 1: %.1f%%, "
              "uniform would be %.1f%%):"
              % (out["positional"]["pct_items_true_at_position_1"],
                 out["positional"]["expected_pct_if_position_is_uniform"]))
        for a in ARMS:
            print("  %-20s named position 1 on %.1f%% of items"
                  % (a, out["positional"]
                     ["per_arm_pct_of_named_answers_at_position_1"][a]))
        print("\npaired:")
        for k, v in out["paired"].items():
            print("  %-56s discordant %3d  p=%.4g"
                  % (k, v["discordant"], v["exact_binomial_two_sided_p"]))
        print("\ndrops: %s" % out["n_dropped"])
        print("leaks remaining per arm: %s" % out["leaks"])
        print("\nINTERPRETATION: %s" % out["interpretation"])

    write_json_atomic(REPORT, out)
    print("\nwrote %s" % REPORT)
    return out


# ------------------------------------------------------------------ degenerate

def degenerate_checks() -> None:
    print("=== degenerate inputs ===")
    no_head = "---\ndate: 2026-09-01\n---\n\nsome prose and no headings at all\n"
    print("rollup with no ## heading -> claim_headings=%r" % claim_headings(no_head))
    only_labels = "## Intent\n\nx\n\n## Outcome\n\ny\n\n## 2026-09-18\n\nz\n"
    print("rollup with only label/date headings -> claim_headings=%r"
          % claim_headings(only_labels))
    print("section_body for a heading that is absent -> %r"
          % section_body("## a real claim about caching\n\nbody\n", "missing"))
    print("empty snippet: identification grade -> %r"
          % grade_identification("", "3b5fdb29-0000-0000-0000-000000000000",
                                 ["3b5fdb29-0000-0000-0000-000000000000",
                                  "ab12cd34-0000-0000-0000-000000000000"]))
    print("rollup with no opening request -> first_user_prompt=%r"
          % first_user_prompt('{"message": {"role": "user", "content": "hi"}}'))
    print("transcript that is not json at all -> first_user_prompt=%r"
          % first_user_prompt("not json\nalso not json\n"))
    print("empty snippet: finding grade -> %r" % grade_finding("", "a claim"))
    print("centre_window('', 500) -> %r   centre_window('abc', 0) -> %r"
          % (centre_window("", 500), centre_window("abc", 0)))
    print("truncate('', 500) -> %r" % truncate("", 500))
    print("overlap with empty reference -> %r" % overlap("", "anything"))
    print("scrub of a transcript-shaped leak -> %r"
          % scrub("cwd /home/user/.claude/worktrees/agent-x/ file "
                  "/home/user/.claude/projects/p/3b5fdb29-0000-0000-0000-000000000000"
                  ".jsonl sessions/2026-09-25_claude-code_3b5fdb29_x.md:12",
                  "2026-09-25_claude-code_3b5fdb29_x",
                  "3b5fdb29-0000-0000-0000-000000000000"))
    print("gate_verdict on a leaking arm -> passed=%s"
          % gate_verdict({"wrong_note_snippet": {"pct": 60.0},
                          "noop": {"pct": 12.0},
                          "noop_opaque": {"pct": 11.0}})["passed"])
    print("gate_verdict on a clean arm -> passed=%s"
          % gate_verdict({"wrong_note_snippet": {"pct": 13.0},
                          "noop": {"pct": 12.0},
                          "noop_opaque": {"pct": 11.0}})["passed"])


# ----------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true",
                    help="write the pre-registration stub and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="3 sessions end to end plus the degenerate checks")
    ap.add_argument("--limit", type=int, default=0,
                    help="attempt only the first N eligible sessions")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--inspect", type=int, default=0,
                    help="print N built items for hand inspection, no API calls")
    args = ap.parse_args()

    if args.prereg:
        if REPORT.exists():
            prior = json.loads(REPORT.read_text())
            if prior.get("preregistration"):
                print("pre-registration already recorded in %s; not overwritten"
                      % REPORT)
                return
        write_json_atomic(REPORT, {
            "experiment": "rq6_attribution_regime",
            "status": "PRE-REGISTRATION ONLY -- no results yet",
            "preregistration": PREREGISTRATION})
        print("wrote pre-registration to %s" % REPORT)
        print(json.dumps(PREREGISTRATION, indent=2))
        return

    if args.smoke:
        degenerate_checks()

    manifest = json.loads(MANIFEST.read_text())
    limit = 3 if args.smoke else args.limit
    items, drops = build_items(manifest, limit)
    print("\neligible sessions: %d (attempting %d)"
          % (len(json.loads(MANIFEST.read_text())["paired_session_ids"]), len(items)))
    print("drops by reason: %s" % {k: len(v) for k, v in sorted(drops.items())})

    sizes = {a: [len(i["contexts"][a]) for i in items] for a in ARMS}
    for a in ARMS:
        v = sorted(sizes[a])
        print("  %-20s median %4d  min %4d  max %4d"
              % (a, v[len(v) // 2], v[0], v[-1]))
    bad = {a: [i["session_id"] for i in items if i["leaks"][a]] for a in ARMS}
    bad = {a: v for a, v in bad.items() if v}
    print("  identifier leaks after scrubbing: %s" % (bad or "none"))
    if bad:
        print("FATAL: an arm still names its own session. Refusing to spend calls.")
        sys.exit(1)

    if args.inspect:
        # Read these by hand. An automated leak check cannot see a question that
        # is answerable from its own wording, which is the failure mode the
        # candidate labels already had once.
        for i in items[:args.inspect]:
            print("\n" + "=" * 78)
            print("TRUE = sessions/%s.md" % i["session_id"][:8])
            print("true label     : %s" % candidate_label(i["session_id"],
                                                          i["intent"]))
            print("claim heading  : %s" % i["heading"])
            print("snippet/label overlap  rollup %.2f  stripped %.2f  wrong %.2f"
                  % (i["intent_in_snippet"]["rollup_snippet"],
                     i["intent_in_snippet"]["stripped_chunk"],
                     i["intent_in_snippet"]["wrong_note_snippet"]))
            print("-- candidates --")
            for l in i["candidate_labels"]:
                print("   %s%s" % ("*" if l.startswith("sessions/%s."
                                                       % i["session_id"][:8])
                                   else " ", l))
            print("-- rollup_snippet --\n%s" % i["contexts"]["rollup_snippet"])
            print("-- stripped_chunk --\n%s" % i["contexts"]["stripped_chunk"])
        return

    if args.dry_run or args.summarise_only:
        if args.dry_run:
            print("\n=== first item, every arm ===")
            for a in ARMS:
                print("\n--- %s (%d chars) ---\n%s"
                      % (a, len(items[0]["contexts"][a]), items[0]["contexts"][a]))
            print("\n--- identification prompt, rollup_snippet ---")
            print(IDENT_PROMPT.format(
                candidates="\n".join("- " + l
                                     for l in items[0]["candidate_labels"]),
                context=items[0]["contexts"]["rollup_snippet"]))
            print("\n--dry-run: no API calls made")
            return
        # Spend is summed from the recorded rows rather than declared unknown, so
        # re-publishing a report never drops the cost of the run that produced it.
        rows = load_records(OUT) if OUT.exists() else []
        summarise(items, drops, {
            "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens") or 0
                                 for r in rows),
            "completion_tokens": sum(
                (r.get("usage") or {}).get("completion_tokens") or 0
                for r in rows),
            "n_calls": len(rows),
            "source": "summed from %s" % OUT.name})
        return

    done = set()
    if OUT.exists():
        for r in load_records(OUT):
            done.add((r.get("task"), r.get("arm"), r.get("session_id")))
        print("resuming: %d cells already recorded" % len(done))

    todo = [(t, a, i) for i in items for t in TASKS for a in ARMS
            if (t, a, i["session_id"]) not in done]
    print("to run: %d cells\n" % len(todo))

    t0 = time.time()
    tin = tout = errors = 0
    for n, (task, arm, item) in enumerate(todo, 1):
        reply, usage = ask(task, item, arm)
        tin += usage.get("prompt_tokens") or 0
        tout += usage.get("completion_tokens") or 0
        if not reply:
            # Never recorded, so a rerun retries it. Scoring an empty reply as a
            # wrong answer would manufacture the finding under measurement.
            errors += 1
            print("[%d/%d] %s %s %s -> API_ERROR (not recorded)"
                  % (n, len(todo), item["session_id"][:8], task, arm))
            continue
        rec = {"session_id": item["session_id"], "note_stem": item["note_stem"],
               "task": task, "arm": arm, "context_chars":
                   len(item["contexts"][arm]), "reply": reply,
               "usage": usage, "model": JUDGE_MODEL}
        if task == "identification":
            rec.update(grade_identification(reply, item["session_id"],
                                            item["candidates"]))
            rec["true_position"] = item["true_position"]
            rec["named_positions"] = [
                item["candidates"].index(c) + 1
                for c in item["candidates"] if c[:8] in rec["named"]]
        else:
            rec.update(grade_finding(reply, item["heading"]))
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        mark = rec.get("correct", rec.get("recalled"))
        print("[%d/%d] %s %-14s %-18s -> %s  (%.0fs in=%d out=%d)"
              % (n, len(todo), item["session_id"][:8], task, arm, mark,
                 time.time() - t0, tin, tout))

    print("\nAPI errors (unrecorded, retryable): %d" % errors)
    summarise(items, drops, {"prompt_tokens": tin, "completion_tokens": tout,
                             "api_errors": errors})


if __name__ == "__main__":
    main()
