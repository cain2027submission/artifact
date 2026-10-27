#!/usr/bin/env python3
"""Assertions for rq6_attribution_regime.py.

Every check here exists because the corresponding mistake would have produced a
publishable-looking number that was wrong, and most of them have already happened
once somewhere in this paper:

  * the identification grader must be deterministic and must not credit a reply
    that hedges across candidates
  * path comparison must survive the repo name's two casings, a
    `.claude/worktrees/<name>/` prefix and a `:LINE` suffix
  * the session set must come from the frozen manifest, never from a glob of
    `sessions/` (which has since more than doubled)
  * no arm may leak the identifier that IS the answer to the identification task
  * degenerate inputs must return a value rather than raise
  * the `wrong_note_snippet` gate must be enforced in code, not by eye

Run:  python3 scripts/paper-rollups/test_rq6_attribution_regime.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import paths as P  # noqa: E402
import rq6_attribution_regime as R  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print("%s %s%s" % ("PASS" if cond else "FAIL", name,
                       ("  -- " + str(detail)) if detail and not cond else ""))


# ------------------------------------------------- 1. identification grader

TRUE = "3b5fdb29-1111-2222-3333-444444444444"
OTHER = "ab12cd34-1111-2222-3333-444444444444"
CANDS = [TRUE, OTHER, "ffffffff-1111-2222-3333-444444444444"]
STEM = "2026-09-25_claude-code_3b5fdb29_a-claim-about-caching"

check("grader: exact single correct candidate -> correct",
      R.grade_identification("Reasoning.\nsessions/%s.md" % TRUE[:8],
                             TRUE, CANDS)["correct"])

check("grader: single WRONG candidate -> not correct",
      not R.grade_identification("sessions/%s.md" % OTHER[:8], TRUE, CANDS)["correct"])

multi = R.grade_identification(
    "could be sessions/%s.md or sessions/%s.md" % (TRUE[:8], OTHER[:8]), TRUE, CANDS)
check("grader: hedging across two candidates -> not correct (n_named=2)",
      (not multi["correct"]) and multi["n_named"] == 2, multi)

check("grader: naming every candidate is not an identification",
      not R.grade_identification(" ".join(c[:8] for c in CANDS),
                                 TRUE, CANDS)["correct"])

check("grader: empty reply -> not correct, n_named=0",
      R.grade_identification("", TRUE, CANDS) == {"named": [], "n_named": 0,
                                                  "correct": False})

check("grader: refusal text -> not correct",
      not R.grade_identification("I cannot tell from this.", TRUE, CANDS)["correct"])

check("grader: is deterministic across repeated calls",
      R.grade_identification("sessions/%s.md" % TRUE[:8], TRUE, CANDS)
      == R.grade_identification("sessions/%s.md" % TRUE[:8], TRUE, CANDS))

check("grader: the same id named twice is still ONE identification",
      R.grade_identification("sessions/%s.md, i.e. %s" % (TRUE[:8], TRUE[:8]),
                             TRUE, CANDS)["correct"])

check("grader: a full uuid in the reply still matches its 8-hex candidate",
      R.grade_identification("session %s" % TRUE, TRUE, CANDS)["correct"])

# ------------------------------------------------------ 2. path normalisation

# Configure CORPUS_REPO_ALIASES to test the fold actually works
_orig_aliases = P.CORPUS_REPO_ALIASES[:]
P.CORPUS_REPO_ALIASES[:] = ["MyNotes", "mynotes"]
check("norm: two casings of the repo name fold to the same token",
      R.norm_path_token("/Users/testuser/MyNotes/sessions/%s.md" % STEM)
      == R.norm_path_token("/Users/testuser/mynotes/sessions/%s.md" % STEM))
P.CORPUS_REPO_ALIASES[:] = _orig_aliases

check("norm: a :LINE suffix does not break the match",
      R.grade_identification("sessions/%s.md:12" % TRUE[:8], TRUE, CANDS)["correct"])

check("norm: a .claude/worktrees/<name>/ prefix does not break the match",
      R.grade_identification(
          ".claude/worktrees/agent-a5d4bbd4/sessions/%s.md" % TRUE[:8],
          TRUE, CANDS)["correct"])

check("norm: an absolute repo path with case variation still matches",
      R.grade_identification(
          "/Users/testuser/vault/sessions/%s.md" % TRUE[:8],
          TRUE, CANDS)["correct"])

check("norm: punctuation and backticks around the path still match",
      R.grade_identification("`sessions/%s.md`." % TRUE[:8], TRUE, CANDS)["correct"])

# ------------------------------------- 2b. the candidate label must not be the key

check("label: the candidate label carries no claim slug and no date",
      "a-claim-about-caching" not in R.candidate_label(TRUE, "do a thing please")
      and "2026" not in R.candidate_label(TRUE, "do a thing please"))
check("label: the candidate label does carry the 8-hex id the grader keys on",
      TRUE[:8] in R.candidate_label(TRUE, "do a thing please"))
check("label: the opening request is capped so a long prompt cannot dominate",
      len(R.candidate_label(TRUE, "x" * 900)) < 250)

check("intent: the opening request is read from the transcript, not the note",
      R.first_user_prompt(json.dumps(
          {"message": {"role": "user",
                       "content": "please review all 434 repos and report which "
                                  "ones are stale"}})) ==
      "please review all 434 repos and report which ones are stale")
check("intent: a tool_result user record is not mistaken for the opening request",
      R.first_user_prompt("\n".join([
          json.dumps({"message": {"role": "user", "content": [
              {"type": "tool_result", "content": "a" * 200}]}}),
          json.dumps({"message": {"role": "user",
                                  "content": "the real opening request here, "
                                             "long enough to qualify"}})]))
      == "the real opening request here, long enough to qualify")
check("intent: a system-reminder is stripped out of the opening request",
      "system-reminder" not in R.first_user_prompt(json.dumps(
          {"message": {"role": "user",
                       "content": "<system-reminder>ignore me</system-reminder> "
                                  "the actual request is this one here"}})))
check("intent: a slash-command stub is skipped",
      R.first_user_prompt(json.dumps(
          {"message": {"role": "user",
                       "content": "<command-name>/loop</command-name> run it "
                                  "again and again until it works"}})) == "")
check("intent: an isMeta record is skipped",
      R.first_user_prompt(json.dumps(
          {"isMeta": True,
           "message": {"role": "user", "content": "a long meta preamble line"}}))
      == "")
check("degenerate: a too-short opening request yields '' (session is dropped)",
      R.first_user_prompt(json.dumps({"message": {"role": "user",
                                                  "content": "hi"}})) == "")
check("degenerate: a transcript that is not JSON yields '' rather than raising",
      R.first_user_prompt("not json at all\n") == "")
check("degenerate: an empty transcript yields ''", R.first_user_prompt("") == "")

check("norm: WORKTREE regex strips the prefix",
      "worktrees" not in R.normalise_vault(".claude/worktrees/agent-x/sessions/a.md"))

check("norm: LINESUF regex strips :12 from a .md path",
      R.normalise_vault("sessions/a.md:12").endswith("a.md"))

# ------------------------------------------------------ 3. scrubbing / leaks

SID = "3b5fdb29-1111-2222-3333-444444444444"
DIRTY = ("cwd /home/user/.claude/worktrees/agent-x/ transcript "
         "/home/user/.claude/projects/-home-user/%s.jsonl note sessions/%s.md:8 "
         "session_id: %s prefix 3b5fdb29" % (SID, TRUE, SID))
CLEAN = R.scrub(DIRTY, TRUE, SID)

check("scrub: the full session uuid is gone", SID not in CLEAN, CLEAN)
check("scrub: the 8-hex session prefix is gone", "3b5fdb29" not in CLEAN, CLEAN)
check("scrub: the .jsonl transcript path is gone", ".jsonl" not in CLEAN, CLEAN)
check("scrub: the note stem is gone", TRUE not in CLEAN, CLEAN)
check("scrub: leak_scan reports nothing on a scrubbed context",
      R.leak_scan(CLEAN, TRUE, SID) == [], R.leak_scan(CLEAN, TRUE, SID))
check("scrub: leak_scan DOES fire on the unscrubbed original",
      set(R.leak_scan(DIRTY, TRUE, SID)) >= {"note_stem", "session_uuid",
                                             "jsonl_path"})
check("scrub: a disguised positive is still caught (stem with spaces for dashes)",
      "<REDACTED>" in R.scrub("note 2026-09-25 claude-code 3b5fdb29 a claim about "
                              "caching here", TRUE, SID))
check("scrub: the shipped controls carry no identifier",
      R.leak_scan("- sessions/<REDACTED>.md", TRUE, SID) == []
      and R.leak_scan("(no context was retrieved for this prompt)", TRUE, SID) == [])

# ------------------------------------------------------- 4. heading selection

check("headings: a claim-shaped heading is kept",
      R.claim_headings("## the daemon goes cold in ten seconds\n\nbody\n")
      == ["the daemon goes cold in ten seconds"])
check("headings: structural labels are rejected (boring_heading)",
      R.claim_headings("## Intent\n\nx\n\n## Outcome\n\ny\n") == [])
check("headings: a bare date heading is rejected",
      R.claim_headings("## 2026-09-18\n\nx\n") == [])
check("headings: a two-word heading is rejected as carrying no claim",
      R.claim_headings("## cold start\n\nx\n") == [])
check("headings: a note with NO ## heading at all returns [] rather than raising",
      R.claim_headings("---\ndate: x\n---\n\njust prose\n") == [])
check("headings: None-safe", R.claim_headings(None) == [])
check("section_body: absent heading returns '' rather than raising",
      R.section_body("## a real claim about caching\n\nbody\n", "nope") == "")
check("section_body: frontmatter and wikilinks are cleaned out of the body",
      "[[" not in R.section_body("## a real claim about caching\n\nsee [[a note]]\n",
                                 "a real claim about caching"))

# --------------------------------------------------------- 5. budget / window

LONG = ("alpha bravo charlie delta echo foxtrot golf hotel india juliett " * 40)
check("budget: truncate never exceeds MAX_BODY_CHARS + the ellipsis",
      len(R.truncate(LONG)) <= R.BUDGET + 1, len(R.truncate(LONG)))
check("budget: centre_window never exceeds MAX_BODY_CHARS",
      len(R.centre_window(LONG)) <= R.BUDGET, len(R.centre_window(LONG)))
check("budget: centre_window is actually the CENTRE, not the head",
      not LONG.startswith(R.centre_window(LONG)[:40]))
check("budget: MAX_BODY_CHARS is the hook's 500", R.BUDGET == 500, R.BUDGET)
check("degenerate: centre_window('') -> ''", R.centre_window("") == "")
check("degenerate: centre_window(text, 0) -> ''", R.centre_window("abc", 0) == "")
check("degenerate: truncate('') -> ''", R.truncate("") == "")
check("degenerate: truncate(text, 0) -> ''", R.truncate("abc", 0) == "")

# ------------------------------------------------------- 6. finding grader

REF = "the daemon goes cold in ten seconds and straddles the hook deadline"
check("finding: a close paraphrase clears the 0.5 threshold",
      R.grade_finding("The daemon goes cold after ten seconds, straddling the "
                      "hook's deadline.", REF)["recalled"])
check("finding: an unrelated sentence does not",
      not R.grade_finding("The paper needs a new table.", REF)["recalled"])
check("finding: NOT ANSWERABLE scores exactly 0 and is flagged refused",
      R.grade_finding("NOT ANSWERABLE", REF) ==
      {"overlap": 0.0, "refused": True, "recalled": False})
check("finding: a refusal that also happens to contain the words scores 0",
      R.grade_finding("NOT ANSWERABLE " + REF, REF)["overlap"] == 0.0)
check("degenerate: empty reply scores 0", R.grade_finding("", REF)["overlap"] == 0.0)
check("degenerate: empty reference returns 0.0 rather than dividing by zero",
      R.overlap("", "anything") == 0.0)
check("finding: grading is deterministic",
      R.grade_finding("the daemon goes cold in ten seconds", REF)
      == R.grade_finding("the daemon goes cold in ten seconds", REF))
check("finding: stopwords cannot carry a score on their own",
      R.grade_finding("the and of to in on at by for with", REF)["overlap"] == 0.0)

# ------------------------------------------------------------ 7. statistics

check("stats: wilson(0, 89) matches the prior experiment's [0.0, 4.1]",
      R.wilson(0, 89) == (0.0, 4.1), R.wilson(0, 89))
check("stats: wilson(0, 0) is (0.0, 0.0) rather than a ZeroDivisionError",
      R.wilson(0, 0) == (0.0, 0.0))
check("stats: exact_binomial(14, 18) reproduces rq3's p=0.5966",
      abs(R.exact_binomial(14, 18) - 0.5966) < 0.001, R.exact_binomial(14, 18))
check("stats: exact_binomial with no discordant pairs is p=1.0",
      R.exact_binomial(0, 0) == 1.0)
check("stats: exact_binomial is symmetric",
      R.exact_binomial(3, 20) == R.exact_binomial(20, 3))
check("stats: rate() on an empty denominator does not raise",
      R.rate(0, 0)["pct"] == 0.0)

# ------------------------------------------------- 8. the gate, in code

hi = R.gate_verdict({"wrong_note_snippet": {"pct": 60.0}, "noop": {"pct": 12.0},
                     "noop_opaque": {"pct": 11.0}})
lo = R.gate_verdict({"wrong_note_snippet": {"pct": 13.0}, "noop": {"pct": 12.0},
                     "noop_opaque": {"pct": 11.0}})
edge = R.gate_verdict({"wrong_note_snippet": {"pct": 22.0}, "noop": {"pct": 12.0},
                       "noop_opaque": {"pct": 11.0}})
over = R.gate_verdict({"wrong_note_snippet": {"pct": 22.1}, "noop": {"pct": 12.0},
                       "noop_opaque": {"pct": 11.0}})
check("gate: a guessable question set FAILS the gate", not hi["passed"], hi)
check("gate: a clean wrong-note arm PASSES the gate", lo["passed"], lo)
check("gate: exactly at the %.0f pp margin still passes" % R.GATE_MARGIN_PP,
      edge["passed"], edge)
check("gate: one tenth of a point over the margin fails", not over["passed"], over)
check("gate: the margin is measured against the BETTER no-op",
      hi["best_noop_pct"] == 12.0)

# The second gate condition, added because the first has a hole this suite found:
# a margin against a no-op that is itself at ceiling proves nothing.
ceiling = R.gate_verdict({"wrong_note_snippet": {"pct": 100.0},
                          "noop": {"pct": 100.0}, "noop_opaque": {"pct": 100.0}})
check("gate: a no-op at ceiling FAILS even though its margin is zero",
      (not ceiling["passed"]) and ceiling["margin_over_noop_pp"] == 0.0, ceiling)
check("gate: a ceiling no-op is diagnosed as a leaking prompt, not a guessable one",
      "leaks the answer" in ceiling["meaning"], ceiling["meaning"])
check("gate: a no-op well above chance fails on the chance condition",
      not R.gate_verdict({"wrong_note_snippet": {"pct": 30.0},
                          "noop": {"pct": 30.0},
                          "noop_opaque": {"pct": 28.0}})["passed"])
check("gate: both conditions are reported separately",
      set(ceiling) >= {"wrong_note_within_margin", "noops_near_chance"})
check("gate: a failed gate says so in words", "GATE FAILED" in hi["meaning"])
check("gate: chance is recorded as 12.5% for 8 candidates",
      lo["chance_pct"] == 12.5 and R.N_CANDIDATES == 8)

# ------------------- 9. the gate suppresses the interpretation in summarise()

def fake_corpus(tmp: Path, n_good: int = 3, n_generic: int = 0,
                n_dup: int = 0) -> dict:
    """A manifest-shaped fixture: good sessions, plus one of every drop reason.

    `n_generic` adds sessions sharing one bland opening request and `n_dup` adds
    exact duplicates of the first session's request. Both only misbehave at
    realistic pool size -- genericness IS high document frequency, so a single
    bland prompt is rare by construction and cannot be detected at all.
    """
    sess = tmp / "sessions"
    sess.mkdir()
    proj = tmp / "projects"
    proj.mkdir()
    paired, rollups, transcripts, tbytes = [], {}, [], {}
    body = ("this is enough prose to fill a five hundred character window and "
            "then some, repeated so the stripped transcript clears the budget. ") * 12

    def add(sid, note_text, with_note=True, with_transcript=True, tiny=False,
            prompt=None):
        paired.append(sid)
        if with_note:
            p = sess / ("2026-09-25_claude-code_%s_fixture-%s.md" % (sid[:8], sid[:4]))
            p.write_text(note_text)
            rollups[sid] = str(p)
        if with_transcript:
            t = proj / ("%s.jsonl" % sid)
            text = "short" if tiny else body
            lines = [json.dumps({"message": {
                "role": "user",
                "content": prompt or ("please investigate the %s subsystem and "
                                      "report back" % sid[:8])}})]
            lines.append(json.dumps({"message": {"role": "assistant",
                                                 "content": [{"type": "text",
                                                              "text": text}]}}))
            t.write_text("\n".join(lines) + "\n")
            transcripts.append(str(t))
            tbytes[str(t)] = t.stat().st_size

    # Distinct, discriminative opening requests: a shared scaffold plus three
    # words unique to each session, so each clears MIN_INTENT_RARE_TOKENS while
    # staying below the duplicate and distractor Jaccard ceilings.
    def good_prompt(i: int) -> str:
        return ("reconcile the zqa%dx ledger against the zqb%dx snapshot and "
                "chart zqc%dx drift" % (i, i, i))

    def note_for(i):
        return ("---\nd: x\n---\n\n## a genuine claim about fixture number %d\n\n"
                "%s\n" % (i, body))

    for i in range(n_good):
        add("%08d-0000-0000-0000-000000000000" % (10000000 + i),
            note_for(i), prompt=good_prompt(i))
    for i in range(n_generic):
        add("%08d-0000-0000-0000-000000000000" % (20000000 + i),
            note_for(1000 + i),
            prompt="get latest, then start on the new task you pulled")
    for i in range(n_dup):
        add("%08d-0000-0000-0000-000000000000" % (30000000 + i),
            note_for(2000 + i), prompt=good_prompt(0))
    add("aaaaaaaa-0000-0000-0000-000000000000", "## Intent\n\nx\n\n## Outcome\n\ny\n")
    add("bbbbbbbb-0000-0000-0000-000000000000", "## a genuine claim about a pruned "
        "transcript\n\nbody\n", with_transcript=False)
    add("cccccccc-0000-0000-0000-000000000000", "## a genuine claim about a tiny "
        "transcript\n\nbody\n", tiny=True)
    add("dddddddd-0000-0000-0000-000000000000", "", with_note=False)
    add("eeeeeeee-0000-0000-0000-000000000000", "## a genuine claim about a "
        "session with no usable opening request\n\n%s\n" % body, prompt="hi")
    return {"paired_session_ids": paired, "rollup_note_paths": rollups,
            "transcripts": transcripts, "transcript_bytes": tbytes,
            "rollup_files": [Path(p).name for p in rollups.values()]}


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    man = fake_corpus(tmp)
    items, drops = R.build_items(man)

    check("manifest: only sessions with a claim heading AND a live transcript survive",
          len(items) == 3, (len(items), {k: len(v) for k, v in drops.items()}))
    check("manifest: a label-only note is dropped as no_claim_shaped_heading",
          len(drops.get("no_claim_shaped_heading", [])) == 1, drops)
    check("degenerate: a PRUNED transcript is dropped, not absorbed as a zero",
          len(drops.get("transcript_pruned", [])) == 1, drops)
    check("degenerate: a sub-budget stripped transcript is dropped and recorded",
          len(drops.get("stripped_transcript_under_budget", [])) == 1, drops)
    check("manifest: a missing rollup note is dropped and recorded",
          len(drops.get("no_rollup_note_on_disk", [])) == 1, drops)
    check("degenerate: a session with no usable opening request is dropped",
          len(drops.get("no_usable_opening_request", [])) == 1, drops)

    check("jaccard: identical sets are 1.0 and disjoint sets are 0.0",
          R.jaccard({"a"}, {"a"}) == 1.0 and R.jaccard({"a"}, {"b"}) == 0.0)
    check("jaccard: an empty set is 0.0 rather than a ZeroDivisionError",
          R.jaccard(set(), {"a"}) == 0.0)

    check("manifest: every drop is recorded by reason, none silently vanish",
          len(items) + sum(len(v) for v in drops.values())
          == len(man["paired_session_ids"]))
    check("manifest: no session appears both kept and dropped",
          not ({i["session_id"] for i in items}
               & {s for v in drops.values() for s in v}))

    ids = [i["session_id"] for i in items]
    check("manifest: every kept id came from paired_session_ids",
          set(ids) <= set(man["paired_session_ids"]))

    check("arms: all five arms are built for every item",
          all(set(i["contexts"]) == set(R.ARMS) for i in items))
    check("arms: no arm leaks its own session's identifiers",
          all(not i["leaks"][a] for i in items for a in R.ARMS),
          [(i["session_id"], a, i["leaks"][a]) for i in items for a in R.ARMS
           if i["leaks"][a]])
    check("arms: every context is within the 500-char budget",
          all(len(i["contexts"][a]) <= R.BUDGET + 1 for i in items for a in R.ARMS))
    check("derangement: no session ever donates the wrong snippet to itself",
          all(i["donor_session_id"] != i["session_id"] for i in items))
    check("derangement: the wrong snippet differs from the true snippet",
          all(i["contexts"]["wrong_note_snippet"] != i["contexts"]["rollup_snippet"]
              for i in items))
    check("candidates: the true session is always among the candidates",
          all(i["session_id"] in i["candidates"] for i in items))
    # The smoke run refuted the earlier version of this assertion, which required
    # the list to be SORTED by id "so position leaks nothing". Items are built in
    # sorted id order, so an early item's true session sorts ahead of most of its
    # own distractors and lands at position 1; a zero-information no-op scored
    # 33.3% by answering first-in-list. Order must now be a per-item shuffle.
    check("candidates: the list is NOT sorted by id (position would track the id)",
          any(i["candidates"] != sorted(i["candidates"], key=lambda s: s[:8])
              for i in items),
          [i["candidates"] for i in items])
    check("candidates: true_position is recorded and points at the true session",
          all(i["candidates"][i["true_position"] - 1] == i["session_id"]
              for i in items))
    check("candidates: the donor of the wrong snippet is NOT a candidate",
          all(i["donor_session_id"] not in i["candidates"] for i in items))
    check("candidates: one label per candidate",
          all(len(i["candidate_labels"]) == len(i["candidates"]) for i in items))
    check("candidates: no candidate label restates the note's claim heading "
          "(the tautology the first dry run caught)",
          all(R.overlap(i["heading"], " ".join(i["candidate_labels"])) < 0.9
              for i in items),
          [(i["session_id"], R.overlap(i["heading"], " ".join(i["candidate_labels"])))
           for i in items])
    check("candidates: no candidate label contains a claim slug or a date",
          all("fixture-" not in l and "2026" not in l
              for i in items for l in i["candidate_labels"]))
    check("candidates: no duplicate candidates",
          all(len(set(i["candidates"])) == len(i["candidates"]) for i in items))
    check("candidates: candidate selection is seeded, so a rerun is identical",
          R.build_items(man)[0][0]["candidates"] == items[0]["candidates"])

    check("contamination: the rollup snippet DOES contain its own heading",
          all(i["heading_in_snippet"]["rollup_snippet"] >= 0.9 for i in items))
    check("contamination: the no-op controls contain none of it",
          all(i["heading_in_snippet"]["noop"] == 0.0 for i in items))

    # The prompts must not name the answer.
    p = R.IDENT_PROMPT.format(candidates="- sessions/x.md", context="ctx")
    check("leak: the identification prompt does not reveal which candidate is true",
          "correct" not in p.lower() and "true" not in p.lower())
    check("leak: the finding prompt does not contain the reference heading",
          REF not in R.FINDING_PROMPT)

    # summarise() must refuse to interpret a failed gate. Redirect module paths so
    # the published files are untouched.
    real_out, real_report = R.OUT, R.REPORT
    try:
        R.OUT = tmp / "arms.jsonl"
        R.REPORT = tmp / "report.json"
        rows = []
        for i in items:
            for arm in R.ARMS:
                # Everything correct, including wrong_note_snippet: a gate failure.
                rows.append({"session_id": i["session_id"], "task": "identification",
                             "arm": arm, "correct": True, "n_named": 1})
                rows.append({"session_id": i["session_id"], "task": "finding_recall",
                             "arm": arm, "overlap": 1.0, "recalled": True,
                             "refused": False})
        R.OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
        rep = R.summarise(items, drops, {}, verbose=False)
        check("gate: summarise() REFUSES to interpret when the gate fails",
              rep["interpretation"].startswith("NONE") and not rep["gate"]["passed"],
              rep["interpretation"])
        check("gate: the gate verdict is present in the emitted report",
              "gate" in rep and "passed" in rep["gate"])
        check("report: deviations and limitations are emitted, not left implicit",
              len(rep["deviations"]) >= 4 and len(rep["limitations"]) >= 3)
        check("report: drops are emitted by reason with counts",
              set(rep["n_dropped"]) == set(drops))
        check("report: each pre-registered clause is scored SUPPORTED or REFUTED",
              all(v["verdict"].startswith(("SUPPORTED", "REFUTED"))
                  for k, v in rep["prediction_vs_outcome"].items()
                  if isinstance(v, dict)),
              rep.get("prediction_vs_outcome"))
        check("report: the positional-guessing check is emitted per arm",
              set(rep["positional"]["per_arm_pct_of_named_answers_at_position_1"])
              == set(R.ARMS))
        check("report: the paired rollup-vs-stripped test is emitted for both tasks",
              "identification|rollup_snippet_vs_stripped_chunk" in rep["paired"]
              and "finding_recall|rollup_snippet_vs_stripped_chunk" in rep["paired"])
        check("report: every arm gets a Wilson interval on both tasks",
              all(len(rep["arms"][t][a]["ci95"]) == 2
                  for t in R.TASKS for a in R.ARMS))

        # A partial jsonl must not be scored as failure: incomplete sessions are
        # excluded from the denominator rather than counted as wrong.
        R.OUT.write_text("".join(json.dumps(r) + "\n" for r in rows[:2]))
        rep2 = R.summarise(items, drops, {}, verbose=False)
        check("resume: a session missing some arms is excluded, not scored zero",
              rep2["n_complete"]["identification"] == 0
              and rep2["n_eligible_sessions"] == 3, rep2["n_complete"])

        # An existing published file must never be truncated by a failed write.
        R.REPORT.write_text('{"preregistration": {"kept": true}}')
        try:
            R.write_json_atomic(R.REPORT, {"bad": {1, 2}})
        except TypeError:
            pass
        check("atomicity: a failed write leaves the existing report intact",
              json.loads(R.REPORT.read_text())["preregistration"]["kept"] is True)
        check("atomicity: no .tmp file is left behind",
              not list(R.REPORT.parent.glob("*.tmp")))
        check("prereg: load_prereg() prefers the block already on disk",
              R.load_prereg() == {"kept": True})
    finally:
        R.OUT, R.REPORT = real_out, real_report

# A generic opening request and a duplicated one both make an item unanswerable
# for EVERY arm, so both must be dropped rather than scored. Tested at realistic
# pool size because genericness IS document frequency: a single bland prompt is
# rare by construction and no df-based rule can see it.
with tempfile.TemporaryDirectory() as td2:
    big = fake_corpus(Path(td2), n_good=22, n_generic=4, n_dup=1)
    kept_b, drops_b = R.build_items(big)
    gen_ids = {"%08d-0000-0000-0000-000000000000" % (20000000 + i)
               for i in range(4)}
    dup_id = "30000000-0000-0000-0000-000000000000"
    check("eligibility: sessions sharing one bland opening request are all dropped",
          gen_ids <= set(drops_b.get("opening_request_not_discriminative", [])),
          drops_b.get("opening_request_not_discriminative"))
    check("eligibility: an exact duplicate of another opening request is dropped",
          dup_id in drops_b.get("duplicate_opening_request", []), drops_b)
    check("eligibility: the 22 discriminative sessions all survive",
          len(kept_b) == 22, len(kept_b))
    check("eligibility: no distractor near-duplicates the true label",
          all(R.jaccard(R.content_tokens(i["intent"]),
                        R.content_tokens(j["intent"])) < R.DISTRACTOR_MAX_JACCARD
              for i in kept_b for j in kept_b
              if j["session_id"] in i["candidates"]
              and j["session_id"] != i["session_id"]))
    check("candidates: a realistic pool yields the full 8-way choice",
          all(len(i["candidates"]) == R.N_CANDIDATES for i in kept_b),
          sorted({len(i["candidates"]) for i in kept_b}))
    # The defect the smoke run found: with candidates sorted by id, items built in
    # id order put the true session first almost always. Answering "first" then
    # beats chance with no information at all. Under 1/8 = 12.5% uniform, 22 items
    # allow a generous ceiling; the sorted version scored 20/22 here.
    at_1 = sum(1 for i in kept_b if i["true_position"] == 1)
    check("candidates: the true session is not concentrated at position 1",
          at_1 <= 6, "%d/%d items have the answer first" % (at_1, len(kept_b)))
    check("candidates: the true position varies across items",
          len({i["true_position"] for i in kept_b}) >= 4,
          sorted(i["true_position"] for i in kept_b))

# -------------------------------------- 10. source-level reproducibility rules

SRC = (HERE.parent / "rq6_attribution_regime.py").read_text()
check("source: sessions/ is never globbed",
      'glob("*.md")' not in SRC and "glob('*.md')" not in SRC
      and "iterdir" not in SRC)
check("source: the published report is never opened in mode 'w'",
      'REPORT.open("w")' not in SRC and "REPORT.open('w')" not in SRC
      and 'REPORT.write_text' not in SRC)
check("source: the arms jsonl is opened append-only",
      'OUT.open("a")' in SRC and 'OUT.open("w")' not in SRC)
check("source: results are flushed and fsynced so a killed batch keeps its work",
      "fsync" in SRC)
check("source: an API error is not recorded, so a rerun retries it",
      "API_ERROR (not recorded)" in SRC)
check("source: no temperature is sent (rejected on Claude 5 models)",
      "temperature" not in SRC)
check("source: the guard env vars are never set",
      "PKM_ALLOW_MASS_DELETE" not in SRC and "PKM_ALLOW_LEAK_TERM" not in SRC)
check("source: the pre-registered prediction and threshold are in the file",
      "PREREGISTRATION" in SRC and "OVERLAP_THRESHOLD = 0.5" in SRC)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED: " + "; ".join(FAIL))
sys.exit(1 if FAIL else 0)
