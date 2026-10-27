#!/usr/bin/env python3
"""Gate for the corrected RQ3 path recall.

Run with python3:

    python3 scripts/paper-rollups/test_path_recall.py

Every case here corresponds to one of the four defects in the withdrawn 44.5%
figure, and the last one is a control: it asserts the OLD extractor *fails* the
same cases, so a pass means the suite is testing something rather than agreeing
with whatever the code currently does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import path_extract as px
import part1_structural_loss as p1
from freeze_corpus import ROLLUPS, load_manifest, parse_session_id, read_pinned
from paths import CORPUS_HOME
from rq3_path_recall import OUT as CORRECTED_JSON
from rq3_path_recall import load_pairs

VAULT = "/Users/testuser/vault"
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  ok    {msg}")


# ---------------------------------------------------------------------------
def test_normalisation_unifies_relative_and_absolute() -> None:
    """A relative and an absolute form of one file must share one key.

    This is the defect that made session 1758702b score 233%: the rollup said
    `AGENTS.md` and the transcript held an absolute path, and nothing mapped
    those together, so the intersection of two sets naming the same four files
    was empty while the ratio of their sizes was 7/3.
    """
    print("normalisation maps relative and absolute to one key")
    rel = px.normalise("scripts/paper-rollups/freeze_corpus.py", root=VAULT)
    ab = px.normalise(f"{VAULT}/scripts/paper-rollups/freeze_corpus.py")
    check(rel is not None and rel == ab,
          f"relative {rel!r} == absolute {ab!r}")

    # ~ expansion, a trailing line reference, and the two repo name casings
    # (82.8% of transcript mentions use the second) all collapse too.
    check(px.normalise("~/AGENTS.md") == px.normalise(f"{CORPUS_HOME}/AGENTS.md"),
          "~/ expands to the home directory")
    check(px.normalise(f"{VAULT}/AGENTS.md:66")
          == px.normalise(f"{VAULT}/AGENTS.md"),
          "a trailing :LINE reference is dropped")
    check(px.normalise("/Users/testuser/vault/AGENTS.md")
          == px.normalise("/Users/testuser/Vault/AGENTS.md"),
          "two casings of the repo name are one key")
    # A worktree is the same repo at another path.
    check(px.normalise("/repo/.claude/worktrees/wt-name/src/a.ts")
          == px.normalise("/repo/src/a.ts"),
          "a .claude/worktrees/<name>/ segment collapses")
    # Directories and prose are not paths.
    for junk in ["/Users/testuser", "learnings/", "and/or", "2026/09/28",
                 "https://example.com/a/b.py", "see section 3.2",
                 "local models.", "*.py", ".md"]:
        check(px.normalise(junk, root=VAULT) is None, f"{junk!r} is not a path")


def test_path_only_in_tool_argument_is_found() -> None:
    """A path that appears solely inside a tool-call argument must be found.

    Bash is 54% of tool calls in this corpus and its paths live inside the
    `command` string. The old transcript extractor read only structured
    `file_path`/`path`/`paths` keys, so the single commonest way a session touches
    a file was invisible to it -- session 715f5e5e is 38 Bash calls, 2 Edits and
    1 Read, and the old extractor found one path in the whole transcript.
    """
    print("a path only in a tool-call argument is found on the transcript side")
    entries = [
        {
            "type": "assistant",
            "cwd": VAULT,
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Bash", "input": {
                    "command": "grep -rn 'recall' scripts/paper-rollups/rq3_path_recall.py",
                    "description": "search",
                }},
            ]},
        },
        {
            "type": "assistant",
            "cwd": VAULT,
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Read", "input": {
                    "file_path": f"{VAULT}/paper/main.tex"}},
            ]},
        },
    ]
    found = px.transcript_paths(entries, surface=px.SURFACE_ACTED)
    want_bash = px.normalise("scripts/paper-rollups/rq3_path_recall.py", root=VAULT)
    want_read = px.normalise(f"{VAULT}/paper/main.tex")
    check(want_bash in found,
          f"path inside a Bash command found ({want_bash!r} in {sorted(found)})")
    check(want_read in found, "path in a structured file_path argument found")

    # And the prose surface must be off by default, or the denominator silently
    # includes files the session only talked about.
    prose = [{"type": "assistant", "cwd": VAULT, "message": {
        "role": "assistant", "content": [
            {"type": "text", "text": "I should look at scripts/only_mentioned.py next"}]}}]
    acted = px.transcript_paths(prose, surface=px.SURFACE_ACTED)
    authored = px.transcript_paths(prose, surface=px.SURFACE_AUTHORED)
    check(not acted, "prose-only path absent from the `acted` denominator")
    check(px.normalise("scripts/only_mentioned.py", root=VAULT) in authored,
          "prose-only path present in the `authored` denominator")

    # A path only in a tool *result* belongs to the widest arm alone.
    res = [{"type": "user", "cwd": VAULT, "message": {"role": "user", "content": [
        {"type": "tool_result", "content": "scripts/from_a_result.py\n"}]}}]
    check(not px.transcript_paths(res, surface=px.SURFACE_AUTHORED),
          "tool-result path absent from the `authored` denominator")
    check(px.normalise("scripts/from_a_result.py", root=VAULT)
          in px.transcript_paths(res, surface=px.SURFACE_ALL),
          "tool-result path present in the `all` denominator")


def test_fixture_scores_exactly_fifty_percent() -> None:
    """Hand-built: the rollup names 2 of the transcript's 4 paths -> 50.0%.

    Deliberately exercises the shapes that broke the original: one path cited in
    the rollup relative where the transcript has it absolute, one cited through a
    worktree-qualified path, and two the rollup never mentions. It also includes
    the trace-backlink boilerplate, which the old rollup extractor counted as a
    cited artifact -- if it were counted here the numerator would be wrong.
    """
    print("a 2-of-4 fixture scores exactly 50%")
    entries = [{
        "type": "assistant",
        "cwd": VAULT,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read",
             "input": {"file_path": f"{VAULT}/scripts/paper-rollups/freeze_corpus.py"}},
            {"type": "tool_use", "name": "Edit",
             "input": {"file_path": f"{VAULT}/paper/main.tex"}},
            {"type": "tool_use", "name": "Read",
             "input": {"file_path": f"{VAULT}/AGENTS.md"}},
            {"type": "tool_use", "name": "Bash",
             "input": {"command": f"wc -l {VAULT}/data/paper-rollups/README.md"}},
        ]},
    }]
    note = (
        "---\n"
        "session_id: fixture-0000\n"
        f"project_dir: {VAULT}\n"
        "---\n\n"
        "## a fixture\n\n"
        "Edited `scripts/paper-rollups/freeze_corpus.py` and rewrote the\n"
        "section in `.claude/worktrees/some-wt/paper/main.tex`.\n\n"
        "## the full trace is at ~/.cache/agent-session-traces/traces/fixture-0000\n"
    )
    r_paths, root = px.rollup_paths(note)
    t_paths = px.transcript_paths(entries, surface=px.SURFACE_ACTED)
    num, den, rec = px.path_recall(r_paths, t_paths)
    check(root == VAULT, f"project_dir read from frontmatter ({root!r})")
    check(den == 4, f"denominator is 4 transcript paths, got {den}: {sorted(t_paths)}")
    check(num == 2, f"numerator is 2 named paths, got {num}: "
                    f"{sorted(px.match_transcript_paths(r_paths, t_paths))}")
    check(abs(rec - 0.5) < 1e-9, f"recall is exactly 50%, got {100 * rec:.4f}%")

    trace = px.normalise("~/.cache/agent-session-traces/traces/fixture-0000")
    check(trace is None or trace not in r_paths,
          "the trace-backlink path is not counted as a cited artifact")


def test_every_session_is_at_most_one() -> None:
    """No session in the frozen paired set may exceed 100%.

    Two assertions, because the artifact alone would only prove the script agreed
    with itself: every committed per-session row, and a live recomputation over
    the 35 sessions the withdrawn metric scored above 100% on -- the hardest
    cases, not a random sample.
    """
    print("per-session recall <= 1.0 over the frozen paired set")
    if not CORRECTED_JSON.exists():
        FAILURES.append(f"{CORRECTED_JSON} missing; run rq3_path_recall.py first")
        print(f"  FAIL  {CORRECTED_JSON} missing; run rq3_path_recall.py first")
        return
    data = json.loads(CORRECTED_JSON.read_text())
    m = load_manifest()
    check(data["corpus_snapshot_utc"] == m["snapshot_utc"],
          "artifact was computed over the current frozen snapshot")

    rows = data["per_session"]
    check(len(rows) > 0, f"artifact holds {len(rows)} sessions")
    bad = []
    for r in rows:
        for arm in list(px.SURFACES) + ["acted_with_recaps"]:
            v = r[arm]["recall"]
            if v > 1.0:
                bad.append(f"session {r['session_id']} arm {arm} scored "
                           f"{100 * v:.1f}% ({r[arm]['numerator']}/"
                           f"{r[arm]['denominator']}) -- above 100% is impossible "
                           f"for a recall")
            if r[arm]["numerator"] > r[arm]["denominator"]:
                bad.append(f"session {r['session_id']} arm {arm} numerator "
                           f"{r[arm]['numerator']} exceeds denominator "
                           f"{r[arm]['denominator']}")
    check(not bad, "no committed session exceeds 100%"
                   + ("" if not bad else " -- " + "; ".join(bad[:5])))

    # Live recomputation on the sessions that proved the defect.
    worst = [w["session_id"] for w in
             data["withdrawn_figure"]["recomputed_here"]["worst_sessions"]]
    hard = set(worst)
    checked = 0
    live_bad = []
    for sid, notes, tpath in load_pairs():
        if sid not in hard:
            continue
        text = "\n".join(p.read_text(errors="replace") for p in notes)
        entries = []
        for line in read_pinned(str(tpath)).splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        r_paths, _ = px.rollup_paths(text)
        for arm in px.SURFACES:
            num, den, rec = px.path_recall(
                r_paths, px.transcript_paths(entries, surface=arm))
            if rec > 1.0 or num > den:
                live_bad.append(f"session {sid} arm {arm} scored {100 * rec:.1f}% "
                                f"({num}/{den}) on live recomputation")
        checked += 1
    check(checked > 0, f"live recomputation covered {checked} of the worst sessions")
    check(not live_bad, "no live-recomputed session exceeds 100%"
                        + ("" if not live_bad else " -- " + "; ".join(live_bad[:5])))


def test_old_extractor_fails_these_cases() -> None:
    """Control: the OLD code must fail at least one case above.

    Without this the suite could be green against a no-op fix. It asserts three
    specific failures of `part1_structural_loss.py`, each mapping to a defect:
    the metric exceeds 1.0 on a real session, the transcript side misses a Bash
    path, and the rollup side counts a directory and the trace boilerplate as
    cited artifacts.
    """
    print("control: the old extractor fails these cases")
    old_failures = 0

    # 1. The old statistic exceeds 1.0 where the true recall cannot.
    entries = [{
        "type": "assistant",
        "cwd": VAULT,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit",
             "input": {"file_path": f"{VAULT}/paper/main.tex"}}]},
    }]
    note = (f"---\nproject_dir: {VAULT}\n---\n\n"
            "Touched `/home/user`, `scene.ts`, `src/mahjong` and\n"
            "`~/.cache/traces/x`.\n")
    old_r, _ = p1.extract_files_from_rollup(note)
    old_t = p1.extract_files_from_transcript(entries)
    old_val = len(old_r) / len(old_t) if old_t else 0.0
    if old_val > 1.0:
        old_failures += 1
    check(old_val > 1.0,
          f"old metric scores {100 * old_val:.0f}% on a fixture whose ceiling is "
          f"100% (rollup set {len(old_r)}, transcript set {len(old_t)})")
    _, _, new_val = px.path_recall(*(lambda rp: (rp[0], px.transcript_paths(
        entries, surface=px.SURFACE_ACTED)))(px.rollup_paths(note)))
    check(new_val <= 1.0, f"new metric stays at {100 * new_val:.0f}% on the same fixture")

    # 2. The old transcript side cannot see a path inside a Bash command.
    bash_only = [{
        "type": "assistant", "cwd": VAULT,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Bash",
             "input": {"command": f"wc -l {VAULT}/paper/main.tex"}}]},
    }]
    old_seen = p1.extract_files_from_transcript(bash_only)
    if not old_seen:
        old_failures += 1
    check(not old_seen,
          f"old transcript extractor finds nothing in a Bash-only turn ({old_seen})")
    check(px.normalise(f"{VAULT}/paper/main.tex")
          in px.transcript_paths(bash_only, surface=px.SURFACE_ACTED),
          "new transcript extractor finds it")

    # 3. The old rollup side counts a directory and the trace boilerplate.
    junk = {p for p in old_r if p in ("/home/user", "src/mahjong")
            or "traces" in p}
    if junk:
        old_failures += 1
    check(bool(junk),
          f"old rollup extractor counts directories and trace boilerplate as "
          f"cited artifacts ({sorted(junk)})")

    # 4. The old normalisation cannot pair a relative citation with an absolute.
    rel_note = f"---\nproject_dir: {VAULT}\n---\n\nEdited `AGENTS.md`.\n"
    rel_entries = [{
        "type": "assistant", "cwd": VAULT,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read",
             "input": {"file_path": f"{VAULT}/AGENTS.md"}}]},
    }]
    o_r, _ = p1.extract_files_from_rollup(rel_note)
    o_t = p1.extract_files_from_transcript(rel_entries)
    if not (o_r & o_t):
        old_failures += 1
    check(not (o_r & o_t),
          f"old code's two sides share no key for one file ({sorted(o_r)} vs "
          f"{sorted(o_t)})")
    r2, _ = px.rollup_paths(rel_note)
    n2, d2, _ = px.path_recall(r2, px.transcript_paths(
        rel_entries, surface=px.SURFACE_ACTED))
    check((n2, d2) == (1, 1), f"new code pairs them (got {n2}/{d2})")

    check(old_failures >= 1,
          f"the old extractor fails {old_failures} of these cases, so the suite "
          f"is testing something")


def main() -> None:
    for fn in (test_normalisation_unifies_relative_and_absolute,
               test_path_only_in_tool_argument_is_found,
               test_fixture_scores_exactly_fifty_percent,
               test_every_session_is_at_most_one,
               test_old_extractor_fails_these_cases):
        fn()
        print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} assertion(s)")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
