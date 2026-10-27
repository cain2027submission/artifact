#!/usr/bin/env python3
"""Gate: no author-identifying string reaches the paper or the released scripts.

The submission is double-anonymous and the artifact repository has to be
anonymous too, so an absolute home path, the notes repo's name or the author's
organisation leaking into either one is a desk-reject risk. A one-time cleanup
decays; this runs as a gate so a later compression pass cannot quietly paste a
path back in.

What counts as identifying comes from paths.py, which resolves it from the
machine and the optional local config. This file therefore contains none of the
terms it bans, which is what lets it ship inside the artifact it checks.

A neutral placeholder home such as /Users/alice in a test fixture is NOT a
finding: it identifies nobody. Only the configured corpus home is banned.

Usage:
    check_anonymity.py --worktree     # scan the working tree
    check_anonymity.py --origin       # scan what is on origin/main
    check_anonymity.py --data         # also scan data/paper-rollups (see below)
There is deliberately no default source: the two disagree whenever work is
unpushed, and a gate that picks for you grades the wrong file.

``--data`` is separate rather than part of the default scan because the data
files are not currently releasable and failing the script gate on them would
train everyone to ignore it. What leaks there is record *fields* -- transcript
paths and session-note titles -- so it needs a field-level rewrite against a
session-id mapping, not a substitution pass. Use this mode to size that job.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from paths import (
    CORPUS_HOME,
    CORPUS_SENSITIVE_TERMS,
    PAPER_DIR,
    REPO_ROOT,
)

# Files the artifact would publish. Anything here must be anonymous.
SCRIPT_DIR = "scripts/paper-rollups"
PAPER_FILES = ["paper/main.tex"]
# Scan all file types in scripts/paper-rollups
SCAN_EXTS = {".py", ".sh", ".md", ".json"}

# The local config is gitignored and is the designated home for identifying
# values, so it is exempt. The example config carries placeholders only.
EXEMPT = {"scripts/paper-rollups/paper_config.json"}


def _boundary(term: str) -> str:
    """Escape a term and stop it matching inside a longer word.

    A one-letter home such as `/Users/a` must not flag `/Users/alice`, and a
    repo name must not flag a longer name that contains it.

    A trailing dash is deliberately NOT excluded. Excluding it was a bug, twice:
    the repo name appears dash-joined inside session-note filenames
    (`...-the-<repo>-review-note.md`) and the home path appears dash-joined in a
    transcript directory name, and both are real leaks. Only letters, digits and
    underscore end the match.
    """
    return re.escape(term) + r"(?![A-Za-z0-9_])"


def rules() -> list[tuple[str, re.Pattern]]:
    """Build the ban list from configuration."""
    out: list[tuple[str, re.Pattern]] = []

    home = CORPUS_HOME.rstrip("/")
    if home:
        out.append(("corpus home path", re.compile(_boundary(home))))
        # Claude Code names a transcript directory after the project path with
        # every separator replaced by a dash, so the same leak reappears in a
        # slug such as `-Users-a-Documents-GitHub-Repo`, which the slash form
        # above cannot see. This was a live blind spot: an audit grepping only
        # the slash form undercounted the affected files.
        # The trailing boundary here must allow a dash, because in slug form a
        # dash is exactly what follows the home prefix. Letters and digits are
        # still excluded, so a one-letter home does not match a longer username
        # that merely starts with the same letter.
        out.append(
            (
                "corpus home path, slug form",
                re.compile(re.escape(home.replace("/", "-")) + r"(?![A-Za-z0-9_])"),
            )
        )

    for term in CORPUS_SENSITIVE_TERMS:
        if term:
            out.append(
                (f"sensitive term ({len(term)} chars)", re.compile(_boundary(term), re.I))
            )

    # Machine-specific interpreter and scratch directories. These identify a
    # particular machine rather than a person, but they also mean the script
    # cannot run anywhere else, so they are failures either way.
    out.append(("absolute venv interpreter", re.compile(r"/\.venvs?/[^\s\"']+/bin/python")))
    out.append(("job scratch directory", re.compile(r"\.claude/jobs/[0-9a-f]{6,}")))
    out.append(("unresolved draft marker", re.compile(r"\\(?:gap|pending)\{")))

    return out


def scan(text: str, label: str, checks: list[tuple[str, re.Pattern]]) -> list[str]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, pat in checks:
            if pat.search(line):
                findings.append(f"{label}:{lineno}: {name}")
    return findings


def from_origin(path: str) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"origin/main:{path}"],
        capture_output=True,
        text=True,
    )
    return r.stdout if r.returncode == 0 else None


def origin_files() -> list[str]:
    r = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", "origin/main"],
        capture_output=True,
        text=True,
        check=True,
    )
    names = r.stdout.splitlines()
    keep = [
        n for n in names
        if n.startswith(SCRIPT_DIR + "/") and Path(n).suffix in SCAN_EXTS
    ]
    return sorted(set(keep + [p for p in PAPER_FILES if p in names]))


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] not in ("--worktree", "--origin", "--data"):
        print(__doc__)
        print("  to gate the working tree:  check_anonymity.py --worktree")
        print("  to gate what is on main:    check_anonymity.py --origin")
        print("  to size the data release:   check_anonymity.py --data")
        return 2

    if args[0] == "--data":
        checks = rules()
        data = REPO_ROOT / "data" / "paper-rollups"
        files = n_leaking = total = 0
        worst: list[tuple[int, str]] = []
        for p in sorted(data.iterdir()):
            if not p.is_file():
                continue
            files += 1
            hits = scan(p.read_text(errors="replace"), p.name, checks)
            if hits:
                n_leaking += 1
                total += len(hits)
                worst.append((len(hits), p.name))
        print(f"=== anonymity scan (data), {files} files, {len(checks)} rules ===")
        print(f"{n_leaking} of {files} files carry an identifying string, "
              f"{total} line-level occurrences")
        for n, name in sorted(worst, reverse=True)[:10]:
            print(f"  {n:6}  {name}")
        print("\nThis is a sizing report, not a gate: releasing these needs a "
              "field-level rewrite, not a substitution pass.")
        return 0

    checks = rules()
    source = args[0]
    findings: list[str] = []
    scanned = 0

    if source == "--origin":
        for rel in origin_files():
            if rel in EXEMPT:
                continue
            text = from_origin(rel)
            if text is None:
                continue
            scanned += 1
            findings += scan(text, rel, checks)
    else:
        script_dir = REPO_ROOT / SCRIPT_DIR
        targets = []
        for ext in SCAN_EXTS:
            targets.extend(script_dir.glob(f"*{ext}"))
        targets += [REPO_ROOT / p for p in PAPER_FILES]
        for p in sorted(targets):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel in EXEMPT or not p.exists():
                continue
            scanned += 1
            findings += scan(p.read_text(errors="replace"), rel, checks)

    print(f"=== anonymity gate ({source[2:]}), {scanned} files, {len(checks)} rules ===")
    if not findings:
        print(f"PASS  0 findings in {scanned} files")
        return 0
    for f in findings:
        print("FAIL  " + f)
    print(f"\n{len(findings)} finding(s) in {scanned} files")
    return 1


if __name__ == "__main__":
    sys.exit(main())
