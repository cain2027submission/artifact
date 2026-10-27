#!/usr/bin/env python3
"""Regression test for delegation.py, built out of disguised positives.

The detector keys on a directory layout, so the way it fails is by keying on a
label instead: matching anything with "subagents" in the path, or anything named
`agent-*.jsonl`. Both of those bugs were live during this change and both were
caught here rather than by reading the code -- the second one (a `subagents/`
directory nested one level too deep) made the manifest form invent a delegating
session called "nested".

Every negative below is a plausible-but-wrong shape, not an obviously unrelated
path. Case I is the opposite trap: a real layout that looks wrong. One of the 158
delegating sessions in the frozen corpus writes its subagent transcripts under a
different project directory than its own transcript, because the session changed
working directory mid-run, and a first version of the fix scored it as solo.

Run:
    python3 test_delegation.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delegation import delegating_ids_from_manifest, is_delegating  # noqa: E402


def touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"x":1}\n')
    return p


def build(root: Path) -> tuple[dict[str, tuple[Path, bool]], set[str]]:
    proj = root / "projects"
    a = proj / "-Users-a-one"
    cases: dict[str, tuple[Path, bool]] = {}

    p = touch(a / "aaaaaaaa.jsonl")
    touch(a / "aaaaaaaa" / "subagent" / "agent-x.jsonl")        # singular dir
    cases["singular subagent/ dir"] = (p, False)

    p = touch(a / "bbbbbbbb.jsonl")
    touch(a / "agent-bbbb.jsonl")                               # beside parent
    cases["agent-*.jsonl beside the parent"] = (p, False)

    p = touch(a / "cccccccc.jsonl")
    (a / "cccccccc" / "subagents").mkdir(parents=True)           # empty dir
    cases["empty subagents/ dir"] = (p, False)

    p = touch(a / "dddddddd.jsonl")
    touch(a / "dddddddd" / "subagents" / "sidechain-1.jsonl")     # wrong name
    cases["subagents/ holding a non-agent file"] = (p, False)

    p = touch(a / "eeeeeeee.jsonl")
    touch(a / "eeeeeeee" / "nested" / "subagents" / "agent-y.jsonl")
    cases["subagents/ one level too deep"] = (p, False)

    p = touch(a / "ffffffff.jsonl")
    (a / "ffffffff" / "subagents" / "agent-f.jsonl").mkdir(parents=True)
    cases["agent-*.jsonl that is a directory"] = (p, False)

    touch(a / "hhhhhhhh" / "subagents" / "agent-z.jsonl")         # no parent
    cases["orphan subagents/ with no parent transcript"] = (
        a / "hhhhhhhh.jsonl", False)

    p = touch(a / "99999999.jsonl")
    touch(a / "99999999" / "subagents" / "agent-general-purpose-1.jsonl")
    cases["the real shape"] = (p, True)

    p = touch(a / "11111111.jsonl")
    touch(proj / "-Users-a-two" / "11111111" / "subagents" / "agent-w.jsonl")
    cases["cross-project, session changed cwd"] = (p, True)

    return cases, {"99999999", "11111111"}


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="delegation-test-"))
    failures = 0
    try:
        cases, expect_ids = build(root)
        for name, (path, expected) in sorted(cases.items()):
            got = is_delegating(path)
            ok = got == expected
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'}  is_delegating  {name:44s} "
                  f"expected={expected} got={got}")

        manifest = sorted(str(p) for p in root.rglob("*.jsonl") if p.is_file())
        ids = delegating_ids_from_manifest(manifest)
        ok = ids == expect_ids
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  manifest form  "
              f"{'':44s} expected={sorted(expect_ids)} got={sorted(ids)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{'all passed' if not failures else str(failures) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
