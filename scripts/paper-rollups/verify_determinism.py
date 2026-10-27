#!/usr/bin/env python3
"""Verify that each RQ script produces byte-identical output across consecutive runs.

This is the gate behind the paper's reproducibility claim, so it is worth being
explicit about what it does and about the way its first version did not do it.

The first version ran each script twice and only then hashed the outputs, so it
compared nothing: it printed the hash left on disk by run 2 and reported success
unconditionally. Every script "passed" whether or not it was deterministic. The
fix is the whole point of the file -- hash after run 1, hash again after run 2,
and compare the two dicts.

Determinism here means byte-identical output, which is stricter than
reproducible numbers. It catches the failure this study actually hit: a script
that re-globs a live corpus picks up sessions written since the last run, so its
denominator moves under it. Passing this gate is evidence that the script reads
the frozen manifest rather than the filesystem.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PYTHON
REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO / "scripts" / "paper-rollups"
DATA_DIR = REPO / "data" / "paper-rollups"

SCRIPTS = [
    ("rq1_signal_composition_corrected.py", [
        "rq1_corrected_summary.json",
        "rq1_corrected_per_session.csv",
    ]),
    ("rq2_compression_corrected.py", [
        "rq2_compression_corrected.json",
        "rq2_compression_corrected.csv",
    ]),
    ("rq5_generation_cost.py", [
        "rq5_generation_cost.json",
    ]),
    ("rq0_query_workload.py", [
        "rq0_query_workload.json",
    ]),
    ("rq3_retention_summary.py", [
        "rq3_retention_summary.json",
    ]),
    ("rq3_artifact_citation.py", [
        "rq3_artifact_citation.json",
    ]),
]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(output_files) -> dict:
    """Hash every declared output, recording misses rather than raising."""
    out = {}
    for name in output_files:
        p = DATA_DIR / name
        out[name] = hash_file(p) if p.exists() else "<missing>"
    return out


def run_once(script_path: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [PYTHON, str(script_path)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return r.returncode == 0, r.stderr


def main() -> int:
    failures = []

    for script_name, output_files in SCRIPTS:
        script_path = SCRIPTS_DIR / script_name
        print(f"\n{script_name}")

        if not script_path.exists():
            print("  SKIP  script not found")
            failures.append(f"{script_name}: script not found")
            continue

        ok, err = run_once(script_path)
        if not ok:
            print(f"  FAIL  run 1 exited non-zero\n{err.strip()[:800]}")
            failures.append(f"{script_name}: run 1 failed")
            continue
        first = snapshot(output_files)

        ok, err = run_once(script_path)
        if not ok:
            print(f"  FAIL  run 2 exited non-zero\n{err.strip()[:800]}")
            failures.append(f"{script_name}: run 2 failed")
            continue
        second = snapshot(output_files)

        for name in output_files:
            a, b = first[name], second[name]
            if a == "<missing>" or b == "<missing>":
                print(f"  FAIL  {name:<42} output missing")
                failures.append(f"{script_name}/{name}: missing")
            elif a == b:
                print(f"  ok    {name:<42} {a[:16]}")
            else:
                print(f"  FAIL  {name:<42} {a[:16]} != {b[:16]}")
                failures.append(f"{script_name}/{name}: differs between runs")

    print()
    if failures:
        print(f"NOT DETERMINISTIC ({len(failures)} failure(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"deterministic: {len(SCRIPTS)} scripts, byte-identical across two runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
