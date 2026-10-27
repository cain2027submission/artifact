#!/usr/bin/env python3
"""SUPERSEDED by rq3_heldout_equivalence_rebuild.py, which rebuilds this file (and the two
no-digest files) under the current `grade` field, records the field, keeps the old
`grade_published` result under `stale_grade_published` and adds a session-clustered CI.
This script now just calls it. Original description follows.

Execute the equivalence test pre-registered in
tasks/2026-10-04 five strengthening runs pre-registered... .md, null 1 (+-10pts)
and null 3 (+-5pts) on action record vs output-stripped: that session was
abandoned before running it. No new API calls -- this is a bootstrap
reanalysis of the already-graded held-out pairs (rq3_heldout_action_record_arm
.jsonl, rq3_heldout_stripped_arm.jsonl, n=68, the same pairs behind the
published 82.4% vs 85.3%, p=0.72 McNemar result).

TOST via percentile bootstrap: resample paired record indices with
replacement, recompute each arm's recoverable rate, record the signed
difference (action_record - output_stripped). Equivalence at margin M holds
iff the 90% CI (two one-sided tests at alpha=0.05 each <-> a 90% CI) lies
entirely within [-M, +M].

Writes: data/paper-rollups/rq3_heldout_equivalence.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_heldout_equivalence_rebuild import main  # noqa: E402

if __name__ == "__main__":
    main()
