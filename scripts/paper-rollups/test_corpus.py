#!/usr/bin/env python3
"""Test corpus building."""

import sys
from pathlib import Path

from paths import (
    PROJECTS_DIR,
    PUBLIC_SKILLS_DIR,
    SESSIONS_DIR,
)

sys.path.insert(0, str(PUBLIC_SKILLS_DIR))

from pathlib import Path
from rq4_experiment import build_corpus

print("Starting corpus build...", flush=True)

rollup_dir = SESSIONS_DIR
transcript_dir = PROJECTS_DIR

print(f"Rollup dir exists: {rollup_dir.exists()}", flush=True)
print(f"Transcript dir exists: {transcript_dir.exists()}", flush=True)

sessions = build_corpus(rollup_dir, transcript_dir)

print(f"\nFound {len(sessions)} sessions", flush=True)

if sessions:
    sess = sessions[0]
    print(f"\nFirst session: {sess.session_id}", flush=True)
    print(f"Rollup length: {len(sess.rollup_text)} chars", flush=True)
    print(f"Transcript length: {len(sess.transcript_text)} chars", flush=True)
    print(f"Transcript no-tool length: {len(sess.transcript_no_tool)} chars", flush=True)

print("\nDone!", flush=True)
