#!/usr/bin/env python3
"""Rebuild the 124 rq3 action-record contexts and tokenize them with real
tiktoken cl100k_base, instead of the chars/2.45 estimate in substrate_value.json.

The action-record context strings were never persisted (rq3_action_record_arm.jsonl
keeps action_record_chars and usage but not the context itself), so this rebuilds
each one from its transcript path via action_record.py:build_action_record(),
then counts real cl100k tokens.

Usage: python3 measure_action_record_cl100k.py
Writes: data/paper-rollups/action_record_cl100k.json
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_record import build_action_record  # noqa: E402

import tiktoken

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/paper-rollups/rq3_action_record_arm.jsonl"
OUTPUT = ROOT / "data/paper-rollups/action_record_cl100k.json"

enc = tiktoken.get_encoding("cl100k_base")

records = []
missing = []
errors = []

with open(INPUT) as f:
    rows = [json.loads(line) for line in f]

for row in rows:
    transcript_path = row["transcript"]
    if not Path(transcript_path).exists():
        missing.append(row["record_id"])
        continue
    try:
        text, stats = build_action_record(transcript_path)
    except Exception as e:
        errors.append((row["record_id"], str(e)))
        continue
    n_tokens = len(enc.encode(text, disallowed_special=()))
    records.append({
        "record_id": row["record_id"],
        "session_id": row["session_id"],
        "chars": len(text),
        "chars_stored_in_jsonl": row.get("action_record_chars"),
        "cl100k_tokens": n_tokens,
    })

char_counts = [r["chars"] for r in records]
token_counts = [r["cl100k_tokens"] for r in records]
ratios = [r["chars"] / r["cl100k_tokens"] for r in records if r["cl100k_tokens"] > 0]

result = {
    "n": len(records),
    "missing_transcript": missing,
    "build_errors": errors,
    "median_chars": statistics.median(char_counts) if char_counts else None,
    "mean_chars": statistics.mean(char_counts) if char_counts else None,
    "median_cl100k_tokens": statistics.median(token_counts) if token_counts else None,
    "mean_cl100k_tokens": statistics.mean(token_counts) if token_counts else None,
    "median_chars_per_token": statistics.median(ratios) if ratios else None,
    "tokenization_method": "tiktoken cl100k_base, real encode() over rebuilt action_record.py contexts",
    "used_estimation": False,
    "comparison": {
        "paper_substrate_value_json_estimate_chars_div_2.45": 9757,
        "note": "substrate_value.json's 9757 used chars/2.45 estimation (used_estimation=true); this file is the first real cl100k count over actual rebuilt action-record text.",
    },
    "records": records,
}

OUTPUT.write_text(json.dumps(result, indent=2))
print(f"n={result['n']} missing={len(missing)} errors={len(errors)}")
print(f"median_chars={result['median_chars']}")
print(f"median_cl100k_tokens={result['median_cl100k_tokens']}")
print(f"mean_cl100k_tokens={result['mean_cl100k_tokens']}")
print(f"median_chars_per_token={result['median_chars_per_token']}")
print(f"wrote {OUTPUT}")
