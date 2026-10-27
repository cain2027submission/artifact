#!/usr/bin/env python3
"""Real cl100k measurement of the action-record-minus-result-digests ablation
variant, across all 124 rq3 questions. Reuses the already-verified
build_action_record_no_result_digests() from spinoff_actionrecord_ablation.py
instead of its stdout-only /2.45 estimate.

Writes: data/paper-rollups/action_record_no_digests_cl100k.json
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_record import build_action_record  # noqa: E402
from spinoff_actionrecord_ablation import build_action_record_no_result_digests  # noqa: E402

import tiktoken

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/paper-rollups/rq3_action_record_arm.jsonl"
OUTPUT = ROOT / "data/paper-rollups/action_record_no_digests_cl100k.json"

enc = tiktoken.get_encoding("cl100k_base")

with open(INPUT) as f:
    rows = [json.loads(line) for line in f]

records = []
for row in rows:
    transcript_path = row["transcript"]
    full_text, _ = build_action_record(transcript_path)
    stripped_text, _ = build_action_record_no_result_digests(transcript_path)
    records.append({
        "record_id": row["record_id"],
        "full_chars": len(full_text),
        "full_cl100k_tokens": len(enc.encode(full_text, disallowed_special=())),
        "no_digest_chars": len(stripped_text),
        "no_digest_cl100k_tokens": len(enc.encode(stripped_text, disallowed_special=())),
    })

full_chars = [r["full_chars"] for r in records]
full_tok = [r["full_cl100k_tokens"] for r in records]
nd_chars = [r["no_digest_chars"] for r in records]
nd_tok = [r["no_digest_cl100k_tokens"] for r in records]

result = {
    "n": len(records),
    "full_action_record": {
        "median_chars": statistics.median(full_chars),
        "median_cl100k_tokens": statistics.median(full_tok),
    },
    "no_result_digests_variant": {
        "median_chars": statistics.median(nd_chars),
        "median_cl100k_tokens": statistics.median(nd_tok),
    },
    "savings_vs_full": {
        "chars": statistics.median(full_chars) - statistics.median(nd_chars),
        "cl100k_tokens": statistics.median(full_tok) - statistics.median(nd_tok),
    },
    "output_stripped_baseline_median_cl100k_tokens": 8880,
    "note": "accuracy/recall of the no-digest variant is UNMEASURED here -- this is size only, no grading was run",
    "records": records,
}

OUTPUT.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
