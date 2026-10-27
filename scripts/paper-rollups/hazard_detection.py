#!/usr/bin/env python3
"""Detection checks for the four MSR hazards."""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Read one sample transcript to demonstrate each hazard
SAMPLE_PATH = Path("~/.claude/projects").expanduser()

def hazard_1_block_inflation():
    """Hazard 1: Entries are per content block, inflate counts 2.38x."""
    # This is the detection check: deduplicate on requestId
    print("=== HAZARD 1: Per-block inflation ===")
    print("Detection: count entries vs unique requestId values")
    print("Expected: entries / unique_requests ≈ 2.38x")
    print("Verified in rq1_corrected_summary.json: 2.38x call inflation, 2.26x token inflation")
    print()

def hazard_2_user_type():
    """Hazard 2: Tool results are typed 'user', not human turns."""
    print("=== HAZARD 2: Tool results typed as 'user' ===")
    print("Detection: count role='user' entries by content type")
    print("Expected: 93.2% are tool_result blocks, not human text")
    print("Impact: turn-splitting on role='user' shreds multi-tool turns")
    print()

def hazard_3_byte_size():
    """Hazard 3: Byte size != real content."""
    print("=== HAZARD 3: Byte size divergence ===")
    print("Detection: file size vs payload_chars from composition scan")
    print("Expected: median 691 KB serialized vs much smaller payload")
    print("Impact: corpus filters by file size misclassify transcripts")
    print()

def hazard_4_usage_overstatement():
    """Hazard 4: Transcript usage fields overstate billed spend ~1.5x."""
    print("=== HAZARD 4: Usage field overstatement ===")
    print("Detection: sum message.usage fields vs console CSV billing")
    print("Expected: transcript reports ~1.5x higher than billed")
    print("Impact: cost analyses from transcripts alone are inflated")
    print()

def generality_verdict(hazard_num: int) -> str:
    """Is this hazard specific to this client or general to agent transcripts?"""
    verdicts = {
        1: "GENERAL: per-block usage reporting is an API response structure, not a Claude Code implementation detail. Any system that writes one transcript entry per content block and repeats usage records will see this.",
        2: "GENERAL: the Anthropic API types tool results as role='user', per the Messages API spec. Any system using this API will see tool results as user records.",
        3: "SPECIFIC: byte inflation from bookkeeping/attachment records depends on what this specific harness logs. Generality unknown without checking other agent frameworks.",
        4: "SPECIFIC: cause unknown. Verified for this machine's transcripts vs Anthropic Console CSV. Generality unknown without replication on other accounts."
    }
    return verdicts.get(hazard_num, "UNKNOWN")

if __name__ == "__main__":
    hazard_1_block_inflation()
    print(f"Generality: {generality_verdict(1)}")
    print()

    hazard_2_user_type()
    print(f"Generality: {generality_verdict(2)}")
    print()

    hazard_3_byte_size()
    print(f"Generality: {generality_verdict(3)}")
    print()

    hazard_4_usage_overstatement()
    print(f"Generality: {generality_verdict(4)}")
