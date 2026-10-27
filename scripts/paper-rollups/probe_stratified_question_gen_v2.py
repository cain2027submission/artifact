#!/usr/bin/env python3
"""Generate position-stratified questions for RQ3 probe-window bias measurement.

CORRECTED VERSION: Uses the SAME context format as part2_question_retention.py
(the original generator), changing ONLY the sampling window to span the whole
transcript instead of just the first 30k chars.

The original format includes:
- USER text
- ASSISTANT text
- THINKING blocks
- TOOL use with name + input (truncated to 200 chars)
- TOOL_RESULT (truncated to 500 chars)

This format is proven answerable by both arms: 94.4% output-stripped, 56.5% rollup.

Usage:
    probe_stratified_question_gen_v2.py --dry-run
    probe_stratified_question_gen_v2.py --limit 5   # 5 sessions, 3 questions each = 15 total
    probe_stratified_question_gen_v2.py            # full run on 25 sessions
"""

import argparse
import json
import hashlib
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-sonnet-5"
SEED = 42

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
OUTPUT = DATA / "probe_stratified_questions_v2.jsonl"

# Transcript sources
SOURCES = [
    PROJECTS_DIR,
    Path.home() / "Backups" / "claude-transcripts" / "projects",
]

# The 25 graded sessions from rq3_retention_summary.json
GRADED_SESSIONS = [
    "2026-09-15_claude-code_98a92be4_refactor-wifi-mapping-into-git-native",
    "2026-09-15_claude-code_b766c7ce_mirror-two-displays-built-in-3024",
    "2026-09-17_claude-code_88f820e3_explain-how-to-stop-recurring-loops",
    "2026-09-18_claude-code_37d5326d_diagnose-whether-concurrent-agents-were-token",
    "2026-09-18_claude-code_53f752ef_move-an-untitled-note-containing-ux",
    "2026-09-18_claude-code_a8273208_explore-the-idea-of-using-a",
    "2026-09-18_claude-code_ab7bad8a_continue-rewriting-the-vaultnotes-review-note",
    "2026-09-18_claude-code_bc746e37_check-home-assistant-s-zigbee-wifi-signal",
    "2026-09-19_claude-code_9576c5a8_a-loop-running-every-30-minutes",
    "2026-09-20_claude-code_4d4c46f6_update-a-stale-comment-in-the",
    "2026-09-20_claude-code_714318cd_apply-lever-c-100k-main-thread",
    "2026-09-20_claude-code_e7486f80_how-to-add-compaction-microcompaction-support",
    "2026-09-20_claude-code_f5b8c8d0_anything-from-the-last-3-days",
    "2026-09-21_claude-code_57153a94_many-junk-pkm-vault-rollup-notes-reading",
    "2026-09-21_claude-code_b9c67733_mirror-claude-code-s-15-auto",
    "2026-09-21_claude-code_c99a79d3_pull-the-newly-pushed-pkm-vault-task",
    "2026-09-21_claude-code_f4a20cd4_extend-a-public-vault-note-on-cognitive",
    "2026-09-23_claude-code_757d3aaf_play-pr-179-scale-board-shake",
    "2026-09-25_claude-code_1327c110_confident-narrated-finding-was-wrong-every",
    "2026-09-25_claude-code_3b5fdb29_grep-on-the-append-only-ledger",
    "2026-09-25_claude-code_9d4c165a_session-was-just-two-scripted-echo",
    "2026-09-25_claude-code_d7aa4f50_background-sessions-lack-full-disk-access",
    "2026-09-26_claude-code_6d7b783e_sessions-offered-to-write-up-notes",
    "2026-09-26_claude-code_a596be9f_session-lists-three-mcp-server-prefixes",
    "2026-09-26_claude-code_f8828ed8_agent-s-context-problem-is-naming",
]

# Position stratification: 3 equal thirds
N_BANDS = 3
QUESTIONS_PER_SESSION = 3  # 1 per band

# Maximum tokens for question generation (raised from published 2048)
QUESTION_MAX_TOKENS = 4096


def post(url: str, payload: dict, timeout: int = 120) -> dict:
    """Post JSON to gateway endpoint."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def call_llm(prompt: str, timeout: int = 300) -> Tuple[str, dict]:
    """Call LLM via gateway. Returns (response, usage)."""
    url = f"{GATEWAY}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "max_tokens": QUESTION_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        data = post(url, payload, timeout=timeout)
        content = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        finish_reason = data['choices'][0].get('finish_reason')

        if not content or not content.strip():
            print(f"  WARNING: Empty response, finish_reason={finish_reason}")
            return "", usage

        if finish_reason == 'length':
            print(f"  WARNING: Hit max_tokens={QUESTION_MAX_TOKENS}, reply truncated")

        return content.strip(), usage

    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        print(f"  ERROR calling {MODEL}: {error}")
        return "", {}


def transcript_to_text(entries: List[Dict]) -> str:
    """Convert transcript to readable text. IDENTICAL to part2_question_retention.py.

    This is the original context format that yields 94.4%/56.5% on the published run.
    """
    parts = []

    for entry in entries:
        entry_type = entry.get('type')

        if entry_type not in ['user', 'assistant']:
            continue

        msg = entry.get('message', {})
        role = msg.get('role')
        content = msg.get('content', '')

        if role == 'user':
            if isinstance(content, str):
                parts.append(f"USER: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            parts.append(f"USER: {block.get('text', '')}")
                        elif block.get('type') == 'tool_result':
                            # Include tool results, but truncated
                            result = str(block.get('content', ''))[:500]
                            parts.append(f"TOOL_RESULT: {result}")

        elif role == 'assistant':
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    if block.get('type') == 'text':
                        parts.append(f"ASSISTANT: {block.get('text', '')}")
                    elif block.get('type') == 'thinking':
                        parts.append(f"THINKING: {block.get('thinking', '')}")
                    elif block.get('type') == 'tool_use':
                        tool_name = block.get('name', 'unknown')
                        tool_input = json.dumps(block.get('input', {}))[:200]
                        parts.append(f"TOOL: {tool_name} {tool_input}")

    return '\n\n'.join(parts)


def load_transcript(path: Path) -> List[Dict]:
    """Load JSONL transcript."""
    entries = []
    for line in path.read_text(errors='replace').splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def stratify_transcript_by_position(transcript_text: str,
                                    n_bands: int = N_BANDS) -> Dict[int, Tuple[int, int]]:
    """Divide transcript into position bands and return char offsets."""
    text_len = len(transcript_text)
    band_size = text_len // n_bands

    bands = {}
    for i in range(n_bands):
        start = i * band_size
        # Last band takes remainder
        end = text_len if i == n_bands - 1 else (i + 1) * band_size
        bands[i] = (start, end)

    return bands


def sample_from_band(transcript_text: str, start: int, end: int) -> Tuple[str, float]:
    """Sample a 3000-char window from a band. Returns (context, position)."""
    band_len = end - start
    if band_len <= 3000:
        # Band fits entirely
        context = transcript_text[start:end]
        position = (start + end) / 2 / len(transcript_text)
        return context, position

    # Sample a 3000-char window from within the band
    window_start = start + random.randint(0, band_len - 3000)
    window_end = window_start + 3000
    context = transcript_text[window_start:window_end]
    position = (window_start + window_end) / 2 / len(transcript_text)
    return context, position


def generate_question(context: str, position: float, band: int,
                     pair_id: str, dry_run: bool = False) -> Optional[dict]:
    """Generate ONE question from context."""
    if dry_run:
        return {
            "pair_id": pair_id,
            "band": band,
            "position": position,
            "question": "[DRY RUN]",
            "ground_truth": "[DRY RUN]",
        }

    prompt = f"""You are generating a question for an agent transcript evaluation study.

Below is an excerpt from an agent session showing actual work being done:

{context}

Generate ONE question about what the agent DID or what actually HAPPENED in THIS session.

DO NOT ask about:
- System prompts, instructions, or reminders
- Tool definitions or API schemas
- General rules or configuration
- Documentation content
- Metadata like token counts, timing, or technical internals

Make the question answerable ONLY from what actually happened in this session.
Keep it concrete and specific, including numbers, file names, or error messages when present.

Return your response as JSON in this exact format:
{{
  "question": "What...",
  "ground_truth": "The answer is..."
}}

Return ONLY the JSON, no other text."""

    response, usage = call_llm(prompt, timeout=300)
    if not response:
        return None

    try:
        json_text = response
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    json_text = part[4:].strip()
                elif part.startswith("{"):
                    json_text = part

        data = json.loads(json_text)

        return {
            "pair_id": pair_id,
            "band": band,
            "position": position,
            "question": data["question"],
            "ground_truth": data["ground_truth"],
            "usage": usage,
        }

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ERROR parsing response: {e}")
        return None


def index_transcripts() -> Dict[str, Path]:
    """Index transcripts by session slug (e.g., '98a92be4')."""
    by_slug: Dict[str, Path] = {}

    for src in SOURCES:
        if not src.is_dir():
            continue
        for p in src.rglob("*.jsonl"):
            # Match against graded session IDs
            for sid in GRADED_SESSIONS:
                # Extract 8-hex slug from session ID
                parts = sid.split('_')
                if len(parts) >= 3:
                    slug = parts[2]
                    if slug in p.stem:
                        by_slug[sid] = p
                        break

    return by_slug


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                       help="Check coverage and sizes without API calls")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit to N sessions for smoke test")
    args = parser.parse_args()

    random.seed(SEED)

    print("Indexing transcripts for 25 graded sessions...")
    transcript_index = index_transcripts()
    print(f"  Found {len(transcript_index)}/{len(GRADED_SESSIONS)} transcripts")

    missing = set(GRADED_SESSIONS) - set(transcript_index.keys())
    if missing:
        print(f"\n  WARNING: Missing transcripts for {len(missing)} sessions:")
        for sid in sorted(missing):
            print(f"    {sid}")
        print()

    sessions_to_process = sorted(transcript_index.keys())
    if args.limit:
        sessions_to_process = sessions_to_process[:args.limit]
        print(f"\nLimited to {len(sessions_to_process)} sessions")

    if args.dry_run:
        print("\nDRY RUN: Checking transcript coverage and band sizes\n")
        for session_id in sessions_to_process:
            transcript_path = transcript_index[session_id]
            entries = load_transcript(transcript_path)
            text = transcript_to_text(entries)
            bands = stratify_transcript_by_position(text, N_BANDS)

            print(f"{session_id[:50]}")
            print(f"  Transcript: {len(text):,} chars, {len(entries)} entries")
            for band_idx in sorted(bands.keys()):
                start, end = bands[band_idx]
                print(f"  Band {band_idx}: chars {start:,}-{end:,} ({end-start:,} chars)")

        print(f"\nDRY RUN complete. Would generate {len(sessions_to_process) * QUESTIONS_PER_SESSION} questions.")
        return 0

    # Generate questions
    total_cost_usd = 0.0
    questions_generated = 0

    print(f"\nGenerating {QUESTIONS_PER_SESSION} questions per session ({N_BANDS} bands)")
    print()

    for i, session_id in enumerate(sessions_to_process, 1):
        print(f"[{i}/{len(sessions_to_process)}] {session_id[:60]}")

        transcript_path = transcript_index[session_id]
        entries = load_transcript(transcript_path)
        text = transcript_to_text(entries)
        bands = stratify_transcript_by_position(text, N_BANDS)

        print(f"  Transcript: {len(text):,} chars")

        # Generate one question per band
        for band_idx in sorted(bands.keys()):
            start, end = bands[band_idx]
            context, position = sample_from_band(text, start, end)

            pair_id = f"probe_{session_id}_{band_idx:02d}"

            question = generate_question(context, position, band_idx, pair_id,
                                        dry_run=False)

            if question:
                # Write to output immediately
                with OUTPUT.open("a") as f:
                    record = {
                        "session_id": session_id,
                        "pair_id": pair_id,
                        "band": band_idx,
                        "position": position,
                        "question": question["question"],
                        "ground_truth": question["ground_truth"],
                    }
                    f.write(json.dumps(record) + "\n")

                questions_generated += 1

                # Estimate cost (rough)
                usage = question.get("usage", {})
                input_tok = usage.get("prompt_tokens", 0) + usage.get("input_tokens", 0)
                output_tok = usage.get("completion_tokens", 0) + usage.get("output_tokens", 0)
                cost = (input_tok * 3.0 + output_tok * 15.0) / 1e6
                total_cost_usd += cost

                print(f"    Band {band_idx} @ {position:.3f}: OK (${cost:.4f})")
            else:
                print(f"    Band {band_idx}: FAILED")

        print(f"  Total cost so far: ${total_cost_usd:.2f}")
        print()

        # Budget check
        if total_cost_usd >= 30.0:
            print(f"STOP: Reached $30 budget cap")
            break

    print(f"\n{'='*70}")
    print(f"Generated {questions_generated} questions from {i} sessions")
    print(f"Total cost: ${total_cost_usd:.2f}")
    print(f"Output: {OUTPUT}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
