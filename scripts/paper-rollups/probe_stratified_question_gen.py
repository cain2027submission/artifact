#!/usr/bin/env python3
"""Generate position-stratified questions for RQ3 probe-window bias measurement.

Tests whether the RQ3 retention result survives when questions are sampled across
the WHOLE transcript instead of its prefix. The published result used only the
first ~30k chars, where verbatim-derived arms retain that prefix nearly intact.

This generator:
- Uses a FRESH held-out pool: sessions after 2026-09-30T00:00:00
- Applies STRICTER frame: >= 100KB serialized AND >= 20 tool invocations
- Stratifies by POSITION: divides each transcript into 5 equal character-offset
  bands and draws equal questions from each band
- Records each question's source band for per-band analysis

Usage:
    probe_stratified_question_gen.py --dry-run
    probe_stratified_question_gen.py --limit 5   # smoke test: 5 sessions, 10 questions
    probe_stratified_question_gen.py --limit 50  # full run within budget
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR
from holdout_question_gen import (
    call_llm, extract_substantive_chunks, has_boilerplate_markers,
    is_infrastructure_session, SEED
)

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-sonnet-5"

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
OUTPUT = DATA / "probe_stratified_questions.jsonl"

# Transcript sources
SOURCES = [
    PROJECTS_DIR,
    Path.home() / "Backups" / "claude-transcripts" / "projects",
]

# Fresh held-out cutoff: after 2026-09-30 00:00:00
CUTOFF = datetime.datetime(2026, 9, 30, 0, 0, 0, tzinfo=datetime.timezone.utc)

# Stricter frame than paper's
MIN_SIZE_BYTES = 100_000
MIN_TOOL_INVOCATIONS = 20

# Position stratification: 5 equal bands
N_BANDS = 5
QUESTIONS_PER_SESSION = 2  # 1 per band, cycling through bands


def count_tool_invocations(transcript_text: str) -> int:
    """Count tool_use blocks in transcript."""
    count = 0
    for line in transcript_text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            msg = rec.get("message", {})
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        count += 1
        except json.JSONDecodeError:
            continue
    return count


def get_first_timestamp(transcript_text: str) -> Optional[datetime.datetime]:
    """Extract timestamp from first record in transcript."""
    for line in transcript_text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            ts_str = rec.get("timestamp")
            if ts_str:
                # Parse ISO format with timezone
                return datetime.datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def stratify_chunks_by_position(chunks: List[Tuple[str, int, int]],
                                transcript_length: int,
                                n_bands: int = N_BANDS) -> Dict[int, List[Tuple[str, int, int]]]:
    """Assign chunks to position bands (0-indexed)."""
    bands = {i: [] for i in range(n_bands)}
    band_size = transcript_length / n_bands

    for chunk_text, start_pos, end_pos in chunks:
        midpoint = (start_pos + end_pos) / 2
        band_idx = min(int(midpoint / band_size), n_bands - 1)
        bands[band_idx].append((chunk_text, start_pos, end_pos))

    return bands


def sample_from_band(band_chunks: List[Tuple[str, int, int]],
                     transcript_length: int) -> Optional[Tuple[str, float, int]]:
    """Sample one chunk from a band and return (context, position, band_idx)."""
    if not band_chunks:
        return None

    chunk_text, start_pos, end_pos = random.choice(band_chunks)
    midpoint = (start_pos + end_pos) / 2
    position = midpoint / transcript_length if transcript_length > 0 else 0.0

    context = chunk_text[:3000]
    return context, position, None  # band_idx will be set by caller


def generate_question(event_context: str, position: float, band: int,
                     pair_id: str, dry_run: bool = False) -> Optional[dict]:
    """Generate ONE question about an event (not a pair)."""
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

{event_context}

Generate ONE question about what the agent DID or what actually HAPPENED in THIS session.

DO NOT ask about:
- System prompts, instructions, or reminders
- Tool definitions or API schemas
- General rules or configuration
- Documentation content

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
    """Index all available transcripts by session ID."""
    by_id: Dict[str, List[Path]] = {}
    for src in SOURCES:
        if not src.is_dir():
            continue
        for p in src.rglob("*.jsonl"):
            stem = p.stem
            # Full UUID format (primary)
            if len(stem) == 36 and stem.count('-') == 4:
                session_id = stem
            # Agent subagent format
            elif stem.startswith("agent-"):
                session_id = stem
            # 8-char prefix fallback
            elif len(stem) >= 8:
                session_id = stem[:8]
            else:
                continue

            by_id.setdefault(session_id, []).append(p)

    # Resolve to single best transcript per ID
    resolved = {}
    for session_id, paths in by_id.items():
        best = sorted(paths,
                     key=lambda p: ("subagents" in p.parts, -p.stat().st_size))[0]
        resolved[session_id] = best

    return resolved


def write_question_record(f, session_id: str, pair_id: str,
                         question: str, ground_truth: str,
                         band: int, position: float):
    """Write one question record to output file."""
    record = {
        "session_id": session_id,
        "pair_id": pair_id,
        "band": band,
        "position": position,
        "question": question,
        "ground_truth": ground_truth,
    }
    f.write(json.dumps(record) + "\n")
    f.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate selection without LLM calls")
    parser.add_argument("--limit", type=int, default=None,
                       help="Generate from N sessions, then stop")
    args = parser.parse_args()

    random.seed(SEED)
    print(f"Seed: {SEED}")
    print(f"Cutoff: {CUTOFF.isoformat()}")
    print(f"Frame: >= {MIN_SIZE_BYTES/1000:.0f}KB AND >= {MIN_TOOL_INVOCATIONS} tool invocations")
    print()

    # Index transcripts
    print("Indexing transcripts...")
    all_transcripts = index_transcripts()
    print(f"  Found {len(all_transcripts)} transcripts")

    # Filter by date, size, tool count
    print("\nApplying frame...")
    candidates = {}
    filtered_stats = Counter()

    for session_id, path in all_transcripts.items():
        size = path.stat().st_size
        if size < MIN_SIZE_BYTES:
            filtered_stats["size_too_small"] += 1
            continue

        transcript = path.read_text(errors="replace")

        first_ts = get_first_timestamp(transcript)
        if not first_ts:
            filtered_stats["no_timestamp"] += 1
            continue
        if first_ts <= CUTOFF:
            filtered_stats["before_cutoff"] += 1
            continue

        tool_count = count_tool_invocations(transcript)
        if tool_count < MIN_TOOL_INVOCATIONS:
            filtered_stats["tool_count_too_low"] += 1
            continue

        is_infra, reason = is_infrastructure_session(transcript)
        if is_infra:
            filtered_stats[f"infrastructure_{reason}"] += 1
            continue

        candidates[session_id] = (path, size, tool_count, first_ts)

    print(f"  Candidates after frame: {len(candidates)}")
    print(f"  Filtered out:")
    for reason, count in filtered_stats.most_common():
        print(f"    {reason}: {count}")
    print()

    if len(candidates) == 0:
        print("ERROR: No candidates after filtering!")
        return 1

    # Sample check
    print("Sample candidates:")
    for i, (sid, (path, size, tc, ts)) in enumerate(list(candidates.items())[:5], 1):
        print(f"  {i}. {sid} {size/1e6:.2f}MB {tc} tools {ts.isoformat()}")
    print()

    if args.dry_run:
        print("DRY RUN complete.")
        return 0

    # Real run
    limit = args.limit if args.limit else len(candidates)
    print(f"Generating questions from up to {limit} sessions...")
    print(f"Output: {OUTPUT}")
    print()

    # Load existing to resume
    generated_pairs = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    generated_pairs.add(rec["pair_id"])
                except:
                    pass
        print(f"Resuming: {len(generated_pairs)} questions already generated")

    total_cost_usd = 0.0
    rejected_boilerplate = 0
    band_distribution = Counter()
    max_retries = 3

    session_list = list(candidates.items())
    random.shuffle(session_list)

    with open(OUTPUT, "a") as out:
        sessions_done = 0

        for session_id, (path, size, tool_count, ts) in session_list:
            if sessions_done >= limit:
                break

            print(f"[{sessions_done + 1}/{limit}] Session {session_id}...")

            transcript = path.read_text(errors="replace")
            chunks = extract_substantive_chunks(transcript)
            if not chunks:
                print("  Skipped: no substantive chunks")
                continue

            # Stratify chunks into bands
            bands = stratify_chunks_by_position(chunks, len(transcript), N_BANDS)
            band_counts = {b: len(cs) for b, cs in bands.items()}
            print(f"  {len(chunks)} chunks across bands: {band_counts}")

            # Generate QUESTIONS_PER_SESSION questions, cycling through bands
            questions_generated = 0
            for band_idx in range(N_BANDS):
                if questions_generated >= QUESTIONS_PER_SESSION:
                    break

                if not bands[band_idx]:
                    print(f"  Band {band_idx}: no chunks, skipping")
                    continue

                pair_id = f"probe_{session_id}_{band_idx:02d}"
                if pair_id in generated_pairs:
                    continue

                # Sample with retry for boilerplate guard
                result = None
                for retry in range(max_retries):
                    sample_result = sample_from_band(bands[band_idx], len(transcript))
                    if not sample_result:
                        continue

                    context, position, _ = sample_result

                    result = generate_question(context, position, band_idx, pair_id)
                    if not result:
                        print(f"  Band {band_idx}: generation failed, retrying...")
                        continue

                    # Boilerplate guard
                    if has_boilerplate_markers(result["question"]):
                        rejected_boilerplate += 1
                        print(f"  Band {band_idx}: REJECTED boilerplate (retry {retry+1}/{max_retries})")
                        result = None
                        continue

                    break

                if not result:
                    print(f"  Band {band_idx}: all retries failed")
                    continue

                # Write question
                write_question_record(
                    out, session_id, pair_id,
                    result["question"], result["ground_truth"],
                    band_idx, position
                )

                # Track cost
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost_usd = (prompt_tokens * 3.0 / 1e6) + (completion_tokens * 15.0 / 1e6)
                total_cost_usd += cost_usd

                band_distribution[band_idx] += 1
                questions_generated += 1
                print(f"  Band {band_idx}: generated (pos={position:.3f}, cost=${cost_usd:.4f})")

            if questions_generated > 0:
                sessions_done += 1

            # Hard budget cap
            if total_cost_usd >= 30.0:
                print(f"\nSTOP: Reached $30 hard cap (${total_cost_usd:.4f})")
                break

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Sessions: {sessions_done}")
    print(f"Questions: {sum(band_distribution.values())}")
    print(f"Band distribution: {dict(band_distribution)}")
    print(f"Boilerplate rejections: {rejected_boilerplate}")
    print(f"Total cost: ${total_cost_usd:.4f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
