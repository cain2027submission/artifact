#!/usr/bin/env python3
"""Generate held-out question pairs for RQ3 evaluation.

Fixes three defects in the existing question set:
1. Questions were visible during substrate engineering (circular)
2. Questions sampled only the first ~30k chars (probe-window bias)
3. Difficulty was never balanced across classes

This generator:
- Uses project's own substantive-session filter (rq1_corrected_per_session.csv)
- Excludes infrastructure sessions (rollup/summariser/drain)
- Samples SUBSTANTIVE CHUNKS uniformly within sessions
- Records true character position to prove whole-transcript coverage
- Emits difficulty-matched pairs (attribution + detail about same event)
- Guards against boilerplate questions

Usage:
    holdout_question_gen.py --dry-run           # zero cost validation
    holdout_question_gen.py --limit 10          # pilot: 10 pairs (20 questions)
    holdout_question_gen.py --limit 150         # full run: 150 pairs (300 questions)
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR

# Gateway and model configuration
GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-sonnet-5"

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
EXISTING_QUESTIONS = DATA / "rq3_questions.jsonl"
PER_SESSION_METRICS = DATA / "rq1_corrected_per_session.csv"
OUTPUT = DATA / "holdout_questions_pilot_v3.jsonl"

# Transcript sources
SOURCES = [
    PROJECTS_DIR,
    Path.home() / "Backups" / "claude-transcripts" / "projects",
]

# Session ID pattern from slug
SLUG_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z-]+_([0-9a-f]{8})_")

# Seed for reproducibility
SEED = int(os.environ.get("SEED", 42))

# Boilerplate guard
BOILERPLATE_PHRASES = [
    "system prompt",
    "system-reminder",
    "system instructions",
    "tool definition",
    "input schema",
    "according to the instructions",
    "CLAUDE.md",
    "AGENTS.md",
    "MEMORY.md",
    "tool schema",
    "parameters schema",
    "properties defined",
]


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    """Post JSON to gateway."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def call_llm(prompt: str, model: str = MODEL, timeout: int = 600) -> Tuple[str, dict]:
    """Call the gateway. No temperature parameter (rejected on Claude 5 models)."""
    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        data = post(f"{GATEWAY}/v1/chat/completions", payload, timeout=timeout)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if not content or not content.strip():
            print(f"  WARNING: empty response, finish_reason={choice.get('finish_reason')}")
            return "", data.get("usage", {})
        return content.strip(), data.get("usage", {})
    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        print(f"  ERROR calling {model}: {error}")
        return "", {}


def load_existing_session_ids() -> set:
    """Load session_ids from existing questions to exclude them."""
    session_ids = set()
    if not EXISTING_QUESTIONS.exists():
        return session_ids

    with open(EXISTING_QUESTIONS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                session_ids.add(record["session_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    return session_ids


def load_substantive_sessions() -> Set[str]:
    """Load substantive session IDs from project's per-session metrics.

    Uses the project's own predicate from rq1_signal_composition_corrected.py:
        tool_calls >= 1 and payload_chars >= 5000, parent sessions only
    """
    substantive = set()

    with open(PER_SESSION_METRICS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_parent = int(row["is_subagent"]) == 0
            tool_calls = int(row["tool_calls"])
            payload_chars = int(row["payload_chars"])

            if is_parent and tool_calls >= 1 and payload_chars >= 5000:
                substantive.add(row["session_id"])

    return substantive


def is_infrastructure_session(transcript_text: str) -> Tuple[bool, str]:
    """Check if transcript is an infrastructure session (rollup/summariser/drain).

    Returns (is_infrastructure, reason).
    """
    # Check first 3000 chars for infrastructure markers
    head = transcript_text[:3000]

    # Rollup-summariser sessions
    if "rollup-summariser" in head or "rollup_summariser" in head:
        return True, "rollup-summariser"

    # Queue operation sessions
    if '"type":"queue-operation"' in head and '"operation":"enqueue"' in head:
        return True, "queue-operation"

    # Agent role identifiers for summarisers
    if '"agentSetting":"rollup' in head or 'agent-setting","agentSetting":"drain' in head:
        return True, "agent-setting-infrastructure"

    return False, ""


def is_boilerplate_chunk(chunk: str) -> bool:
    """Check if a transcript chunk is boilerplate."""
    if "<system-reminder>" in chunk or "</system-reminder>" in chunk:
        return True

    # Tool definitions (have JSON schema structure)
    schema_markers = ['"parameters":', '"$schema":', '"additionalProperties":']
    if sum(1 for m in schema_markers if m in chunk) >= 2:
        return True

    # MCP instructions
    if "mcp_instructions_delta" in chunk:
        return True

    # Large instructional blocks
    if len(chunk) > 5000:
        instruction_markers = [
            "## how a rule in this file is written",
            "## finding a note",
            "Contents of ~/.claude/CLAUDE.md",
        ]
        if any(m in chunk for m in instruction_markers):
            return True

    return False


def extract_substantive_chunks(transcript_text: str) -> List[Tuple[str, int, int]]:
    """Extract transcript chunks that are NOT boilerplate."""
    chunks = []
    char_pos = 0

    for line in transcript_text.splitlines():
        line_start = char_pos
        line_end = char_pos + len(line)
        char_pos = line_end + 1

        if not line.strip():
            continue

        # Keep lines >= 200 chars that aren't boilerplate
        if len(line) >= 200 and not is_boilerplate_chunk(line):
            chunks.append((line, line_start, line_end))

    return chunks


def sample_chunk(chunks: List[Tuple[str, int, int]],
                transcript_length: int) -> Optional[Tuple[str, float]]:
    """Sample one chunk uniformly and return it with its position."""
    if not chunks:
        return None

    chunk_text, start_pos, end_pos = random.choice(chunks)
    midpoint = (start_pos + end_pos) / 2
    position = midpoint / transcript_length if transcript_length > 0 else 0.0

    context = chunk_text[:3000]
    return context, position


def has_boilerplate_markers(text: str) -> bool:
    """Check if question text contains boilerplate markers."""
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in BOILERPLATE_PHRASES)


def generate_question_pair(event_context: str, position: float,
                          pair_id: str, dry_run: bool = False) -> Optional[dict]:
    """Generate an attribution + detail question pair."""
    if dry_run:
        return {
            "pair_id": pair_id,
            "evidence_position": position,
            "attribution": {"question": "[DRY RUN]", "ground_truth": "[DRY RUN]"},
            "detail": {"question": "[DRY RUN]", "ground_truth": "[DRY RUN]"},
        }

    prompt = f"""You are generating questions for an agent transcript evaluation study.

Below is an excerpt from an agent session showing actual work being done:

{event_context}

Generate TWO questions about the SAME event or action in this excerpt:
1. An ATTRIBUTION question: which tool, file, command, error, or person was involved
2. A DETAIL question: what specific value, path, result, count, or parameter was present

CRITICAL: Ask about what the agent DID or what actually HAPPENED in THIS session.
DO NOT ask about:
- System prompts, instructions, or reminders
- Tool definitions or API schemas
- General rules or configuration
- Documentation content

For each question:
- Make it answerable ONLY from what actually happened in this session
- Provide the ground truth answer
- Keep questions concrete and specific
- Include numbers, file names, or error messages when present

Return your response as JSON in this exact format:
{{
  "attribution_question": "Which file...",
  "attribution_ground_truth": "The file was...",
  "detail_question": "What was the specific...",
  "detail_ground_truth": "The value was..."
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
            "evidence_position": position,
            "attribution": {
                "question": data["attribution_question"],
                "ground_truth": data["attribution_ground_truth"],
            },
            "detail": {
                "question": data["detail_question"],
                "ground_truth": data["detail_ground_truth"],
            },
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
            # Extract session ID from filename
            stem = p.stem
            # Full UUID format (primary)
            if len(stem) == 36 and stem.count('-') == 4:
                session_id = stem
            # Agent subagent format: agent-a<hex>
            elif stem.startswith("agent-"):
                session_id = stem
            # 8-char prefix (fallback for older format)
            elif len(stem) >= 8:
                session_id = stem[:8]
            else:
                continue

            by_id.setdefault(session_id, []).append(p)

    # Resolve to single best transcript per ID
    resolved = {}
    for session_id, paths in by_id.items():
        # Prefer non-subagent, then largest
        best = sorted(paths,
                     key=lambda p: ("subagents" in p.parts, -p.stat().st_size))[0]
        resolved[session_id] = best

    return resolved


def extract_session_id_prefix(slug: str) -> Optional[str]:
    """Extract 8-char session ID from rollup note slug."""
    m = SLUG_PATTERN.match(slug)
    return m.group(1) if m else None


def load_transcript(path: Path) -> str:
    """Load full transcript as text."""
    try:
        return path.read_text(errors="replace")
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return ""


def get_first_user_turn(transcript_text: str, max_chars: int = 200) -> str:
    """Extract first real user turn from transcript."""
    for line in transcript_text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            # Claude Code transcript format: type="user" with nested message object
            if record.get("type") == "user":
                message = record.get("message", {})
                if message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str) and len(content.strip()) > 20:
                        # Skip system reminders and metadata
                        if "<system-reminder>" in content or content.startswith("{"):
                            continue
                        return content[:max_chars]
        except json.JSONDecodeError:
            continue

    return "(could not extract first user turn)"


def write_question_record(f, session_id: str, pair_id: str,
                         question_class: str, question: str,
                         ground_truth: str, evidence_position: float):
    """Write one question record to output file."""
    record = {
        "session_id": session_id,
        "pair_id": pair_id,
        "class": question_class,
        "question": question,
        "ground_truth": ground_truth,
        "evidence_position": evidence_position,
    }
    f.write(json.dumps(record) + "\n")
    f.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate selection and sampling without LLM calls")
    parser.add_argument("--limit", type=int, default=None,
                       help="Generate N pairs (2N questions), then stop")
    args = parser.parse_args()

    random.seed(SEED)
    print(f"Seed: {SEED}")
    print()

    # Load existing questions to exclude
    print("Loading existing questions...")
    existing_ids = load_existing_session_ids()
    print(f"  Existing questions: {len(existing_ids)} sessions to exclude")

    # Load project's substantive sessions
    print("\nLoading substantive sessions (project's own filter)...")
    print("  Predicate: tool_calls >= 1 AND payload_chars >= 5000 AND is_subagent == 0")
    substantive_ids = load_substantive_sessions()
    print(f"  Substantive sessions: {len(substantive_ids)}")

    # Index transcripts
    print("\nIndexing transcripts...")
    all_transcripts = index_transcripts()
    print(f"  Found {len(all_transcripts)} transcripts")

    # Filter to substantive, held-out sessions
    candidate_ids = substantive_ids - existing_ids
    candidates = {sid: path for sid, path in all_transcripts.items() if sid in candidate_ids}
    print(f"  Held-out substantive candidates: {len(candidates)}")

    # Filter out infrastructure sessions
    print("\nFiltering infrastructure sessions...")
    infra_counts = Counter()
    work_sessions = {}

    for session_id, path in candidates.items():
        transcript = load_transcript(path)
        if not transcript:
            continue

        is_infra, reason = is_infrastructure_session(transcript)
        if is_infra:
            infra_counts[reason] += 1
        else:
            work_sessions[session_id] = path

    print(f"  Infrastructure sessions excluded:")
    for reason, count in infra_counts.most_common():
        print(f"    {reason}: {count}")
    print(f"  Work sessions remaining: {len(work_sessions)}")
    print()

    if len(work_sessions) == 0:
        print("ERROR: No work sessions available after filtering!")
        return 1

    # Sanity check: show first user turns
    print("Sanity check: First user turns from 5 random sessions:")
    sample_sessions = random.sample(list(work_sessions.items()), min(5, len(work_sessions)))
    for i, (session_id, path) in enumerate(sample_sessions, 1):
        transcript = load_transcript(path)
        first_turn = get_first_user_turn(transcript)
        print(f"  {i}. [{session_id}] {first_turn}")
    print()

    if args.dry_run:
        print("DRY RUN complete. Proceeding would generate from these sessions.")
        return 0

    # Real run: generate questions
    limit = args.limit if args.limit else len(work_sessions)
    print(f"Generating up to {limit} question pairs...")
    print(f"Output: {OUTPUT}")
    print()

    # Load existing output to resume
    generated_pairs = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    generated_pairs.add(rec["pair_id"])
                except:
                    pass
        print(f"Resuming: {len(generated_pairs) // 2} pairs already generated")

    total_cost_usd = 0.0
    total_tokens = {"prompt": 0, "completion": 0}
    rejected_boilerplate = 0
    max_retries = 3

    session_list = list(work_sessions.items())
    random.shuffle(session_list)

    with open(OUTPUT, "a") as out:
        pairs_generated = len(generated_pairs) // 2

        for session_id, path in session_list:
            if pairs_generated >= limit:
                break

            pair_id = f"holdout_{session_id}_{pairs_generated:03d}"
            if pair_id in generated_pairs:
                continue

            print(f"[{pairs_generated + 1}/{limit}] Session {session_id}...")

            transcript = load_transcript(path)
            if not transcript:
                print("  Skipped: could not load")
                continue

            chunks = extract_substantive_chunks(transcript)
            if not chunks:
                print("  Skipped: no substantive chunks")
                continue

            print(f"  Found {len(chunks)} chunks")

            # Sample with retry for boilerplate guard
            result = None
            for retry in range(max_retries):
                sample_result = sample_chunk(chunks, len(transcript))
                if not sample_result:
                    continue

                context, position = sample_result
                print(f"  Sampled at position {position:.3f}")

                result = generate_question_pair(context, position, pair_id)
                if not result:
                    print("  Generation failed, retrying...")
                    continue

                # Boilerplate guard
                attr_q = result["attribution"]["question"]
                detail_q = result["detail"]["question"]

                if has_boilerplate_markers(attr_q) or has_boilerplate_markers(detail_q):
                    rejected_boilerplate += 1
                    print(f"  REJECTED: boilerplate (retry {retry+1}/{max_retries})")
                    result = None
                    continue

                break

            if not result:
                print("  Skipped: all retries failed")
                continue

            # Write both questions
            session_slug = f"holdout_{session_id}"

            write_question_record(
                out, session_slug, pair_id, "attribution",
                result["attribution"]["question"],
                result["attribution"]["ground_truth"],
                position
            )

            write_question_record(
                out, session_slug, pair_id, "detail",
                result["detail"]["question"],
                result["detail"]["ground_truth"],
                position
            )

            # Track cost
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens["prompt"] += prompt_tokens
            total_tokens["completion"] += completion_tokens

            cost_usd = (prompt_tokens * 3.0 / 1e6) + (completion_tokens * 15.0 / 1e6)
            total_cost_usd += cost_usd

            print(f"  Generated pair")
            print(f"  Cost: ${cost_usd:.4f}")
            print()

            pairs_generated += 1

            if total_cost_usd >= 5.0:
                print(f"STOP: Reached $5 ceiling (${total_cost_usd:.4f})")
                break

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Pairs generated: {pairs_generated}")
    print(f"Questions generated: {pairs_generated * 2}")
    print(f"Boilerplate rejections: {rejected_boilerplate}")
    print(f"Total cost: ${total_cost_usd:.4f}")
    print()

    if pairs_generated > 0:
        cost_per_pair = total_cost_usd / pairs_generated
        extrapolated_300 = cost_per_pair * 150
        print(f"Extrapolated cost for 150 pairs (300 questions): ${extrapolated_300:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
