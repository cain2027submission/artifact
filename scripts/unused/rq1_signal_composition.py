#!/usr/bin/env python3
"""Measure signal composition in Claude Code transcripts.

Categorizes every content block by type and measures bytes:
- tool_result: output from tools (the noise hypothesis)
- tool_use: agent's tool invocation params
- thinking: model reasoning (thinking, redacted_thinking)
- assistant_text: assistant prose
- user_text: genuine human turns
- system_injected: system-reminder, isMeta entries, tool results in user role
- other: unclassified content (should be small)

Measures both:
1. Serialized bytes: len(json.dumps(block)) for structural fidelity
2. Payload text bytes: length of human-readable content

Per transcript tracks: path, project, bytes, entries, deduplicated API calls,
turns, has subagents, compaction entries, token totals.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR


@dataclass
class SessionStats:
    """Per-session statistics."""
    file_path: str
    project: str
    total_serialized_bytes: int
    total_payload_bytes: int
    entry_count: int
    api_requests: int  # deduplicated on requestId
    turns: int  # distinct assistant requestIds
    has_subagents: bool
    compaction_entries: int
    usage_input_tokens: int
    usage_output_tokens: int

    # Serialized bytes by category
    tool_result_bytes: int = 0
    tool_use_bytes: int = 0
    thinking_bytes: int = 0
    assistant_text_bytes: int = 0
    user_text_bytes: int = 0
    system_injected_bytes: int = 0
    other_bytes: int = 0

    # Payload text bytes by category
    tool_result_payload: int = 0
    tool_use_payload: int = 0
    thinking_payload: int = 0
    assistant_text_payload: int = 0
    user_text_payload: int = 0
    system_injected_payload: int = 0
    other_payload: int = 0


def extract_payload_text(block):
    """Extract the human-readable text from a content block."""
    block_type = block.get('type')

    if block_type == 'text':
        return block.get('text', '')
    elif block_type == 'thinking':
        return block.get('thinking', '')
    elif block_type == 'redacted_thinking':
        return block.get('redacted_thinking', '')
    elif block_type == 'tool_use':
        # Input parameters as JSON
        return json.dumps(block.get('input', {}))
    elif block_type == 'tool_result':
        content = block.get('content', '')
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Content can be list of blocks
            return ' '.join(str(c) for c in content)
        return str(content)
    else:
        # For unknown types, try to extract text
        for key in ['text', 'content', 'message']:
            if key in block:
                val = block[key]
                if isinstance(val, str):
                    return val
                return str(val)
        return ''


def is_system_injected_text(text):
    """Check if text is system-injected content."""
    markers = [
        '<system-reminder>',
        '<local-command-caveat>',
        '<budget:token_budget>',
        '<total_tokens>',
        'Caveat: The messages below were generated'
    ]
    return any(marker in text for marker in markers)


def categorize_block(block, entry):
    """Categorize a content block and return (category, serialized_bytes, payload_bytes)."""
    block_type = block.get('type')
    serialized_bytes = len(json.dumps(block))
    payload_text = extract_payload_text(block)
    payload_bytes = len(payload_text.encode('utf-8'))

    # Categorize based on block type and context
    if block_type == 'tool_result':
        return 'tool_result', serialized_bytes, payload_bytes

    elif block_type == 'tool_use':
        return 'tool_use', serialized_bytes, payload_bytes

    elif block_type in ['thinking', 'redacted_thinking']:
        return 'thinking', serialized_bytes, payload_bytes

    elif block_type == 'text':
        # Text blocks can be assistant or user, and may be system-injected
        entry_type = entry.get('type')
        is_meta = entry.get('isMeta', False)

        # Check if it's system-injected
        if is_meta or is_system_injected_text(payload_text):
            return 'system_injected', serialized_bytes, payload_bytes

        # Genuine user or assistant text
        role = entry.get('message', {}).get('role')
        if role == 'assistant':
            return 'assistant_text', serialized_bytes, payload_bytes
        elif role == 'user':
            return 'user_text', serialized_bytes, payload_bytes
        else:
            return 'other', serialized_bytes, payload_bytes

    else:
        # Unknown block type
        return 'other', serialized_bytes, payload_bytes


def process_transcript(file_path):
    """Process a single transcript file and return SessionStats."""
    path = Path(file_path)
    project = path.parent.name

    # Accumulators
    category_serialized = defaultdict(int)
    category_payload = defaultdict(int)
    entry_count = 0
    request_ids = set()
    assistant_request_ids = set()
    has_subagents = False
    compaction_entries = 0

    # Track usage deduplicated by requestId
    usage_by_request = {}

    with open(file_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_count += 1

            # Track metadata
            if entry.get('isSidechain'):
                has_subagents = True

            entry_type = entry.get('type')
            if entry_type in ['compaction', 'summary']:
                compaction_entries += 1

            # Track requestId
            request_id = entry.get('requestId')
            if request_id:
                request_ids.add(request_id)

                # Track assistant turns
                if entry.get('message', {}).get('role') == 'assistant':
                    assistant_request_ids.add(request_id)

                # Deduplicate usage
                usage = entry.get('usage')
                if usage and request_id not in usage_by_request:
                    usage_by_request[request_id] = usage

            # Process content blocks
            content = entry.get('message', {}).get('content')

            if isinstance(content, str):
                # String content - treat as single text block
                block = {'type': 'text', 'text': content}
                cat, ser_bytes, pay_bytes = categorize_block(block, entry)
                category_serialized[cat] += ser_bytes
                category_payload[cat] += pay_bytes

            elif isinstance(content, list):
                # List of content blocks
                for block in content:
                    cat, ser_bytes, pay_bytes = categorize_block(block, entry)
                    category_serialized[cat] += ser_bytes
                    category_payload[cat] += pay_bytes

    # Sum usage tokens
    total_input = sum(u.get('input_tokens', 0) for u in usage_by_request.values())
    total_output = sum(u.get('output_tokens', 0) for u in usage_by_request.values())

    # Calculate totals
    total_serialized = sum(category_serialized.values())
    total_payload = sum(category_payload.values())

    return SessionStats(
        file_path=str(file_path),
        project=project,
        total_serialized_bytes=total_serialized,
        total_payload_bytes=total_payload,
        entry_count=entry_count,
        api_requests=len(request_ids),
        turns=len(assistant_request_ids),
        has_subagents=has_subagents,
        compaction_entries=compaction_entries,
        usage_input_tokens=total_input,
        usage_output_tokens=total_output,
        tool_result_bytes=category_serialized['tool_result'],
        tool_use_bytes=category_serialized['tool_use'],
        thinking_bytes=category_serialized['thinking'],
        assistant_text_bytes=category_serialized['assistant_text'],
        user_text_bytes=category_serialized['user_text'],
        system_injected_bytes=category_serialized['system_injected'],
        other_bytes=category_serialized['other'],
        tool_result_payload=category_payload['tool_result'],
        tool_use_payload=category_payload['tool_use'],
        thinking_payload=category_payload['thinking'],
        assistant_text_payload=category_payload['assistant_text'],
        user_text_payload=category_payload['user_text'],
        system_injected_payload=category_payload['system_injected'],
        other_payload=category_payload['other'],
    )


def compute_percentiles(values, p):
    """Compute percentile p (0-100) of sorted values."""
    if not values:
        return 0
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(values):
        return values[-1]
    return values[f] + (k - f) * (values[c] - values[f])


def compute_summary(all_stats):
    """Compute aggregate statistics."""
    n = len(all_stats)

    # Categories to summarize
    categories = ['tool_result', 'tool_use', 'thinking', 'assistant_text',
                  'user_text', 'system_injected', 'other']

    # Compute shares for each session
    payload_shares = defaultdict(list)
    for stats in all_stats:
        total = stats.total_payload_bytes
        if total > 0:
            for cat in categories:
                bytes_val = getattr(stats, f'{cat}_payload')
                share = bytes_val / total
                payload_shares[cat].append(share)

    # Overall statistics
    summary = {
        'n_sessions': n,
        'total_corpus_bytes': sum(s.total_payload_bytes for s in all_stats),
        'total_api_requests': sum(s.api_requests for s in all_stats),
        'total_turns': sum(s.turns for s in all_stats),
        'sessions_with_subagents': sum(1 for s in all_stats if s.has_subagents),
    }

    # Per-category statistics (using payload bytes as primary measure)
    for cat in categories:
        shares = payload_shares[cat]
        summary[f'{cat}_mean'] = sum(shares) / len(shares) if shares else 0
        summary[f'{cat}_median'] = compute_percentiles(shares, 50)
        summary[f'{cat}_p10'] = compute_percentiles(shares, 10)
        summary[f'{cat}_p90'] = compute_percentiles(shares, 90)

    # Breakdown by session size (deciles)
    sorted_by_size = sorted(all_stats, key=lambda s: s.total_payload_bytes)
    decile_size = len(sorted_by_size) // 10

    summary['by_size_decile'] = []
    for i in range(10):
        start = i * decile_size
        end = start + decile_size if i < 9 else len(sorted_by_size)
        decile_sessions = sorted_by_size[start:end]

        decile_summary = {'decile': i + 1}
        for cat in categories:
            shares = []
            for s in decile_sessions:
                if s.total_payload_bytes > 0:
                    shares.append(getattr(s, f'{cat}_payload') / s.total_payload_bytes)
            decile_summary[f'{cat}_mean'] = sum(shares) / len(shares) if shares else 0

        summary['by_size_decile'].append(decile_summary)

    # Breakdown by subagent usage
    with_subagents = [s for s in all_stats if s.has_subagents]
    without_subagents = [s for s in all_stats if not s.has_subagents]

    summary['with_subagents'] = {}
    summary['without_subagents'] = {}

    for cat in categories:
        # With subagents
        shares = []
        for s in with_subagents:
            if s.total_payload_bytes > 0:
                shares.append(getattr(s, f'{cat}_payload') / s.total_payload_bytes)
        summary['with_subagents'][f'{cat}_mean'] = sum(shares) / len(shares) if shares else 0

        # Without subagents
        shares = []
        for s in without_subagents:
            if s.total_payload_bytes > 0:
                shares.append(getattr(s, f'{cat}_payload') / s.total_payload_bytes)
        summary['without_subagents'][f'{cat}_mean'] = sum(shares) / len(shares) if shares else 0

    return summary


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # Find all transcript files recursively
    projects_dir = PROJECTS_DIR
    transcript_files = list(projects_dir.rglob('*.jsonl'))

    print(f"Found {len(transcript_files)} transcript files")
    print(f"Processing...")

    all_stats = []
    for i, file_path in enumerate(transcript_files):
        if i % 100 == 0:
            print(f"  {i}/{len(transcript_files)}", file=sys.stderr)

        try:
            stats = process_transcript(file_path)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            continue

    print(f"Processed {len(all_stats)} sessions successfully")

    # Write per-session CSV
    output_dir = repo_root / "data" / "paper-rollups"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'rq1_per_session.csv'
    with open(csv_path, 'w', newline='') as f:
        if all_stats:
            writer = csv.DictWriter(f, fieldnames=asdict(all_stats[0]).keys())
            writer.writeheader()
            for stats in all_stats:
                writer.writerow(asdict(stats))

    print(f"Wrote {csv_path}")

    # Compute and write summary
    summary = compute_summary(all_stats)

    json_path = output_dir / 'rq1_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {json_path}")

    # Print key findings
    print("\n=== Key Findings ===")
    print(f"N sessions: {summary['n_sessions']}")
    print(f"Total corpus: {summary['total_corpus_bytes']:,} bytes")
    print(f"Sessions with subagents: {summary['sessions_with_subagents']}")
    print()
    print("Signal composition (payload bytes, median/mean):")
    for cat in ['tool_result', 'tool_use', 'thinking', 'assistant_text', 'user_text', 'system_injected', 'other']:
        median = summary[f'{cat}_median'] * 100
        mean = summary[f'{cat}_mean'] * 100
        print(f"  {cat:20s}: {median:5.1f}% / {mean:5.1f}%")


if __name__ == '__main__':
    main()
