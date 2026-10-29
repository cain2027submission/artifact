#!/usr/bin/env python3
"""Part 1: Structural loss accounting - classify what rollups structurally cannot contain."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_corpus import load_manifest, read_pinned
from paths import PROJECTS_DIR, SESSIONS_DIR


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoder."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def parse_frontmatter(content: str) -> Dict:
    """Extract YAML frontmatter from markdown."""
    if not content.startswith('---'):
        return {}

    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}

    fm_text = parts[1]
    result = {}
    for line in fm_text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result


def extract_files_from_rollup(content: str) -> Tuple[set, str]:
    """Extract file paths from rollup. Returns (paths, schema_format).

    Handles two schema generations:
    1. ## Files heading (newer format)
    2. **files edited:** bold label (older format)

    Also scans entire body for backticked paths.

    OVERCOUNTS, and needs the same fix as the transcript side -- which is the
    point: the two sides were different code with different rules, so they could
    not agree about what a path is. This one takes any backticked span containing
    a slash, so `/Users/alice`, `learnings/` and `src/game` become cited artifacts,
    and it scans the frontmatter and the trace-backlink boilerplate, whose literal
    trace path is in every note and in no transcript. Making both sides call one
    normalisation is the fix. Use path_extract.rollup_paths.
    """
    files = set()
    schema_format = "unknown"

    # Check which schema format this rollup uses
    has_files_heading = '## Files' in content
    has_files_label = '**files edited:**' in content or '**Files:**' in content

    if has_files_heading:
        schema_format = "heading"
    elif has_files_label:
        schema_format = "label"

    # Scan entire content for backticked paths
    # Match paths like `/path/to/file` or `path/to/file.ext`
    for match in re.finditer(r'`([^`]+)`', content):
        path = match.group(1)
        # Filter for path-like strings (contains / or ends in common extensions)
        if '/' in path or re.search(r'\.(py|js|ts|md|json|yaml|txt|sh|css|html)$', path):
            files.add(path)

    return files, schema_format


def extract_files_from_transcript(entries: List[Dict]) -> set:
    """Extract file paths from tool_use blocks.

    UNDERCOUNTS, and is kept only so rq3_path_recall.py can show the old and new
    numbers side by side. It reads three structured keys and nothing else, so it
    is blind to every path inside a Bash `command` string -- and Bash is 54% of
    tool calls in this corpus. Session 372ca0e6 is 38 Bash calls, 2 Edits and 1
    Read; this function finds one path in the whole transcript. It also does not
    normalise, so a relative citation can never be paired with an absolute one.
    Use path_extract.transcript_paths.
    """
    files = set()

    for entry in entries:
        if entry.get('type') != 'assistant':
            continue

        msg = entry.get('message', {})
        content = msg.get('content', [])

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'tool_use':
                continue

            input_data = block.get('input', {})
            # Check various tool parameters that contain file paths
            for key in ['file_path', 'path', 'paths']:
                if key in input_data:
                    val = input_data[key]
                    if isinstance(val, str):
                        files.add(val)
                    elif isinstance(val, list):
                        files.update(val)

    return files


def count_commands_in_transcript(entries: List[Dict]) -> int:
    """Count shell commands and tool invocations."""
    count = 0

    for entry in entries:
        if entry.get('type') != 'assistant':
            continue

        msg = entry.get('message', {})
        content = msg.get('content', [])

        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                count += 1

    return count


def count_commands_in_rollup(content: str) -> int:
    """Count shell commands mentioned in rollup (rough heuristic)."""
    # Look for code blocks and inline code that looks like commands
    count = 0

    # Count code blocks
    count += len(re.findall(r'```(?:bash|sh)\n', content))

    # Count inline commands that start with common CLI tools
    common_tools = r'\b(?:git|npm|python|pytest|curl|grep|find|sed|awk|make|cargo|go)\s'
    count += len(re.findall(common_tools, content))

    return count


def analyze_transcript_composition(entries: List[Dict]) -> Dict[str, int]:
    """Break down transcript by content type and token count."""
    composition = {
        'tool_result': 0,
        'tool_use': 0,
        'thinking': 0,
        'assistant_prose': 0,
        'user_text': 0,
    }

    for entry in entries:
        # Skip non-message entries
        if entry.get('type') not in ['user', 'assistant', 'attachment']:
            continue

        # Extract the actual message
        msg = entry.get('message', {})
        role = msg.get('role')
        content = msg.get('content', '')

        if role == 'user':
            # User content can be a string or list of blocks (including tool_result)
            if isinstance(content, str):
                composition['user_text'] += count_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get('type')

                    if block_type == 'text':
                        composition['user_text'] += count_tokens(block.get('text', ''))
                    elif block_type == 'tool_result':
                        # Tool results are in user messages
                        result_content = block.get('content', '')
                        if isinstance(result_content, list):
                            result_content = json.dumps(result_content)
                        composition['tool_result'] += count_tokens(str(result_content))

        elif role == 'assistant':
            # Assistant content is an array of blocks
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get('type')

                    if block_type == 'text':
                        composition['assistant_prose'] += count_tokens(block.get('text', ''))
                    elif block_type == 'thinking':
                        composition['thinking'] += count_tokens(block.get('thinking', ''))
                    elif block_type == 'tool_use':
                        # Tool use includes name + input
                        tool_text = json.dumps(block)
                        composition['tool_use'] += count_tokens(tool_text)

        # NOTE: tool_result blocks are now parsed above in user messages

    return composition


def find_paired_sessions() -> List[Tuple[Path, Path]]:
    """Take the paired set from the frozen manifest, not from a live glob.

    This used to glob sessions/ and projects/ directly, which meant RQ3 ran over
    whatever existed at the moment it was invoked: it reported 302 sessions while
    RQ1, RQ2 and RQ5 all reported the manifest's 300, so the paper's research
    questions described three slightly different corpora. Reading the manifest
    makes the corpus identical across every RQ and reproducible across runs.
    """
    manifest = load_manifest()
    rollups_dir = SESSIONS_DIR
    transcripts_base = PROJECTS_DIR

    # session_id -> transcript path, from the snapshot rather than a fresh rglob.
    transcript_by_sid = {}
    for p in manifest['transcripts']:
        transcript_by_sid[Path(p).stem] = Path(p)

    # Grouped by session, because two sessions in this corpus have two rollup
    # notes each (the pipeline writes one note per finding). Iterating rollup
    # files counted those two transcripts twice and reported 302 units for 300
    # sessions. A session's generated artifact is all of its notes, so they are
    # concatenated rather than one being dropped -- dropping one would shrink the
    # rollup and inflate that session's compression ratio.
    paired = set(manifest['paired_session_ids'])
    by_session: Dict[str, List[Path]] = {}
    for rollup_path in sorted(rollups_dir.glob('*.md')):
        fm = parse_frontmatter(rollup_path.read_text())
        session_id = fm.get('session_id')
        if not session_id or session_id not in paired:
            continue
        if session_id in transcript_by_sid:
            by_session.setdefault(session_id, []).append(rollup_path)

    return [(sorted(rollups), transcript_by_sid[sid])
            for sid, rollups in sorted(by_session.items())]


def main():
    print("Finding paired sessions...")
    pairs = find_paired_sessions()
    print(f"Found {len(pairs)} paired sessions")

    results = {
        # Recorded so this file can be matched to the corpus it was computed
        # over, as RQ1, RQ2 and RQ5 already do.
        'corpus_snapshot_utc': load_manifest()['snapshot_utc'],
        'n_sessions': len(pairs),
        'sessions': [],
        'schema_formats': {
            'heading': 0,  # ## Files format
            'label': 0,    # **files edited:** format
            'unknown': 0,
        },
        'aggregate': {
            'total_transcript_tokens': 0,
            'total_rollup_tokens': 0,
            'dropped_tokens': 0,
            'compressed_tokens': 0,
            'tool_result_tokens': 0,
            'tool_use_tokens': 0,
            'thinking_tokens': 0,
            'assistant_prose_tokens': 0,
            'user_text_tokens': 0,
            'files_in_transcripts': 0,
            'files_in_rollups': 0,
            'commands_in_transcripts': 0,
            'commands_in_rollups': 0,
        }
    }

    for i, (rollup_paths, transcript_path) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(pairs)}")

        # Read rollup(s). Usually one note; two sessions have two.
        rollup_path = rollup_paths[0]
        rollup_content = "\n".join(p.read_text() for p in rollup_paths)
        fm = parse_frontmatter(rollup_content)
        session_id = fm.get('session_id', rollup_path.stem)  # Fall back to filename if no frontmatter

        rollup_tokens = count_tokens(rollup_content)
        rollup_files, schema_format = extract_files_from_rollup(rollup_content)
        rollup_commands = count_commands_in_rollup(rollup_content)

        # Track schema format
        results['schema_formats'][schema_format] += 1

        # Read the transcript as it stood at snapshot time. A live read here lets
        # a session that is still being written change RQ3's totals between runs.
        entries = []
        for line in read_pinned(str(transcript_path)).splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # A pinned snapshot can still leave a truncated line; drop
                    # just that record rather than aborting the whole run.
                    print(f"  WARNING: malformed JSONL line in {transcript_path}, skipping")

        composition = analyze_transcript_composition(entries)
        transcript_files = extract_files_from_transcript(entries)
        transcript_commands = count_commands_in_transcript(entries)

        total_transcript = sum(composition.values())

        # DROPPED: tool_result + tool_use
        dropped = composition['tool_result'] + composition['tool_use']

        # COMPRESSED: thinking + assistant_prose + user_text
        compressed = composition['thinking'] + composition['assistant_prose'] + composition['user_text']

        # Fair compression ratio: compressed / rollup
        fair_compression = compressed / rollup_tokens if rollup_tokens > 0 else 0

        session_result = {
            'session_id': session_id,
            'rollup_filename': rollup_path.stem,
            'rollup_tokens': rollup_tokens,
            'transcript_tokens': total_transcript,
            'composition': composition,
            'dropped_tokens': dropped,
            'compressed_tokens': compressed,
            'fair_compression_ratio': fair_compression,
            'files_transcript': len(transcript_files),
            'files_rollup': len(rollup_files),
            # NOT a recall, and withdrawn from the paper. This is
            # len(rollup_set)/len(transcript_set) -- a ratio of two set
            # cardinalities with no intersection in it -- so it exceeds 1.0
            # whenever the rollup names more path-like tokens than this file's
            # transcript extractor found, which it did on 36 of 300 sessions (up
            # to 400%). On the four worst the true intersection of the two sets
            # was empty, so their real recall was 0%. Renamed rather than
            # deleted so an old artifact can still be read, and left computed so
            # rq3_path_recall.py can put old and new side by side. Nothing
            # consumes it. Use scripts/paper-rollups/rq3_path_recall.py.
            'files_ratio_not_a_recall_withdrawn': (
                len(rollup_files) / len(transcript_files) if transcript_files else 0),
            'commands_transcript': transcript_commands,
            'commands_rollup': rollup_commands,
            'schema_format': schema_format,
        }

        results['sessions'].append(session_result)

        # Aggregate
        agg = results['aggregate']
        agg['total_transcript_tokens'] += total_transcript
        agg['total_rollup_tokens'] += rollup_tokens
        agg['dropped_tokens'] += dropped
        agg['compressed_tokens'] += compressed
        for key in composition:
            agg[f'{key}_tokens'] += composition[key]
        agg['files_in_transcripts'] += len(transcript_files)
        agg['files_in_rollups'] += len(rollup_files)
        agg['commands_in_transcripts'] += transcript_commands
        agg['commands_in_rollups'] += rollup_commands

    # Compute aggregate percentages
    total = results['aggregate']['total_transcript_tokens']
    if total > 0:
        agg = results['aggregate']
        agg['dropped_share'] = agg['dropped_tokens'] / total
        agg['compressed_share'] = agg['compressed_tokens'] / total
        agg['tool_result_share'] = agg['tool_result_tokens'] / total
        agg['tool_use_share'] = agg['tool_use_tokens'] / total
        agg['thinking_share'] = agg['thinking_tokens'] / total
        agg['assistant_prose_share'] = agg['assistant_prose_tokens'] / total
        agg['user_text_share'] = agg['user_text_tokens'] / total

        agg['fair_compression_ratio'] = agg['compressed_tokens'] / agg['total_rollup_tokens']
        # See the per-session note above: this is the pooled form of the same
        # non-recall, and it is the number that was circulated as 44.5%.
        agg['file_ratio_not_a_recall_withdrawn'] = agg['files_in_rollups'] / agg['files_in_transcripts'] if agg['files_in_transcripts'] > 0 else 0

    # Save
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "data" / "paper-rollups" / "rq3_structural_loss.json"
    output_path.write_text(json.dumps(results, indent=2))

    print(f"\nStructural Analysis Complete")
    print(f"  Sessions analyzed: {results['n_sessions']}")
    print(f"  Total transcript tokens: {agg['total_transcript_tokens']:,}")
    print(f"  Total rollup tokens: {agg['total_rollup_tokens']:,}")
    print(f"\nToken Distribution:")
    print(f"  DROPPED (tool_result + tool_use): {agg['dropped_share']:.1%}")
    print(f"    - tool_result: {agg['tool_result_share']:.1%}")
    print(f"    - tool_use: {agg['tool_use_share']:.1%}")
    print(f"  COMPRESSED (thinking + prose + user): {agg['compressed_share']:.1%}")
    print(f"    - thinking: {agg['thinking_share']:.1%}")
    print(f"    - assistant_prose: {agg['assistant_prose_share']:.1%}")
    print(f"    - user_text: {agg['user_text_share']:.1%}")
    print(f"\nCompression Ratios:")
    print(f"  Naive (total transcript / rollup): {total / agg['total_rollup_tokens']:.1f}x")
    print(f"  Fair (compressed only / rollup): {agg['fair_compression_ratio']:.1f}x")
    print(f"\nSet-size Ratios (NOT recalls -- see rq3_path_recall.py):")
    print(f"  File paths: {agg['file_ratio_not_a_recall_withdrawn']:.1%} ({agg['files_in_rollups']:,}/{agg['files_in_transcripts']:,})")
    print(f"  Commands: {agg['commands_in_rollups']:,}/{agg['commands_in_transcripts']:,}")

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
