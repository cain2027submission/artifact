#!/usr/bin/env python3
"""
SUPERSEDED 2026-10-04. Kept as the stage 1 artifact, not as live code.

Replaced by `correction_detector.py` (detection) and `correction_pairs.py`
(pairing). Two things in this file are wrong and are the reason for the swap:

1. `CORRECTION_MARKERS` is a negation vocabulary, and the human whose turns it
   scans corrects by proposing the better alternative instead ("but ideally",
   "it feels", "could we instead"). It flagged 708 of 3,641 turns while missing
   the one correction independently known to exist.

2. `build_candidate_list` gates on `session_id == '6d7b783e'` while `session_id`
   holds a full uuid, so `positive_control_hit` was unreachable and the control
   reported FAIL as a constant. A perfect detector would have been halted by the
   same line, so the stage 1 `FAIL` carries no information about the detector.
   `positive_control_session_found` was computed and never written out, so the
   artifact could not even show whether the control session was in the frame.

Its denominator is also wrong: "user text turns" counted peer hand-backs,
harness continuation summaries and this project's own benchmark probes, all of
which arrive as `type: "user"`. See `human_turns.py` for the corrected count.

Correction survival detector for the pre-registered rollup test.

This script implements stage 1 of the pre-registered test documented in:
tasks/2026-10-04 pre-registered test of whether session rollups drop corrected turns.md

Frame: sessions/*.md files with date >= 2026-09-21 and resolvable transcripts.
Detector: frozen lexical marker list applied to user text turns only (not tool results).
Output: paired correction+control candidates for blind grading in stage 2.

Control selection rule (documented per pre-registration):
For each correction at turn index C, the control is selected from the nearest
substantive assistant text turn BEFORE the correction (searching backward from C-1)
that: (1) was not flagged by the correction detector on its preceding user turn,
(2) contains >= 20 characters of text content, and (3) is within 2x the character
length of the correction text. If no qualifying turn is found searching backward,
search forward from C+1 using the same criteria.
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


# FROZEN correction markers per pre-registration
CORRECTION_MARKERS = [
    "no,",
    "that's wrong",
    "thats wrong",
    "incorrect",
    "not true",
    "you're wrong",
    "actually,",
    "i said",
    "revert",
    "undo",
    "that is not",
    "isn't right",
    "wrong",
]


def load_yaml_frontmatter(md_path: str) -> Dict:
    """Extract YAML frontmatter from markdown file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---\n'):
        return {}

    # Find the closing ---
    end_idx = content.find('\n---\n', 4)
    if end_idx == -1:
        return {}

    fm_text = content[4:end_idx]

    # Parse minimal YAML subset (date and session_id)
    result = {}
    for line in fm_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()

    return result


def find_transcript_path(session_id: str) -> Optional[str]:
    """Find transcript for session_id in ~/.claude/projects/*/<session_id>.jsonl"""
    pattern = os.path.expanduser(f'~/.claude/projects/*/{session_id}.jsonl')
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def resolve_rollup_note_by_session_id(session_id: str, sessions_dir: str) -> Optional[str]:
    """
    Resolve rollup note by session ID using glob pattern.
    Returns None if not found or if multiple notes exist (ambiguous case).
    Raises RuntimeError only if zero matches (true error per pre-registration).

    Multiple matches occur when one session produces multiple rollup notes.
    This case was not specified in the pre-registration, so we conservatively
    skip these sessions to preserve the paired design.
    """
    pattern = os.path.join(sessions_dir, f'*_{session_id[:8]}_*.md')
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise RuntimeError(f"SCRIPT ERROR: No rollup note found for session_id {session_id} (pattern: {pattern})")

    if len(matches) > 1:
        # Return None to signal ambiguous case (will be tracked in stats)
        return None

    return matches[0]


def is_tool_result_turn(message: Dict) -> bool:
    """Check if this user message is actually a tool result."""
    if message.get('type') != 'user':
        return False

    content = message.get('message', {}).get('content', [])
    if not isinstance(content, list):
        return False

    # Check if any content block is a tool_result
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_result':
            return True

    return False


def is_system_reminder_only(message: Dict) -> bool:
    """Check if this user message contains only system-reminder content."""
    if message.get('type') != 'user':
        return False

    content = message.get('message', {}).get('content', '')

    # Handle string content
    if isinstance(content, str):
        # Remove system-reminder blocks and check if anything remains
        text = re.sub(r'<system-reminder>.*?</system-reminder>', '', content, flags=re.DOTALL)
        return not text.strip()

    # Handle list content
    if not isinstance(content, list):
        return False

    # Check if all text blocks are system-reminders or cache_control
    for block in content:
        if isinstance(block, dict):
            if block.get('type') == 'text':
                text = block.get('text', '')
                # If there's non-empty text that's not wrapped in system-reminder tags, it's real content
                if text and '<system-reminder>' not in text:
                    return False

    return True


def extract_user_text(message: Dict) -> str:
    """Extract text content from a user message, excluding system-reminders."""
    content = message.get('message', {}).get('content', '')

    # Handle string content (direct text)
    if isinstance(content, str):
        # Remove system-reminder blocks
        text = re.sub(r'<system-reminder>.*?</system-reminder>', '', content, flags=re.DOTALL)
        return text.strip()

    # Handle list content (blocks)
    if not isinstance(content, list):
        return ""

    texts = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'text':
            text = block.get('text', '')
            # Remove system-reminder blocks
            text = re.sub(r'<system-reminder>.*?</system-reminder>', '', text, flags=re.DOTALL)
            text = text.strip()
            if text:
                texts.append(text)

    return ' '.join(texts)


def extract_assistant_text(message: Dict) -> str:
    """Extract text content from an assistant message."""
    content = message.get('message', {}).get('content', '')

    # Handle string content (direct text)
    if isinstance(content, str):
        return content.strip()

    # Handle list content (blocks)
    if not isinstance(content, list):
        return ""

    texts = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'text':
            text = block.get('text', '')
            if text:
                texts.append(text)

    return ' '.join(texts)


def is_correction(text: str) -> bool:
    """Check if text matches any correction marker (case-insensitive)."""
    text_lower = text.lower()
    return any(marker.lower() in text_lower for marker in CORRECTION_MARKERS)


def find_control_fact(transcript: List[Dict], correction_idx: int, correction_text: str,
                      flagged_turns: set) -> Optional[Tuple[int, str]]:
    """
    Find a control fact from the same transcript, selected by position.

    Per pre-registration: search backward from correction for nearest substantive
    assistant text (>=20 chars, within 2x length, not after flagged user turn).
    If not found backward, search forward.
    """
    correction_len = len(correction_text)
    min_len = 20
    max_len = correction_len * 2

    # Search backward from correction_idx - 1
    for i in range(correction_idx - 1, -1, -1):
        msg = transcript[i]
        if msg.get('type') != 'assistant':
            continue

        # Check if this assistant turn followed a flagged user turn
        if i > 0 and i - 1 in flagged_turns:
            continue

        text = extract_assistant_text(msg)
        text_len = len(text)

        if min_len <= text_len <= max_len:
            return (i, text)

    # Search forward from correction_idx + 1
    for i in range(correction_idx + 1, len(transcript)):
        msg = transcript[i]
        if msg.get('type') != 'assistant':
            continue

        # Check if this assistant turn followed a flagged user turn
        if i > 0 and i - 1 in flagged_turns:
            continue

        text = extract_assistant_text(msg)
        text_len = len(text)

        if min_len <= text_len <= max_len:
            return (i, text)

    return None


def load_transcript(transcript_path: str) -> List[Dict]:
    """Load transcript from jsonl file."""
    transcript = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                transcript.append(json.loads(line))
    return transcript


def build_candidate_list(sessions_dir: str, output_path: str) -> Dict:
    """
    Build the candidate list per pre-registration specification.
    Returns validation statistics.
    """
    stats = {
        'total_sessions_in_frame': 0,
        'sessions_with_transcripts': 0,
        'sessions_skipped_multiple_notes': 0,
        'user_text_turns_scanned': 0,
        'corrections_detected': 0,
        'sessions_with_corrections': 0,
        'pairs_emitted': 0,
        'positive_control_hit': False,
        'positive_control_session_found': False,
    }

    # Find all session notes
    note_files = glob.glob(os.path.join(sessions_dir, '*.md'))

    candidates = []

    for note_path in note_files:
        fm = load_yaml_frontmatter(note_path)

        # Check date >= 2026-09-21
        date_str = fm.get('date', '')
        if date_str < '2026-09-21':
            continue

        stats['total_sessions_in_frame'] += 1

        session_id = fm.get('session_id', '').strip()
        if not session_id:
            continue

        # Find transcript
        transcript_path = find_transcript_path(session_id)
        if not transcript_path:
            continue

        stats['sessions_with_transcripts'] += 1

        # Load transcript
        try:
            transcript = load_transcript(transcript_path)
        except Exception as e:
            print(f"Warning: Could not load transcript for {session_id}: {e}", file=sys.stderr)
            continue

        # Scan for corrections in user text turns
        corrections_in_session = []
        flagged_turn_indices = set()

        for i, msg in enumerate(transcript):
            if msg.get('type') != 'user':
                continue

            # Skip tool results
            if is_tool_result_turn(msg):
                continue

            # Skip system-reminder-only turns
            if is_system_reminder_only(msg):
                continue

            # Extract user text
            user_text = extract_user_text(msg)
            if not user_text:
                continue

            stats['user_text_turns_scanned'] += 1

            # Check for correction
            if is_correction(user_text):
                corrections_in_session.append((i, user_text))
                flagged_turn_indices.add(i)
                stats['corrections_detected'] += 1

                # Check positive control
                if session_id == '6d7b783e':
                    stats['positive_control_hit'] = True

        if session_id == '6d7b783e':
            stats['positive_control_session_found'] = True

        if corrections_in_session:
            stats['sessions_with_corrections'] += 1

            # Resolve rollup note by session ID
            try:
                resolved_note_path = resolve_rollup_note_by_session_id(session_id, sessions_dir)
            except RuntimeError as e:
                print(str(e), file=sys.stderr)
                raise

            # Skip if multiple notes (ambiguous case)
            if resolved_note_path is None:
                stats['sessions_skipped_multiple_notes'] += 1
                continue

            # For each correction, find a control fact
            for correction_idx, correction_text in corrections_in_session:
                control = find_control_fact(transcript, correction_idx, correction_text, flagged_turn_indices)

                if control:
                    control_idx, control_text = control

                    candidate = {
                        'session_id': session_id,
                        'note_path': resolved_note_path,
                        'correction_text': correction_text,
                        'control_text': control_text,
                        'correction_turn_index': correction_idx,
                        'control_turn_index': control_idx,
                    }
                    candidates.append(candidate)
                    stats['pairs_emitted'] += 1

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for candidate in candidates:
            f.write(json.dumps(candidate) + '\n')

    return stats


def validate_detector() -> Dict:
    """
    Validate the detector per pre-registration:
    1. Positive control: session 6d7b783e must be detected
    2. Negative control: paraphrased correction with no markers must be missed
    """
    # Test negative control (disguised correction)
    disguised_correction = "hmm, the desktop count was 12 not 14"
    disguised_detected = is_correction(disguised_correction)

    result = {
        'positive_control': {
            'session_id': '6d7b783e',
            'description': 'Session 6d7b783e must be detected as containing a correction',
            'status': 'pending',  # Will be updated during build
        },
        'negative_control': {
            'test_text': disguised_correction,
            'description': 'Paraphrased correction using no marker words must be MISSED',
            'detected': disguised_detected,
            'status': 'PASS' if not disguised_detected else 'FAIL',
        },
        'detector_recall_ceiling': 'Limited by frozen marker list; will miss paraphrases like the negative control',
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Correction survival detector')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    parser.add_argument('--build', action='store_true', help='Build candidate list')
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent.parent.parent  # Go up to repo root
    sessions_dir = script_dir / 'sessions'
    output_dir = script_dir / 'data' / 'paper-rollups'
    candidates_path = output_dir / 'correction_survival_candidates.jsonl'
    validation_path = output_dir / 'correction_survival_detector_validation.json'

    if args.validate:
        # Run validation
        validation_result = validate_detector()

        # Print validation result
        print(json.dumps(validation_result, indent=2))

        # Save validation result
        os.makedirs(output_dir, exist_ok=True)
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, indent=2)

        # Exit with success
        sys.exit(0)

    elif args.build:
        # Build candidate list
        stats = build_candidate_list(str(sessions_dir), str(candidates_path))

        # Load and update validation result
        validation_result = validate_detector()
        validation_result['positive_control']['status'] = 'PASS' if stats['positive_control_hit'] else 'FAIL'

        # Add statistics to validation result
        validation_result['statistics'] = {
            'total_sessions_in_frame': stats['total_sessions_in_frame'],
            'sessions_with_transcripts': stats['sessions_with_transcripts'],
            'sessions_skipped_multiple_notes': stats['sessions_skipped_multiple_notes'],
            'user_text_turns_scanned': stats['user_text_turns_scanned'],
            'corrections_detected': stats['corrections_detected'],
            'sessions_with_corrections': stats['sessions_with_corrections'],
            'pairs_emitted': stats['pairs_emitted'],
        }

        # Save validation result
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_result, f, indent=2)

        # Print summary
        print(json.dumps(validation_result, indent=2))

        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
