#!/usr/bin/env python3
"""Build the correction/control pair list for blind grading.

Reads the model-labelled frame from ``correction_frame_labels.jsonl`` and emits
one pair per detected correction: the correction itself, and a non-correction
control fact drawn from the same session, so the primary outcome can be the
**paired** difference in survival rather than a bare survival rate. A rollup is
a lossy summary that is supposed to drop most of its input, so "35% of
corrections survive" is not a finding until the same number exists for material
that was not a correction.

Two defects in the stage 1 pairing are fixed here.

**The length window could not be satisfied.** Stage 1 used
``min_len = 20`` and ``max_len = len(correction) * 2``, so any correction under
10 characters had an empty window and silently produced no pair -- and short
turns are exactly where the blunt corrections live. Worse, there was no lower
bound tied to the correction's own length, so a 20-character control could be
paired against a 2,000-character correction. "Comparable length" has to bind on
both sides or the control arm is systematically easier or harder to recover than
the arm it is controlling for, which moves the headline number directly.

**The flagged-turn test looked at the wrong index.** Stage 1 skipped an
assistant turn when ``i - 1`` was a flagged user turn, which assumes a user turn
is always immediately followed by its assistant reply. Tool results interleave,
so the assistant reply to a flagged turn usually sits several records later and
was not skipped at all. The test now excludes any assistant turn falling between
a flagged user turn and the next user turn of any kind.

Usage:
    python correction_pairs.py --build
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from human_turns import load_transcript

# A control must sit inside this multiplicative band of the correction's length,
# with an absolute floor so a two-word control is never drawn.
CONTROL_MIN_RATIO = 0.5
CONTROL_MAX_RATIO = 2.0
CONTROL_ABS_MIN_CHARS = 40


def extract_assistant_text(record: Dict) -> str:
    """Text blocks of an assistant record, thinking and tool calls excluded."""
    content = record.get('message', {}).get('content', '')
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ''
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'text':
            text = block.get('text', '')
            if text:
                parts.append(text)
    return ' '.join(parts).strip()


def excluded_assistant_indices(transcript: List[Dict], flagged: set) -> set:
    """Assistant records that answer a flagged turn, so cannot be controls.

    An assistant record is excluded if the nearest preceding user record is a
    flagged (correction) turn. Stage 1 tested only ``i - 1``, which tool results
    almost always occupy, so the real reply to a correction was left eligible as
    its own control.
    """
    excluded = set()
    last_user: Optional[int] = None
    for index, record in enumerate(transcript):
        kind = record.get('type')
        if kind == 'user':
            last_user = index
        elif kind == 'assistant' and last_user is not None and last_user in flagged:
            excluded.add(index)
    return excluded


def find_control(transcript: List[Dict], correction_index: int, correction_len: int,
                 flagged: set) -> Optional[Tuple[int, str]]:
    """Nearest positionally-eligible assistant turn of comparable length."""
    low = max(CONTROL_ABS_MIN_CHARS, int(correction_len * CONTROL_MIN_RATIO))
    high = int(correction_len * CONTROL_MAX_RATIO)
    if high < low:
        # A correction too short for any band to be satisfiable. Stage 1 dropped
        # these silently; they are now reported as a named exclusion instead.
        return None

    blocked = excluded_assistant_indices(transcript, flagged)

    def eligible(index: int) -> Optional[str]:
        record = transcript[index]
        if record.get('type') != 'assistant' or index in blocked:
            return None
        text = extract_assistant_text(record)
        return text if low <= len(text) <= high else None

    for index in range(correction_index - 1, -1, -1):
        text = eligible(index)
        if text:
            return index, text

    for index in range(correction_index + 1, len(transcript)):
        text = eligible(index)
        if text:
            return index, text

    return None


def build(repo_root: Path) -> Dict:
    labels_path = repo_root / 'data' / 'paper-rollups' / 'correction_frame_labels.jsonl'
    out_path = repo_root / 'data' / 'paper-rollups' / 'correction_survival_pairs.jsonl'

    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]

    by_session: Dict[str, List[Dict]] = {}
    for row in rows:
        by_session.setdefault(row['session_id'], []).append(row)

    stats = {
        'sessions': len(by_session),
        'turns_labelled': len(rows),
        'corrections': sum(1 for r in rows if r['label'] == 1),
        'sessions_with_corrections': 0,
        'pairs_emitted': 0,
        'dropped_no_control': 0,
        'dropped_band_unsatisfiable': 0,
        'dropped_unreadable_transcript': 0,
        'resumed_sessions_in_pairs': 0,
    }

    pairs = []
    projects_dir = Path(os.path.expanduser('~/.claude/projects'))

    for session_id, session_rows in sorted(by_session.items()):
        corrections = [r for r in session_rows if r['label'] == 1]
        if not corrections:
            continue
        stats['sessions_with_corrections'] += 1

        matches = list(projects_dir.glob(f'*/{session_id}.jsonl'))
        if not matches:
            stats['dropped_unreadable_transcript'] += len(corrections)
            continue
        try:
            transcript = load_transcript(str(matches[0]))
        except Exception:
            stats['dropped_unreadable_transcript'] += len(corrections)
            continue

        flagged = {r['turn_index'] for r in corrections}

        for row in corrections:
            correction_len = len(row['text'])
            control = find_control(transcript, row['turn_index'], correction_len, flagged)
            if control is None:
                high = int(correction_len * CONTROL_MAX_RATIO)
                low = max(CONTROL_ABS_MIN_CHARS, int(correction_len * CONTROL_MIN_RATIO))
                if high < low:
                    stats['dropped_band_unsatisfiable'] += 1
                else:
                    stats['dropped_no_control'] += 1
                continue

            control_index, control_text = control
            if row['note_count'] > 1:
                stats['resumed_sessions_in_pairs'] += 1

            pairs.append({
                'pair_id': f'{session_id[:8]}-{row["turn_index"]}',
                'session_id': session_id,
                # The union of every note the session produced. A resumed session
                # legitimately has several, and resumption is the mechanism by
                # which a late correction gets written into a note at all, so
                # scoring against one note would drop the positive cases.
                'note_paths': row['note_paths'],
                'note_count': row['note_count'],
                'correction_text': row['text'],
                'correction_turn_index': row['turn_index'],
                'correction_chars': correction_len,
                'detector_why': row.get('why', ''),
                'control_text': control_text,
                'control_turn_index': control_index,
                'control_chars': len(control_text),
            })
            stats['pairs_emitted'] += 1

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + '\n')

    if stats['pairs_emitted']:
        ratios = [p['control_chars'] / p['correction_chars'] for p in pairs]
        ratios.sort()
        stats['control_to_correction_length_ratio'] = {
            'p10': round(ratios[int(len(ratios) * 0.10)], 3),
            'p50': round(ratios[int(len(ratios) * 0.50)], 3),
            'p90': round(ratios[int(len(ratios) * 0.90)], 3),
        }

    stats['output'] = str(out_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()

    if not args.build:
        parser.print_help()
        sys.exit(1)

    stats = build(Path(__file__).parent.parent.parent)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
