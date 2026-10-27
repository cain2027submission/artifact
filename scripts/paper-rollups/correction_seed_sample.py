#!/usr/bin/env python3
"""Draw the hand-label sample for the model-labelled correction detector.

The sample is drawn by a fixed seed and committed **unlabelled** before any
label is written and before the detector prompt exists, so that the draw
provably predates both. That ordering is the same discipline the
pre-registration applies to the candidate list: the selection has to predate
the selection's outcome or it cannot be distinguished from fitting.

The draw is split in two halves by the shuffled order:

- the first ``DEV_N`` turns are the **development** half. the detector prompt
  may be read against these and revised until it agrees with the hand labels.
- the remaining turns are the **held-out** half. they are scored exactly once,
  and the prompt is never revised after seeing that score.

The held-out half is what replaces session ``6d7b783e`` as the positive
control. ``6d7b783e``'s correction is quoted verbatim in the pre-registration,
so any detector written afterwards has read the answer; it is kept only as a
named sanity check and its result carries no weight.

Usage:
    python correction_seed_sample.py --draw
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from human_turns import frame_sessions, human_turns, load_transcript

# Fixed and recorded so the draw is reproducible and cannot be re-rolled until
# it gives a convenient sample.
#
# The sample was drawn twice and the seed is deliberately unchanged between the
# two. The first draw (committed unlabelled at f28cb76e) was read, and 13 of its
# 60 turns turned out not to be human-authored at all: six harness continuation
# summaries, four benchmark probes from this project's own answer_speed harness,
# two `!` shell passthroughs and a task notification. That is a defect in the
# population, not in the draw, so `human_turns.py` was fixed and the draw rerun
# over the corrected 1,268-turn population under the same seed. The first draw
# stays in git history, which is the only thing that makes the redraw checkable
# rather than a quiet re-roll: no label had been written when it happened.
SEED = 20261004
SAMPLE_N = 60
DEV_N = 40

# Each turn is capped head-and-tail for labelling. The median human turn is 162
# characters but p90 is 20,846: the tail is pasted logs and documents, not the
# human's prose, and a correction does not live 20,000 characters into a paste.
# Capping is applied to the stored sample too, so a hand label and a model label
# are made from byte-identical text.
HEAD_CHARS = 1500
TAIL_CHARS = 500


def cap_turn(text: str) -> str:
    """Cap a turn head-and-tail, marking the elision."""
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    cut = len(text) - HEAD_CHARS - TAIL_CHARS
    return (
        text[:HEAD_CHARS]
        + f'\n\n[... {cut} characters elided ...]\n\n'
        + text[-TAIL_CHARS:]
    )


def collect_turns(sessions_dir: str, projects_dir: str) -> list:
    """Every human-authored turn in the frame, as label-ready records."""
    turns = []
    for entry in frame_sessions(sessions_dir, projects_dir):
        try:
            transcript = load_transcript(entry['transcript_path'])
        except Exception as error:
            print(f"warning: unreadable transcript {entry['session_id']}: {error}",
                  file=sys.stderr)
            continue
        for index, text in human_turns(transcript):
            turns.append({
                'session_id': entry['session_id'],
                'turn_index': index,
                'note_count': entry['note_count'],
                'text': cap_turn(text),
                'original_chars': len(text),
            })
    return turns


def draw(sessions_dir: str, projects_dir: str, output_path: str) -> dict:
    """Draw the sample and write it unlabelled."""
    turns = collect_turns(sessions_dir, projects_dir)

    # Sort before shuffling so the population order does not depend on the
    # filesystem's glob order, which is not stable across machines.
    turns.sort(key=lambda t: (t['session_id'], t['turn_index']))

    rng = random.Random(SEED)
    picked = rng.sample(range(len(turns)), min(SAMPLE_N, len(turns)))
    rng.shuffle(picked)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        for position, source_index in enumerate(picked):
            record = dict(turns[source_index])
            record['sample_position'] = position
            record['split'] = 'dev' if position < DEV_N else 'heldout'
            record['label'] = None  # hand-filled in the labelled copy
            handle.write(json.dumps(record) + '\n')

    return {
        'population_turns': len(turns),
        'sample_n': len(picked),
        'dev_n': min(DEV_N, len(picked)),
        'heldout_n': max(0, len(picked) - DEV_N),
        'seed': SEED,
        'head_chars': HEAD_CHARS,
        'tail_chars': TAIL_CHARS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draw', action='store_true', help='draw the sample')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    sessions_dir = repo_root / 'sessions'
    projects_dir = Path(os.path.expanduser('~/.claude/projects'))
    output = repo_root / 'data' / 'paper-rollups' / 'correction_seed_sample.jsonl'

    if not args.draw:
        parser.print_help()
        sys.exit(1)

    stats = draw(str(sessions_dir), str(projects_dir), str(output))
    print(json.dumps(stats, indent=2))
    print(f'\nwrote {output}')


if __name__ == '__main__':
    main()
