#!/usr/bin/env python3
"""Model-labelled correction detector for the pre-registered rollup test.

This replaces the frozen lexical marker list of stage 1, which failed because
the register it had to detect has no fixed vocabulary: this human corrects by
proposing the better alternative ("but ideally", "it feels", "could we
instead"), and a detector built from negation words measures a vocabulary
nobody here uses. A marker list cannot be repaired by adding markers.

What is frozen here instead is the **prompt**, committed before it was ever run
against the held-out half of the hand-labelled sample. The prompt may be revised
against the 40 development turns; it is never revised after the 17 held-out
turns have been scored. ``--validate`` enforces the split by refusing to score
the held-out half unless asked for explicitly.

The gate is rewritten as well, because stage 1's gate could not have passed.
Its check was ``if session_id == '6d7b783e'`` against a full uuid, so
``positive_control_hit`` was unreachable and ``FAIL`` was a constant: a perfect
detector would have been halted by the same line. So this gate is run against
two deliberately broken detectors as well as the real one -- one that labels
everything a correction, which must make the gate pass, and one that labels
nothing, which must make it fail. A gate that cannot be made to pass is
indistinguishable from a detector that cannot work.

Session ``6d7b783e`` is no longer the positive control. The pre-registration
quotes its correction verbatim and explains which register makes it one, so any
detector written afterwards has read the answer; clearing it is fitting, not
validating. It is kept as a named sanity check whose result carries no weight.

Usage:
    python correction_detector.py --validate            # dev half only
    python correction_detector.py --validate --heldout  # scores the sealed half
    python correction_detector.py --label-frame         # label all human turns
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from human_turns import frame_sessions, human_turns, load_transcript

GATEWAY = os.environ.get('GATEWAY', 'http://127.0.0.1:8080')

# The paper's existing grader is claude-sonnet-5 and the detector is the
# instrument the whole result rests on, so it uses the same model rather than a
# cheaper one. At a 2,000-character cap over 1,268 turns the whole frame is
# roughly 455k input tokens, about $1.40, which is why the cheaper option buys
# nothing worth the second instrument family.
DETECTOR_MODEL = 'claude-sonnet-5'

TURNS_PER_CALL = 10

# Separate from and not drawn against the pre-registration's $10 / 200-call
# grading ceiling. Labelling the instrument is not grading.
#
# Raised from $3.00 to $5.00 *before* the frame run, not during it. The $3 was a
# same-turn estimate of mine rather than anything in the greenlit design, and
# the measured dev rate projects the full 1,268-turn frame at roughly $2.40 -
# close enough that a mid-run budget stop was likely. That matters more than the
# dollar, because `label_frame` walks sessions in sorted id order: a stop part
# way through truncates the candidate list by session id rather than at random,
# which is a selection effect on the thing being measured. Raising a ceiling
# after seeing a result is the move this whole design exists to prevent, so it
# is done here with the arithmetic stated and the grading budget untouched.
MAX_LABEL_CALLS = 200
MAX_LABEL_USD = 5.00
USD_PER_MTOK_IN = 3.00
USD_PER_MTOK_OUT = 15.00

# ---------------------------------------------------------------------------
# The frozen prompt. Committed before any held-out turn was scored.
#
# It describes the register rather than listing phrases, which is the whole
# point of the swap: a phrase list is what stage 1 already tried. It is also
# told nothing about rollups, survival, or the hypothesis, so a label cannot be
# shaded by what the study wants the answer to be.
# ---------------------------------------------------------------------------
CORRECTION_PROMPT = """You are labelling turns from a developer's chat with a coding assistant.

For each turn, decide whether the person is CORRECTING the assistant.

A turn is a correction (1) if it asserts or implies that something the assistant
did, chose, produced or claimed is wrong, unnecessary, or should be different.
This register takes many forms and all of them count:

- a direct negation: "no, that's wrong", "that's not right", "revert that"
- a question that implies something should not be there: "do we need all this in
  the abstract?", "but does it need to be in the intro?"
- a stated preference against the assistant's choice: "i lean to cut it",
  "i'd rather we didn't"
- a proposed alternative in place of the assistant's approach: "but ideally we
  don't load all of it", "instead of changing the timestamps, could we...",
  "lets not rely on it for now, lets focus on something simpler"
- an objection to style, tone or wording: "that reads like a blog post, don't
  use that kind of language"
- a report that something the assistant built or delivered does not work
- a pasted error or failure output from a command the assistant supplied

A turn is NOT a correction (0) if it is:

- a new task or request, however forcefully phrased
- a question seeking information the assistant has not given yet
- an approval or a continuation: "do it", "ok", "sounds good", "continue"
- a status report about the person's own actions: "i restarted i think"
- pasted reference material with no objection attached
- a complaint about a person other than the assistant

Judge the register, not the vocabulary. A polite question can be a correction and
a blunt instruction can fail to be one. The person writes informally, with typos
and missing punctuation; that carries no signal either way.

Return a JSON array, one object per turn, and nothing else:
[{"id": <the turn's id>, "label": 0 or 1, "why": "<at most 12 words>"}]

Turns:
"""


def post(url: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Budget:
    """Hard call and spend ceiling, checked before every call."""

    def __init__(self, max_calls: int = MAX_LABEL_CALLS, max_usd: float = MAX_LABEL_USD):
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def usd(self) -> float:
        return (self.tokens_in / 1e6) * USD_PER_MTOK_IN + (self.tokens_out / 1e6) * USD_PER_MTOK_OUT

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise RuntimeError(f'BUDGET STOP: {self.calls} calls reached the {self.max_calls} ceiling')
        if self.usd >= self.max_usd:
            raise RuntimeError(f'BUDGET STOP: ${self.usd:.2f} reached the ${self.max_usd:.2f} ceiling')

    def record(self, usage: dict) -> None:
        self.calls += 1
        self.tokens_in += usage.get('prompt_tokens', 0)
        self.tokens_out += usage.get('completion_tokens', 0)

    def summary(self) -> dict:
        return {
            'calls': self.calls,
            'tokens_in': self.tokens_in,
            'tokens_out': self.tokens_out,
            'usd_estimated': round(self.usd, 4),
            'call_ceiling': self.max_calls,
            'usd_ceiling': self.max_usd,
        }


def label_batch(turns: List[Dict], budget: Budget, model: str = DETECTOR_MODEL) -> Dict[int, Dict]:
    """Label one batch of turns. Returns ``{id: {label, why}}``.

    A gateway failure returns no labels for the batch rather than a default,
    because a failed call is not a finding about a turn. The caller records the
    batch as unlabelled and the turns are excluded from the denominator with
    that reason named, which is the distinction stage 1's silent skips lost.
    """
    budget.check()

    body = []
    for turn in turns:
        body.append(f'--- id: {turn["id"]}\n{turn["text"]}\n')
    prompt = CORRECTION_PROMPT + '\n'.join(body)

    payload = {
        'model': model,
        'max_tokens': 2000,
        'messages': [{'role': 'user', 'content': prompt}],
    }

    try:
        data = post(f'{GATEWAY}/v1/chat/completions', payload)
    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        print(f'  gateway error: {error}', file=sys.stderr)
        budget.calls += 1
        return {}

    budget.record(data.get('usage', {}) or {})

    choice = (data.get('choices') or [{}])[0]
    content = (choice.get('message') or {}).get('content') or ''
    if choice.get('finish_reason') == 'length':
        print('  warning: reply truncated at max_tokens; batch dropped', file=sys.stderr)
        return {}

    text = content.strip()
    start, end = text.find('['), text.rfind(']')
    if start == -1 or end == -1:
        print('  warning: no JSON array in reply; batch dropped', file=sys.stderr)
        return {}

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        print(f'  warning: unparseable reply ({error}); batch dropped', file=sys.stderr)
        return {}

    out = {}
    for item in parsed:
        if not isinstance(item, dict) or 'id' not in item:
            continue
        try:
            turn_id = int(item['id'])
        except (TypeError, ValueError):
            continue
        out[turn_id] = {
            'label': 1 if item.get('label') in (1, '1', True) else 0,
            'why': str(item.get('why', ''))[:120],
        }
    return out


def label_turns(turns: List[Dict], budget: Budget, model: str = DETECTOR_MODEL) -> Dict[int, Dict]:
    """Label a list of ``{id, text}`` turns in batches."""
    labels: Dict[int, Dict] = {}
    for start in range(0, len(turns), TURNS_PER_CALL):
        batch = turns[start:start + TURNS_PER_CALL]
        got = label_batch(batch, budget, model)
        missing = [t['id'] for t in batch if t['id'] not in got]
        if missing:
            print(f'  batch {start // TURNS_PER_CALL}: {len(missing)} turn(s) unlabelled', file=sys.stderr)
        labels.update(got)
        time.sleep(0.2)
    return labels


# ---------------------------------------------------------------------------
# Deliberately broken detectors, used to prove the gate can move.
# ---------------------------------------------------------------------------

def always_correction(turns: List[Dict], budget: Budget, model: str = '') -> Dict[int, Dict]:
    return {t['id']: {'label': 1, 'why': 'stub: labels everything'} for t in turns}


def never_correction(turns: List[Dict], budget: Budget, model: str = '') -> Dict[int, Dict]:
    return {t['id']: {'label': 0, 'why': 'stub: labels nothing'} for t in turns}


def oracle(truth: Dict[int, int]) -> Dict[int, Dict]:
    """A stub that reproduces the hand labels exactly.

    This is the control that proves the gate is *able* to emit PASS. The first
    version of this self-test used `always_correction` for that job and the
    self-test failed, correctly: a detector that flags every turn has perfect
    recall and a false-positive rate of 1.0, so a gate with a false-positive arm
    is supposed to reject it. Needing an oracle rather than a flag-everything
    stub is the point - PASS has to be reachable by being right, not by being
    indiscriminate.
    """
    return {turn_id: {'label': actual, 'why': 'stub: copies hand labels'}
            for turn_id, actual in truth.items()}


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

# Stated before the gate was first run. Five positives in 17 held-out turns is
# thin, so the thresholds are set to what that sample can actually support: the
# detector must recover a majority of the positives and must not flood the
# negatives. It cannot distinguish a good detector from an excellent one.
GATE_MIN_RECALL = 0.60
GATE_MAX_FPR = 0.34


def score(labels: Dict[int, Dict], truth: Dict[int, int]) -> Dict:
    """Confusion counts and rates of a label set against the hand labels."""
    tp = fp = tn = fn = unlabelled = 0
    for turn_id, actual in truth.items():
        if turn_id not in labels:
            unlabelled += 1
            continue
        predicted = labels[turn_id]['label']
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            tn += 1

    positives = tp + fn
    negatives = fp + tn
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'unlabelled': unlabelled,
        'positives': positives,
        'negatives': negatives,
        'recall': round(tp / positives, 3) if positives else None,
        'false_positive_rate': round(fp / negatives, 3) if negatives else None,
        'precision': round(tp / (tp + fp), 3) if (tp + fp) else None,
    }


def gate_verdict(scored: Dict) -> Dict:
    """Apply the pre-stated thresholds. Returns pass/fail with the reason."""
    recall = scored['recall']
    fpr = scored['false_positive_rate']
    reasons = []

    if recall is None:
        reasons.append('no positives to measure recall against')
    elif recall < GATE_MIN_RECALL:
        reasons.append(f'recall {recall} below {GATE_MIN_RECALL}')

    if fpr is None:
        reasons.append('no negatives to measure false positives against')
    elif fpr > GATE_MAX_FPR:
        reasons.append(f'false-positive rate {fpr} above {GATE_MAX_FPR}')

    return {
        'status': 'PASS' if not reasons else 'FAIL',
        'reasons': reasons,
        'thresholds': {'min_recall': GATE_MIN_RECALL, 'max_false_positive_rate': GATE_MAX_FPR},
    }


def load_sample(repo_root: Path) -> List[Dict]:
    path = repo_root / 'data' / 'paper-rollups' / 'correction_seed_sample.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_labels(repo_root: Path) -> Dict:
    path = repo_root / 'data' / 'paper-rollups' / 'correction_seed_labels.json'
    return json.loads(path.read_text())


def sanity_turn_6d7b783e() -> Optional[Dict]:
    """The stage 1 positive control, kept as a fitted sanity check only."""
    projects = Path(os.path.expanduser('~/.claude/projects'))
    matches = list(projects.glob('*/6d7b783e-*.jsonl'))
    if not matches:
        return None
    transcript = load_transcript(str(matches[0]))
    for index, text in human_turns(transcript):
        if 'ideally we dont load all infrastructure data' in text:
            return {'id': 900001, 'text': text, 'turn_index': index}
    return None


def disguised_negatives() -> List[Dict]:
    """Non-corrections that contain the stage 1 marker words.

    The old gate's "negative control" asserted that a disguised *correction*
    must be MISSED and scored that a pass, which rewards exactly the failure
    that halted stage 1. The useful negative control is the opposite: ordinary
    prose carrying the marker vocabulary must not be flagged. All three of these
    contain a frozen stage 1 marker and none is a correction.
    """
    return [
        {'id': 900101, 'text': 'the build was already wrong before i started, no idea who broke it'},
        {'id': 900102, 'text': "i said i'd check the dashboard today so let me go do that now"},
        {'id': 900103, 'text': 'can you add an undo button to the settings screen'},
    ]


def validate(repo_root: Path, score_heldout: bool, model: str = DETECTOR_MODEL) -> Dict:
    sample = load_sample(repo_root)
    label_file = load_labels(repo_root)
    hand = label_file['labels']
    excluded = set(label_file['excluded_as_non_human'].keys())

    def turns_for(split: str) -> List[Dict]:
        out = []
        for record in sample:
            position = str(record['sample_position'])
            if position in excluded or position not in hand:
                continue
            if hand[position]['split'] != split:
                continue
            out.append({'id': record['sample_position'], 'text': record['text']})
        return out

    def truth_for(split: str) -> Dict[int, int]:
        return {
            int(position): entry['label']
            for position, entry in hand.items()
            if entry['split'] == split and position not in excluded
        }

    result: Dict = {
        'model': model,
        'prompt_sha': __import__('hashlib').sha256(CORRECTION_PROMPT.encode()).hexdigest()[:16],
        'gate_thresholds': {'min_recall': GATE_MIN_RECALL, 'max_false_positive_rate': GATE_MAX_FPR},
    }

    budget = Budget()

    # --- gate self-test: the gate must be capable of both outcomes -----------
    dev_turns = turns_for('dev')
    dev_truth = truth_for('dev')

    oracle_scored = score(oracle(dev_truth), dev_truth)
    all_scored = score(always_correction(dev_turns, budget), dev_truth)
    none_scored = score(never_correction(dev_turns, budget), dev_truth)
    oracle_verdict = gate_verdict(oracle_scored)
    all_verdict = gate_verdict(all_scored)
    none_verdict = gate_verdict(none_scored)

    self_test_ok = (
        oracle_verdict['status'] == 'PASS'
        and all_verdict['status'] == 'FAIL'
        and none_verdict['status'] == 'FAIL'
    )

    result['gate_self_test'] = {
        'purpose': 'stage 1 gate compared a full uuid to an 8-char prefix and so '
                   'reported FAIL unconditionally; a gate is not trusted until it '
                   'is shown to produce both outcomes, and until each of its arms '
                   'is shown to bite on its own',
        'oracle_must_PASS': {'scored': oracle_scored, 'verdict': oracle_verdict,
                             'tests': 'that PASS is reachable at all'},
        'labels_nothing_must_FAIL': {'scored': none_scored, 'verdict': none_verdict,
                                     'tests': 'that the recall arm bites'},
        'labels_everything_must_FAIL': {'scored': all_scored, 'verdict': all_verdict,
                                        'tests': 'that the false-positive arm bites'},
        'status': 'PASS' if self_test_ok else 'FAIL',
    }

    # --- the real detector on the development half --------------------------
    dev_labels = label_turns(dev_turns, budget, model)
    dev_scored = score(dev_labels, dev_truth)
    result['development_half'] = {
        'n': len(dev_truth),
        'scored': dev_scored,
        'verdict': gate_verdict(dev_scored),
        'note': 'the prompt may be revised against this half; it is not the gate',
    }

    # --- disguised negatives -------------------------------------------------
    dn = disguised_negatives()
    dn_labels = label_turns(dn, budget, model)
    dn_results = []
    for turn in dn:
        got = dn_labels.get(turn['id'])
        dn_results.append({
            'text': turn['text'],
            'flagged': None if got is None else bool(got['label']),
            'status': 'PASS' if (got is not None and got['label'] == 0) else 'FAIL',
        })
    result['disguised_negatives'] = {
        'purpose': 'ordinary prose containing a stage 1 marker word must not be flagged; '
                   'stage 1 instead tested that a disguised correction was missed and '
                   'scored that a pass, which rewards the failure that halted it',
        'cases': dn_results,
        'status': 'PASS' if all(c['status'] == 'PASS' for c in dn_results) else 'FAIL',
    }

    # --- fitted sanity check on 6d7b783e ------------------------------------
    sanity = sanity_turn_6d7b783e()
    if sanity is None:
        result['fitted_sanity_6d7b783e'] = {'status': 'TRANSCRIPT NOT FOUND'}
    else:
        sanity_labels = label_turns([{'id': sanity['id'], 'text': sanity['text']}], budget, model)
        got = sanity_labels.get(sanity['id'])
        result['fitted_sanity_6d7b783e'] = {
            'turn_index': sanity['turn_index'],
            'detected': None if got is None else bool(got['label']),
            'weight': 'none; this turn is quoted verbatim in the pre-registration, so any '
                      'detector written afterwards has read the answer. reported, not counted.',
        }

    # --- the held-out half, scored once -------------------------------------
    if score_heldout:
        heldout_turns = turns_for('heldout')
        heldout_truth = truth_for('heldout')
        heldout_labels = label_turns(heldout_turns, budget, model)
        heldout_scored = score(heldout_labels, heldout_truth)
        heldout_verdict = gate_verdict(heldout_scored)
        result['heldout_half'] = {
            'n': len(heldout_truth),
            'scored': heldout_scored,
            'verdict': heldout_verdict,
            'note': 'scored once. the prompt is not revised after this number exists.',
        }
        result['GATE'] = {
            'status': 'PASS' if (
                heldout_verdict['status'] == 'PASS'
                and result['gate_self_test']['status'] == 'PASS'
                and result['disguised_negatives']['status'] == 'PASS'
            ) else 'FAIL',
            'components': {
                'heldout': heldout_verdict['status'],
                'gate_self_test': result['gate_self_test']['status'],
                'disguised_negatives': result['disguised_negatives']['status'],
            },
        }
    else:
        result['heldout_half'] = {'status': 'NOT SCORED; pass --heldout to seal and score it'}
        result['GATE'] = {'status': 'NOT RUN'}

    result['budget'] = budget.summary()
    return result


def label_frame(repo_root: Path, model: str = DETECTOR_MODEL, limit: Optional[int] = None) -> Dict:
    """Label every human-authored turn in the frame, caching as it goes."""
    sessions_dir = repo_root / 'sessions'
    projects_dir = Path(os.path.expanduser('~/.claude/projects'))
    out_path = repo_root / 'data' / 'paper-rollups' / 'correction_frame_labels.jsonl'

    from correction_seed_sample import cap_turn

    records = []
    for entry in frame_sessions(str(sessions_dir), str(projects_dir)):
        try:
            transcript = load_transcript(entry['transcript_path'])
        except Exception:
            continue
        for index, text in human_turns(transcript):
            records.append({
                'session_id': entry['session_id'],
                'turn_index': index,
                'note_count': entry['note_count'],
                'note_paths': entry['note_paths'],
                'text': cap_turn(text),
                'original_chars': len(text),
            })

    if limit:
        records = records[:limit]

    budget = Budget()
    os.makedirs(out_path.parent, exist_ok=True)
    written = 0
    unlabelled = 0

    with open(out_path, 'w', encoding='utf-8') as handle:
        for start in range(0, len(records), TURNS_PER_CALL):
            chunk = records[start:start + TURNS_PER_CALL]
            try:
                budget.check()
            except RuntimeError as error:
                print(str(error), file=sys.stderr)
                unlabelled += len(records) - start
                break

            got = label_batch(
                [{'id': i, 'text': r['text']} for i, r in enumerate(chunk)], budget, model
            )
            for i, record in enumerate(chunk):
                label = got.get(i)
                if label is None:
                    unlabelled += 1
                    continue
                out = dict(record)
                out['label'] = label['label']
                out['why'] = label['why']
                handle.write(json.dumps(out) + '\n')
                written += 1

            if (start // TURNS_PER_CALL) % 20 == 0:
                print(f'  {start + len(chunk)}/{len(records)} turns, '
                      f'{budget.calls} calls, ${budget.usd:.2f}', flush=True)

    return {
        'turns_total': len(records),
        'turns_labelled': written,
        'turns_unlabelled': unlabelled,
        'corrections': None,
        'budget': budget.summary(),
        'output': str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--heldout', action='store_true',
                        help='score the sealed half; do not revise the prompt afterwards')
    parser.add_argument('--label-frame', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model', default=DETECTOR_MODEL)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    out_dir = repo_root / 'data' / 'paper-rollups'
    os.makedirs(out_dir, exist_ok=True)

    if args.validate:
        result = validate(repo_root, args.heldout, args.model)
        print(json.dumps(result, indent=2))
        name = 'correction_detector_validation.json' if args.heldout else 'correction_detector_validation_dev.json'
        (out_dir / name).write_text(json.dumps(result, indent=2))
        print(f'\nwrote {out_dir / name}')
        sys.exit(0 if result['GATE']['status'] in ('PASS', 'NOT RUN') else 1)

    if args.label_frame:
        stats = label_frame(repo_root, args.model, args.limit)
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
