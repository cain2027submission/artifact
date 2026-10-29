#!/usr/bin/env python3
"""
Clause splitter for the pre-registered correction-survival test.

Implements the 2026-10-04 amendment in
tasks/2026-10-04 pre-registered test of whether session rollups drop corrected turns, against a paired non-correction control.md

The amendment decides that each correction is scored twice, once on its request
clause and once on its corrective clause. This script produces the splits, and
the splits are committed and pushed BEFORE any grading call is made, so a
splitter cannot be tuned against a grade it has already seen.

Procedure, per the amendment:

1. segment `correction_text` on `(?<=[.!?])\\s+` and on runs of newlines; a
   segment under 15 characters merges into the preceding segment, or into the
   following one when it is first.
2. label each sentence exactly one of corrective / request / neither, by a model
   call with the frozen prompt below. The prompt is told nothing about rollups,
   survival, controls or the hypothesis.
3. a sentence that carries both readings is labelled corrective, and the count
   of such sentences is recorded as `tie_forced`.
4. the request clause is every request sentence in original order; the
   corrective clause is every corrective sentence in original order. A turn
   yields at most two clauses however many sentences it has.
5. a missing clause is `null` and enters no denominator. A turn with neither is
   recorded `clause_split: "neither"`. A batch the gateway could not label is
   recorded `clause_split: "unlabelled"` rather than defaulted either way.

Controls are not split: an assistant turn has no request-versus-corrective
structure, so each control is carried through whole and graded once.

Usage:
    python correction_clause_split.py --out data/paper-rollups/correction_survival_clause_splits.jsonl
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

GATEWAY = os.environ.get('GATEWAY', 'http://127.0.0.1:8080')

# Same model family as the detector and the paper's grader, so the instrument
# chain does not mix model families for no stated reason.
SPLIT_MODEL = 'claude-sonnet-5'

PAIRS_PER_CALL = 6

# Named exclusion from the amendment: the flagged turn is a harness notice, not
# a human correction. 1 of 294, 0.3%.
EXCLUDED_PAIRS = {'fc0fb41b-85'}

MIN_SEGMENT_CHARS = 15

MAX_SPLIT_CALLS = 120
MAX_SPLIT_USD = 2.00
USD_PER_MTOK_IN = 3.00
USD_PER_MTOK_OUT = 15.00

# ---------------------------------------------------------------------------
# The frozen clause prompt. Committed with the splits, before any grade exists.
# It names no hypothesis, no rollup and no control.
# ---------------------------------------------------------------------------
CLAUSE_PROMPT = """You are segmenting turns a developer wrote to a coding assistant.

Each turn is given to you already split into numbered sentences. Label every
sentence with exactly one of three labels.

CORRECTIVE - the sentence asserts or implies that something the assistant did,
chose, produced or claimed is wrong, unnecessary, or should have been different.
This includes a question that implies it ("why were you searching my private
details?"), a stated preference against the assistant's choice, a report that
something the assistant built does not work, and pasted error or failure output.

REQUEST - the sentence asks for new work, or for information the assistant has
not given yet, however forcefully phrased.

NEITHER - greetings, approvals and continuations ("ok", "do it", "sounds good"),
status reports about the person's own actions, and pasted reference material
with no objection attached.

If one sentence does both jobs at once ("don't do X, do Y instead"), label it
CORRECTIVE and set "both" to true. Otherwise "both" is false.

Judge the register, not the vocabulary. The person writes informally, with typos
and missing punctuation; that carries no signal either way.

Return a JSON array, one object per sentence, and nothing else:
[{"id": <the sentence's id>, "label": "corrective" | "request" | "neither", "both": true|false}]

Sentences:
"""


def post(url: str, payload: dict, timeout: int = 180) -> dict:
    """POST to the gateway's native messages endpoint.

    The native endpoint is used rather than the OpenAI-compatible one because
    `/v1/chat/completions` turns adaptive thinking on by default: the same
    19-sentence batch cost 1,795 output tokens there against 335 here, a 5.4x
    difference on a labelling task that needs no reasoning trace. Neither
    `effort` nor `thinking.adaptive.effort` is accepted on this gateway, so the
    endpoint choice is the only lever.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Budget:
    def __init__(self, max_calls: int = MAX_SPLIT_CALLS, max_usd: float = MAX_SPLIT_USD):
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
        self.tokens_in += usage.get('input_tokens', 0)
        self.tokens_out += usage.get('output_tokens', 0)

    def summary(self) -> dict:
        return {
            'calls': self.calls,
            'tokens_in': self.tokens_in,
            'tokens_out': self.tokens_out,
            'usd_estimated': round(self.usd, 4),
            'call_ceiling': self.max_calls,
            'usd_ceiling': self.max_usd,
        }


def segment(text: str) -> List[str]:
    """Sentence segmentation, frozen by the amendment."""
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+|\n+', text) if p.strip()]
    out: List[str] = []
    for part in parts:
        if len(part) < MIN_SEGMENT_CHARS and out:
            out[-1] = out[-1] + ' ' + part
        else:
            out.append(part)
    if len(out) > 1 and len(out[0]) < MIN_SEGMENT_CHARS:
        out = [out[0] + ' ' + out[1]] + out[2:]
    return out


def label_batch(items: List[Dict], budget: Budget, model: str) -> Dict[int, Dict]:
    """Label one batch of sentences. ``items`` is ``[{sid, text, turn_id}]``."""
    budget.check()

    lines: List[str] = []
    current_turn = None
    for item in items:
        if item['turn_id'] != current_turn:
            current_turn = item['turn_id']
            lines.append(f'--- turn {current_turn} ---')
        lines.append(f'{item["sid"]}. {item["text"]}')
    prompt = CLAUSE_PROMPT + '\n'.join(lines)

    payload = {
        'model': model,
        'max_tokens': 4000,
        'thinking': {'type': 'disabled'},
        'messages': [{'role': 'user', 'content': prompt}],
    }

    try:
        data = post(f'{GATEWAY}/v1/messages', payload)
    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        print(f'  gateway error: {error}', file=sys.stderr)
        budget.calls += 1
        return {}

    budget.record(data.get('usage', {}) or {})

    if data.get('stop_reason') == 'max_tokens':
        print('  warning: reply truncated at max_tokens; batch dropped', file=sys.stderr)
        return {}

    text = ''.join(
        block.get('text', '') for block in (data.get('content') or []) if block.get('type') == 'text'
    ).strip()
    start, end = text.find('['), text.rfind(']')
    if start == -1 or end == -1:
        print('  warning: no JSON array in reply; batch dropped', file=sys.stderr)
        return {}
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        print(f'  warning: unparseable reply ({error}); batch dropped', file=sys.stderr)
        return {}

    out: Dict[int, Dict] = {}
    for entry in parsed:
        if not isinstance(entry, dict) or 'id' not in entry:
            continue
        try:
            sid = int(entry['id'])
        except (TypeError, ValueError):
            continue
        label = str(entry.get('label', '')).strip().lower()
        if label not in ('corrective', 'request', 'neither'):
            continue
        both = entry.get('both') in (True, 'true', 'True', 1, '1')
        if both:
            label = 'corrective'
        out[sid] = {'label': label, 'both': both}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', default='data/paper-rollups/correction_survival_pairs.jsonl')
    parser.add_argument('--out', default='data/paper-rollups/correction_survival_clause_splits.jsonl')
    parser.add_argument('--meta', default='data/paper-rollups/correction_survival_clause_splits_meta.json')
    parser.add_argument('--model', default=SPLIT_MODEL)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    pairs = [json.loads(line) for line in open(args.pairs)]
    kept = [p for p in pairs if p['pair_id'] not in EXCLUDED_PAIRS]
    excluded = [p['pair_id'] for p in pairs if p['pair_id'] in EXCLUDED_PAIRS]
    if args.limit:
        kept = kept[:args.limit]

    # Build the flat sentence list, with a global sentence id per sentence.
    sentences: List[Dict] = []
    per_pair: Dict[str, List[int]] = {}
    sid = 0
    for pair in kept:
        ids = []
        for text in segment(pair['correction_text']):
            sentences.append({'sid': sid, 'text': text, 'turn_id': pair['pair_id']})
            ids.append(sid)
            sid += 1
        per_pair[pair['pair_id']] = ids

    print(f'{len(kept)} pairs, {len(sentences)} sentences, excluded {excluded}')

    budget = Budget()
    labels: Dict[int, Dict] = {}

    # Batch by pair so the model always sees a whole turn at once.
    batch: List[Dict] = []
    batch_pairs = 0
    last_turn = None

    def flush(batch_items: List[Dict]) -> None:
        if not batch_items:
            return
        got = label_batch(batch_items, budget, args.model)
        missing = [item for item in batch_items if item['sid'] not in got]
        if missing and len(batch_items) > 1:
            # retry the missing half once, in smaller pieces, before giving up
            for item in missing:
                retry = label_batch([item], budget, args.model)
                got.update(retry)
        labels.update(got)
        print(f'  batch of {len(batch_items)} -> {len(got)} labelled, ${budget.usd:.3f}')

    for item in sentences:
        if item['turn_id'] != last_turn:
            if batch_pairs >= PAIRS_PER_CALL:
                flush(batch)
                batch, batch_pairs = [], 0
            last_turn = item['turn_id']
            batch_pairs += 1
        batch.append(item)
    flush(batch)

    # Assemble clauses.
    records = []
    counts = {'both': 0, 'request_only': 0, 'corrective_only': 0, 'neither': 0, 'unlabelled': 0}
    tie_forced_total = 0
    for pair in kept:
        ids = per_pair[pair['pair_id']]
        sent_texts = {i: sentences[i]['text'] for i in ids}
        if any(i not in labels for i in ids):
            split_kind = 'unlabelled'
            request_clause = corrective_clause = None
            tie_forced = 0
            labelled = [{'text': sent_texts[i], 'label': labels.get(i, {}).get('label')} for i in ids]
        else:
            req = [sent_texts[i] for i in ids if labels[i]['label'] == 'request']
            cor = [sent_texts[i] for i in ids if labels[i]['label'] == 'corrective']
            tie_forced = sum(1 for i in ids if labels[i]['both'])
            tie_forced_total += tie_forced
            request_clause = ' '.join(req) if req else None
            corrective_clause = ' '.join(cor) if cor else None
            if request_clause and corrective_clause:
                split_kind = 'both'
            elif request_clause:
                split_kind = 'request_only'
            elif corrective_clause:
                split_kind = 'corrective_only'
            else:
                split_kind = 'neither'
            labelled = [{'text': sent_texts[i], 'label': labels[i]['label'], 'both': labels[i]['both']} for i in ids]
        counts[split_kind] += 1
        records.append({
            'pair_id': pair['pair_id'],
            'session_id': pair['session_id'],
            'note_count': pair['note_count'],
            'clause_split': split_kind,
            'n_sentences': len(ids),
            'tie_forced': tie_forced,
            'request_clause': request_clause,
            'corrective_clause': corrective_clause,
            'control_text': pair['control_text'],
            'correction_text': pair['correction_text'],
            'detector_why': pair.get('detector_why', ''),
            'sentences': labelled,
        })

    with open(args.out, 'w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')

    meta = {
        'stage': 'clause split, committed before any grading call per the 2026-10-04 amendment',
        'model': args.model,
        'pairs_in': len(pairs),
        'pairs_excluded': excluded,
        'pairs_split': len(kept),
        'sentences': len(sentences),
        'sentences_labelled': len(labels),
        'split_counts': counts,
        'tie_forced_sentences': tie_forced_total,
        'multi_sentence_pairs_ge3': sum(1 for r in records if r['n_sentences'] >= 3),
        'budget': budget.summary(),
        'segmentation_rule': r"split on (?<=[.!?])\s+ and newline runs; segments under 15 chars merge into the preceding segment, or the following one when first",
        'tie_break': 'a sentence carrying both readings is labelled corrective',
        'controls': 'not split; each control is carried through whole and graded once',
    }
    with open(args.meta, 'w') as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta, indent=2)[:2000])


if __name__ == '__main__':
    main()
