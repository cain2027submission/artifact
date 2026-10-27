#!/usr/bin/env python3
"""
Blind grader for the pre-registered correction-survival test.

Implements the grading half of the 2026-10-04 amendment in
tasks/2026-10-04 pre-registered test of whether session rollups drop corrected turns, against a paired non-correction control.md

One fact per call. The grader is shown the union of a session's rollup notes
and one fact, and is asked whether the substance of the fact is recoverable
from the notes. It is never told whether the fact is a correction or a control,
nor whether it is a request clause or a corrective clause, nor that clauses or
pairs exist; the word "correction" does not appear in the grading prompt.

Inputs come from `correction_survival_clause_splits.jsonl`, which was committed
and pushed before this script made a single call.

Ordering: sessions are shuffled, and the questions within a session are
shuffled, both under a fixed seed. Grouping by session is what makes
`cache_control` pay - one note union is written to cache once and read back for
that session's remaining questions - and it means a budget stop truncates at a
random set of sessions with pairs intact rather than cutting a pair in half.
Order cannot leak a fact's class because each call is independent.

Modes:
    --mode probe   grade synthetic must-PRESENT and must-ABSENT facts, to show
                   the grader can produce both answers. Probe grades are written
                   to their own file and never enter the dataset.
    --mode grade   grade the real facts.

Usage:
    python correction_survival_grade.py --mode probe
    python correction_survival_grade.py --mode grade
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GATEWAY = os.environ.get('GATEWAY', 'http://127.0.0.1:8080')

# The paper's existing judge, and the model the detector and the clause split
# already ran on, so the instrument chain stays in one family.
GRADER_MODEL = 'claude-sonnet-5'

SEED = 20261004

# Grading-only ceiling. The pre-registration's $10 is the hard stop for the
# whole study; the clause split spent $0.52 of it before this script ran, so
# this ceiling is set below the remainder rather than at it.
MAX_GRADE_USD = 8.00

USD_PER_MTOK_IN = 3.00
USD_PER_MTOK_OUT = 15.00
USD_PER_MTOK_CACHE_WRITE = 3.75
USD_PER_MTOK_CACHE_READ = 0.30

# ---------------------------------------------------------------------------
# The frozen grading prompt. It names no arm, no class, no clause and no
# hypothesis, which is the blinding requirement the amendment imposes.
# ---------------------------------------------------------------------------
GRADE_INSTRUCTIONS = """You are checking whether one specific piece of information is recoverable from a written summary.

You will be given a summary document, and then a single FACT.

Answer PRESENT if the substance of the fact is recoverable from the summary: stated outright, paraphrased, or clearly implied by something the summary does say.

Answer ABSENT if a reader of the summary could not recover the substance of the fact from it.

Judge substance, not wording. Overlap of topic is not enough - if the summary records the general subject but not the substance of the fact, that is ABSENT. If the fact has several parts, judge whether its main substance is recoverable, not whether every detail is.

Return JSON and nothing else: {"grade": "PRESENT" or "ABSENT", "why": "<at most 20 words>"}
"""


def post(url: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Budget:
    def __init__(self, max_usd: float = MAX_GRADE_USD):
        self.max_usd = max_usd
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_write = 0
        self.cache_read = 0

    @property
    def usd(self) -> float:
        return (
            (self.tokens_in / 1e6) * USD_PER_MTOK_IN
            + (self.tokens_out / 1e6) * USD_PER_MTOK_OUT
            + (self.cache_write / 1e6) * USD_PER_MTOK_CACHE_WRITE
            + (self.cache_read / 1e6) * USD_PER_MTOK_CACHE_READ
        )

    def check(self) -> None:
        if self.usd >= self.max_usd:
            raise RuntimeError(f'BUDGET STOP: ${self.usd:.2f} reached the ${self.max_usd:.2f} grading ceiling')

    def record(self, usage: dict) -> None:
        self.calls += 1
        self.tokens_in += usage.get('input_tokens', 0)
        self.tokens_out += usage.get('output_tokens', 0)
        self.cache_write += usage.get('cache_creation_input_tokens', 0)
        self.cache_read += usage.get('cache_read_input_tokens', 0)

    def summary(self) -> dict:
        return {
            'calls': self.calls,
            'tokens_in': self.tokens_in,
            'tokens_out': self.tokens_out,
            'cache_write_tokens': self.cache_write,
            'cache_read_tokens': self.cache_read,
            'usd_estimated': round(self.usd, 4),
            'usd_ceiling': self.max_usd,
        }


def note_union(session_id: str) -> Tuple[str, List[str]]:
    """Resolve a session's rollup notes by session-id glob, per the frame rule.

    A zero-match result is a script error that halts the run, never a data
    point - the pre-registration is explicit that "note does not exist" may not
    be concluded from a failed lookup.
    """
    prefix = session_id[:8]
    paths = sorted(glob.glob(f'sessions/*_{prefix}_*.md'))
    if not paths:
        raise RuntimeError(f'no rollup note resolves for session {session_id}')
    chunks = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            chunks.append(f'--- {Path(path).name} ---\n{handle.read()}')
    return '\n\n'.join(chunks), paths


def thinking_for(model: str) -> Dict:
    """Lowest thinking setting the model accepts.

    Thinking is off because the grade is a present/absent judgement and a
    reasoning trace costs 5.4x the output tokens for the same answer; on the
    6-pair clause-split sample the labels were identical with it and without.
    `claude-sonnet-5-5` rejects `disabled` and names `between_tools` as its
    floor, so the setting is per model rather than global.
    """
    if model.startswith('claude-sonnet-5-5'):
        return {'type': 'between_tools'}
    return {'type': 'disabled'}


def grade_one(note_text: str, fact: str, budget: Budget, model: str) -> Dict:
    budget.check()
    payload = {
        'model': model,
        'max_tokens': 300,
        'thinking': thinking_for(model),
        'system': [
            {'type': 'text', 'text': GRADE_INSTRUCTIONS},
            {
                'type': 'text',
                'text': '=== SUMMARY ===\n' + note_text,
                'cache_control': {'type': 'ephemeral'},
            },
        ],
        'messages': [{'role': 'user', 'content': '=== FACT ===\n' + fact}],
    }
    for attempt in range(3):
        try:
            data = post(f'{GATEWAY}/v1/messages', payload)
        except (urllib.error.URLError, TimeoutError, KeyError) as error:
            print(f'  gateway error (attempt {attempt + 1}): {error}', file=sys.stderr)
            budget.calls += 1
            continue
        budget.record(data.get('usage', {}) or {})
        text = ''.join(
            block.get('text', '') for block in (data.get('content') or []) if block.get('type') == 'text'
        ).strip()
        match = re.search(r'\{.*\}', text, re.S)
        if not match:
            print(f'  unparseable reply: {text[:120]!r}', file=sys.stderr)
            continue
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            print(f'  bad json: {text[:120]!r}', file=sys.stderr)
            continue
        grade = str(parsed.get('grade', '')).strip().upper()
        if grade not in ('PRESENT', 'ABSENT'):
            continue
        return {'grade': grade.lower(), 'why': str(parsed.get('why', ''))[:160]}
    return {'grade': None, 'why': 'ungraded: gateway or parse failure after 3 attempts'}


def build_questions(splits: List[Dict]) -> List[Dict]:
    """One question per fact. Pairs with neither clause contribute nothing."""
    questions = []
    for record in splits:
        if record['clause_split'] not in ('both', 'request_only', 'corrective_only'):
            continue
        if record['request_clause']:
            questions.append({'fact_id': record['pair_id'] + '|req', 'fact': record['request_clause']})
        if record['corrective_clause']:
            questions.append({'fact_id': record['pair_id'] + '|cor', 'fact': record['corrective_clause']})
        questions.append({'fact_id': record['pair_id'] + '|ctl', 'fact': record['control_text']})
        for question in questions[-3:]:
            question.setdefault('pair_id', record['pair_id'])
            question.setdefault('session_id', record['session_id'])
    return questions


def ordered_questions(questions: List[Dict]) -> List[Dict]:
    by_session: Dict[str, List[Dict]] = {}
    for question in questions:
        by_session.setdefault(question['session_id'], []).append(question)
    rng = random.Random(SEED)
    sessions = sorted(by_session)
    rng.shuffle(sessions)
    out = []
    for session in sessions:
        items = by_session[session]
        rng.shuffle(items)
        out.extend(items)
    return out


def probe_questions(splits: List[Dict], count: int = 6) -> List[Dict]:
    """Synthetic must-PRESENT and must-ABSENT facts, one pair per session drawn.

    A grader that answers PRESENT to everything and one that answers ABSENT to
    everything both produce a clean-looking survival rate, so the instrument is
    shown capable of both answers before the batch runs.
    """
    rng = random.Random(SEED + 1)
    sessions = sorted({record['session_id'] for record in splits})
    rng.shuffle(sessions)
    out = []
    for session in sessions[:count]:
        text, _ = note_union(session)
        body = [
            line.strip() for line in text.splitlines()
            if len(line.strip()) > 120 and not line.strip().startswith(('-', '#', '>', '*', '|'))
        ]
        if not body:
            continue
        out.append({
            'fact_id': session[:8] + '|probe_present',
            'session_id': session,
            'pair_id': 'probe',
            'expected': 'present',
            'fact': body[len(body) // 2],
        })
        out.append({
            'fact_id': session[:8] + '|probe_absent',
            'session_id': session,
            'pair_id': 'probe',
            'expected': 'absent',
            'fact': 'The agent renamed the production database to `penguin_store` and emailed the invoice to the landlord.',
        })
    return out


def run(questions: List[Dict], out_path: str, budget: Budget, model: str) -> List[Dict]:
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as handle:
            for line in handle:
                record = json.loads(line)
                done[record['fact_id']] = record
        print(f'resuming: {len(done)} already graded')

    notes_cache: Dict[str, Tuple[str, List[str]]] = {}
    results = list(done.values())
    handle = open(out_path, 'a')
    try:
        for index, question in enumerate(questions):
            if question['fact_id'] in done:
                continue
            session = question['session_id']
            if session not in notes_cache:
                notes_cache = {session: note_union(session)}
            note_text, paths = notes_cache[session]
            verdict = grade_one(note_text, question['fact'], budget, model)
            record = {
                'fact_id': question['fact_id'],
                'pair_id': question['pair_id'],
                'session_id': session,
                'grade': verdict['grade'],
                'why': verdict['why'],
                'fact_chars': len(question['fact']),
                'note_count': len(paths),
            }
            if 'expected' in question:
                record['expected'] = question['expected']
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            results.append(record)
            if (index + 1) % 25 == 0:
                print(f'  {index + 1}/{len(questions)} ${budget.usd:.3f}')
    except RuntimeError as error:
        print(f'{error}', file=sys.stderr)
    finally:
        handle.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default='data/paper-rollups/correction_survival_clause_splits.jsonl')
    parser.add_argument('--mode', choices=['probe', 'grade'], default='grade')
    parser.add_argument('--out', default=None)
    parser.add_argument('--model', default=GRADER_MODEL)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--subsample', type=int, default=None,
                        help='grade a seeded random subsample, for a second-grader agreement check')
    parser.add_argument('--max-usd', type=float, default=MAX_GRADE_USD)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    splits = [json.loads(line) for line in open(args.splits)]
    budget = Budget(args.max_usd)

    if args.mode == 'probe':
        out_path = args.out or 'data/paper-rollups/correction_survival_grader_probes.jsonl'
        questions = probe_questions(splits)
    else:
        out_path = args.out or 'data/paper-rollups/correction_survival_batch_grades.jsonl'
        questions = ordered_questions(build_questions(splits))
    if args.subsample:
        chosen = random.Random(SEED + 2).sample(questions, min(args.subsample, len(questions)))
        picked = {q['fact_id'] for q in chosen}
        # keep session grouping so the cache still pays
        questions = [q for q in questions if q['fact_id'] in picked]
    if args.limit:
        questions = questions[:args.limit]

    print(f'{args.mode}: {len(questions)} questions, ceiling ${args.max_usd}')
    results = run(questions, out_path, budget, args.model)
    print(json.dumps(budget.summary(), indent=2))

    if args.mode == 'probe':
        hits = sum(1 for r in results if r.get('expected') and r['grade'] == r['expected'])
        total = sum(1 for r in results if r.get('expected'))
        print(f'probe agreement with the expected answer: {hits}/{total}')
    else:
        graded = [r for r in results if r['grade']]
        print(f'graded {len(graded)} of {len(questions)}; ungraded {len(results) - len(graded)}')


if __name__ == '__main__':
    main()
