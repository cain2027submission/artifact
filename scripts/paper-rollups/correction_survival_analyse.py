#!/usr/bin/env python3
"""
Analysis for the pre-registered correction-survival test, per-clause.

Reads the committed clause splits and the blind grades, and reports survival
per clause with its own denominator, the discordant-pair counts, and exact
two-sided McNemar on each paired comparison.

Three comparisons are reported. The first two are the pre-registered
correction-versus-control contrast, now run once per clause; the third is the
asymmetry the 2026-10-04 amendment makes the question of interest, and it is
the only one where both members of the comparison come from the same turn in
the same session graded against the same notes.

    request clause   vs control        (pairs carrying a request clause)
    corrective clause vs control       (pairs carrying a corrective clause)
    request clause   vs corrective     (the 108 pairs carrying both)

The control grade is shared between the first two tables, so those two tests
are not independent of each other and no combined p-value is computed.

Usage:
    python correction_survival_analyse.py
"""

import argparse
import json
import os
from math import comb
from pathlib import Path
from typing import Dict, List, Optional


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Exact two-sided McNemar: a binomial sign test on the discordant pairs."""
    n = b + c
    if n == 0:
        return None
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def table(pairs: List[Dict], left: str, right: str) -> Dict:
    """Paired 2x2 over pairs where both arms are graded."""
    usable = [p for p in pairs if p.get(left) in ('present', 'absent') and p.get(right) in ('present', 'absent')]
    both = sum(1 for p in usable if p[left] == 'present' and p[right] == 'present')
    left_only = sum(1 for p in usable if p[left] == 'present' and p[right] == 'absent')
    right_only = sum(1 for p in usable if p[left] == 'absent' and p[right] == 'present')
    neither = sum(1 for p in usable if p[left] == 'absent' and p[right] == 'absent')
    discordant = left_only + right_only
    return {
        'arms': f'{left} vs {right}',
        'n_pairs': len(usable),
        'left_present': both + left_only,
        'right_present': both + right_only,
        'left_survival': round((both + left_only) / len(usable), 4) if usable else None,
        'right_survival': round((both + right_only) / len(usable), 4) if usable else None,
        'both_present': both,
        'left_only': left_only,
        'right_only': right_only,
        'neither': neither,
        'discordant_pairs': discordant,
        'mcnemar_p_exact_two_sided': mcnemar_exact(left_only, right_only),
        'threshold_20_discordant_met': discordant >= 20,
    }


def second_grader(primary_path: str, second_path: str) -> Dict:
    """Agreement between the batch grader and an independent second grader.

    A single grader that has not been checked against another can be
    systematically lenient or systematically strict without anything in the
    output showing it, and a survival rate is exactly the quantity that moves.
    """
    if not os.path.exists(second_path):
        return {'available': False}
    primary = {r['fact_id']: r['grade'] for r in map(json.loads, open(primary_path))}
    second = {r['fact_id']: r['grade'] for r in map(json.loads, open(second_path))}
    overlap = [k for k in second if second[k] and primary.get(k)]
    agree = sum(1 for k in overlap if primary[k] == second[k])
    absent_to_present = sum(1 for k in overlap if primary[k] == 'absent' and second[k] == 'present')
    present_to_absent = sum(1 for k in overlap if primary[k] == 'present' and second[k] == 'absent')
    per_class = {}
    for suffix, name in (('|req', 'request'), ('|cor', 'corrective'), ('|ctl', 'control')):
        keys = [k for k in overlap if k.endswith(suffix)]
        per_class[name] = {
            'n': len(keys),
            'agree': sum(1 for k in keys if primary[k] == second[k]),
            'primary_present': sum(1 for k in keys if primary[k] == 'present'),
            'second_present': sum(1 for k in keys if second[k] == 'present'),
        }
    return {
        'available': True,
        'model': 'claude-sonnet-5-5',
        'n_overlap': len(overlap),
        'agree': agree,
        'agreement_rate': round(agree / len(overlap), 4) if overlap else None,
        'primary_absent_second_present': absent_to_present,
        'primary_present_second_absent': present_to_absent,
        'per_class': per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default='data/paper-rollups/correction_survival_clause_splits.jsonl')
    parser.add_argument('--grades', default='data/paper-rollups/correction_survival_batch_grades.jsonl')
    parser.add_argument('--probes', default='data/paper-rollups/correction_survival_grader_probes.jsonl')
    parser.add_argument('--second', default='data/paper-rollups/correction_survival_second_grader.jsonl')
    parser.add_argument('--out', default='data/paper-rollups/correction_survival_batch_summary.json')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    splits = {record['pair_id']: record for record in map(json.loads, open(args.splits))}
    grades: Dict[str, Dict] = {}
    for record in map(json.loads, open(args.grades)):
        grades[record['fact_id']] = record

    pairs = []
    for pair_id, record in splits.items():
        if record['clause_split'] not in ('both', 'request_only', 'corrective_only'):
            continue
        entry = {
            'pair_id': pair_id,
            'session_id': record['session_id'],
            'clause_split': record['clause_split'],
            'note_count': record['note_count'],
            'request': grades.get(pair_id + '|req', {}).get('grade'),
            'corrective': grades.get(pair_id + '|cor', {}).get('grade'),
            'control': grades.get(pair_id + '|ctl', {}).get('grade'),
        }
        pairs.append(entry)

    def rate(key: str, subset: List[Dict]) -> Dict:
        graded = [p for p in subset if p[key] in ('present', 'absent')]
        present = sum(1 for p in graded if p[key] == 'present')
        return {
            'present': present,
            'graded': len(graded),
            'survival': round(present / len(graded), 4) if graded else None,
        }

    with_request = [p for p in pairs if p['clause_split'] in ('both', 'request_only')]
    with_corrective = [p for p in pairs if p['clause_split'] in ('both', 'corrective_only')]
    with_both = [p for p in pairs if p['clause_split'] == 'both']

    probes = []
    if os.path.exists(args.probes):
        probes = [r for r in map(json.loads, open(args.probes)) if r.get('expected')]

    missing = [p['pair_id'] for p in pairs if p['control'] is None]

    summary = {
        'stage': 'full 293-pair batch, blind model grading, per-clause per the 2026-10-04 amendment',
        'grader': 'claude-sonnet-5 via the local gateway, one fact per call, thinking disabled, class hidden from the grader',
        'pairs_in_file': len(splits),
        'named_exclusions': {
            'fc0fb41b-85': 'harness notice, not a human correction; excluded in the split step, 1 of 294, 0.3%'
        },
        'pairs_with_a_clause': len(pairs),
        'pairs_excluded_neither_clause': sum(
            1 for r in splits.values() if r['clause_split'] == 'neither'
        ),
        'pairs_excluded_neither_clause_ids': [
            pid for pid, r in splits.items() if r['clause_split'] == 'neither'
        ],
        'clause_counts': {
            'both_clauses': len(with_both),
            'request_only': sum(1 for p in pairs if p['clause_split'] == 'request_only'),
            'corrective_only': sum(1 for p in pairs if p['clause_split'] == 'corrective_only'),
        },
        'survival': {
            'request_clause': rate('request', with_request),
            'corrective_clause': rate('corrective', with_corrective),
            'control_on_request_subset': rate('control', with_request),
            'control_on_corrective_subset': rate('control', with_corrective),
            'control_all_pairs': rate('control', pairs),
            'request_clause_on_both_subset': rate('request', with_both),
            'corrective_clause_on_both_subset': rate('corrective', with_both),
            'control_on_both_subset': rate('control', with_both),
        },
        'by_note_count': {
            'single_note_sessions': {
                'pairs': sum(1 for p in pairs if p['note_count'] == 1),
                'request_clause': rate('request', [p for p in with_request if p['note_count'] == 1]),
                'corrective_clause': rate('corrective', [p for p in with_corrective if p['note_count'] == 1]),
                'control': rate('control', [p for p in pairs if p['note_count'] == 1]),
            },
            'resumed_sessions_multi_note': {
                'pairs': sum(1 for p in pairs if p['note_count'] > 1),
                'request_clause': rate('request', [p for p in with_request if p['note_count'] > 1]),
                'corrective_clause': rate('corrective', [p for p in with_corrective if p['note_count'] > 1]),
                'control': rate('control', [p for p in pairs if p['note_count'] > 1]),
            },
        },
        'paired_tests': [
            table(with_request, 'request', 'control'),
            table(with_corrective, 'corrective', 'control'),
            table(with_both, 'request', 'corrective'),
        ],
        'ungraded_facts': {
            'controls_missing': len(missing),
            'control_pair_ids_missing': missing[:20],
            'requests_missing': sum(
                1 for p in with_request if p['request'] not in ('present', 'absent')
            ),
            'correctives_missing': sum(
                1 for p in with_corrective if p['corrective'] not in ('present', 'absent')
            ),
        },
        'second_grader': second_grader(args.grades, args.second),
        'grader_probes': {
            'n': len(probes),
            'agreed_with_expected': sum(1 for r in probes if r['grade'] == r['expected']),
            'note': 'synthetic must-PRESENT and must-ABSENT facts, graded before the batch, never pooled with it',
        },
    }

    with open(args.out, 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
