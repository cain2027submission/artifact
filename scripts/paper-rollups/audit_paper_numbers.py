#!/usr/bin/env python3
"""
Comprehensive audit of all numeric claims in paper/main.tex against data files.
Reports ONLY mismatches and NOT-IN-DATA findings.
"""
import json
import re
import sys
from pathlib import Path

# Load all data files
with open('data/paper-rollups/rq0_query_workload.json') as f:
    rq0 = json.load(f)
with open('data/paper-rollups/rq1_corrected_summary.json') as f:
    rq1 = json.load(f)
with open('data/paper-rollups/rq2_compression_corrected.json') as f:
    rq2 = json.load(f)
with open('data/paper-rollups/rq3_structural_loss.json') as f:
    rq3 = json.load(f)
with open('data/paper-rollups/rq5_generation_cost.json') as f:
    rq5 = json.load(f)

findings = []
checks_run = 0

def mismatch(location, draft, data, path, note=''):
    findings.append({
        'type': 'MISMATCH',
        'location': location,
        'draft': draft,
        'data': data,
        'path': path,
        'note': note
    })

def not_in_data(location, draft, note):
    findings.append({
        'type': 'NOT-IN-DATA',
        'location': location,
        'draft': draft,
        'path': 'N/A',
        'note': note
    })

# Stale checks removed 2026-10-02:
# - 550 rollups claim was stale (data now shows 300)
# - 286 pairing count was stale (data now shows 300)
# These hardcoded expectations no longer match the current paper or data.

# === REASONING TEXT SIZES ===
parent_kb_actual = rq1['thinking_persistence']['parent']['retained_chars'] / 1024
subagent_mb_actual = rq1['thinking_persistence']['subagent']['retained_chars'] / (1024**2)

checks_run += 1
if abs(683 - parent_kb_actual) / parent_kb_actual > 0.02:
    mismatch(
        'Line 88 - parent reasoning size',
        '683 KB',
        f'{parent_kb_actual:.1f} KB',
        'rq1.thinking_persistence.parent.retained_chars',
        f'Draft says 683 KB, should be {int(parent_kb_actual)} KB'
    )

checks_run += 1
if abs(9.5 - subagent_mb_actual) / subagent_mb_actual > 0.02:
    mismatch(
        'Line 89 - subagent reasoning size',
        '9.5 MB',
        f'{subagent_mb_actual:.1f} MB',
        'rq1.thinking_persistence.subagent.retained_chars',
        f'Draft says 9.5 MB, should be {subagent_mb_actual:.1f} MB'
    )

# === DERIVED CLAIMS ===
agg = rq3['aggregate']
dropped_share = agg['dropped_share'] * 100
checks_run += 1
if abs(89.5 - dropped_share) / dropped_share > 0.02:
    mismatch(
        'Line 52 - mechanical deletion share',
        '89.5%',
        f'{dropped_share:.1f}%',
        'rq3.aggregate.dropped_share',
        ''
    )

parent_reasoning = rq1['thinking_persistence']['parent']['retained_chars']
subagent_reasoning = rq1['thinking_persistence']['subagent']['retained_chars']
total_reasoning = parent_reasoning + subagent_reasoning
subagent_pct = (subagent_reasoning / total_reasoning) * 100

checks_run += 1
if abs(93 - subagent_pct) / subagent_pct > 0.02:
    mismatch(
        'Line 433 - reasoning in subagents',
        '93%',
        f'{subagent_pct:.1f}%',
        'Derived from rq1.thinking_persistence',
        ''
    )

# === CHECK SPECIFIC ABSTRACT AND TEXT CLAIMS ===
checks_run += 1
if 98.1 != rq1['thinking_persistence']['parent']['empty_pct']:
    mismatch(
        'Abstract line 40',
        98.1,
        rq1['thinking_persistence']['parent']['empty_pct'],
        'rq1.thinking_persistence.parent.empty_pct',
        ''
    )

checks_run += 1
if 17148 != rq1['thinking_persistence']['parent']['blocks']:
    mismatch(
        'Abstract line 40',
        17148,
        rq1['thinking_persistence']['parent']['blocks'],
        'rq1.thinking_persistence.parent.blocks',
        ''
    )

tool_io = rq1['signal_vs_noise_substantive']['tool_io_pooled'] * 100
reasoning_prose = rq1['signal_vs_noise_substantive']['reasoning_and_prose_pooled'] * 100

checks_run += 1
if abs(73.4 - tool_io) / tool_io > 0.01:
    mismatch(
        'Abstract line 46',
        '73.4%',
        f'{tool_io:.1f}%',
        'rq1.signal_vs_noise_substantive.tool_io_pooled',
        ''
    )

checks_run += 1
if abs(7.3 - reasoning_prose) / reasoning_prose > 0.01:
    mismatch(
        'Abstract line 46',
        '7.3%',
        f'{reasoning_prose:.1f}%',
        'rq1.signal_vs_noise_substantive.reasoning_and_prose_pooled',
        ''
    )

checks_run += 1
if abs(33.0 - rq2['compression_tokens']['median']) / rq2['compression_tokens']['median'] > 0.01:
    mismatch(
        'Abstract line 50',
        '33.0x',
        f'{rq2["compression_tokens"]["median"]:.2f}x',
        'rq2.compression_tokens.median',
        ''
    )

checks_run += 1
if abs(6.69 - rq2['inflation_serialized_over_payload']) / rq2['inflation_serialized_over_payload'] > 0.01:
    mismatch(
        'Abstract line 50',
        '6.69x',
        f'{rq2["inflation_serialized_over_payload"]:.2f}x',
        'rq2.inflation_serialized_over_payload',
        ''
    )

checks_run += 1
if abs(14.0 - rq2['sessions_rollup_expands']['pct']) / rq2['sessions_rollup_expands']['pct'] > 0.01:
    mismatch(
        'Abstract line 54',
        '14.0%',
        f'{rq2["sessions_rollup_expands"]["pct"]}%',
        'rq2.sessions_rollup_expands.pct',
        ''
    )

checks_run += 1
if 2458 != rq0['organic']['n_queries']:
    mismatch(
        'Abstract line 56',
        2458,
        rq0['organic']['n_queries'],
        'rq0.organic.n_queries',
        ''
    )

checks_run += 1
if 6 != rq0['organic']['words']['median']:
    mismatch(
        'Abstract line 56',
        6,
        rq0['organic']['words']['median'],
        'rq0.organic.words.median',
        ''
    )

checks_run += 1
if abs(31.7 - rq0['organic']['results_logged']['empty_pct_of_logged']) / rq0['organic']['results_logged']['empty_pct_of_logged'] > 0.01:
    mismatch(
        'Abstract line 56',
        '31.7%',
        f'{rq0["organic"]["results_logged"]["empty_pct_of_logged"]}%',
        'rq0.organic.results_logged.empty_pct_of_logged',
        ''
    )

checks_run += 1
if abs(0.03 - rq5['generation_cost_usd_cached']['median']) / rq5['generation_cost_usd_cached']['median'] > 0.01:
    mismatch(
        'Abstract line 60',
        '$0.03',
        f'${rq5["generation_cost_usd_cached"]["median"]:.2f}',
        'rq5.generation_cost_usd_cached.median',
        ''
    )

checks_run += 1
if abs(0.23 - rq5['break_even_retrievals_cached']['median']) / rq5['break_even_retrievals_cached']['median'] > 0.01:
    mismatch(
        'Abstract line 60',
        '0.23',
        f'{rq5["break_even_retrievals_cached"]["median"]:.2f}',
        'rq5.break_even_retrievals_cached.median',
        ''
    )

checks_run += 1
if abs(93.8 - rq5['sessions_break_even_le_1_cached']['pct']) / rq5['sessions_break_even_le_1_cached']['pct'] > 0.01:
    mismatch(
        'Abstract line 61',
        '93.8%',
        f'{rq5["sessions_break_even_le_1_cached"]["pct"]}%',
        'rq5.sessions_break_even_le_1_cached.pct',
        ''
    )

# === TEXT-CONSISTENCY CHECKS ===
MAIN_TEX = Path(__file__).resolve().parents[2] / 'paper' / 'main.tex'
tex = MAIN_TEX.read_text() if MAIN_TEX.exists() else ''

def all_occurrences(pattern: str, label: str) -> list[float]:
    return [float(m.group(1)) for m in re.finditer(pattern, tex)]

def check_all_equal(label: str, pattern: str, canonical: float, tol: float = 0.05):
    global checks_run
    checks_run += 1
    vals = all_occurrences(pattern, label)
    if not vals:
        not_in_data(label, f'(no occurrences found matching {pattern!r})',
                    'pattern may be stale if main.tex prose changed')
        return
    bad = [v for v in vals if abs(v - canonical) > tol]
    if bad:
        mismatch(
            label,
            sorted(set(vals)),
            canonical,
            label,
            f'{len(bad)}/{len(vals)} occurrences in main.tex do not match the '
            f'canonical value within {tol}'
        )

check_all_equal(
    'reasoning-empty share (parent thinking blocks)',
    r'(9[78]\.\d\d)\\%\s+of(?:\s+\d[\d{,}]*)?\s+parent-transcript reasoning',
    rq1['thinking_persistence']['parent']['empty_pct'],
)
check_all_equal(
    'reasoning-empty share (all restatements, e.g. Conclusion)',
    r'in (9[78]\.\d\d)\\%\s+of parent-transcript reasoning blocks',
    rq1['thinking_persistence']['parent']['empty_pct'],
)
check_all_equal(
    'tool I/O share',
    r'(7[0-9]\.\d\d)\\%\s+(?:mechanical )?tool I/?O',
    round(rq1['signal_vs_noise_substantive']['tool_io_pooled'] * 100, 2),
)
check_all_equal(
    'headline (serialized-byte) compression ratio',
    r'(?:not the |reporting )(\d{3})\$\\times\$',
    round(rq2['compression_serialized_bytes']['median']),
    tol=1.0,
)
check_all_equal(
    'paired sample size (RQ2 pairs)',
    r'(\d{3})\s+(?:paired samples|pairs\b|paired session/rollup samples)',
    rq2['compression_serialized_bytes']['n'],
    tol=0.5,
)

# === PRINT RESULTS ===
print('='*80)
print('AUDIT COMPLETE')
print('='*80)
print()

if len(findings) == 0:
    print('✓ ALL CLAIMS VERIFIED - NO MISMATCHES OR MISSING DATA')
    print()
else:
    print(f'Found {len(findings)} issues:')
    print()

    findings.sort(key=lambda x: (x['type'] != 'NOT-IN-DATA', x['location']))

    for f in findings:
        print(f"[{f['type']}] {f['location']}")
        print(f"  Draft:  {f['draft']}")
        print(f"  Data:   {f.get('data', 'N/A')}")
        print(f"  Path:   {f['path']}")
        if f['note']:
            print(f"  Note:   {f['note']}")
        print()

print('='*80)
print('SUMMARY')
print('='*80)
mismatches = sum(1 for f in findings if f["type"] == "MISMATCH")
not_in_data_count = sum(1 for f in findings if f["type"] == "NOT-IN-DATA")
matches = checks_run - len(findings)
print(f'Total claims checked: {checks_run}')
print(f'Mismatches: {mismatches}')
print(f'Not in data: {not_in_data_count}')
print(f'Matches: {matches}')

# Exit non-zero if any checks failed
sys.exit(1 if len(findings) > 0 else 0)
