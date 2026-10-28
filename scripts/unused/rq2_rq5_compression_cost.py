#!/usr/bin/env python3
"""RQ2 & RQ5 analysis: compression ratios and generation costs for session rollups.

Computes compression statistics (transcript bytes vs rollup bytes, token-based compression)
and cost analysis for a research paper. Precision matters.

Usage:
    python scripts/paper-rollups/rq2_rq5_compression_cost.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

SESSIONS_DIR = SESSIONS_DIR
PROJECTS_DIR = Path.home() / ".claude/projects"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "paper-rollups"


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter as a dict."""
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}

    fm = {}
    lines = match.group(1).split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if ':' not in line:
            i += 1
            continue

        key, _, val = line.partition(':')
        key = key.strip()
        val = val.strip()

        # Handle nested dicts (tokens, compactions, etc)
        if not val or val == '':
            # This is a nested structure
            nested = {}
            i += 1
            while i < len(lines) and lines[i].startswith('  '):
                nested_line = lines[i].strip()
                if ':' in nested_line:
                    nk, _, nv = nested_line.partition(':')
                    nk = nk.strip()
                    nv = nv.strip()
                    # Handle deeply nested (tokens_by_model)
                    if not nv:
                        sub_nested = {}
                        i += 1
                        while i < len(lines) and lines[i].startswith('    '):
                            sub_line = lines[i].strip()
                            if ':' in sub_line:
                                sk, _, sv = sub_line.partition(':')
                                try:
                                    sub_nested[sk.strip()] = float(sv.strip()) if '.' in sv else int(sv.strip())
                                except ValueError:
                                    sub_nested[sk.strip()] = sv.strip()
                            i += 1
                        nested[nk] = sub_nested
                        continue
                    try:
                        nested[nk] = float(nv) if '.' in nv else int(nv)
                    except ValueError:
                        nested[nk] = nv
                i += 1
            fm[key] = nested
            continue

        # Handle lists
        if val.startswith('-'):
            list_items = [val.lstrip('- ')]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('-'):
                list_items.append(lines[i].strip().lstrip('- '))
                i += 1
            fm[key] = list_items
            continue

        # Handle scalars
        try:
            fm[key] = float(val) if '.' in val else int(val)
        except ValueError:
            fm[key] = val
        i += 1

    return fm


def get_rollup_byte_size(filepath: Path) -> int:
    """Get the byte size of a rollup note (entire file including frontmatter)."""
    return filepath.stat().st_size


def get_transcript_bytes_from_file(session_id: str) -> int | None:
    """Compute transcript byte size by finding and reading the jsonl file."""
    # Search for transcript in all project directories
    pattern = f"**/{session_id}.jsonl"
    matches = list(PROJECTS_DIR.glob(pattern))

    if not matches:
        return None

    # Take the first match (there should only be one)
    transcript_path = matches[0]
    return transcript_path.stat().st_size


def compute_token_compression(session_id: str, rollup_bytes: int) -> float | None:
    """Compute token-based compression: transcript tokens / rollup tokens (estimated)."""
    pattern = f"**/{session_id}.jsonl"
    matches = list(PROJECTS_DIR.glob(pattern))

    if not matches:
        return None

    transcript_path = matches[0]

    # Count unique request tokens from transcript
    total_input = 0
    total_output = 0
    seen_ids = set()

    try:
        with open(transcript_path, 'r', errors='replace') as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if d.get("type") != "assistant":
                    continue

                msg = d.get("message", {})
                msg_id = msg.get("id")

                # Dedupe by message.id to avoid overcounting
                if msg_id and msg_id in seen_ids:
                    continue
                if msg_id:
                    seen_ids.add(msg_id)

                usage = msg.get("usage", {})
                total_input += usage.get("input_tokens", 0) or 0
                total_output += usage.get("output_tokens", 0) or 0
    except Exception:
        return None

    transcript_tokens = total_input + total_output
    if transcript_tokens == 0:
        return None

    # Estimate rollup tokens at ~4 chars/token
    rollup_tokens = rollup_bytes / 4.0

    return transcript_tokens / rollup_tokens if rollup_tokens > 0 else None


def extract_intent_section(content: str) -> str:
    """Extract the ## Intent section for session type classification."""
    match = re.search(r'^## Intent\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""


def classify_session_type(intent: str) -> str:
    """Classify session into coarse types using keyword heuristics."""
    intent_lower = intent.lower()

    # Order matters: more specific first
    if any(kw in intent_lower for kw in ['debug', 'fix', 'error', 'bug', 'troubleshoot', 'diagnose']):
        return 'debugging'
    if any(kw in intent_lower for kw in ['refactor', 'reorgani', 'restructur', 'simplif', 'clean up']):
        return 'refactoring'
    if any(kw in intent_lower for kw in ['setup', 'install', 'config', 'environment', 'deploy']):
        return 'environment/setup'
    if any(kw in intent_lower for kw in ['analyz', 'analysis', 'measure', 'benchmark', 'metric', 'statistic', 'investigat']):
        return 'analysis/measurement'
    if any(kw in intent_lower for kw in ['write', 'document', 'doc ', 'README', 'comment', 'note']):
        return 'writing/docs'
    if any(kw in intent_lower for kw in ['implement', 'add', 'create', 'build', 'feature', 'develop']):
        return 'feature/implementation'

    return 'other'


def main():
    print("Loading rollup notes...")

    rows = []

    for rollup_path in sorted(SESSIONS_DIR.glob("*.md")):
        content = rollup_path.read_text(errors='replace')
        fm = parse_frontmatter(content)

        # Extract key fields
        session_id = fm.get('session_id') or fm.get('recap_session')
        if not session_id:
            continue

        # Skip unrecoverable stubs
        if fm.get('rollup_status') == 'unrecoverable':
            continue

        # Get rollup byte size
        rollup_bytes = get_rollup_byte_size(rollup_path)

        # Get transcript bytes from frontmatter
        fm_transcript_bytes = fm.get('transcript_bytes')

        # Compute transcript bytes ourselves
        computed_transcript_bytes = get_transcript_bytes_from_file(session_id)

        # Use computed if available, otherwise frontmatter
        transcript_bytes = computed_transcript_bytes if computed_transcript_bytes else fm_transcript_bytes

        # Compute compression ratio
        compression_ratio = None
        if transcript_bytes and rollup_bytes > 0:
            compression_ratio = transcript_bytes / rollup_bytes

        # Compute token-based compression
        token_compression = compute_token_compression(session_id, rollup_bytes)

        # Extract intent section for classification
        intent_text = extract_intent_section(content)
        session_type = classify_session_type(intent_text)

        # Extract other metadata
        agent = fm.get('agent', '')
        machine = fm.get('machine', '')
        turns = fm.get('turns')
        subagents = fm.get('subagents', 0)
        compactions = fm.get('compactions', {}).get('count', 0) if isinstance(fm.get('compactions'), dict) else 0
        duration_active = fm.get('duration_active')
        cost_usd = fm.get('cost_usd')
        model = fm.get('model')

        # Extract tokens
        tokens = fm.get('tokens', {})
        if isinstance(tokens, dict):
            tokens_input = tokens.get('input', 0)
            tokens_output = tokens.get('output', 0)
            tokens_cache_read = tokens.get('cache_read', 0)
            tokens_cache_creation = tokens.get('cache_creation', 0)
            tokens_total = tokens_input + tokens_output + tokens_cache_read + tokens_cache_creation
        else:
            tokens_input = tokens_output = tokens_cache_read = tokens_cache_creation = tokens_total = 0

        cost_recomputed = fm.get('cost_recomputed', False)

        rows.append({
            'filename': rollup_path.name,
            'session_id': session_id,
            'agent': agent,
            'machine': machine,
            'session_type': session_type,
            'rollup_bytes': rollup_bytes,
            'fm_transcript_bytes': fm_transcript_bytes,
            'computed_transcript_bytes': computed_transcript_bytes,
            'transcript_bytes': transcript_bytes,
            'compression_ratio': compression_ratio,
            'token_compression': token_compression,
            'turns': turns,
            'subagents': subagents,
            'compactions': compactions,
            'duration_active': duration_active,
            'tokens_total': tokens_total,
            'tokens_input': tokens_input,
            'tokens_output': tokens_output,
            'tokens_cache_read': tokens_cache_read,
            'tokens_cache_creation': tokens_cache_creation,
            'cost_usd': cost_usd,
            'cost_recomputed': cost_recomputed,
            'model': model,
        })

    print(f"Loaded {len(rows)} rollup notes")

    # Filter to those with valid compression ratios
    compression_rows = [r for r in rows if r['compression_ratio'] is not None]

    print(f"{len(compression_rows)} have valid compression ratios")

    # Save per-rollup data
    csv_path = OUTPUT_DIR / "rq2_compression.csv"
    if compression_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=compression_rows[0].keys())
            writer.writeheader()
            writer.writerows(compression_rows)
    print(f"Wrote {csv_path}")

    # Helper functions for statistics
    def percentile(data, p):
        return np.percentile(data, p * 100) if len(data) > 0 else None

    def get_values(rows, key):
        return [r[key] for r in rows if r[key] is not None]

    # Compute summary statistics
    summary = {}

    # RQ2: Compression ratios
    compression_vals = get_values(compression_rows, 'compression_ratio')
    if compression_vals:
        summary['rq2_compression'] = {
            'n': len(compression_vals),
            'min': float(np.min(compression_vals)),
            'p10': float(percentile(compression_vals, 0.1)),
            'median': float(np.median(compression_vals)),
            'mean': float(np.mean(compression_vals)),
            'p90': float(percentile(compression_vals, 0.9)),
            'max': float(np.max(compression_vals)),
        }

    # By agent
    summary['rq2_by_agent'] = {}
    agents = set(r['agent'] for r in compression_rows)
    for agent in agents:
        agent_vals = [r['compression_ratio'] for r in compression_rows if r['agent'] == agent]
        if agent_vals:
            summary['rq2_by_agent'][agent] = {
                'n': len(agent_vals),
                'median': float(np.median(agent_vals)),
                'mean': float(np.mean(agent_vals)),
            }

    # By machine
    summary['rq2_by_machine'] = {}
    machines = set(r['machine'] for r in compression_rows if r['machine'])
    for machine in machines:
        machine_vals = [r['compression_ratio'] for r in compression_rows if r['machine'] == machine]
        if machine_vals:
            summary['rq2_by_machine'][str(machine)] = {
                'n': len(machine_vals),
                'median': float(np.median(machine_vals)),
                'mean': float(np.mean(machine_vals)),
            }

    # By session type
    summary['rq2_by_session_type'] = {}
    stypes = set(r['session_type'] for r in compression_rows)
    for stype in stypes:
        stype_vals = [r['compression_ratio'] for r in compression_rows if r['session_type'] == stype]
        if stype_vals:
            summary['rq2_by_session_type'][stype] = {
                'n': len(stype_vals),
                'median': float(np.median(stype_vals)),
                'mean': float(np.mean(stype_vals)),
            }

    # Correlations
    corr_vars = ['turns', 'subagents', 'compactions', 'transcript_bytes']
    summary['rq2_correlations'] = {}

    for var in corr_vars:
        valid_pairs = [(r['compression_ratio'], r[var]) for r in compression_rows
                       if r['compression_ratio'] is not None and r[var] is not None]
        if len(valid_pairs) > 10:
            comp_vals = [p[0] for p in valid_pairs]
            var_vals = [p[1] for p in valid_pairs]
            corr, pval = spearmanr(comp_vals, var_vals)
            summary['rq2_correlations'][var] = {
                'n': len(valid_pairs),
                'spearman_r': float(corr),
                'p_value': float(pval),
            }

    # Frontmatter vs computed transcript_bytes comparison
    fm_vs_comp = [r for r in compression_rows
                  if r['fm_transcript_bytes'] is not None and r['computed_transcript_bytes'] is not None]

    if fm_vs_comp:
        diffs = [r['fm_transcript_bytes'] - r['computed_transcript_bytes'] for r in fm_vs_comp]
        ratios = [r['fm_transcript_bytes'] / r['computed_transcript_bytes'] for r in fm_vs_comp]
        exact_matches = sum(1 for d in diffs if d == 0)

        summary['rq2_frontmatter_vs_computed'] = {
            'n_both_available': len(fm_vs_comp),
            'mean_diff': float(np.mean(diffs)),
            'median_diff': float(np.median(diffs)),
            'mean_ratio': float(np.mean(ratios)),
            'median_ratio': float(np.median(ratios)),
            'n_exact_match': exact_matches,
            'explanation': 'ratio > 1 means frontmatter overstates; ratio < 1 means it understates',
        }

    # Token-based compression
    token_comp_vals = get_values(compression_rows, 'token_compression')
    if token_comp_vals:
        summary['rq2_token_compression'] = {
            'n': len(token_comp_vals),
            'min': float(np.min(token_comp_vals)),
            'p10': float(percentile(token_comp_vals, 0.1)),
            'median': float(np.median(token_comp_vals)),
            'mean': float(np.mean(token_comp_vals)),
            'p90': float(percentile(token_comp_vals, 0.9)),
            'max': float(np.max(token_comp_vals)),
        }

    # RQ5: Cost analysis
    cost_rows = [r for r in rows if r['cost_usd'] is not None]

    if cost_rows:
        cost_vals = get_values(cost_rows, 'cost_usd')
        summary['rq5_cost'] = {
            'n': len(cost_vals),
            'median': float(np.median(cost_vals)),
            'mean': float(np.mean(cost_vals)),
            'total': float(np.sum(cost_vals)),
            'min': float(np.min(cost_vals)),
            'max': float(np.max(cost_vals)),
        }

        # By model
        summary['rq5_by_model'] = {}
        models = set(r['model'] for r in cost_rows if r['model'])
        for model in models:
            model_vals = [r['cost_usd'] for r in cost_rows if r['model'] == model]
            if model_vals:
                summary['rq5_by_model'][str(model)] = {
                    'n': len(model_vals),
                    'median': float(np.median(model_vals)),
                    'mean': float(np.mean(model_vals)),
                    'total': float(np.sum(model_vals)),
                }

        # cost_usd vs cost_recomputed
        recomputed = [r for r in cost_rows if r['cost_recomputed'] == True]
        if recomputed:
            summary['rq5_cost_recomputed_flag'] = {
                'n_with_recomputed_flag': len(recomputed),
                'note': 'cost_recomputed=true indicates values backfilled after 2026-09-20 fix',
            }

    # CRITICAL CAVEAT
    summary['rq5_cost_caveat'] = {
        'what_cost_usd_represents': 'ENTIRE SESSION COST, not rollup generation cost',
        'evidence': [
            'backfill_session_costs.py lines 46-87: computes cost by reading entire transcript and summing all assistant message usage',
            'rollup_note.py lines 347-353: total_tokens aggregated from all epochs in the session',
            'rollup_note.py line 485: cost_usd = estimate_cost_usd(total_tokens, primary_model)',
            'The rollup generation cost is NOT tracked in the frontmatter'
        ],
        'implication': 'cost_usd in rollup notes measures SESSION execution cost, NOT the cost of generating the summary',
        'rollup_generation_cost_location': 'Not currently tracked; rollup drain log shows "wrote <filename>" but no cost figures',
    }

    # Session type distribution
    summary['session_type_distribution'] = {}
    type_counts = defaultdict(int)
    for r in rows:
        type_counts[r['session_type']] += 1
    summary['session_type_distribution'] = dict(type_counts)

    # Methodology note
    summary['methodology'] = {
        'classification_method': 'keyword heuristics over ## Intent section',
        'classification_is_heuristic': True,
        'compression_formula': 'transcript_bytes / rollup_bytes',
        'token_compression_formula': 'sum(deduped usage.input_tokens + output_tokens) / (rollup_bytes / 4.0)',
        'deduplication': 'message.id used to dedupe transcript entries (naive counting overcounts ~2.4x)',
    }

    # Save summary
    json_path = OUTPUT_DIR / "rq2_rq5_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
