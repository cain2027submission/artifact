import json

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

mismatches = []

def check(draft_val, data_val, tolerance=0.001):
    """Check if values match within relative tolerance"""
    if isinstance(data_val, (int, float)) and isinstance(draft_val, (int, float)):
        if data_val == 0:
            return abs(draft_val) < tolerance
        return abs((draft_val - data_val) / data_val) < tolerance
    return draft_val == data_val

# ===  CORPUS TABLE (Table 1) ===
print("Checking Table 1 corpus counts...")
corpus_checks = [
    ('Transcripts parsed', 2219, rq1['n_transcripts_parsed'], 'rq1.n_transcripts_parsed'),
    ('Parent sessions', 1144, rq1['n_parent_transcripts'], 'rq1.n_parent_transcripts'),
    ('Subagent files', 1075, rq1['n_subagent_transcripts'], 'rq1.n_subagent_transcripts'),
    ('Substantive', 236, rq1['n_substantive'], 'rq1.n_substantive'),
    ('Trivial excluded', 908, rq1['n_trivial_excluded'], 'rq1.n_trivial_excluded'),
    ('Serialized bytes', 512300275, rq1['totals_substantive']['serialized_bytes'], 'rq1.totals_substantive.serialized_bytes'),
    ('Payload chars', 73727014, rq1['totals_substantive']['payload_chars'], 'rq1.totals_substantive.payload_chars'),
    ('API requests', 24649, rq1['totals_substantive']['api_requests'], 'rq1.totals_substantive.api_requests'),
    ('Tool calls', 25935, rq1['totals_substantive']['tool_calls'], 'rq1.totals_substantive.tool_calls'),
]

for label, draft, data, path in corpus_checks:
    if not check(draft, data):
        mismatches.append({
            'location': f'Table 1 - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# Rollup count check
if 550 != rq3['n_sessions']:
    mismatches.append({
        'location': 'Table 1 & Abstract - Generated rollup notes',
        'draft': 550,
        'data': rq3['n_sessions'],
        'path': 'rq3.n_sessions',
    })

# Paired samples count
if 286 != rq5['n_sessions']:
    mismatches.append({
        'location': 'Table 4 caption - Paired samples',
        'draft': 286,
        'data': rq5['n_sessions'],
        'path': 'rq5.n_sessions',
    })

# === THINKING PERSISTENCE (Table 2) ===
print("Checking Table 2 thinking persistence...")
thinking_checks = [
    ('Parent blocks', 17148, rq1['thinking_persistence']['parent']['blocks'], 'rq1.thinking_persistence.parent.blocks'),
    ('Parent empty', 16822, rq1['thinking_persistence']['parent']['empty'], 'rq1.thinking_persistence.parent.empty'),
    ('Parent empty %', 98.1, rq1['thinking_persistence']['parent']['empty_pct'], 'rq1.thinking_persistence.parent.empty_pct'),
    ('Subagent blocks', 16665, rq1['thinking_persistence']['subagent']['blocks'], 'rq1.thinking_persistence.subagent.blocks'),
    ('Subagent empty', 519, rq1['thinking_persistence']['subagent']['empty'], 'rq1.thinking_persistence.subagent.empty'),
    ('Subagent empty %', 3.1, rq1['thinking_persistence']['subagent']['empty_pct'], 'rq1.thinking_persistence.subagent.empty_pct'),
]

for label, draft, data, path in thinking_checks:
    if not check(draft, data, tolerance=0.02):
        mismatches.append({
            'location': f'Table 2 - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# === COMPOSITION (Table 3) ===
print("Checking Table 3 composition...")
p = rq1['shares_substantive']['_pooled']
composition_checks = [
    ('tool_result %', 43.3, p['tool_result']*100, 'shares_substantive._pooled.tool_result'),
    ('tool_use %', 30.1, p['tool_use']*100, 'shares_substantive._pooled.tool_use'),
    ('assistant_text %', 7.2, p['assistant_text']*100, 'shares_substantive._pooled.assistant_text'),
    ('thinking %', 0.09, p['thinking']*100, 'shares_substantive._pooled.thinking'),
    ('user_text %', 16.0, p['user_text']*100, 'shares_substantive._pooled.user_text'),
    ('system_injected %', 3.2, p['system_injected']*100, 'shares_substantive._pooled.system_injected'),
    ('Tool I/O %', 73.4, rq1['signal_vs_noise_substantive']['tool_io_pooled']*100, 'signal_vs_noise_substantive.tool_io_pooled'),
    ('Reasoning+prose %', 7.3, rq1['signal_vs_noise_substantive']['reasoning_and_prose_pooled']*100, 'signal_vs_noise_substantive.reasoning_and_prose_pooled'),
]

for label, draft, data, path in composition_checks:
    if not check(draft, data, tolerance=0.02):
        mismatches.append({
            'location': f'Table 3 - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# === COMPRESSION (Table 4) ===
print("Checking compression numbers...")
compression_checks = [
    ('Serialized median', 206.3, rq2['compression_serialized_bytes']['median'], 'rq2.compression_serialized_bytes.median'),
    ('Payload median', 34.8, rq2['compression_payload_bytes']['median'], 'rq2.compression_payload_bytes.median'),
    ('Tokens median', 33.0, rq2['compression_tokens']['median'], 'rq2.compression_tokens.median'),
    ('Tokens p10', 0.21, rq2['compression_tokens']['p10'], 'rq2.compression_tokens.p10'),
    ('Tokens p90', 92.8, rq2['compression_tokens']['p90'], 'rq2.compression_tokens.p90'),
    ('Tokens max', 538, rq2['compression_tokens']['max'], 'rq2.compression_tokens.max'),
    ('Inflation factor', 6.69, rq2['inflation_serialized_over_payload'], 'rq2.inflation_serialized_over_payload'),
    ('Expanding sessions n', 40, rq2['sessions_rollup_expands']['n'], 'rq2.sessions_rollup_expands.n'),
    ('Expanding sessions %', 14.0, rq2['sessions_rollup_expands']['pct'], 'rq2.sessions_rollup_expands.pct'),
]

for label, draft, data, path in compression_checks:
    if not check(draft, data, tolerance=0.02):
        mismatches.append({
            'location': f'Table 4 / §RQ2 - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# === SIZE BANDS (Table 5) ===
print("Checking size bands...")
bands_checks = [
    ('<10KB median', 0.35, rq2['compression_payload_by_size']['under_10KB']['median'], 'rq2.compression_payload_by_size.under_10KB.median'),
    ('10-100KB median', 19.04, rq2['compression_payload_by_size']['10KB_to_100KB']['median'], 'rq2.compression_payload_by_size.10KB_to_100KB.median'),
    ('100KB-1MB median', 53.96, rq2['compression_payload_by_size']['100KB_to_1MB']['median'], 'rq2.compression_payload_by_size.100KB_to_1MB.median'),
    ('>1MB median', 126.6, rq2['compression_payload_by_size']['over_1MB']['median'], 'rq2.compression_payload_by_size.over_1MB.median'),
]

for label, draft, data, path in bands_checks:
    if not check(draft, data, tolerance=0.02):
        mismatches.append({
            'location': f'Table 5 - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# === QUERIES (Table 6) ===
print("Checking query workload...")
org = rq0['organic']
query_checks = [
    ('n_queries', 2458, org['n_queries'], 'rq0.organic.n_queries'),
    ('n_distinct', 823, org['n_distinct'], 'rq0.organic.n_distinct'),
    ('Median words', 6, org['words']['median'], 'rq0.organic.words.median'),
    ('Keyword %', 77.4, org['shape']['keyword']['pct'], 'rq0.organic.shape.keyword.pct'),
    ('Question %', 22.6, org['shape']['question']['pct'], 'rq0.organic.shape.question.pct'),
    ('Over 8 tokens %', 30.8, org['over_8_tokens']['pct'], 'rq0.organic.over_8_tokens.pct'),
    ('Empty results %', 31.7, org['results_logged']['empty_pct_of_logged'], 'rq0.organic.results_logged.empty_pct_of_logged'),
    ('Median latency ms', 1150, org['live_latency_ms']['median'], 'rq0.organic.live_latency_ms.median'),
    ('p95 latency ms', 4120, org['live_latency_ms']['p95'], 'rq0.organic.live_latency_ms.p95'),
]

for label, draft, data, path in query_checks:
    if not check(draft, data, tolerance=0.02):
        mismatches.append({
            'location': f'Table 6 / Query workload - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# === RQ3 STRUCTURAL LOSS ===
print("Checking structural loss...")
agg = rq3['aggregate']
# Draft claims 89.5% dropped, 10.5% compressed
# Draft claims tool_result 73.7%, tool_use 15.8% of dropped
dropped_pct = agg['dropped_share'] * 100
compressed_pct = agg['compressed_share'] * 100
tool_result_dropped_pct = (agg['tool_result_tokens'] / agg['total_transcript_tokens']) * 100
tool_use_dropped_pct = (agg['tool_use_tokens'] / agg['total_transcript_tokens']) * 100

structural_checks = [
    ('Dropped %', 89.5, dropped_pct, 'rq3.aggregate.dropped_share'),
    ('Compressed %', 10.5, compressed_pct, 'rq3.aggregate.compressed_share'),
    ('Tool result dropped %', 73.7, tool_result_dropped_pct, 'rq3.aggregate.tool_result_tokens/total'),
    ('Tool use dropped %', 15.8, tool_use_dropped_pct, 'rq3.aggregate.tool_use_tokens/total'),
    ('Fair compression ratio', 15.9, agg['fair_compression_ratio'], 'rq3.aggregate.fair_compression_ratio'),
]

for label, draft, data, path in structural_checks:
    if not check(draft, data, tolerance=0.02):
        mismatches.append({
            'location': f'§RQ2 Structural loss - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# === RQ5 GENERATION COST ===
print("Checking generation cost...")
gen_checks = [
    ('Median cost uncached', 0.12, rq5['generation_cost_usd']['median'], 'rq5.generation_cost_usd.median'),
    ('Median cost cached', 0.03, rq5['generation_cost_usd_cached']['median'], 'rq5.generation_cost_usd_cached.median'),
    ('p90 cost uncached', 0.52, rq5['generation_cost_usd']['p90'], 'rq5.generation_cost_usd.p90'),
    ('p90 cost cached', 0.08, rq5['generation_cost_usd_cached']['p90'], 'rq5.generation_cost_usd_cached.p90'),
    ('Median input tokens', 35812, rq5['generation_input_tokens']['median'], 'rq5.generation_input_tokens.median'),
    ('Median output tokens', 1016, rq5['generation_output_tokens']['median'], 'rq5.generation_output_tokens.median'),
    ('Break-even uncached median', 1.16, rq5['break_even_retrievals']['median'], 'rq5.break_even_retrievals.median'),
    ('Break-even cached median', 0.23, rq5['break_even_retrievals_cached']['median'], 'rq5.break_even_retrievals_cached.median'),
    ('Break-even uncached p10', 1.06, rq5['break_even_retrievals']['p10'], 'rq5.break_even_retrievals.p10'),
    ('Break-even cached p10', 0.15, rq5['break_even_retrievals_cached']['p10'], 'rq5.break_even_retrievals_cached.p10'),
    ('Break-even uncached p90', 1.68, rq5['break_even_retrievals']['p90'], 'rq5.break_even_retrievals.p90'),
    ('Break-even cached p90', 0.68, rq5['break_even_retrievals_cached']['p90'], 'rq5.break_even_retrievals_cached.p90'),
    ('Sessions <=1 cached n', 227, rq5['sessions_break_even_le_1_cached']['n'], 'rq5.sessions_break_even_le_1_cached.n'),
    ('Sessions <=1 cached %', 93.8, rq5['sessions_break_even_le_1_cached']['pct'], 'rq5.sessions_break_even_le_1_cached.pct'),
]

for label, draft, data, path in gen_checks:
    if not check(draft, data, tolerance=0.02):
        mismatches.append({
            'location': f'Table breakeven / §RQ5 - {label}',
            'draft': draft,
            'data': data,
            'path': path,
        })

# === ADDITIONAL CHECKS ===
print("Checking derived claims...")

# 93% reasoning in subagents
parent_reasoning = rq1['thinking_persistence']['parent']['retained_chars']
subagent_reasoning = rq1['thinking_persistence']['subagent']['retained_chars']
total_reasoning = parent_reasoning + subagent_reasoning
subagent_pct = (subagent_reasoning / total_reasoning) * 100
if not check(93, subagent_pct, tolerance=0.02):
    mismatches.append({
        'location': 'Abstract & §RQ1 - 93% reasoning in subagents',
        'draft': 93,
        'data': round(subagent_pct, 1),
        'path': 'Derived: subagent_reasoning / (parent + subagent)',
    })

# 55.07% subagent volume
subagent_vol_pct = rq1['subagent_volume']['share_of_total_corpus_payload'] * 100
if not check(55.07, subagent_vol_pct, tolerance=0.02):
    mismatches.append({
        'location': 'Abstract & §RQ1 - 55.07% subagent volume',
        'draft': 55.07,
        'data': round(subagent_vol_pct, 2),
        'path': 'rq1.subagent_volume.share_of_total_corpus_payload',
    })

# PRINT RESULTS
print(f"\n{'='*80}")
if len(mismatches) == 0:
    print("✓ ALL CHECKS PASSED")
else:
    print(f"✗ FOUND {len(mismatches)} MISMATCHES")
print(f"{'='*80}\n")

for m in mismatches:
    print(f"Location: {m['location']}")
    print(f"  Draft: {m['draft']}")
    print(f"  Data:  {m['data']}")
    print(f"  Path:  {m['path']}")
    if isinstance(m['draft'], (int, float)) and isinstance(m['data'], (int, float)):
        print(f"  Diff:  {m['draft'] - m['data']}")
    print()
