# Replication package — CAIN 2027 submission

Analysis scripts for a CAIN 2027 submission currently under double-anonymous review.
Author names and affiliations are withheld for review.

## Where each result comes from

All scripts live in `scripts/paper-rollups/`. Each writes its output under
`data/paper-rollups/` (not included, see below).

| RQ / section | Paper item | Scripts (run in this order) |
|---|---|---|
| Corpus | "The corpus", all RQs | [`freeze_corpus.py`](scripts/paper-rollups/freeze_corpus.py) freezes the session manifest every RQ is computed over |
| RQ1 transcripts are mostly machine output | Table `composition` | [`rq1_signal_composition_corrected.py`](scripts/paper-rollups/rq1_signal_composition_corrected.py) (our sessions), [`public_composition.py`](scripts/paper-rollups/public_composition.py) (public dataset) |
| RQ2 compression is mostly deletion | Fig. `deletion` | [`rq2_compression_corrected.py`](scripts/paper-rollups/rq2_compression_corrected.py) → [`make_figures.py`](scripts/paper-rollups/make_figures.py) |
| RQ3 what a later session recovers | Fig. `retention`, headline recall | arm runners `rq3_*_arm.py` → grading [`part2_streamlined.py`](scripts/paper-rollups/part2_streamlined.py), [`rq3_majority_vote_grades.py`](scripts/paper-rollups/rq3_majority_vote_grades.py) → [`rq3_retention_summary.py`](scripts/paper-rollups/rq3_retention_summary.py) → [`make_figures.py`](scripts/paper-rollups/make_figures.py) |
| RQ3 | Table `paired` | [`rq3_paired_comparison.py`](scripts/paper-rollups/rq3_paired_comparison.py), [`rq3_stripped_paired.py`](scripts/paper-rollups/rq3_stripped_paired.py), [`rq3_three_way_paired.py`](scripts/paper-rollups/rq3_three_way_paired.py) |
| RQ3 | Table `failmode` (wrong answers, declines) | [`rq3_error_rates_by_set.py`](scripts/paper-rollups/rq3_error_rates_by_set.py), [`rq3_error_rate_fix.py`](scripts/paper-rollups/rq3_error_rate_fix.py) |
| RQ3 no-digest action record | Table `nodigest` | [`rq3_action_record_no_digest_arm.py`](scripts/paper-rollups/rq3_action_record_no_digest_arm.py), [`rq3_heldout_no_digest_extend.py`](scripts/paper-rollups/rq3_heldout_no_digest_extend.py); replication: [`replication_freeze.py`](scripts/paper-rollups/replication_freeze.py) → [`replication_run.py`](scripts/paper-rollups/replication_run.py) → [`replication_analysis.py`](scripts/paper-rollups/replication_analysis.py) |
| RQ3 position vs length | Table `position` | [`rq3_late_window.py`](scripts/paper-rollups/rq3_late_window.py), [`rq3_late_control.py`](scripts/paper-rollups/rq3_late_control.py) → [`rq3_late_analyse.py`](scripts/paper-rollups/rq3_late_analyse.py), [`rq3_late_control_analyse.py`](scripts/paper-rollups/rq3_late_control_analyse.py) |
| RQ3 generator and length | Tables `frontier`, `length`; Fig. `budget` | [`rq3_read_budget_arms.py`](scripts/paper-rollups/rq3_read_budget_arms.py), [`rq3_prose_only_budget_ladder.py`](scripts/paper-rollups/rq3_prose_only_budget_ladder.py), [`rq3_sonnet_rollup_124.py`](scripts/paper-rollups/rq3_sonnet_rollup_124.py), [`frontier_bands.py`](scripts/paper-rollups/frontier_bands.py) → [`make_figures.py`](scripts/paper-rollups/make_figures.py) |
| RQ3 under a read cap | "Under a read cap…" | [`rq3_cap400_run.py`](scripts/paper-rollups/rq3_cap400_run.py), [`rq3_cap400_control.py`](scripts/paper-rollups/rq3_cap400_control.py) |
| RQ4 retrieval rankings | Table `rq4` | [`freeze_corpus_rq4.py`](scripts/paper-rollups/freeze_corpus_rq4.py) → [`rq4_known_item.py`](scripts/paper-rollups/rq4_known_item.py), [`rq4_bm25_no_tool.py`](scripts/paper-rollups/rq4_bm25_no_tool.py), [`rq4_batched.py`](scripts/paper-rollups/rq4_batched.py) |
| Discussion: pricing | "Pricing each artifact…" | [`rq5_generation_cost.py`](scripts/paper-rollups/rq5_generation_cost.py), [`rq5_cold_cache_correction.py`](scripts/paper-rollups/rq5_cold_cache_correction.py) |
| Statistics | CIs, equivalence, per-class tests | [`cluster_bootstrap.py`](scripts/paper-rollups/cluster_bootstrap.py), [`rq3_heldout_equivalence_test.py`](scripts/paper-rollups/rq3_heldout_equivalence_test.py) (TOST), [`rq3_per_class_mcnemar.py`](scripts/paper-rollups/rq3_per_class_mcnemar.py) |

Table and figure names are the paper's LaTeX labels (`tab:` / `fig:` dropped).

Other files in `scripts/paper-rollups/`:

- **shared modules** imported by the scripts above, e.g. [`paths.py`](scripts/paper-rollups/paths.py) (all paths and
  machine-specific settings), [`grading_common.py`](scripts/paper-rollups/grading_common.py), [`action_record.py`](scripts/paper-rollups/action_record.py)
- **pipeline steps** the scripts above read data from: question generation, held-out
  and replication sets, regrading
- **checks**: `test_*.py` (unit tests) and [`verify_determinism.py`](scripts/paper-rollups/verify_determinism.py) (verifies byte-identical reproduction)

## What is not here, and why

The study corpus is not released. It consists of production coding-agent transcripts
(Claude Code `.jsonl` session logs) and the session notes generated from them. These
contain personal data and private source code, so they are withheld for privacy reasons.

For reference, the frozen corpus (snapshot 2026-09-27) contains 2,602 transcripts and
499 session notes, of which 300 sessions are paired (both a transcript and a note).

## Running on your own data

The scripts read standard Claude Code transcripts, so they can be pointed at any
corpus of your own sessions (by default `~/.claude/projects/`).

1. Python 3.13 and `pip install -r requirements.txt`.
2. Copy `scripts/paper-rollups/paper_config.example.json` to `scripts/paper-rollups/paper_config.json` and set the
   paths for your machine. Every key can also be set as a `PAPER_<KEY>` environment
   variable. A fresh clone runs with no config file except where noted in the example.
3. Freeze a corpus with [`freeze_corpus.py`](scripts/paper-rollups/freeze_corpus.py), then run the scripts in the table above.

Some scripts drive the note-generation pipeline under study and import its modules
(`rollup_*`, `regrade`, `rq4_experiment`), which are not part of this package; those
scripts document the step but will not run standalone. Scripts that call a model
need an API key for the provider they name.

## License

MIT for code.
