# Replication package — CAIN 2027 submission

Analysis scripts for a CAIN 2027 submission currently under double-anonymous review.
Author names and affiliations are withheld for review.

## What is here

`scripts/paper-rollups/` contains every script used to build the corpus, run the experimental arms,
grade answers and compute the numbers, tables and figures reported in the paper, plus
the gates used to check the manuscript (`check_paper_*.py`, `check_anonymity.py`).

## What is not here, and why

The study corpus is not released. It consists of real coding-agent transcripts
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
3. Freeze a corpus with `freeze_corpus.py`, then run the analysis scripts.

Some scripts drive the note-generation pipeline under study and import its modules
(`rollup_*`, `regrade`, `rq4_experiment`), which are not part of this package; those
scripts document the step but will not run standalone. Scripts that call a model
need an API key for the provider they name.

## License

MIT for code.
