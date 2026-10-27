#!/usr/bin/env python3
"""Generate the HELD-OUT question set for the 25 sessions in rq3_heldout_manifest.json.

Why this file exists. paper/main.tex states the problem it answers plainly: "the
action record was designed with these questions visible...so a held-out question
set is required before this dominance is settled." The 124-question design set in
rq3_questions.jsonl was visible while the action record format was being written,
so the action record's 96.8% against output-stripped's 94.4% cannot settle
anything on its own. These questions are drawn from sessions dated after the last
graded session, by the same generator, and no arm has seen them.

What is held fixed, because a question-generation difference would be a confound
larger than the effect under test: the generator is
part2_question_retention.generate_questions, imported rather than copied, so
QUESTION_MODEL ("claude-sonnet-5"), the 5-class prompt, and the
transcript_text[:30000] truncation are the published ones by construction. The
transcript rendering is part2's transcript_to_text, also imported.

Output goes to a NEW file, rq3_heldout_questions.jsonl. rq3_questions.jsonl is
PUBLISHED DATA and is opened with 'w' by three scripts in this directory
(part2_question_retention.py:370, part2_streamlined.py:228, resume_rq3.py:81), so
nothing here goes near it and no existing default output path is repointed.

Schema matches rq3_questions.jsonl so the arm scripts can consume it unchanged:
session_id (the rollup note slug, which rq3_raw_transcript_arm.SLUG matches to
recover the 8-hex prefix), class, question, ground_truth, judge_answer, rating,
grade. The last three are null here and carry no information: there is no rollup
arm in this validation, so no published grade exists to compare against. Two keys
are added, session_prefix and session_uuid, so the prebuilt artifacts can be
addressed without re-deriving the prefix from a regex.

Usage:
    rq3_heldout_questions.py --dry-run     # resolve sessions, no API calls
    rq3_heldout_questions.py --limit 2     # smoke test: first 2 sessions
    rq3_heldout_questions.py               # all 25 (resumable, appends)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part2_question_retention as gen  # noqa: E402
from part2_question_retention import (  # noqa: E402
    QUESTION_MODEL, generate_questions, load_transcript, transcript_to_text,
)
from paths import SESSIONS_DIR  # noqa: E402
from rq3_raw_transcript_arm import DATA, load_records  # noqa: E402

# The one disclosed deviation from the published generator, and it is a defect fix
# rather than a change of method. At the published max_tokens=2048 the reply is cut
# mid-string on these sessions -- 5 questions each quoting a verbatim ground truth
# does not fit -- json.loads raises "Unterminated string", generate_questions
# returns [], and the session yields nothing. Measured on f7bd7d1f: truncated at
# char 1759, inside question 1 of 5. Nothing else about generation moves: same
# model, same prompt, same transcript_text[:30000]. The prompt, the model and the
# truncation point are what a question set has to hold fixed to be comparable; an
# output cap that silently deletes whole sessions is not a property worth
# reproducing, and reproducing it here would bias the held-out set toward sessions
# whose ground truths happen to be short.
gen.QUESTION_MAX_TOKENS = 8192

MANIFEST = DATA / "rq3_heldout_manifest.json"
OUT = DATA / "rq3_heldout_questions.jsonl"
SLUG_STEM = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z-]+_([0-9a-f]{8})_")

CLASSES = ["decision", "dead_end", "outcome", "artifact_detail", "sequence"]


def rollup_slugs() -> Dict[str, str]:
    """prefix -> rollup note stem. The stem is the arms' session_id, because
    rq3_raw_transcript_arm.SLUG is what recovers the prefix downstream."""
    found: Dict[str, List[str]] = {}
    for p in SESSIONS_DIR.glob("*.md"):
        m = SLUG_STEM.match(p.stem)
        if m:
            found.setdefault(m.group(1), []).append(p.stem)
    # A prefix with two notes would make session_id ambiguous, and record_id is
    # derived from session_id, so pick deterministically rather than by glob order.
    return {k: sorted(v)[0] for k, v in found.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated 8-hex prefixes")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    sessions = manifest["per_session"]
    slugs = rollup_slugs()

    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        sessions = [s for s in sessions if s["session_prefix"] in want]
    sessions.sort(key=lambda s: s["session_prefix"])
    if args.limit:
        sessions = sessions[:args.limit]

    done = set()
    if OUT.exists():
        done = {r["session_id"] for r in load_records(OUT)}
        print(f"resuming: {len(done)} sessions already have questions")
    else:
        OUT.touch()

    missing_slug = [s["session_prefix"] for s in sessions
                    if s["session_prefix"] not in slugs]
    missing_tx = [s["session_prefix"] for s in sessions
                  if not Path(s["transcript"]).exists()]
    print(f"sessions {len(sessions)}  missing rollup slug {len(missing_slug)}  "
          f"missing transcript {len(missing_tx)}")
    if missing_slug:
        print(f"  no slug: {missing_slug}")
    if missing_tx:
        print(f"  no transcript: {missing_tx}")

    if args.dry_run:
        print(f"\n{'sess':10s} {'stratum':10s} {'raw MB':>8s}  session_id")
        for s in sessions:
            print(f"{s['session_prefix']:10s} {s['stratum']:10s} "
                  f"{s['raw_bytes']/1e6:8.2f}  "
                  f"{slugs.get(s['session_prefix'], '-')}")
        print(f"\nquestion model: {QUESTION_MODEL}")
        print("--dry-run: no API calls made")
        return

    n_written = 0
    for i, s in enumerate(sessions, 1):
        prefix = s["session_prefix"]
        slug = slugs.get(prefix)
        tx = Path(s["transcript"])
        if slug is None or not tx.exists():
            print(f"[{i}/{len(sessions)}] {prefix} SKIP (no slug or transcript)")
            continue
        if slug in done:
            print(f"[{i}/{len(sessions)}] {prefix} already done")
            continue

        text = transcript_to_text(load_transcript(tx))
        print(f"[{i}/{len(sessions)}] {prefix} transcript_text {len(text):,} chars "
              f"-> generator sees {min(len(text), 30000):,}")
        questions = generate_questions(text, slug)
        if not questions:
            print(f"  FAILED, no questions for {prefix}")
            continue

        got = []
        with OUT.open("a") as fh:
            for q in questions:
                q_class = (q.get("class") or "").strip().lower()
                if q_class not in CLASSES:
                    print(f"  WARNING: unrecognised class {q.get('class')!r}, kept as-is")
                rec = {
                    "session_id": slug,
                    "class": q_class,
                    "question": q.get("question", ""),
                    "ground_truth": q.get("ground_truth", ""),
                    # Null by construction: this validation has no rollup arm, so
                    # there is no published grade for the arms to print alongside.
                    "judge_answer": None,
                    "rating": None,
                    "grade": None,
                    "session_prefix": prefix,
                    "session_uuid": s["session_id"],
                }
                fh.write(json.dumps(rec) + "\n")
                got.append(q_class)
                n_written += 1
        print(f"  wrote {len(got)} questions: {got}")

    rows = load_records(OUT)
    from collections import Counter
    print(f"\nwrote {n_written} this run; {OUT.name} now holds {len(rows)} "
          f"questions over {len({r['session_id'] for r in rows})} sessions")
    print(f"class counts: {dict(Counter(r['class'] for r in rows))}")
    print(f"  {OUT}")


if __name__ == "__main__":
    main()
