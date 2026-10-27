#!/usr/bin/env python3
"""Part 2: Question-class retention - does compression lose what matters?"""

import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR, SESSIONS_DIR
from rq3_raw_transcript_arm import extract_confidence  # noqa: E402
from grading_common import (  # noqa: E402
    GRADE_ERROR, PARSER_VERSION, PROMPT_VERSION, FailureCounter, GradeError,
    grade_with,
)


GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
QUESTION_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-sonnet-5"

# How a self-declared PARTIALLY_ANSWERED is graded. The original code branched on
# NOT_ANSWERABLE alone, so a partial went to the grader unconstrained and could
# come back CORRECT -- a judge calling its own answer incomplete and then being
# scored fully right. Capping it at `partially_correct` is the only change:
#   - it stays `attempted`, because a partial answer is NOT a visible miss. The
#     whole point of wrong-of-attempted is that a calling agent cannot tell a
#     wrong answer from a right one, and it cannot tell a partial one either.
#   - `wrong` and `absent` are untouched, so wrong-of-attempted is arithmetically
#     identical with the cap on or off. Only `correct` moves, into
#     `partially_correct`, whose sum (`recoverable`) is also unchanged.
# Set False to reproduce the published, effectively binary grading exactly.
CAP_PARTIAL_AT_PARTIALLY_CORRECT = True

# Output cap for generate_questions only. 2048 is the published value and stays the
# default so nothing already produced changes. It is also demonstrably too small
# for this prompt: 5 questions each carrying a verbatim ground-truth quote runs past
# it, the reply is cut mid-string, json.loads raises "Unterminated string", and
# generate_questions returns [] -- so a session whose ground truths are long
# contributes NO questions and the loss is silent, because call_llm only tested for
# an empty reply and never looked at finish_reason. rq3_heldout_questions.py raises
# this; see its docstring for why that is a disclosed deviation rather than a fix
# to the published set.
QUESTION_MAX_TOKENS = 2048


def post(url: str, payload: dict, timeout: int = 120) -> dict:
    """Post JSON to gateway endpoint."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def call_llm(prompt: str, model: str, max_tokens: int = 2048) -> str:
    """Call LLM via gateway. No temperature for Claude 5 models.

    `max_tokens` defaults to the published 2048 so every existing caller is
    unaffected; only generate_questions passes anything else.
    """
    url = f"{GATEWAY}/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        data = post(url, payload, timeout=120)
        content = data['choices'][0]['message']['content']
        finish_reason = data['choices'][0].get('finish_reason')

        if not content or not content.strip():
            print(f"  WARNING: Empty response from {model}, finish_reason={finish_reason}")
            return ""

        if finish_reason == 'length':
            # Previously invisible. A truncated reply is not an empty one, so this
            # fell straight through to a JSONDecodeError in generate_questions and
            # the session was dropped with no indication that the cap, rather than
            # the model, was the cause.
            print(f"  WARNING: {model} hit max_tokens={max_tokens}, reply truncated")

        return content.strip()

    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        print(f"  ERROR calling {model}: {error}")
        return ""


def parse_frontmatter(content: str) -> Dict:
    """Extract YAML frontmatter."""
    if not content.startswith('---'):
        return {}
    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}

    result = {}
    for line in parts[1].strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result


def load_transcript(path: Path) -> List[Dict]:
    """Load JSONL transcript."""
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def transcript_to_text(entries: List[Dict]) -> str:
    """Convert transcript to readable text for question generation."""
    parts = []

    for entry in entries:
        entry_type = entry.get('type')

        if entry_type not in ['user', 'assistant']:
            continue

        msg = entry.get('message', {})
        role = msg.get('role')
        content = msg.get('content', '')

        if role == 'user':
            if isinstance(content, str):
                parts.append(f"USER: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            parts.append(f"USER: {block.get('text', '')}")
                        elif block.get('type') == 'tool_result':
                            # Include tool results, but truncated
                            result = str(block.get('content', ''))[:500]
                            parts.append(f"TOOL_RESULT: {result}")

        elif role == 'assistant':
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    if block.get('type') == 'text':
                        parts.append(f"ASSISTANT: {block.get('text', '')}")
                    elif block.get('type') == 'thinking':
                        parts.append(f"THINKING: {block.get('thinking', '')}")
                    elif block.get('type') == 'tool_use':
                        tool_name = block.get('name', 'unknown')
                        tool_input = json.dumps(block.get('input', {}))[:200]
                        parts.append(f"TOOL: {tool_name} {tool_input}")

    return '\n\n'.join(parts)


def generate_questions(transcript_text: str, session_id: str,
                       offset: int = 0, excerpt_note: str = "") -> List[Dict]:
    """Generate 5 questions from transcript only. NEVER show rollup here.

    `offset` selects which 30,000-character window of transcript_text the
    generator sees. The default 0 is the published behaviour (the first 30,000
    characters); rq3_late_window.py passes a later offset for its MID and LATE
    windows. `excerpt_note`, empty by default (so the published prompt is
    byte-identical), is inserted before the TRANSCRIPT block to tell the
    generator its text is an excerpt with more session after it."""

    prompt = f"""Below is the raw transcript of an agent session. Read it carefully and generate EXACTLY 5 questions, one for each class below. Each question must be answerable from this transcript.

Question classes:
1. DECISION - Why one approach was chosen over another
2. DEAD_END - What was tried and failed, and why
3. OUTCOME - The final state/result of the session
4. ARTIFACT_DETAIL - A specific concrete value: exact error string, flag, path, version number, or measured figure
5. SEQUENCE - The order in which things were done

For each question, provide:
- The question class (DECISION, DEAD_END, OUTCOME, ARTIFACT_DETAIL, or SEQUENCE)
- The question text
- The ground-truth answer extracted from the transcript

Format your response as JSON:
[
  {{
    "class": "DECISION",
    "question": "...",
    "ground_truth": "..."
  }},
  ...
]

{excerpt_note}TRANSCRIPT:
{transcript_text[offset:offset + 30000]}

Generate the 5 questions as JSON now:"""

    response = call_llm(prompt, QUESTION_MODEL, max_tokens=QUESTION_MAX_TOKENS)

    if not response:
        print(f"  FAILED to generate questions for {session_id}")
        return []

    # Parse JSON from response
    try:
        # Try to extract JSON block
        if '```json' in response:
            json_text = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_text = response.split('```')[1].split('```')[0]
        else:
            json_text = response

        questions = json.loads(json_text.strip())

        if not isinstance(questions, list) or len(questions) != 5:
            print(f"  WARNING: Got {len(questions) if isinstance(questions, list) else 0} questions instead of 5 for {session_id}")

        return questions

    except json.JSONDecodeError as e:
        print(f"  ERROR parsing questions JSON for {session_id}: {e}")
        print(f"  Response was: {response[:500]}")
        return []


def judge_question_from_rollup(question: str,
                               rollup_content: str) -> Tuple[str, str, str]:
    """Present ONLY the rollup to judge. Return (answer, confidence, raw_reply).

    The raw reply is returned and stored because it was not, and that is why the
    contamination this fix removes can never be quantified over the records
    already on disk: rq3_questions.jsonl keeps only the post-parse `judge_answer`,
    so re-parsing it is circular. Keeping the bytes makes the next parser change
    measurable instead of merely arguable.

    The confidence marker is parsed by `extract_confidence`, imported from
    rq3_raw_transcript_arm so all four RQ3 code paths share one implementation.
    It replaces a substring test that had three defects, all of which pushed this
    file's output in a specific direction:
      1. "ANSWERED" is a substring of "PARTIALLY_ANSWERED" and was tested first
         with a `break`, so every partial collapsed to a full ANSWERED. The nine
         records this script contributed to rq3_questions.jsonl carry zero
         PARTIALLY_ANSWERED, against 27 of the 124 published records written by
         the differently-parsed resume/streamlined scripts.
      2. "ANSWERED" also substring-matches prose -- "cannot be answered",
         "unanswered". Record 126 of rq3_questions.jsonl has the stored line
         'ANSWERED note: The note does not mention any command-line tools...',
         which this test would have read as a confident answer had it landed on
         the final line. An honest refusal graded as a hallucination.
      3. Only lines[-1] was inspected and lines[:-1] was kept as the answer, so a
         reply whose marker was not last lost a line of real content.
    """

    prompt = f"""Someone asks: "{question}"

Below is a compressed note summarizing an agent session. Answer the question based ONLY on this note. If the note does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

NOTE:
{rollup_content}

Provide your answer and confidence now:"""

    response = call_llm(prompt, JUDGE_MODEL)

    if not response:
        # Critically NOT "NOT_ANSWERABLE". An empty reply means the gateway
        # errored or timed out, and NOT_ANSWERABLE grades as `absent` -- the very
        # outcome this experiment measures. API_ERROR is never graded.
        return "", "API_ERROR", ""

    answer, confidence = extract_confidence(response)
    return answer, confidence, response


def grade_answer_row(judge_answer: str, ground_truth: str, confidence: str,
                     question: str = "") -> dict:
    """Grade via the shared grading_common module. Returns the row fragment;
    raises GradeError (never returns `absent`) when the grader call fails, and
    `api_error` is passed through for a failed answering call."""
    if confidence == "API_ERROR":
        # A gateway failure is not a finding about the rollup. The caller drops it.
        return {"grade": "api_error", "contradictory": False,
                "grader_response": None, "prompt_version": PROMPT_VERSION,
                "parser_version": PARSER_VERSION}
    return grade_with(lambda prompt: call_llm(prompt, JUDGE_MODEL),
                      question, ground_truth, judge_answer, confidence,
                      cap_partial=CAP_PARTIAL_AT_PARTIALLY_CORRECT)


def grade_answer(judge_answer: str, ground_truth: str, confidence: str,
                 question: str = "") -> str:
    """Grade-only wrapper. Returns: correct, partially_correct, wrong, absent,
    or api_error. Raises GradeError when the grader call fails."""
    return grade_answer_row(judge_answer, ground_truth, confidence,
                            question)["grade"]


def sample_sessions(pairs: List[Tuple[Path, Path]], n: int = 60, seed: int = 42) -> List[Tuple[Path, Path]]:
    """Sample sessions stratified by size (100KB-1MB and >1MB)."""
    random.seed(seed)

    # Categorize by transcript size
    small = []  # 100KB - 1MB
    large = []  # > 1MB

    for rollup_path, transcript_path in pairs:
        size = transcript_path.stat().st_size

        if 100_000 <= size < 1_000_000:
            small.append((rollup_path, transcript_path))
        elif size >= 1_000_000:
            large.append((rollup_path, transcript_path))

    print(f"  Found {len(small)} sessions 100KB-1MB, {len(large)} sessions >1MB")

    # Sample proportionally
    n_small = min(len(small), n // 2)
    n_large = min(len(large), n - n_small)

    sample = random.sample(small, n_small) + random.sample(large, n_large)
    random.shuffle(sample)

    print(f"  Sampled {len(sample)} sessions ({n_small} small, {n_large} large)")
    return sample


def main():
    seed = int(os.environ.get('SEED', '42'))
    print(f"Using seed: {seed}")

    # Load pairs from Part 1 results
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "data" / "paper-rollups" / "rq3_structural_loss.json"

    if not results_path.exists():
        print("ERROR: Run part1_structural_loss.py first")
        return

    structural = json.loads(results_path.read_text())

    # Rebuild pairs
    pairs = []
    rollups_dir = SESSIONS_DIR
    transcripts_base = PROJECTS_DIR

    for session in structural['sessions']:
        session_id = session['session_id']
        rollup_filename = session.get('rollup_filename', session_id)  # Fallback to session_id

        rollup_path = rollups_dir / f"{rollup_filename}.md"
        transcript_matches = list(transcripts_base.rglob(f'{session_id}.jsonl'))

        if rollup_path.exists() and transcript_matches:
            pairs.append((rollup_path, transcript_matches[0]))

    print(f"Found {len(pairs)} paired sessions")

    # Sample
    sample = sample_sessions(pairs, n=60, seed=seed)

    # Process each session
    questions_output = repo_root / "data" / "paper-rollups" / "rq3_questions.jsonl"
    questions_file = questions_output.open('w')

    retention_counts = {
        'decision': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'dead_end': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'outcome': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'artifact_detail': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'sequence': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
    }
    api_errors = 0
    grade_errors = 0
    grade_failures = FailureCounter("part2 grader")

    for i, (rollup_path, transcript_path) in enumerate(sample):
        session_id = rollup_path.stem
        print(f"\n[{i+1}/{len(sample)}] Processing {session_id}")

        # Load transcript (NEVER show rollup to question generator)
        entries = load_transcript(transcript_path)
        transcript_text = transcript_to_text(entries)

        # Generate questions from transcript only
        print(f"  Generating questions from transcript...")
        questions = generate_questions(transcript_text, session_id)

        if not questions:
            continue

        # Load rollup
        rollup_content = rollup_path.read_text()

        # Judge each question from rollup only
        for q in questions:
            q_class = q.get('class', '').lower()
            question_text = q.get('question', '')
            ground_truth = q.get('ground_truth', '')

            print(f"  Judging {q_class.upper()} question...")

            judge_answer, confidence, raw_reply = judge_question_from_rollup(
                question_text, rollup_content)
            try:
                graded = grade_answer_row(judge_answer, ground_truth,
                                          confidence, question_text)
            except GradeError:
                # Not written and not counted as a grade; a rerun retries it.
                grade_failures.record(GRADE_ERROR)
                print(f"    -> grade_error, record dropped")
                grade_errors += 1
                grade_failures.check()
                continue
            grade = graded['grade']
            if graded['grader_response'] is not None:
                grade_failures.record(grade)

            if confidence == "API_ERROR":
                # Not written and not counted. Scoring a gateway failure as
                # `absent` would inflate exactly the rate this measures.
                print(f"    -> API_ERROR, record dropped")
                api_errors += 1
                continue

            # Record
            record = {
                'session_id': session_id,
                'class': q_class,
                'question': question_text,
                'ground_truth': ground_truth,
                'raw_judge_reply': raw_reply,
                'judge_answer': judge_answer,
                'confidence': confidence,
                'grade': grade,
                'contradictory': graded['contradictory'],
                'grader_response': graded['grader_response'],
                'prompt_version': graded['prompt_version'],
                'parser_version': graded['parser_version'],
            }

            questions_file.write(json.dumps(record) + '\n')
            questions_file.flush()

            # Update counts
            if q_class in retention_counts:
                retention_counts[q_class][grade] += 1

    questions_file.close()
    grade_failures.check(final=True)

    # Compute summary
    summary = {
        'n_sessions': len(sample),
        'n_api_errors_dropped': api_errors,
        'n_grade_errors_dropped': grade_errors,
        'retention_by_class': {},
        'hallucination_rate': 0,
    }

    total_wrong = 0
    total_questions = 0

    for q_class, counts in retention_counts.items():
        total = sum(counts.values())
        if total > 0:
            summary['retention_by_class'][q_class] = {
                'total': total,
                'correct': counts['correct'],
                'partially_correct': counts['partially_correct'],
                'wrong': counts['wrong'],
                'absent': counts['absent'],
                'correct_rate': counts['correct'] / total,
                'partial_rate': counts['partially_correct'] / total,
                'wrong_rate': counts['wrong'] / total,
                'absent_rate': counts['absent'] / total,
            }

            total_wrong += counts['wrong']
            total_questions += total

    if total_questions > 0:
        summary['hallucination_rate'] = total_wrong / total_questions

    # Save summary (will be merged with Part 3)
    summary_path = repo_root / "data" / "paper-rollups" / "rq3_retention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n\nPart 2 Complete")
    print(f"  Sessions: {summary['n_sessions']}")
    print(f"  Total questions: {total_questions}")
    print(f"  API errors dropped (not scored as absent): {api_errors}")
    print(f"  Hallucination rate: {summary['hallucination_rate']:.1%}\n")

    for q_class, stats in summary['retention_by_class'].items():
        print(f"{q_class.upper():20} correct={stats['correct_rate']:.1%} partial={stats['partial_rate']:.1%} wrong={stats['wrong_rate']:.1%} absent={stats['absent_rate']:.1%}")

    print(f"\nResults saved to:")
    print(f"  {questions_output}")
    print(f"  {summary_path}")


if __name__ == '__main__':
    main()
