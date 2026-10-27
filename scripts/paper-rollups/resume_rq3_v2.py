#!/usr/bin/env python3
"""Resume RQ3 with improved JSON parsing robustness."""

import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR, SESSIONS_DIR
from grading_common import GRADE_ERROR, FailureCounter, GradeError, grade_with

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-sonnet-5"
START_TIME = time.time()
DEADLINE_MINUTES = 75

def log(msg, end='\n'):
    """Print with immediate flush."""
    print(msg, end=end, flush=True)

def call_llm(prompt: str, timeout=120) -> str:
    """Call LLM via gateway. NEVER send temperature field."""
    url = f"{GATEWAY}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
            content = data['choices'][0]['message']['content']
            return content.strip() if content else ""
    except Exception as e:
        log(f"  ERROR: {e}")
        return ""

def load_existing_sessions(questions_path):
    """Load existing records and return set of session_ids already done."""
    if not questions_path.exists():
        return set(), []

    raw = questions_path.read_text(errors="replace")
    dec = json.JSONDecoder()
    rows = []
    i = 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \n\r\t":
            i += 1
        if i >= len(raw):
            break
        try:
            obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            log(f"  Truncation at byte {i}, stopping load")
            break
        rows.append(obj)

    session_ids = {r['session_id'] for r in rows if 'session_id' in r}
    log(f"Loaded {len(rows)} existing records from {len(session_ids)} sessions")
    return session_ids, rows

def load_pairs():
    """Load session pairs from Part 1 results."""
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "data" / "paper-rollups" / "rq3_structural_loss.json"
    structural = json.load(results_path.open())

    pairs = []
    rollups_dir = SESSIONS_DIR
    transcripts_base = PROJECTS_DIR

    for session in structural['sessions']:
        session_id = session['session_id']
        rollup_filename = session.get('rollup_filename', session_id)

        rollup_path = rollups_dir / f'{rollup_filename}.md'
        transcript_matches = list(transcripts_base.rglob(f'{session_id}.jsonl'))

        if rollup_path.exists() and transcript_matches:
            pairs.append((rollup_path, transcript_matches[0]))

    return pairs

def sample_sessions_resume(pairs, existing_sessions, target=70, seed=42):
    """Sample sessions stratified by size, skipping those already done."""
    random.seed(seed)

    small, large = [], []
    for rollup_path, transcript_path in pairs:
        session_id = rollup_path.stem
        if session_id in existing_sessions:
            continue

        size = transcript_path.stat().st_size
        if 100_000 <= size < 1_000_000:
            small.append((rollup_path, transcript_path))
        elif size >= 1_000_000:
            large.append((rollup_path, transcript_path))

    needed = target - len(existing_sessions)
    if needed <= 0:
        return []

    n_small = min(len(small), needed // 2)
    n_large = min(len(large), needed - n_small)

    sample = random.sample(small, n_small) + random.sample(large, n_large)
    random.shuffle(sample)

    log(f"Sampled {len(sample)} NEW sessions ({n_small} small, {n_large} large)")
    return sample

def transcript_to_text(transcript_path):
    """Convert transcript to text for question generation."""
    entries = []
    for line in transcript_path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except:
                pass

    parts = []
    for entry in entries:
        if entry.get('type') == 'user':
            msg = entry.get('message', {})
            content = msg.get('content', '')
            if isinstance(content, str):
                parts.append(f"USER: {content}")

        elif entry.get('type') == 'assistant':
            msg = entry.get('message', {})
            content = msg.get('content', [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            parts.append(f"ASSISTANT: {block.get('text', '')}")
                        elif block.get('type') == 'thinking':
                            parts.append(f"THINKING: {block.get('thinking', '')}")

    return '\n\n'.join(parts)

def extract_json_robust(text):
    """Try multiple strategies to extract JSON from response."""
    # Strategy 1: Look for ```json block
    if '```json' in text:
        try:
            json_text = text.split('```json')[1].split('```')[0]
            return json.loads(json_text.strip())
        except:
            pass

    # Strategy 2: Look for any ``` block
    if '```' in text:
        try:
            json_text = text.split('```')[1].split('```')[0]
            return json.loads(json_text.strip())
        except:
            pass

    # Strategy 3: Look for [ ... ] array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    # Strategy 4: Try parsing the whole thing
    try:
        return json.loads(text.strip())
    except:
        pass

    return None

def generate_questions(transcript_text, session_id):
    """Generate 5 questions from transcript with improved robustness."""
    # Use a simpler, more explicit prompt
    prompt = f"""Generate 5 questions about this session transcript. Return ONLY a JSON array.

For each question, specify:
- class: one of DECISION, DEAD_END, OUTCOME, ARTIFACT_DETAIL, SEQUENCE
- question: the question text
- answer: ground truth answer from the transcript

DECISION = why a choice was made
DEAD_END = what failed and why
OUTCOME = final result
ARTIFACT_DETAIL = specific path/value/number
SEQUENCE = order of steps

Transcript (first 25000 chars):
{transcript_text[:25000]}

Return JSON array only:
[{{"class": "DECISION", "question": "...", "answer": "..."}}, ...]"""

    response = call_llm(prompt, timeout=180)
    if not response:
        log("  No response from LLM")
        return []

    questions = extract_json_robust(response)
    if questions and isinstance(questions, list):
        log(f"  Extracted {len(questions)} questions")
        return questions
    else:
        log(f"  JSON parse failed. Response preview: {response[:200]}")
        return []

def judge_question(question_text, rollup_content):
    """Judge if question is answerable from rollup."""
    prompt = f"""Answer this question from the note, or say you can't.

QUESTION: {question_text}

NOTE:
{rollup_content[:15000]}

Format:
ANSWER: [your answer or "Cannot answer"]
RATING: ANSWERED or PARTIALLY_ANSWERED or NOT_ANSWERABLE"""

    response = call_llm(prompt, timeout=120)
    if not response:
        return "", "NOT_ANSWERABLE"

    lines = response.strip().split('\n')
    rating = "NOT_ANSWERABLE"
    answer_lines = []

    for line in lines:
        if line.strip().startswith("RATING:"):
            rating_text = line.split(":", 1)[1].strip().upper()
            if "NOT" in rating_text:
                rating = "NOT_ANSWERABLE"
            elif "PARTIALLY" in rating_text:
                rating = "PARTIALLY_ANSWERED"
            elif "ANSWERED" in rating_text:
                rating = "ANSWERED"
        elif not line.strip().startswith("RATING:"):
            if line.strip().startswith("ANSWER:"):
                answer_lines.append(line.split(":", 1)[1].strip())
            elif answer_lines or line.strip():
                answer_lines.append(line)

    answer = '\n'.join(answer_lines).strip()
    return answer, rating

def grade_answer(judge_answer, ground_truth, rating, question=""):
    """Grade via the shared grading_common module. Returns the row fragment:
    raises GradeError (never returns `absent`) when the grader call fails."""
    return grade_with(lambda prompt: call_llm(prompt, timeout=120),
                      question, ground_truth, judge_answer, rating)

def main():
    repo_root = Path(__file__).resolve().parents[2]
    questions_path = repo_root / "data" / "paper-rollups" / "rq3_questions.jsonl"

    log("=" * 60)
    log("RQ3 Question Retention (RESUME V2)")
    log("=" * 60)

    # Load existing
    existing_sessions, existing_rows = load_existing_sessions(questions_path)

    # Load all pairs
    all_pairs = load_pairs()
    log(f"Loaded {len(all_pairs)} session pairs")

    # Sample new sessions
    new_sample = sample_sessions_resume(all_pairs, existing_sessions, target=70, seed=42)

    if not new_sample:
        log("\nNo new sessions to process.")
        return 0

    # Open file in append mode
    grade_failures = FailureCounter("resume_v2 grader")
    with questions_path.open('a') as questions_file:
        processed = 0
        for i, (rollup_path, transcript_path) in enumerate(new_sample):
            # Check deadline
            elapsed_min = (time.time() - START_TIME) / 60
            if elapsed_min >= DEADLINE_MINUTES:
                log(f"\nDEADLINE at {elapsed_min:.1f}min. Stopping.")
                break

            session_id = rollup_path.stem
            total = len(existing_sessions) + processed + 1
            log(f"\n[{total}] {session_id} ({i+1}/{len(new_sample)}, {elapsed_min:.1f}min)")

            transcript_text = transcript_to_text(transcript_path)
            log(f"  Transcript: {len(transcript_text):,} chars")

            questions = generate_questions(transcript_text, session_id)
            if not questions:
                log("  SKIP: No questions")
                continue

            rollup_content = rollup_path.read_text()

            for q in questions:
                q_class = q.get('class', '').lower()
                question_text = q.get('question', '')
                ground_truth = q.get('answer', '')

                if not question_text:
                    continue

                log(f"    {q_class}...", end='', flush=True)

                judge_answer, rating = judge_question(question_text, rollup_content)
                try:
                    graded = grade_answer(judge_answer, ground_truth, rating,
                                          question_text)
                except GradeError:
                    # Not written, so a rerun retries the question.
                    grade_failures.record(GRADE_ERROR)
                    log(" grade_error (not recorded)")
                    grade_failures.check()
                    continue
                grade = graded['grade']
                if graded['grader_response'] is not None:
                    grade_failures.record(grade)

                record = {
                    'session_id': session_id,
                    'class': q_class,
                    'question': question_text,
                    'ground_truth': ground_truth,
                    'judge_answer': judge_answer,
                    'rating': rating,
                    'grade': grade,
                    'contradictory': graded['contradictory'],
                    'grader_response': graded['grader_response'],
                    'prompt_version': graded['prompt_version'],
                    'parser_version': graded['parser_version'],
                }

                questions_file.write(json.dumps(record) + '\n')
                questions_file.flush()
                os.fsync(questions_file.fileno())

                log(f" {grade}")

            processed += 1

    grade_failures.check(final=True)
    log("\n" + "=" * 60)
    log(f"Processed {processed} new sessions")
    log("=" * 60)
    return 0

if __name__ == '__main__':
    sys.exit(main())
