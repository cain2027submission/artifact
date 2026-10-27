#!/usr/bin/env python3
"""Resume RQ3 question generation from where part2_streamlined.py left off.

Reads existing records, skips those sessions, and extends the sample to 55-70 sessions.
Repairs the truncated tail before appending.
Writes atomically per record with fsync.
"""

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR, SESSIONS_DIR

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-sonnet-5"
START_TIME = time.time()
DEADLINE_MINUTES = 75

def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)

def call_llm(prompt: str, timeout=120) -> str:
    """Call LLM via gateway. NEVER send temperature field."""
    url = f"{GATEWAY}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "max_tokens": 512,
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

def repair_and_rewrite(questions_path, rows):
    """Rewrite file with only complete records."""
    log(f"Rewriting {questions_path} with {len(rows)} complete records...")
    with questions_path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
        f.flush()
        os.fsync(f.fileno())
    log("  Repaired.")

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
            continue  # Skip already-done sessions

        size = transcript_path.stat().st_size
        if 100_000 <= size < 1_000_000:
            small.append((rollup_path, transcript_path))
        elif size >= 1_000_000:
            large.append((rollup_path, transcript_path))

    # Calculate how many we need from each stratum
    needed = target - len(existing_sessions)
    if needed <= 0:
        log(f"Already have {len(existing_sessions)} sessions, target is {target}")
        return []

    n_small = min(len(small), needed // 2)
    n_large = min(len(large), needed - n_small)

    sample = random.sample(small, n_small) + random.sample(large, n_large)
    random.shuffle(sample)

    log(f"Sampled {len(sample)} NEW sessions ({n_small} small, {n_large} large)")
    log(f"Total will be {len(existing_sessions)} + {len(sample)} = {len(existing_sessions) + len(sample)}")
    return sample

def transcript_to_text(transcript_path):
    """Convert transcript to text for question generation."""
    entries = []
    for line in transcript_path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))

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

def generate_questions(transcript_text, session_id):
    """Generate 5 questions from transcript."""
    prompt = f"""From this transcript, generate EXACTLY 5 questions, one for each class below.

Classes:
1. DECISION - why one approach was chosen
2. DEAD_END - what failed and why
3. OUTCOME - final result
4. ARTIFACT_DETAIL - specific value (error, path, number)
5. SEQUENCE - order of actions

For each, provide the question and ground-truth answer. Format as JSON:
[{{"class":"DECISION","question":"...","answer":"..."}}, ...]

TRANSCRIPT (excerpt):
{transcript_text[:25000]}

JSON:"""

    response = call_llm(prompt)
    if not response:
        return []

    try:
        if '```json' in response:
            json_text = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_text = response.split('```')[1].split('```')[0]
        else:
            json_text = response

        questions = json.loads(json_text.strip())
        return questions if isinstance(questions, list) else []
    except:
        log(f"  Failed to parse questions JSON")
        return []

def judge_question(question_text, rollup_content):
    """Judge if question is answerable from rollup."""
    prompt = f"""Can you answer this question from the note below?

QUESTION: {question_text}

NOTE:
{rollup_content}

Answer the question if possible, then rate yourself:
ANSWERED / PARTIALLY_ANSWERED / NOT_ANSWERABLE

Format:
[your answer]

RATING: [rating]"""

    response = call_llm(prompt)
    if not response:
        return "", "NOT_ANSWERABLE"

    lines = response.strip().split('\n')
    rating = "NOT_ANSWERABLE"
    for line in reversed(lines):
        if "ANSWERED" in line.upper():
            if "NOT" in line.upper():
                rating = "NOT_ANSWERABLE"
            elif "PARTIALLY" in line.upper():
                rating = "PARTIALLY_ANSWERED"
            else:
                rating = "ANSWERED"
            break

    answer = '\n'.join([l for l in lines if 'RATING:' not in l]).strip()
    return answer, rating

def grade_answer(judge_answer, ground_truth, rating):
    """Grade judge answer. Returns: correct, partially_correct, wrong, absent."""
    if rating == "NOT_ANSWERABLE" or not judge_answer:
        return "absent"

    prompt = f"""Grade this answer against ground truth.

GROUND TRUTH:
{ground_truth}

ANSWER:
{judge_answer}

Grade as ONE WORD: CORRECT / PARTIALLY_CORRECT / WRONG / ABSENT"""

    response = call_llm(prompt)
    response_upper = response.strip().upper()

    for grade in ["PARTIALLY_CORRECT", "CORRECT", "WRONG", "ABSENT"]:
        if grade in response_upper:
            return grade.lower()

    return "absent"

def test_gateway():
    """Test that gateway is responsive."""
    log("Testing gateway...")
    response = call_llm("Say OK", timeout=30)
    if response:
        log(f"  Gateway OK: '{response[:50]}'")
        return True
    else:
        log("  ERROR: Gateway did not respond")
        return False

def main():
    repo_root = Path(__file__).resolve().parents[2]
    questions_path = repo_root / "data" / "paper-rollups" / "rq3_questions.jsonl"

    log("=" * 60)
    log("RQ3 Question Retention (RESUME)")
    log("=" * 60)

    # Test gateway first
    if not test_gateway():
        log("\nERROR: Gateway not responding. Stopping.")
        return 1

    # Load existing and repair
    existing_sessions, existing_rows = load_existing_sessions(questions_path)
    repair_and_rewrite(questions_path, existing_rows)

    # Load all pairs
    all_pairs = load_pairs()
    log(f"Loaded {len(all_pairs)} session pairs from structural loss data")

    # Sample new sessions to add
    new_sample = sample_sessions_resume(all_pairs, existing_sessions, target=70, seed=42)

    if not new_sample:
        log("\nNo new sessions to process.")
        return 0

    # Open file in append mode
    with questions_path.open('a') as questions_file:
        processed = 0
        for i, (rollup_path, transcript_path) in enumerate(new_sample):
            # Check deadline
            elapsed_min = (time.time() - START_TIME) / 60
            if elapsed_min >= DEADLINE_MINUTES:
                log(f"\nDEADLINE reached at {elapsed_min:.1f} minutes. Stopping.")
                break

            session_id = rollup_path.stem
            total_sessions = len(existing_sessions) + i + 1
            log(f"\n[{total_sessions}] {session_id} (new {i+1}/{len(new_sample)}, {elapsed_min:.1f}min elapsed)")

            # Load transcript
            log("  Loading transcript...")
            transcript_text = transcript_to_text(transcript_path)
            log(f"  Transcript: {len(transcript_text):,} chars")

            # Generate questions
            log("  Generating questions...")
            questions = generate_questions(transcript_text, session_id)
            if not questions:
                log("  SKIP: No questions generated")
                continue

            log(f"  Generated {len(questions)} questions")

            # Load rollup
            rollup_content = rollup_path.read_text()

            # Judge each question
            for q in questions:
                q_class = q.get('class', '').lower()
                question_text = q.get('question', '')
                ground_truth = q.get('answer', '')

                if not question_text:
                    continue

                log(f"    Judging {q_class}...")

                judge_answer, rating = judge_question(question_text, rollup_content)
                grade = grade_answer(judge_answer, ground_truth, rating)

                record = {
                    'session_id': session_id,
                    'class': q_class,
                    'question': question_text,
                    'ground_truth': ground_truth,
                    'judge_answer': judge_answer,
                    'rating': rating,
                    'grade': grade,
                }

                # Atomic write with fsync
                questions_file.write(json.dumps(record) + '\n')
                questions_file.flush()
                os.fsync(questions_file.fileno())

                log(f"      -> {grade}")

            processed += 1

    log("\n" + "=" * 60)
    log(f"Resume complete: processed {processed} new sessions")
    log(f"Total sessions now: {len(existing_sessions) + processed}")
    log("=" * 60)
    return 0

if __name__ == '__main__':
    sys.exit(main())
