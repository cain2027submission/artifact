#!/usr/bin/env python3
"""Part 2 streamlined: Question retention with progress reporting."""

import json
import os
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR, SESSIONS_DIR

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-sonnet-5"

def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)

def call_llm(prompt: str, timeout=120) -> str:
    """Call LLM via gateway."""
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

def sample_sessions(pairs, n=30, seed=42):
    """Sample sessions stratified by size."""
    random.seed(seed)

    small, large = [], []
    for rollup_path, transcript_path in pairs:
        size = transcript_path.stat().st_size
        if 100_000 <= size < 1_000_000:
            small.append((rollup_path, transcript_path))
        elif size >= 1_000_000:
            large.append((rollup_path, transcript_path))

    n_small = min(len(small), n // 2)
    n_large = min(len(large), n - n_small)

    sample = random.sample(small, n_small) + random.sample(large, n_large)
    random.shuffle(sample)

    log(f"Sampled {len(sample)} sessions ({n_small} small, {n_large} large)")
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

def main():
    seed = int(os.environ.get('SEED', '42'))
    n_sessions = int(os.environ.get('N', '30'))  # Default to 30 for faster run

    repo_root = Path(__file__).resolve().parents[2]

    log(f"Part 2 Question Retention (seed={seed}, n={n_sessions})")
    log("=" * 60)

    pairs = load_pairs()
    log(f"Loaded {len(pairs)} session pairs")

    sample = sample_sessions(pairs, n=n_sessions, seed=seed)

    questions_output = repo_root / "data" / "paper-rollups" / "rq3_questions.jsonl"
    questions_file = questions_output.open('w')

    counts = {
        'decision': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'dead_end': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'outcome': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'artifact_detail': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
        'sequence': {'correct': 0, 'partially_correct': 0, 'wrong': 0, 'absent': 0},
    }

    for i, (rollup_path, transcript_path) in enumerate(sample):
        session_id = rollup_path.stem
        log(f"\n[{i+1}/{len(sample)}] {session_id}")

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

            questions_file.write(json.dumps(record) + '\n')
            questions_file.flush()

            if q_class in counts:
                counts[q_class][grade] += 1

            log(f"      -> {grade}")

    questions_file.close()

    # Compute summary
    total_questions = sum(sum(c.values()) for c in counts.values())
    total_wrong = sum(c['wrong'] for c in counts.values())

    summary = {
        'n_sessions': len(sample),
        'retention_by_class': {},
        'hallucination_rate': total_wrong / total_questions if total_questions > 0 else 0,
    }

    for q_class, class_counts in counts.items():
        total = sum(class_counts.values())
        if total > 0:
            summary['retention_by_class'][q_class] = {
                'total': total,
                **class_counts,
                'correct_rate': class_counts['correct'] / total,
                'wrong_rate': class_counts['wrong'] / total,
                'absent_rate': class_counts['absent'] / total,
            }

    summary_path = repo_root / "data" / "paper-rollups" / "rq3_retention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log("\n" + "=" * 60)
    log(f"Part 2 Complete: {total_questions} questions from {len(sample)} sessions")
    log(f"Hallucination rate: {summary['hallucination_rate']:.1%}\n")

    for q_class, stats in summary['retention_by_class'].items():
        log(f"{q_class:15} correct={stats['correct_rate']:.1%} wrong={stats['wrong_rate']:.1%} absent={stats['absent_rate']:.1%} (n={stats['total']})")

    log(f"\nResults: {questions_output}")

if __name__ == '__main__':
    main()
