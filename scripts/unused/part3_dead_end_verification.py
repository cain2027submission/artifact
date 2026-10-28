#!/usr/bin/env python3
"""Part 3: Dead end hallucination check - verify claims against transcripts."""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR, SESSIONS_DIR

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
JUDGE_MODEL = "claude-sonnet-5"


def call_llm(prompt: str, model: str) -> str:
    """Call LLM via gateway. No temperature for Claude 5."""
    url = f"{GATEWAY}/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": 1024,  # Must be >= 512 for adaptive thinking
        "messages": [{"role": "user", "content": prompt}],
    }
    # NEVER send temperature - gateway rejects it

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read())
            content = data['choices'][0]['message']['content']
            return content.strip() if content else ""
    except Exception as e:
        print(f"  ERROR calling {model}: {e}")
        return ""


def extract_dead_ends_from_rollup(content: str) -> List[str]:
    """Extract dead end claims from ## Thinking & dead ends section."""
    dead_ends = []
    in_section = False

    for line in content.split('\n'):
        if re.match(r'^##\s+Thinking.*dead end', line, re.IGNORECASE):
            in_section = True
            continue
        elif line.startswith('## ') and in_section:
            break
        elif in_section and line.strip():
            # Extract bullet points or sentences
            if line.strip().startswith(('-', '*', '•')):
                dead_ends.append(line.strip()[1:].strip())
            elif len(line.strip()) > 20:  # Likely a sentence
                dead_ends.append(line.strip())

    return dead_ends


def load_transcript_text(path: Path) -> str:
    """Load transcript as readable text."""
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))

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
                    if isinstance(block, dict) and block.get('type') == 'text':
                        parts.append(f"USER: {block.get('text', '')}")

        elif role == 'assistant':
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    if block.get('type') == 'text':
                        parts.append(f"ASSISTANT: {block.get('text', '')}")
                    elif block.get('type') == 'thinking':
                        parts.append(f"THINKING: {block.get('thinking', '')}")

    return '\n\n'.join(parts)


def verify_dead_end_claim(claim: str, transcript_text: str) -> str:
    """Verify one dead end claim against transcript. Returns: supported, unsupported, contradicted."""

    prompt = f"""A summary note claims this dead end occurred during a session:

CLAIM: {claim}

Below is the raw session transcript. Verify if this claim is:
- SUPPORTED: The transcript shows this was tried and failed
- UNSUPPORTED: The transcript does not mention this
- CONTRADICTED: The transcript shows this actually succeeded or never happened

TRANSCRIPT (excerpt):
{transcript_text[:20000]}

Respond with one word: SUPPORTED, UNSUPPORTED, or CONTRADICTED"""

    response = call_llm(prompt, JUDGE_MODEL)

    if not response:
        return "unsupported"

    response_upper = response.strip().upper()
    for verdict in ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]:
        if verdict in response_upper:
            return verdict.lower()

    return "unsupported"


def extract_dead_ends_from_transcript(transcript_text: str) -> List[str]:
    """Estimate dead ends from transcript for recall. This is judgement-heavy."""

    prompt = f"""Below is a raw agent session transcript. Identify dead ends: things that were tried and failed, approaches that didn't work, or paths that were abandoned.

List each dead end as a brief bullet point.

TRANSCRIPT (excerpt):
{transcript_text[:25000]}

List the dead ends now, one per line starting with "-":"""

    response = call_llm(prompt, JUDGE_MODEL)

    if not response:
        return []

    dead_ends = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith(('-', '*', '•')):
            dead_ends.append(line[1:].strip())

    return dead_ends


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # Load Part 2 samples
    questions_path = repo_root / "data" / "paper-rollups" / "rq3_questions.jsonl"

    if not questions_path.exists():
        print("ERROR: Run part2_question_retention.py first")
        return

    # Get session IDs from Part 2
    session_ids = set()
    for line in questions_path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            session_ids.add(record['session_id'])

    print(f"Verifying dead ends for {len(session_ids)} sessions from Part 2")

    rollups_dir = SESSIONS_DIR
    transcripts_base = PROJECTS_DIR

    precision_results = []
    recall_results = []

    for i, session_id in enumerate(session_ids):
        print(f"\n[{i+1}/{len(session_ids)}] Verifying {session_id}")

        rollup_path = rollups_dir / f"{session_id}.md"
        transcript_matches = list(transcripts_base.rglob(f'{session_id}.jsonl'))

        if not rollup_path.exists() or not transcript_matches:
            continue

        rollup_content = rollup_path.read_text()
        transcript_text = load_transcript_text(transcript_matches[0])

        # Extract dead ends from rollup
        rollup_dead_ends = extract_dead_ends_from_rollup(rollup_content)
        print(f"  Found {len(rollup_dead_ends)} dead end claims in rollup")

        if not rollup_dead_ends:
            continue

        # PRECISION: Verify each rollup claim
        precision_record = {
            'session_id': session_id,
            'claims': [],
            'supported': 0,
            'unsupported': 0,
            'contradicted': 0,
        }

        for claim in rollup_dead_ends[:10]:  # Limit to avoid excessive API calls
            print(f"    Verifying: {claim[:80]}...")
            verdict = verify_dead_end_claim(claim, transcript_text)

            precision_record['claims'].append({
                'claim': claim,
                'verdict': verdict,
            })

            precision_record[verdict] += 1

        precision_results.append(precision_record)

        # RECALL: Extract dead ends from transcript and check coverage
        print(f"  Extracting dead ends from transcript for recall estimate...")
        transcript_dead_ends = extract_dead_ends_from_transcript(transcript_text)
        print(f"  Found {len(transcript_dead_ends)} dead ends in transcript")

        # Rough recall: how many transcript dead ends are mentioned in rollup?
        covered = 0
        for t_dead_end in transcript_dead_ends[:15]:  # Limit
            # Simple text overlap check (not perfect, but bounded)
            if any(word in rollup_content.lower() for word in t_dead_end.lower().split() if len(word) > 4):
                covered += 1

        recall_results.append({
            'session_id': session_id,
            'transcript_dead_ends': len(transcript_dead_ends),
            'rollup_dead_ends': len(rollup_dead_ends),
            'rough_covered': covered,
        })

    # Compute aggregates
    total_claims = 0
    total_supported = 0
    total_unsupported = 0
    total_contradicted = 0

    for result in precision_results:
        total_claims += len(result['claims'])
        total_supported += result['supported']
        total_unsupported += result['unsupported']
        total_contradicted += result['contradicted']

    precision = total_supported / total_claims if total_claims > 0 else 0

    total_transcript_dead_ends = sum(r['transcript_dead_ends'] for r in recall_results)
    total_rough_covered = sum(r['rough_covered'] for r in recall_results)
    rough_recall = total_rough_covered / total_transcript_dead_ends if total_transcript_dead_ends > 0 else 0

    # Merge with Part 2 summary
    summary_path = repo_root / "data" / "paper-rollups" / "rq3_retention_summary.json"
    summary = json.loads(summary_path.read_text())

    summary['dead_end_precision'] = {
        'total_claims': total_claims,
        'supported': total_supported,
        'unsupported': total_unsupported,
        'contradicted': total_contradicted,
        'precision': precision,
        'hallucination_rate': total_contradicted / total_claims if total_claims > 0 else 0,
    }

    summary['dead_end_recall'] = {
        'note': 'Recall is bounded estimate only - enumerating all dead ends is judgement-heavy',
        'method': 'LLM extraction from transcript + keyword overlap check',
        'total_transcript_dead_ends': total_transcript_dead_ends,
        'rough_covered': total_rough_covered,
        'rough_recall': rough_recall,
    }

    summary['precision_details'] = precision_results
    summary['recall_details'] = recall_results

    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n\nPart 3 Complete")
    print(f"  Sessions analyzed: {len(session_ids)}")
    print(f"\nDead End Precision:")
    print(f"  Total claims verified: {total_claims}")
    print(f"  Supported: {total_supported} ({total_supported/total_claims:.1%})")
    print(f"  Unsupported: {total_unsupported} ({total_unsupported/total_claims:.1%})")
    print(f"  Contradicted (HALLUCINATED): {total_contradicted} ({total_contradicted/total_claims:.1%})")
    print(f"  Precision: {precision:.1%}")
    print(f"\nDead End Recall (bounded estimate):")
    print(f"  Method: LLM extraction + keyword overlap")
    print(f"  Rough recall: {rough_recall:.1%} ({total_rough_covered}/{total_transcript_dead_ends})")
    print(f"  NOTE: This is a weak measure - enumerating all true dead ends is judgement-heavy")

    print(f"\nResults saved to {summary_path}")


if __name__ == '__main__':
    main()
