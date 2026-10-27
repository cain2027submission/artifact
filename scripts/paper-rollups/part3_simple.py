#!/usr/bin/env python3
"""Part 3: Dead-end precision - verify rollup claims against transcripts."""

import json
import os
import re
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

def call_llm(prompt: str) -> str:
    """Call LLM via gateway."""
    url = f"{GATEWAY}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

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
        log(f"  ERROR: {e}")
        return ""

def extract_dead_ends_from_rollup(content: str):
    """Extract dead end claims from rollup."""
    dead_ends = []

    # Look for "Thinking & dead ends" or "thinking and dead ends" section
    in_section = False
    for line in content.split('\n'):
        if re.search(r'(##|[\*]{2})\s*[Tt]hinking.*dead end', line, re.IGNORECASE):
            in_section = True
            continue
        elif (line.startswith('## ') or line.startswith('**')) and in_section:
            break
        elif in_section and line.strip():
            # Extract bullet points
            if line.strip().startswith(('-', '*', '•')):
                dead_ends.append(line.strip()[1:].strip())
            elif len(line.strip()) > 20:
                dead_ends.append(line.strip())

    return dead_ends

def transcript_to_text(transcript_path):
    """Load transcript as text."""
    entries = []
    for line in transcript_path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except:
                continue

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

def verify_dead_end(claim: str, transcript_text: str) -> str:
    """Verify dead end claim. Returns: supported, unsupported, contradicted."""
    prompt = f"""A rollup claims this dead end:

CLAIM: {claim}

Does the transcript support this?

TRANSCRIPT (excerpt):
{transcript_text[:20000]}

Answer ONE WORD: SUPPORTED / UNSUPPORTED / CONTRADICTED"""

    response = call_llm(prompt)
    response_upper = response.strip().upper()

    for verdict in ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]:
        if verdict in response_upper:
            return verdict.lower()

    return "unsupported"

def main():
    log("Part 3: Dead-End Precision")
    log("=" * 60)

    repo_root = Path(__file__).resolve().parents[2]

    # Get sessions from Part 2 via reading the questions file carefully
    questions_path = repo_root / "data" / "paper-rollups" / "rq3_questions.jsonl"

    if not questions_path.exists():
        log("ERROR: Run part2 first")
        return

    session_filenames = set()
    for line in questions_path.read_text().splitlines():
        if line.strip():
            try:
                record = json.loads(line)
                session_filenames.add(record['session_id'])
            except Exception as e:
                log(f"Skipping malformed line: {e}")
                continue

    log(f"Found {len(session_filenames)} unique sessions from Part 2\n")

    # Load session mapping from Part 1
    results_path = repo_root / "data" / "paper-rollups" / "rq3_structural_loss.json"
    structural = json.load(results_path.open())

    filename_to_id = {}
    for session in structural['sessions']:
        filename = session.get('rollup_filename', '')
        if filename:
            filename_to_id[filename] = session['session_id']

    rollups_dir = SESSIONS_DIR
    transcripts_base = PROJECTS_DIR

    precision_results = []

    for i, rollup_filename in enumerate(sorted(session_filenames)):
        session_id = filename_to_id.get(rollup_filename, rollup_filename)

        log(f"[{i+1}/{len(session_filenames)}] {rollup_filename}")

        rollup_path = rollups_dir / f"{rollup_filename}.md"
        transcript_matches = list(transcripts_base.rglob(f'{session_id}.jsonl'))

        if not rollup_path.exists() or not transcript_matches:
            log("  SKIP: Missing files")
            continue

        rollup_content = rollup_path.read_text()
        transcript_text = transcript_to_text(transcript_matches[0])

        # Extract dead ends
        dead_ends = extract_dead_ends_from_rollup(rollup_content)
        log(f"  Found {len(dead_ends)} dead-end claims")

        if not dead_ends:
            continue

        # Verify each (limit to 5 per session)
        verified = {'supported': 0, 'unsupported': 0, 'contradicted': 0}

        for claim in dead_ends[:5]:
            log(f"    Verifying: {claim[:60]}...")
            verdict = verify_dead_end(claim, transcript_text)
            verified[verdict] += 1
            log(f"      -> {verdict}")

        precision_results.append({
            'session_id': session_id,
            'rollup_filename': rollup_filename,
            'n_claims': len(dead_ends),
            **verified
        })

    # Aggregate
    total_claims = sum(r['supported'] + r['unsupported'] + r['contradicted'] for r in precision_results)
    total_supported = sum(r['supported'] for r in precision_results)
    total_unsupported = sum(r['unsupported'] for r in precision_results)
    total_contradicted = sum(r['contradicted'] for r in precision_results)

    precision = total_supported / total_claims if total_claims > 0 else 0

    # Update summary
    summary_path = repo_root / "data" / "paper-rollups" / "rq3_retention_summary.json"
    summary = json.load(summary_path.open())

    summary['dead_end_precision'] = {
        'total_claims_verified': total_claims,
        'supported': total_supported,
        'unsupported': total_unsupported,
        'contradicted': total_contradicted,
        'precision': precision,
        'hallucination_rate': total_contradicted / total_claims if total_claims > 0 else 0,
        'note': 'Limited to 5 claims per session to reduce API calls'
    }

    summary['precision_details'] = precision_results

    summary_path.write_text(json.dumps(summary, indent=2))

    log("\n" + "=" * 60)
    log(f"Part 3 Complete: {total_claims} dead-end claims verified")
    log(f"  Supported: {total_supported} ({total_supported/total_claims:.1%})")
    log(f"  Unsupported: {total_unsupported} ({total_unsupported/total_claims:.1%})")
    log(f"  Contradicted: {total_contradicted} ({total_contradicted/total_claims:.1%})")
    log(f"  Precision: {precision:.1%}")
    log(f"\nResults: {summary_path}")

if __name__ == '__main__':
    main()
