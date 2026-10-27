#!/usr/bin/env python3
"""Test Part 2 with just one session."""

import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR, SESSIONS_DIR

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-sonnet-5"

def call_llm(prompt: str) -> str:
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
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read())
            content = data['choices'][0]['message']['content']
            return content.strip() if content else ""
    except Exception as e:
        print(f"ERROR: {e}")
        return ""

# Load one session
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
        size = transcript_matches[0].stat().st_size
        if 500_000 <= size < 1_000_000:  # Mid-size
            pairs.append((rollup_path, transcript_matches[0]))
            if len(pairs) >= 1:
                break

if not pairs:
    print("No sessions found")
    exit(1)

rollup_path, transcript_path = pairs[0]
print(f"Testing with session: {rollup_path.stem}")
print(f"Transcript size: {transcript_path.stat().st_size:,} bytes")

# Load transcript
entries = []
for line in transcript_path.read_text().splitlines():
    if line.strip():
        entries.append(json.loads(line))

print(f"Transcript entries: {len(entries)}")

# Build simple transcript text
parts = []
for entry in entries[:100]:  # Limit for test
    if entry.get('type') == 'user':
        msg = entry.get('message', {})
        content = msg.get('content', '')
        if isinstance(content, str):
            parts.append(f"USER: {content[:200]}")

    elif entry.get('type') == 'assistant':
        msg = entry.get('message', {})
        content = msg.get('content', [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    parts.append(f"ASSISTANT: {block.get('text', '')[:200]}")

transcript_text = '\n\n'.join(parts)
print(f"Transcript text length: {len(transcript_text):,} chars")

# Test question generation
prompt = f"""Generate ONE question from this transcript excerpt. Return only the question, nothing else.

TRANSCRIPT:
{transcript_text[:5000]}

Generate ONE question now:"""

print("\nCalling LLM to generate question...")
response = call_llm(prompt)
print(f"Response: {response}")
