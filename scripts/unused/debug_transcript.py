#!/usr/bin/env python3
"""Debug transcript format to understand where tool results are."""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR, PRIMARY_CHECKOUT, project_slug as derive_project_slug

# Find a sample transcript (update session ID as needed)
# Project slug derived from repository directory name
project_slug = derive_project_slug(PRIMARY_CHECKOUT)
transcript_path = PROJECTS_DIR / project_slug / "e8f53eb5-5e7b-4e09-b905-ba639d7e9547.jsonl"

print(f"Analyzing: {transcript_path.name}\n")

# Count entry types
entry_types = Counter()
user_content_types = Counter()
assistant_content_types = Counter()

for line in transcript_path.read_text().splitlines():
    if not line.strip():
        continue

    entry = json.loads(line)
    entry_type = entry.get('type')
    entry_types[entry_type] += 1

    # Check user messages
    if entry_type == 'user':
        msg = entry.get('message', {})
        content = msg.get('content', '')

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    user_content_types[block.get('type')] += 1

    # Check assistant messages
    elif entry_type == 'assistant':
        msg = entry.get('message', {})
        content = msg.get('content', [])

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    assistant_content_types[block.get('type')] += 1

print("Entry types:")
for t, count in entry_types.most_common():
    print(f"  {t}: {count}")

print("\nUser message content block types:")
for t, count in user_content_types.most_common():
    print(f"  {t}: {count}")

print("\nAssistant message content block types:")
for t, count in assistant_content_types.most_common():
    print(f"  {t}: {count}")

# Show a sample user message with tool_result
print("\n\nSample user message with tool_result:")
for line in transcript_path.read_text().splitlines():
    if not line.strip():
        continue

    entry = json.loads(line)
    if entry.get('type') == 'user':
        msg = entry.get('message', {})
        content = msg.get('content', [])

        if isinstance(content, list):
            has_tool_result = any(b.get('type') == 'tool_result' for b in content if isinstance(b, dict))
            if has_tool_result:
                print(json.dumps(entry, indent=2)[:2000])
                break
