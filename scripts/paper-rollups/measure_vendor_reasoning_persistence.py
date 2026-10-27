#!/usr/bin/env python3
"""Count empty vs non-empty persisted reasoning in Antigravity CLI transcripts.

Mirrors what the paper measures over Claude Code: for every record that carries a
reasoning/thinking field at all, is the serialized form empty?
"""
import json
import pathlib
import collections

root = pathlib.Path.home() / ".gemini" / "antigravity-cli" / "brain"
files = sorted(root.glob("*/.system_generated/logs/transcript.jsonl"))

CANDIDATES = ("thinking", "reasoning", "reasoning_content", "thought",
              "thoughts", "redacted_thinking", "encrypted_reasoning")

field_present = collections.Counter()
field_empty = collections.Counter()
lens = []
bad_lines = 0
records = 0


def walk(obj):
    """Yield every dict nested anywhere in obj."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


for f in files:
    try:
        text = f.read_text(errors="replace")
    except OSError:
        continue
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        records += 1
        for d in walk(rec):
            for key in CANDIDATES:
                if key in d:
                    val = d[key]
                    if isinstance(val, bool):
                        continue  # a flag, not content
                    field_present[key] += 1
                    s = val if isinstance(val, str) else json.dumps(val)
                    if not s or not s.strip() or s in ("null", '""', "[]", "{}"):
                        field_empty[key] += 1
                    else:
                        lens.append(len(s))

print(f"files            {len(files)}")
print(f"json records     {records}")
print(f"unparseable      {bad_lines}")
print()
for key in sorted(field_present, key=lambda k: -field_present[k]):
    present = field_present[key]
    empty = field_empty[key]
    pct = 100.0 * empty / present if present else 0.0
    print(f"{key:22s} present {present:7d}   empty {empty:7d}   {pct:6.2f}% empty")

if lens:
    lens.sort()
    n = len(lens)
    print()
    print(f"non-empty blocks  {n}")
    print(f"median chars      {lens[n // 2]}")
    print(f"mean chars        {sum(lens) / n:.0f}")
    print(f"total chars       {sum(lens)}")
if not field_present:
    print("NO reasoning-like field found under any candidate name")
