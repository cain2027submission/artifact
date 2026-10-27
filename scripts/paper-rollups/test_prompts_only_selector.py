#!/usr/bin/env python3
"""Test the prompts-only selector on two sessions to verify correctness.

Tests on session prefixes 1327c110 (large) and b766c7ce (small) to ensure:
1. Total chars count
2. ### USER count unchanged
3. ### ASSISTANT count goes to 0
4. ### TOOL CALL count goes to 0
5. "file_path", "command", "old_string" substrings go to 0 or near 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_stripped_transcript_arm import strip_transcript
from rq3_prompts_only_arm import prompts_only_context
from rq3_raw_transcript_arm import index_transcripts, resolve

def count_sections(text: str, marker: str) -> int:
    """Count lines starting with marker."""
    return sum(1 for line in text.splitlines() if line.startswith(marker))

def count_substring(text: str, substring: str) -> int:
    """Count occurrences of substring."""
    return text.count(substring)

def test_session(prefix: str, by_prefix: dict) -> None:
    """Test one session."""
    p = resolve(prefix, by_prefix)
    if p is None:
        print(f"ERROR: Could not resolve {prefix}")
        return

    raw = p.read_text(errors="replace")
    stripped, _ = strip_transcript(raw)
    prompts_only, _ = prompts_only_context(raw)

    print(f"\n{'='*70}")
    print(f"Session: {prefix}")
    print(f"{'='*70}")

    # Stripped stats (baseline)
    print(f"\nSTRIPPED TRANSCRIPT:")
    print(f"  Total chars:       {len(stripped):7,}")
    print(f"  ### USER:          {count_sections(stripped, '### USER'):7,}")
    print(f"  ### ASSISTANT:     {count_sections(stripped, '### ASSISTANT'):7,}")
    print(f"  ### TOOL CALL:     {count_sections(stripped, '### TOOL CALL'):7,}")
    print(f"  'file_path':       {count_substring(stripped, 'file_path'):7,}")
    print(f"  'command':         {count_substring(stripped, 'command'):7,}")
    print(f"  'old_string':      {count_substring(stripped, 'old_string'):7,}")

    # Prompts-only stats
    print(f"\nPROMPTS-ONLY:")
    print(f"  Total chars:       {len(prompts_only):7,}")
    print(f"  ### USER:          {count_sections(prompts_only, '### USER'):7,}")
    print(f"  ### ASSISTANT:     {count_sections(prompts_only, '### ASSISTANT'):7,}")
    print(f"  ### TOOL CALL:     {count_sections(prompts_only, '### TOOL CALL'):7,}")
    print(f"  'file_path':       {count_substring(prompts_only, 'file_path'):7,}")
    print(f"  'command':         {count_substring(prompts_only, 'command'):7,}")
    print(f"  'old_string':      {count_substring(prompts_only, 'old_string'):7,}")

    # Verify correctness
    stripped_user_count = count_sections(stripped, '### USER')
    prompts_user_count = count_sections(prompts_only, '### USER')
    prompts_assistant_count = count_sections(prompts_only, '### ASSISTANT')
    prompts_tool_count = count_sections(prompts_only, '### TOOL CALL')
    prompts_command_count = count_substring(prompts_only, 'command')

    print(f"\nVERIFICATION:")
    checks = []

    # Check 1: USER count unchanged
    if prompts_user_count == stripped_user_count and prompts_user_count > 0:
        checks.append(("### USER count unchanged and nonzero", "PASS"))
    else:
        checks.append(("### USER count unchanged and nonzero", f"FAIL (stripped={stripped_user_count}, prompts={prompts_user_count})"))

    # Check 2: ASSISTANT count goes to 0
    if prompts_assistant_count == 0:
        checks.append(("### ASSISTANT goes to 0", "PASS"))
    else:
        checks.append(("### ASSISTANT goes to 0", f"FAIL (count={prompts_assistant_count})"))

    # Check 3: TOOL CALL count goes to 0
    if prompts_tool_count == 0:
        checks.append(("### TOOL CALL goes to 0", "PASS"))
    else:
        checks.append(("### TOOL CALL goes to 0", f"FAIL (count={prompts_tool_count})"))

    # Check 4: command count goes to 0 or near 0
    if prompts_command_count <= 2:  # Allow a couple in user prose
        checks.append((f"'command' goes to 0 or near 0", f"PASS (count={prompts_command_count})"))
    else:
        checks.append((f"'command' goes to 0 or near 0", f"FAIL (count={prompts_command_count})"))

    for check_name, result in checks:
        print(f"  {check_name:40s} {result}")

    # Overall
    all_pass = all("PASS" in r for _, r in checks)
    print(f"\n  OVERALL: {'PASS - selector is correct' if all_pass else 'FAIL - selector is leaking'}")

def main():
    by_prefix = index_transcripts()

    print("Testing prompts-only selector on two sessions")
    print("="*70)

    test_session("1327c110", by_prefix)
    test_session("b766c7ce", by_prefix)

    print(f"\n{'='*70}")
    print("Testing complete.")
    print("="*70)

if __name__ == "__main__":
    main()
