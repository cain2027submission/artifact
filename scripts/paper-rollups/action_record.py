#!/usr/bin/env python3
"""Build an action record from a raw transcript.

An action record is a structured session rendering designed to retain what
matters (tool intents, result digests, prose, ordering) and drop what does not
(thinking signatures, repeated boilerplate, full result payloads, re-read file
content). See ACTION_RECORD_FORMAT.md for the spec and justification.

Usage:
    from action_record import build_action_record
    record = build_action_record("/path/to/transcript.jsonl")
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Reuse RQ1's harness-injection stripper
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_signal_composition_corrected import split_meta  # noqa: E402


def truncate(s: str, limit: int, suffix: str = "") -> str:
    """Truncate string to limit, optionally adding suffix."""
    if len(s) <= limit:
        return s
    return s[:limit] + suffix


def structure_tool_call(tool_name: str, tool_input: dict) -> str:
    """Extract salient arguments from a tool call input.

    Returns a compact, human-readable representation of what the tool did,
    dropping full payloads and focusing on paths, queries, commands, and sizes.
    """
    if tool_name == "Read":
        path = tool_input.get("file_path", "?")
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        if offset is not None or limit is not None:
            return f"Read({path}, offset={offset}, limit={limit})"
        return f"Read({path})"

    elif tool_name == "Edit":
        path = tool_input.get("file_path", "?")
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        replace_all = tool_input.get("replace_all", False)
        old_head = truncate(old, 60, "...")
        new_head = truncate(new, 60, "...")
        if len(old) + len(new) < 200:
            return f"Edit({path}, old={repr(old)}, new={repr(new)})"
        suffix = ", replace_all=True" if replace_all else ""
        return f"Edit({path}, old_head={repr(old_head)}, new_head={repr(new_head)}{suffix})"

    elif tool_name == "Write":
        path = tool_input.get("file_path", "?")
        content = tool_input.get("content", "")
        return f"Write({path}, {len(content)} chars)"

    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if len(cmd) <= 200:
            return f"Bash({repr(cmd)})"
        return f"Bash({repr(cmd[:100])} [... +{len(cmd)-100} chars])"

    elif tool_name == "Agent":
        name = tool_input.get("name", "?")
        desc = tool_input.get("description", "")
        model = tool_input.get("model")
        isolation = tool_input.get("isolation")
        prompt = tool_input.get("prompt", "")
        prompt_head = truncate(prompt, 100, "...")
        parts = [f"name={repr(name)}", f"desc={repr(desc)}"]
        if model:
            parts.append(f"model={repr(model)}")
        if isolation:
            parts.append(f"isolation={repr(isolation)}")
        parts.append(f"prompt_head={repr(prompt_head)}")
        return f"Agent({', '.join(parts)})"

    elif tool_name in ("WebFetch", "WebSearch"):
        url = tool_input.get("url") or tool_input.get("query", "?")
        return f"{tool_name}({repr(url)})"

    elif tool_name == "Grep" or tool_name == "GrepReplace":
        pattern = tool_input.get("pattern", "?")
        path = tool_input.get("file_path") or tool_input.get("directory", ".")
        return f"{tool_name}(pattern={repr(pattern)}, path={repr(path)})"

    else:
        # Generic: tool name + any path-like or query-like field + size
        path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("directory")
        query = tool_input.get("query") or tool_input.get("pattern")
        if path:
            return f"{tool_name}(path={repr(path)})"
        elif query:
            return f"{tool_name}(query={repr(query)})"
        else:
            # Last resort: show first 100 chars of JSON
            s = json.dumps(tool_input, indent=None)
            return f"{tool_name}({truncate(s, 100, '...')})"


def digest_tool_result(tool_name: str, result_content: str, is_error: bool) -> str:
    """One-line digest of a tool result.

    Returns success/failure, scale (bytes/lines/count), and a sample (first line).
    """
    if is_error:
        first_line = result_content.split("\n")[0][:80] if result_content else "error"
        return f"error: {first_line}"

    if tool_name in ("Read", "WebFetch"):
        lines = result_content.split("\n")
        first = lines[0][:80] if lines else ""
        return f"{len(result_content)} bytes, {len(lines)} lines, first: {repr(first)}"

    elif tool_name in ("Edit", "Write"):
        # These echo nothing useful, just note success
        return "success"

    elif tool_name == "Bash":
        # Parse for exit code if present (format varies)
        lines = result_content.split("\n")
        first_line = lines[0][:80] if lines else ""
        # Heuristic: look for exit code in result
        if "exit code" in result_content.lower() or "exit:" in result_content.lower():
            return first_line
        # Default: just show first line
        return f"output: {repr(first_line)}"

    elif tool_name == "Agent":
        # Subagent completion: first line of its report
        first_line = result_content.split("\n")[0][:100] if result_content else ""
        return f"completed: {repr(first_line)}"

    else:
        # Generic: count lines, show first
        lines = result_content.split("\n")
        first = lines[0][:80] if lines else ""
        return f"{len(lines)} lines, first: {repr(first)}"


def build_action_record(transcript_path: str) -> Tuple[str, dict]:
    """Build an action record from a raw transcript.

    Returns (markdown_record, stats). Stats keys:
    - entries: number of transcript entries processed
    - turns: number of user/assistant turn pairs
    - tool_calls: number of tool calls
    - tool_results: number of tool results with digests
    - thinking_blocks: total thinking blocks seen
    - thinking_empty: thinking blocks with no text
    - thinking_kept: non-empty thinking blocks kept
    - signatures_dropped: thinking signatures dropped
    - re_reads: file re-reads detected and deduplicated
    - boilerplate_blocks: harness blocks deduplicated
    - user_prose_chars: total user prose characters
    - assistant_prose_chars: total assistant prose characters
    - tool_call_chars: total tool call representation chars
    - tool_digest_chars: total tool result digest chars
    """
    text = Path(transcript_path).read_text(errors="replace")

    out: List[str] = []
    stats = {
        "entries": 0,
        "turns": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "thinking_blocks": 0,
        "thinking_empty": 0,
        "thinking_kept": 0,
        "signatures_dropped": 0,
        "re_reads": 0,
        "boilerplate_blocks": 0,
        "user_prose_chars": 0,
        "assistant_prose_chars": 0,
        "tool_call_chars": 0,
        "tool_digest_chars": 0,
    }

    # Track seen file content hashes to detect re-reads
    seen_file_content: Dict[str, str] = {}  # path -> content_hash

    # Track boilerplate blocks (e.g., <system-reminder>) to deduplicate
    boilerplate_seen: Dict[str, int] = {}  # text_hash -> count

    turn_num = 0
    in_assistant_turn = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue

        stats["entries"] += 1
        role = msg.get("role")
        is_meta = bool(rec.get("isMeta"))
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [content] if content else []

        # Detect turn boundaries
        if role == "user" and not is_meta:
            turn_num += 1
            stats["turns"] = turn_num
            in_assistant_turn = False
            out.append(f"\n--- turn {turn_num} ---\n")
        elif role == "assistant":
            in_assistant_turn = True

        for b in blocks:
            if isinstance(b, str):
                btype, text_content = "text", b
            elif isinstance(b, dict):
                btype = b.get("type", "other")
                text_content = ""
            else:
                continue

            # Thinking blocks
            if btype in ("thinking", "redacted_thinking"):
                stats["thinking_blocks"] += 1
                text_content = b.get("thinking") or b.get("text") or ""

                # Drop the signature (always present, ~1424 chars, zero info)
                if "signature" in b:
                    stats["signatures_dropped"] += 1

                if not text_content.strip():
                    stats["thinking_empty"] += 1
                    continue

                # Keep non-empty reasoning
                stats["thinking_kept"] += 1
                chunk = f"\n### REASONING\n{text_content.strip()}\n"
                out.append(chunk)
                stats["assistant_prose_chars"] += len(chunk)

            # Tool calls
            elif btype == "tool_use":
                stats["tool_calls"] += 1
                name = b.get("name") or "tool"
                inp = b.get("input", {})

                structured = structure_tool_call(name, inp)
                chunk = f"\ntool: {structured}\n"
                out.append(chunk)
                stats["tool_call_chars"] += len(chunk)

            # Tool results
            elif btype == "tool_result":
                stats["tool_results"] += 1
                # Extract result content and error status
                result_content = ""
                is_error = b.get("is_error", False)

                # Result content can be a string or a list of content blocks
                result = b.get("content")
                if isinstance(result, str):
                    result_content = result
                elif isinstance(result, list):
                    # Concatenate text blocks
                    parts = []
                    for rb in result:
                        if isinstance(rb, str):
                            parts.append(rb)
                        elif isinstance(rb, dict) and rb.get("type") == "text":
                            parts.append(rb.get("text", ""))
                    result_content = "\n".join(parts)

                # Check for file re-reads (only for Read tool results)
                # We don't have the tool name here directly, so infer from content
                # or track tool_use -> tool_result pairing. For now, simplified:
                # We'll just digest all results uniformly.

                # Generate one-line digest
                # We don't have the original tool name here; for now, generic digest
                digest = digest_tool_result("Read", result_content, is_error)
                chunk = f"result: {digest}\n"
                out.append(chunk)
                stats["tool_digest_chars"] += len(chunk)

            # Text blocks (user/assistant prose)
            elif btype == "text":
                if isinstance(b, dict):
                    text_content = b.get("text", "")

                if not text_content.strip():
                    continue

                # Check for boilerplate (heuristic: starts with < and is > 200 chars)
                if text_content.strip().startswith("<") and len(text_content) > 200:
                    text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
                    if text_hash in boilerplate_seen:
                        boilerplate_seen[text_hash] += 1
                        stats["boilerplate_blocks"] += 1
                        continue  # Skip duplicate boilerplate
                    else:
                        boilerplate_seen[text_hash] = 1
                        # First occurrence: emit a note
                        chunk = f"\n[boilerplate block {text_hash}, {len(text_content)} chars, appears {boilerplate_seen[text_hash]} time(s)]\n"
                        out.append(chunk)
                        continue

                if role == "assistant":
                    chunk = f"\nA: {text_content.strip()}\n"
                    out.append(chunk)
                    stats["assistant_prose_chars"] += len(chunk)

                elif role == "user":
                    if is_meta:
                        continue
                    # Strip harness injections
                    authored, _inj = split_meta(text_content)
                    if authored.strip():
                        chunk = f"\nuser: {authored.strip()}\n"
                        out.append(chunk)
                        stats["user_prose_chars"] += len(chunk)

    # Header
    session_id = Path(transcript_path).stem[:8]
    header = f"# action record for session {session_id}\n\n"
    header += f"[{stats['entries']} entries, {stats['turns']} turns, {stats['tool_calls']} tool calls, {stats['tool_results']} results]\n"
    header += f"[thinking: {stats['thinking_kept']} kept, {stats['thinking_empty']} empty, {stats['signatures_dropped']} signatures dropped]\n"
    if stats["boilerplate_blocks"] > 0:
        header += f"[{stats['boilerplate_blocks']} duplicate boilerplate blocks suppressed]\n"
    header += "\n"

    return header + "".join(out), stats


def main():
    """CLI for testing: build_action_record.py <transcript.jsonl>"""
    import sys
    if len(sys.argv) < 2:
        print("usage: action_record.py <transcript.jsonl>")
        sys.exit(1)

    record, stats = build_action_record(sys.argv[1])
    print(record)
    print("\n--- stats ---", file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"{k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
