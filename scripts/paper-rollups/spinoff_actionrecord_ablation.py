#!/usr/bin/env python3
"""Generate ablation variants of the action record.

The action record has three components:
1. Turn index (turn numbers prefixing each exchange)
2. Result digests (one-line summaries: exit code, byte count, first 80 chars)
3. Structured arguments (salient tool args extracted, not full input dicts)

This script generates three leave-one-out variants and measures their sizes
to estimate the cost of running a full ablation study.

Usage:
    spinoff_actionrecord_ablation.py <transcript.jsonl>   # smoke test on one
    spinoff_actionrecord_ablation.py --sample 2           # run on 2 transcripts
"""

import argparse
import sys
from pathlib import Path

# Import action_record builder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts/paper-rollups"))
from action_record import build_action_record  # noqa: E402

def build_action_record_no_turn_index(transcript_path: str):
    """Build action record without turn numbers."""
    from action_record import (
        json, split_meta, truncate, structure_tool_call, digest_tool_result,
        hashlib
    )

    text = Path(transcript_path).read_text(errors="replace")
    out = []
    stats = {
        "entries": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "tool_call_chars": 0,
        "tool_digest_chars": 0,
        "assistant_prose_chars": 0,
        "user_prose_chars": 0,
    }

    seen_file_content = {}
    boilerplate_seen = {}

    # NO turn_num tracking, NO turn markers
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

        # Skip user/assistant turn markers entirely

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
                text_content = b.get("thinking") or b.get("text") or ""
                if not text_content.strip():
                    continue
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
                result_content = ""
                is_error = b.get("is_error", False)
                result = b.get("content")
                if isinstance(result, str):
                    result_content = result
                elif isinstance(result, list):
                    parts = []
                    for rb in result:
                        if isinstance(rb, str):
                            parts.append(rb)
                        elif isinstance(rb, dict) and rb.get("type") == "text":
                            parts.append(rb.get("text", ""))
                    result_content = "\n".join(parts)

                digest = digest_tool_result("Read", result_content, is_error)
                chunk = f"result: {digest}\n"
                out.append(chunk)
                stats["tool_digest_chars"] += len(chunk)

            # Text blocks
            elif btype == "text":
                if isinstance(b, dict):
                    text_content = b.get("text", "")
                if not text_content.strip():
                    continue
                if text_content.strip().startswith("<") and len(text_content) > 200:
                    text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
                    if text_hash in boilerplate_seen:
                        boilerplate_seen[text_hash] += 1
                        continue
                    else:
                        boilerplate_seen[text_hash] = 1
                        chunk = f"\n[boilerplate block {text_hash}, {len(text_content)} chars]\n"
                        out.append(chunk)
                        continue

                if role == "assistant":
                    chunk = f"\nA: {text_content.strip()}\n"
                    out.append(chunk)
                    stats["assistant_prose_chars"] += len(chunk)
                elif role == "user":
                    if is_meta:
                        continue
                    authored, _inj = split_meta(text_content)
                    if authored.strip():
                        chunk = f"\nuser: {authored.strip()}\n"
                        out.append(chunk)
                        stats["user_prose_chars"] += len(chunk)

    session_id = Path(transcript_path).stem[:8]
    header = f"# action record (NO TURN INDEX) for session {session_id}\n\n"
    header += f"[{stats['entries']} entries, {stats['tool_calls']} tool calls, {stats['tool_results']} results]\n\n"

    return header + "".join(out), stats


def build_action_record_no_result_digests(transcript_path: str):
    """Build action record without result digests (replicate output-stripped)."""
    from action_record import (
        json, split_meta, structure_tool_call, hashlib
    )

    text = Path(transcript_path).read_text(errors="replace")
    out = []
    stats = {
        "entries": 0,
        "turns": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "tool_call_chars": 0,
        "tool_digest_chars": 0,
        "assistant_prose_chars": 0,
        "user_prose_chars": 0,
    }

    boilerplate_seen = {}
    turn_num = 0

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

        # Turn boundaries
        if role == "user" and not is_meta:
            turn_num += 1
            stats["turns"] = turn_num
            out.append(f"\n--- turn {turn_num} ---\n")

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
                text_content = b.get("thinking") or b.get("text") or ""
                if not text_content.strip():
                    continue
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

            # Tool results - SKIP (no digests)
            elif btype == "tool_result":
                stats["tool_results"] += 1
                # NO DIGEST - just count it, don't append anything

            # Text blocks
            elif btype == "text":
                if isinstance(b, dict):
                    text_content = b.get("text", "")
                if not text_content.strip():
                    continue
                if text_content.strip().startswith("<") and len(text_content) > 200:
                    text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
                    if text_hash in boilerplate_seen:
                        boilerplate_seen[text_hash] += 1
                        continue
                    else:
                        boilerplate_seen[text_hash] = 1
                        chunk = f"\n[boilerplate block {text_hash}]\n"
                        out.append(chunk)
                        continue

                if role == "assistant":
                    chunk = f"\nA: {text_content.strip()}\n"
                    out.append(chunk)
                    stats["assistant_prose_chars"] += len(chunk)
                elif role == "user":
                    if is_meta:
                        continue
                    authored, _inj = split_meta(text_content)
                    if authored.strip():
                        chunk = f"\nuser: {authored.strip()}\n"
                        out.append(chunk)
                        stats["user_prose_chars"] += len(chunk)

    session_id = Path(transcript_path).stem[:8]
    header = f"# action record (NO RESULT DIGESTS) for session {session_id}\n\n"
    header += f"[{stats['entries']} entries, {stats['turns']} turns, {stats['tool_calls']} tool calls, {stats['tool_results']} results]\n\n"

    return header + "".join(out), stats


def build_action_record_no_structured_args(transcript_path: str):
    """Build action record without structured args (full tool inputs, capped at 2000 chars like output-stripped)."""
    from action_record import (
        json, split_meta, digest_tool_result, hashlib
    )

    text = Path(transcript_path).read_text(errors="replace")
    out = []
    stats = {
        "entries": 0,
        "turns": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "tool_call_chars": 0,
        "tool_digest_chars": 0,
        "assistant_prose_chars": 0,
        "user_prose_chars": 0,
    }

    boilerplate_seen = {}
    turn_num = 0

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

        # Turn boundaries
        if role == "user" and not is_meta:
            turn_num += 1
            stats["turns"] = turn_num
            out.append(f"\n--- turn {turn_num} ---\n")

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
                text_content = b.get("thinking") or b.get("text") or ""
                if not text_content.strip():
                    continue
                chunk = f"\n### REASONING\n{text_content.strip()}\n"
                out.append(chunk)
                stats["assistant_prose_chars"] += len(chunk)

            # Tool calls - FULL INPUT, capped at 2000 chars
            elif btype == "tool_use":
                stats["tool_calls"] += 1
                name = b.get("name") or "tool"
                inp = b.get("input", {})
                full_input = json.dumps(inp, indent=None)
                capped_input = full_input[:2000]
                if len(full_input) > 2000:
                    capped_input += f"... [+{len(full_input)-2000} chars]"
                chunk = f"\ntool: {name}({capped_input})\n"
                out.append(chunk)
                stats["tool_call_chars"] += len(chunk)

            # Tool results
            elif btype == "tool_result":
                stats["tool_results"] += 1
                result_content = ""
                is_error = b.get("is_error", False)
                result = b.get("content")
                if isinstance(result, str):
                    result_content = result
                elif isinstance(result, list):
                    parts = []
                    for rb in result:
                        if isinstance(rb, str):
                            parts.append(rb)
                        elif isinstance(rb, dict) and rb.get("type") == "text":
                            parts.append(rb.get("text", ""))
                    result_content = "\n".join(parts)

                digest = digest_tool_result("Read", result_content, is_error)
                chunk = f"result: {digest}\n"
                out.append(chunk)
                stats["tool_digest_chars"] += len(chunk)

            # Text blocks
            elif btype == "text":
                if isinstance(b, dict):
                    text_content = b.get("text", "")
                if not text_content.strip():
                    continue
                if text_content.strip().startswith("<") and len(text_content) > 200:
                    text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
                    if text_hash in boilerplate_seen:
                        boilerplate_seen[text_hash] += 1
                        continue
                    else:
                        boilerplate_seen[text_hash] = 1
                        chunk = f"\n[boilerplate block {text_hash}]\n"
                        out.append(chunk)
                        continue

                if role == "assistant":
                    chunk = f"\nA: {text_content.strip()}\n"
                    out.append(chunk)
                    stats["assistant_prose_chars"] += len(chunk)
                elif role == "user":
                    if is_meta:
                        continue
                    authored, _inj = split_meta(text_content)
                    if authored.strip():
                        chunk = f"\nuser: {authored.strip()}\n"
                        out.append(chunk)
                        stats["user_prose_chars"] += len(chunk)

    session_id = Path(transcript_path).stem[:8]
    header = f"# action record (NO STRUCTURED ARGS, full inputs capped 2k) for session {session_id}\n\n"
    header += f"[{stats['entries']} entries, {stats['turns']} turns, {stats['tool_calls']} tool calls]\n\n"

    return header + "".join(out), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", nargs="?", help="Single transcript to test")
    ap.add_argument("--sample", type=int, help="Run on N sample transcripts from graded set")
    args = ap.parse_args()

    if args.transcript:
        transcripts = [args.transcript]
    elif args.sample:
        # Load graded sessions
        import json
        from paths import DATA_DIR
        data_dir = DATA_DIR
        summary = json.loads((data_dir / "rq3_action_record_arm_summary.json").read_text())
        action_arm = [json.loads(line) for line in (data_dir / "rq3_action_record_arm.jsonl").read_text().splitlines() if line.strip()]

        # Get unique transcripts
        unique_transcripts = list({r["transcript"] for r in action_arm})
        transcripts = unique_transcripts[:args.sample]
    else:
        print("usage: spinoff_actionrecord_ablation.py <transcript.jsonl>")
        print("       spinoff_actionrecord_ablation.py --sample N")
        sys.exit(1)

    print(f"## Ablation Size Estimates (n={len(transcripts)} transcripts)\n")
    print(f"{'Session':10s} {'Full':>8s} {'No-Turn':>10s} {'No-Digest':>10s} {'No-Struct':>10s}")
    print("-" * 60)

    totals = {"full": 0, "no_turn": 0, "no_digest": 0, "no_struct": 0}

    for t in transcripts:
        session_id = Path(t).stem[:8]

        full_record, full_stats = build_action_record(t)
        no_turn_record, no_turn_stats = build_action_record_no_turn_index(t)
        no_digest_record, no_digest_stats = build_action_record_no_result_digests(t)
        no_struct_record, no_struct_stats = build_action_record_no_structured_args(t)

        full_chars = len(full_record)
        no_turn_chars = len(no_turn_record)
        no_digest_chars = len(no_digest_record)
        no_struct_chars = len(no_struct_record)

        totals["full"] += full_chars
        totals["no_turn"] += no_turn_chars
        totals["no_digest"] += no_digest_chars
        totals["no_struct"] += no_struct_chars

        print(f"{session_id:10s} {full_chars:8,} {no_turn_chars:10,} {no_digest_chars:10,} {no_struct_chars:10,}")

    print("-" * 60)
    median_idx = len(transcripts) // 2

    # For medians, we'd need to sort, but for now just show means
    mean_full = totals["full"] // len(transcripts)
    mean_no_turn = totals["no_turn"] // len(transcripts)
    mean_no_digest = totals["no_digest"] // len(transcripts)
    mean_no_struct = totals["no_struct"] // len(transcripts)

    print(f"{'MEAN':10s} {mean_full:8,} {mean_no_turn:10,} {mean_no_digest:10,} {mean_no_struct:10,}")

    # Component contributions
    turn_contrib = mean_full - mean_no_turn
    digest_contrib = mean_full - mean_no_digest
    struct_contrib = mean_full - mean_no_struct

    print(f"\n## Component Size Contributions (chars)\n")
    print(f"Turn index:        {turn_contrib:+8,}  ({100*turn_contrib/mean_full:+5.1f}%)")
    print(f"Result digests:    {digest_contrib:+8,}  ({100*digest_contrib/mean_full:+5.1f}%)")
    print(f"Structured args:   {struct_contrib:+8,}  ({100*struct_contrib/mean_full:+5.1f}%)")

    # Token estimates (2.45 bytes/token rough)
    print(f"\n## Token Estimates (÷2.45 bytes/token)\n")
    print(f"Full action record:            {mean_full/2.45:8,.0f} tokens")
    print(f"No turn index:                 {mean_no_turn/2.45:8,.0f} tokens")
    print(f"No result digests:             {mean_no_digest/2.45:8,.0f} tokens")
    print(f"No structured args (full 2k):  {mean_no_struct/2.45:8,.0f} tokens")

if __name__ == "__main__":
    main()
