#!/usr/bin/env python3
"""RQ2, corrected: compression of session rollups against raw transcripts.

Three separate ratios, because conflating them is how this measurement goes
wrong, and it did go wrong on the first pass:

1. `serialized` -- rollup bytes vs raw .jsonl bytes. This is the number the
   earlier draft led with (~200x). It is inflated by JSON structure: keys,
   quoting, escapes and per-entry metadata that no memory system would ever
   inject into a prompt. Reported only so the inflation is visible.
2. `payload` -- rollup bytes vs the concatenated human-readable text of every
   content block. This is what "inject the whole transcript instead" actually
   costs, and it is the defensible byte ratio.
3. `tokens` -- rollup tokens vs payload tokens, using a chars/token factor
   calibrated against the API's own accounting rather than assumed.

A fourth quantity, `billed_context_tokens`, is reported but deliberately NOT
called compression. It sums input + cache-read + cache-creation + output across
deduplicated requests, so it grows quadratically with turns because the whole
context is re-sent every call. It measures what the session cost to run, not how
much text it contains.

The trap that produced a 0.02x "compression ratio" on the first pass: Claude's
`input_tokens` EXCLUDES `cache_read_input_tokens` and
`cache_creation_input_tokens`. A heavily cached session reports input_tokens=2
against a 136 KB transcript, so any denominator built from input+output alone
collapses. Cache fields must be added in.

Run:
    python3 rq2_compression_corrected.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from freeze_corpus import load_manifest, read_pinned, read_pinned_rollup
OUTDIR = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"

TEXT_BLOCK_KEYS = ("text", "thinking")

# Calibrating chars/token from the API's own output_tokens does not work: extended
# thinking is billed as output but is not fully persisted in the transcript, so the
# ratio comes out at 1.3 chars/token instead of the expected ~3.5 and the shortfall
# is unrecoverable. Instead tokenize both sides with one real code-aware BPE. The
# absolute counts will not match Claude's proprietary tokenizer, but the RATIO is
# what the paper claims, and a single consistent tokenizer measures the part that
# a character ratio cannot: tool output and code tokenize far denser than the
# prose a rollup is made of, so the token ratio is not simply the byte ratio.
TOKENIZER_PATH = (str(Path.home() / ".cache" / "huggingface" / "hub") + "/"
                  "models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit/"
                  "snapshots/019cc73c45c770444708a6dd8690c66243cc5c80/"
                  "tokenizer.json")


def load_tokenizer():
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(TOKENIZER_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: tokenizer unavailable ({exc}); token ratios omitted")
        return None


def count_tokens(tok, text: str) -> int:
    """Token count, chunked because the tokenizer rejects very long inputs."""
    if tok is None or not text:
        return 0
    total = 0
    step = 200_000
    for i in range(0, len(text), step):
        total += len(tok.encode(text[i:i + step], add_special_tokens=False).ids)
    return total


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    fm = {}
    for line in text[3:end].splitlines():
        m = re.match(r"^([a-z_]+):\s*(.*)$", line.strip())
        if m and m.group(2):
            fm[m.group(1)] = m.group(2).strip()
    return fm


def block_payload(block) -> tuple[str, str]:
    """Return (category, human-readable text) for one content block."""
    if isinstance(block, str):
        return "text", block
    if not isinstance(block, dict):
        return "other", ""
    btype = block.get("type", "other")
    if btype in ("text", "thinking", "redacted_thinking"):
        for k in TEXT_BLOCK_KEYS:
            if isinstance(block.get(k), str):
                return btype, block[k]
        return btype, ""
    if btype == "tool_use":
        return btype, json.dumps(block.get("input", ""))
    if btype == "tool_result":
        c = block.get("content")
        if isinstance(c, str):
            return btype, c
        if isinstance(c, list):
            return btype, "".join(
                b.get("text", "") for b in c if isinstance(b, dict))
        return btype, json.dumps(c) if c is not None else ""
    return "other", ""


def scan_transcript(raw: str, tok=None) -> dict | None:
    payload_chars = 0
    payload_tokens = 0
    assistant_chars = 0          # assistant text + thinking, for calibration
    by_cat: dict[str, int] = defaultdict(int)
    by_cat_tokens: dict[str, int] = defaultdict(int)
    usage_by_request: dict[str, dict] = {}
    entries = 0

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries += 1
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [content]
        for b in blocks:
            cat, text = block_payload(b)
            n = len(text)
            payload_chars += n
            by_cat[cat] += n
            if tok is not None:
                t = count_tokens(tok, text)
                payload_tokens += t
                by_cat_tokens[cat] += t
            # Everything the model emitted counts toward output_tokens, and
            # that includes tool_use inputs -- a single Write call carries a
            # whole file as generated output. Leaving tool_use out of this sum
            # put the calibration at 0.26 chars/token, i.e. four tokens per
            # character, which is what exposed the omission.
            if cat in ("text", "thinking", "redacted_thinking", "tool_use") \
                    and msg.get("role") == "assistant":
                assistant_chars += n
        # Each content block of one API response repeats that response's usage
        # record, so keying by requestId is what stops a ~2.4x overcount.
        rid = rec.get("requestId")
        usage = msg.get("usage")
        if rid and isinstance(usage, dict):
            usage_by_request[rid] = usage

    billed = 0
    out_tokens = 0
    for u in usage_by_request.values():
        billed += (u.get("input_tokens", 0) or 0)
        billed += (u.get("cache_read_input_tokens", 0) or 0)
        billed += (u.get("cache_creation_input_tokens", 0) or 0)
        billed += (u.get("output_tokens", 0) or 0)
        out_tokens += (u.get("output_tokens", 0) or 0)

    return {
        "serialized_bytes": len(raw),
        "payload_chars": payload_chars,
        "payload_tokens_measured": payload_tokens,
        "assistant_chars": assistant_chars,
        "by_cat": dict(by_cat),
        "by_cat_tokens": dict(by_cat_tokens),
        "requests": len(usage_by_request),
        "billed_context_tokens": billed,
        "output_tokens": out_tokens,
        "entries": entries,
    }


def index_transcripts(manifest: dict) -> dict[str, Path]:
    """Map session id -> transcript path (filename stem is the session id)."""
    idx: dict[str, Path] = {}
    for p_str in manifest["transcripts"]:
        p = Path(p_str)
        idx.setdefault(p.stem, p)
    return idx


def q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


def summarise(vals: list[float]) -> dict:
    vals = [v for v in vals if v == v and v not in (float("inf"),)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": round(min(vals), 2),
        "p10": round(q(vals, 0.10), 2),
        "median": round(statistics.median(vals), 2),
        "mean": round(statistics.mean(vals), 2),
        "p90": round(q(vals, 0.90), 2),
        "max": round(max(vals), 2),
    }


def main() -> None:
    manifest = load_manifest()
    tindex = index_transcripts(manifest)
    paired_ids = set(manifest["paired_session_ids"])
    tok = load_tokenizer()
    rows = []

    # Rollups are read through read_pinned_rollup, keyed by session_id against
    # the frozen manifest, because the rollup drain rewrites and renames notes
    # in place on a cadence of minutes -- a live glob() re-run drifts (see
    # freeze_corpus.read_pinned_rollup's docstring). A session with no valid
    # pin (renamed or rewritten since the snapshot) is dropped from the corpus
    # rather than read live, the same way a session with no transcript already
    # is below. This also retires this function's own manual first-note-wins
    # dedup for sessions with two rollup notes: freeze_corpus.build() already
    # resolves one rollup path per session_id into the manifest, so there is
    # nothing left here to deduplicate.
    for sid in sorted(paired_ids):
        if sid not in tindex:
            continue
        rollup_raw = read_pinned_rollup(sid, manifest)
        if rollup_raw is None:
            continue
        fm = parse_frontmatter(rollup_raw)
        transcript_raw = read_pinned(str(tindex[sid]), manifest)
        scan = scan_transcript(transcript_raw, tok)
        if not scan or scan["payload_chars"] == 0:
            continue
        rollup_path = manifest.get("rollup_note_paths", {}).get(sid, sid)
        rows.append({
            "file": Path(rollup_path).name,
            "session_id": sid,
            "agent": fm.get("agent", ""),
            "model": fm.get("model", ""),
            "turns": fm.get("turns", ""),
            "subagents": fm.get("subagents", ""),
            "compactions": fm.get("compactions", ""),
            "rollup_bytes": len(rollup_raw),
            "rollup_tokens_measured": count_tokens(tok, rollup_raw),
            **{k: v for k, v in scan.items()
               if k not in ("by_cat", "by_cat_tokens")},
            "tool_result_chars": scan["by_cat"].get("tool_result", 0),
            "tool_result_tokens": scan["by_cat_tokens"].get("tool_result", 0),
        })

    if not rows:
        raise SystemExit("no rollup/transcript pairs matched")

    # Calibrate chars/token against the API's own output accounting. Assistant
    # text and thinking characters are exactly what produced output_tokens, so
    # their ratio is a measured constant for this corpus rather than the
    # customary guess of 4.
    cal = [(r["assistant_chars"], r["output_tokens"]) for r in rows
           if r["output_tokens"] > 1000 and r["assistant_chars"] > 1000]
    per_session = [a / o for a, o in cal]
    pooled = sum(a for a, _ in cal) / sum(o for _, o in cal) if cal else 4.0
    chars_per_token = pooled

    for r in rows:
        r["comp_serialized"] = r["serialized_bytes"] / r["rollup_bytes"]
        r["comp_payload"] = r["payload_chars"] / r["rollup_bytes"]
        r["comp_tokens"] = (r["payload_tokens_measured"] / r["rollup_tokens_measured"]
                            if r.get("rollup_tokens_measured") else float("nan"))
        r["tool_share"] = (r["tool_result_chars"] / r["payload_chars"]
                           if r["payload_chars"] else 0)
        # A rollup that is bigger than the session it summarises is not a
        # compression failure to hide; it is the honest floor of the technique.
        # Counted on comp_tokens, not comp_payload: comp_payload divides
        # characters by *bytes*, so on any session whose rollup contains
        # multi-byte characters the two disagree near the 1.0 boundary. Counting
        # on comp_payload gave 40 of 286 where the token ratio -- the unit the
        # paper's headline compression is stated in -- gives 42.
        r["expands"] = 1 if r["comp_tokens"] < 1.0 else 0

    summary = {
        "corpus_snapshot_utc": manifest["snapshot_utc"],
        "n_pairs": len(rows),
        "calibration": {
            "method": "assistant text+thinking chars / deduplicated API "
                      "output_tokens, pooled over sessions with >1000 of each",
            "n_sessions": len(cal),
            "chars_per_token_pooled": round(chars_per_token, 3),
            "chars_per_token_median_per_session": round(
                statistics.median(per_session), 3) if per_session else None,
            "note": "comp_tokens is the ratio the paper quotes, because token "
                    "budgets are the unit a memory system is constrained by. "
                    "comp_payload is close to it but not a rescaling of it: "
                    "its denominator is rollup *bytes* while its numerator is "
                    "payload *characters*, so the two rank sessions slightly "
                    "differently wherever a rollup carries non-ASCII text",
        },
        "compression_serialized_bytes": summarise([r["comp_serialized"] for r in rows]),
        "compression_payload_bytes": summarise([r["comp_payload"] for r in rows]),
        "compression_tokens": summarise([r["comp_tokens"] for r in rows]),
        "inflation_serialized_over_payload": round(
            statistics.median([r["serialized_bytes"] / max(1, r["payload_chars"])
                               for r in rows]), 2),
        # The share of sessions a rollup makes bigger rather than smaller. These
        # are short sessions whose fixed note scaffolding (frontmatter, five
        # headings, a file list) outweighs the conversation being summarised.
        "sessions_rollup_expands": {
            "n": sum(r["expands"] for r in rows),
            "pct": round(100.0 * sum(r["expands"] for r in rows) / len(rows), 1),
        },
        # Compression conditioned on session size, since the technique is only
        # claimed to pay off on substantial sessions.
        "compression_payload_by_size": {
            label: summarise([r["comp_payload"] for r in rows
                              if lo <= r["payload_chars"] < hi])
            for label, lo, hi in (
                ("under_10KB", 0, 10_000),
                ("10KB_to_100KB", 10_000, 100_000),
                ("100KB_to_1MB", 100_000, 1_000_000),
                ("over_1MB", 1_000_000, float("inf")),
            )
        },
        # The same bands in the token unit. The paper quotes token compression
        # everywhere else, and a table headed "token compression" that was
        # actually sourced from the payload-character bands is the kind of unit
        # slip that survives review and then cannot be reproduced.
        "compression_tokens_by_size": {
            label: summarise([r["comp_tokens"] for r in rows
                              if lo <= r["payload_tokens_measured"] < hi])
            for label, lo, hi in (
                ("under_2500_tokens", 0, 2_500),
                ("2500_to_25k_tokens", 2_500, 25_000),
                ("25k_to_250k_tokens", 25_000, 250_000),
                ("over_250k_tokens", 250_000, float("inf")),
            )
        },
        "billed_context_tokens": summarise(
            [float(r["billed_context_tokens"]) for r in rows]),
        "totals": {
            "serialized_bytes": sum(r["serialized_bytes"] for r in rows),
            "payload_chars": sum(r["payload_chars"] for r in rows),
            "rollup_bytes": sum(r["rollup_bytes"] for r in rows),
            "requests": sum(r["requests"] for r in rows),
        },
    }

    by_agent: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_agent[r["agent"] or "unknown"].append(r["comp_payload"])
    summary["compression_payload_by_agent"] = {
        k: summarise(v) for k, v in sorted(by_agent.items())}

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "rq2_compression_corrected.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUTDIR / "rq2_compression_corrected.json").write_text(
        json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
