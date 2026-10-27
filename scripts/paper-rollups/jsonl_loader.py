#!/usr/bin/env python3
"""Strict JSONL loader that counts and reports parse failures.

Every data-loading script should use this instead of silently filtering
unparseable lines, because silent filtering hides data loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


class LoadResult(NamedTuple):
    """Result of loading a JSONL file."""
    records: list[dict]
    attempted: int
    parsed: int
    failed: int


def load_jsonl_strict(
    path: Path,
    *,
    allow_unparseable: bool = False,
    show_first_bytes: int = 60,
    required_keys: set[str] | None = None
) -> LoadResult:
    """Load a JSONL file with strict validation by default.

    Args:
        path: Path to the JSONL file
        allow_unparseable: If False (default), raise on any unparseable line.
                          If True, skip unparseable lines and return stats.
        show_first_bytes: Number of bytes to show in error message (default 60)
        required_keys: Optional set of keys that must be present in each record
                      dict. When supplied, records missing any key or having an
                      empty-string value for a required key are failures.

    Returns:
        LoadResult with records, counts of attempted/parsed/failed lines

    Raises:
        ValueError: If any line fails to parse and allow_unparseable is False,
                   or if a line is not a dict, or if required keys are missing
                   or empty. The error message includes the line number and
                   details of the failure.

    This exists because every consumer used to filter unparseable lines WITHOUT
    COUNTING them, so total data loss looked like a clean read. The corruption
    in rq3_questions.jsonl line 125 (32,343 NUL bytes) survived unnoticed for
    exactly this reason.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    attempted = len(lines)
    records = []
    failed_lines = []

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            # Empty lines don't count as attempts
            attempted -= 1
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            failed_lines.append((i, line, e))
            if not allow_unparseable:
                # Show first N bytes of the bad line in error message
                preview = line[:show_first_bytes]
                if len(line) > show_first_bytes:
                    preview += "..."
                raise ValueError(
                    f"{path.name}: line {i} failed to parse: {e}\n"
                    f"First {show_first_bytes} bytes: {repr(preview)}"
                ) from e
            continue

        # Structural validation: must be a dict
        if not isinstance(obj, dict):
            error_msg = (
                f"{path.name}: line {i} is not a dict (got {type(obj).__name__})"
            )
            failed_lines.append((i, line, error_msg))
            if not allow_unparseable:
                raise ValueError(error_msg)
            continue

        # Required keys validation
        if required_keys:
            missing = required_keys - obj.keys()
            if missing:
                error_msg = (
                    f"{path.name}: line {i} missing required keys: "
                    f"{', '.join(sorted(missing))}"
                )
                failed_lines.append((i, line, error_msg))
                if not allow_unparseable:
                    raise ValueError(error_msg)
                continue

            # Check for empty string values in required keys
            empty_keys = [k for k in required_keys if obj.get(k) == ""]
            if empty_keys:
                error_msg = (
                    f"{path.name}: line {i} has empty-string value(s) for "
                    f"required key(s): {', '.join(sorted(empty_keys))}"
                )
                failed_lines.append((i, line, error_msg))
                if not allow_unparseable:
                    raise ValueError(error_msg)
                continue

        # Empty dict check: fail if the dict is empty (no keys at all)
        if not obj:
            error_msg = f"{path.name}: line {i} is an empty dict"
            failed_lines.append((i, line, error_msg))
            if not allow_unparseable:
                raise ValueError(error_msg)
            continue

        records.append(obj)

    failed = len(failed_lines)
    parsed = len(records)

    return LoadResult(
        records=records,
        attempted=attempted,
        parsed=parsed,
        failed=failed
    )


def load_jsonl(path: Path) -> list[dict]:
    """Convenience wrapper that returns just the records list.

    Raises on any parse failure. Use load_jsonl_strict() if you need
    the detailed counts or want to allow unparseable lines.
    """
    result = load_jsonl_strict(path, allow_unparseable=False)
    return result.records
