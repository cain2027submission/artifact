#!/usr/bin/env python3
"""Measure sanitization burden for MSR data release.

Checks what fraction of transcripts contain patterns requiring redaction:
- Absolute home paths
- Repository and organization names
- API key patterns
- Proprietary terms

Reports COUNTS only, never extracts actual content.
"""

import json
import re
import random
from pathlib import Path
from collections import Counter

from paths import (
    DATA_DIR, PROJECTS_DIR, CORPUS_HOME, CORPUS_SENSITIVE_TERMS,
    home_path_pattern
)

# Load the frozen corpus manifest
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
if MANIFEST_PATH.exists():
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    transcript_paths = manifest["transcripts"]
else:
    # Fallback if no manifest
    transcript_paths = [str(p) for p in PROJECTS_DIR.rglob("*.jsonl")]

# Redaction patterns built from config so they match the actual corpus
_sensitive_alternation = "|".join(re.escape(term) for term in CORPUS_SENSITIVE_TERMS)
_project_base = re.escape(CORPUS_HOME.rstrip("/") + "/Documents/GitHub/")

PATTERNS = {
    "home_path": home_path_pattern(),
    "repo_name": re.compile(_sensitive_alternation, re.IGNORECASE) if _sensitive_alternation else re.compile(r"(?!)"),
    "api_key": re.compile(r"(?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_-]{20,}"),
    "project_path": re.compile(_project_base + r"\w+"),
}

def scan_transcript(path: Path) -> dict:
    """Scan one transcript and return pattern match counts."""
    if not path.exists():
        return None

    try:
        text = path.read_text()
    except Exception:
        return None

    hits = {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}
    hits["char_count"] = len(text)
    hits["line_count"] = text.count("\n")

    # Binary flags: does this transcript need redaction?
    hits["needs_redaction"] = any(hits[k] > 0 for k in PATTERNS.keys())

    return hits

def main():
    print("=== SANITIZATION BURDEN AUDIT ===")
    print(f"Sampling {min(50, len(transcript_paths))} transcripts from {len(transcript_paths)} total")
    print()

    # Sample transcripts
    sample_size = min(50, len(transcript_paths))
    sample = random.sample(transcript_paths, sample_size)

    results = []
    for path_str in sample:
        path = Path(path_str)
        result = scan_transcript(path)
        if result:
            results.append(result)

    if not results:
        print("ERROR: No transcripts could be read")
        return

    # Aggregate statistics
    needs_redaction = sum(1 for r in results if r["needs_redaction"])
    total = len(results)
    pct = 100 * needs_redaction / total

    print(f"Transcripts needing redaction: {needs_redaction} / {total} ({pct:.1f}%)")
    print()

    # Pattern-specific breakdown
    print("Pattern frequencies (mean hits per transcript):")
    for name in PATTERNS.keys():
        mean_hits = sum(r[name] for r in results) / total
        affected = sum(1 for r in results if r[name] > 0)
        print(f"  {name}: {mean_hits:.1f} hits/transcript ({affected}/{total} transcripts)")
    print()

    # Estimate per-session burden
    print("Estimated per-session redaction burden:")
    if needs_redaction > 0:
        avg_char = sum(r["char_count"] for r in results if r["needs_redaction"]) / needs_redaction
        print(f"  Mean size of sessions needing redaction: {avg_char/1024:.1f} KB")
        print(f"  Estimated total: {needs_redaction} × manual review + pattern replacement")

    # Viability verdict
    print()
    print("=== DATA RELEASE VIABILITY ===")
    if pct < 20:
        print(f"VIABLE: Only {pct:.1f}% need redaction. Can release sanitized subset.")
    elif pct < 50:
        print(f"CHALLENGING: {pct:.1f}% need redaction. Automated patterns + spot-checks feasible.")
    else:
        print(f"BLOCKED: {pct:.1f}% need redaction. Manual burden too high for 100+ session release.")

if __name__ == "__main__":
    random.seed(42)  # Reproducible sample
    main()
