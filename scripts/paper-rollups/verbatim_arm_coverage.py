#!/usr/bin/env python3
"""Measure the verbatim arm's real coverage against the paper's graded session sets.

Answers the question the candidates note left open: of the 25 RQ3 graded
sessions and the 300 RQ2/RQ4 paired sessions, how many have a verbatim.md
on disk in the traces repo?
"""
import json
import os
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, TRACES_DIR  # noqa: E402

TRACES = TRACES_DIR
DATA = DATA_DIR

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
SHORT_RE = re.compile(r"\b([0-9a-f]{8})\b", re.I)

# --- index the traces repo -------------------------------------------------
dirs = [d for d in TRACES.iterdir() if d.is_dir()]
full_uuid, short_uuid, no_id = {}, {}, []
for d in dirs:
    has_verbatim = (d / "verbatim.md").exists()
    m = UUID_RE.search(d.name)
    if m:
        full_uuid[m.group(0).lower()] = has_verbatim
        short_uuid.setdefault(m.group(0).lower()[:8], has_verbatim)
        continue
    # fall back to an 8-hex field, which is how rollup-style slugs name sessions
    parts = d.name.split("_")
    hit = next((p for p in parts if SHORT_RE.fullmatch(p)), None)
    if hit:
        short_uuid.setdefault(hit.lower(), has_verbatim)
    else:
        no_id.append(d.name)

n_verbatim = sum(1 for d in dirs if (d / "verbatim.md").exists())
print(f"trace directories:            {len(dirs)}")
print(f"  with verbatim.md:           {n_verbatim}")
print(f"  full UUID in dir name:      {len(full_uuid)}")
print(f"  only an 8-hex short id:     {len(short_uuid) - len(full_uuid)}")
print(f"  no parseable id at all:     {len(no_id)}")


def resolve(sid):
    """Return (matched, has_verbatim) for a session id in either form."""
    s = sid.lower()
    m = UUID_RE.search(s)
    if m and m.group(0) in full_uuid:
        return True, full_uuid[m.group(0)]
    # rollup-style slug: YYYY-MM-DD_client_<8hex>_title
    for p in s.split("_"):
        if SHORT_RE.fullmatch(p) and p in short_uuid:
            return True, short_uuid[p]
    if m and m.group(0)[:8] in short_uuid:
        return True, short_uuid[m.group(0)[:8]]
    return False, False


def report(label, ids):
    ids = sorted(set(ids))
    matched = [i for i in ids if resolve(i)[0]]
    withv = [i for i in ids if resolve(i)[1]]
    print(f"\n{label}: {len(ids)} sessions")
    print(f"  found in traces repo:       {len(matched)}")
    print(f"  of those, have verbatim.md: {len(withv)}"
          f"  ({100.0 * len(withv) / len(ids):.1f}% of the set)")
    return withv


# --- RQ3 graded set -------------------------------------------------------
rq3 = set()
for name in ("rq3_majority_uniform.jsonl", "rq3_majority_stripped.jsonl",
             "rq3_majority_raw.jsonl"):
    p = DATA / name
    if not p.exists():
        continue
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for k in ("session_id", "session", "sid", "slug"):
            if isinstance(rec.get(k), str):
                rq3.add(rec[k])
                break
report("RQ3 graded", rq3)

# --- RQ2 / RQ4 paired set -------------------------------------------------
snap = json.load((DATA / "corpus_snapshot.json").open())
paired = snap.get("paired_session_ids") or []
if not paired:  # find any list of uuid-looking strings
    for v in snap.values():
        if isinstance(v, list) and v and isinstance(v[0], str) and UUID_RE.search(v[0]):
            paired = v
            break
report("RQ2/RQ4 paired", paired)
