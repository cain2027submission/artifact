#!/usr/bin/env python3
"""Replication 2 supply count. Free, read-only: same eligibility rules as replication_draw.scan(),
plus exclusion of every Replication 1 session prefix (answered and unanswered).
Writes only data/paper-rollups/replication2/supply.json (prefixes and counts, no draw, no questions)."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parent))
from paths import PROJECTS_DIR
import replication_draw as rd
DATA = rd.DATA; R1 = DATA / "replication"; OUT = DATA / "replication2" / "supply.json"

def main():
    used = rd.prefixes_used()
    r1 = set()
    # the 100 session prefixes in the frozen Replication 1 question set (98 answered by both arms + 2 unanswered);
    # 5 further sessions had questions generated but were never frozen, and are not excluded (reported separately)
    for l in (R1 / "questions.jsonl").open():
        if l.strip(): r1.add(json.loads(l)["session_prefix"])
    used_all = used | r1
    now = time.time(); loss = {}; rows = []; seen = set()
    files = [p for p in PROJECTS_DIR.glob("**/*.jsonl") if "subagents" not in p.parts and p.stat().st_size >= 100_000]
    loss["ge_100KB"] = len(files)
    for p in sorted(files):
        pre = p.stem[:8]
        if pre in used: loss["design_or_heldout"] = loss.get("design_or_heldout", 0) + 1; continue
        if pre in r1: loss["replication1"] = loss.get("replication1", 0) + 1; continue
        if now - p.stat().st_mtime < 3600: loss["modified_last_hour"] = loss.get("modified_last_hour", 0) + 1; continue
        ts, synth, users, tools, paper = rd.inspect(p)
        if ts is None: loss["no_timestamp"] = loss.get("no_timestamp", 0) + 1; continue
        if synth or tools < 5 or users < 2: loss["synthetic_or_low_activity"] = loss.get("synthetic_or_low_activity", 0) + 1; continue
        if paper: loss["touched_paper_or_experiment"] = loss.get("touched_paper_or_experiment", 0) + 1; continue
        if pre in seen: loss["dup_prefix"] = loss.get("dup_prefix", 0) + 1; continue
        seen.add(pre); rows.append(pre)
    out = {"scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "design_heldout_prefixes": len(used),
           "replication1_prefixes": len(r1), "loss": loss, "n_eligible": len(rows),
           "max_questions_at_2_per_session": 2 * len(rows), "max_questions_at_3_per_session": 3 * len(rows),
           "eligible_prefixes": sorted(rows)}
    OUT.write_text(json.dumps(out, indent=1)); print({k: v for k, v in out.items() if k != "eligible_prefixes"})
main()
