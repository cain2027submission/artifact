#!/usr/bin/env python3
"""Where do the questions of three sets fall inside their sessions, and does
the production-rollup arm's recall depend on that position?

Sets: design (rq3, 124 questions), held-out (125), Replication 1 (196 answered).
All three are generated from the first GEN_WINDOW characters of the rendered
transcript (part2_question_retention.py), so a question's position is
estimated as follows: split the rendered transcript into blank-line-separated
blocks, score each block that starts inside the window by the idf-weighted
share of the question's ground-truth tokens it contains, and take the best
block's start offset divided by the full rendered length. The position is a
heuristic (lexical match, not provenance) and is confounded with session
identity: 25 sessions per set, several questions per session.

Reads only local files (no model calls). Writes
data/paper-rollups/position-heldout-vs-design/results.json.

Usage:
    position_heldout_vs_design.py [--out PATH]
"""

import argparse
import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import REPO_ROOT  # noqa: E402
import part2_question_retention as p2  # noqa: E402
import rq3_raw_transcript_arm as raw  # noqa: E402

DATA = REPO_ROOT / "data" / "paper-rollups"
# Input files: (set name) -> (questions file, rollup-arm file). Resolved
# against this checkout first, then against sibling worktrees, then the
# primary checkout, as other paper scripts do.
SETS = {
    "design": ("rq3_questions.jsonl", "rq3_rollup_uniform.jsonl"),
    "held": ("rq3_heldout_questions.jsonl", "heldout-rollup/rollup_arm.jsonl"),
    "rep": ("replication/questions.jsonl", "replication-rollup/rollup_arm.jsonl"),
}
GEN_WINDOW = 30000  # chars of rendered transcript the generator saw
BUCKETS = [(0, .10), (.10, .25), (.25, .50), (.50, 1.01)]
OUT_DEFAULT = DATA / "position-heldout-vs-design" / "results.json"


def find(rel):
    cands = [REPO_ROOT / "data" / "paper-rollups" / rel]
    wt = REPO_ROOT.parent  # .../.claude/worktrees
    if wt.name == "worktrees":
        cands += sorted(wt.glob("*/data/paper-rollups/" + rel))
        cands.append(wt.parents[1] / "data" / "paper-rollups" / rel)
    for c in cands:
        if c.exists():
            return c
    raise SystemExit("input not found: " + rel)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


STOP = set("the a an and or of to in for on with that this was were is are it its as at by from be been had has have not but which after then than into their about what why how did does assistant session".split())


def toks(s):
    return [t for t in re.findall(r"[A-Za-z0-9_./~-]{4,}", s.lower()) if t not in STOP]


BYP = raw.index_transcripts()
CACHE = {}


def tpath(r):
    if r.get("transcript"):
        return Path(r["transcript"])
    pre = r.get("session_prefix") or re.search(r"_([0-9a-f]{8})_", r["session_id"]).group(1)
    return raw.resolve(pre, BYP)


def sess(path):
    if path in CACHE:
        return CACHE[path]
    parts = p2.transcript_to_text(p2.load_transcript(path)).split("\n\n")
    offs, o = [], 0
    for x in parts:
        offs.append(o)
        o += len(x) + 2
    ps = [set(toks(x)) for x in parts]
    df = collections.Counter()
    for s in ps:
        df.update(s)
    CACHE[path] = (parts, offs, o, ps, df)
    return CACHE[path]


def locate(r):
    """Return (loc dict or None, reason when None)."""
    p = tpath(r)
    if p is None or not Path(p).exists():
        return None, "transcript_missing"
    parts, offs, L, ps, df = sess(p)
    gt = set(toks(r["ground_truth"]))
    n = len(parts)
    if not gt or n == 0:
        return None, "no_ground_truth_tokens"
    best = None
    for i, s in enumerate(ps):
        if offs[i] >= GEN_WINDOW:
            break
        sc = sum(math.log(1 + n / df[t]) for t in gt if t in s)
        if best is None or sc > best[0]:
            best = (sc, i)
    if best is None or best[0] == 0:
        return None, "no_block_match"
    i = best[1]
    return dict(frac=offs[i] / L, turn=i, nparts=n, L=L, win=min(1, GEN_WINDOW / L),
                nmatch=sum(1 for t in gt if t in ps[i]), ngt=len(gt)), None


def quart(x):
    q = st.quantiles(x, n=4, method="inclusive")
    return [round(v, 3) for v in q]


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    res = {"method": {
        "generation_window_chars": GEN_WINDOW,
        "position": "start offset of the best-matching blank-line block / full rendered length",
        "match": "idf-weighted count of ground-truth tokens (len>=4, stopwords removed) in the block; only blocks starting inside the window",
        "recall": "rollup arm grade in (correct, partially_correct)",
        "buckets": [list(b[:1]) + [min(b[1], 1.0)] for b in BUCKETS],
        "caveats": "positions are heuristic; 25 sessions per set; position is confounded with session identity; no significance tests"},
        "sets": {}}
    for name, (qf, af) in SETS.items():
        qs = jl(find(qf))
        arm = jl(find(af))
        qloc, reasons = {}, collections.Counter()
        missing_t = set()
        for r in qs:
            try:
                loc, why = locate(r)
            except Exception as e:  # unreadable transcript
                loc, why = None, "error:" + type(e).__name__
            if loc is None:
                reasons[why] += 1
                if why == "transcript_missing":
                    missing_t.add(r.get("session_prefix") or r["session_id"])
            qloc[r["question"]] = loc
        rows = [(qloc.get(x["question"]), x["grade"] in ("correct", "partially_correct"), x["session_id"]) for x in arm]
        ok = [r for r in rows if r[0]]
        sessions = {r[2]: r[0] for r in ok}
        fr = [r[0]["frac"] for r in ok]
        ws = [v["win"] for v in sessions.values()]
        Ls = [v["L"] for v in sessions.values()]
        s = {
            "questions_in_file": len(qs),
            "arm_rows": len(rows),
            "arm_rows_located": len(ok),
            "unlocated_question_reasons": dict(reasons),
            "sessions_with_missing_transcript": sorted(missing_t),
            "sessions_located": len(sessions),
            "relative_position": {"median": round(st.median(fr), 3), "q1_q3": [quart(fr)[0], quart(fr)[2]]},
            "share_first_25pct": {"n": sum(f < .25 for f in fr), "of": len(fr), "pct": pct(sum(f < .25 for f in fr), len(fr))},
            "share_first_10pct": {"n": sum(f < .10 for f in fr), "of": len(fr), "pct": pct(sum(f < .10 for f in fr), len(fr))},
            "session_length_chars": {"median": st.median(Ls), "q1_q3": [quart(Ls)[0], quart(Ls)[2]]},
            "window_coverage": {"median": round(st.median(ws), 3), "q1_q3": [quart(ws)[0], quart(ws)[2]],
                                "sessions_fully_covered": sum(w >= 1 for w in ws)},
            "recall_overall_located_pct": pct(sum(r[1] for r in ok), len(ok)),
            "recall_by_position_bucket": [],
            "match_quality": {
                "median_token_share_of_ground_truth_in_best_block": round(st.median(r[0]["nmatch"] / r[0]["ngt"] for r in ok), 3)},
        }
        for lo, hi in BUCKETS:
            b = [r for r in ok if lo <= r[0]["frac"] < hi]
            s["recall_by_position_bucket"].append({
                "bucket": "%d-%d%%" % (lo * 100, min(hi, 1) * 100), "n": len(b),
                "recall_pct": pct(sum(r[1] for r in b), len(b))})
        res["sets"][name] = s
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["sets"], indent=1))


if __name__ == "__main__":
    main()
