#!/usr/bin/env python3
"""Select 2 questions per session (seeded), compute artifact sizes and a cost estimate,
write questions.jsonl, hash. Usage: replication_freeze.py estimate | freeze --n 200"""
import argparse, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replication_common import RDIR
from spinoff_actionrecord_ablation import build_action_record_no_result_digests
from rq3_stripped_transcript_arm import strip_transcript

SEED = 20261013
PER_SESSION = 2
BPT = 3.6


def sizes():
    f = RDIR / "artifact_sizes.json"
    if f.exists():
        return json.loads(f.read_text())
    rows = [json.loads(l) for l in (RDIR / "questions_raw.jsonl").open()]
    out = {}
    for r in rows:
        p = r["transcript"]
        if p in out: continue
        nd, _ = build_action_record_no_result_digests(p)
        st, _ = strip_transcript(Path(p).read_text(errors="replace"))
        out[p] = {"nd_chars": len(nd), "st_chars": len(st)}
    f.write_text(json.dumps(out))
    return out


def pick(n):
    rows = [json.loads(l) for l in (RDIR / "questions_raw.jsonl").open()]
    order = json.loads((RDIR / "draw_order.json").read_text())
    by = {}
    for r in rows: by.setdefault(r["session_prefix"], []).append(r)
    rng = random.Random(SEED); sel = []
    for pre in order:
        if pre not in by or len(sel) >= n: continue
        qs = by[pre]; rng.shuffle(qs)
        sel += qs[:min(PER_SESSION, n - len(sel))]
    return sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    sz = sizes(); sel = pick(a.n)
    cost = 0.0
    for q in sel:
        s = sz[q["transcript"]]
        for ch in (s["nd_chars"], s["st_chars"]):
            cost += ((ch / BPT + 250) * 3 + 500 * 15) / 1e6
    print("questions", len(sel), "sessions", len({q["session_prefix"] for q in sel}),
          "mean nd chars", sum(sz[q["transcript"]]["nd_chars"] for q in sel) / len(sel),
          "mean st chars", sum(sz[q["transcript"]]["st_chars"] for q in sel) / len(sel),
          "answer cost est $%.2f" % cost, "max st tokens", max(sz[q["transcript"]]["st_chars"] for q in sel) / BPT)
    if a.mode == "freeze":
        out = []
        for i, q in enumerate(sel):
            rid = hashlib.sha256(f"{q['session_id']}|{q['class']}|{q['question']}".encode()).hexdigest()[:16]
            out.append({"question_id": rid, "record_id": rid, "session_id": q["session_id"], "session_prefix": q["session_prefix"],
                        "class": q["class"], "question": q["question"], "ground_truth": q["ground_truth"],
                        "transcript": q["transcript"], "order": i})
        p = RDIR / "questions.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in out))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        (RDIR / "questions.sha256").write_text(h + "  questions.jsonl\n")
        print("sha256", h)
