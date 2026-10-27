#!/usr/bin/env python3
"""Step 2 of the replication pre-registration: eligible-session scan, seeded draw, and
question generation. Stages:
  scan      free. list eligible sessions -> data/paper-rollups/replication/candidates.json
  generate  paid. questions for the drawn sessions -> replication/questions_raw.jsonl
  freeze    free. select <=3 (here 2) per session, write questions.jsonl + sha256

Eligibility (pre-registration "question and grading rules" + held-out manifest predicate):
  top-level transcript (not subagents) >= 100,000 bytes; first record timestamp
  present (NO date cutoff: with the held-out manifest's > 2026-09-26 cutoff only 54 sessions
  qualify, short of the >= 100 targeted, and the pre-registration requires only absence from
  the design and held-out sets, so the cutoff is dropped and written up as a deviation); not synthetic
  (first authored user turn not a rollup-summariser / "reply with the single word: ok"
  call; >= 5 tool_use blocks; >= 2 authored user turns); 8-hex prefix in neither the 25
  design-set sessions, the 25 held-out sessions nor any session in rq3_questions.jsonl /
  rq3_heldout_questions.jsonl; file not modified in the last 60 min (still being written);
  no tool call whose input mentions paper/, paper-rollups, cain-replication or a
  tasks/ note about this paper (conservative superset of 'edited'); raw transcript on disk.
"""
import argparse, json, random, re, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path
import time

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from paths import PROJECTS_DIR  # noqa: E402
from replication_common import RDIR, DATA, call_paid  # noqa: E402

CUTOFF = datetime.fromisoformat("2026-09-26T23:59:59+00:00")
SEED = 20261013
PAPER_PAT = re.compile(r"paper-rollups|/paper/|paper/main\.tex|cain-replication|cain no-digest|cain output-stripped|"
                       r"replication_power_sim|rq3_heldout|2026-10-13 a fresh-session", re.I)
SYNTH = re.compile(r"\[rollup-summariser\] You are summarizing|reply with the single word:\s*ok", re.I)


def prefixes_used():
    used = set()
    for f in ("rq3_questions.jsonl", "rq3_heldout_questions.jsonl", "rq3_heldout_manifest.json"):
        t = (DATA / f).read_text(errors="replace")
        used |= set(re.findall(r"(?<![0-9a-f])([0-9a-f]{8})(?:-[0-9a-f]{4}-|_[a-z])", t))
        used |= set(re.findall(r'"session_prefix":\s*"([0-9a-f]{8})"', t))
    return used


def inspect(p):
    ts = None; synth = False; users = 0; tools = 0; paper = False; first_user = None
    with p.open(errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if ts is None and d.get("timestamp"):
                try: ts = datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00"))
                except Exception: pass
            m = d.get("message") or {}
            c = m.get("content")
            if d.get("type") == "user" and m.get("role") == "user":
                txt = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text") if isinstance(c, list) else ""
                if txt.strip() and not d.get("isMeta") and not txt.lstrip().startswith("<"):
                    users += 1
                    if first_user is None:
                        first_user = txt; synth = bool(SYNTH.search(txt[:400]))
            if d.get("type") == "assistant" and isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tools += 1
                        if PAPER_PAT.search(json.dumps(b.get("input", {}))):
                            paper = True
    return ts, synth, users, tools, paper


def scan():
    used = prefixes_used()
    now = time.time(); rows = []; loss = {}
    files = [p for p in PROJECTS_DIR.glob("**/*.jsonl") if "subagents" not in p.parts and p.stat().st_size >= 100_000]
    loss["ge_100KB"] = len(files)
    seen_pref = set()
    for p in sorted(files):
        pre = p.stem[:8]
        if pre in used: loss["already_used"] = loss.get("already_used", 0) + 1; continue
        if now - p.stat().st_mtime < 3600: loss["modified_last_hour"] = loss.get("modified_last_hour", 0) + 1; continue
        ts, synth, users, tools, paper = inspect(p)
        if ts is None: loss["not_after_cutoff"] = loss.get("not_after_cutoff", 0) + 1; continue
        if synth or tools < 5 or users < 2: loss["synthetic_or_low_activity"] = loss.get("synthetic_or_low_activity", 0) + 1; continue
        if paper: loss["touched_paper_or_experiment"] = loss.get("touched_paper_or_experiment", 0) + 1; continue
        if pre in seen_pref: loss["dup_prefix"] = loss.get("dup_prefix", 0) + 1; continue
        seen_pref.add(pre)
        rows.append({"session_prefix": pre, "session_id": p.stem, "transcript": str(p), "raw_bytes": p.stat().st_size,
                     "first_ts": ts.isoformat(), "tool_calls": tools, "user_turns": users})
    out = {"cutoff": CUTOFF.isoformat(), "loss": loss, "n_eligible": len(rows), "candidates": rows}
    (RDIR / "candidates.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(loss), "eligible", len(rows))


def generate(n_sessions, limit_cost):
    import part2_question_retention as gen
    from part2_question_retention import generate_questions, load_transcript, transcript_to_text
    gen.QUESTION_MAX_TOKENS = 8192   # same disclosed defect fix as rq3_heldout_questions.py
    gen.call_llm = lambda prompt, model, max_tokens=2048: call_paid("questiongen", prompt, model, max_tokens, 120)[0]
    cands = json.loads((RDIR / "candidates.json").read_text())["candidates"]
    rng = random.Random(SEED)
    order = cands[:]; rng.shuffle(order)
    (RDIR / "draw_order.json").write_text(json.dumps([c["session_prefix"] for c in order]))
    out = RDIR / "questions_raw.jsonl"
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lk = threading.Lock()
    from replication_common import gate
    gate("questiongen", 0.06 * n_sessions)

    def one(c):
        try:
            text = transcript_to_text(load_transcript(Path(c["transcript"])))
            qs = generate_questions(text, c["session_id"])
        except Exception as e:
            print("skip", c["session_prefix"], e); return 0
        if not qs: print("no questions", c["session_prefix"]); return 0
        with lk, out.open("a") as fh:
            for q in qs:
                fh.write(json.dumps({"session_id": c["session_id"], "session_prefix": c["session_prefix"],
                    "class": (q.get("class") or "").strip().lower(), "question": q.get("question", ""),
                    "ground_truth": q.get("ground_truth", ""), "transcript": c["transcript"]}) + "\n")
        return 1

    pos = 0; ok = 0
    while ok < n_sessions and pos < len(order):
        batch = order[pos:pos + (n_sessions - ok)]; pos += len(batch)
        with ThreadPoolExecutor(6) as ex:
            ok += sum(ex.map(one, batch))
        print("sessions with questions:", ok, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "generate"])
    ap.add_argument("--sessions", type=int, default=105)
    a = ap.parse_args()
    scan() if a.stage == "scan" else generate(a.sessions, None)
