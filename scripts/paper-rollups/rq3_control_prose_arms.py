#!/usr/bin/env python3
"""No-op control + length-matched prose arms + held-back-225 prose arm.

Subcommands (all token lengths are tiktoken cl100k_base, nothing else):
  sizes      no model calls: cl100k sizes of production notes and prose artifacts
  control    regenerate rollups with the production summarise() (arm prod_pipeline_control)
  prose      prose_matched_per_record / prose_matched_median on the 124
  heldback   prose_only_heldback225 on the 225 held-back records
  analyze    tables (no model calls)
Every model-calling subcommand takes --limit N (N records) and appends to
data/paper-rollups/rq3_arm_<arm>.jsonl (artifacts) and ..._graded.jsonl (grades).
A spend ledger (data/paper-rollups/rq3_control_prose_spend.json) enforces the $12 cap.
"""
import argparse
import json
import math
import statistics
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROLLUP_SRC_DIR))

import tiktoken  # noqa: E402
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, call_llm, extract_confidence, grade_answer,
    index_transcripts, load_records, record_id, resolve, wilson,
)
from rq3_prose_only_arm import prose_only_context  # noqa: E402
from rq3_uniform_rollup_arm import find_rollup  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
JUDGE_MODEL = "claude-sonnet-5"
GEN_MODEL = "claude-haiku-4-5@20251001"  # same as the prod_input_2x arm's generator
PRICE_IN, PRICE_OUT = 3.00e-6, 15.00e-6  # conservative: sonnet rates for every call
CAP_USD = 12.0
LEDGER = DATA / "rq3_control_prose_spend.json"
ANSWER_CONTEXT_TOKENS = 1_000_000  # answer-model context limit used for skip decisions
MARK = "…"


def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))


def arm_paths(arm):
    return DATA / f"rq3_arm_{arm}.jsonl", DATA / f"rq3_arm_{arm}_graded.jsonl"


def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ---------------- spend ledger ----------------
def ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"prompt": 0, "completion": 0, "usd": 0.0, "by_arm": {}}


def spend(arm, usage_list):
    L = ledger()
    for u in usage_list:
        p, c = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        L["prompt"] += p
        L["completion"] += c
        usd = p * PRICE_IN + c * PRICE_OUT
        L["usd"] += usd
        a = L["by_arm"].setdefault(arm, {"prompt": 0, "completion": 0, "usd": 0.0})
        a["prompt"] += p
        a["completion"] += c
        a["usd"] += usd
    LEDGER.write_text(json.dumps(L, indent=1))
    return L["usd"]


def guard(est_usd):
    L = ledger()
    if L["usd"] + est_usd > CAP_USD:
        print(f"STOP: spent ${L['usd']:.2f} + next est ${est_usd:.2f} would exceed ${CAP_USD}")
        sys.exit(3)


# ---------------- data ----------------
def graded_ids():
    return set(json.loads(SUMMARY.read_text())["graded_record_ids"])


def recs124():
    g = graded_ids()
    return sorted([r for r in load_records(QUESTIONS) if record_id(r) in g], key=record_id)


def recs225():
    g = graded_ids()
    return sorted([r for r in load_records(QUESTIONS) if record_id(r) not in g], key=record_id)


_by_prefix = None


def transcript(r):
    global _by_prefix
    if _by_prefix is None:
        _by_prefix = index_transcripts()
    m = SLUG.match(r.get("session_id", ""))
    return resolve(m.group(1), _by_prefix) if m else None


_prose_cache = {}


def prose_for(r):
    p = transcript(r)
    if p is None:
        return None
    if p not in _prose_cache:
        _prose_cache[p] = prose_only_context(p.read_text(errors="replace"), keep_continuation=False)[0]
    return _prose_cache[p]


def prod_note(r):
    p = find_rollup(r["session_id"])
    return p.read_text(errors="replace") if p else None


# ---------------- answering (prompts verbatim) ----------------
def answer_rollup(question, note):
    prompt = f"""Someone asks: "{question}"

Below is a compressed note summarizing an agent session. Answer the question based ONLY on this note. If the note does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

NOTE:
{note}

Provide your answer and confidence now:"""
    resp, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=180)
    if not resp:
        return "", "API_ERROR", usage
    a, c = extract_confidence(resp)
    return a, c, usage


def answer_prose(question, prose):
    prompt = f"""Someone asks: "{question}"

Below is a transcript with only the prose (user and assistant text). Tool calls and their results have been removed. Answer the question based ONLY on this prose. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

PROSE:
{prose}

Provide your answer and confidence now:"""
    resp, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not resp:
        return "", "API_ERROR", usage
    a, c = extract_confidence(resp)
    return a, c, usage


def grade_row(arm, r, artifact, artifact_tokens, fn):
    answer, conf, usage = fn(r["question"], artifact)
    if conf == "API_ERROR":
        return None, [usage]
    grade = grade_answer(answer, r.get("ground_truth", ""), conf)
    up = answer.upper()
    if conf != "NOT_ANSWERABLE" and ("NOT ANSWERABLE" in up or "NOT_ANSWERABLE" in up):
        cc, cg = "NOT_ANSWERABLE", "absent"
    else:
        cc, cg = conf, grade
    return {"arm": arm, "record_id": record_id(r), "session_id": r.get("session_id"),
            "class": r.get("class"), "question": r.get("question"),
            "ground_truth": r.get("ground_truth"), "artifact_cl100k_tokens": artifact_tokens,
            "answer": answer, "confidence": conf, "grade": grade,
            "corrected_confidence": cc, "corrected_grade": cg,
            "usage": usage, "model": JUDGE_MODEL}, [usage]


def run_grading(arm, items, fn, limit):
    """items: list of (record, artifact_str, artifact_tokens). Resumable."""
    _, gpath = arm_paths(arm)
    done = {x["record_id"] for x in rows(gpath)}
    todo = [it for it in items if record_id(it[0]) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{arm}] to grade {len(todo)} (already {len(done)})")
    for i, (r, art, nt) in enumerate(todo, 1):
        est = (nt + 600) * PRICE_IN + 400 * PRICE_OUT + 0.002
        guard(est)
        row, us = grade_row(arm, r, art, nt, fn)
        # grade_answer() discards its usage; ledger an estimate of one grading call
        # (~ground truth + answer + instructions in, one word out) unless it short-circuited
        if row is not None and row["confidence"] != "NOT_ANSWERABLE" and row["answer"]:
            us = us + [{"prompt_tokens": 350, "completion_tokens": 5}]
        total = spend(arm, us)
        if row is None:
            print(f"[{i}/{len(todo)}] {record_id(r)} API_ERROR (not recorded)")
            continue
        with gpath.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(todo)}] {row['record_id']} tok={nt} {row['confidence']:18s} {row['grade']:18s} corr={row['corrected_grade']} spend=${total:.3f}")


# ---------------- subcommands ----------------
def cmd_sizes(a):
    r124 = recs124()
    sess = {}
    for r in r124:
        sess.setdefault(r["session_id"], r)
    pt = {s: ntok(prod_note(r)) for s, r in sess.items()}
    qt = [pt[r["session_id"]] for r in r124]
    print(f"124 production note cl100k per question: median {statistics.median(qt)} mean {statistics.mean(qt):.0f} min {min(qt)} max {max(qt)}")
    pr = {s: ntok(prose_for(r)) for s, r in sess.items()}
    print(f"124 prose full per question: median {statistics.median([pr[r['session_id']] for r in r124])} max {max(pr.values())}")
    r225 = recs225()
    s225 = {}
    for r in r225:
        s225.setdefault(r["session_id"], r)
    miss = [s for s, r in s225.items() if transcript(r) is None]
    print(f"225 records, {len(s225)} sessions, transcripts missing: {miss}")
    p225 = {s: ntok(prose_for(r)) for s, r in s225.items() if transcript(r)}
    per_q = [p225[r["session_id"]] for r in r225 if r["session_id"] in p225]
    print(f"225 prose cl100k per question: median {statistics.median(per_q)} max {max(per_q)} total {sum(per_q):,}")
    print(f"est input cost 225 @ $3/M: ${sum(per_q)*PRICE_IN:.2f}")
    over = [s for s, t in p225.items() if t > ANSWER_CONTEXT_TOKENS - 2000]
    print("sessions over context:", over)
    for s, t in sorted(p225.items(), key=lambda x: -x[1])[:8]:
        print(f"  {s[:60]} {t:,}")


class _Shim:
    """anthropic-client shape for rollup_summarise.summarise(client=...), routed through
    the gateway. Records usage; a failed call returns '' (summarise degrades to a
    'not valid JSON' stub, which summary_failed() catches)."""
    def __init__(self):
        self.usage = []
        self.messages = self

    def create(self, model, max_tokens, messages):
        text, u = call_llm(messages[0]["content"], model=model, max_tokens=max_tokens)
        self.usage.append(u)
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])


def render_artifact(session_id, path, epochs, summaries, meta):
    import rollup_note
    from rollup_types import Session
    s = Session(session_id=session_id, agent="claude-code", transcript=str(path))
    return rollup_note.render(s, epochs, summaries, meta, signals=[], trace_dir=None)


def cmd_control(a):
    import rollup_read
    import rollup_summarise as rs
    from rollup_types import split_epochs
    arm = "prod_pipeline_control"
    gen_path, _ = arm_paths(arm)
    r124 = recs124()
    sess = {}
    for r in r124:
        sess.setdefault(r["session_id"], r)
    gens = {x["session_id"]: x for x in rows(gen_path)}
    sids = list(sess)
    if a.limit:
        # smoke: the 3 records' sessions only
        sids = list(dict.fromkeys(r["session_id"] for r in r124[:a.limit]))
    for sid in sids:
        if sid in gens:
            continue
        r = sess[sid]
        p = transcript(r)
        turns = [t for t in rollup_read.read_turns(p) if t.text.strip()]
        epochs = split_epochs(turns)
        # pre-flight estimate from the real prompts (chars/2.7 only to size the cap check)
        est_in = sum(len(rs.build_summary_prompt(c, rs.HAIKU_WINDOW - 2000)) for e in epochs
                     for c in rs.chunk_epoch(e, rs.HAIKU_WINDOW - 2000)) / 2.7
        guard(est_in * 1e-6 * 1.0 * 1.5 + 0.02)  # haiku-rate would be lower; this is conservative
        shim = _Shim()
        import os
        os.environ["PKM_ROLLUP_MODEL"] = GEN_MODEL
        summaries = [rs.summarise(e, client=shim) for e in epochs]
        failed = [rs.summary_failed(x) for x in summaries]
        total = spend(arm + "_gen", shim.usage)
        if any(failed):
            print(f"GEN FAILED {sid}: {summaries}")
            continue
        meta = dict(rollup_read.session_meta(p), subagents=0, turns=len(turns))
        art = render_artifact(sid, p, epochs, summaries, meta)
        with gen_path.open("a") as fh:
            fh.write(json.dumps({"session_id": sid, "model": GEN_MODEL, "n_epochs": len(epochs),
                                 "n_turns": len(turns), "summaries": summaries, "artifact": art,
                                 "artifact_cl100k_tokens": ntok(art), "usage": shim.usage}) + "\n")
        print(f"gen {sid[:50]} epochs={len(epochs)} calls={len(shim.usage)} art_tok={ntok(art)} spend=${total:.3f}")
    gens = {x["session_id"]: x for x in rows(gen_path)}
    items = [(r, gens[r["session_id"]]["artifact"], gens[r["session_id"]]["artifact_cl100k_tokens"])
             for r in r124 if r["session_id"] in gens]
    run_grading(arm, items, answer_rollup, a.limit)


def truncate_tokens(text, n):
    ids = ENC.encode(text, disallowed_special=())
    if len(ids) <= n:
        return text
    return ENC.decode(ids[:n]) + MARK


def cmd_prose(a):
    r124 = recs124()
    prod_t = {}
    for r in r124:
        prod_t[record_id(r)] = ntok(prod_note(r))
    med = int(statistics.median(prod_t.values()))
    print(f"production note cl100k tokens: median {med}")
    for arm, lenf in (("prose_matched_per_record", lambda r: prod_t[record_id(r)]),
                      ("prose_matched_median", lambda r: med)):
        items = []
        for r in r124:
            full = prose_for(r)
            art = truncate_tokens(full, lenf(r))
            items.append((r, art, ntok(art)))
        if a.limit:
            items = items[:a.limit]
        run_grading(arm, items, answer_prose, 0)
        # artifact file
        apath, _ = arm_paths(arm)
        if not apath.exists() or not a.limit:
            apath.write_text("".join(json.dumps({"record_id": record_id(r), "session_id": r["session_id"],
                                                 "target_tokens": lenf(r), "artifact_cl100k_tokens": nt,
                                                 "artifact": art}) + "\n" for r, art, nt in items))


def cmd_heldback(a):
    arm = "prose_only_heldback225"
    items, skipped = [], []
    for r in recs225():
        full = prose_for(r)
        if full is None:
            skipped.append({"record_id": record_id(r), "reason": "no_transcript"})
            continue
        nt = ntok(full)
        if nt > ANSWER_CONTEXT_TOKENS - 2000:
            skipped.append({"record_id": record_id(r), "reason": f"exceeds_context {nt}"})
            continue
        items.append((r, full, nt))
    apath, _ = arm_paths(arm)
    if not apath.exists():
        apath.write_text(json.dumps({"skipped": skipped, "n_graded_candidates": len(items)}) + "\n")
    print(f"heldback: {len(items)} to run, skipped {skipped}")
    if a.limit:
        # smoke: shortest artifacts first so a smoke test is cheap
        items = sorted(items, key=lambda x: x[2])
    run_grading(arm, items, answer_prose, a.limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sizes", "control", "prose", "heldback"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"sizes": cmd_sizes, "control": cmd_control, "prose": cmd_prose, "heldback": cmd_heldback}[a.cmd](a)


if __name__ == "__main__":
    main()
