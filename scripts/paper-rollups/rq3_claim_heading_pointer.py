#!/usr/bin/env python3
"""RQ3 read-budget re-run, one added arm: claim_heading_pointer.

Same 89 records, same pointer shape as rollup_pointer (path + heading, no body),
same answer prompt and answerer (rq3_read_budget_arms.answer), same grader
(grading_common, claude-sonnet-5, majority of three draws, draws 3-4 on a tie).
Only the heading text differs: each rollup note's `##` headings are rewritten by
claude-sonnet-5-5 as one-line claims, from the NOTE TEXT ONLY. The generator never
sees a question or a ground-truth answer.

Stages (resumable):  gate | headings | answer | grade | report | all
Gate (a): rebuild the pointer/hookshape contexts and regrade the stored answers
through this code path (cache hits are free), and confirm 0.0% / 31.5%.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "data" / "paper-rollups" / "regrade-2026-10-09"))
import rq3_read_budget_arms as rb  # noqa: E402
import regrade as rg  # noqa: E402
from rq3_raw_transcript_arm import call_llm, rates, wilson  # noqa: E402

gc = rg.gc
DATA = rb.DATA
MAIN_INDEX = f"file:{OBSIDIAN_DIR / 'pkm_index.db'}?mode=ro"
rb.INDEX_DB = MAIN_INDEX   # the worktree has no index; the main checkout's is the one the paper run used

OLD_CACHE = DATA / "regrade-2026-10-09" / "grade_cache.jsonl"
NEW_DIR = DATA / "claim-heading-pointer"
NEW_DIR.mkdir(exist_ok=True)
HEADINGS = NEW_DIR / "claim_headings.json"
ANSWERS = NEW_DIR / "claim_heading_pointer_arms.jsonl"
GRADES = NEW_DIR / "grade_cache.jsonl"
REPORT = NEW_DIR / "claim_heading_pointer_summary.json"
rg.LEDGER = NEW_DIR / "ledger.jsonl"
rg.CACHE = GRADES
GEN_MODEL = "claude-sonnet-5-5"
ARM = "claim_heading_pointer"

GEN_PROMPT = """Below is a note. Rewrite each `##` heading as a one-line claim that states the finding of that section, lowercase. Keep the number of headings and their order. Change nothing else.

Return ONLY a JSON array of objects, one per `##` heading in order: {{"original": "<heading text without ##>", "claim": "<claim>"}}.

NOTE:
{note}"""


def cache_all():
    c = rg.load_cache()
    if OLD_CACHE.exists():
        for l in OLD_CACHE.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                c.setdefault(r["key"], r)
    return c


def norm(h):
    return " ".join((h or "").lower().replace("#", " ").split()).strip(" :-—")


def note_headings(text):
    return [m.group(1).strip() for m in re.finditer(r"^##\s+(.+?)\s*$", text, re.M)]


def gen_headings(items):
    have = json.loads(HEADINGS.read_text()) if HEADINGS.exists() else {}
    notes = sorted({i["note"] for i in items})
    calls = 0
    for n in notes:
        if n in have:
            continue
        text = (rb.SESSIONS_DIR / Path(n).name).read_text()
        hs = note_headings(text)
        resp, usage = call_llm(GEN_PROMPT.format(note=text), model=GEN_MODEL, timeout=180)
        calls += 1
        try:
            arr = json.loads(re.search(r"\[.*\]", resp, re.S).group(0))
            claims = [a["claim"].strip() for a in arr]
        except Exception:
            claims = None
        have[n] = {"originals": hs, "claims": claims, "n_ok": claims is not None and len(claims) == len(hs),
                   "usage": usage}
        HEADINGS.write_text(json.dumps(have, indent=1))
    print("heading-generation calls this run:", calls)
    return have


def claim_pointer(con, item, have):
    """rollup_pointer rendering with the selected section's heading swapped for its claim."""
    note, q = item["note"], item["question"]
    heading, _ = rb.pick_section(con, note, q)
    if heading is None:
        heading, _ = rb.any_section(con, note)
    meta = have[note]
    mp = {norm(o): c for o, c in zip(meta["originals"], meta["claims"] or [])} if meta["n_ok"] else {}
    claim = mp.get(norm(heading))
    swapped = claim is not None
    detail = rb.clean(claim if swapped else (heading or ""))
    if not detail or rb.boring_heading(detail):
        detail = rb.clean(rb.note_description(con, note)) or rb.filename_title(note) or detail
    restated = bool(detail) and rb._restates_path(detail, note)
    if restated:
        detail = ""
    if len(detail) > rb.MAX_SNIPPET_CHARS:
        detail = detail[:rb.MAX_SNIPPET_CHARS].rstrip() + "…"
    ptr = "- %s — %s" % (note, detail) if detail else "- %s" % note
    return ptr, {"orig_heading": heading, "claim": claim, "swapped": swapped, "restated_dropped": restated}


def majority3(items, stage, cache, tag):
    """3 draws (+3,4 on a tie) through grading_common; returns {rid: (majority, draws)}."""
    draws = {}
    for d in range(3):
        draws[d] = rg.run_batch(items, rg.SONNET, d, stage, cache)
        for _ in range(2):
            bad = [i for i in items if draws[d][(i["arm"], i["record_id"])]["grade"] == gc.GRADE_ERROR]
            if not bad:
                break
            draws[d].update(rg.run_batch(bad, rg.SONNET, d, stage, cache, workers=4))
    out = {}
    for it in items:
        k = (it["arm"], it["record_id"])
        gs = [draws[d][k]["grade"] for d in range(3)]
        if gc.GRADE_ERROR not in gs and rg.majority(gs) is None:
            for d in (3, 4):
                r = rg.run_batch([it], rg.SONNET, d, stage, cache)
                gs.append(r[k]["grade"])
        valid = [g for g in gs if g != gc.GRADE_ERROR]
        out[it["record_id"]] = (rg.majority(valid) if len(valid) >= 2 else None, gs)
    return out


def gate(items, cache):
    stored = defaultdict(dict)
    for l in open(DATA / "rq3_read_budget_arms.jsonl"):
        r = json.loads(l)
        stored[r["arm"]][r["record_id"]] = r
    res = {}
    for arm in ("rollup_pointer", "rollup_hookshape"):
        mism = sum(1 for i in items if len(i["contexts"][arm]) != stored[arm][i["record_id"]]["context_chars"])
        gi = [{"arm": "rq3_read_budget_arms|" + arm, "record_id": i["record_id"], "question": i["question"],
               "gt": i["ground_truth"], "answer": stored[arm][i["record_id"]]["answer"],
               "confidence": stored[arm][i["record_id"]]["confidence"]} for i in items]
        n_model = len(rg.model_items(gi))
        mj = majority3(gi, "gate", cache, arm)
        mine = {k: v[0] for k, v in mj.items()}
        stored_maj = {k: stored[arm][k]["grade"] for k in mine}
        agree = sum(1 for k in mine if mine[k] == stored_maj[k])
        rec = sum(1 for g in mine.values() if g in rb.RECOVERABLE)
        res[arm] = {"context_len_mismatches_vs_stored": mism, "model_graded": n_model,
                    "regrade_matches_stored": "%d/%d" % (agree, len(mine)),
                    "recoverable_regrade": rec, "recoverable_stored":
                    sum(1 for g in stored_maj.values() if g in rb.RECOVERABLE), "n": len(mine),
                    "pct": round(100 * rec / len(mine), 1)}
        print("GATE", arm, res[arm])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "headings", "answer", "grade", "report", "all"])
    a = ap.parse_args()
    items, skipped = rb.build_items()
    assert len(items) == 89 and not skipped, (len(items), skipped)
    cache = cache_all()
    rg.TOTAL_CAP_USD = 1e9   # this script's own ceiling is enforced by call counts below

    gate_res = None
    if a.stage in ("gate", "all"):
        gate_res = gate(items, cache)
        (NEW_DIR / "gate_a.json").write_text(json.dumps(gate_res, indent=1))
        if a.stage == "gate":
            return
    have = gen_headings(items) if a.stage in ("headings", "answer", "grade", "report", "all") else None
    if a.stage == "headings":
        bad = [n for n, v in have.items() if not v["n_ok"]]
        print("notes", len(have), "heading-count mismatches/parse failures:", len(bad))
        return
    con = sqlite3.connect(MAIN_INDEX, uri=True)
    ctxs = {}
    for it in items:
        ctxs[it["record_id"]] = claim_pointer(con, it, have)
    print("claim pointers: swapped %d/89, restates-path dropped %d/89"
          % (sum(m["swapped"] for _, m in ctxs.values()), sum(m["restated_dropped"] for _, m in ctxs.values())))

    done = {json.loads(l)["record_id"] for l in open(ANSWERS)} if ANSWERS.exists() else set()
    for it in items:
        if it["record_id"] in done:
            continue
        ptr, meta = ctxs[it["record_id"]]
        ans, conf, usage = rb.answer(it["question"], ptr)
        if conf == "API_ERROR":
            print("API_ERROR", it["record_id"])
            continue
        rec = {"record_id": it["record_id"], "arm": ARM, "session_id": it["session_id"], "class": it["class"],
               "context": ptr, "context_chars": len(ptr), "confidence": conf, "answer": ans, "usage": usage,
               "model": rb.JUDGE_MODEL, **meta}
        with ANSWERS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    rows = {json.loads(l)["record_id"]: json.loads(l) for l in open(ANSWERS)}
    print("answers:", len(rows))
    if a.stage == "answer":
        return

    gi = [{"arm": "rq3_read_budget_arms|" + ARM, "record_id": it["record_id"], "question": it["question"],
           "gt": it["ground_truth"], "answer": rows[it["record_id"]]["answer"],
           "confidence": rows[it["record_id"]]["confidence"]} for it in items if it["record_id"] in rows]
    mj = majority3(gi, "claim", cache, ARM)
    graded = []
    for it in items:
        r = rows.get(it["record_id"])
        g, draws = mj.get(it["record_id"], (None, []))
        if r is None or g is None:
            continue
        graded.append({**r, "grade": g, "grade_draws": draws[:3]})
    (NEW_DIR / "claim_heading_pointer_graded.jsonl").write_text(
        "\n".join(json.dumps(r) for r in graded) + "\n")
    print("graded", len(graded), "of 89")

    # report
    stored_ptr = {}
    for l in open(DATA / "rq3_read_budget_arms.jsonl"):
        r = json.loads(l)
        if r["arm"] == "rollup_pointer":
            stored_ptr[r["record_id"]] = r
    ids = [r["record_id"] for r in graded]
    rt = rates(graded)
    sess = {r["record_id"]: r["session_id"] for r in graded}
    d = {i: int(next(g for g in graded if g["record_id"] == i)["grade"] in rb.RECOVERABLE)
         - int(stored_ptr[i]["grade"] in rb.RECOVERABLE) for i in ids}
    cl = defaultdict(list)
    for i in ids:
        cl[sess[i]].append(d[i])
    clusters = list(cl.values())
    est = sum(d.values()) / len(ids)
    rng = random.Random(0)
    bs = []
    for _ in range(10000):
        pick = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        tot = sum(len(c) for c in pick)
        bs.append(sum(sum(c) for c in pick) / tot)
    bs.sort()
    ci = [round(100 * bs[249], 1), round(100 * bs[9749], 1)]
    out = {"n": len(ids), "n_sessions": len(clusters), "arm": rt,
           "paired_vs_pointer": {"diff_pp": round(100 * est, 1), "cluster_bootstrap95_pp": ci,
                                 "claim_only": sum(1 for v in d.values() if v == 1),
                                 "pointer_only": sum(1 for v in d.values() if v == -1)},
           "claim_pointers_swapped": sum(r["swapped"] for r in graded),
           "restates_path_dropped": sum(r["restated_dropped"] for r in graded),
           "gate_a": json.loads((NEW_DIR / "gate_a.json").read_text()) if (NEW_DIR / "gate_a.json").exists() else None}
    REPORT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n", "n_sessions", "paired_vs_pointer", "claim_pointers_swapped",
                                           "restates_path_dropped")}, indent=1))
    print("recoverable", rt["recoverable"], "hallucination_of_attempted", rt["hallucination_of_attempted"])
    print("grades", {g: rt[g]["n"] for g in gc.GRADES})


if __name__ == "__main__":
    main()
