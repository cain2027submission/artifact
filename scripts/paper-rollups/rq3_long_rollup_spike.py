#!/usr/bin/env python3
"""SPIKE: a Sonnet-written rollup at ~5% of raw transcript tokens, compared with every
existing RQ3 arm on the same questions.

Does a model-written summary at ~5% of raw do as well as the output-stripped
transcript at a similar size (~6.5% of raw)?

Design (see data/paper-rollups/rq3_long_rollup_spike.md):
  generator  claude-sonnet-5-5, no temperature, input = strip_transcript() of the
             session (the output-stripped arm's construction), prompt = the production
             build_summary_prompt() with the three per-section sentence counts replaced
             by "up to N tokens", N = 5% of the session's raw cl100k tokens.
             If the stripped input has fewer than N tokens, N = 0.6 x stripped tokens.
             N is capped at MAX_TARGET (a model cannot write a 90k-token summary).
  answerer   claude-sonnet-5, the rollup arm's answer prompt.
  judge      claude-sonnet-5, 3 draws + majority (2 more on a 1-1-1 split), as
             rq3_majority_vote_grades.py. By-rule absent when NOT_ANSWERABLE.

Modes:
  --table                     per-session size/delegation table, no model calls
  --select                    pick the 8 sessions (fixed seed), print them
  --generate [--only SID...]  generate rollups (resumable)
  --grade    [--only SID...]  answer + 3-draw judge the sessions' frozen questions
  --analyse                   no model calls: per-arm table + paired tests
Run from a checkout that has data/paper-rollups (the worktree is origin/main; the
uncommitted peer-spike data is read from the main checkout, READ-ONLY).
"""

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROLLUP_SRC_DIR))

from rq3_raw_transcript_arm import (  # noqa: E402
    call_llm, extract_confidence, wilson,
)
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402
from rq3_regrade_rollup_arm import FIXED_ORDER, grading_prompt, parse  # noqa: E402
from grading_common import GRADE_ERROR, GradeError  # noqa: E402


def parse_draw(resp, order):
    """One draw, or GradeError: a failed draw must not enter the vote."""
    g = parse(resp, order)
    if g == GRADE_ERROR:
        raise GradeError("empty or unparseable grader reply", resp)
    return g
from rq3_rollup_from_stripped_spike import stripped_to_epoch  # noqa: E402
import rollup_summarise as rs  # noqa: E402

REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
MAIN_DATA = DATA_DIR  # read-only fallback

GEN_OUT = DATA / "rq3_long_rollup_spike.jsonl"
GRADED = DATA / "rq3_long_rollup_spike_graded.jsonl"
SUMMARY_OUT = DATA / "rq3_long_rollup_spike_summary.json"
SESSIONS_CACHE = DATA / "rq3_long_rollup_spike_sessions.json"

GEN_MODEL = "claude-sonnet-5"   # claude-sonnet-5-5 returns finish_reason=content_filter on the unchanged production prompt
ANSWER_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-sonnet-5"
SEED = 42
FRACTION = 0.05
SHORT_INPUT_FRACTION = 0.60
MAX_TARGET = 20_000          # output-token cap on N (documented deviation)
CHUNK_BUDGET = 600_000       # estimate_tokens units: force a single call
PRICE_IN, PRICE_OUT = 3.00 / 1e6, 15.00 / 1e6   # assumed Sonnet pricing (rq5_generation_cost.py)

_ENC = None


def ntok(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode_ordinary(text))


def rows(p: Path):
    if not p.exists():
        p = MAIN_DATA / p.name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def src(name: str) -> Path:
    p = DATA / name
    return p if p.exists() else MAIN_DATA / name


# ------------------------------------------------------------------ sessions
def session_table(refresh=False):
    if SESSIONS_CACHE.exists() and not refresh:
        return json.loads(SESSIONS_CACHE.read_text())
    strip_rows = rows(src("rq3_questions_stripped_arm.jsonl"))
    raw_cov = {r["session_id"] for r in rows(src("rq3_majority_raw.jsonl"))}
    deleg = json.loads(src("rq3_delegation_reweight.json").read_text())["sample_composition"]["per_session"]
    qcount = Counter(r["session_id"] for r in strip_rows)
    out = {}
    for r in strip_rows:
        sid = r["session_id"]
        if sid in out:
            continue
        path = Path(r["transcript"])
        raw = path.read_text(errors="replace")
        stripped, st = strip_transcript(raw)
        out[sid] = {"session_id": sid, "transcript": str(path), "n_questions": qcount[sid],
                    "raw_tokens": ntok(raw), "stripped_tokens": ntok(stripped),
                    "tool_calls": st["tool_calls"], "raw_arm_covered": sid in raw_cov,
                    "delegating": bool(deleg[sid]["delegating"])}
        print(f"  {sid[:60]:60s} raw {out[sid]['raw_tokens']:>9,} strip {out[sid]['stripped_tokens']:>8,} "
              f"tools {st['tool_calls']:>4} raw_cov {sid in raw_cov} deleg {out[sid]['delegating']}")
    SESSIONS_CACHE.write_text(json.dumps(out, indent=1))
    return out


def select(table):
    elig = sorted((s for s in table.values() if s["tool_calls"] > 0), key=lambda s: s["raw_tokens"])
    n = len(elig)
    tiers = {"small": elig[: n // 3], "medium": elig[n // 3: 2 * n // 3], "large": elig[2 * n // 3:]}
    quota = {"small": 3, "medium": 3, "large": 2}
    rng = random.Random(SEED)
    for attempt in range(10_000):
        pick = []
        for t, k in quota.items():
            pick += rng.sample(tiers[t], k)
        if (sum(s["raw_arm_covered"] for s in pick) >= 5 and sum(s["delegating"] for s in pick) >= 2
                and all(s["n_questions"] >= 3 for s in pick)):
            for t, ss in tiers.items():
                for s in pick:
                    if s in ss:
                        s["tier"] = t
            return sorted(pick, key=lambda s: s["raw_tokens"]), attempt
    raise SystemExit("no selection satisfied the constraints")


# ------------------------------------------------------------------ generation
_LEN = None


def build_prompt(stripped: str, n_target: int) -> str:
    import re
    epoch = stripped_to_epoch(stripped)
    chunks = rs.chunk_epoch(epoch, CHUNK_BUDGET)
    assert len(chunks) == 1, f"{len(chunks)} chunks"
    p = rs.build_summary_prompt(chunks[0], CHUNK_BUDGET)
    head, sep, tail = p.partition("\n\nTranscript:\n")
    head = re.sub(r" \(\d-\d sentences\)", "", head)
    head += (f"\n\nLength: this note is deliberately long. Write about {n_target} tokens "
             f"(roughly {int(n_target * 0.75)} words) in total across the fields, never fewer than "
             f"{int(n_target * 0.8)} tokens. Each of the intent, thinking and outcome fields should be several "
             f"paragraphs, walking through the session in order: every phase, each file touched, "
             f"commands run, numbers and identifiers, decisions with their reasons, and each dead end. "
             f"Do not compress to a short paragraph.")
    return head + sep + tail


def target_for(s):
    n = int(FRACTION * s["raw_tokens"])
    capped = n > MAX_TARGET
    n = min(n, MAX_TARGET)
    short = s["stripped_tokens"] < n
    if short:
        n = int(SHORT_INPUT_FRACTION * s["stripped_tokens"])
    return n, short, capped


def generate(sessions):
    done = {r["session_id"] for r in rows(GEN_OUT)} if GEN_OUT.exists() else set()
    for s in sessions:
        if s["session_id"] in done:
            continue
        raw = Path(s["transcript"]).read_text(errors="replace")
        stripped, _ = strip_transcript(raw)
        n, short, capped = target_for(s)
        prompt = build_prompt(stripped, n)
        for _try in range(3):
            text, usage = call_llm(prompt, model=GEN_MODEL, timeout=900, max_tokens=max(2048, int(n * 1.6)))
            if text and usage.get("finish_reason") == "stop":
                break
            text = ""
        if not text:
            print(f"GENERATION FAILED {s['session_id']} {usage}")
            continue
        rec = {"session_id": s["session_id"], "tier": s.get("tier"), "model": GEN_MODEL,
               "raw_tokens": s["raw_tokens"], "stripped_tokens": s["stripped_tokens"],
               "target_tokens": n, "target_is_short_input_rule": short, "target_capped_at_max": capped,
               "rollup_cl100k_tokens": ntok(text), "rollup_chars": len(text),
               "pct_of_raw": 100 * ntok(text) / s["raw_tokens"],
               "completion_tokens": usage.get("completion_tokens"), "usage": usage,
               "rollup_raw": text}
        with GEN_OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{s['session_id'][:50]} target {n} actual {rec['rollup_cl100k_tokens']} "
              f"({rec['pct_of_raw']:.2f}% of raw) usage {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}")


# ------------------------------------------------------------------ answer + grade
def answer_from_rollup(question: str, rollup: str):
    prompt = f"""Someone asks: "{question}"

Below is a compressed note summarizing an agent session. Answer the question based ONLY on this note. If the note does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

NOTE:
{rollup}

Provide your answer and confidence now:"""
    response, usage = call_llm(prompt, model=ANSWER_MODEL, timeout=300)
    if not response:
        return "", "API_ERROR", usage
    a, c = extract_confidence(response)
    return a, c, usage


def majority_grade(answer: str, truth: str, question: str):
    """Raises GradeError if any draw fails; the caller leaves the record unwritten."""
    prompt = grading_prompt(answer, truth, question)
    draws = []
    for _ in range(3):
        resp, u = call_llm(prompt, model=JUDGE_MODEL, timeout=120)
        draws.append(parse_draw(resp, FIXED_ORDER))
    c = Counter(draws)
    top, k = c.most_common(1)[0]
    if sum(1 for n in c.values() if n == k) > 1:
        for _ in range(2):
            resp, u = call_llm(prompt, model=JUDGE_MODEL, timeout=120)
            draws.append(parse_draw(resp, FIXED_ORDER))
        c = Counter(draws)
        top, k = c.most_common(1)[0]
    unresolved = sum(1 for n in c.values() if n == k) > 1
    return (None if unresolved else top), draws, unresolved


def grade(sessions):
    rolls = {r["session_id"]: r["rollup_raw"] for r in rows(GEN_OUT)}
    done = {r["record_id"] for r in rows(GRADED)} if GRADED.exists() else set()
    qs = rows(src("rq3_rollup_uniform.jsonl"))
    want = {s["session_id"] for s in sessions}
    todo = [q for q in qs if q["session_id"] in want and q["session_id"] in rolls and q["record_id"] not in done]
    print(f"to grade {len(todo)}")
    for i, q in enumerate(todo, 1):
        ans, conf, usage = answer_from_rollup(q["question"], rolls[q["session_id"]])
        if conf == "API_ERROR":
            print(f"[{i}] {q['record_id']} API_ERROR (retry on rerun)")
            continue
        if conf == "NOT_ANSWERABLE" or not ans:
            g, draws, unres = "absent", [], False
            by_rule = True
        else:
            try:
                g, draws, unres = majority_grade(ans, q["ground_truth"],
                                                 q["question"])
            except GradeError:
                print(f"[{i}] {q['record_id']} grade_error (not recorded, retry on rerun)")
                continue
            by_rule = False
        rec = {"record_id": q["record_id"], "session_id": q["session_id"], "class": q["class"],
               "question": q["question"], "ground_truth": q["ground_truth"], "answer": ans,
               "confidence": conf, "by_rule_absent": by_rule, "draws": draws, "grade": g,
               "unresolved": unres, "single_draw_grade": (draws[0] if draws else "absent"),
               "usage": usage, "answer_model": ANSWER_MODEL, "judge_model": JUDGE_MODEL}
        with GRADED.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{i}/{len(todo)}] {q['record_id']} {conf:17s} {g} {draws}")


# ------------------------------------------------------------------ analysis
REC = ("correct", "partially_correct")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# arm label -> (file, grade-source). grade-source: ("file", key) = grade key in the arm's
# own file; ("maj", file) = three-draw majority file; usage always from the answer file.
ARMS = {
    "long Sonnet rollup (new)": None,
    "production rollup": ("rq3_rollup_uniform.jsonl", "rq3_majority_uniform.jsonl"),
    "dialogue record": ("rq3_verbatim_arm.jsonl", None),
    "raw transcript": ("rq3_questions_raw_arm.jsonl", "rq3_majority_raw.jsonl"),
    "output-stripped": ("rq3_questions_stripped_arm.jsonl", "rq3_majority_stripped.jsonl"),
    "call-only": ("rq3_tool_inputs_only_arm.jsonl", None),
    "prose-only": ("rq3_prose_only_arm.jsonl", None),
    "answer-only": ("rq3_answer_only_arm.jsonl", None),
    "prompts-only": ("rq3_prompts_only_arm.jsonl", None),
    "zero context": ("rq3_zero_context_arm.jsonl", None),
    "action record": ("rq3_action_record_arm.jsonl", None),
    "prose cut to 933 tok": ("rq3_arm_prose_matched_median_graded.jsonl", None),
}


def load_arm(label, ids):
    """-> {record_id: (grade, prompt_tokens)} restricted to ids."""
    f, maj = ARMS[label]
    base = {r["record_id"]: r for r in rows(src(f))}
    mj = {r["record_id"]: r for r in rows(src(maj))} if maj else {}
    out = {}
    for rid in ids:
        if rid not in base:
            continue
        r = base[rid]
        g = mj[rid]["grade"] if rid in mj else r.get("grade")
        if g is None:      # unresolved majority vote: fall back to the arm's own draw
            g = r.get("grade")
        out[rid] = (g, (r.get("usage") or {}).get("prompt_tokens"))
    return out


def cluster_boot(pairs, sess, iters=10_000, seed=SEED):
    """pairs: {rid: (a_rec, b_rec)}; diff = mean(a) - mean(b), resampling sessions."""
    by = {}
    for rid, (a, b) in pairs.items():
        by.setdefault(sess[rid], []).append((a, b))
    ks = list(by)
    rng = random.Random(seed)
    ds = []
    for _ in range(iters):
        sa = sb = n = 0
        for k in (rng.choice(ks) for _ in ks):
            for a, b in by[k]:
                sa += a
                sb += b
                n += 1
        ds.append(100 * (sa - sb) / n)
    ds.sort()
    return ds[int(0.025 * iters)], ds[int(0.975 * iters) - 1]


def analyse():
    gen = rows(GEN_OUT)
    new = {r["record_id"]: r for r in rows(GRADED)}
    ids = sorted(new)
    sess = {rid: new[rid]["session_id"] for rid in ids}
    sessions = sorted(set(sess.values()))
    tab = json.loads(SESSIONS_CACHE.read_text())
    summary = {"n_questions": len(ids), "n_sessions": len(sessions), "sessions": sessions,
               "caveat": "8 sessions: direction only, not an estimate of the corpus rate"}

    # generation sizes
    summary["generation"] = {
        "median_target_tokens": statistics.median(g["target_tokens"] for g in gen),
        "median_actual_tokens": statistics.median(g["rollup_cl100k_tokens"] for g in gen),
        "median_pct_of_raw": statistics.median(g["pct_of_raw"] for g in gen),
        "per_session": [{k: g[k] for k in ("session_id", "tier", "raw_tokens", "stripped_tokens", "target_tokens",
                                           "target_is_short_input_rule", "target_capped_at_max",
                                           "rollup_cl100k_tokens", "pct_of_raw")} for g in gen]}

    grades = {"long Sonnet rollup (new)": {rid: (new[rid]["grade"] if new[rid]["grade"] else "absent",
                                                 new[rid]["usage"].get("prompt_tokens")) for rid in ids}}
    for lab in list(ARMS)[1:]:
        grades[lab] = load_arm(lab, ids)

    def stats(g):
        n = len(g)
        k = sum(v[0] in REC for v in g.values())
        wrong = sum(v[0] == "wrong" for v in g.values())
        absent = sum(v[0] == "absent" for v in g.values())
        att = n - absent
        toks = [v[1] for v in g.values() if v[1]]
        lo, hi = wilson(k, n) if n else (0, 0)
        wl, wh = wilson(wrong, att) if att else (0, 0)
        return {"n": n, "recoverable": k, "recoverable_pct": round(100 * k / n, 1) if n else None,
                "wilson95": [lo, hi], "wrong": wrong, "attempted": att,
                "wrong_of_attempted_pct": round(100 * wrong / att, 1) if att else None,
                "wrong_of_attempted_wilson95": [wl, wh],
                "median_read_tokens": statistics.median(toks) if toks else None}

    summary["arms"] = {lab: stats(g) for lab, g in grades.items()}
    summary["new_arm_single_draw_sensitivity"] = {
        "recoverable_pct": round(100 * sum(new[r]["single_draw_grade"] in REC for r in ids) / len(ids), 1)}

    paired = {}
    for lab in ("output-stripped", "production rollup", "prose-only"):
        common = [r for r in ids if r in grades[lab]]
        a = {r: grades["long Sonnet rollup (new)"][r][0] in REC for r in common}
        b = {r: grades[lab][r][0] in REC for r in common}
        n_only = sum(a[r] and not b[r] for r in common)
        o_only = sum(b[r] and not a[r] for r in common)
        lo, hi = cluster_boot({r: (a[r], b[r]) for r in common}, sess)
        paired[f"new vs {lab}"] = {
            "n": len(common), "both": sum(a[r] and b[r] for r in common), "new_only": n_only,
            "other_only": o_only, "neither": sum(not a[r] and not b[r] for r in common),
            "diff_pct_points": round(100 * (sum(a.values()) - sum(b.values())) / len(common), 1),
            "mcnemar_exact_p": round(mcnemar_exact(n_only, o_only), 4),
            "session_cluster_bootstrap95_pp": [round(lo, 1), round(hi, 1)],
            "bootstrap": "10000 resamples of sessions, seed 42; 8 sessions so direction only"}
    summary["paired"] = paired

    u_in = sum(g["usage"].get("prompt_tokens", 0) for g in gen)
    u_out = sum(g["usage"].get("completion_tokens", 0) for g in gen)
    a_in = sum(r["usage"].get("prompt_tokens", 0) for r in new.values())
    a_out = sum(r["usage"].get("completion_tokens", 0) for r in new.values())
    n_judge = sum(len(r["draws"]) for r in new.values())
    j_in = n_judge * 450          # estimated grading prompt, judge calls not usage-logged
    j_out = n_judge * 5
    summary["spend_usd_estimated_at_3_15_per_M"] = {
        "generation": round(u_in * PRICE_IN + u_out * PRICE_OUT, 3),
        "answering": round(a_in * PRICE_IN + a_out * PRICE_OUT, 3),
        "judging_estimated": round(j_in * PRICE_IN + j_out * PRICE_OUT, 3),
        "gen_tokens_in_out": [u_in, u_out], "answer_tokens_in_out": [a_in, a_out], "judge_calls": n_judge}
    summary["spend_usd_estimated_at_3_15_per_M"]["total"] = round(
        sum(summary["spend_usd_estimated_at_3_15_per_M"][k] for k in ("generation", "answering", "judging_estimated")), 2)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=1))

    print(f"questions {len(ids)} sessions {len(sessions)}")
    print(json.dumps(summary["generation"]["median_target_tokens"]), summary["generation"]["median_actual_tokens"],
          summary["generation"]["median_pct_of_raw"])
    print(f"{'arm':28s} {'n':>4s} {'rec%':>6s} {'wilson':>14s} {'medtok':>8s} {'wrong/att':>10s}")
    for lab, s in summary["arms"].items():
        print(f"{lab:28s} {s['n']:4d} {s['recoverable_pct']!s:>6s} {str(s['wilson95']):>14s} "
              f"{s['median_read_tokens']!s:>8s} {s['wrong']}/{s['attempted']}={s['wrong_of_attempted_pct']}")
    for k, v in paired.items():
        print(k, v)
    print("sensitivity single draw", summary["new_arm_single_draw_sensitivity"])
    print("spend", summary["spend_usd_estimated_at_3_15_per_M"])


def main():
    ap = argparse.ArgumentParser()
    for f in ("table", "select", "generate", "grade", "analyse", "estimate"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    table = session_table()
    chosen, attempt = select(table)
    if a.only:
        chosen = [s for s in chosen if any(s["session_id"].startswith(o) or o in s["session_id"] for o in a.only)]
    if a.table or a.select:
        for s in chosen:
            n, short, cap = target_for(s)
            print(f"{s['tier']:6s} {s['session_id'][:62]:62s} raw {s['raw_tokens']:>9,} strip {s['stripped_tokens']:>7,} "
                  f"q {s['n_questions']} tools {s['tool_calls']} deleg {s['delegating']} rawcov {s['raw_arm_covered']} "
                  f"target {n} short={short} cap={cap}")
        print("draw attempt", attempt)
    if a.estimate:
        gi = go = 0
        for s in chosen:
            n, _, _ = target_for(s)
            gi += s["stripped_tokens"] * 1.15 + 1500
            go += n * 1.1
        qs = sum(s["n_questions"] for s in chosen)
        ai = sum(s["n_questions"] * (min(target_for(s)[0], s["raw_tokens"]) * 1.2 + 300) for s in chosen)
        est = gi * PRICE_IN + go * PRICE_OUT + ai * PRICE_IN + qs * 4 * 450 * PRICE_IN + qs * 400 * PRICE_OUT
        print(f"gen in {gi:,.0f} out {go:,.0f}; answer in {ai:,.0f}; questions {qs}; estimated total ${est:.2f}")
    if a.generate:
        generate(chosen)
    if a.grade:
        grade(chosen)
    if a.analyse:
        analyse()


if __name__ == "__main__":
    main()
