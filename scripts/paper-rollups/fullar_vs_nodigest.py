#!/usr/bin/env python3
"""EXPLORATORY, not part of the main pre-registration: full action record vs
no-digest action record vs output-stripped transcript, like for like on `grade`
(3-draw majority, grade_protocol v2). Isolates the contribution of the result digests.

Modes:
  --part1             design set (124 q), no model calls
  --plan-run          print held-out coverage + cheapest-first run order + projected spend
  --run N             answer + grade the next N missing held-out full-record questions
                      (cheapest artifact first, hard USD cap, cost log)
  --part2             pairwise analysis on held-out questions with all three arms graded
  --write             write results.json (runs part1 + part2)

Answering prompt = answer_from_action_record in rq3_action_record_arm.py (the one the 68
existing held-out full-record answers used, via rq3_heldout_arms.py); mechanism = gateway POST
as in rq3_heldout_no_digest_extend.py (rq3_raw_transcript_arm.call_llm): no temperature.
Artifacts = prebuilt <prefix>.action.md files the 68 existing answers were read from.
Grading = replication_run.grade logic: 3 draws, plurality, 2 more on a 1-1-1 tie.
Recall = correct + partially_correct. Wrong-given-attempted = wrong / (n - absent).
CI = session-clustered paired bootstrap, B=10000, seed 20261013, 90% percentile.
McNemar = exact two-sided binomial on discordant recall pairs.
"""
import argparse, hashlib, json, random, sys, time
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper-rollups"
OUTD = DATA / "fullar-vs-nodigest"
ANSWERS = OUTD / "heldout_fullar_new.jsonl"
COST_LOG = OUTD / "cost_log.jsonl"
CALLS = OUTD / "calls.jsonl"
RESULTS = OUTD / "results.json"
ARTIFACTS = JOBS_DIR / "3f652bec" / "tmp" / "heldout" / "artifacts"
GATEWAY = "http://127.0.0.1:8080"
MODEL = "claude-sonnet-5"
CAP_USD = 6.0
HEADROOM_USD = 5.75  # projection must stay under this so estimate error cannot cross the cap
PRICE_IN, PRICE_OUT = 3.0 / 1e6, 15.0 / 1e6
CHARS_PER_TOKEN = 2.46      # measured on these artifacts (rq3_heldout_arms.py: 2.459)
OVERHEAD_USD = 0.03         # output ~500 tok + 3 grading calls, conservative
B, SEED = 10000, 20261013
OK = ("correct", "partially_correct")
GRADE_FILES = {
    "design": {"full": "rq3_action_record_arm.jsonl", "nodigest": "rq3_action_record_no_digest_arm.jsonl",
               "stripped": "rq3_questions_stripped_arm.jsonl"},
    "heldout": {"full": "rq3_heldout_action_record_arm.jsonl", "nodigest": "rq3_heldout_no_digest_arm.jsonl",
                "stripped": "rq3_heldout_stripped_arm_extended.jsonl"},
}
PAIRS = [("full", "nodigest"), ("full", "stripped"), ("nodigest", "stripped")]


def rows(p):
    out = []
    for l in Path(p).read_text(errors="replace").splitlines():
        if l.strip() and not l.startswith("\x00"):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def load(name):
    return {r["record_id"]: r for r in rows(DATA / name)}


def rid(r):
    return hashlib.sha256(f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}".encode()).hexdigest()[:16]


# ---------------- statistics ----------------
def rates(rs):
    n = len(rs)
    c = Counter(r["grade"] for r in rs)
    att = n - c["absent"]
    return {"n": n, "recall": (c["correct"] + c["partially_correct"]) / n if n else None,
            "counts": dict(c), "absent": c["absent"] / n if n else None,
            "wrong_given_attempted": c["wrong"] / att if att else None, "n_attempted": att}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(a, b, sess):
    rng = random.Random(SEED)
    by = {}
    for i, s in enumerate(sess):
        by.setdefault(s, []).append(i)
    keys = list(by)
    d = []
    for _ in range(B):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in by[k]]
        d.append(100 * (sum(a[i] for i in idx) - sum(b[i] for i in idx)) / len(idx))
    d.sort()

    def pct(p):
        k = p * (B - 1); lo = int(k)
        return d[lo] * (1 - (k - lo)) + d[lo + 1] * (k - lo) if lo + 1 < B else d[lo]
    return [round(pct(.05), 2), round(pct(.95), 2)]


def analyse(arms, ids, label):
    """arms: name -> {record_id: row}; ids: record ids present in all three."""
    ids = sorted(ids)
    sess = [arms["full"][i]["session_id"] for i in ids]
    ok = {a: [1 if arms[a][i]["grade"] in OK else 0 for i in ids] for a in arms}
    out = {"label": label, "n": len(ids), "n_sessions": len(set(sess)), "grade_field": "grade",
           "bootstrap": {"B": B, "seed": SEED, "type": "session-clustered paired, percentile 90%"},
           "arms": {a: rates([arms[a][i] for i in ids]) for a in arms}, "pairs": {}}
    for x, y in PAIRS:
        b = sum(1 for p, q in zip(ok[x], ok[y]) if p and not q)
        c = sum(1 for p, q in zip(ok[x], ok[y]) if q and not p)
        ci = boot_ci(ok[x], ok[y], sess)
        out["pairs"][f"{x}_minus_{y}"] = {
            "recall_diff_pts": round(100 * (sum(ok[x]) - sum(ok[y])) / len(ids), 2), "ci90_pts": ci,
            "ci90_inside_pm10": ci[0] > -10 and ci[1] < 10, "ci90_inside_pm5": ci[0] > -5 and ci[1] < 5,
            "mcnemar": {f"{x}_only": b, f"{y}_only": c, "both": sum(1 for p, q in zip(ok[x], ok[y]) if p and q),
                        "neither": sum(1 for p, q in zip(ok[x], ok[y]) if not p and not q),
                        "p_exact_two_sided": round(mcnemar_exact(b, c), 4)}}
    return out


def merged_heldout():
    full = load(GRADE_FILES["heldout"]["full"])
    if ANSWERS.exists():
        for r in rows(ANSWERS):
            if r.get("grade") not in (None, "grade_error", "unresolved"):
                full.setdefault(r["record_id"], r)
    return full


def part1():
    arms = {a: load(f) for a, f in GRADE_FILES["design"].items()}
    ids = set.intersection(*[set(v) for v in arms.values()])
    return analyse(arms, ids, "design set (circular for both action-record arms)")


def part2():
    arms = {a: (merged_heldout() if a == "full" else load(f)) for a, f in GRADE_FILES["heldout"].items()}
    ids = set.intersection(*[set(v) for v in arms.values()])
    return analyse(arms, ids, "held-out")


# ---------------- paid run ----------------
def cum_usd():
    return sum(json.loads(l)["usd"] for l in CALLS.open() if l.strip()) if CALLS.exists() else 0.0


def call_gateway(stage, prompt, max_tokens, timeout):
    import urllib.request, urllib.error
    cap = max_tokens
    for attempt in range(2):  # one retry only
        if cum_usd() >= CAP_USD:
            raise SystemExit(f"CAP REACHED {cum_usd():.2f}")
        try:
            req = urllib.request.Request(f"{GATEWAY}/v1/chat/completions", headers={"Content-Type": "application/json"},
                data=json.dumps({"model": MODEL, "max_tokens": cap, "messages": [{"role": "user", "content": prompt}]}).encode())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            ch, u = d["choices"][0], d.get("usage", {})
            u["finish_reason"] = ch.get("finish_reason")
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            with CALLS.open("a") as f:
                f.write(json.dumps({"t": time.time(), "stage": stage, "in": tin, "out": tout,
                                    "usd": tin * PRICE_IN + tout * PRICE_OUT, "attempt": attempt}) + "\n")
            text = (ch["message"]["content"] or "").strip()
            if u["finish_reason"] == "length" and attempt == 0:
                cap *= 2
                continue
            if u["finish_reason"] == "length" or not text:
                return "", u
            return text, u
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
            with CALLS.open("a") as f:
                f.write(json.dumps({"t": time.time(), "stage": stage, "in": 0, "out": 0, "usd": 0, "error": str(e)[:160], "attempt": attempt}) + "\n")
            time.sleep(3)
    return "", {"failed": True}


def grade_majority(gc, q, answer, conf, raw):
    def call(p):
        return call_gateway("grade", p, 1024, 180)[0]

    def one():
        for _ in range(2):
            try:
                return gc.grade_with(call, q["question"], q["ground_truth"], answer, confidence=conf, raw_reply=raw)
            except gc.GradeError:
                continue
        return None
    first = one()
    if first is None:
        return "grade_error", []
    if first["grader_response"] is None:
        return first["grade"], []
    draws = [first["grade"]]
    for _ in range(2):
        r = one()
        if r is None:
            return "grade_error", draws
        draws.append(r["grade"])
    c = Counter(draws).most_common()
    if len(c) > 1 and c[0][1] == c[1][1] == 1:
        for _ in range(2):
            r = one()
            if r is None:
                return "grade_error", draws
            draws.append(r["grade"])
        c = Counter(draws).most_common()
    if len(c) == 1 or c[0][1] > c[1][1]:
        return c[0][0], draws
    return "unresolved", draws


def coverage():
    qs = rows(DATA / "rq3_heldout_questions.jsonl")
    for q in qs:
        q["_rid"] = rid(q)
    have = merged_heldout()
    miss = [q for q in qs if q["_rid"] not in have]
    for q in miss:
        p = ARTIFACTS / f"{q['session_prefix']}.action.md"
        q["_art"], q["_chars"] = p, (p.stat().st_size if p.exists() else None)
        q["_est"] = (q["_chars"] / CHARS_PER_TOKEN * PRICE_IN + OVERHEAD_USD) if q["_chars"] else None
    # order depends on artifact size only, never on a grade
    miss = sorted([q for q in miss if q["_chars"]], key=lambda q: (q["_chars"], q["_rid"]))
    return qs, have, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1", action="store_true")
    ap.add_argument("--plan-run", action="store_true")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--part2", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    if a.part1:
        print(json.dumps(part1(), indent=1))
    if a.plan_run:
        qs, have, miss = coverage()
        run, tot = [], cum_usd()
        for q in miss:
            if tot + q["_est"] > HEADROOM_USD:
                break
            run.append(q); tot += q["_est"]
        print(f"pool {len(qs)}, existing full-record graded {len(have)}, missing {len(miss)}; "
              f"fit under projected ${HEADROOM_USD}: {len(run)} (projected total ${tot:.2f}, "
              f"all-missing ${sum(q['_est'] for q in miss):.2f})")
    if a.run:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import grading_common as gc
        from rq3_action_record_arm import answer_from_action_record  # noqa: F401  (prompt source)
        from rq3_raw_transcript_arm import extract_confidence
        import inspect, re
        # reuse the exact prompt text of answer_from_action_record
        src = inspect.getsource(answer_from_action_record)
        assert "one-line result digests" in src
        protocol = f"{gc.PROMPT_VERSION}+{gc.PARSER_VERSION}+majority3+{MODEL}"
        _, have, miss = coverage()
        done = {r["record_id"] for r in rows(ANSWERS)} if ANSWERS.exists() else set()
        todo = [q for q in miss if q["_rid"] not in done][:a.run]
        proj = sum(q["_est"] for q in todo)
        print(f"batch: {len(todo)} questions, projected ${proj:.2f}, running total before ${cum_usd():.2f}, cap ${CAP_USD}")
        if cum_usd() + proj > CAP_USD:
            sys.exit("projected total exceeds cap, refusing")
        t0, c0 = time.time(), cum_usd()
        n_calls0 = sum(1 for _ in CALLS.open()) if CALLS.exists() else 0
        for q in todo:
            ctx = q["_art"].read_text(errors="replace")
            prompt = f"""Someone asks: "{q['question']}"

Below is a structured action record of an agent session, with tool calls shown as name + salient arguments, one-line result digests, and turn ordering. Answer the question based ONLY on this record. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

ACTION RECORD:
{ctx}

Provide your answer and confidence now:"""
            raw, u = call_gateway("answer", prompt, 2048, 600)
            if not raw:
                print("API_ERROR", q["session_prefix"], q["class"]); continue
            ans, conf = extract_confidence(raw)
            g, draws = grade_majority(gc, q, ans, conf, raw)
            row = {"record_id": q["_rid"], "session_id": q["session_id"], "session_prefix": q["session_prefix"],
                   "class": q["class"], "question": q["question"], "ground_truth": q["ground_truth"],
                   "arm": "action_record", "artifact": str(q["_art"]), "artifact_chars": len(ctx), "answer": ans,
                   "rating": conf, "grade": g, "grade_draws": draws, "grade_protocol": protocol,
                   "raw_reply": raw, "usage": u, "model": MODEL}
            with ANSWERS.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{q['session_prefix']} {q['class']:15s} {conf:18s} {g:17s} running ${cum_usd():.3f}")
        calls = [json.loads(l) for l in CALLS.open()][n_calls0:]
        with COST_LOG.open("a") as f:
            f.write(json.dumps({"stage": f"heldout_fullar_batch_{len(todo)}", "calls": len(calls),
                                "input_tokens": sum(c["in"] for c in calls), "output_tokens": sum(c["out"] for c in calls),
                                "est_usd": round(cum_usd() - c0, 4), "pre_batch_estimate_usd": round(proj, 4),
                                "cumulative_usd": round(cum_usd(), 4), "price": "claude-sonnet-5 $3/$15 per M, gateway-reported usage (about 1.5x above real spend)",
                                "t": time.time()}) + "\n")
        print(f"batch done: ${cum_usd()-c0:.3f}, cumulative ${cum_usd():.3f}")
    if a.part2:
        print(json.dumps(part2(), indent=1))
    if a.write:
        qs, have, miss = coverage()
        new = rows(ANSWERS) if ANSWERS.exists() else []
        res = {"status": "EXPLORATORY, not part of the main pre-registration; no pass/fail margin, "
                         "reports only whether each pairwise 90% CI lies inside +-10 and +-5 points",
               "design_set_caveat": "design set is circular for both action-record arms (designed with these "
                                    "questions visible); Part 1 isolates the digest effect on seen questions only",
               "recall": "correct + partially_correct on field `grade` (3-draw majority, grade_protocol v2)",
               "part1_design": part1(), "part2_heldout": part2(),
               "part2_coverage": {"pool": len(qs), "existing_full_graded": len(have) - len(new), "newly_answered": len(new),
                                  "n_all_three_arms": part2()["n"], "still_missing_full": len(miss) - 0,
                                  "selection_rule": "missing questions run cheapest artifact first (by chars), cap-limited; never by grade"},
               "spend": {"estimated_usd": round(cum_usd(), 4), "cap_usd": CAP_USD}}
        RESULTS.write_text(json.dumps(res, indent=1))
        print("wrote", RESULTS)


if __name__ == "__main__":
    main()
