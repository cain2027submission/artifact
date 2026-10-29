#!/usr/bin/env python3
"""RQ3 at a PRODUCTION READ BUDGET: what the retrieval hook actually injects.

The published RQ3 arms hand the grader the WHOLE artifact -- a 1-6KB rollup note
or a tens-of-KB tool-stripped transcript. Neither is ever read that way in
production. The live hook (~/.claude/scripts/prompt-context.py) injects a median
881 chars for a median 5 notes -- ~176 chars per note, a path plus one heading --
and includes body text (capped at MAX_BODY_CHARS = 500) on only 16.7% of
injections. So "which artifact retains more" is not the operative question.
This script asks: at that budget, which artifact ANSWERS more?

Four arms, one shared answer prompt, one draw per item. Only the context varies:

  noop             `- <path>`                       (control: no heading, no body)
  rollup_pointer   `- <path> — <heading>`           (the 84% case)
  rollup_hookshape pointer + `From <path>:\n<body>` (the 16.7% case, <=500 chars)
  stripped_chunk   byte-matched chunk of the tool-stripped transcript

Fidelity notes, because each one is a place this could have cheated:

  * The rollup arms' rendering is lifted from render() in prompt-career-context:
    MAX_SNIPPET_CHARS = 160 on the heading, boring_heading() dropping structural
    labels, the description -> filename_title fallback chain, _restates_path()
    dropping a heading that only echoes the filename, MAX_BODY_CHARS = 500 with
    the same frontmatter/link/blank-line cleaning and word-boundary truncation.
  * The rollup SECTION is chosen by the hook's own keyword path: bm25 over
    sections_fts in the live index, restricted to that note. Not by hand.
  * The stripped transcript is NOT in the index (0 of 21,512 sections have a
    .jsonl path), so no retrieval mechanism can pick its chunk. The chunk is
    picked by keyword overlap with the QUESTION instead. That is a GENEROUS
    proxy -- it is oracle-ish selection the production system cannot perform --
    and it favours the stripped arm, so a stripped-arm loss here is a floor.
  * A null/errored verdict is recorded API_ERROR and never scored. An unanswered
    call scored as `absent` would manufacture the finding under measurement.

Usage:
    rq3_read_budget_arms.py --dry-run           # build contexts, no API calls
    rq3_read_budget_arms.py --limit 3           # smoke test
    rq3_read_budget_arms.py                     # full run (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, call_llm, grade_answer, index_transcripts,
    load_records, rates, record_id, resolve,
)
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402
from paths import REPO_ROOT, SESSIONS_DIR  # noqa: E402

OUT = DATA / "rq3_read_budget_arms.jsonl"
REPORT = DATA / "rq3_read_budget_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

# Using SESSIONS_DIR from paths.py directly, no local redefinition
INDEX_DB = f"file:{REPO_ROOT}/.obsidian/pkm_index.db?mode=ro"

# Verbatim from prompt-context.py. Changing either would stop this measuring the
# hook and start measuring a budget nobody uses.
MAX_SNIPPET_CHARS = 160
MAX_BODY_CHARS = 500

RECOVERABLE = ("correct", "partially_correct")
# noop_opaque exists because `noop` turned out NOT to be a no-op. A session
# rollup's filename is derived from its claim heading, so "- sessions/2026-09-25_
# claude-code_170637e9_example-claim-heading.md" ships the finding in the
# path -- and the hook's own _restates_path() then drops the heading as redundant,
# collapsing rollup_pointer onto noop for 45 of 89 records. noop_opaque truncates
# the filename at the session id, so it asserts a session note exists and carries
# nothing else. It is the arm that decides whether any of this is measuring
# retrieval rather than the grader's priors.
ARMS = ("noop_opaque", "noop", "rollup_pointer", "rollup_hookshape",
        "stripped_chunk")

STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "was",
    "what", "which", "who", "whom", "why", "how", "when", "where", "did", "do",
    "does", "done", "had", "has", "have", "not", "no", "yes", "he", "she", "they",
    "them", "his", "her", "their", "you", "your", "i", "we", "our", "us", "me",
    "so", "there", "here", "about", "into", "over", "under", "after", "before",
    "during", "while", "also", "only", "very", "more", "most", "some", "any",
    "all", "both", "each", "other", "such", "own", "same", "too", "can", "will",
    "just", "should", "would", "could", "may", "might", "must", "one", "two",
}

# prompt-context.py's FTS_BORING, imported rather than retyped where possible.
FTS_BORING = {
    "intent", "outcome", "commits", "files edited", "files", "recaps", "summary",
    "thinking", "thinking & dead ends", "dead ends", "delegated", "next",
    "notes", "context", "why", "eli5", "todo", "tasks", "links", "references",
}
BORING_MAX_WORDS = 4


def clean(text: str) -> str:
    return " ".join((text or "").split())


def boring_heading(head: str) -> bool:
    """True when a heading is a structural label rather than a finding."""
    low = (head or "").lower().strip(" #:-\u2014\u2013.")
    if low in FTS_BORING:
        return True
    words = [w for w in re.split(r"[^a-z0-9]+", low) if w]
    return len(words) <= BORING_MAX_WORDS and any(b in low for b in FTS_BORING)


FILENAME_NOISE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})?[_ -]*(claude[- ]code|antigravity|codex|copilot|cursor)?"
    r"[_ -]*([0-9a-f]{8})?[_ -]*", re.I)


def filename_title(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    stem = FILENAME_NOISE.sub("", stem)
    return " ".join(stem.replace("_", " ").replace("-", " ").split())


def _sig(text: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _restates_path(detail: str, path: str) -> bool:
    """Verbatim intent of prompt-context._restates_path: drop a heading that only
    echoes the filename it sits beside."""
    d, s = _sig(detail), _sig(os.path.splitext(os.path.basename(path or ""))[0])
    if not d or not s:
        return False
    if len(" ".join(d)) < 12:
        return False
    return d <= s or s <= d


def terms(text: str, cap: int = 10) -> list:
    out = []
    for w in re.split(r"[^A-Za-z0-9]+", (text or "").lower()):
        if len(w) < 3 or w in STOP or w in out:
            continue
        out.append(w)
    return out[:cap]


def pick_section(con, note_path: str, question: str):
    """The hook's keyword path, restricted to one note: bm25 over sections_fts.

    Returns (heading, content) for the best-matching section, or (None, None) when
    no section of the note matches any question term -- which is itself a
    production outcome, since the hook then falls back to the filename.
    """
    ts = terms(question)
    if not ts:
        return None, None
    match = " OR ".join('"%s"' % t for t in ts)
    sql = ("SELECT s.heading, sections_fts.content FROM sections_fts "
           "JOIN sections s ON s.id = sections_fts.section_id "
           "WHERE sections_fts MATCH ? AND s.path = ? "
           "ORDER BY bm25(sections_fts) LIMIT 1")
    try:
        row = con.execute(sql, (match, note_path)).fetchone()
    except sqlite3.OperationalError:
        return None, None
    return (row[0], row[1]) if row else (None, None)


def note_description(con, note_path: str) -> str:
    row = con.execute("SELECT description FROM notes WHERE path = ?",
                      (note_path,)).fetchone()
    return (row[0] or "") if row else ""


def any_section(con, note_path: str):
    """First non-boring section of the note, for when no term matched. The hook's
    semantic arm would still return *a* section; this keeps the arm from being
    handed nothing for a reason that is an artifact of keyword matching."""
    rows = con.execute(
        "SELECT s.heading, sections_fts.content FROM sections s "
        "JOIN sections_fts ON sections_fts.section_id = s.id "
        "WHERE s.path = ? ORDER BY s.id", (note_path,)).fetchall()
    for h, c in rows:
        if h and not boring_heading(h):
            return h, c
    return (rows[0] if rows else (None, None))


def clean_body(content: str) -> str:
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content or "", flags=re.DOTALL)
    content = re.sub(r"\n\n+", "\n\n", content)
    content = re.sub(r"\[\[([^\]]+)\]\]", r"\1", content)
    content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
    return clean(content)


def build_rollup_contexts(con, vault_rel: str, question: str) -> dict:
    """Render the three rollup-side contexts exactly as render() would."""
    heading, content = pick_section(con, vault_rel, question)
    matched = heading is not None
    if not matched:
        heading, content = any_section(con, vault_rel)

    detail = clean(heading or "")
    if not detail or boring_heading(detail):
        detail = clean(note_description(con, vault_rel)) or filename_title(vault_rel) or detail
    if detail and _restates_path(detail, vault_rel):
        detail = ""
    if len(detail) > MAX_SNIPPET_CHARS:
        detail = detail[:MAX_SNIPPET_CHARS].rstrip() + "\u2026"

    noop = "- %s" % vault_rel
    # Cut the filename right after the 8-hex session id, so the claim baked into
    # the slug is gone but the path still says "a session note exists".
    stem = os.path.splitext(os.path.basename(vault_rel))[0]
    m = re.match(r"^(\d{4}-\d{2}-\d{2}_[a-z-]+_[0-9a-f]{8})", stem)
    opaque = "- %s" % (os.path.dirname(vault_rel) + "/" + (m.group(1) if m else stem[:8])
                       + ".md")
    pointer = "- %s \u2014 %s" % (vault_rel, detail) if detail else noop

    body = clean_body(content or "")
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + "\u2026"
    hookshape = pointer + ("\n\nFrom %s:\n%s" % (vault_rel, body) if body else "")
    return {"noop_opaque": opaque, "noop": noop, "rollup_pointer": pointer,
            "rollup_hookshape": hookshape, "_detail": detail,
            "_body_chars": len(body), "_fts_matched": matched}


def best_chunk(stripped: str, question: str, budget: int) -> str:
    """Highest keyword-overlap window of the stripped transcript, `budget` chars.

    Windows stride by half a budget so no scoring position is unreachable.
    Scored by distinct question terms present (primary) then total occurrences
    (tie-break). This is the GENEROUS proxy: the production hook cannot do this
    at all, because the transcript is not indexed.
    """
    if budget <= 0 or not stripped:
        return ""
    ts = terms(question)
    low = stripped.lower()
    stride = max(1, budget // 2)
    best, best_score = stripped[:budget], (-1, -1)
    for start in range(0, max(1, len(stripped) - budget + stride), stride):
        window = low[start:start + budget]
        if not window:
            break
        distinct = sum(1 for t in ts if t in window)
        total = sum(window.count(t) for t in ts)
        if (distinct, total) > best_score:
            best_score = (distinct, total)
            best = stripped[start:start + budget]
    return best


def build_stripped_context(tpath: str, stripped: str, question: str,
                           budget_chars: int) -> str:
    """Byte-matched to the rollup_hookshape context, same two-part shape."""
    head = "- %s" % tpath
    intro = "\n\nFrom %s:\n" % tpath
    room = budget_chars - len(head) - len(intro)
    if room <= 0:
        return head[:budget_chars]
    return head + intro + best_chunk(stripped, question, room)


ANSWER_PROMPT = """Someone asks: "{question}"

Below is the context block that was automatically retrieved and injected alongside that question. Answer the question based ONLY on this context. If the context does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

CONTEXT:
{context}

Provide your answer and confidence now:"""


def answer(question: str, context: str):
    """One prompt for all four arms, so only the context differs between them.

    The parse is verbatim from rq3_raw_transcript_arm.answer_from_transcript:
    confidence from the LAST line, API_ERROR (never NOT_ANSWERABLE) on empty.
    """
    response, usage = call_llm(ANSWER_PROMPT.format(question=question,
                                                    context=context),
                               model=JUDGE_MODEL, timeout=180)
    if not response:
        return "", "API_ERROR", usage
    lines = response.strip().split("\n")
    confidence = "NOT_ANSWERABLE"
    for keyword in ["ANSWERED", "PARTIALLY_ANSWERED", "NOT_ANSWERABLE"]:
        if keyword in lines[-1].upper():
            confidence = keyword
            break
    return "\n".join(lines[:-1]).strip(), confidence, usage


def exact_binomial(b: int, c: int) -> float:
    """Two-sided exact binomial on discordant pairs, verbatim from the sibling."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(p, 1.0)


def resolve_rollup(session_id: str):
    p = SESSIONS_DIR / (session_id + ".md")
    if p.exists():
        return p
    m = SLUG.match(session_id)
    if not m:
        return None
    hits = sorted(SESSIONS_DIR.glob("*%s*.md" % m.group(1)))
    return hits[0] if hits else None


def build_items(limit: int = 0) -> list:
    """The 89-subset, defined exactly as rq3_stripped_paired.main() defines it."""
    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    published = {record_id(r): r for r in load_records(QUESTIONS)
                 if record_id(r) in graded}
    uniform = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_uniform.jsonl")}
    raw = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_raw.jsonl")}
    strip = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_stripped.jsonl")}
    unresolved = sorted({i for d in (uniform, raw, strip) for i, r in d.items()
                         if r.get("grade") is None})
    for d in (uniform, raw, strip):
        for i in unresolved:
            d.pop(i, None)
    shared89 = sorted(set(strip) & set(raw) & set(uniform) & set(published))
    print("89-subset: %d records, %d unresolved dropped upstream"
          % (len(shared89), len(unresolved)))

    by_prefix = index_transcripts()
    con = sqlite3.connect(INDEX_DB, uri=True)
    cache = {}
    items, skipped = [], []
    for rid in shared89:
        r = published[rid]
        sid = r["session_id"]
        note = resolve_rollup(sid)
        m = SLUG.match(sid)
        tpath = resolve(m.group(1), by_prefix) if m else None
        if note is None or tpath is None:
            skipped.append({"record_id": rid, "session_id": sid,
                            "reason": "no rollup note" if note is None
                                      else "no transcript"})
            continue
        vault_rel = "sessions/" + note.name
        if tpath not in cache:
            cache[str(tpath)] = strip_transcript(tpath.read_text(errors="replace"))[0]
        stripped = cache[str(tpath)]

        ctx = build_rollup_contexts(con, vault_rel, r["question"])
        budget = len(ctx["rollup_hookshape"])
        ctx["stripped_chunk"] = build_stripped_context(
            str(tpath), stripped, r["question"], budget)
        items.append({
            "record_id": rid, "session_id": sid, "class": r.get("class"),
            "question": r["question"], "ground_truth": r.get("ground_truth", ""),
            "rollup_grade_published": r.get("grade"),
            "strip_grade_whole": strip[rid].get("grade"),
            "raw_grade_whole": raw[rid].get("grade"),
            "note": vault_rel, "transcript": str(tpath),
            "stripped_chars": len(stripped),
            "detail": ctx["_detail"], "body_chars": ctx["_body_chars"],
            "fts_matched": ctx["_fts_matched"],
            "contexts": {a: ctx[a] for a in ARMS},
        })
    con.close()
    if skipped:
        print("SKIPPED %d: %s" % (len(skipped), skipped))
    if limit:
        items = items[:limit]
    return items, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--degenerate", action="store_true",
                    help="smoke the context builders on empty/garbage inputs")
    args = ap.parse_args()

    if args.degenerate:
        con = sqlite3.connect(INDEX_DB, uri=True)
        for note, q in [("sessions/does-not-exist.md", "what happened?"),
                        ("sessions/does-not-exist.md", ""),
                        ("", "?????")]:
            c = build_rollup_contexts(con, note, q)
            print("note=%r q=%r -> noop=%d pointer=%d hookshape=%d detail=%r"
                  % (note, q, len(c["noop"]), len(c["rollup_pointer"]),
                     len(c["rollup_hookshape"]), c["_detail"][:60]))
        for stripped, budget in [("", 500), ("abc", 500), ("x" * 10, 3), ("y" * 50, 0)]:
            s = build_stripped_context("/tmp/t.jsonl", stripped, "what failed", budget)
            print("stripped=%d budget=%d -> %d chars %r"
                  % (len(stripped), budget, len(s), s[:40]))
        con.close()
        return

    items, skipped = build_items(args.limit)
    sizes = {a: [len(i["contexts"][a]) for i in items] for a in ARMS}
    print("\ncontext sizes (chars), n=%d" % len(items))
    for a in ARMS:
        v = sorted(sizes[a])
        print("  %-17s median %4d  min %4d  max %4d"
              % (a, v[len(v) // 2], v[0], v[-1]))
    nb = sum(1 for i in items if i["body_chars"])
    print("  rollup body present: %d/%d (%.1f%%)  fts matched a section: %d/%d"
          % (nb, len(items), 100.0 * nb / max(1, len(items)),
             sum(1 for i in items if i["fts_matched"]), len(items)))
    byte_mismatch = [i["record_id"] for i in items
                     if len(i["contexts"]["stripped_chunk"])
                     != len(i["contexts"]["rollup_hookshape"])]
    print("  byte-match failures (hookshape vs stripped_chunk): %d" % len(byte_mismatch))

    if args.dry_run:
        print("\n--dry-run: no API calls made")
        print("\n=== sample contexts (first item) ===")
        for a in ARMS:
            print("\n--- %s (%d chars) ---\n%s"
                  % (a, len(items[0]["contexts"][a]), items[0]["contexts"][a]))
        return

    done = set()
    if OUT.exists():
        for r in load_records(OUT):
            done.add((r.get("record_id"), r.get("arm")))
        print("resuming: %d (record, arm) cells already graded" % len(done))

    todo = [(i, a) for i in items for a in ARMS if (i["record_id"], a) not in done]
    print("to grade: %d cells\n" % len(todo))

    t0 = time.time()
    tokens_in = tokens_out = 0
    errors = 0
    for n, (item, arm) in enumerate(todo, 1):
        ctx = item["contexts"][arm]
        ans, conf, usage = answer(item["question"], ctx)
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("completion_tokens") or 0
        if conf == "API_ERROR":
            # Not written, so a rerun retries it. Never scored as `absent`: that
            # is the outcome under measurement and a silent API failure would
            # read as a real finding.
            errors += 1
            print("[%d/%d] %s %s -> API_ERROR (not recorded)"
                  % (n, len(todo), item["record_id"], arm))
            continue
        grade = grade_answer(ans, item["ground_truth"], conf)
        rec = {"record_id": item["record_id"], "arm": arm,
               "session_id": item["session_id"], "class": item["class"],
               "context_chars": len(ctx), "confidence": conf, "grade": grade,
               "answer": ans, "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print("[%d/%d] %-16s %-16s %-10s %s  (%.0fs, in=%d out=%d)"
              % (n, len(todo), item["record_id"], arm, conf, grade,
                 time.time() - t0, tokens_in, tokens_out))

    print("\nAPI errors (unrecorded, retryable): %d" % errors)
    print("tokens: prompt=%d completion=%d" % (tokens_in, tokens_out))
    summarise(items, skipped, {"prompt_tokens": tokens_in,
                               "completion_tokens": tokens_out,
                               "api_errors": errors})


def summarise(items: list, skipped: list, spend: dict) -> None:
    rows = load_records(OUT)
    by_arm = {a: {} for a in ARMS}
    for r in rows:
        if r.get("arm") in by_arm:
            by_arm[r["arm"]][r["record_id"]] = r
    attempted = {i["record_id"] for i in items}
    complete = sorted(set.intersection(*[set(by_arm[a]) & attempted for a in ARMS]))
    print("\nattempted %d records x %d arms = %d cells; %d records have all 4 arms"
          % (len(attempted), len(ARMS), len(attempted) * len(ARMS), len(complete)))
    for a in ARMS:
        miss = sorted(attempted - set(by_arm[a]))
        if miss:
            print("  %s missing %d: %s" % (a, len(miss), miss[:6]))

    out = {"n_attempted_records": len(attempted),
           "n_complete_records": len(complete),
           "skipped_records": skipped, "spend": spend, "arms": {}, "paired": {},
           "per_class": {}}
    hdr = ("%-17s %5s  %3s %3s %3s %3s  %-8s %-14s %s"
           % ("arm", "n", "cor", "par", "wrg", "abs", "recover", "wilson95",
              "median chars"))
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for a in ARMS:
        rs = [by_arm[a][i] for i in complete]
        d = rates(rs)
        chars = sorted(r["context_chars"] for r in rs)
        d["median_context_chars"] = chars[len(chars) // 2] if chars else 0
        out["arms"][a] = d
        print("%-17s %5d  %3d %3d %3d %3d  %7.1f%% %-14s %d"
              % (a, d["n"], d["correct"]["n"], d["partially_correct"]["n"],
                 d["wrong"]["n"], d["absent"]["n"], d["recoverable"]["pct"],
                 str(d["recoverable"]["ci95"]), d["median_context_chars"]))

    for la, lb in [("rollup_hookshape", "stripped_chunk"),
                   ("rollup_pointer", "stripped_chunk"),
                   ("rollup_pointer", "noop"),
                   ("rollup_hookshape", "noop"),
                   ("rollup_hookshape", "rollup_pointer"),
                   ("noop", "noop_opaque")]:
        a_ok = {i for i in complete if by_arm[la][i]["grade"] in RECOVERABLE}
        b_ok = {i for i in complete if by_arm[lb][i]["grade"] in RECOVERABLE}
        b, c = len(b_ok - a_ok), len(a_ok - b_ok)
        p = exact_binomial(b, c)
        out["paired"]["%s_vs_%s" % (la, lb)] = {
            "n": len(complete), "both": len(a_ok & b_ok), "%s_only" % la: c,
            "%s_only" % lb: b, "neither": len(complete) - len(a_ok | b_ok),
            "discordant": b + c, "exact_binomial_two_sided_p": p}
        print("\npaired %s vs %s over %d: both %d  %s only %d  %s only %d  "
              "neither %d  discordant %d  p=%.4g"
              % (la, lb, len(complete), len(a_ok & b_ok), la, c, lb, b,
                 len(complete) - len(a_ok | b_ok), b + c, p))

    print("\nper class (recoverable %):")
    classes = sorted({by_arm[ARMS[0]][i].get("class") for i in complete})
    for cl in classes:
        ids = [i for i in complete if by_arm[ARMS[0]][i].get("class") == cl]
        row = {"n": len(ids)}
        line = "  %-16s n=%2d " % (cl, len(ids))
        for a in ARMS:
            d = rates([by_arm[a][i] for i in ids])
            row[a] = d
            line += " %s %5.1f%%" % (a[:9], d["recoverable"]["pct"])
        out["per_class"][cl] = row
        print(line)

    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote %s" % REPORT)


if __name__ == "__main__":
    main()
