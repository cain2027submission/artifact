#!/usr/bin/env python3
"""Regenerate the 25 held-out sessions' rollup notes with Haiku 4.5, without touching sessions/.

Uses the production summariser's own code (agent-session-rollup: rollup_drain.rollup ->
rollup_drain.child_turns -> rollup_summarise.summarise -> build_summary_prompt -> rollup_note.render),
so the input is the rule-filtered view the drain feeds the model (read_turns, empty blocks dropped,
subagents folded as brief summaries, epochs split on gaps). Two things differ from a production
drain run and are recorded, not hidden:
  * the model call goes through the local gateway (POST /v1/chat/completions, the route every Haiku
    arm in this directory uses) instead of the `claude -p` CLI backend, via an injected client;
    summarise() takes a client for exactly this. max_tokens is raised to 4096 (the injected-client
    path hard-codes 1024, the CLI has no such cap), to avoid truncating JSON.
  * no queue record, so no `commits observed` / branch / head metadata (production gets those from
    the queue at session time; they are not model output). The transcript is also read as it is
    today, which can be longer than at production time for a resumed session.
Output: data/paper-rollups/heldout-rollup-haiku/rollups/<session_id>.md plus generations.jsonl
(one row per model call: model returned, usage) and cost_log.jsonl.

Usage: heldout_rollup_haiku_generate.py --dry-run | [--only PREFIX ...]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "paper-rollups"
OUTDIR = DATA / "heldout-rollup-haiku"
ROLLUPS = OUTDIR / "rollups"
GENLOG = OUTDIR / "generations.jsonl"
COST = OUTDIR / "cost_log.jsonl"
MISSING = OUTDIR / "missing.json"
QUESTIONS = DATA / "rq3_heldout_questions.jsonl"
SRC = ROLLUP_SRC_DIR
GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
MODEL = "claude-haiku-4-5@20251001"   # gateway spelling of claude-haiku-4-5 (rollup_summarise.HAIKU_VERTEX_MODEL)
PRICE_IN, PRICE_OUT = 1.0, 5.0         # USD per 1M tokens, assumed, not billed
CAP_USD = 2.2                          # generation share of the USD 4 total (step C reserves the rest)
GEN_MAX_TOKENS = 4096

sys.path.insert(0, str(SRC))
os.environ.setdefault("PKM_ROLLUP_BACKEND", "cli")
os.environ.pop("PKM_ROLLUP_MODEL", None)
import rollup_drain as drain  # noqa: E402
import rollup_note as note  # noqa: E402
import rollup_read as read  # noqa: E402
import rollup_summarise as rs  # noqa: E402
import rollup_discover as disc  # noqa: E402
from rollup_types import Session  # noqa: E402

lock = threading.Lock()
LOCAL = threading.local()


class Msg:
    def __init__(self, text):
        self.content = [SimpleNamespace(text=text)]


class GatewayClient:
    """messages.create() shim over the gateway; logs model returned and usage per call."""

    def __init__(self, dry=False):
        self.dry = dry
        self.messages = self

    def create(self, model, max_tokens, messages):
        prompt = messages[0]["content"]
        if self.dry:
            self._log(prompt, {"prompt_tokens": len(prompt) // 3, "completion_tokens": 450}, MODEL, "dry")
            return Msg('{"heading": "x", "intent": "x", "thinking": "x", "outcome": "x"}')
        body = json.dumps({"model": MODEL, "max_tokens": GEN_MAX_TOKENS,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(f"{GATEWAY}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        ch = data["choices"][0]
        self._log(prompt, data.get("usage", {}), data.get("model"), ch.get("finish_reason"), time.time() - t0)
        return Msg(ch["message"]["content"] or "")

    def _log(self, prompt, usage, returned, finish, secs=0.0):
        row = {"session": getattr(LOCAL, "sid", None), "model_requested": MODEL, "model_returned": returned,
               "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
               "finish_reason": finish, "seconds": round(secs, 1), "dry": self.dry}
        LOCAL.calls.append(row)
        if not self.dry:
            with lock, GENLOG.open("a") as fh:
                fh.write(json.dumps(row) + "\n")


class Summ:
    """Stand-in for the rollup_summarise module as rollup_drain.rollup sees it."""

    def __init__(self, client):
        self.client = client

    def summarise(self, epoch, dry_run=False, brief=False, **kw):
        return rs.summarise(epoch, client=self.client, brief=brief)


def sessions_for(ids):
    out = {}
    for sid in ids:
        pre = sid.split("_")[2]
        hits = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{pre}*.jsonl"))
        if not hits:
            out[sid] = None
            continue
        p = Path(hits[0])
        pd = disc.extract_project_dir(p, p.parent.parent)
        s = Session(session_id=p.stem, agent=disc.AGENT, transcript=str(p), project_dir=pd,
                    worktree=disc.extract_worktree(pd), machine="")
        kids = []
        for k in sorted(p.parent.glob(f"{p.stem}/subagents/*.jsonl")):
            if k.stat().st_size:
                kids.append(Session(session_id=f"{p.stem}/{k.stem}", agent=disc.AGENT, transcript=str(k),
                                    project_dir=pd, worktree=s.worktree, parent_id=p.stem))
        out[sid] = (s, kids)
    return out


def one(sid, s, kids, dry, tmp):
    LOCAL.sid, LOCAL.calls = sid, []
    drain.NOTES = tmp
    client = GatewayClient(dry)
    path = drain.rollup(s, dry_run=False, read=read, note=note, summarise=Summ(client), children=kids,
                        traces_out=None, traces_root=None, queue_record=None)
    return sid, path, LOCAL.calls


def est(calls):
    return sum(c["prompt_tokens"] * PRICE_IN + c["completion_tokens"] * PRICE_OUT for c in calls) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    ids = sorted({json.loads(l)["session_id"] for l in QUESTIONS.read_text().splitlines() if l.strip()})
    if args.only:
        ids = [i for i in ids if any(o in i for o in args.only)]
    found = sessions_for(ids)
    missing = [i for i, v in found.items() if v is None]
    todo = [i for i in ids if found[i] and not (ROLLUPS / f"{i}.md").exists()]
    print(f"sessions {len(ids)}  missing transcript {len(missing)}  todo {len(todo)}")
    ROLLUPS.mkdir(parents=True, exist_ok=True)
    tmp = OUTDIR / "_drain_out"
    if args.dry_run:
        tmp = Path("/tmp/heldout_haiku_dry")
    tmp.mkdir(parents=True, exist_ok=True)
    spent = sum(json.loads(l)["usd"] for l in COST.read_text().splitlines() if l.strip()) if COST.exists() else 0.0
    with ThreadPoolExecutor(1 if args.dry_run else 4) as ex:
        res = list(ex.map(lambda i: one(i, *found[i], args.dry_run, tmp), todo))
    total = 0.0
    for sid, path, calls in res:
        e = est(calls)
        total += e
        print(f"  {sid[:60]:60s} calls {len(calls):3d} in {sum(c['prompt_tokens'] for c in calls):8d} est ${e:.3f} note {'ok' if path else 'NONE'}")
        if args.dry_run or not path:
            continue
        (ROLLUPS / f"{sid}.md").write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"batch estimate ${total:.3f} (dry={args.dry_run}), running total before ${spent:.3f}, cap ${CAP_USD}")
    if not args.dry_run:
        with COST.open("a") as fh:
            fh.write(json.dumps({"stage": "generate", "calls": sum(len(c) for _, _, c in res),
                                 "usd": round(total, 5)}) + "\n")
        MISSING.write_text(json.dumps({"missing_transcript": missing}, indent=1))


if __name__ == "__main__":
    main()
