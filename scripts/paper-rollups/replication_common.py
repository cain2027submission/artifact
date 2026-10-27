"""Shared paid-call wrapper for the replication: one gateway call, per-call ledger,
hard USD cap. Mechanism is the one in rq3_heldout_no_digest_extend.py (rq3_raw_transcript_arm
.call_llm): POST {GATEWAY}/v1/chat/completions, no temperature, claude-sonnet-5."""
import json, os, threading, time, urllib.error, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "paper-rollups"
RDIR = DATA / "replication"
RDIR.mkdir(parents=True, exist_ok=True)
GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
SONNET = "claude-sonnet-5"
PRICE_IN, PRICE_OUT = 3.0 / 1e6, 15.0 / 1e6
CAP_USD = 70.0
CALLS = RDIR / "calls.jsonl"
COST_LOG = RDIR / "cost_log.jsonl"
_lock = threading.Lock()


def cumulative_usd():
    if not CALLS.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in CALLS.open() if l.strip())


def usd(tin, tout):
    return tin * PRICE_IN + tout * PRICE_OUT


def call_paid(stage, prompt, model=SONNET, max_tokens=2048, timeout=600):
    """-> (text, usage). Returns ("", usage) on failure; never scores a failure. One retry
    only (at double max_tokens for finish=length, otherwise a single plain retry)."""
    cap = max_tokens
    for attempt in range(2):
        if cumulative_usd() >= CAP_USD:
            raise SystemExit(f"CAP REACHED: ${cumulative_usd():.2f} >= ${CAP_USD}")
        try:
            req = urllib.request.Request(f"{GATEWAY}/v1/chat/completions",
                data=json.dumps({"model": model, "max_tokens": cap,
                                 "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]; u = d.get("usage", {}); fin = ch.get("finish_reason")
            u["finish_reason"] = fin
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            text = (ch["message"]["content"] or "").strip()
            with _lock, CALLS.open("a") as f:
                f.write(json.dumps({"t": time.time(), "stage": stage, "in": tin, "out": tout,
                                    "usd": usd(tin, tout), "finish": fin, "attempt": attempt}) + "\n")
            if fin == "length" and attempt == 0:
                cap = max_tokens * 2
                continue
            if fin == "length" or not text:
                u["failed"] = True
                return "", u
            return text, u
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
            with _lock, CALLS.open("a") as f:
                f.write(json.dumps({"t": time.time(), "stage": stage, "in": 0, "out": 0, "usd": 0,
                                    "error": str(e)[:160], "attempt": attempt}) + "\n")
            time.sleep(3)
    return "", {"failed": True}


def stage_totals(stage, since=0.0):
    n = tin = tout = 0; u = 0.0
    for l in CALLS.open():
        r = json.loads(l)
        if r["t"] >= since and (since or r["stage"] == stage):
            n += 1; tin += r["in"]; tout += r["out"]; u += r["usd"]
    return n, tin, tout, u


def log_batch(stage, est_usd, note="", since=0.0):
    n, tin, tout, u = stage_totals(stage, since)
    row = {"stage": stage, "calls": n, "input_tokens": tin, "output_tokens": tout, "est_usd": round(u, 4),
           "pre_batch_estimate_usd": round(est_usd, 4), "cumulative_usd": round(cumulative_usd(), 4),
           "price": "claude-sonnet-5 $3/$15 per M; gateway-reported usage", "note": note, "t": time.time()}
    with COST_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("COST", json.dumps(row))


def gate(stage, est_usd):
    cum = cumulative_usd()
    print(f"[gate] {stage}: estimate ${est_usd:.2f}, cumulative ${cum:.2f}, would be ${cum+est_usd:.2f} / ${CAP_USD}")
    if cum + est_usd > CAP_USD:
        raise SystemExit(f"STOP: {stage} would exceed ${CAP_USD} cap")
