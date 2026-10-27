"""CAIN RQ1 replication: payload-character composition on public agent trajectories.

Usage: python -I public_composition.py <shard.parquet> <out.json> [n=1000] [seed=0]
Zero model calls; the parquet is untrusted data, only read via pyarrow.

Dataset: nvidia/Open-SWE-Traces, data/openhands/qwen36_27b/swe-rebench-v2 shard 0
(OpenHands harness, Qwen3.6-27B). Each row has `messages` in chat format.

Field -> category mapping (unit = characters of the string payload):
  role=system, content                    -> scaffolding (harness system prompt)
  role=user, content (task statement and
     any later user turns)                -> user_text (includes OpenHands' fixed
                                             task wrapper; not split out)
  role=assistant, content                 -> assistant_prose
  role=assistant, reasoning_content       -> reasoning
  role=assistant, tool_calls[].function   -> tool_use (name + arguments string)
  role=tool, content                      -> tool_result
List-of-parts content: text parts' "text" are concatenated; other parts are
JSON-dumped. tool I/O = tool_result + tool_use.
Shares: pooled (sum chars over sampled trajectories) and per-trajectory median.
"""
import json, random, statistics, sys
import pyarrow.parquet as pq

CATS = ["tool_result", "tool_use", "user_text", "assistant_prose", "reasoning", "scaffolding"]

def txt(c):
    if c is None: return 0
    if isinstance(c, str): return len(c)
    if isinstance(c, list):
        return sum(len(p["text"]) if isinstance(p, dict) and isinstance(p.get("text"), str) else len(json.dumps(p)) for p in c)
    return len(json.dumps(c))

def classify(msgs):
    c = dict.fromkeys(CATS, 0)
    for m in msgs:
        r = m.get("role")
        if r == "system": c["scaffolding"] += txt(m.get("content"))
        elif r == "user": c["user_text"] += txt(m.get("content"))
        elif r == "tool": c["tool_result"] += txt(m.get("content"))
        elif r == "assistant":
            c["assistant_prose"] += txt(m.get("content"))
            c["reasoning"] += txt(m.get("reasoning_content"))
            for t in m.get("tool_calls") or []:
                f = t.get("function") or {}
                c["tool_use"] += len(f.get("name") or "") + txt(f.get("arguments"))
    return c

def main():
    shard, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    pf = pq.ParquetFile(shard)
    total = pf.metadata.num_rows
    pick = set(random.Random(seed).sample(range(total), min(n, total)))
    rows, turns, idx = [], 0, 0
    for g in range(pf.num_row_groups):
        for rec in pf.read_row_group(g, columns=["messages"]).column("messages").to_pylist():
            if idx in pick:
                rows.append(classify(rec))
                turns += sum(1 for m in rec if m.get("role") == "assistant")
            idx += 1
    pooled_tot = sum(sum(r.values()) for r in rows)
    pooled = {k: 100 * sum(r[k] for r in rows) / pooled_tot for k in CATS}
    med = {}
    for k in CATS:
        med[k] = statistics.median(100 * r[k] / sum(r.values()) for r in rows if sum(r.values()))
    res = {"n_trajectories": len(rows), "assistant_turns": turns, "total_chars": pooled_tot,
           "pooled_pct": pooled, "median_pct": med,
           "tool_io_pooled_pct": pooled["tool_result"] + pooled["tool_use"],
           "tool_io_median_pct": statistics.median(100 * (r["tool_result"] + r["tool_use"]) / sum(r.values()) for r in rows if sum(r.values())),
           "pooled_pct_excl_scaffolding": {k: 100 * sum(r[k] for r in rows) / (pooled_tot - sum(r["scaffolding"] for r in rows)) for k in CATS if k != "scaffolding"},
           "seed": seed, "shard_rows": total}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))

main()
