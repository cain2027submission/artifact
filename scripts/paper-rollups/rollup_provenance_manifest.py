#!/usr/bin/env python3
"""Write data/paper-rollups/rollup-provenance.jsonl: one row per rollup any paper arm reads.

arms: design-uniform (25 production notes, rq3_rollup_uniform.jsonl), heldout-production (25 notes,
sessions/<id>.md for the held-out sessions), heldout-haiku-regen (25 notes regenerated with Haiku 4.5
by heldout_rollup_haiku_generate.py, with the model id the API response reported).
Existing notes carry no per-note generator field, so their rows are provenance_status=inferred.
Rerunnable; recomputes hashes from the files as they are now.
"""
import glob, hashlib, json, os, re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "paper-rollups"
H = D / "heldout-rollup-haiku"
ENC = tiktoken.get_encoding("cl100k_base")
SUMM = ROLLUP_SRC_DIR.parent.parent
ASR_SHA = os.popen(f"git -C {SUMM} rev-parse HEAD").read().strip()


def base(arm, path, sid):
    b = path.read_bytes()
    t = b.decode("utf-8", "replace")
    m = re.search(r"^date: (\S+)", t, re.M)
    return {"session_id": sid, "arm": arm, "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "cl100k_tokens": len(ENC.encode(t)),
            "note_date": m.group(1) if m else None}


rows = []
design = sorted({os.path.basename(json.loads(l)["rollup"]) for l in (D / "rq3_rollup_uniform.jsonl").read_text().splitlines() if l.strip()})
for fn in design:
    r = base("design-uniform", ROOT / "sessions" / fn, fn[:-3])
    r.update(generator_model_requested="claude-haiku-4-5 (inferred)", generator_model_returned=None,
             provenance_status="inferred",
             evidence=("no per-note generator field and no per-note call log found (the summariser only began recording calls 2026-09-27). "
                       "inferred from the summariser's code default HAIKU_MODEL=claude-haiku-4-5 "
                       "(agent-session-rollup src/agent_session_rollup/rollup_summarise.py:135, model_for():379-395) "
                       "and from paper/main.tex:578 and :731 which state the inference. NOT established: the date the "
                       "drain launch agent's PKM_ROLLUP_MODEL=claude-sonnet-5 pin was added is unknown, so a note "
                       "written 2026-09-15..26 may be Sonnet 5 output if the pin predates it"),
             generated_at=r["note_date"], summariser_code_commit=None, backend="unknown (cli by launchd pin if pinned)")
    rows.append(r)

qs = sorted({json.loads(l)["session_id"] for l in (D / "rq3_heldout_questions.jsonl").read_text().splitlines() if l.strip()})
for sid in qs:
    r = base("heldout-production", ROOT / "sessions" / f"{sid}.md", sid)
    r.update(generator_model_requested="claude-sonnet-5 (inferred)", generator_model_returned=None,
             provenance_status="inferred",
             evidence=("~/Library/LaunchAgents/com.agent-session-rollup.drain.plist lines 54-57 pin "
                       "PKM_ROLLUP_BACKEND=cli and PKM_ROLLUP_MODEL=claude-sonnet-5 (file mtime 2026-10-01 23:08; its "
                       "comment lines 11-27 say the pin was in force until 2026-10-01 and that the default "
                       "claude-haiku-4-5 answered 403 'not allowed by the caller's organization access profile'). "
                       "notes written 2026-09-27..30 fall inside the pin; no per-note log, so not attributable per note"),
             generated_at=r["note_date"], summariser_code_commit=None, backend="cli (inferred from plist pin)")
    rows.append(r)

gens = defaultdict(list)
for l in (H / "generations.jsonl").read_text().splitlines():
    if l.strip():
        g = json.loads(l); gens[g["session"]].append(g)
for sid in qs:
    p = H / "rollups" / f"{sid}.md"
    if not p.exists():
        continue
    r = base("heldout-haiku-regen", p, sid)
    calls = gens.get(sid, [])
    mods = sorted({c["model_returned"] for c in calls if c["model_returned"]})
    r.update(generator_model_requested="claude-haiku-4-5@20251001", generator_model_returned=mods[0] if len(mods) == 1 else mods,
             provenance_status="logged",
             evidence=f"generations.jsonl rows for this session ({len(calls)} calls, finish_reason {sorted({c['finish_reason'] for c in calls})}); model id is the gateway response's `model` field",
             generated_at=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
             summariser_code_commit=ASR_SHA,
             backend="gateway /v1/chat/completions at 127.0.0.1:8080 via an injected client into rollup_summarise.summarise (not the cli backend)",
             n_calls=len(calls), prompt_tokens=sum(c["prompt_tokens"] for c in calls),
             completion_tokens=sum(c["completion_tokens"] for c in calls))
    rows.append(r)

(D / "rollup-provenance.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
from collections import Counter
print(Counter((r["arm"], r["provenance_status"]) for r in rows))
