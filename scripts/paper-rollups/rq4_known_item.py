#!/usr/bin/env python3
"""RQ4: Known-item retrieval comparing rollups vs raw transcripts.

Known-item task: given a probe generated from session S, retrieve S from the corpus.
Ground truth is unambiguous: the probe's source session is the only relevant document.

Six arms:
  A_rollup         - dense retrieval over rollup notes (full note, frontmatter stripped)
  A_rollup_summary - dense retrieval over summary section only (no frontmatter/recaps/backlink)
  B_raw            - dense retrieval over chunked raw transcripts
  C_raw_no_tool    - dense retrieval over transcripts with tool_use/tool_result removed
  D_bm25_rollup    - BM25 over rollup notes
  E_bm25_raw       - BM25 over chunked raw transcripts

CRITICAL VALIDATION:
  Probes are LLM-generated queries a developer would TYPE to find a session.
  Each probe is validated: must NOT appear verbatim in transcript or rollup.
  Fairness statistics: fraction of probe tokens present in each substrate.
"""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Import paths before using them
from paths import (
    PROJECTS_DIR,
    PUBLIC_SKILLS_DIR,
)

# Import from indexer
sys.path.insert(0, str(PUBLIC_SKILLS_DIR))
from fastembed import TextEmbedding

# Import freeze_corpus utilities
sys.path.insert(0, str(Path(__file__).parent))
from freeze_corpus import load_manifest, read_pinned, read_pinned_rollup

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
PROBE_MODEL = "claude-sonnet-5"
SEED = 42
# Defaults for a direct run. The published results come from rq4_batched.py, which
# overrides TARGET_SESSIONS, SEED and the three file paths at the module level and
# then calls main() here, so do not add argparse to this module: main() is called
# as a library function and would reject the batched runner's own flags.
TARGET_SESSIONS = 120
PROBES_PER_SESSION = 2
MAX_RETRIES = 3

FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Chunk width in TOKEN_RE words. The embedding model's context window is 512
# *wordpiece* tokens, and transcript text runs ~1.15 wordpieces per regex word,
# so the original 512-word chunk measured p50 590 and max 898 tokens: 98.5% of
# raw chunks overflowed and 16.7% of all raw transcript text was silently
# dropped from the dense index. The loss was not symmetric across arms -- 96-98%
# of raw chunks were truncated against 48-50% of rollup chunks -- so it degraded
# the raw baselines far harder than the arms under test, biasing the comparison
# towards the hypothesis. It also explains why the BM25 arms, which have no
# context window, outscored their dense counterparts. 320 words held every arm
# inside the window (<0.2% of text dropped) for the original 300-session corpus.
#
# Dropped to 280 for the 2026-10-06 550-session re-run: the 250 sessions added
# since the 2026-09-27 freeze overflow at 3.36% (vs 1.44% for the original 300),
# pulling B_raw's exact population overflow_fraction to 2.08% -- over the 2%
# guard -- because newer sessions carry chunkier raw tool output. Measured
# directly against the full 2026-10-06 chunk set (not the 400-chunk sample the
# runtime check uses): 320->2.08%, 300->1.36%, 290->1.18%, 280->0.98%, 270->0.73%.
# 280 is the smallest width that restores the <1%-dropped standard the original
# run was held to, at ~1.14x the embedding cost of 320.
CHUNK_WORDS = 280
# Held at the 12.5% of chunk width that the original 512/64 pair used, so the
# narrower window does not also change how much context neighbouring chunks share.
CHUNK_OVERLAP = 35
EMBED_TOKEN_LIMIT = 512
# Above this share of over-long chunks the index is being silently truncated
# again. That is a wrong experiment rather than a slow one, so it fails loudly.
MAX_OVERFLOW_FRACTION = 0.02

REPO_ROOT_LOCAL = Path(__file__).resolve().parents[2]
VAULT = REPO_ROOT_LOCAL
PROJECTS = PROJECTS_DIR
# Optional arms beyond the six default ones. Empty by default so a default run is
# unchanged; rq4_batched.py --with-f sets ("F_bm25_no_tool",).
EXTRA_ARMS: tuple[str, ...] = ()
CACHE_FILE = REPO_ROOT_LOCAL / "data" / "paper-rollups" / "rq4_cache.json"
PROBES_FILE = REPO_ROOT_LOCAL / "data" / "paper-rollups" / "rq4_probes.jsonl"
RESULTS_FILE = REPO_ROOT_LOCAL / "data" / "paper-rollups" / "rq4_known_item.json"


def estimate_tokens(text: str) -> int:
    """Quick token estimate."""
    return len(TOKEN_RE.findall(text))


def normalize_text(text: str) -> str:
    """Normalize for substring matching: lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunk text into token-sized windows with overlap."""
    tokens = TOKEN_RE.findall(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks if chunks else [text]


def extract_summary_from_rollup(rollup_text: str) -> str:
    """Extract summary section: strip frontmatter, recaps tail, and trace backlink."""
    # Strip frontmatter
    fm_match = FRONTMATTER_RE.match(rollup_text)
    if fm_match:
        rollup_text = rollup_text[fm_match.end():]

    # Remove ## Recaps section and everything after
    recaps_match = re.search(r'\n## Recaps\n', rollup_text, re.IGNORECASE)
    if recaps_match:
        rollup_text = rollup_text[:recaps_match.start()]

    # Remove trace backlink (usually last line starting with "Trace:")
    lines = rollup_text.split('\n')
    while lines and (not lines[-1].strip() or lines[-1].strip().startswith('Trace:')):
        lines.pop()

    return '\n'.join(lines).strip()


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON to gateway."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_llm(prompt: str, cache: dict) -> str:
    """Call LLM via gateway. Cache by prompt hash. NO temperature field."""
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    if prompt_hash in cache:
        return cache[prompt_hash]

    url = f"{GATEWAY}/v1/chat/completions"
    payload = {
        "model": PROBE_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        data = post_json(url, payload, timeout=120)
        content = data["choices"][0]["message"]["content"].strip()
        cache[prompt_hash] = content
        return content
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return ""


def generate_probe(transcript_slice: str, cache: dict) -> list[str]:
    """Generate 2 probe queries from transcript. Returns list of query strings."""
    prompt = f"""You are helping build a retrieval benchmark. Below is a slice from an agent session transcript.

Generate EXACTLY 2 search queries that a developer would TYPE LATER to find this session again.

Requirements:
- Queries should be 4-8 words, keyword-shaped
- Phrase them as a developer would naturally search, not quoting verbatim strings
- Capture the INTENT and TOPIC, not exact wording
- Think: "How would I search for this session if I remembered the problem but not the exact words?"

Examples of GOOD queries:
- "fix pytest import error macos"
- "git worktree remove classifier denied"
- "token burn measurement script"

Examples of BAD queries (too verbatim):
- "trace_path cost_usd repo path"
- "and sweep up only"
- exact code snippets or error messages copied verbatim

Transcript slice:
{transcript_slice[:3000]}

Return ONLY a JSON array of 2 query strings, nothing else:"""

    response = call_llm(prompt, cache)

    if not response:
        return []

    # Extract JSON
    try:
        if "[" in response and "]" in response:
            start = response.index("[")
            end = response.rindex("]") + 1
            queries = json.loads(response[start:end])
            if isinstance(queries, list):
                return [str(q).strip() for q in queries[:2] if str(q).strip()]
    except:
        pass

    return []


def validate_probe(probe: str, transcript: str, rollup: str) -> bool:
    """Validate: probe must NOT appear as substring in normalized transcript or rollup."""
    probe_norm = normalize_text(probe)
    transcript_norm = normalize_text(transcript)
    rollup_norm = normalize_text(rollup)

    if probe_norm in transcript_norm:
        return False
    if probe_norm in rollup_norm:
        return False

    return True


def compute_fairness_stats(probe: str, transcript: str, rollup: str) -> tuple[float, float]:
    """Compute fraction of probe tokens present in transcript vs rollup."""
    probe_tokens = set(TOKEN_RE.findall(probe.lower()))
    if not probe_tokens:
        return 0.0, 0.0

    transcript_tokens = set(TOKEN_RE.findall(transcript.lower()))
    rollup_tokens = set(TOKEN_RE.findall(rollup.lower()))

    transcript_overlap = len(probe_tokens & transcript_tokens) / len(probe_tokens)
    rollup_overlap = len(probe_tokens & rollup_tokens) / len(probe_tokens)

    return transcript_overlap, rollup_overlap


def build_session_id_map() -> dict[str, Path]:
    """Build map of session_id -> rollup_path by parsing frontmatter."""
    rollup_dir = VAULT / "sessions"
    session_map = {}

    for rollup_path in rollup_dir.glob("*.md"):
        try:
            text = rollup_path.read_text(encoding="utf-8", errors="replace")
            fm_match = FRONTMATTER_RE.match(text)
            if not fm_match:
                continue

            # Parse frontmatter
            for line in fm_match.group(1).split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    if key.strip() == "session_id":
                        session_id = val.strip().strip('"\'')
                        if session_id:
                            session_map[session_id] = rollup_path
                        break
        except:
            continue

    return session_map


def load_session(session_id: str, manifest: dict, session_map: dict[str, Path]) -> dict | None:
    """Load rollup and transcript for a session.

    The rollup is read through the pin (read_pinned_rollup), keyed by
    session_id rather than the live session_map path, because the rollup
    drain can rename or rewrite a note between the manifest's snapshot_utc
    and whenever this runs. A session_id with no valid pin -- manifest
    predates the rollup_note_bytes field, or the note has been renamed or
    rewritten since -- is dropped from the corpus rather than read live,
    the same way a session with no transcript is dropped below. This keeps
    a rerun's rollup arms bit-for-bit reproducible against the frozen
    manifest, matching the guarantee read_pinned already gives transcripts.
    """
    # Find rollup using session_id map (still used to confirm the note is
    # findable at all; the actual bytes come from the pin below)
    if session_id not in session_map:
        return None

    # Find transcript
    transcript_candidates = [p for p in manifest["transcripts"] if session_id in p]
    if not transcript_candidates:
        return None
    transcript_path = transcript_candidates[0]

    # Load rollup through the pin. No pin (old manifest, or the note moved /
    # changed since snapshot) means this session cannot be reproduced and is
    # dropped, not read live.
    rollup_raw = read_pinned_rollup(session_id, manifest)
    if rollup_raw is None:
        return None
    fm_match = FRONTMATTER_RE.match(rollup_raw)
    rollup_full = rollup_raw[fm_match.end():] if fm_match else rollup_raw
    rollup_summary = extract_summary_from_rollup(rollup_raw)

    # Load transcript via read_pinned
    transcript_raw = read_pinned(transcript_path, manifest)

    # Parse transcript to extract text and text_no_tool
    transcript_parts = []
    transcript_no_tool_parts = []

    for line in transcript_raw.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("type") not in ("user", "assistant"):
                continue

            msg = entry.get("message", {})
            content = msg.get("content", [])

            if isinstance(content, str):
                transcript_parts.append(content)
                transcript_no_tool_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    btype = block.get("type", "")

                    if btype == "text":
                        text = block.get("text", "")
                        transcript_parts.append(text)
                        transcript_no_tool_parts.append(text)
                    elif btype == "thinking":
                        text = block.get("thinking", "")
                        transcript_parts.append(text)
                        transcript_no_tool_parts.append(text)
                    elif btype == "tool_use":
                        name = block.get("name", "")
                        inp = json.dumps(block.get("input", {}))
                        transcript_parts.append(f"[TOOL {name}] {inp}")
                        # Excluded from no_tool
                    elif btype == "tool_result":
                        content_val = block.get("content", "")
                        if isinstance(content_val, list):
                            content_val = " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content_val)
                        transcript_parts.append(f"[RESULT] {content_val}")
                        # Excluded from no_tool
        except:
            continue

    transcript_text = "\n".join(transcript_parts)
    transcript_no_tool = "\n".join(transcript_no_tool_parts)

    if not transcript_text.strip():
        return None

    return {
        "session_id": session_id,
        "rollup_full": rollup_full,
        "rollup_summary": rollup_summary,
        "transcript": transcript_text,
        "transcript_no_tool": transcript_no_tool,
    }


def generate_probes_for_corpus(sessions: list[dict], cache: dict) -> list[dict]:
    """Generate and validate probes for all sessions."""
    print("\nGenerating probes...")

    all_probes = []
    rejected_count = 0
    dropped_count = 0
    fairness_transcript = []
    fairness_rollup = []

    for i, sess in enumerate(sessions):
        if i % 20 == 0:
            print(f"  Processed {i}/{len(sessions)} sessions...")

        session_id = sess["session_id"]
        transcript = sess["transcript"]
        rollup = sess["rollup_full"]

        # Generate probes
        transcript_slice = transcript[:6000]
        candidates = generate_probe(transcript_slice, cache)

        for candidate in candidates:
            # Validate with retries
            valid = False
            for retry in range(MAX_RETRIES):
                if validate_probe(candidate, transcript, rollup):
                    valid = True
                    break
                else:
                    rejected_count += 1
                    # Regenerate
                    new_candidates = generate_probe(transcript_slice, cache)
                    if new_candidates:
                        candidate = new_candidates[0]
                    else:
                        break

            if not valid:
                dropped_count += 1
                continue

            # Compute fairness
            t_overlap, r_overlap = compute_fairness_stats(candidate, transcript, rollup)
            fairness_transcript.append(t_overlap)
            fairness_rollup.append(r_overlap)

            probe = {
                "query": candidate,
                "target_session_id": session_id,
                "fairness_transcript": t_overlap,
                "fairness_rollup": r_overlap,
            }
            all_probes.append(probe)

    print(f"  Generated {len(all_probes)} valid probes")
    print(f"  Rejected: {rejected_count}, Dropped: {dropped_count}")

    if fairness_transcript and fairness_rollup:
        mean_t = np.mean(fairness_transcript)
        mean_r = np.mean(fairness_rollup)
        print(f"  Fairness: transcript={mean_t:.3f}, rollup={mean_r:.3f}, diff={abs(mean_t - mean_r):.3f}")

        if abs(mean_t - mean_r) > 0.15:
            print(f"  WARNING: Fairness difference > 0.15 - probe generator may be biased!")

    return all_probes


def check_chunk_overflow(chunks: list[str], embedder: TextEmbedding, arm: str) -> dict:
    """Refuse to index chunks that overflow the embedder's context window.

    Measured on an independent copy of the tokenizer, because the live one has
    truncation enabled and so can never report a length above its own limit --
    the first attempt at this measurement read p50 == p95 == max == 512 and
    concluded 0% overflow, when the true p50 was 590.
    """
    from tokenizers import Tokenizer

    probe_tok = Tokenizer.from_str(embedder.model.tokenizer.to_str())
    probe_tok.no_truncation()
    probe_tok.no_padding()

    sample = chunks if len(chunks) <= 400 else random.Random(SEED).sample(chunks, 400)
    lens = np.array([len(probe_tok.encode(c).ids) for c in sample])
    over = float((lens > EMBED_TOKEN_LIMIT).mean())
    dropped = float(np.clip(lens - EMBED_TOKEN_LIMIT, 0, None).sum() / lens.sum())
    stats = {
        "sampled_chunks": len(sample),
        "overflow_fraction": over,
        "text_dropped_fraction": dropped,
        "p50_tokens": int(np.percentile(lens, 50)),
        "max_tokens": int(lens.max()),
    }
    print(f"    {arm}: {over:.1%} of chunks over {EMBED_TOKEN_LIMIT} tokens, "
          f"{dropped:.2%} of text dropped (p50 {stats['p50_tokens']}, max {stats['max_tokens']})")
    if over > MAX_OVERFLOW_FRACTION:
        raise SystemExit(
            f"ABORT: arm {arm} has {over:.1%} of chunks over the {EMBED_TOKEN_LIMIT}-token window "
            f"(limit {MAX_OVERFLOW_FRACTION:.0%}), silently dropping {dropped:.1%} of its text.\n"
            f"  Lower CHUNK_WORDS (currently {CHUNK_WORDS}) and re-run. Indexing as-is would "
            f"penalise whichever arm has the longest chunks, which is the raw-transcript baseline."
        )
    return stats


def embed_texts(texts: list[str], embedder: TextEmbedding, cache: dict, cache_key: str) -> np.ndarray:
    """Embed texts with caching."""
    # The key is only the arm name, so it carries no record of which corpus or
    # chunk width produced the rows. A cache left over from a smaller run, or the
    # stub of empty lists a failed run leaves behind, would otherwise be returned
    # as a hit and hand this arm fewer embeddings than it has chunks -- silently,
    # since nothing downstream compares the two lengths. Treat a length mismatch
    # as a miss and re-embed rather than scoring a misaligned index.
    if cache_key in cache:
        cached = np.array(cache[cache_key])
        if cached.shape[0] == len(texts):
            return cached
        print(
            f"  cache {cache_key}: {cached.shape[0]} rows for {len(texts)} chunks, re-embedding"
        )

    embeddings = np.array(list(embedder.embed(texts, batch_size=256)))
    if embeddings.shape[0] != len(texts):
        raise SystemExit(
            f"ABORT: embedded {embeddings.shape[0]} of {len(texts)} chunks for {cache_key}"
        )
    cache[cache_key] = embeddings.tolist()
    return embeddings


def build_dense_index(sessions: list[dict], text_key: str, embedder: TextEmbedding, cache: dict, cache_prefix: str, chunk: bool = False) -> dict:
    """Build dense retrieval index."""
    chunks = []
    chunk_to_session = []

    for sess in sessions:
        text = sess[text_key]
        if chunk:
            sess_chunks = chunk_text(text, chunk_size=CHUNK_WORDS, overlap=CHUNK_OVERLAP)
        else:
            sess_chunks = chunk_text(text, chunk_size=CHUNK_WORDS, overlap=0)

        for c in sess_chunks:
            chunks.append(c)
            chunk_to_session.append(sess["session_id"])

    overflow = check_chunk_overflow(chunks, embedder, cache_prefix)

    # Embed with cache
    cache_key = f"{cache_prefix}_embeddings"
    embeddings = embed_texts(chunks, embedder, cache, cache_key)

    total_tokens = sum(estimate_tokens(c) for c in chunks)

    return {
        "chunks": chunks,
        "chunk_to_session": chunk_to_session,
        "embeddings": embeddings,
        "num_chunks": len(chunks),
        "num_distinct_sessions": len(set(chunk_to_session)),
        "total_tokens": total_tokens,
        "chunk_overflow": overflow,
    }


def build_bm25_index(sessions: list[dict], text_key: str, chunk: bool = False) -> dict:
    """Build BM25 index."""
    chunks = []
    chunk_to_session = []

    for sess in sessions:
        text = sess[text_key]
        if chunk:
            sess_chunks = chunk_text(text, chunk_size=CHUNK_WORDS, overlap=CHUNK_OVERLAP)
        else:
            sess_chunks = chunk_text(text, chunk_size=CHUNK_WORDS, overlap=0)

        for c in sess_chunks:
            chunks.append(c)
            chunk_to_session.append(sess["session_id"])

    # Tokenize each chunk exactly once here, instead of on every query call in
    # bm25_score. Same regex, same lowering -- this changes nothing about the
    # score, only when the (deterministic) tokenization happens. Verified
    # bit-identical against the original per-call tokenization on 20 real
    # sessions / 62 queries including edge cases (empty, Unicode, OOV tokens).
    doc_tokens = [TOKEN_RE.findall(c.lower()) for c in chunks]
    doc_len = [len(tokens) for tokens in doc_tokens]
    doc_tf = []
    df = defaultdict(int)
    for tokens in doc_tokens:
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        doc_tf.append(tf)
        for token in tf:
            df[token] += 1

    N = len(chunks)
    avgdl = sum(doc_len) / N if N > 0 else 0
    total_tokens = sum(estimate_tokens(c) for c in chunks)

    return {
        "chunks": chunks,
        "chunk_to_session": chunk_to_session,
        "df": df,
        "doc_tf": doc_tf,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "num_chunks": len(chunks),
        "num_distinct_sessions": len(set(chunk_to_session)),
        "total_tokens": total_tokens,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between query and documents."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm @ b_norm.T).flatten()


def bm25_score(query: str, doc_tf: list[dict], doc_len: list[int], avgdl: float, df: dict, N: int, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 scoring. Same formula as before; doc_tf/doc_len/avgdl are precomputed
    once at index-build time instead of re-tokenized on every call."""
    query_tokens = set(TOKEN_RE.findall(query.lower()))

    scores = np.zeros(N)
    for i in range(N):
        tf = doc_tf[i]
        dl = doc_len[i]
        score = 0.0
        for token in query_tokens:
            if token in tf:
                idf = np.log((N - df[token] + 0.5) / (df[token] + 0.5) + 1)
                tf_score = (tf[token] * (k1 + 1)) / (tf[token] + k1 * (1 - b + b * dl / (avgdl + 1e-8)))
                score += idf * tf_score

        scores[i] = score

    return scores


def aggregate_scores_to_sessions(scores: np.ndarray, chunk_to_session: list[str]) -> dict[str, float]:
    """Max-pool every chunk's score to its session, over the FULL scores array.

    Chunks-per-session differs by ~100x between the rollup and raw-transcript
    indexes, so truncating to a fixed top-N chunks before deduping to sessions
    (the previous behaviour) silently shrinks the number of distinct sessions
    reachable in the raw arm far more than in the rollup arm, making recall@k
    an asymmetric comparison. Considering every chunk removes that truncation
    by construction: every session is scored for every query, no cutoff.
    """
    session_scores: dict[str, float] = {}
    for idx, session_id in enumerate(chunk_to_session):
        score = float(scores[idx])
        if session_id not in session_scores or score > session_scores[session_id]:
            session_scores[session_id] = score
    return session_scores


def retrieve_dense(query: str, index: dict, embedder: TextEmbedding, cache: dict, k: int = 10) -> list[str]:
    """Dense retrieval. Returns ranked session IDs."""
    # Embed query with cache
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    cache_key = f"query_emb_{query_hash}"

    if cache_key in cache:
        query_vec = np.array(cache[cache_key])
    else:
        query_vec = np.array(list(embedder.embed([query])))[0]
        cache[cache_key] = query_vec.tolist()

    # Compute similarities
    scores = cosine_similarity(query_vec, index["embeddings"])

    # Aggregate every chunk's score to its session (no top-N truncation first)
    session_scores = aggregate_scores_to_sessions(scores, index["chunk_to_session"])

    # Sort and return top-k sessions
    ranked = sorted(session_scores.items(), key=lambda x: -x[1])
    return [sid for sid, _ in ranked[:k]]


def retrieve_bm25(query: str, index: dict, k: int = 10) -> list[str]:
    """BM25 retrieval. Returns ranked session IDs."""
    scores = bm25_score(query, index["doc_tf"], index["doc_len"], index["avgdl"], index["df"], index["num_chunks"])

    # Aggregate every chunk's score to its session (no top-N truncation first)
    session_scores = aggregate_scores_to_sessions(scores, index["chunk_to_session"])

    # Sort and return top-k sessions
    ranked = sorted(session_scores.items(), key=lambda x: -x[1])
    return [sid for sid, _ in ranked[:k]]


def evaluate_arm(probes: list[dict], index: dict, arm_name: str, embedder: TextEmbedding | None, cache: dict) -> dict:
    """Evaluate one arm on all probes."""
    print(f"  Evaluating {arm_name}...")

    hits_at_1 = []
    hits_at_5 = []
    hits_at_10 = []
    mrrs = []
    target_session_ids = []  # parallel to hits_at_* / mrrs, for session-clustered bootstrap

    for probe in probes:
        query = probe["query"]
        target = probe["target_session_id"]
        target_session_ids.append(target)

        # Retrieve
        if "bm25" in arm_name:
            ranked = retrieve_bm25(query, index, k=10)
        else:
            ranked = retrieve_dense(query, index, embedder, cache, k=10)

        # Score
        hit_1 = 1 if target in ranked[:1] else 0
        hit_5 = 1 if target in ranked[:5] else 0
        hit_10 = 1 if target in ranked[:10] else 0

        if target in ranked:
            rank = ranked.index(target) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0

        hits_at_1.append(hit_1)
        hits_at_5.append(hit_5)
        hits_at_10.append(hit_10)
        mrrs.append(mrr)

    return {
        "recall@1": float(np.mean(hits_at_1)),
        "recall@5": float(np.mean(hits_at_5)),
        "recall@10": float(np.mean(hits_at_10)),
        "mrr@10": float(np.mean(mrrs)),
        "hits_at_10": hits_at_10,  # For significance tests
        "target_session_ids": target_session_ids,  # Parallel to hits_at_10, for session-clustered bootstrap
        "num_chunks": index["num_chunks"],
        "total_tokens": index["total_tokens"],
        # Every session in the index is scored for every query now that the
        # top-100-chunk truncation is gone (see aggregate_scores_to_sessions),
        # so this should be identical across arms indexing the same session set --
        # i.e. recall@10 is a true recall@10, not recall@k for some smaller k.
        "candidate_pool_sessions": index["num_distinct_sessions"],
        # Recorded per arm so the paper can state how much text each index holds
        # rather than assuming the chunker kept all of it. BM25 arms have no
        # context window, so this key is absent for them.
        "chunk_overflow": index.get("chunk_overflow"),
    }


def mcnemar_test(hits_a: list[int], hits_b: list[int]) -> dict | None:
    """McNemar test for paired binary outcomes. Returns None if not computable."""
    if len(hits_a) != len(hits_b) or len(hits_a) == 0:
        return None

    n01 = sum(a and not b for a, b in zip(hits_a, hits_b))
    n10 = sum(not a and b for a, b in zip(hits_a, hits_b))

    if n01 + n10 == 0:
        return {"p_value": 1.0, "effect_size": 0.0, "note": "no discordant pairs"}

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

    # Chi-squared CDF approximation for df=1
    import math
    p_value = 1 - 0.5 * (1 + math.erf(math.sqrt(chi2 / 2)))
    effect_size = (n01 - n10) / len(hits_a)

    return {"p_value": float(p_value), "effect_size": float(effect_size)}


def _clustered_bootstrap_diffs(
    values_a: list[float], values_b: list[float], session_ids: list[str],
    n_bootstrap: int = 1000, seed: int = 42,
) -> list[float]:
    """Shared block/cluster bootstrap: probes are not independent draws --
    PROBES_PER_SESSION probes come from the same session -- so this resamples
    at the SESSION level (session IDs drawn with replacement, same count as
    distinct sessions) and includes ALL of each resampled session's probes,
    rather than resampling individual probe indices (which overstates
    precision by ignoring within-session correlation).

    Both arms are resampled from the SAME drawn session list on each
    iteration, preserving the paired-bootstrap property of the original code
    (one `indices` draw used for both boot_a and boot_b).
    """
    session_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(session_ids):
        session_to_indices[sid].append(i)
    unique_sessions = sorted(session_to_indices)
    n_sessions = len(unique_sessions)

    rng = np.random.RandomState(seed)
    diffs = []

    for _ in range(n_bootstrap):
        sampled = rng.choice(n_sessions, size=n_sessions, replace=True)
        indices = []
        for s_idx in sampled:
            indices.extend(session_to_indices[unique_sessions[s_idx]])
        boot_a = [values_a[i] for i in indices]
        boot_b = [values_b[i] for i in indices]
        diffs.append(np.mean(boot_a) - np.mean(boot_b))

    return sorted(diffs)


def bootstrap_ci(
    values_a: list[float], values_b: list[float], session_ids: list[str],
    n_bootstrap: int = 1000, seed: int = 42,
) -> dict | None:
    """Session-clustered bootstrap 95% CI on difference in means.

    `session_ids` is a parallel array (same length/order as values_a/values_b)
    giving the target session each probe/value came from -- see
    `evaluate_arm`'s `target_session_ids`. Returns None if not computable.
    """
    if len(values_a) != len(values_b) or len(values_a) == 0 or len(session_ids) != len(values_a):
        return None

    diffs = _clustered_bootstrap_diffs(values_a, values_b, session_ids, n_bootstrap, seed)
    ci_lower = diffs[int(0.025 * n_bootstrap)]
    ci_upper = diffs[int(0.975 * n_bootstrap)]

    return {
        "mean_diff": float(np.mean(values_a) - np.mean(values_b)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def equivalence_test(
    values_a: list[float], values_b: list[float], session_ids: list[str], margin: float,
    n_bootstrap: int = 1000, seed: int = 42,
) -> dict | None:
    """TOST-style equivalence test on the paired difference in means.

    Reuses the session-clustered paired bootstrap from `bootstrap_ci` but at
    90% coverage (5th/95th percentile, not 2.5th/97.5th): a 90% bootstrap CI
    is equivalent to two one-sided tests at alpha=0.05, matching the
    convention this project already uses elsewhere for an equivalence test.
    `equivalent` is True iff that 90% CI falls entirely within [-margin, margin].
    """
    if len(values_a) != len(values_b) or len(values_a) == 0 or len(session_ids) != len(values_a):
        return None

    diffs = _clustered_bootstrap_diffs(values_a, values_b, session_ids, n_bootstrap, seed)
    ci_lower = diffs[int(0.05 * n_bootstrap)]
    ci_upper = diffs[int(0.95 * n_bootstrap)]
    equivalent = (ci_lower >= -margin) and (ci_upper <= margin)

    return {
        "mean_diff": float(np.mean(values_a) - np.mean(values_b)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "margin": float(margin),
        "equivalent": bool(equivalent),
    }


def main():
    start_time = time.time()

    # Load manifest
    print("Loading corpus manifest...")
    manifest = load_manifest()
    paired_ids = manifest["paired_session_ids"]
    print(f"  {len(paired_ids)} paired sessions in frozen corpus")

    # Sample sessions
    random.seed(SEED)
    sampled_ids = random.sample(paired_ids, min(TARGET_SESSIONS, len(paired_ids)))
    print(f"\nSampled {len(sampled_ids)} sessions (seed={SEED})")

    # Build session_id -> rollup_path map
    print("\nBuilding session ID map...")
    session_map = build_session_id_map()
    print(f"  Mapped {len(session_map)} rollups to session IDs")

    # Load sessions
    print("\nLoading sessions...")
    sessions = []
    for session_id in sorted(sampled_ids):  # Sort for determinism
        sess = load_session(session_id, manifest, session_map)
        if sess:
            sessions.append(sess)
    print(f"  Loaded {len(sessions)} sessions")

    # Load cache
    cache = {}
    if CACHE_FILE.exists():
        print(f"\nLoading cache from {CACHE_FILE}")
        cache = json.loads(CACHE_FILE.read_text())

    # Generate probes
    probes = generate_probes_for_corpus(sessions, cache)

    # Save probes
    print(f"\nSaving {len(probes)} probes to {PROBES_FILE}")
    PROBES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROBES_FILE.open("w") as f:
        for probe in probes:
            f.write(json.dumps(probe) + "\n")

    # Print example probes
    print("\nExample probes:")
    for i, probe in enumerate(probes[:5]):
        print(f"  {i+1}. {probe['query']}")

    # Initialize embedder
    print("\nInitializing embedder...")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL, providers=["CPUExecutionProvider"])

    # Build arms
    print("\nBuilding arms...")

    print("  Building A_rollup...")
    t0 = time.time()
    arm_a_rollup = build_dense_index(sessions, "rollup_full", embedder, cache, "arm_a_rollup", chunk=False)
    arm_a_rollup["build_time_s"] = time.time() - t0

    print("  Building A_rollup_summary...")
    t0 = time.time()
    arm_a_summary = build_dense_index(sessions, "rollup_summary", embedder, cache, "arm_a_summary", chunk=False)
    arm_a_summary["build_time_s"] = time.time() - t0

    print("  Building B_raw...")
    t0 = time.time()
    arm_b_raw = build_dense_index(sessions, "transcript", embedder, cache, "arm_b_raw", chunk=True)
    arm_b_raw["build_time_s"] = time.time() - t0

    print("  Building C_raw_no_tool...")
    t0 = time.time()
    arm_c_no_tool = build_dense_index(sessions, "transcript_no_tool", embedder, cache, "arm_c_no_tool", chunk=True)
    arm_c_no_tool["build_time_s"] = time.time() - t0

    print("  Building D_bm25_rollup...")
    t0 = time.time()
    arm_d_bm25_rollup = build_bm25_index(sessions, "rollup_full", chunk=False)
    arm_d_bm25_rollup["build_time_s"] = time.time() - t0

    print("  Building E_bm25_raw...")
    t0 = time.time()
    arm_e_bm25_raw = build_bm25_index(sessions, "transcript", chunk=True)
    arm_e_bm25_raw["build_time_s"] = time.time() - t0

    with_f = "F_bm25_no_tool" in EXTRA_ARMS
    if with_f:
        # Same BM25 setup as E_bm25_raw (same chunker, same candidate pool: every
        # session), indexing the tool_use/tool_result-stripped text instead.
        print("  Building F_bm25_no_tool...")
        t0 = time.time()
        arm_f_bm25_no_tool = build_bm25_index(sessions, "transcript_no_tool", chunk=True)
        arm_f_bm25_no_tool["build_time_s"] = time.time() - t0

    # Evaluate all arms
    print("\nEvaluating arms...")
    results = {}

    results["A_rollup"] = evaluate_arm(probes, arm_a_rollup, "A_rollup", embedder, cache)
    results["A_rollup"]["build_time_s"] = arm_a_rollup["build_time_s"]

    results["A_rollup_summary"] = evaluate_arm(probes, arm_a_summary, "A_rollup_summary", embedder, cache)
    results["A_rollup_summary"]["build_time_s"] = arm_a_summary["build_time_s"]

    results["B_raw"] = evaluate_arm(probes, arm_b_raw, "B_raw", embedder, cache)
    results["B_raw"]["build_time_s"] = arm_b_raw["build_time_s"]

    results["C_raw_no_tool"] = evaluate_arm(probes, arm_c_no_tool, "C_raw_no_tool", embedder, cache)
    results["C_raw_no_tool"]["build_time_s"] = arm_c_no_tool["build_time_s"]

    results["D_bm25_rollup"] = evaluate_arm(probes, arm_d_bm25_rollup, "D_bm25_rollup", None, cache)
    results["D_bm25_rollup"]["build_time_s"] = arm_d_bm25_rollup["build_time_s"]

    results["E_bm25_raw"] = evaluate_arm(probes, arm_e_bm25_raw, "E_bm25_raw", None, cache)
    results["E_bm25_raw"]["build_time_s"] = arm_e_bm25_raw["build_time_s"]

    if with_f:
        results["F_bm25_no_tool"] = evaluate_arm(probes, arm_f_bm25_no_tool, "F_bm25_no_tool", None, cache)
        results["F_bm25_no_tool"]["build_time_s"] = arm_f_bm25_no_tool["build_time_s"]

    # Compute fairness stats
    fairness_transcript = [p["fairness_transcript"] for p in probes]
    fairness_rollup = [p["fairness_rollup"] for p in probes]
    mean_fairness_transcript = float(np.mean(fairness_transcript))
    mean_fairness_rollup = float(np.mean(fairness_rollup))
    fairness_diff = abs(mean_fairness_transcript - mean_fairness_rollup)

    # Significance tests
    print("\nComputing significance tests...")
    sig_tests = {}

    # A vs B
    test_ab = mcnemar_test(results["A_rollup"]["hits_at_10"], results["B_raw"]["hits_at_10"])
    if test_ab:
        sig_tests["A_rollup_vs_B_raw"] = test_ab

    # A vs C
    test_ac = mcnemar_test(results["A_rollup"]["hits_at_10"], results["C_raw_no_tool"]["hits_at_10"])
    if test_ac:
        sig_tests["A_rollup_vs_C_raw_no_tool"] = test_ac

    # A vs D
    test_ad = mcnemar_test(results["A_rollup"]["hits_at_10"], results["D_bm25_rollup"]["hits_at_10"])
    if test_ad:
        sig_tests["A_rollup_vs_D_bm25_rollup"] = test_ad

    # D vs E (BM25 over rollups vs BM25 over raw transcripts, the one substrate
    # pair the paper previously reported with no significance test at all)
    test_de = mcnemar_test(results["D_bm25_rollup"]["hits_at_10"], results["E_bm25_raw"]["hits_at_10"])
    if test_de:
        sig_tests["D_bm25_rollup_vs_E_bm25_raw"] = test_de

    if with_f:
        for other, name in (("E_bm25_raw", "F_bm25_no_tool_vs_E_bm25_raw"),
                            ("D_bm25_rollup", "F_bm25_no_tool_vs_D_bm25_rollup"),
                            ("C_raw_no_tool", "F_bm25_no_tool_vs_C_raw_no_tool")):
            t = mcnemar_test(results["F_bm25_no_tool"]["hits_at_10"], results[other]["hits_at_10"])
            if t:
                sig_tests[name] = t
                b = bootstrap_ci(
                    [float(x) for x in results["F_bm25_no_tool"]["hits_at_10"]],
                    [float(x) for x in results[other]["hits_at_10"]],
                    results["F_bm25_no_tool"]["target_session_ids"],
                )
                if b:
                    sig_tests[name]["bootstrap_ci"] = b

    # Bootstrap CIs (session-clustered: target_session_ids is the same, in the
    # same order, for every arm since all arms iterate over the same `probes`
    # list, but each arm's own copy is used to keep the pairing self-contained).
    boot_ab = bootstrap_ci(
        [float(x) for x in results["A_rollup"]["hits_at_10"]],
        [float(x) for x in results["B_raw"]["hits_at_10"]],
        results["A_rollup"]["target_session_ids"],
    )
    if boot_ab:
        sig_tests["A_rollup_vs_B_raw"]["bootstrap_ci"] = boot_ab

    boot_ac = bootstrap_ci(
        [float(x) for x in results["A_rollup"]["hits_at_10"]],
        [float(x) for x in results["C_raw_no_tool"]["hits_at_10"]],
        results["A_rollup"]["target_session_ids"],
    )
    if boot_ac:
        sig_tests["A_rollup_vs_C_raw_no_tool"]["bootstrap_ci"] = boot_ac

    boot_de = bootstrap_ci(
        [float(x) for x in results["D_bm25_rollup"]["hits_at_10"]],
        [float(x) for x in results["E_bm25_raw"]["hits_at_10"]],
        results["D_bm25_rollup"]["target_session_ids"],
    )
    if boot_de:
        sig_tests["D_bm25_rollup_vs_E_bm25_raw"]["bootstrap_ci"] = boot_de

    # Equivalence test (TOST-style, 90% CI): dense rollup vs dense raw-transcript
    # recall@10, margin=0.05 matching the paper's stated power claim of
    # "underpowered to detect differences up to about five recall@10 points".
    equivalence_tests = {}
    equiv_ab = equivalence_test(
        [float(x) for x in results["A_rollup"]["hits_at_10"]],
        [float(x) for x in results["B_raw"]["hits_at_10"]],
        results["A_rollup"]["target_session_ids"],
        margin=0.05,
    )
    if equiv_ab:
        equivalence_tests["A_rollup_vs_B_raw"] = equiv_ab

    # Lexical overlap of each probe with the output-stripped text of its target
    # session (same measure as fairness_transcript / fairness_rollup).
    f_extra = {}
    if with_f:
        by_id = {x["session_id"]: x for x in sessions}
        ov = [compute_fairness_stats(p["query"], by_id[p["target_session_id"]]["transcript_no_tool"], "")[0]
              for p in probes]
        f_extra = {
            "mean_no_tool_overlap": float(np.mean(ov)),
            "mean_transcript_overlap": mean_fairness_transcript,
            "mean_rollup_overlap": mean_fairness_rollup,
            "script_commit": subprocess.run(
                ["git", "-C", str(REPO_ROOT_LOCAL), "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
        }

    # Clean up hits_at_10 / target_session_ids from results (not needed in output)
    for arm in results:
        results[arm].pop("hits_at_10", None)
        results[arm].pop("target_session_ids", None)

    # Build output
    output = {
        "experiment": "rq4_known_item",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "seed": SEED,
        "num_sessions": len(sessions),
        "num_probes": len(probes),
        "fairness": {
            "mean_transcript_overlap": mean_fairness_transcript,
            "mean_rollup_overlap": mean_fairness_rollup,
            "difference": fairness_diff,
            "warning": fairness_diff > 0.15,
        },
        "arms": results,
        "significance_tests": sig_tests,
        "equivalence_tests": equivalence_tests,
        "runtime_s": time.time() - start_time,
    }
    if with_f:
        output["arm_f_overlap"] = f_extra

    # Save results
    print(f"\nSaving results to {RESULTS_FILE}")
    RESULTS_FILE.write_text(json.dumps(output, indent=2, sort_keys=True))

    # Save cache
    print(f"Saving cache to {CACHE_FILE}")
    CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))

    # Print summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nSessions: {len(sessions)}")
    print(f"Probes: {len(probes)}")
    print(f"\nFairness:")
    print(f"  Transcript overlap: {mean_fairness_transcript:.3f}")
    print(f"  Rollup overlap: {mean_fairness_rollup:.3f}")
    print(f"  Difference: {fairness_diff:.3f} {'[WARNING]' if fairness_diff > 0.15 else ''}")

    print(f"\nRecall@10 by arm:")
    for arm in ["A_rollup", "A_rollup_summary", "B_raw", "C_raw_no_tool", "D_bm25_rollup", "E_bm25_raw", *(["F_bm25_no_tool"] if with_f else [])]:
        r10 = results[arm]["recall@10"]
        print(f"  {arm:20} {r10:.3f}")

    print(f"\nA_rollup vs C_raw_no_tool:")
    if "A_rollup_vs_C_raw_no_tool" in sig_tests:
        test = sig_tests["A_rollup_vs_C_raw_no_tool"]
        print(f"  McNemar p-value: {test['p_value']:.4f}")
        print(f"  Effect size: {test['effect_size']:.3f}")

    print(f"\nRuntime: {output['runtime_s']:.1f}s")

    # Compute hash for determinism check
    result_bytes = RESULTS_FILE.read_bytes()
    result_hash = hashlib.sha256(result_bytes).hexdigest()
    print(f"\nResult SHA256: {result_hash[:16]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
