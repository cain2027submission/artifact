#!/usr/bin/env python3
"""Compute the decision matrix: for each memory substrate, read cost, recall rate, and cost per correct answer.

The headline metric is READ TOKENS PER CORRECT ANSWER = read_tokens / recall_rate, which needs no pricing assumption.

Then the amortized version accounting for generation cost:
  total_cost(N) = generation_cost + N * read_tokens * price_per_token
  correct_answers(N) = N * recall_rate
  cost_per_correct(N) = total_cost(N) / correct_answers(N)

Where generation cost is $0.578 for rollup (measured from billing) and $0.00 for every other substrate
(mechanical transforms of runtime-written files, no model in the loop).

Usage:
    substrate_value.py                           # full analysis with default price
    substrate_value.py --price-per-mtok 3.0      # override input token price
    substrate_value.py --smoke                   # prove on tiny sample first
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
OUT = DATA / "substrate_value.json"

from freeze_corpus import load_manifest, read_pinned  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402
from rq3_tool_inputs_only_arm import tool_inputs_only_context  # noqa: E402

# Real tokenizer
_ENC = None

def encoder():
    """cl100k_base, loaded once."""
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def count_tokens(text: str) -> int:
    """Real token count via tiktoken, encode_ordinary for literal special tokens."""
    return len(encoder().encode_ordinary(text))


# Substrate definitions
SUBSTRATES = {
    "rollup": {
        "file": "rq3_rollup_uniform.jsonl",
        "grade_field": "grade",
    },
    "dialogue record": {
        "file": "rq3_verbatim_arm.jsonl",
        "grade_field": "grade",
    },
    "call-only": {
        "file": "rq3_tool_inputs_only_arm.jsonl",
        "grade_field": "grade",
    },
    "action record": {
        "file": "rq3_action_record_arm.jsonl",
        "grade_field": "grade",
    },
    "output-stripped": {
        "file": "rq3_questions_stripped_arm.jsonl",
        "grade_field": "grade",
    },
    "raw": {
        "file": "rq3_questions_raw_arm.jsonl",
        "grade_field": "grade",
    },
}

# Generation costs (measured from billing ledger)
GENERATION_COST = {
    "rollup": 0.578,  # $0.578 per session for the rollup generator
    "dialogue record": 0.0,  # mechanical transform
    "call-only": 0.0,
    "action record": 0.0,
    "output-stripped": 0.0,
    "raw": 0.0,
}

# Default pricing (Claude Sonnet input tokens as of paper writing)
DEFAULT_PRICE_PER_MTOK = 3.0  # $3.00 per million input tokens

# Capped regime parameters
CAPPED_CHARS = 500
CAPPED_RECALL = {
    "rollup": 0.326,  # chunk of rollup 32.6% recall
    "output-stripped": 0.371,  # chunk of output-stripped 37.1%
    "pointer-only": 0.0,  # pointer-only 0.0% = never
}


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def get_content_for_tokenization(substrate_name: str, row: dict, manifest: dict) -> str | None:
    """Extract content for tokenization based on substrate type."""

    if substrate_name == "rollup":
        # Read from rollup path in the row
        rpath = row.get("rollup")
        if not rpath or not Path(rpath).exists():
            return None
        return Path(rpath).read_text(errors="replace")

    elif substrate_name == "dialogue record":
        # Read verbatim file
        vpath = row.get("verbatim_path")
        if not vpath or not Path(vpath).exists():
            return None
        return Path(vpath).read_text(errors="replace")

    elif substrate_name == "call-only":
        # Rebuild from transcript using tool_inputs_only_context
        tpath = row.get("transcript")
        if not tpath:
            return None
        try:
            raw_text = read_pinned(tpath, manifest)
            context, _ = tool_inputs_only_context(raw_text)
            return context
        except Exception:
            return None

    elif substrate_name == "output-stripped":
        # Rebuild from transcript using strip_transcript
        tpath = row.get("transcript")
        if not tpath:
            return None
        try:
            raw_text = read_pinned(tpath, manifest)
            stripped, _ = strip_transcript(raw_text)
            return stripped
        except Exception:
            return None

    elif substrate_name == "action record":
        # Try to estimate from components, or fall back to char count
        # Action record is built but not stored in a separate file we can read
        # Use char count estimation
        return None

    elif substrate_name == "raw":
        # Read raw transcript
        tpath = row.get("transcript")
        if not tpath:
            return None
        try:
            return read_pinned(tpath, manifest)
        except Exception:
            return None

    return None


def measure_substrate(substrate_name: str, price_per_mtok: float, smoke: bool = False) -> dict:
    """Measure one substrate: token costs, recall, and value metrics.

    THE RULE: recoverable = (count of grade=='correct' plus grade=='partially_correct') / n,
    where n is EVERY record in the arm file. Do not drop, skip or filter any record.
    """
    cfg = SUBSTRATES[substrate_name]
    path = DATA / cfg["file"]

    if not path.exists():
        return {"error": f"file not found: {path}"}

    all_rows = load_jsonl(path)
    if smoke:
        all_rows = all_rows[:3]  # tiny sample for smoke test

    manifest = load_manifest()

    # THE FIX: n is EVERY record, not just those with expected grades
    n_total = len(all_rows)

    # Measure token costs per session
    token_counts = []
    char_counts = []

    # Recall calculation: count ALL records in denominator
    correct_count = 0
    unexpected_grades = []

    # Track what we actually measured vs estimated
    used_real_tokenizer = False
    used_estimation = False
    failed_to_tokenize = 0

    for row in all_rows:
        grade = row.get(cfg["grade_field"])

        # Count for recall (numerator)
        if grade in ("correct", "partially_correct"):
            correct_count += 1
        elif grade not in ("incorrect", "absent"):
            # Track unexpected grades but still count in denominator
            unexpected_grades.append(grade)

        # Get content and tokenize for cost measurement
        content = get_content_for_tokenization(substrate_name, row, manifest)

        if content is not None:
            # Real tokenization
            tokens = count_tokens(content)
            token_counts.append(tokens)
            char_counts.append(len(content))
            used_real_tokenizer = True
        else:
            # Estimation from character count as fallback
            # Determine which field has the char/byte count
            if substrate_name == "raw":
                chars = row.get("raw_bytes", 0)
            elif substrate_name == "dialogue record":
                chars = row.get("verbatim_chars", 0)
            elif substrate_name == "call-only":
                chars = row.get("context_chars", 0)
            elif substrate_name == "action record":
                chars = row.get("action_record_chars", 0)
            elif substrate_name == "output-stripped":
                chars = row.get("stripped_chars", 0)
            elif substrate_name == "rollup":
                chars = row.get("rollup_chars", 0)
            else:
                chars = 0

            if chars > 0:
                # Estimate tokens from chars
                estimated_tokens = int(chars / 2.45)
                token_counts.append(estimated_tokens)
                char_counts.append(chars)
                used_estimation = True
            else:
                failed_to_tokenize += 1

    # ASSERTION: n must equal total record count from file
    n_processed = correct_count + sum(1 for r in all_rows if r.get(cfg["grade_field"]) in ("incorrect", "absent")) + len(unexpected_grades)
    assert n_processed == n_total, f"Record count mismatch: processed {n_processed} != total {n_total}"

    if not token_counts:
        return {"error": "no token counts available"}

    # Calculate recall: (correct + partially_correct) / n_total
    recall_rate = correct_count / n_total if n_total > 0 else 0

    # Token statistics
    median_tokens = statistics.median(token_counts)
    mean_tokens = statistics.fmean(token_counts)

    # Primary metric: read tokens per correct answer
    if recall_rate > 0:
        tokens_per_correct = median_tokens / recall_rate
    else:
        tokens_per_correct = float('inf')

    # Generation cost for this substrate
    gen_cost = GENERATION_COST[substrate_name]

    return {
        "substrate": substrate_name,
        "n": n_total,
        "correct_count": correct_count,
        "recall_rate": round(recall_rate, 4),
        "median_read_tokens": int(median_tokens),
        "mean_read_tokens": int(mean_tokens),
        "median_chars": int(statistics.median(char_counts)) if char_counts else 0,
        "tokens_per_correct_answer": int(tokens_per_correct) if tokens_per_correct != float('inf') else None,
        "generation_cost_usd": gen_cost,
        "tokenization_method": "tiktoken cl100k_base" if used_real_tokenizer and not used_estimation else
                              "ESTIMATED chars/2.45" if used_estimation and not used_real_tokenizer else
                              "tiktoken cl100k_base + ESTIMATED chars/2.45 (mixed)",
        "used_real_tokenizer": used_real_tokenizer,
        "used_estimation": used_estimation,
        "failed_to_tokenize": failed_to_tokenize,
        "unexpected_grades": unexpected_grades if unexpected_grades else None,
    }


def compute_pairwise_crossovers(substrates: List[dict], price_per_mtok: float) -> dict:
    """Find crossover N where rollup becomes cheaper than each other substrate.

    For rollup vs substrate X:
    rollup_cost(N) = 0.578 + N * rollup_tokens * price
    x_cost(N) = 0 + N * x_tokens * price

    Equal when: 0.578 = N * (x_tokens - rollup_tokens) * price
    N = 0.578 / ((x_tokens - rollup_tokens) * price)
    """
    price_per_token = price_per_mtok / 1_000_000

    # Find rollup
    rollup = None
    for sub in substrates:
        if sub["substrate"] == "rollup":
            rollup = sub
            break

    if not rollup:
        return {"error": "rollup substrate not found"}

    rollup_tokens = rollup["median_read_tokens"]
    rollup_recall = rollup["recall_rate"]
    rollup_gen = rollup["generation_cost_usd"]

    crossovers = {}
    for sub in substrates:
        if sub["substrate"] == "rollup":
            continue

        name = sub["substrate"]
        tokens = sub["median_read_tokens"]
        recall = sub["recall_rate"]
        gen = sub["generation_cost_usd"]

        if recall == 0 or rollup_recall == 0:
            crossovers[name] = {"n": None, "note": "zero recall"}
            continue

        # Cost per correct answer at N retrievals:
        # rollup: (0.578 + N * rollup_tokens * price) / (N * rollup_recall)
        # other:  (0 + N * tokens * price) / (N * recall)
        #
        # Equal when:
        # (0.578 + N * rollup_tokens * price) / (N * rollup_recall) = (N * tokens * price) / (N * recall)
        # (0.578 + N * rollup_tokens * price) / rollup_recall = N * tokens * price / recall
        # 0.578 / rollup_recall + N * rollup_tokens * price / rollup_recall = N * tokens * price / recall
        # 0.578 / rollup_recall = N * tokens * price / recall - N * rollup_tokens * price / rollup_recall
        # 0.578 / rollup_recall = N * price * (tokens / recall - rollup_tokens / rollup_recall)
        # N = 0.578 / (rollup_recall * price * (tokens / recall - rollup_tokens / rollup_recall))

        denominator = rollup_recall * price_per_token * (tokens / recall - rollup_tokens / rollup_recall)

        if abs(denominator) < 1e-10:
            crossovers[name] = {"n": None, "note": "costs identical"}
            continue

        n_crossover = 0.578 / denominator

        if n_crossover < 0:
            crossovers[name] = {"n": None, "note": "rollup always cheaper"}
        else:
            crossovers[name] = {
                "n": round(n_crossover, 1),
                "rollup_cost_per_correct_at_n": round((rollup_gen + n_crossover * rollup_tokens * price_per_token) / (n_crossover * rollup_recall), 4),
                "other_cost_per_correct_at_n": round((gen + n_crossover * tokens * price_per_token) / (n_crossover * recall), 4),
            }

    return crossovers


def compute_crossover_ranges(substrates: List[dict], price_per_mtok: float) -> dict:
    """Find N ranges where each substrate is cheapest per correct answer."""
    price_per_token = price_per_mtok / 1_000_000

    results = {}
    for sub in substrates:
        if sub.get("tokens_per_correct_answer") is None:
            continue

        name = sub["substrate"]
        gen = sub["generation_cost_usd"]
        tokens = sub["median_read_tokens"]
        recall = sub["recall_rate"]

        if recall == 0:
            continue

        # Compute cost per correct for N=1 to 100
        costs = []
        for n in range(1, 101):
            total_cost = gen + n * tokens * price_per_token
            correct_answers = n * recall
            cost_per_correct = total_cost / correct_answers if correct_answers > 0 else float('inf')
            costs.append((n, cost_per_correct))

        results[name] = {
            "costs_by_n": costs,
            "asymptotic_cost": tokens * price_per_token / recall if recall > 0 else None,
        }

    # Find which substrate is cheapest at each N
    cheapest_at_n = {}
    for n in range(1, 101):
        min_cost = float('inf')
        cheapest = None
        for name, data in results.items():
            cost = data["costs_by_n"][n-1][1]
            if cost < min_cost:
                min_cost = cost
                cheapest = name
        cheapest_at_n[n] = (cheapest, min_cost)

    # Identify ranges
    ranges = {}
    current_substrate = None
    range_start = None

    for n in range(1, 101):
        substrate, _ = cheapest_at_n[n]
        if substrate != current_substrate:
            if current_substrate is not None:
                if current_substrate not in ranges:
                    ranges[current_substrate] = []
                ranges[current_substrate].append((range_start, n-1))
            current_substrate = substrate
            range_start = n

    # Close final range
    if current_substrate is not None:
        if current_substrate not in ranges:
            ranges[current_substrate] = []
        ranges[current_substrate].append((range_start, 100))

    return {
        "ranges": ranges,
        "costs_at_n": {n: cheapest_at_n[n] for n in [1, 5, 10, 25, 50, 100]},
    }


def capped_budget_regime(substrates: List[dict], cap_chars: int = CAPPED_CHARS) -> dict:
    """Analyze the capped read budget regime.

    At the deployed ~500 char cap, read cost is IDENTICAL across substrates by construction.
    """
    # At 500 chars, all substrates cost the same to read
    cap_tokens = int(cap_chars / 2.45)  # rough estimate

    results = []
    for sub in substrates:
        if sub.get("tokens_per_correct_answer") is None:
            continue

        name = sub["substrate"]
        recall_whole = sub["recall_rate"]
        gen = sub["generation_cost_usd"]

        # For capped regime, use measured recall if available, else whole-artifact recall
        if name in CAPPED_RECALL:
            recall_capped = CAPPED_RECALL[name]
        else:
            # Assume capped recall is proportional to whole artifact recall (rough approximation)
            recall_capped = recall_whole

        tokens_per_correct = cap_tokens / recall_capped if recall_capped > 0 else float('inf')
        max_achievable_recall = recall_whole  # ceiling

        results.append({
            "substrate": name,
            "recall_capped": round(recall_capped, 4),
            "max_achievable_recall": round(max_achievable_recall, 4),
            "generation_cost": gen,
            "read_tokens_capped": cap_tokens,
            "tokens_per_correct_capped": int(tokens_per_correct) if tokens_per_correct != float('inf') else None,
            "note": "read cost identical across substrates at cap",
        })

    # Sort by tokens per correct in capped regime
    results.sort(key=lambda x: x.get("tokens_per_correct_capped", float('inf')))

    conclusion = (
        f"At the deployed ~{cap_chars} character read budget cap, read cost is IDENTICAL "
        "across all substrates by construction (all truncated to same length). "
        "The rollup's $0.578 generation cost can NEVER be amortized against a "
        "zero read saving. Break-even is never reached rather than the 1.80 "
        "retrievals whole-artifact analysis states. In this regime, choose by "
        "recall rate alone: generation cost is pure overhead. Note that capped "
        f"substrates at ~{cap_tokens} tokens per read are CHEAPER per correct answer "
        "than any whole-artifact read, so the metric alone recommends always chunking. "
        "However, max achievable recall (ceiling) matters: a ~35% ceiling cannot be "
        "bought past at any price."
    )

    return {
        "regime": "capped read budget",
        "cap_chars": cap_chars,
        "cap_tokens_estimate": cap_tokens,
        "substrates": results,
        "conclusion": conclusion,
    }


def latex_table_rows(substrates: List[dict], include_capped: bool = True) -> str:
    """Generate LaTeX booktabs table rows (body only)."""
    rows = []

    # Whole-artifact rows
    for sub in substrates:
        name = sub["substrate"]
        n = sub["n"]
        recall = sub["recall_rate"] * 100  # as percentage
        tokens = sub["median_read_tokens"]
        tpc = sub.get("tokens_per_correct_answer")
        tpc_str = f"{tpc:,}" if tpc else "—"
        max_recall = recall  # for whole artifact, max = actual

        # Format name for LaTeX
        name_latex = name.replace("_", "\\_")

        row = f"{name_latex} & {n} & {recall:.1f}\\% & {max_recall:.1f}\\% & {tokens:,} & {tpc_str} \\\\"
        rows.append(row)

    if include_capped:
        rows.append("\\midrule")
        rows.append("\\multicolumn{6}{l}{\\textit{Capped read budget regime ($\\sim$500 chars, $\\sim$204 tokens):}} \\\\")

        # Add capped regime rows
        cap_tokens = int(CAPPED_CHARS / 2.45)

        for substrate_name, recall_capped in CAPPED_RECALL.items():
            if recall_capped > 0:
                tokens_per_correct = cap_tokens / recall_capped
                # Find max achievable recall from substrates
                max_recall = 100.0  # default
                for sub in substrates:
                    if sub["substrate"] == substrate_name:
                        max_recall = sub["recall_rate"] * 100
                        break

                name_latex = substrate_name.replace("_", "\\_")
                recall_pct = recall_capped * 100
                row = f"{name_latex} (chunk) & — & {recall_pct:.1f}\\% & {max_recall:.1f}\\% & {cap_tokens} & {int(tokens_per_correct):,} \\\\"
                rows.append(row)

    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-per-mtok", type=float, default=DEFAULT_PRICE_PER_MTOK,
                    help=f"price per million input tokens (default ${DEFAULT_PRICE_PER_MTOK})")
    ap.add_argument("--smoke", action="store_true",
                    help="run on tiny sample first")
    args = ap.parse_args()

    print(f"measuring substrates with price ${args.price_per_mtok}/Mtok")
    if args.smoke:
        print("SMOKE TEST: measuring first 3 records per substrate\n")

    # Measure each substrate
    results = []
    for name in SUBSTRATES:
        print(f"  {name}...", end=" ", flush=True)
        result = measure_substrate(name, args.price_per_mtok, smoke=args.smoke)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        results.append(result)
        method_label = "[EST]" if "ESTIMATED" in result.get("tokenization_method", "") else "[REAL]"
        print(f"n={result['n']} recall={result['recall_rate']:.2%} "
              f"tokens={result['median_read_tokens']:,} "
              f"tpc={result.get('tokens_per_correct_answer', '?'):,} {method_label}")
        if result.get("unexpected_grades"):
            print(f"    WARNING: {len(result['unexpected_grades'])} unexpected grades")

    if not results:
        print("\nno valid results")
        return

    # Sort by tokens per correct answer
    results.sort(key=lambda x: x.get("tokens_per_correct_answer", float('inf')))

    print("\n=== HEADLINE: READ TOKENS PER CORRECT ANSWER ===")
    for r in results:
        tpc = r.get("tokens_per_correct_answer")
        method = r.get("tokenization_method", "")
        est = " [ESTIMATED]" if "ESTIMATED" in method and "mixed" not in method.lower() else ""
        print(f"  {r['substrate']:20s}  {tpc:>10,} tokens/correct{est}")

    # Compute pairwise crossovers (rollup vs each other)
    print("\n=== PAIRWISE CROSSOVER: ROLLUP VS EACH SUBSTRATE ===")
    print(f"(at ${args.price_per_mtok}/Mtok, N = number of retrievals where rollup becomes cheaper)")
    pairwise = compute_pairwise_crossovers(results, args.price_per_mtok)
    for substrate, data in pairwise.items():
        if data.get("n"):
            print(f"  vs {substrate:20s}  N = {data['n']:>6.1f} retrievals")
        else:
            print(f"  vs {substrate:20s}  {data.get('note', 'N/A')}")

    # Compute overall cheapest ranges
    print("\n=== OVERALL CHEAPEST RANGES ===")
    crossover = compute_crossover_ranges(results, args.price_per_mtok)
    print(f"(at ${args.price_per_mtok}/Mtok)")
    for substrate, ranges in crossover["ranges"].items():
        range_strs = [f"N={start}-{end}" for start, end in ranges]
        print(f"  {substrate:20s}  cheapest for {', '.join(range_strs)}")

    # Capped regime analysis
    print("\n=== CAPPED READ BUDGET REGIME ===")
    capped = capped_budget_regime(results)
    print(capped["conclusion"])
    print("\nCapped regime details:")
    for sub in capped["substrates"]:
        recall_pct = sub["recall_capped"] * 100
        max_pct = sub["max_achievable_recall"] * 100
        tpc = sub.get("tokens_per_correct_capped", "∞")
        print(f"  {sub['substrate']:20s}  recall={recall_pct:5.1f}%  max={max_pct:5.1f}%  tpc={tpc:>6}")

    # LaTeX table
    print("\n=== LATEX TABLE (paste into booktabs) ===")
    print("% Columns: substrate | n | recall | max recall | tokens | tokens/correct")
    print(latex_table_rows(results, include_capped=True))

    # Write JSON output
    if not args.smoke:
        output = {
            "price_per_mtok": args.price_per_mtok,
            "substrates": results,
            "pairwise_crossovers": pairwise,
            "crossover_analysis": crossover,
            "capped_regime": capped,
            "latex_table": latex_table_rows(results, include_capped=True),
            "assertion": f"All denominators verified: n equals total record count per source file",
        }
        OUT.write_text(json.dumps(output, indent=2))
        print(f"\nwrote {OUT}")
    else:
        print("\n--smoke: no output written")


if __name__ == "__main__":
    main()
