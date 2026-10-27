#!/usr/bin/env python3
"""Cost estimates for four candidate RQ3 arms.

Estimates cost and wall time for:
(a) action-record arm — the new format
(b) tool-inputs-only arm — isolates the active ingredient
(c) prose-only arm — control showing tool inputs are necessary
(d) signature-free raw arm — may recover more of the 7 over-window sessions

All arms answer 124 questions with one answer call + one grading call each,
using claude-sonnet-5 via the local gateway (no temperature parameter).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_action_record_sizes import SESSIONS, load_baseline_sizes, index_transcripts, resolve

# Model pricing from claude-sonnet-5 (as of 2026-09-30)
# These are enterprise API gateway prices which may differ from public list prices
INPUT_PRICE_PER_1M = 3.00  # $3 per 1M input tokens
OUTPUT_PRICE_PER_1M = 15.00  # $15 per 1M output tokens

BYTES_PER_TOKEN = 2.45  # Measured from raw transcript arm
N_QUESTIONS = 124
AVG_ANSWER_TOKENS = 150  # Observed from existing arms
AVG_GRADING_TOKENS = 120  # Observed from existing arms


def estimate_arm(name: str, median_chars: int, n_questions: int = N_QUESTIONS) -> dict:
    """Estimate cost and time for an arm.

    Each question requires:
    1. Answer call: context + answer completion
    2. Grading call: context + previous answer + grading completion
    """
    median_tokens = median_chars / BYTES_PER_TOKEN

    # Answer calls
    answer_prompt_tokens = n_questions * median_tokens
    answer_completion_tokens = n_questions * AVG_ANSWER_TOKENS
    answer_total = answer_prompt_tokens + answer_completion_tokens

    # Grading calls (includes the answer in context)
    grading_prompt_tokens = n_questions * (median_tokens + AVG_ANSWER_TOKENS)
    grading_completion_tokens = n_questions * AVG_GRADING_TOKENS
    grading_total = grading_prompt_tokens + grading_completion_tokens

    total_input_tokens = answer_prompt_tokens + grading_prompt_tokens
    total_output_tokens = answer_completion_tokens + grading_completion_tokens
    total_tokens = total_input_tokens + total_output_tokens

    # Cost
    input_cost = (total_input_tokens / 1e6) * INPUT_PRICE_PER_1M
    output_cost = (total_output_tokens / 1e6) * OUTPUT_PRICE_PER_1M
    total_cost = input_cost + output_cost

    # Wall time estimate (assuming ~30 tokens/sec output, serial execution)
    # Answer calls: prompt processing (~instant with cache) + generation
    # Grading calls: prompt processing + generation
    # Total: 2 calls per question, dominated by generation time
    output_time_sec = total_output_tokens / 30  # 30 tokens/sec generation
    # Add overhead for prompt processing (say 2s per call on average)
    overhead_sec = n_questions * 2 * 2  # 2 calls per question, 2s each
    total_time_sec = output_time_sec + overhead_sec
    total_time_min = total_time_sec / 60

    return {
        "name": name,
        "median_chars": median_chars,
        "median_tokens": int(median_tokens),
        "n_questions": n_questions,
        "total_input_tokens": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "total_tokens": int(total_tokens),
        "input_cost_usd": round(input_cost, 2),
        "output_cost_usd": round(output_cost, 2),
        "total_cost_usd": round(total_cost, 2),
        "wall_time_min": round(total_time_min, 1),
        "cost_per_question_cents": round(total_cost * 100 / n_questions, 2),
    }


def main():
    # Load measured sizes
    by_prefix = index_transcripts()
    verbatim_sizes, stripped_sizes = load_baseline_sizes()

    # Calculate action record median
    action_sizes = []
    for prefix in SESSIONS:
        path = resolve(prefix, by_prefix)
        if path:
            from action_record import build_action_record
            try:
                record, _ = build_action_record(str(path))
                action_sizes.append(len(record))
            except Exception:
                pass
    action_sizes.sort()
    action_median = action_sizes[len(action_sizes) // 2] if action_sizes else 23905

    # Calculate stripped median (for reference)
    stripped_vals = [v for v in stripped_sizes.values() if v > 0]
    stripped_vals.sort()
    stripped_median = stripped_vals[len(stripped_vals) // 2] if stripped_vals else 33443

    # Verbatim median (from summary)
    verbatim_median = 14379

    # (a) Action-record arm
    action = estimate_arm("(a) action-record", action_median)

    # (b) Tool-inputs-only arm: ~46.9% of stripped
    tool_inputs_only_chars = int(stripped_median * 0.469)
    tool_inputs = estimate_arm("(b) tool-inputs-only", tool_inputs_only_chars)

    # (c) Prose-only arm: stripped minus tool inputs, ~12.4 KB
    # From spec: user prose 12.3% + assistant prose 24.9% = 37.2% of stripped
    prose_only_chars = int(stripped_median * 0.372)
    prose = estimate_arm("(c) prose-only", prose_only_chars)

    # (d) Signature-free raw arm: this is expensive
    # Median raw size for sessions that fit: estimate ~700k chars based on raw arm data
    # But this only covers 89 of 124 questions (7 sessions overflow even without signatures)
    # Conservatively estimate: median raw 700k chars, signatures save ~50k per session
    # So signature-free median ~650k chars
    raw_median_chars = 650000
    signature_free = estimate_arm("(d) signature-free-raw", raw_median_chars, n_questions=89)
    # Note: this arm would attempt all 124 but may still overflow some, so report as 89-124 range

    # Print results
    arms = [action, tool_inputs, prose, signature_free]

    print("\nCost estimates for four candidate RQ3 arms")
    print("=" * 100)
    print(f"Model: claude-sonnet-5 via local gateway (no temperature)")
    print(f"Pricing: ${INPUT_PRICE_PER_1M:.2f}/1M input, ${OUTPUT_PRICE_PER_1M:.2f}/1M output")
    print(f"Questions: {N_QUESTIONS} (except signature-free-raw: 89-124)")
    print("=" * 100)
    print()

    print(f"{'Arm':<25s} {'Median':>10s} {'Total':>12s} {'Cost':>10s} {'Time':>10s} {'$/Q':>10s}")
    print(f"{'':25s} {'chars':>10s} {'tokens':>12s} {'(USD)':>10s} {'(min)':>10s} {'(cents)':>10s}")
    print("-" * 100)

    for a in arms:
        print(f"{a['name']:<25s} {a['median_chars']:>10,} {a['total_tokens']:>12,} "
              f"${a['total_cost_usd']:>9.2f} {a['wall_time_min']:>10.1f} {a['cost_per_question_cents']:>10.2f}")

    print()
    print("Breakdown by arm:")
    print("-" * 100)

    for a in arms:
        print(f"\n{a['name']}")
        print(f"  Median context: {a['median_chars']:,} chars = {a['median_tokens']:,} tokens")
        print(f"  Input tokens: {a['total_input_tokens']:,} (${a['input_cost_usd']:.2f})")
        print(f"  Output tokens: {a['total_output_tokens']:,} (${a['output_cost_usd']:.2f})")
        print(f"  Total: {a['total_tokens']:,} tokens, ${a['total_cost_usd']:.2f}, {a['wall_time_min']:.1f} min")
        print(f"  Cost per question: {a['cost_per_question_cents']:.2f} cents")

    print()
    print("=" * 100)
    print("Information gained per dollar (ranked):")
    print("=" * 100)
    print()

    # Rank by cost, then add interpretation
    sorted_arms = sorted(arms, key=lambda x: x["total_cost_usd"])

    for rank, a in enumerate(sorted_arms, 1):
        cost_ratio = a["total_cost_usd"] / sorted_arms[0]["total_cost_usd"]
        print(f"{rank}. {a['name']} — ${a['total_cost_usd']:.2f} ({cost_ratio:.1f}x cheapest)")

        if a["name"] == "(a) action-record":
            print(f"   Novel format. Expected to match or exceed stripped's 90.3% recovery")
            print(f"   in {a['median_tokens']:,} tokens (0.71x stripped). Main experiment.")

        elif a["name"] == "(b) tool-inputs-only":
            print(f"   Isolates the active ingredient. Tool inputs are 46.9% of stripped")
            print(f"   and distinguish it from verbatim (71.8%). Pure test of the hypothesis.")

        elif a["name"] == "(c) prose-only":
            print(f"   Control. Prose alone is ~12.4 KB and verbatim (pure prose) scores")
            print(f"   71.8%. Should land near verbatim, confirming tool inputs matter.")

        elif a["name"] == "(d) signature-free-raw":
            print(f"   May recover the 7 over-window sessions raw arm couldn't fit.")
            print(f"   Signatures are ~1,424 chars × 2,193 blocks = 3.1 MB saved per corpus.")
            print(f"   Expensive ({a['median_tokens']:,} tokens/question) but high ceiling.")
            print(f"   Coverage: 89 certain, up to 124 if signatures were the only blocker.")

        print()

    print("=" * 100)
    print("Recommended run order:")
    print("=" * 100)
    print()
    print("1. (c) prose-only — $", prose["total_cost_usd"], "— cheapest, validates the control")
    print("   If it lands near verbatim's 71.8%, tool inputs are proven necessary.")
    print()
    print("2. (b) tool-inputs-only — $", tool_inputs["total_cost_usd"], "— isolates the ingredient")
    print("   If it matches stripped's 90.3%, prose adds nothing. If it falls short,")
    print("   we know both are needed. Either way, this brackets the action record.")
    print()
    print("3. (a) action-record — $", action["total_cost_usd"], "— the main arm")
    print("   Should match or beat stripped (90.3%) at 0.71x the cost. The paper's")
    print("   contribution: a format that retains tool context + result digests + ordering")
    print("   in a durable, readable form smaller than stripped.")
    print()
    print("4. (d) signature-free-raw — $", signature_free["total_cost_usd"], "— expensive, high ceiling")
    print("   Run only if recovering the 7 over-window sessions is worth 20x the cost.")
    print("   Signatures are provably zero-information, so stripping them is honest.")
    print("   But at ~650k tokens/question, this is a budget question, not a science one.")
    print()


if __name__ == "__main__":
    main()
