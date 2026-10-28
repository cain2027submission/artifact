"""Canonical naming for RQ3 context artifact arms.

Maps internal code keys to paper-facing display names. One source of truth
for all human-facing labels, documentation, and reports.
"""

# Code key -> paper display name
ARM_NAMES = {
    "raw": "raw transcript",
    "stripped": "output-stripped transcript",
    "verbatim": "dialogue record",
    "tool_inputs_only": "call-only",
    "prose_only": "prose-only",
    "action_record": "action record",
    "rollup_hookshape": "rollup summary",
    "rollup_pointer": "rollup pointer",
    "stripped_chunk": "output-stripped chunk",
}

# One-line construction summary per arm
ARM_CONSTRUCTION = {
    "raw": "the .jsonl, everything including JSON scaffolding and base64 thinking signatures",
    "stripped": "strip_transcript() output: user/assistant prose, non-empty reasoning under ### REASONING, tool calls under ### TOOL CALL with inputs capped at 2000 chars; drops all tool_result content, empty thinking blocks, JSON scaffolding",
    "verbatim": "rollup_verbatim.py output: human prompts, assistant's last prose answer per exchange, tool-call bullet index truncated to 120 chars; drops all reasoning and tool results",
    "tool_inputs_only": "tool_use inputs only, zero prose, capped at 2000 chars per call (negative control)",
    "prose_only": "output-stripped minus tool calls",
    "action_record": "like output-stripped but adds one-line tool-result digests, structures salient args, adds turn numbers; drops signature",
    "rollup_hookshape": "rollup note FTS section, max 160 char snippet",
    "rollup_pointer": "rollup summary plus body up to 500 chars",
    "stripped_chunk": "keyword-matched chunk of output-stripped transcript, budget-matched to rollup summary length",
}


def paper_name(key):
    """Return paper display name for a code key. Raises KeyError if unknown."""
    if key not in ARM_NAMES:
        raise KeyError(f"Unknown arm key: {key!r}. Valid keys: {list(ARM_NAMES.keys())}")
    return ARM_NAMES[key]
