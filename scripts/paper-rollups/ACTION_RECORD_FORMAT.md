# action record format specification (corrected 2026-09-30)

## motivation

the output-stripped transcript arm (rq3) keeps tool call inputs capped at 2000 chars and drops tool results entirely. this recovers 94.4% of ground truth (uncorrected) or 90.3% (confidence-corrected) in a median 33,443 chars, against the dialogue record's 71.8% in 14,379 chars. tool inputs are the active ingredient (46.9% of output-stripped's volume), but the blind 2000-char cap treats a one-line bash command and a 5000-line Write payload identically. tool results, dropped entirely, are the overwhelming majority of raw transcript bytes and plausibly hold ground truth for `artifact_detail` and `outcome` question classes.

the action record is the missing middle: structured tool calls with salient arguments extracted, plus one-line result digests that convey success/failure and scale without the full payload.

**framing — critical**: the action record compresses differently depending on baseline:
- **vs RAW TRANSCRIPT**: large compression. result digests replace full result bodies (21.68% of raw as content: 11,713,000 of 54,037,811 chars measured from `toolUseResult` fields plus `tool_result` content blocks). signatures dropped (8.71% of raw, measured 4,708,852 of 54,037,811 chars). measured action:raw ratio 0.034x (23,905 / 700,000).
- **vs OUTPUT-STRIPPED TRANSCRIPT**: enrichment, not compression. adds result digests (+2k-4k chars) and turn ordering (+100 chars). saves bytes via structured tool calls (smarter than output-stripped's blind 2000-char cap), NOT via dropping signatures (already 0.00% of output-stripped because `strip_transcript()` skips empty thinking blocks) or boilerplate (measured 3.04% ceiling). measured action:output-stripped ratio 0.71x (23,905 / 33,443), so action IS smaller, but the claim is higher recovery at modestly different cost, not same recovery for fewer bytes.

## measured justifications

all measurements from rq3 analysis over 25 graded sessions, 124 questions:

| format | median chars | recoverable % | notes |
|--------|-------------|---------------|-------|
| raw transcript | ~700k | 95.5 (uncorrected) / 94.4 (corrected) | 89 of 124 questions; 35 dropped overflow context window and are systematically harder (+16.0pp to +26.3pp worse across other arms) |
| output-stripped transcript | 33,443 | 90.3 | tool inputs capped 2000 chars, results dropped |
| **action record** | **23,905** | **UNRUN** | **structured tool calls, result digests, turn ordering** |
| dialogue record | 14,379 | 71.8 | turn structure, prose, tool calls inline, no results |
| rollup summary | ~3k | 68.5 | generated summary |

composition of output-stripped transcript (median 33,443 chars):
- tool call inputs: 46.9%
- assistant prose: 24.9%
- user prose: 12.3%
- thinking text: 0.0%
- thinking signatures: 0.0% (output-stripped already skips them via line-121 continue)
- other: ~16% (under investigation, treat as unverified)

**measured corrections 2026-09-30**:
- **thinking block signatures**: 8.71% of raw transcript (4,708,852 of 54,037,811 chars, cross-checked to 0.00pp, up to 13.53% in largest sessions). BUT 0.00% of output-stripped because `strip_transcript()` already skips empty thinking blocks (line 121 `continue`). removing them rescues 1 of 5 over-window sessions vs raw. NO saving vs output-stripped.
- **harness boilerplate**: measured at 3.04% of output-stripped artifact (per-session range 0.00% to 14.33%). harness-injected blocks like `<system-reminder>` are already 0.00% because `split_meta()` removes them before rendering. true repeated chunks of 100+ chars are the remaining 3.04%. NOT a significant share.

thinking blocks: 3,659 total across n=12 sessions, 3,659 with empty text (99.8%), all carrying a base64 `signature` averaging ~1,287 chars (see the warning under [what the action record drops](#thinking-block-signatures) before reusing the ~2,125 median that used to be quoted here). zero information. **the claim that the output-stripped artifact is "60.7% thinking signatures / 17.8% tool inputs" is WRONG** — it was measured on raw-JSONL-minus-tool_result, a different object.

worst-recovered question class: `sequence` at 37.5% across arms, suggesting ordering information is load-bearing.

## what the action record keeps

### user prose
harness-injected `<local-command-caveat>` blocks and similar meta-commentary stripped using `rq1_signal_composition_corrected.split_meta()`. keeps only authored user text.

**justification**: user prose is 12.3% of output-stripped and is the intent. no measured benefit to dropping it.

**byte effect**: neutral. already minimal.

### assistant prose
all assistant text blocks, in full.

**justification**: assistant prose is 24.9% of output-stripped and carries findings. dialogue record is 14.4 KB of pure prose and scores 18.5 points lower than output-stripped's 12.4 KB prose + 46.9% tool inputs, demonstrating prose alone is insufficient.

**byte effect**: neutral.

### tool calls — structured
tool name, salient arguments only. not the full input dict.

per tool:
- **Read**: `Read(path, offset=N, limit=M)` or `Read(path)` when whole-file.
- **Edit**: `Edit(path, old_head, new_head, replace_all)` where `old_head` and `new_head` are the first 60 chars of the old and new strings. full strings for edits under 200 chars total.
- **Write**: `Write(path, size=N chars)`. do NOT inline content. the renderer will note "writing X chars" rather than replaying them.
- **Bash**: full command if under 200 chars. for longer commands, the first 100 chars + `[... +N chars]`.
- **Agent**: `Agent(name, description, model, isolation)`. prompt is summarized as `prompt_head` (first 100 chars).
- **other tools**: tool name + any path/query/pattern argument + a size/count marker for payload fields.

**justification**: tool inputs are 46.9% of output-stripped and are the active ingredient that distinguishes output-stripped (90.3%) from dialogue record (71.8%). but much of that 46.9% is full file writes, re-reads of the same file, and edit payloads where the first 60 chars convey the intent. the blind 2000-char cap in output-stripped spends tokens on the 800th char of a bash script Write when the fact that a bash script was written is the signal. **this is WHERE the action record saves bytes vs output-stripped**, not from dropping signatures or boilerplate.

**byte effect vs output-stripped**: measured 0.71x (23,905 / 33,443 chars). the saving comes from structured tool calls (e.g., `Write(path, 5000 chars)` instead of capping the full content at 2000 chars), not from dropping signatures (0.00% of output-stripped) or boilerplate (3.04%). estimated 40-60% reduction against output-stripped's tool input volume when that volume is Write/Edit payloads, neutral when it is already-compact commands.

### tool results — one-line digests
- **Read/WebFetch**: `lines: N` or `bytes: N`, first 80 chars of content (one line), whether it was an error.
- **Edit/Write**: success or error message only. no content echo.
- **Bash**: `exit: N`, first 80 chars of stdout, first 80 chars of stderr if nonzero exit.
- **Agent**: completion status (`success`, `error`, `timeout`), agent name, one-line summary of what it reported (first 100 chars of its SubagentHandback message).
- **other tools**: success/error, a count (rows returned, files matched, bytes transferred), first line of output or error.

**justification** (corrected 2026-09-30): tool results are 21.68% of raw transcript as content (11,713,000 of 54,037,811 chars, measured from `toolUseResult` and `tool_result` blocks), and are dropped entirely by output-stripped, yet `artifact_detail` and `outcome` question classes plausibly have ground truth in those results (e.g., "what did the baseline gate run output?" needs the output). a one-line digest conveys success/failure, scale, and a sample without the full payload. the action record's motivation is size (0.71x output-stripped at 23,905 vs 33,443 chars median) and that specific question classes (`artifact_detail`, `outcome`) may benefit from result digests. the original claim that output-stripped (90.3%, n=124) trails raw (95.5%, n=89) by 5.2 points is falsified: on the paired 89 questions output-stripped scores 98.9% against raw's 95.5%, so output-stripped is already ahead and there is no gap for result digests to close.

**byte effect**: measured +2k-4k chars (one line per tool call, ~311 tool calls per large session at ~10-15 chars per line). **this is an addition vs output-stripped, not a deletion.**

### errors and refusals
kept in full. these are short (one line to one paragraph), high-information, and diagnostic.

**justification**: errors and refusals are rare and signal dead ends or gates hit. no reason to truncate them.

**byte effect**: negligible. errors are <1% of transcripts.

### turn index
an explicit turn number prefixes each user/assistant exchange.

**justification**: `sequence` is the worst-recovered class at 37.5%, suggesting that ordering is nearly free to encode (one int per turn) and plausibly load-bearing.

**byte effect**: ~50-100 bytes (25-50 turns × 2 bytes each).

### reasoning blocks (if non-empty)
kept in full when `thinking` or `redacted_thinking` blocks contain non-empty text.

**justification**: 3,659 of 3,666 thinking blocks have empty text = 99.8% (n=12 sessions). non-empty reasoning is rare and may be diagnostic. empty blocks with signatures are dead weight.

**byte effect**: measured as 0.0% of output-stripped. keeping the rare non-empty ones costs <500 chars across a session.

## what the action record drops

### thinking block signatures
every thinking block carries a base64 `signature` field of roughly 1,287 chars (mean, derived from the cross-checked total below), carrying zero information. the signature total over the 3,659 empty blocks in these n=12 sessions is the measured 4,708,852 chars, i.e. ~4.7 MB.

**do not multiply the ~2,125-char median by the block count.** this line read `~2,125-char ... median measured` and `3,659 × 2,125 = ~7.8 MB` until 2026-10-01, and that product contradicts the measured sum on the very next line by 1.65x: 4,708,852 / 3,659 = 1,287 chars per block, not 2,125. a median times a count is not a sum, and here the two differ by more than 3 MB. the sum is the figure to trust, because it is the one cross-checked to 0.00pp; the 2,125 median is either over a different block set (all thinking blocks rather than the empty ones) or simply wrong, and nobody has gone back to the raw data to say which. the paper's `paper/main.tex` figures — a signature of roughly 1.2 KB, and 21.4 MB over the 17,608 empty blocks of its own larger population, which implies 1,215 chars per block — agree with the sum and not with the median, so they are correct as printed and were briefly and wrongly flagged as a 1.8x overstatement during the 2026-10-01 CAIN review.

**justification**: measured at 8.71% of raw transcript (4,708,852 of 54,037,811 chars, cross-checked to 0.00pp, per-session up to 13.53%). `strip_transcript()` already skips them (it `continue`s past empty-text thinking blocks around line 121), so they are 0.00% of output-stripped.

**byte effect vs raw**: -8.71% (up to 13.53% in largest sessions). saves 1 of 5 over-window sessions.
**byte effect vs output-stripped**: 0.00%. no saving. output-stripped already removes them.

### repeated harness boilerplate
`<system-reminder>`, `<local-command-caveat>`, and other meta blocks that appear in multiple turns can be emitted once at the top with a note "this block appears in N turns" rather than repeated.

**justification**: measured at 3.04% of the output-stripped artifact (per-session range 0.00% to 14.33%). harness-injected blocks like `<system-reminder>` are already 0.00% because `split_meta()` removes them before rendering. true repeated chunks of 100+ chars are the remaining 3.04%. the rule is cheap to implement but buys only 3%, not a significant share.

**byte effect vs output-stripped**: -3.04% ceiling, ~1k chars for a session at the high end. keep the rule if cheap, but this is not a major saving.

### tool result bodies beyond the digest
full stdout, full file contents returned by Read, full API responses — replaced by the one-line digests above.

**justification**: tool results are 21.68% of raw as content (11,713,000 of 54,037,811 chars). output-stripped drops them entirely (0.00% of output-stripped). the action record drops the bodies (like output-stripped) but adds one-line digests (unlike output-stripped). this is a large compression vs raw, and a modest addition vs output-stripped.

**byte effect vs raw**: -21.68% as content + one-line digest cost (~2k-4k chars). net saving ~18-19%.
**byte effect vs output-stripped**: +2k-4k chars. **this is an enrichment, not a deletion.**

### re-reads of a file already recorded
when a file is Read multiple times in one session, the second and later reads are noted as `Read(path) [content-hash: abc123, already shown]` rather than inlining the content again.

**justification**: sessions often re-read a file after editing it or read the same file in a loop. the content is recorded once; later reads note the hash to convey "this file was accessed again" without replaying its bytes.

**byte effect**: estimated -1k-3k chars for sessions with file re-reads (common in long sessions). **not yet implemented in action_record.py**, so not reflected in the measured 23,905 chars.

## measured size vs existing formats

**measured over 25 rq3 sessions**:
- action record median: 23,905 chars
- output-stripped median: 33,443 chars
- dialogue record median: 14,379 chars
- raw transcript median (estimated): ~700,000 chars

**ratios**:
- action:raw = 0.034x (23,905 / 700,000). **action record is a large compression vs raw transcript.**
- action:output-stripped = 0.71x (23,905 / 33,443). **action record is smaller than output-stripped.**
- action:dialogue = 1.66x (23,905 / 14,379). action record is larger, but adds tool context.

**why action < output-stripped when output-stripped already removes signatures (0.00%) and boilerplate is only 3.04%**:
the saving comes from **structured tool calls**. output-stripped caps tool inputs at 2000 chars blindly, spending tokens on the 800th char of a Write payload. action record instead renders `Write(path, 5000 chars)`, conveying the intent in ~30 chars. measured tool call representation in action record: ~6k chars vs output-stripped's ~15.7k (46.9% of 33k). that 9.7k saving, minus the +2k-4k for result digests and +100 for turn markers, nets to action at 0.71x output-stripped.

**large session example (9c37ec9f)**:
- raw: 4.74 MB
- output-stripped: 368,861 chars
- action: 193,439 chars (0.52x output-stripped)

large sessions benefit more from structured tool calls because they have more Write/Edit payloads.

**small session example (12a5c2da)**:
- raw: 121 KB
- output-stripped: 1,297 chars
- action: 1,277 chars (0.98x output-stripped)

small sessions save little because they have few tool calls to structure.

## degenerate cases to test

before running the full 25 sessions, prove the builder on:
- one small session (12a5c2da, 121 KB raw) ✓ tested
- one large session (9c37ec9f, 4.7 MB raw, 368k stripped) ✓ tested
- a session with no tool calls (or mock one by stripping them from a real transcript)
- a session with a single turn (or mock one)

these cover: normal operation, volume stress, missing tools, missing structure.
