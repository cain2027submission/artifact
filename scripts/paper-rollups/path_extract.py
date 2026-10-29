"""One path extractor and one normalisation, shared by both sides of path recall.

Why this module exists. Path recall is
(transcript paths the rollup also names) / (transcript paths), so its ceiling is
100% by construction. The measurement that reported 44.5% scored up to 400% on
individual sessions, and there were four independent reasons, all of which are
fixed here rather than patched at one call site:

1. It was not a recall. `part1_structural_loss.py` computed
   `len(rollup_files) / len(transcript_files)` -- a ratio of two set
   *cardinalities* with no intersection anywhere in it. A rollup naming four
   path-like tokens against a transcript the extractor read one path out of
   scored 400%, and the true intersection of those two sets was empty. This is
   the dominant defect: the metric never measured overlap at all.

2. The transcript side read only `file_path`/`path`/`paths` off `tool_use`
   blocks. Bash is 54% of tool calls in this corpus and its paths are inside the
   `command` string, so the single most common way a session touches a file was
   invisible. Session 372ca0e6 is 38 Bash calls, 2 Edits and 1 Read; the old
   extractor found exactly one path in it.

3. Neither side normalised. A rollup writes `AGENTS.md` or `src/game/scene.ts`
   where the transcript holds an absolute home-qualified path and a
   `.claude/worktrees/<name>/` -qualified absolute path. Same file, no match.

4. The rollup side emitted directory fragments (home dirs, `learnings/`) and the
   boilerplate trace-backlink path as if they were cited artifacts, inflating the
   numerator with tokens that can never appear in a transcript.

Both sides therefore call `normalise` and nothing else, and matching is a
segment-boundary suffix relation so a prose citation of a file by name counts as
naming it. The numerator is counted over *transcript* paths, which makes
`recall <= 1.0` true by construction rather than by luck.

Run the gate with python3:
    python3 test_path_recall.py
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Iterator, Sequence

from paths import CORPUS_HOME

# The `~` expansion target. Resolved rather than hardcoded: normalisation has to
# agree with whatever home directory the transcripts were written under. Import
# from config so it matches the actual corpus home on the author's machine.
HOME = os.environ.get("PKM_HOME_OVERRIDE") or CORPUS_HOME

# Extensions that make a bare filename a path even with no slash in it. Kept
# explicit: a permissive rule here turns ordinary prose ("see section 3.2") into
# a cited artifact and inflates the numerator, which is the direction of error
# that flatters the rollup arm.
EXTS = (
    "py|js|jsx|ts|tsx|mjs|cjs|md|markdown|json|jsonl|ya?ml|toml|ini|cfg|conf|"
    "sh|bash|zsh|fish|tex|bib|sql|csv|tsv|txt|log|html|css|scss|rs|go|java|kt|"
    "rb|php|c|h|cc|cpp|hpp|swift|m|mm|plist|lock|env|gitignore|gitattributes|"
    "png|jpg|jpeg|svg|gif|pdf|zip|db|sqlite|sqlite3|pkl|npy|parquet|ipynb"
)

# Code and prose idioms that the extension rule would otherwise admit as files:
# `console.log(...)` ends in a `log` extension, `process.env` in `env`,
# `llama.cpp` in `cpp`. They land in the denominator and depress recall, so this
# is a conservative-direction error rather than a flattering one, but it is still
# wrong and cheap to remove.
BASENAME_DENY = {
    "console.log", "process.env", "module.exports", "os.environ", "sys.argv",
    "sys.path", "json.dumps", "json.loads", "json.load", "self.py",
    "window.location", "document.body", "np.array", "pd.read", "e.g", "i.e",
    "etc.md", "llama.cpp", "requirements.in",
}

# Extensionless filenames that are still files. Anything not here and without a
# dot in its last segment is treated as a directory and dropped.
EXTENSIONLESS = {
    "makefile", "dockerfile", "license", "licence", "readme", "gemfile",
    "procfile", "rakefile", "justfile", "cargo", "codeowners", "notice",
}

# A path-like token in free text.
#
# The space handling is the fiddly part and it is not optional: this corpus is a
# note vault whose filenames are sentences ("2026-08-31 topic we can use.md").
# A no-space regex truncated every one of those to `use.md`, so a rollup citing
# a note could never be paired with the transcript's absolute form. But an unbounded
# space-tolerant regex is worse -- the first version of this file turned
# "verified mergeable with passing typecheck/build and no new lint warnings."
# into a cited artifact, because prose contains slashes and full stops. So spaces
# are allowed *only* in a final segment that terminates in a known extension, the
# words inside such a run may not begin or end with a dot (which is what stops a
# match running across a sentence boundary), and directory segments never contain
# spaces.
_WORD = r"[\w\-]+(?:\.[\w\-]+)*"          # a word; dots only between word chars
_DIR = r"[\w.\-]+"                        # directory segment, no spaces
_FILE_SPACED = rf"{_WORD}(?:[ \t]{_WORD}){{0,14}}\.(?:{EXTS})(?![\w])"
_LEAD = r"(?:[~.]{0,2}/)?"
_NOTBEFORE = r"(?<![\w/~.\-])"
CANDIDATE_RE = re.compile(
    # /abs/or/rel/path/to/a sentence filename.md   (spaced final segment)
    rf"{_NOTBEFORE}{_LEAD}(?:{_DIR}/)+{_FILE_SPACED}"
    # /abs/or/rel/path/to/file           (no spaces anywhere)
    rf"|{_NOTBEFORE}{_LEAD}(?:{_DIR}/)+{_DIR}"
    # bare filename.ext, possibly a sentence
    rf"|{_NOTBEFORE}{_FILE_SPACED}"
)

# Obsidian wikilinks. A rollup citing a vault note writes [[title]] far more often
# than it writes title.md, and every vault note is markdown, so a wikilink is a
# path citation. Ignoring them undercounts the rollup side on exactly the sessions
# that touch the vault, which is most of this corpus.
WIKILINK_RE = re.compile(r"\[\[([^\[\]|#]{1,200})(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")

# The extension a normalised key must end in for it to be a file. Enforced in
# `normalise` as well as in the regex, because the structured tool-input keys
# (`file_path` and friends) bypass the regex entirely and a bare
# `description: "local models."` must not become an artifact.
_EXT_OK = re.compile(rf"\.(?:{EXTS})$", re.I)

# Whole-match rejects. Dates (2026/09), prose slashes, URLs, git refs, and the
# `a/b` `b/a` prefixes diff output uses.
NOISE_RE = re.compile(
    r"^(?:and/or|either/or|n/a|w/o|km/h|[\d.]+/[\d.]+|https?://.*|git@.*|"
    r"origin/.*|refs/.*|HEAD~?\d*|[\d/\-]+)$",
    re.I,
)

# A trailing source reference: foo.py:66 or foo.py:66:12. Dropped so a rollup
# citing a line number still names the file.
LINEREF_RE = re.compile(r":(\d+)(?::(\d+))?$")

# .../.claude/worktrees/<name>/rest -> .../rest. A worktree is a checkout of the
# same repo at a different path, so a rollup naming src/game/scene.ts and a
# transcript holding
# /repo/.claude/worktrees/feature-branch/src/game/scene.ts are the
# same file. Without this collapse session 372ca0e6 matches nothing.
WORKTREE_RE = re.compile(r"/\.claude/worktrees/[^/]+/")

# Surrounding syntax a path picks up in markdown, JSON and shell.
STRIP_CHARS = "`'\"()[]{}<>,;:!?*  \t\n\r\\"


def _lexical_resolve(p: str) -> str:
    """Collapse //, . and .. without touching the filesystem.

    Deliberately lexical: `Path.resolve()` follows symlinks and consults the disk,
    so it would give different answers depending on what happens to exist when
    the script is re-run, and the whole point of the frozen corpus is that a
    re-run reproduces the number.
    """
    lead = "/" if p.startswith("/") else ""
    out: list[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if out and out[-1] != "..":
                out.pop()
            elif not lead:
                out.append("..")
            continue
        out.append(seg)
    return lead + "/".join(out)


def normalise(raw: str, root: str | None = None) -> str | None:
    """Canonical comparison key for a path, or None if `raw` is not one.

    The same function is applied to the transcript side and the rollup side --
    that identity is the fix, not an implementation detail. Passing different
    normalisation to the two sides is how the original measurement produced an
    empty intersection between two sets naming the same files.

    `root` resolves a relative path. It is the session's `project_dir`
    (rollup frontmatter) or the entry's `cwd` (transcript), so `AGENTS.md` in a
    repo-rooted session and the transcript's absolute form land on one key.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().strip(STRIP_CHARS).strip()
    if not s:
        return None

    # JSON/shell escaping. A transcript holds these inside a serialised tool
    # input; a rollup can carry them in a fenced block copied out of one.
    s = s.replace("\\/", "/").replace("\\ ", " ").replace("\\\\", "\\")
    s = s.replace("\\n", " ").replace("\\t", " ")
    s = s.strip().strip(STRIP_CHARS).strip()
    if not s or s.startswith("-"):
        return None

    s = LINEREF_RE.sub("", s)
    if NOISE_RE.match(s):
        return None
    if "://" in s or s.startswith("@") or "*" in s or "?" in s:
        return None  # URL or glob, not a concrete path

    if s.startswith("~"):
        s = HOME + s[1:]
    if not s.startswith("/"):
        if root:
            s = _lexical_resolve(root) + "/" + s
        # A relative path with no root stays relative; suffix matching can still
        # pair it with an absolute form on the other side.

    s = _lexical_resolve(s)
    s = WORKTREE_RE.sub("/", s)
    if not s:
        return None

    base = s.rsplit("/", 1)[-1]
    if not base:
        return None
    # Must be a file, and must end in an extension we recognise. Home directories,
    # `learnings/` and `.../src/game` were all counted as cited artifacts by
    # the old rollup extractor and none of them can appear as a file in a
    # transcript; `local models.` and `see section 3.2` are prose that an
    # any-dot-will-do rule admitted.
    if not _EXT_OK.search(base) and base.lower() not in EXTENSIONLESS:
        return None
    if base.lower() in BASENAME_DENY:
        return None
    # A bare extension such as `.md`, left over from a truncated match. Checked
    # against a short list rather than "starts with a dot", because `.gitignore`
    # and `.env` are real files whose whole name looks like an extension.
    if base.lower() in {
        ".md", ".py", ".js", ".ts", ".tsx", ".json", ".jsonl", ".txt", ".log",
        ".csv", ".sh", ".yaml", ".yml", ".html", ".css", ".tex",
    }:
        return None

    # macOS is case-insensitive and this corpus spells the notes repo in two
    # casings (82.8% of mentions use the second). Case-folding the key is what
    # makes those one file instead of two.
    return s.lower()


def _quoted_spans(text: str) -> Iterator[str]:
    """Yield the contents of backtick, single-quote and double-quote spans.

    Run before the bare scan because a quoted path may contain spaces that the
    bare regex is not allowed to cross in unbounded fashion.
    """
    for m in re.finditer(r"`([^`\n]{1,400})`|\"([^\"\n]{1,400})\"|'([^'\n]{1,400})'", text):
        yield next(g for g in m.groups() if g is not None)


def candidates(text: str) -> Iterator[str]:
    """Every path-like token in a blob of free text (prose, a shell command, ...)."""
    if not text:
        return
    for m in WIKILINK_RE.finditer(text):
        yield m.group(1).strip() + ".md"
    for span in _quoted_spans(text):
        span = span.strip()
        if span:
            yield span
            for m in CANDIDATE_RE.finditer(span):
                yield m.group(0)
    for m in CANDIDATE_RE.finditer(text):
        yield m.group(0)


def normalise_variants(raw: str, root: str | None = None) -> set[str]:
    """Every key `raw` may legitimately be compared under.

    A relative citation gets two keys: root-resolved, and left relative. This is
    needed because `project_dir` is the directory the *session started in*, not
    the repo root of the file it touched. Session 372ca0e6 ran from a home
    directory and edited `src/game/scene.ts` in a game worktree; resolving
    that citation against the wrong root fabricates a nonexistent path which is a
    suffix of nothing, so root-resolution alone made the rollup match zero of the
    files it had plainly named. Keeping the unresolved `src/game/scene.ts`
    lets suffix matching find it.

    Aliases are used on the rollup (numerator) side only. The denominator is built
    from single canonical keys, and the numerator is counted over denominator
    members, so extra aliases can never push recall above 1.0.
    """
    out: set[str] = set()
    k = normalise(raw, root)
    if k:
        out.add(k)
    if root and not raw.lstrip("`'\"([{<").startswith(("/", "~")):
        k2 = normalise(raw, None)
        if k2:
            out.add(k2)
    return out


def paths_in_text(text: str, root: str | None = None, variants: bool = False) -> set[str]:
    out: set[str] = set()
    for c in candidates(text):
        if variants:
            out |= normalise_variants(c, root)
        else:
            k = normalise(c, root)
            if k:
                out.add(k)
    return out


# Tool-input keys whose value is a path (or a list of them).
PATH_KEYS = ("file_path", "path", "paths", "file_paths", "notebook_path",
             "filePath", "target_file", "output_path", "dir", "directory")

# Tool-input keys whose value is free text that can *contain* paths. Split into
# two tiers because the denominator of a recall has to be stated, not assumed:
#
#   TARGET_TEXT_KEYS -- arguments that name what the call operates on. `command`
#     is the one that matters: Bash is 54% of tool calls in this corpus and every
#     path it touches is inside that string, so the old extractor (which read
#     only structured file_path/path/paths) was blind to the majority of file
#     access. This tier is the primary denominator.
#   AUTHORED_TEXT_KEYS -- arguments that are prose or file content. A path here
#     was discussed or written about, not necessarily acted on. Reported as a
#     sensitivity arm.
TARGET_TEXT_KEYS = ("command", "pattern", "glob")
AUTHORED_TEXT_KEYS = ("old_string", "new_string", "content", "prompt",
                      "description", "query", "message", "edits", "summary")

# The three denominators, widest last. Every reported figure names which it used.
SURFACE_ACTED = "acted"        # tool-call arguments only
SURFACE_AUTHORED = "authored"  # + assistant prose, thinking, user text, content args
SURFACE_ALL = "all"            # + tool_result output
SURFACES = (SURFACE_ACTED, SURFACE_AUTHORED, SURFACE_ALL)


def _walk_values(val: Any) -> Iterator[str]:
    if isinstance(val, str):
        yield val
    elif isinstance(val, dict):
        for v in val.values():
            yield from _walk_values(v)
    elif isinstance(val, (list, tuple)):
        for v in val:
            yield from _walk_values(v)


def transcript_paths(entries: Sequence[dict], surface: str = SURFACE_ACTED) -> set[str]:
    """Every distinct path the session names, over the chosen surface.

    `surface=SURFACE_ACTED` is the primary denominator: paths appearing in
    tool-call arguments, which is what the old extractor was trying to measure
    before it missed Bash. `SURFACE_AUTHORED` adds prose, thinking, user text and
    file-content arguments. `SURFACE_ALL` adds tool *results*, which one `ls` of a
    large directory can inflate by hundreds of paths nobody named -- it measures
    how much output got dumped rather than what the session did, so it is a
    sensitivity arm and never the headline.
    """
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}; expected one of {SURFACES}")
    authored = surface in (SURFACE_AUTHORED, SURFACE_ALL)
    results = surface == SURFACE_ALL
    text_keys = TARGET_TEXT_KEYS + (AUTHORED_TEXT_KEYS if authored else ())

    out: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        root = e.get("cwd") or None
        msg = e.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if authored:
                out |= paths_in_text(content, root)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                inp = block.get("input")
                if not isinstance(inp, dict):
                    continue
                for key in PATH_KEYS:
                    if key in inp:
                        for v in _walk_values(inp[key]):
                            k = normalise(v, root)
                            if k:
                                out.add(k)
                for key in text_keys:
                    if key in inp:
                        for v in _walk_values(inp[key]):
                            out |= paths_in_text(v, root)
            elif btype == "text" and authored:
                out |= paths_in_text(block.get("text") or "", root)
            elif btype == "thinking" and authored:
                out |= paths_in_text(block.get("thinking") or "", root)
            elif btype == "tool_result" and results:
                rc = block.get("content")
                if not isinstance(rc, str):
                    rc = json.dumps(rc)
                out |= paths_in_text(rc, root)
    return out


# --- rollup side ------------------------------------------------------------
# Same split as rq3_artifact_citation.py, for the same reason: the trace-backlink
# section always contains the literal trace path, so counting it makes every note
# cite a path it never touched, and the `## Recaps` tail is written mid-session by
# a different mechanism than the summariser.
TRACE_SECTION_RE = re.compile(r"^##\s*the full trace is at.*?(?=^##\s|\Z)", re.M | re.S | re.I)
RECAPS_SECTION_RE = re.compile(r"^##\s*Recaps\b.*?(?=\Z)", re.M | re.S | re.I)


def split_note(text: str) -> tuple[str, str, str]:
    """Return (frontmatter, summary_body, recaps_tail)."""
    fm = ""
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end]
            body = text[end + 4:]
    recaps = ""
    m = RECAPS_SECTION_RE.search(body)
    if m:
        recaps = m.group(0)
        body = body[:m.start()]
    body = TRACE_SECTION_RE.sub("", body)
    return fm, body, recaps


def project_dir(frontmatter: str) -> str | None:
    for line in frontmatter.splitlines():
        k, _, v = line.partition(":")
        if k.strip() == "project_dir":
            v = v.strip().strip("\"'")
            return v or None
    return None


def rollup_paths(text: str, include_recaps: bool = False) -> tuple[set[str], str | None]:
    """Paths the rollup's summary names, plus the session's project_dir root.

    Frontmatter is excluded from the path set but read for `project_dir`: the
    frontmatter carries a mechanical `touched:` list, so counting it would
    measure the pipeline's metadata rather than what the summary prose chose to
    say -- the same exclusion `rq3_artifact_citation.py` makes for the 82.2%.
    """
    fm, body, recaps = split_note(text)
    root = project_dir(fm)
    if include_recaps:
        body = body + "\n" + recaps
    return paths_in_text(body, root, variants=True), root


# --- matching ---------------------------------------------------------------


def _segments(key: str) -> list[str]:
    return [s for s in key.split("/") if s]


def _is_segment_suffix(short: str, long: str) -> bool:
    a, b = _segments(short), _segments(long)
    return len(a) <= len(b) and b[len(b) - len(a):] == a


def match_transcript_paths(rollup: Iterable[str], transcript: Iterable[str]) -> set[str]:
    """Which transcript paths the rollup names.

    Returned over the *transcript* set, never the rollup set. That is what makes
    the ratio a recall: the numerator is a subset of the denominator, so it
    cannot exceed 1.0 however badly either extractor behaves. The old code
    divided the rollup set's size by the transcript set's size, which is why it
    reached 400%.

    A match is exact, or a segment-boundary suffix relation in either direction.
    Suffix-either-way is what lets a rollup that says `AGENTS.md` or
    `src/game/scene.ts` count as naming the absolute file, and a rollup that
    quotes a longer worktree-qualified path count as naming the canonical one.
    """
    tset = set(transcript)
    by_base: dict[str, list[str]] = {}
    for t in tset:
        by_base.setdefault(_segments(t)[-1] if _segments(t) else t, []).append(t)

    hit: set[str] = set()
    for r in rollup:
        if r in tset:
            hit.add(r)
            continue
        segs = _segments(r)
        if not segs:
            continue
        for t in by_base.get(segs[-1], ()):
            if _is_segment_suffix(r, t) or _is_segment_suffix(t, r):
                hit.add(t)
    return hit


def path_recall(rollup: Iterable[str], transcript: Iterable[str]) -> tuple[int, int, float]:
    """(numerator, denominator, recall). recall is None-safe: 0.0 on an empty set."""
    tset = set(transcript)
    num = len(match_transcript_paths(rollup, tset))
    den = len(tset)
    return num, den, (num / den if den else 0.0)
