"""Centralized path and corpus-identity configuration for the paper scripts.

This module exists so the reproducibility scripts carry no absolute machine
paths and no author-identifying strings. There are two layers:

1. **Repo layout** -- derived from this file's location, so a clone works
   wherever it is placed and a git worktree resolves to the same data.
2. **Machine and corpus settings** -- directories outside the repo (the agent
   transcript store, the trace archive) and the identity tokens that appear
   *inside* the corpus text. These resolve from, in increasing precedence:
   a documented default, an optional JSON config file, then an environment
   variable.

Precedence:  ``PAPER_<KEY>`` env var  >  ``paper_config.json``  >  default.

The config file lives next to this module and is **not** committed: it is where
a machine-specific or identifying value belongs, which keeps those values out
of the published artifact. ``paper_config.example.json`` documents every key.
Point elsewhere with ``PAPER_CONFIG=/path/to/config.json``.

Usage:
    from paths import REPO_ROOT, DATA_DIR, PROJECTS_DIR, TRACES_DIR, PYTHON
    from paths import CORPUS_HOME, fold_repo_aliases
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------- repo layout

_THIS = Path(__file__).resolve()

# Repo root derived from this file's location (scripts/paper-rollups/paths.py).
REPO_ROOT = _THIS.parents[2]


def _primary_checkout(root: Path) -> Path:
    """Resolve the primary checkout when running inside a git worktree.

    A worktree under ``.claude/worktrees/<name>`` is its own root, so
    ``root.name`` is the branch name rather than the repository's. The
    corpus-identity defaults below derive from the repository name, which must
    mean the same thing in a worktree as in the main checkout.
    """
    if root.parts[-3:-1] == (".claude", "worktrees"):
        return root.parents[2]
    return root


# The repository directory as the user named it. Used only to derive identity
# defaults, never to locate files -- those all hang off REPO_ROOT.
PRIMARY_CHECKOUT = _primary_checkout(REPO_ROOT)

# ------------------------------------------------------------- config loading

CONFIG_PATH = Path(
    os.environ.get("PAPER_CONFIG", _THIS.parent / "paper_config.json")
)

_config: dict = {}
if CONFIG_PATH.exists():
    try:
        _config = json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError) as exc:  # pragma: no cover - operator error
        raise SystemExit(f"paths.py: cannot read {CONFIG_PATH}: {exc}")


def setting(key: str, default):
    """Resolve one setting: env var, then config file, then the default.

    The env var is ``PAPER_`` plus the upper-cased key. A list-valued setting
    accepts a comma-separated env var.
    """
    env = os.environ.get("PAPER_" + key.upper())
    if env is not None:
        if isinstance(default, (list, tuple)):
            return [part for part in (p.strip() for p in env.split(",")) if part]
        return env
    if key in _config:
        return _config[key]
    return default


def _dir(key: str, default: Path) -> Path:
    """Resolve a directory setting, expanding ``~`` and making it absolute."""
    return Path(str(setting(key, default))).expanduser()


# ------------------------------------------------------- in-repo directories

DATA_DIR = _dir("data_dir", REPO_ROOT / "data" / "paper-rollups")
PAPER_DIR = _dir("paper_dir", REPO_ROOT / "paper")
SESSIONS_DIR = _dir("sessions_dir", REPO_ROOT / "sessions")
OBSIDIAN_DIR = _dir("obsidian_dir", REPO_ROOT / ".obsidian")

# Indexer directory name (without full path) for resolution and banning.
# Default is neutral; override with the real name locally.
INDEXER_DIR_NAME = str(setting("indexer_dir_name", "skills-indexer"))

PUBLIC_SKILLS_DIR = _dir(
    "public_skills_dir", REPO_ROOT / "public" / "skills" / INDEXER_DIR_NAME
)

# ------------------------------------------------ directories outside the repo

# The agent's own state directory. ``~/.claude`` is where Claude Code keeps
# transcripts on every platform, so this default is portable, not machine
# specific.
CLAUDE_HOME = _dir("claude_home", Path.home() / ".claude")

# Transcript store. CLAUDE_PROJECTS_DIR is honoured for backwards
# compatibility with scripts and shell invocations that predate this module.
PROJECTS_DIR = Path(
    os.environ.get("CLAUDE_PROJECTS_DIR")
    or str(_dir("projects_dir", CLAUDE_HOME / "projects"))
).expanduser()

# Scratch space for background jobs; some arms stage artifacts here.
JOBS_DIR = _dir("jobs_dir", CLAUDE_HOME / "jobs")

# Prompt-context telemetry log, read by the injection-distribution analysis.
TELEMETRY_LOG = _dir(
    "telemetry_log", CLAUDE_HOME / "prompt-context-telemetry.jsonl"
)

# Level 2/3 trace archive. A separate repository checked out beside this one, so
# the default derives from this checkout's parent rather than from the home
# directory: that is both correct for a side-by-side clone and free of any
# absolute path. A home-relative default was wrong here -- the archive sits
# next to the repo, not at the top of the home directory.
TRACES_DIR = _dir(
    "traces_dir", PRIMARY_CHECKOUT.parent / "agent-session-traces" / "traces"
)

# Rollup level-1 source code directory (agent-session-rollup-level1/src/...).
# Default resolves to a sibling of this repository.
ROLLUP_SRC_DIR = _dir(
    "rollup_src_dir",
    PRIMARY_CHECKOUT.parent / "agent-session-rollup-level1" / "src" / "agent_session_rollup"
)

# --------------------------------------------------------- interpreter choice

# A virtualenv interpreter is required on the authoring machine because
# ``python`` is not on PATH there and the system interpreter lacks numpy.
# Prefer the running interpreter; override with PYTHON_INTERPRETER.
PYTHON = os.environ.get("PYTHON_INTERPRETER") or sys.executable

# ------------------------------------------------------------ corpus identity

# These describe strings that appear *inside* the corpus text, not locations on
# disk. The parsing and scrubbing code has to recognise them to behave
# correctly, so they are configuration rather than literals: the published
# scripts stay free of identifying tokens while behaviour on the real corpus is
# unchanged. Defaults derive from the running machine and the repo directory
# name, which is what the corpus actually contains.

# Home-directory prefix as it appears in transcript text (e.g. cited paths).
CORPUS_HOME = str(setting("corpus_home", str(Path.home())))

# Spellings of the notes repository found in the corpus. macOS paths are
# case-insensitive, so the same repo appears under more than one casing and a
# comparison that does not fold them undercounts matches.
CORPUS_REPO_ALIASES = list(
    setting("corpus_repo_aliases", [PRIMARY_CHECKOUT.name])
)

# Terms whose presence in a release artifact would identify the author or their
# organisation. Consumed by the sanitization audit; extend via config.
# Includes the indexer directory name by default.
_default_sensitive = [PRIMARY_CHECKOUT.name, INDEXER_DIR_NAME]
CORPUS_SENSITIVE_TERMS = list(
    setting("corpus_sensitive_terms", _default_sensitive)
)


def repo_alias_pattern() -> re.Pattern:
    """Case-insensitive alternation over the configured repository spellings."""
    alts = sorted((a for a in CORPUS_REPO_ALIASES if a), key=len, reverse=True)
    if not alts:
        # Match nothing rather than the empty string, which would match
        # everywhere and corrupt every substitution built on this pattern.
        return re.compile(r"(?!)")
    return re.compile("|".join(re.escape(a) for a in alts), re.I)


def fold_repo_aliases(text: str) -> str:
    """Fold every configured spelling of the repo name to one lowercase form.

    Makes two citations of the same file compare equal when they differ only in
    the casing of the repository directory.
    """
    canonical = (CORPUS_REPO_ALIASES[0] if CORPUS_REPO_ALIASES else "").lower()
    return repo_alias_pattern().sub(canonical, text or "")


def project_slug(path) -> str:
    """Claude Code's transcript-directory name for a project path.

    A transcript directory under PROJECTS_DIR is the project's absolute path
    with every ``/`` and ``.`` replaced by ``-``, so
    ``/x/y/.cache/z`` becomes ``-x-y--cache-z``. Deriving it matters because the
    hardcoded form embeds the home directory, and in that spelling a pattern
    looking for ``/`` separators cannot see it.
    """
    return str(path).replace("/", "-").replace(".", "-")


def home_path_pattern() -> re.Pattern:
    """Match an absolute home-directory prefix on macOS, Linux or Windows.

    Used to find paths needing redaction. Includes the configured corpus home
    so the audit still fires on the corpus's own spelling.
    """
    return re.compile(
        "|".join(
            [
                re.escape(CORPUS_HOME.rstrip("/")) + "/",
                r"/Users/[^/]+/",
                r"/home/[^/]+/",
                r"C:\\Users\\[^\\]+\\",
            ]
        )
    )
