#!/usr/bin/env python3
"""Flag LLM-cadence constructions in a paper's prose, so a human can judge them.

Written on 2026-10-02, after the human read the CAIN abstract and said it still
sounded machine-written. The offending sentence was "Deletion omits; generation
invents." -- a short antithetical pair whose content was already carried by the
two failure rates three lines above it. Five such constructions were in the
paper and four of them were in the abstract, which is the part every reader
reads and the part most heavily rewritten, so it accumulates them fastest.

This is a REPORTER, not a gate. It exits 0 whatever it finds, because none of
these patterns is wrong on its own: of the eight it flagged after the first
cleanup, three were real contrasts carrying information ("Their remedy is
better packing, but ours cannot be packed further") and the rest were terse
statements of fact that only tripped the short-sentence heuristic. A gate that
failed on these would train whoever runs it to stop reading the output. Read
each hit and decide; the point is to put them in front of you, not to score
them.

What it looks for, and why each is a tell rather than a style preference:

  semicolon antithesis   two short clauses balanced across a semicolon. The
                         shape asserts a symmetry the measurements rarely have,
                         and it reads as an epigram where a finding belongs.
  aphoristic closer      a very short sentence ending a paragraph. Often a
                         restatement of the number just given, spending a line
                         on rhythm. Many hits here are innocent.
  negation reveal        "it is not the larger half" -- withholding the subject
                         for one beat. Costs the reader a resolution step and
                         buys drama.
  elliptical less/more   "less text read, more answers recovered". Verbless
                         parallelism; compresses to a slogan.
  em-dash reveal         a short clause after --- delivering a twist.

Usage:
  python check_prose_cadence.py <path-to-main.tex>
  python check_prose_cadence.py --origin      # reads git show origin/main:paper/main.tex

There is no default source, deliberately: on 2026-10-02 check_paper_length.py
silently graded origin/main while an edited worktree sat uncommitted, and
reported PASS on prose that was not the prose being checked.
"""
import pathlib
import re
import subprocess
import sys

PATTERNS = [
    ("semicolon antithesis",
     lambda c: ";" in c and len(c.split()) <= 12),
    ("aphoristic closer (<=7 words)",
     lambda c: len(c.split()) <= 7 and c.endswith(".")),
    ("negation reveal",
     lambda c: re.search(r"\bis not\b[^;]*;|\bnot the \w+ half\b|\bis not a\b.*\bbut\b", c)),
    ("elliptical less/more",
     lambda c: re.search(r"\bless \w+ \w+, more \w+|\bmore \w+, less \w+", c)),
    ("em-dash reveal",
     lambda c: "---" in c and len(c.split()) <= 20),
]


def load(arg):
    if arg == "--origin":
        r = subprocess.run(["git", "show", "origin/main:paper/main.tex"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"FAIL: git show failed: {r.stderr.strip()[:200]}")
        return r.stdout, "origin/main:paper/main.tex"
    p = pathlib.Path(arg)
    if not p.exists():
        raise SystemExit(f"FAIL: no such file: {p}")
    return p.read_text(), str(p)


def prose(src):
    """The body, with floats, tabular blocks and comment lines removed."""
    body = src[src.find("\\begin{document}"):]
    body = re.sub(r"\\begin\{(figure\*?|table\*?|tabular|axis)\}.*?\\end\{\1\}",
                  "", body, flags=re.S)
    body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("%"))
    return " ".join(body.split())


def strip_latex(s):
    s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", s)
    return " ".join(re.sub(r"[{}$~\\]", " ", s).split())


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "FAIL: no source given, and there is no default.\n"
            "  to read an edit you have not pushed:  check_prose_cadence.py paper/main.tex\n"
            "  to read what is already on main:      check_prose_cadence.py --origin")

    src, label = load(sys.argv[1])
    print(f"reading {label}\n")

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", prose(src))

    hits, seen = [], set()
    for s in sentences:
        c = strip_latex(s)
        if len(c) < 8 or c in seen:
            continue
        for name, test in PATTERNS:
            try:
                matched = test(c)
            except Exception:
                matched = False
            if matched:
                seen.add(c)
                hits.append((name, c))
                break

    by_pattern = {}
    for name, c in hits:
        by_pattern.setdefault(name, []).append(c)

    for name, _ in PATTERNS:
        found = by_pattern.get(name, [])
        if not found:
            continue
        print(f"== {name}: {len(found)} ==")
        for c in found:
            print(f"  {c[:200]}")
        print()

    print(f"{len(hits)} construction(s) to read. None is automatically wrong; judge each.")


if __name__ == "__main__":
    main()
