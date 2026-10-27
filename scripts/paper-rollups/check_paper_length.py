#!/usr/bin/env python3
"""Measure paper/main.tex against the CAIN page allowance, 10 content + 2 references.

ADVISORY BY DEFAULT since 2026-10-04. It measures and reports, and exits 0 whatever
the counts are. The paper is deliberately over the content allowance and the cuts
that would bring it under are the human's to make by hand. On 2026-10-04 an agent
trying to turn this check green deleted an RQ2 sentence disambiguating the
file-path survival denominator and a whole paragraph carrying the held-out-draw
paired-disagreement reversal plus a prior-work citation, then labelled the commit
as a cut to a section it had never touched. Both cuts were reverted. A gate that
drives that behaviour is worse than no gate, so the blocking exit now needs
--enforce and nothing in the repo passes it.

Written because on 2026-09-30 three sessions independently found the paper at 11
content pages while every check in the repo was green. The failure is specific and
worth naming, because the obvious check is the one that misses it: CAIN's allowance
is usually read as "12 pages total", and the paper builds to exactly 12. References
occupied only 1 of their 2 permitted pages, so the unused reference page silently
absorbed a content page. A total-page assertion passes on a non-compliant paper.

check_paper_invariants.py cannot catch this and should not try: it is a fast regex
gate over figures and audit trail, 45 assertions and no build. Length needs the
engine. Keep them separate so the regex gate stays instant.

Two further traps, both hit while establishing the 11:

  1. tectonic runs multiple passes and prints "Output written" once per pass. The
     first pass of the current paper reports 11 pages and the last reports 12,
     because the bibliography and cross-references are not resolved until later
     passes. Only the LAST pass is the real count. A stream-wide grep, or reading
     main.log, reports the paper as compliant when it is not.
  2. Content pages cannot be derived by subtracting a guessed reference length.
     The only reliable split is a second build with \\bibliography and
     \\bibliographystyle removed, which is what CONTENT_ONLY measures here.

Reads origin/main by default rather than the working copy, because the shared
main checkout is routinely dirty with another session's edits and the plugin's,
so a local build can report a number that is nobody's submission.

Usage:
  python check_paper_length.py              # advisory, builds origin/main:paper/, exits 0
  python check_paper_length.py --worktree   # advisory, builds ./paper/ as it sits on disk
  python check_paper_length.py --enforce    # blocking: non-zero exit when over allowance
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import REPO_ROOT  # noqa: E402

VAULT = str(REPO_ROOT)

# CAIN 2027: 10 pages of content, plus up to 2 further pages for references only.
MAX_CONTENT_PAGES = 10
MAX_REFERENCE_PAGES = 2

BAR = "=" * 78

ADVISORY_BANNER = f"""{BAR}
PAGE ENFORCEMENT IS OFF. THIS IS AN ADVISORY MEASUREMENT, NOT A FAILING CHECK.
NOTHING BELOW IS A FAILURE TO FIX. This script exits 0 whatever the counts are.

The paper is KNOWINGLY over the {MAX_CONTENT_PAGES}-page content allowance. Page cuts are the
human's decision, made by hand, and are not delegated.

NO AGENT MAY CUT, TRIM, SHORTEN OR REWORD CONTENT IN paper/main.tex TO MOVE THESE
NUMBERS. Not a sentence, not a clause, not a paragraph, not a figure. On 2026-10-04
an agent chasing a green gate deleted an RQ2 sentence that disambiguated the
file-path survival denominator and a whole paragraph carrying the held-out-draw
paired-disagreement reversal and a prior-work citation, then described the commit
as a cut to a different section. Both were reverted. If you were asked to make
this check pass, the correct answer is to report that it is advisory and stop.
{BAR}
"""

ENFORCING_BANNER = f"""{BAR}
--enforce: BLOCKING MODE. Exits non-zero when over the allowance.
Even here, the remedy is never an agent editing paper/main.tex. Report the number.
{BAR}
"""


def stage_from_origin(dest):
    """Extract origin/main:paper/ into dest. Needs the whole directory, not just
    main.tex: the build pulls refs.bib and every file under fig/."""
    dest.mkdir(parents=True, exist_ok=True)
    ar = subprocess.run(["git", "-C", VAULT, "archive", "origin/main", "paper"],
                        capture_output=True)
    if ar.returncode != 0:
        raise SystemExit(f"FAIL: git archive origin/main failed: "
                         f"{ar.stderr.decode(errors='replace')[:300]}")
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], input=ar.stdout,
                         capture_output=True)
    if tar.returncode != 0:
        raise SystemExit(f"FAIL: could not unpack archive: "
                         f"{tar.stderr.decode(errors='replace')[:300]}")
    return dest / "paper"


def build(srcdir, workdir, tex_body):
    """Build tex_body against srcdir's assets and return the LAST pass's page count."""
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("refs.bib",):
        if (srcdir / name).exists():
            shutil.copy(srcdir / name, workdir / name)
    if (srcdir / "fig").exists():
        shutil.copytree(srcdir / "fig", workdir / "fig", dirs_exist_ok=True)
    (workdir / "main.tex").write_text(tex_body)
    r = subprocess.run(["tectonic", "--print", "main.tex"],
                       cwd=workdir, capture_output=True, text=True, timeout=900)
    passes = re.findall(r"Output written on \S+ \((\d+) pages",
                        r.stdout + r.stderr)
    if not passes:
        tail = (r.stdout + r.stderr)[-1500:]
        raise SystemExit(f"FAIL: tectonic produced no page count.\n{tail}")
    # Last pass only. See trap 1 in the module docstring.
    return int(passes[-1]), [int(p) for p in passes]


def main():
    use_worktree = "--worktree" in sys.argv
    enforce = "--enforce" in sys.argv

    print(ENFORCING_BANNER if enforce else ADVISORY_BANNER)

    with tempfile.TemporaryDirectory(prefix="paperlen-") as td:
        td = pathlib.Path(td)
        if use_worktree:
            srcdir = pathlib.Path("paper").resolve()
            if not (srcdir / "main.tex").exists():
                raise SystemExit("ERROR: no paper/main.tex in the current directory")
            measured = (f"{srcdir / 'main.tex'}\n"
                        f"               (the WORKING COPY on disk, --worktree was passed)")
        else:
            srcdir = stage_from_origin(td / "origin")
            head = subprocess.run(["git", "-C", VAULT, "rev-parse", "--short",
                                   "origin/main"], capture_output=True, text=True)
            sha = head.stdout.strip() or "unknown"
            measured = (f"origin/main:paper/main.tex  @ {sha}\n"
                        f"               (the COMMITTED file on origin/main, NOT your working\n"
                        f"               copy and NOT your branch. This is the default. Pass\n"
                        f"               --worktree to measure ./paper/main.tex on disk.)")

        tex = (srcdir / "main.tex").read_text()
        # Characters, not on-disk bytes: read_text() applies universal-newline
        # translation, so this runs a little under what `ls -l` reports.
        print(f"  measuring:   {measured}")
        print(f"  size:        {len(tex)} chars\n")

        total, total_passes = build(srcdir, td / "full", tex)

        nobib = re.sub(r"^[ \t]*\\bibliography(style)?\{[^}]*\}[ \t]*$", "",
                       tex, flags=re.M)
        if nobib == tex:
            raise SystemExit("ERROR: found no \\bibliography line to strip, so "
                             "content and reference pages cannot be separated")
        content, content_passes = build(srcdir, td / "nobib", nobib)

    refs = total - content
    print(f"  full build      {total:>3} pages   (passes: {total_passes})")
    print(f"  content only    {content:>3} pages   (passes: {content_passes})")
    print(f"  references       {refs:>3} pages\n")

    bad = 0
    for label, got, cap in (("content pages", content, MAX_CONTENT_PAGES),
                            ("reference pages", refs, MAX_REFERENCE_PAGES)):
        ok = got <= cap
        bad += not ok
        if ok:
            tag = "PASS" if enforce else "INFO"
        else:
            tag = "FAIL" if enforce else "OVER"
        print(f"{tag}  {label}: {got} (allowance <= {cap})")

    if bad:
        over = content - MAX_CONTENT_PAGES
        if over > 0:
            print(f"\nContent is {over} page(s) over. Note that the full build's "
                  f"{total} pages is within the {MAX_CONTENT_PAGES + MAX_REFERENCE_PAGES}-page "
                  f"total, because references use {refs} of their "
                  f"{MAX_REFERENCE_PAGES} permitted pages. A total-page check "
                  f"passes here; this one is the reason why that is not enough.")
    else:
        print(f"\nWithin the {MAX_CONTENT_PAGES}+{MAX_REFERENCE_PAGES} allowance.")

    if not enforce:
        print(f"\n{BAR}\nADVISORY ONLY - exiting 0. The OVER line above is NOT a failing check\n"
              f"and is NOT yours to fix. Do not edit paper/main.tex in response to it;\n"
              f"page cuts are the human's decision. Measured: "
              f"{'./paper/main.tex (working copy)' if use_worktree else 'origin/main:paper/main.tex'}\n{BAR}")
        return 0

    if bad:
        print("\n--enforce was passed, so this exits non-zero. The remedy is the "
              "human's hand-made cut, never an agent's.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
