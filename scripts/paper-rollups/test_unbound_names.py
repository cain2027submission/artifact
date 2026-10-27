#!/usr/bin/env python3
"""Gate: no script may USE a name it never imports or binds.

py_compile cannot catch this -- `PROJECTS_DIR` used without
`from paths import PROJECTS_DIR` is valid syntax and a NameError at import
time. That is exactly the defect the path-centralisation sweep introduced in
10 files, and it took down two of three test suites.

Usage:  test_unbound_names.py [dir]   (defaults to this script's directory)
Exit 0 = clean, exit 1 = unbound names found.
"""
import ast
import builtins
import os
import sys

BUILT = set(dir(builtins))


def bound_names(tree):
    """Names bound anywhere at any scope.

    Deliberately over-approximate (ignores scoping) so the gate cannot produce
    a false positive and get itself disabled. It only has to catch the
    never-bound-anywhere case.
    """
    b = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            b.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b.add(n.name)
        elif isinstance(n, ast.arg):
            b.add(n.arg)
        elif isinstance(n, ast.alias):
            b.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            b.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            b.update(n.names)
    return b


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    problems = {}
    n_files = 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py"):
            continue
        n_files += 1
        src = open(os.path.join(d, f)).read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems[f] = [("<SYNTAX ERROR>", getattr(e, "lineno", 0))]
            continue
        allowed = bound_names(tree) | BUILT | {"__file__", "__name__", "__doc__"}
        used = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                used.setdefault(n.id, n.lineno)
        miss = sorted((name, ln) for name, ln in used.items() if name not in allowed)
        if miss:
            problems[f] = miss

    print(f"scanned {n_files} .py files in {d}")
    if not problems:
        print("PASS: no unbound names")
        return 0
    print(f"FAIL: {len(problems)} files use names they never import or bind")
    for f, miss in problems.items():
        print(f"  {f}")
        for name, ln in miss:
            print(f"      line {ln}: {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
