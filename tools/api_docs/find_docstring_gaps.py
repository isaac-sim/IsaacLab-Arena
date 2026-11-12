# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report public modules, classes and functions that lack a docstring or an ``Args`` section.

Usage: ``python tools/api_docs/find_docstring_gaps.py [--summary] [paths...]``
"""

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = [
    "isaaclab_arena",
    "isaaclab_arena_benchmarking",
    "isaaclab_arena_cosmos",
    "isaaclab_arena_curobo",
    "isaaclab_arena_dreamzero",
    "isaaclab_arena_environments",
    "isaaclab_arena_examples",
    "isaaclab_arena_g1",
    "isaaclab_arena_gr00t",
    "isaaclab_arena_openpi",
    "isaaclab_arena_vlm_agent",
]
SKIP_DIR_NAMES = {"tests", "test", "__pycache__"}


def iter_source_files(roots: list[Path]):
    """Yield non-test Python files below the given roots."""
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            if SKIP_DIR_NAMES.intersection(path.parts):
                continue
            yield path


def has_args_section(docstring: str) -> bool:
    """Return whether the docstring contains a Google-style ``Args:`` section."""
    return re.search(r"^\s*Args:\s*$", docstring, re.MULTILINE) is not None


def parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return the parameter names that need documenting (excluding ``self``, ``cls`` and ``_``-prefixed)."""
    args = node.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    names = [a.arg for a in all_args]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return [n for n in names if n not in ("self", "cls") and not n.startswith("_")]


def find_gaps(path: Path) -> list[tuple[int, str, str]]:
    """Return ``(line, qualified name, problem)`` tuples for one file."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        return [(e.lineno or 0, "<module>", f"syntax error: {e.msg}")]
    gaps = []
    if path.name != "__init__.py" or path.stat().st_size > 0:
        if ast.get_docstring(tree) is None and path.name != "__init__.py":
            gaps.append((1, "<module>", "missing module docstring"))

    def visit(body, prefix: str, in_function: bool):
        for node in body:
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith("_") and not in_function:
                    doc = ast.get_docstring(node)
                    if doc is None:
                        gaps.append((node.lineno, prefix + node.name, "missing class docstring"))
                    visit(node.body, prefix + node.name + ".", False)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_dunder = node.name.startswith("__") and node.name.endswith("__")
                if in_function or (node.name.startswith("_") and not is_dunder):
                    continue
                if is_dunder and node.name != "__init__":
                    continue
                doc = ast.get_docstring(node)
                qualname = prefix + node.name
                if node.name == "__init__":
                    # A constructor only needs a docstring when it takes parameters worth documenting.
                    if parameter_names(node) and (doc is None or not has_args_section(doc)):
                        gaps.append((node.lineno, qualname, "missing Args section"))
                    continue
                if doc is None:
                    gaps.append((node.lineno, qualname, "missing docstring"))
                elif parameter_names(node) and not has_args_section(doc):
                    gaps.append((node.lineno, qualname, "missing Args section"))

    visit(tree.body, "", False)
    return gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Files or directories (default: all first-party packages).")
    parser.add_argument("--summary", action="store_true", help="Print per-file counts only.")
    args = parser.parse_args()
    roots = [Path(p).resolve() for p in args.paths] or [REPO_ROOT / p for p in PACKAGES if (REPO_ROOT / p).exists()]
    total = 0
    for path in iter_source_files(roots):
        gaps = find_gaps(path)
        if not gaps:
            continue
        total += len(gaps)
        rel = path.relative_to(REPO_ROOT)
        if args.summary:
            print(f"{len(gaps):4d}  {rel}")
        else:
            for line, name, problem in gaps:
                print(f"{rel}:{line}: {name}: {problem}")
    print(f"TOTAL gaps: {total}", file=sys.stderr)


if __name__ == "__main__":
    main()
