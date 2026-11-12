# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the code-only source variants used to measure how docstrings and names affect lookup questions.

Usage: ``python build_variants.py <out_dir> <repo_root>``. Creates ``<out_dir>/{c0,c1,c2,c3}``:

- c0: python sources with the new docstrings (commit DOCSTRING_COMMIT).
- c1: python sources with the original docstrings (commit ORIGINAL_COMMIT).
- c2: c0 with every docstring removed (comments kept).
- c3: c2 with all comments removed and the names of classes, functions, methods, module-level variables, class
  attributes and parameters replaced by opaque identifiers, consistently across files. The mapping is written to
  ``<out_dir>/obfuscation_map.json``.

Only ``*.py`` files of the first-party packages are kept (no docs, markdown, yaml or skills), so that no other file can
reveal names or semantics.
"""

import ast
import hashlib
import io
import json
import keyword
import subprocess
import sys
import tarfile
import tokenize
from pathlib import Path

ORIGINAL_COMMIT = "fa0b45dba7"
DOCSTRING_COMMIT = "2c5139b53f"
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
    "osmo",
]

# Parameter / attribute names that double as external API names (torch, gymnasium, Isaac Lab, python itself) and are
# therefore left alone so that the obfuscated code still reads as valid use of those APIs.
EXTERNAL_API_NAMES = {
    "self",
    "cls",
    "args",
    "kwargs",
    "device",
    "dtype",
    "shape",
    "env",
    "envs",
    "cfg",
    "name",
    "size",
    "dim",
    "data",
    "num_envs",
    "env_ids",
    "key",
    "value",
    "values",
    "keys",
    "items",
    "path",
    "file",
    "mode",
    "type",
    "format",
    "id",
    "index",
    "input",
    "output",
    "seed",
    "kwarg",
    "scene",
    "sim",
    "asset",
    "assets",
    "robot",
    "obs",
    "observation",
    "action",
    "actions",
    "step",
    "reset",
    "close",
    "render",
    "main",
    "setup",
    "teardown",
    "run",
    "parse",
    "get",
    "set",
    "add",
    "update",
    "build",
    "load",
    "save",
    "default",
    "config",
    "func",
    "fn",
    "callback",
    "params",
    "param",
    "result",
    "results",
    "message",
    "msg",
    "text",
    "line",
    "lines",
    "count",
    "start",
    "end",
    "min",
    "max",
    "sum",
    "mean",
    "x",
    "y",
    "z",
    "w",
    "i",
    "j",
    "k",
    "n",
    "t",
    "p",
    "q",
    "r",
    "s",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "m",
    "u",
    "v",
    "np",
    "torch",
    "os",
    "sys",
    "re",
    "json",
    "math",
    "time",
    "logging",
    "pytest",
    "simulation_app",
    "args_cli",
    "parser",
    "logger",
    "log",
    "print",
    "len",
    "range",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "object",
    "super",
    "property",
    "classmethod",
    "staticmethod",
    "abstractmethod",
    "dataclass",
    "field",
    "configclass",
    "from_dict",
    "to_dict",
    "main",
    "test",
}


def export_python_sources(repo_root: Path, commit: str, out: Path) -> None:
    """Extract the python files of the first-party packages at ``commit`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    tar_bytes = subprocess.run(
        ["git", "-C", str(repo_root), "archive", commit, *PACKAGES], capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                target = out / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(member).read())


def python_files(root: Path):
    return sorted(root.rglob("*.py"))


def strip_docstrings(source: str) -> str:
    """Remove module, class and function docstrings and attribute docstrings; keep everything else."""
    tree = ast.parse(source)
    lines = source.split("\n")
    edits: list[tuple[int, int, str | None]] = []  # (first line, last line, replacement indent or None to delete)

    def is_doc(stmt):
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)

    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for field_body in [body, getattr(node, "orelse", []), getattr(node, "finalbody", [])]:
            if not isinstance(field_body, list):
                continue
            docs = [s for s in field_body if is_doc(s)]
            for stmt in docs:
                only_statement = len(field_body) == len(docs) and len(docs) == 1
                if only_statement and not isinstance(node, ast.Module):
                    indent = " " * stmt.col_offset
                    edits.append((stmt.lineno, stmt.end_lineno, indent + "pass"))
                else:
                    edits.append((stmt.lineno, stmt.end_lineno, None))
    for first, last, replacement in sorted(set(edits), reverse=True):
        lines[first - 1 : last] = [] if replacement is None else [replacement]
    out = "\n".join(lines)
    ast.parse(out)
    return out


def remove_comments(source: str) -> str:
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = source.split("\n")
    for tok in sorted((t for t in tokens if t.type == tokenize.COMMENT), key=lambda t: t.start, reverse=True):
        row, col = tok.start
        lines[row - 1] = lines[row - 1][:col].rstrip()
    return "\n".join(lines)


def collect_definitions(files: list[Path]) -> dict[str, str]:
    """Return {original name: kind} for every name that is to be renamed."""
    module_names = set()
    for f in files:
        module_names.add(f.stem)
        module_names.update(p for p in f.parent.parts)
    imported = set()
    defined: dict[str, str] = {}

    def add(name, kind):
        if name.startswith("__") and name.endswith("__"):
            return
        if name in EXTERNAL_API_NAMES or keyword.iskeyword(name) or len(name) <= 1:
            return
        defined.setdefault(name, kind)

    for f in files:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                first_party = node.level > 0 if isinstance(node, ast.ImportFrom) else False
                if module and module.split(".")[0] in PACKAGES:
                    first_party = True
                if not first_party:
                    # Names imported from external libraries keep their name; first-party imports are renamed.
                    for alias in node.names:
                        imported.update((alias.asname or alias.name).split("."))
            elif isinstance(node, ast.ClassDef):
                add(node.name, "C")
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        add(stmt.target.id, "a")
                    elif isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            if isinstance(t, ast.Name):
                                add(t.id, "a")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node.name, "f")
                for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    add(a.arg, "p")
                if node.args.vararg:
                    add(node.args.vararg.arg, "p")
                if node.args.kwarg:
                    add(node.args.kwarg.arg, "p")
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        add(t.id, "v")
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                add(stmt.target.id, "v")
    for name in list(defined):
        if name in module_names or name in imported:
            del defined[name]
    return defined


def make_mapping(defined: dict[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name, kind in sorted(defined.items()):
        digest = hashlib.sha1(name.encode()).hexdigest()
        length = 6
        while True:
            candidate = f"{kind}_{digest[:length]}"
            if candidate not in used:
                break
            length += 1
        used.add(candidate)
        mapping[name] = candidate
    return mapping


def rename_tokens(source: str, mapping: dict[str, str]) -> str:
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = source.split("\n")
    hits = [t for t in tokens if t.type == tokenize.NAME and t.string in mapping]
    for tok in sorted(hits, key=lambda t: t.start, reverse=True):
        row, col = tok.start
        line = lines[row - 1]
        lines[row - 1] = line[:col] + mapping[tok.string] + line[col + len(tok.string) :]
    out = "\n".join(lines)
    ast.parse(out)
    return out


def main(out_dir: Path, repo_root: Path) -> None:
    c0, c1, c2, c3 = (out_dir / n for n in ("c0", "c1", "c2", "c3"))
    export_python_sources(repo_root, DOCSTRING_COMMIT, c0)
    export_python_sources(repo_root, ORIGINAL_COMMIT, c1)
    for f in python_files(c0):
        rel = f.relative_to(c0)
        stripped = strip_docstrings(f.read_text())
        (c2 / rel).parent.mkdir(parents=True, exist_ok=True)
        (c2 / rel).write_text(stripped)
    files = python_files(c2)
    mapping = make_mapping(collect_definitions(files))
    for f in files:
        rel = f.relative_to(c2)
        text = remove_comments(rename_tokens(f.read_text(), mapping))
        ast.parse(text)
        (c3 / rel).parent.mkdir(parents=True, exist_ok=True)
        (c3 / rel).write_text(text)
    (out_dir / "obfuscation_map.json").write_text(json.dumps(mapping, indent=1, sort_keys=True))
    print({n: len(python_files(out_dir / n)) for n in ("c0", "c1", "c2", "c3")}, "renamed names:", len(mapping))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
