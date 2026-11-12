# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension that appends a ``Source: path:lines`` link to every sphinx-autoapi documented object.

The path and line range are found by parsing the source with ``ast``; nothing is imported, so the build does not
need Isaac Sim. The link points at the file on GitHub (config value ``source_link_base_url`` + the path + a line
anchor), and the visible text is the repository-relative ``path:line`` so it is also useful in text builds.
"""

import ast
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _find_module_file(repo_root: Path, parts: list[str]) -> tuple[Path | None, int]:
    """Return the file of the longest module prefix of ``parts`` and the number of parts it consumed."""
    for n in range(len(parts), 0, -1):
        base = repo_root.joinpath(*parts[:n])
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py"), n
        if (base / "__init__.py").is_file():
            return base / "__init__.py", n
    return None, 0


def _find_lines(tree: ast.Module, qualname_parts: list[str]) -> tuple[int, int] | None:
    """Return the 1-based line range of a (possibly nested) class, function or assignment, or None."""
    body = tree.body
    node = None
    for part in qualname_parts:
        node = None
        for candidate in body:
            if isinstance(candidate, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if candidate.name == part:
                    node = candidate
            elif isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                targets = candidate.targets if isinstance(candidate, ast.Assign) else [candidate.target]
                if any(isinstance(t, ast.Name) and t.id == part for t in targets):
                    node = candidate
        if node is None:
            return None
        body = getattr(node, "body", [])
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list)
    return start, node.end_lineno


def _add_source_line(app, what, name, obj, skip, options):
    """Append the source link to the object's docstring (idempotent) and leave the skip decision unchanged."""
    if "\nSource: `" in obj.docstring:
        return
    repo_root = Path(app.config.source_link_repo_root).resolve()
    parts = name.split(".")
    file, consumed = _find_module_file(repo_root, parts)
    if file is None:
        return
    rel = file.relative_to(repo_root).as_posix()
    qualname_parts = parts[consumed:]
    span = _find_lines(_parse(file), qualname_parts) if qualname_parts else None
    if qualname_parts and span is None:
        return
    if span:
        label, anchor = f"{rel}:{span[0]}-{span[1]}", f"#L{span[0]}-L{span[1]}"
    else:
        label, anchor = rel, ""
    url = f"{app.config.source_link_base_url.rstrip('/')}/{rel}{anchor}"
    obj.docstring = f"{obj.docstring}\n\nSource: `{label} <{url}>`__"
    return


def setup(app):
    app.add_config_value("source_link_repo_root", ".", "env")
    app.add_config_value("source_link_base_url", "", "env")
    app.connect("autoapi-skip-member", _add_source_line)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
