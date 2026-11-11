# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension that adds each autodoc'd relation's "YAML kind" and checks the relation docstrings."""

import inspect
from typing import Any

from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util import logging

logger = logging.getLogger(__name__)

_SECTION_RUBRICS = (".. rubric:: Validation", ".. rubric:: Restrictions")

_sections_by_class: dict[type, list[str]] = {}
"""Validation and Restrictions lines cut from each class docstring, until its ``__init__`` docstring arrives."""


def _get_relation_names_by_class() -> dict[type, list[str]]:
    """Return every registered relation class with its registered names, primary name first."""
    from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry

    registry = ObjectRelationLibraryRegistry()
    names_by_class: dict[type, list[str]] = {}
    for name in registry.get_all_keys():
        names_by_class.setdefault(registry.get_object_relation_by_name(name), []).append(name)
    for relation_cls, names in names_by_class.items():
        names.sort(key=lambda name: name != relation_cls.name)
    return names_by_class


def _check_docstring(app: Sphinx, relation_cls: type, class_lines: list[str]) -> None:
    if ".. rubric:: Validation" not in class_lines:
        logger.warning("Relation %s needs a 'Validation:' section.", relation_cls.__name__)
    if relation_cls.__init__ is object.__init__:
        return
    init_lines = GoogleDocstring(inspect.getdoc(relation_cls.__init__) or "", app.config, app, "method").lines()
    for name in list(inspect.signature(relation_cls.__init__).parameters)[1:]:
        if not any(line.startswith(f":param {name}:") for line in init_lines):
            logger.warning("Relation %s does not describe '%s' under 'Args:'.", relation_cls.__name__, name)


def _cut_sections(lines: list[str]) -> list[str]:
    """Remove the Validation and Restrictions sections, which end a relation docstring, and return them."""
    start = min((lines.index(rubric) for rubric in _SECTION_RUBRICS if rubric in lines), default=len(lines))
    sections = lines[start:]
    del lines[start:]
    return sections


def process_relation_docstring(app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]) -> None:
    """Add a relation's YAML kind, check its docstring, and move its Validation and Restrictions below its parameters.

    Runs on ``autodoc-process-docstring`` after napoleon. With ``autoclass_content = "both"`` the event fires
    for the class docstring and then the ``__init__`` docstring, both with the class as ``obj``.
    """
    names = _get_relation_names_by_class().get(obj) if what == "class" else None
    if names is None or not lines:
        return
    if lines[0] != inspect.getdoc(obj).split("\n", 1)[0]:
        sections = _sections_by_class.pop(obj, [])
        if sections and lines[-1] != "":
            lines.append("")
        lines.extend(sections)
        return
    _check_docstring(app, obj, lines)
    kind, *aliases = names
    text = f"**YAML kind:** ``{kind}``" + "".join(f", or ``{alias}``" for alias in aliases)
    summary_end = lines.index("") + 1 if "" in lines else len(lines)
    lines[summary_end:summary_end] = [text, ""]
    # Without an __init__ docstring there are no parameters, so the sections stay in place.
    if obj.__init__ is not object.__init__ and inspect.getdoc(obj.__init__):
        _sections_by_class[obj] = _cut_sections(lines)


def check_all_relations_documented(app: Sphinx, env: BuildEnvironment) -> None:
    """Warn for each registered relation that no page documents.

    Skipped when no page documents any relation, as in older versions built by sphinx-multiversion.
    """
    documented_names = env.get_domain("py").objects
    if not any(name.startswith("isaaclab_arena.relations.relations.") for name in documented_names):
        return
    for relation_cls in _get_relation_names_by_class():
        if f"{relation_cls.__module__}.{relation_cls.__qualname__}" not in documented_names:
            logger.warning("Relation %s is registered but not documented with autoclass.", relation_cls.__name__)


def setup(app: Sphinx) -> dict[str, Any]:
    # Napoleon converts docstrings at the default priority of 500; run after it.
    app.connect("autodoc-process-docstring", process_relation_docstring, priority=600)
    app.connect("env-check-consistency", check_all_relations_documented)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
