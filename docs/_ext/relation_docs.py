# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension that adds graph YAML details to autodoc'd relation classes and checks their docstrings.

For each registered relation documented with ``autoclass``, the YAML ``kind``, its aliases and whether the
relation takes a ``reference`` are added below the summary line. The build warns when a relation lacks a
"Validation:" section, leaves a constructor parameter out of "Args:", or is not documented on any page.
"""

import inspect
from typing import Any

from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util import logging

logger = logging.getLogger(__name__)


def _get_relation_names_by_class() -> dict[type, list[str]]:
    """Return every registered relation class with the names it is registered under, primary name first."""
    import isaaclab_arena.relations.relations  # noqa: F401 - importing the module registers the relations.
    from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry

    registry = ObjectRelationLibraryRegistry()
    names_by_class: dict[type, list[str]] = {}
    for name in registry.get_all_keys():
        names_by_class.setdefault(registry.get_object_relation_by_name(name), []).append(name)
    for relation_cls, names in names_by_class.items():
        names.sort(key=lambda name: name != relation_cls.name)
    return names_by_class


def _get_yaml_lines(relation_cls: type, names: list[str]) -> list[str]:
    kind, *aliases = names
    kind_line = f"**YAML kind:** ``{kind}``"
    if aliases:
        kind_line += ", also accepted as " + ", ".join(f"``{alias}``" for alias in aliases)
    if relation_cls.is_unary():
        arity_line = "Unary: the YAML entry has no ``reference``."
    else:
        arity_line = "Binary: the YAML entry's ``reference`` is passed as ``parent``."
    return [kind_line + ". " + arity_line, ""]


def _check_relation_docstring(app: Sphinx, relation_cls: type, class_lines: list[str]) -> None:
    location = f"{relation_cls.__module__}.{relation_cls.__qualname__}"
    if ".. rubric:: Validation" not in class_lines:
        logger.warning("Relation %s's docstring needs a 'Validation:' section.", location)
    init = relation_cls.__init__
    if init is object.__init__:
        return
    init_lines = GoogleDocstring(inspect.getdoc(init) or "", app.config, app, "method").lines()
    for name, parameter in inspect.signature(init).parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if not any(line.startswith(f":param {name}:") for line in init_lines):
            logger.warning("Relation %s does not describe parameter '%s' under 'Args:'.", location, name)


def add_relation_yaml_details(app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]) -> None:
    """Add a relation's YAML kind and arity below its summary line and check its docstring.

    Connected to ``autodoc-process-docstring`` after napoleon, so ``lines`` holds reStructuredText.
    With ``autoclass_content = "both"`` autodoc emits the event for the class docstring and again for
    the ``__init__`` docstring, both with the class as ``obj``; only the class docstring is changed.
    """
    if what != "class":
        return
    names = _get_relation_names_by_class().get(obj)
    if names is None:
        return
    class_summary = (inspect.getdoc(obj) or "").split("\n", 1)[0]
    if not lines or lines[0] != class_summary:
        return
    _check_relation_docstring(app, obj, lines)
    summary_end = lines.index("") + 1 if "" in lines else len(lines)
    lines[summary_end:summary_end] = _get_yaml_lines(obj, names)


def check_all_relations_documented(app: Sphinx, env: BuildEnvironment) -> None:
    """Warn for each registered relation that no page documents."""
    documented = env.get_domain("py").objects
    for relation_cls in _get_relation_names_by_class():
        full_name = f"{relation_cls.__module__}.{relation_cls.__qualname__}"
        if full_name not in documented:
            logger.warning("Relation %s is registered but no page documents it with autoclass.", full_name)


def setup(app: Sphinx) -> dict[str, Any]:
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinx.ext.napoleon")
    # Napoleon converts the docstring at the default priority of 500; run after it.
    app.connect("autodoc-process-docstring", add_relation_yaml_details, priority=600)
    app.connect("env-check-consistency", check_all_relations_documented)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
