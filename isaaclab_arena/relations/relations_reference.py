# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the relations reference page of the docs from the registered relation classes.

Each relation's docstring is the source, parsed as a Google-style docstring: its first line is the
summary, a "Validation:" section names the check that enforces it, an optional "Restrictions:"
section lists what it cannot be combined with, and an optional "Usage:" section holds a Python
example. Parameters, types and defaults come from ``__init__``, and their descriptions from its
"Args:" section.

Regenerate the page after changing a relation:

    python -m isaaclab_arena.relations.relations_reference
"""

from __future__ import annotations

import inspect
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from docstring_parser import Docstring
from docstring_parser.google import DEFAULT_SECTIONS, GoogleParser, Section, SectionType

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "docs/pages/concepts/object_placement/relations_reference.rst"
"""Checked-in page that this module generates."""

_DOCSTRING_PARSER = GoogleParser(
    sections=[
        *DEFAULT_SECTIONS,
        Section("Validation", "validation", SectionType.SINGULAR),
        Section("Restrictions", "restrictions", SectionType.SINGULAR),
        Section("Usage", "usage", SectionType.SINGULAR),
    ]
)

# Parameters that a graph YAML entry sets outside ``params``: ``parent`` is the entry's ``reference``.
_EXCLUDED_PARAMS = frozenset({"self", "parent"})


@dataclass
class RelationParameter:
    """One constructor parameter of a relation, as a graph YAML ``params`` key."""

    name: str
    type_name: str
    default: str | None
    """Python repr of the default, or None when the parameter is required."""
    description: str
    options: list[str] = field(default_factory=list)
    """Accepted values when the parameter is an enum."""


@dataclass
class RelationReferenceEntry:
    """Everything the reference page shows for one relation."""

    kind: str
    class_name: str
    aliases: list[str]
    unary: bool
    summary: str
    description: list[str]
    """Paragraphs between the summary and the Validation paragraph."""
    validation: str
    restrictions: str
    usage: str
    parameters: list[RelationParameter]

    @property
    def validation_check(self) -> str:
        """The check named in the Validation paragraph, or "none"."""
        match = re.search(r"``(\w+)`` check", self.validation)
        return match.group(1) if match else "none"


def build_relation_reference_entries() -> list[RelationReferenceEntry]:
    """Return one entry per registered relation class, sorted by YAML kind."""
    from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry

    registry = ObjectRelationLibraryRegistry()
    names_by_class: dict[type, list[str]] = {}
    for name in registry.get_all_keys():
        names_by_class.setdefault(registry.get_object_relation_by_name(name), []).append(name)
    entries = [
        _build_entry(relation_cls, [name for name in names if name != relation_cls.name])
        for relation_cls, names in names_by_class.items()
    ]
    return sorted(entries, key=lambda entry: entry.kind)


def _build_entry(relation_cls: type, aliases: list[str]) -> RelationReferenceEntry:
    docstring = _parse_docstring(relation_cls.__doc__)
    assert docstring.short_description, f"{relation_cls.__name__} needs a docstring."
    sections = {meta.args[0]: meta.description or "" for meta in docstring.meta}
    assert sections.get("validation"), f"{relation_cls.__name__}'s docstring needs a 'Validation:' section."
    return RelationReferenceEntry(
        kind=relation_cls.name,
        class_name=relation_cls.__name__,
        aliases=aliases,
        unary=relation_cls.is_unary(),
        summary=docstring.short_description,
        description=docstring.long_description.split("\n\n") if docstring.long_description else [],
        validation=sections["validation"],
        restrictions=sections.get("restrictions", ""),
        usage=sections.get("usage", ""),
        parameters=_collect_parameters(relation_cls),
    )


def _parse_docstring(docstring: str | None) -> Docstring:
    """Parse a raw ``__doc__`` as a Google-style docstring with the relation sections.

    The raw string is passed rather than ``inspect.getdoc``'s: the parser rejects a cleaned docstring
    that starts with a section heading, as an ``__init__`` docstring with only "Args:" does.
    """
    return _DOCSTRING_PARSER.parse(docstring or "")


def _collect_parameters(relation_cls: type) -> list[RelationParameter]:
    init = relation_cls.__init__
    if init is object.__init__:
        return []
    descriptions = {
        param.arg_name: " ".join((param.description or "").split()) for param in _parse_docstring(init.__doc__).params
    }
    module_globals = vars(sys.modules[relation_cls.__module__])
    parameters = []
    for name, parameter in inspect.signature(init).parameters.items():
        if name in _EXCLUDED_PARAMS or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        options = []
        for candidate in _get_annotation_members(parameter.annotation, module_globals):
            if isinstance(candidate, type) and issubclass(candidate, Enum):
                options = [str(member.value) for member in candidate]
        parameters.append(
            RelationParameter(
                name=name,
                type_name=str(parameter.annotation).replace("'", ""),
                default=None if parameter.default is parameter.empty else _format_default(parameter.default),
                description=descriptions.get(name, ""),
                options=options,
            )
        )
    return parameters


def _get_annotation_members(annotation: object, module_globals: dict) -> tuple:
    """Return the types in a parameter's annotation, or nothing when it names a type-checking-only import.

    ``get_type_hints`` on the whole signature fails if any annotation does, so each one is resolved alone.
    """
    if isinstance(annotation, str):
        try:
            annotation = eval(annotation, module_globals)  # noqa: S307 - annotations come from Arena's own source.
        except (NameError, TypeError):
            return ()
    return getattr(annotation, "__args__", ()) or (annotation,)


def _format_default(value: object) -> str:
    if isinstance(value, Enum):
        return repr(value.value)
    return repr(value)


def format_relations_reference(entries: list[RelationReferenceEntry]) -> str:
    """Return the reStructuredText page for the given relations."""
    lines = [
        "..",
        "   Generated by isaaclab_arena/relations/relations_reference.py from the relation docstrings.",
        "   Do not edit by hand. After changing a relation, regenerate it with:",
        "   python -m isaaclab_arena.relations.relations_reference",
        "",
        "Relations Reference",
        "===================",
        "",
        "Every relation an environment graph YAML can use, generated from the relation classes in",
        "``isaaclab_arena/relations/relations.py``. ``kind`` is the name in a YAML ``relations`` entry.",
        "A binary relation takes a ``reference`` object; a unary one does not. Constructor arguments go",
        "under ``params``. The validation check is the pre-physics check that rejects layouts breaking",
        "the relation; a relation without one is only a solver objective, so valid layouts can miss it",
        "slightly. Every layout is also checked for overlaps by ``no_overlap``. For how to combine",
        "relations, see :doc:`relations`.",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 10 18 50",
        "",
        "   * - ``kind``",
        "     - Arity",
        "     - Validation check",
        "     - Summary",
    ]
    for entry in entries:
        lines += [
            f"   * - :ref:`{entry.kind} <relation-{entry.kind}>`",
            f"     - {'unary' if entry.unary else 'binary'}",
            f"     - ``{entry.validation_check}``" if entry.validation_check != "none" else "     - none",
            f"     - {entry.summary}",
        ]
    for entry in entries:
        lines += ["", f".. _relation-{entry.kind}:", ""]
        title = f"``{entry.kind}`` ({entry.class_name})"
        lines += [title, "-" * len(title), "", entry.summary, ""]
        for paragraph in entry.description:
            lines += [paragraph, ""]
        lines += [f"**Validation:** {entry.validation}", ""]
        if entry.restrictions:
            lines += [f"**Restrictions:** {entry.restrictions}", ""]
        if entry.aliases:
            lines += ["**Also accepted as:** " + ", ".join(f"``{alias}``" for alias in entry.aliases), ""]
        lines += _format_parameters(entry.parameters)
        lines += _format_yaml_example(entry)
        if entry.usage:
            lines += ["Python:", "", ".. code-block:: python", ""]
            lines += [f"   {line}" if line else "" for line in entry.usage.splitlines()]
            lines += [""]
    return "\n".join(lines).rstrip() + "\n"


def _format_parameters(parameters: list[RelationParameter]) -> list[str]:
    if not parameters:
        return ["Takes no parameters.", ""]
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 22 14 42",
        "",
        "   * - Parameter",
        "     - Type",
        "     - Default",
        "     - Meaning",
    ]
    for parameter in parameters:
        meaning = parameter.description
        if parameter.options:
            meaning += (" " if meaning else "") + "One of " + ", ".join(f"``{o}``" for o in parameter.options) + "."
        lines += [
            f"   * - ``{parameter.name}``",
            f"     - ``{parameter.type_name}``",
            f"     - ``{parameter.default}``" if parameter.default is not None else "     - required",
            f"     - {meaning}" if meaning else "     -",
        ]
    return lines + [""]


def _format_yaml_example(entry: RelationReferenceEntry) -> list[str]:
    lines = [
        "YAML:",
        "",
        ".. code-block:: yaml",
        "",
        "   relations:",
        f"     - kind: {entry.kind!r}" if entry.kind == "on" else f"     - kind: {entry.kind}",
        "       subject: <object>",
    ]
    if not entry.unary:
        lines.append("       reference: <object>")
    required = [parameter for parameter in entry.parameters if parameter.default is None]
    if required:
        lines.append("       params:")
        for parameter in required:
            placeholder = "|".join(parameter.options) if parameter.options else parameter.name
            lines.append(f"         {parameter.name}: <{placeholder}>")
    return lines + [""]


def main() -> None:
    REFERENCE_PATH.write_text(format_relations_reference(build_relation_reference_entries()))
    print(f"Wrote {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
