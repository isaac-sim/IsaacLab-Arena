# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent prompt catalogues built from the live asset, relation, and task registries."""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, get_args, get_type_hints

from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry, TaskRegistry
from isaaclab_arena.relations.relations import RelationBase

# Constructor kwargs already expressed as top-level ArenaEnvGraphTypes fields (not as
# TaskSpec.params / SpatialRelationSpec.params). Keep them out of the agent catalogues.
_TASK_CATALOGUE_EXCLUDED_PARAMS = frozenset({"task_description"})  # CompositeTaskSpec.description
_RELATION_CATALOGUE_EXCLUDED_PARAMS = frozenset({"parent"})  # SpatialRelationSpec.reference


# ---------------------------------------------------------------------------
# Asset catalogue (AssetRegistry → user-prompt blocks)
# ---------------------------------------------------------------------------


@dataclass
class AssetCatalogue:
    """Registered asset vocabulary grouped for the agent prompt."""

    # A list of embodiment names and their tags for agent to choose from.
    embodiments: list[dict[str, Any]] = field(default_factory=list)
    # A list of background names and their tags for agent to choose from.
    backgrounds: list[dict[str, Any]] = field(default_factory=list)
    # A list of object names, object types, and tags for agent to choose from.
    objects: list[dict[str, Any]] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message vocabulary block."""
        embodiment_lines = "\n".join(
            f"- {e['name']}  tags={e['tags']}" for e in sorted(self.embodiments, key=lambda e: e["name"])
        )
        background_lines = "\n".join(
            f"- {b['name']}  tags={b['tags']}" for b in sorted(self.backgrounds, key=lambda b: b["name"])
        )
        object_lines = "\n".join(
            f"- {o['name']}  type={o['object_type']}  tags={o['tags']}"
            for o in sorted(self.objects, key=lambda o: o["name"])
        )
        return (
            f"EMBODIMENTS ({len(self.embodiments)}):\n{embodiment_lines}\n\n"
            f"BACKGROUNDS ({len(self.backgrounds)}):\n{background_lines}\n\n"
            f"OBJECTS ({len(self.objects)}):\n{object_lines}"
        )


def build_asset_catalogue(registry: AssetRegistry | None = None) -> AssetCatalogue:
    """Collect registered embodiments, backgrounds, and pick-up objects from ``AssetRegistry``."""
    registry = registry or AssetRegistry()
    catalogue = AssetCatalogue()
    # TODO(qianl): handle optional lights and hdr images.
    # TODO(qianl): add tag to filter out validated/agent-ready assets only.
    # Classify by registry tags, not issubclass(Background/Object/EmbodimentBase): importing those
    # types pulls in pxr before SimulationApp and breaks unit tests.
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        tags = getattr(cls, "tags", None) or []
        if "embodiment" in tags:
            catalogue.embodiments.append({"name": name, "tags": [t for t in tags if t != "embodiment"]})
        elif "background" in tags:
            catalogue.backgrounds.append({"name": name, "tags": [t for t in tags if t != "background"]})
        # Only assets existed in the catalogue are exposed.
        elif "object" in tags:
            # Exposed so the agent can honour type constraints, e.g. object-set members must be rigid.
            object_type = getattr(cls, "object_type", None)
            catalogue.objects.append({
                "name": name,
                "tags": [t for t in tags if t != "object"],
                "object_type": object_type.value if object_type else "unknown",
            })
    return catalogue


# ---------------------------------------------------------------------------
# Relation catalogue (ObjectRelationLibraryRegistry → user-prompt blocks)
# ---------------------------------------------------------------------------


@dataclass
class RelationCatalogueEntry:
    """One registered spatial relation exposed to the agent."""

    name: str
    unary: bool
    required_params: list[str]
    optional_params: list[str]
    enum_options: dict[str, list[str]]
    summary: str


@dataclass
class RelationCatalogue:
    """Registered object-relation vocabulary for the agent prompt."""

    relations: list[RelationCatalogueEntry] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message RELATIONS block."""
        lines = []
        for entry in sorted(self.relations, key=lambda r: r.name):
            arity = "unary" if entry.unary else "binary"

            def _format_param(name: str) -> str:
                options = entry.enum_options.get(name)
                return f"{name}={{{', '.join(options)}}}" if options else name

            required = ", ".join(_format_param(name) for name in entry.required_params)
            optional = ", ".join(_format_param(name) for name in entry.optional_params)
            params = f"required: {required or 'none'}; optional: {optional or 'none'}"
            lines.append(f"- {entry.name} ({arity}; {params}): {entry.summary}")
        return f"RELATIONS ({len(self.relations)}):\n" + "\n".join(lines)


def build_relation_catalogue(
    registry: ObjectRelationLibraryRegistry | None = None,
) -> RelationCatalogue:
    """Collect agent-ready object relations from ``ObjectRelationLibraryRegistry``."""
    registry = registry or ObjectRelationLibraryRegistry()
    catalogue = RelationCatalogue()
    for name in registry.get_all_keys():
        relation_cls = registry.get_object_relation_by_name(name)
        assert issubclass(relation_cls, RelationBase), f"{name!r} is not a RelationBase subclass"
        if not getattr(relation_cls, "agent_ready", False):
            continue
        required_params, optional_params, enum_options = _collect_init_params(
            relation_cls,
            excluded_params=set(_RELATION_CATALOGUE_EXCLUDED_PARAMS),
        )
        catalogue.relations.append(
            RelationCatalogueEntry(
                name=name,
                unary=relation_cls.is_unary(),
                required_params=required_params,
                optional_params=optional_params,
                enum_options=enum_options,
                summary=_first_docstring_line(relation_cls),
            )
        )
    return catalogue


# ---------------------------------------------------------------------------
# Task catalogue (TaskRegistry → user-prompt blocks)
# ---------------------------------------------------------------------------


@dataclass
class TaskCatalogueEntry:
    """One agent_ready task exposed to the agent."""

    name: str
    required_params: list[str]
    optional_params: list[str]
    enum_options: dict[str, list[str]]
    summary: str


@dataclass
class TaskCatalogue:
    """Agent-ready task vocabulary for the agent prompt."""

    tasks: list[TaskCatalogueEntry] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message TASKS block."""
        lines = []
        for entry in sorted(self.tasks, key=lambda t: t.name):

            def _format_param(name: str) -> str:
                options = entry.enum_options.get(name)
                return f"{name}={{{', '.join(options)}}}" if options else name

            required = ", ".join(_format_param(name) for name in entry.required_params)
            optional = ", ".join(_format_param(name) for name in entry.optional_params)
            params = f"required: {required or 'none'}; optional: {optional or 'none'}"
            lines.append(f"- {entry.name} ({params}): {entry.summary}")
        return f"TASKS ({len(self.tasks)}):\n" + "\n".join(lines)


def agent_ready_task_names(registry: TaskRegistry | None = None) -> frozenset[str]:
    """Return ``TaskRegistry`` keys for tasks marked with ``@agent_ready``."""
    registry = registry or TaskRegistry()
    return frozenset(
        name for name in registry.get_all_keys() if getattr(registry.get_task_by_name(name), "agent_ready", False)
    )


def build_task_catalogue(registry: TaskRegistry | None = None) -> TaskCatalogue:
    """Collect agent_ready tasks from ``TaskRegistry``."""
    registry = registry or TaskRegistry()
    catalogue = TaskCatalogue()
    for name in sorted(agent_ready_task_names(registry)):
        task_cls = registry.get_task_by_name(name)
        required_params, optional_params, enum_options = _collect_init_params(
            task_cls,
            excluded_params=set(_TASK_CATALOGUE_EXCLUDED_PARAMS),
        )
        catalogue.tasks.append(
            TaskCatalogueEntry(
                name=name,
                required_params=required_params,
                optional_params=optional_params,
                enum_options=enum_options,
                summary=_first_docstring_line(task_cls),
            )
        )
    return catalogue


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _first_docstring_line(cls: type) -> str:
    doc = cls.__doc__ or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _collect_init_params(
    cls: type,
    excluded_params: set[str],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Collect required, optional, and Enum-valued constructor parameters."""
    signature = inspect.signature(cls.__init__)
    params = {
        name: param
        for name, param in signature.parameters.items()
        if name != "self"
        and name not in excluded_params
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    required = [name for name, param in params.items() if param.default is inspect.Parameter.empty]
    optional = [name for name, param in params.items() if param.default is not inspect.Parameter.empty]
    try:
        type_hints = get_type_hints(cls.__init__)
    except (NameError, TypeError):
        module_globals = vars(sys.modules[cls.__module__])
        type_hints = {}
        for name, param in params.items():
            type_hints[name] = _resolve_annotation(param.annotation, module_globals)
    enum_options = {
        name: [str(member.value) for member in enum_type]
        for name, annotation in type_hints.items()
        if name in params and (enum_type := _find_enum_type(annotation)) is not None
    }
    return required, optional, enum_options


def _find_enum_type(annotation: Any) -> type[Enum] | None:
    """Return the Enum type contained in an annotation, including union annotations."""
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    for argument in get_args(annotation):
        if (enum_type := _find_enum_type(argument)) is not None:
            return enum_type
    return None


def _resolve_annotation(annotation: Any, module_globals: dict[str, Any]) -> Any:
    """Resolve one trusted internal string annotation when its names are available."""
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, module_globals)  # noqa: S307 — trusted internal annotations
    except (NameError, SyntaxError, TypeError):
        return annotation
