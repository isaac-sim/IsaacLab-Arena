# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent prompt catalogues built from the live asset, relation, and task registries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from isaaclab_arena.agentic_environment_generation.authoring_metadata import (
    accepts_extra_parameters,
    constructor_parameters,
    get_authoring_metadata,
    provided_capabilities,
)
from isaaclab_arena.assets.registries import (
    AssetRegistry,
    EnvironmentRegistry,
    ObjectRelationLibraryRegistry,
    TaskRegistry,
)
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

    def to_dict(self) -> dict[str, Any]:
        """Return machine-readable metadata without constructing registered assets."""
        return asdict(self)

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
    for name in sorted(registry.get_all_keys()):
        cls = registry.get_asset_by_name(name)
        tags = getattr(cls, "tags", None) or []
        # TODO(xinjieyao): Support agentic environment generation consuming procedural assets.
        if "procedural" in tags:
            continue
        metadata = _component_metadata(cls)
        if "embodiment" in tags:
            catalogue.embodiments.append({"name": name, "tags": [t for t in tags if t != "embodiment"], **metadata})
        elif "background" in tags:
            catalogue.backgrounds.append({"name": name, "tags": [t for t in tags if t != "background"], **metadata})
        # Only assets existed in the catalogue are exposed.
        elif "object" in tags:
            # Exposed so the agent can honour type constraints, e.g. object-set members must be rigid.
            object_type = getattr(cls, "object_type", None)
            catalogue.objects.append({
                "name": name,
                "tags": [t for t in tags if t != "object"],
                "object_type": object_type.value if object_type else "unknown",
                **metadata,
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
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    provides: list[str] = field(default_factory=list)
    requires: dict[str, list[str]] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    reset_semantics: str | None = None
    accepts_extra_parameters: bool = False


@dataclass
class RelationCatalogue:
    """Registered object-relation vocabulary for the agent prompt."""

    relations: list[RelationCatalogueEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return machine-readable relation metadata."""
        return asdict(self)

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
        metadata = _component_metadata(relation_cls, _RELATION_CATALOGUE_EXCLUDED_PARAMS)
        required_params, optional_params, enum_options = _parameter_summary(metadata["parameters"])
        catalogue.relations.append(
            RelationCatalogueEntry(
                name=name,
                unary=relation_cls.is_unary(),
                required_params=required_params,
                optional_params=optional_params,
                enum_options=enum_options,
                summary=_first_docstring_line(relation_cls),
                **metadata,
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
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    provides: list[str] = field(default_factory=list)
    requires: dict[str, list[str]] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    reset_semantics: str | None = None
    accepts_extra_parameters: bool = False


@dataclass
class TaskCatalogue:
    """Agent-ready task vocabulary for the agent prompt."""

    tasks: list[TaskCatalogueEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return machine-readable metadata for the same agent-ready tasks."""
        return asdict(self)

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
        metadata = _component_metadata(task_cls, _TASK_CATALOGUE_EXCLUDED_PARAMS)
        required_params, optional_params, enum_options = _parameter_summary(metadata["parameters"])
        catalogue.tasks.append(
            TaskCatalogueEntry(
                name=name,
                required_params=required_params,
                optional_params=optional_params,
                enum_options=enum_options,
                summary=_first_docstring_line(task_cls),
                **metadata,
            )
        )
    return catalogue


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _component_metadata(component: Any, excluded: frozenset[str] = frozenset()) -> dict[str, Any]:
    metadata = get_authoring_metadata(component)
    parameters = constructor_parameters(component, excluded)
    requirements = {}
    for name, schema in parameters.items():
        if schema.get("x-required-capabilities"):
            requirements[name] = schema["x-required-capabilities"]
    return {
        "parameters": parameters,
        "provides": provided_capabilities(component),
        "requires": requirements,
        "constraints": list(metadata.constraints),
        "reset_semantics": metadata.reset_semantics,
        "accepts_extra_parameters": accepts_extra_parameters(component),
    }


def build_catalogue_dict() -> dict[str, Any]:
    """Return the live authoring catalogues as one versioned JSON-compatible mapping."""
    return {
        "schema_version": 1,
        "assets": build_asset_catalogue().to_dict(),
        "environments": build_environment_catalogue(),
        **build_relation_catalogue().to_dict(),
        **build_task_catalogue().to_dict(),
    }


def build_environment_catalogue(registry: EnvironmentRegistry | None = None) -> list[dict[str, Any]]:
    """Describe registered environment factories and their typed configs without constructing either."""
    registry = registry or EnvironmentRegistry()
    entries = []
    for name in sorted(registry.get_all_keys()):
        factory = registry.get_component_by_name(name)
        cfg_type = registry.get_environment_cfg_type(factory)
        metadata = get_authoring_metadata(factory)
        entries.append({
            "name": name,
            "summary": _first_docstring_line(factory),
            "config_type": f"{cfg_type.__module__}.{cfg_type.__qualname__}",
            "parameters": constructor_parameters(cfg_type),
            "provides": provided_capabilities(factory),
            "constraints": list(metadata.constraints),
            "reset_semantics": metadata.reset_semantics,
        })
    return entries


def _first_docstring_line(cls: type) -> str:
    doc = cls.__doc__ or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _parameter_summary(
    parameters: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Derive legacy text fields from the same parameter schemas used by JSON discovery."""
    required = [name for name, schema in parameters.items() if schema["required"]]
    optional = [name for name, schema in parameters.items() if not schema["required"]]
    enum_options = {}
    for name, schema in parameters.items():
        if options := _enum_options(schema):
            enum_options[name] = options
    return required, optional, enum_options


def _enum_options(schema: dict[str, Any]) -> list[str]:
    """Return the first finite choice set, including optional and collection parameters."""
    if "enum" in schema:
        return [str(value) for value in schema["enum"]]
    children = schema.get("anyOf", []) + schema.get("prefixItems", [])
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            children.append(schema[key])
    for child in children:
        if options := _enum_options(child):
            return options
    return []
