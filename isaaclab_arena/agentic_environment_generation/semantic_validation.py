# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Structured static checks against the exact authoring catalogues offered to an agent."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import ValidationError

from isaaclab_arena.agentic_environment_generation.authoring_metadata import object_type_capabilities
from isaaclab_arena.agentic_environment_generation.catalogues import (
    AssetCatalogue,
    RelationCatalogue,
    TaskCatalogue,
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable authoring error with a JSON Pointer into the submitted graph."""

    code: str
    path: str
    message: str
    expected: Any = None
    compatible_choices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the issue as JSON-compatible data."""
        return asdict(self)


def collect_semantic_validation_issues(
    data: dict[str, Any],
    asset_catalog: AssetCatalogue,
    task_catalog: TaskCatalogue,
    relation_catalog: RelationCatalogue,
) -> list[ValidationIssue]:
    """Check registry choices, parameter shapes and declared capability contracts without a build.

    Args:
        data: Environment graph mapping, including mappings that fail structural validation.
        asset_catalog: Exact available asset entries.
        task_catalog: Exact available task entries, normally restricted to ``@agent_ready``.
        relation_catalog: Exact available spatial relations.

    Returns:
        Violations with paths, expected schemas or capabilities, and compatible choices.
    """
    issues = []
    nodes: dict[str, set[str]] = {}
    groups = (
        ("embodiment", asset_catalog.embodiments),
        ("background", asset_catalog.backgrounds),
        ("objects", asset_catalog.objects),
    )
    object_entries = {entry["name"]: entry for entry in asset_catalog.objects}
    for group, entries in groups:
        vocabulary = {entry["name"]: entry for entry in entries}
        values = data.get(group, []) if group == "objects" else [data.get(group)]
        if not isinstance(values, list):
            continue
        for index, asset in enumerate(values):
            if not isinstance(asset, dict):
                continue
            path = f"/{group}/{index}" if group == "objects" else f"/{group}"
            name = asset.get("registry_name")
            entry = vocabulary.get(name) if isinstance(name, str) else None
            node_id = asset.get("id")
            if isinstance(node_id, str):
                nodes[node_id] = _asset_capabilities(entry)
            label = f"Object {node_id!r}" if group == "objects" else group.capitalize()
            catalogue_name = "OBJECTS" if group == "objects" else f"{group.upper()}S"
            if entry is None:
                issues.append(
                    ValidationIssue(
                        "unknown_asset",
                        path + "/registry_name",
                        f"{label} registry_name {name!r} is not in the {catalogue_name} catalog",
                        compatible_choices=sorted(vocabulary),
                    )
                )
            else:
                issues.extend(_parameter_issues(asset.get("params", {}), entry, path + "/params", label))
    for index, object_set in enumerate(_list_value(data.get("object_sets"))):
        if not isinstance(object_set, dict):
            continue
        capabilities = []
        for member_index, member in enumerate(_list_value(object_set.get("members"))):
            entry = object_entries.get(member) if isinstance(member, str) else None
            if entry is None:
                issues.append(
                    ValidationIssue(
                        "unknown_asset",
                        f"/object_sets/{index}/members/{member_index}",
                        f"Object set {object_set.get('id')!r} member registry_name {member!r} is not in the OBJECTS"
                        " catalog",
                        compatible_choices=sorted(object_entries),
                    )
                )
            capabilities.append(_asset_capabilities(entry))
        if isinstance(object_set.get("id"), str):
            nodes[object_set["id"]] = set.intersection(*capabilities) if capabilities else set()
    references = _list_value(data.get("object_references"))
    if references:
        # Reuse the runtime dispatch map; never instantiate references or inspect their USD.
        from isaaclab_arena.agentic_environment_generation.authoring_metadata import provided_capabilities
        from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _AFFORDANCE_REFERENCE_CLASSES

        for index, reference in enumerate(references):
            if not isinstance(reference, dict) or not isinstance(reference.get("id"), str):
                continue
            capabilities = object_type_capabilities(reference.get("object_type"))
            reference_params = reference.get("params")
            reference_params = reference_params if isinstance(reference_params, dict) else {}
            affordance_params = set(reference_params) & set(_AFFORDANCE_REFERENCE_CLASSES)
            if len(affordance_params) > 1:
                issues.append(
                    ValidationIssue(
                        "multiple_affordances",
                        f"/object_references/{index}/params",
                        "An object reference supports one affordance; use separate references for separate joints",
                        expected={"max_affordances": 1},
                    )
                )
            if affordance_params and reference.get("object_type") != "articulation":
                issues.append(
                    ValidationIssue(
                        "affordance_object_type",
                        f"/object_references/{index}/object_type",
                        "Joint affordances require object_type=articulation",
                        compatible_choices=["articulation"],
                    )
                )
            for parameter, reference_class in _AFFORDANCE_REFERENCE_CLASSES.items():
                if parameter in reference_params:
                    capabilities.update(provided_capabilities(reference_class))
            nodes[reference["id"]] = capabilities
    task = data.get("task")
    subtasks = task.get("subtasks", []) if isinstance(task, dict) else []
    for group, values, entries in (
        ("task/subtasks", subtasks, task_catalog.tasks),
        ("relations", data.get("relations", []), relation_catalog.relations),
    ):
        if not isinstance(values, list):
            continue
        vocabulary = {entry.name: asdict(entry) for entry in entries}
        noun = "Task" if group == "task/subtasks" else "Relation"
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                continue
            path = f"/{group}/{index}"
            kind = value.get("kind")
            entry = vocabulary.get(kind) if isinstance(kind, str) else None
            if entry is None:
                issues.append(
                    ValidationIssue(
                        "unsupported_component",
                        path + "/kind",
                        f"{noun} {kind!r} is not in the {noun.upper()}S catalog",
                        compatible_choices=sorted(vocabulary),
                    )
                )
                continue
            params = value.get("params", {})
            issues.extend(_parameter_issues(params, entry, path + "/params", f"{noun} {kind!r}"))
            if isinstance(params, dict):
                for name, schema in entry["parameters"].items():
                    if name in params:
                        issues.extend(_reference_issues(params[name], schema, path + "/params/" + _escape(name), nodes))
            if noun == "Relation":
                for endpoint in ("subject", "reference"):
                    node_id = value.get(endpoint)
                    if isinstance(node_id, str) and node_id not in nodes:
                        issues.append(
                            ValidationIssue(
                                "unknown_node",
                                f"{path}/{endpoint}",
                                f"Unknown graph node {node_id!r}",
                                compatible_choices=sorted(nodes),
                            )
                        )
    return issues


def _asset_capabilities(entry: dict[str, Any] | None) -> set[str]:
    """Read declared class/factory interfaces from current or externally supplied catalogues."""
    if entry is None:
        return set()
    return set(entry.get("provides", [])) | object_type_capabilities(entry.get("object_type"))


def validate_authoring_spec(data: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable static validation report for an environment graph mapping.

    A valid report certifies schema and declared semantics only. Geometry, USD prim existence,
    runtime constraints, and policy feasibility still require the normal build and simulation.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    issues = collect_semantic_validation_issues(
        data, build_asset_catalogue(), build_task_catalogue(), build_relation_catalogue()
    )
    try:
        ArenaEnvGraphSpec.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors(include_context=False, include_input=False):
            path = "/" + "/".join(_escape(str(part)) for part in error["loc"])
            if not any(issue.path == path for issue in issues):
                issues.append(ValidationIssue("schema_error", path, error["msg"]))
    return {
        "schema_version": 1,
        "valid": not issues,
        "validation_scope": "schema_and_declared_semantics",
        "issues": [issue.to_dict() for issue in issues],
    }


def _parameter_issues(params: Any, entry: dict[str, Any], path: str, label: str) -> list[ValidationIssue]:
    if "parameters" not in entry and "required_params" not in entry:
        # Older or dynamically supplied asset catalogues may provide names only.
        return []
    if not isinstance(params, dict):
        return [
            ValidationIssue("parameter_type", path, f"{label} params must be a mapping", expected={"type": "object"})
        ]
    issues = []
    schemas = entry.get("parameters", {})
    required = entry.get("required_params", [name for name, schema in schemas.items() if schema.get("required")])
    optional = entry.get("optional_params", [name for name, schema in schemas.items() if not schema.get("required")])
    supported = set(required) | set(optional)
    for name in required:
        if name not in params:
            issues.append(
                ValidationIssue(
                    "missing_parameter",
                    path + "/" + _escape(name),
                    f"{label} is missing required param {name!r}",
                    expected=schemas.get(name),
                )
            )
    for name, value in params.items():
        param_path = path + "/" + _escape(name)
        if name not in supported and not entry.get("accepts_extra_parameters", False):
            issues.append(
                ValidationIssue(
                    "unsupported_parameter",
                    param_path,
                    f"{label} has unsupported param {name!r}; supported params are {sorted(supported)!r}",
                    compatible_choices=sorted(supported),
                )
            )
        elif name in schemas:
            issues.extend(_schema_issues(value, schemas[name], param_path))
    return issues


def _schema_issues(value: Any, schema: dict[str, Any], path: str) -> list[ValidationIssue]:
    if "anyOf" in schema:
        if not any(not _schema_issues(value, choice, path) for choice in schema["anyOf"]):
            return [
                ValidationIssue(
                    "parameter_type", path, "Value does not match any declared parameter type", expected=schema
                )
            ]
    expected_type = schema.get("type")
    matches = {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, dict),
    }
    if expected_type is not None and not matches[expected_type]:
        return [
            ValidationIssue(
                "parameter_type", path, f"Expected {expected_type}, got {type(value).__name__}", expected=schema
            )
        ]
    if "enum" in schema and (
        value not in schema["enum"]
        or isinstance(value, bool)
        and not any(isinstance(option, bool) for option in schema["enum"])
    ):
        return [
            ValidationIssue(
                "parameter_enum",
                path,
                f"Value must be one of {schema['enum']!r}",
                expected=schema,
                compatible_choices=schema["enum"],
            )
        ]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if (
            not math.isfinite(value)
            or value < schema.get("minimum", -math.inf)
            or value > schema.get("maximum", math.inf)
        ):
            return [
                ValidationIssue(
                    "parameter_range", path, "Value must be finite and within the declared bounds", expected=schema
                )
            ]
    issues = []
    if expected_type == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            return [
                ValidationIssue("parameter_shape", path, "Array length is outside the declared shape", expected=schema)
            ]
        for index, item in enumerate(value):
            item_schema = schema["prefixItems"][index] if "prefixItems" in schema else schema.get("items", {})
            issues.extend(_schema_issues(item, item_schema, f"{path}/{index}"))
    if expected_type == "object" and isinstance(schema.get("additionalProperties"), dict):
        for name, item in value.items():
            issues.extend(_schema_issues(item, schema["additionalProperties"], path + "/" + _escape(name)))
    return issues


def _reference_issues(
    value: Any, schema: dict[str, Any], path: str, nodes: dict[str, set[str]]
) -> list[ValidationIssue]:
    if "anyOf" in schema and not schema.get("x-arena-reference"):
        for branch in schema["anyOf"]:
            if not _schema_issues(value, branch, path):
                return _reference_issues(value, branch, path, nodes)
    if schema.get("type") == "array" and isinstance(value, (list, tuple)):
        issues = []
        for index, item in enumerate(value):
            issues.extend(_reference_issues(item, schema.get("items", {}), f"{path}/{index}", nodes))
        return issues
    if not schema.get("x-arena-reference") or value is None or not isinstance(value, str):
        return []
    requirements = set(schema.get("x-required-capabilities", []))
    compatible = sorted(name for name, capabilities in nodes.items() if requirements <= capabilities)
    if value not in nodes:
        return [
            ValidationIssue(
                "unknown_node",
                path,
                f"Unknown graph node {value!r}",
                expected={"capabilities": sorted(requirements)},
                compatible_choices=compatible,
            )
        ]
    if not requirements <= nodes[value]:
        return [
            ValidationIssue(
                "missing_capability",
                path,
                f"Node {value!r} lacks required capabilities {sorted(requirements - nodes[value])!r}",
                expected={"capabilities": sorted(requirements)},
                compatible_choices=compatible,
            )
        ]
    return []


def _escape(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _list_value(value: Any) -> list:
    return value if isinstance(value, list) else []
