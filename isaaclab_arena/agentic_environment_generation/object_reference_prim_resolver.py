# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pass-2 LLM resolver for object_reference prim_path values."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from isaaclab_arena.agentic_environment_generation.agent_utils import build_strict_schema
from isaaclab_arena.agentic_environment_generation.query_backend import QueryBackend, StructuredOutputRequest
from isaaclab_arena.agentic_environment_generation.spec_validation import format_validation_error
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import ObjectReferenceSpec
from isaaclab_arena.utils.asset_usd import resolve_asset_usd_path

if TYPE_CHECKING:
    from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord


class ResolvedObjectReferences(BaseModel):
    """Pass-2 resolver output: resolved prim_path values for object_reference nodes."""

    object_references: list[ObjectReferenceSpec] = Field(
        description="Resolved object references with prim_path set to a relative suffix under the parent background.",
    )


class ObjectReferencePrimResolver:
    """Pass 2: resolve object_reference prim_path values against background USD."""

    def __init__(self, query_backend: QueryBackend):
        self._query_backend = query_backend
        self._schema = build_strict_schema(ResolvedObjectReferences)

    def infer(
        self,
        spec: ArenaEnvGraphSpec,
        traces: list[str],
    ) -> ArenaEnvGraphSpec | None:
        """Resolve object_reference prim_path values against the background USD prim tree.

        Args:
            spec: Pass-1 spec whose object references carry unresolved prim paths.
            traces: Accumulator for validation error lines, extended in place when the
                model output fails prim-tree validation.

        Returns:
            The spec with resolved prim paths on success, otherwise ``None`` (with
            error lines appended to ``traces``).
        """
        # Defer pxr import until call time to avoid conflict with SimulationApp.
        from isaaclab_arena.utils.usd_prim_tree import load_usd_prim_tree

        usd_path = resolve_asset_usd_path(spec.background)
        prim_tree = load_usd_prim_tree(usd_path)
        data = self._query_backend.run_json(
            StructuredOutputRequest(
                schema_name="ResolvedObjectReferences",
                schema=self._schema,
                system=self._system_prompt(),
                user=self._user_message(spec, prim_tree),
                retry_label="resolve_usd_prim",
            )
        )
        try:
            parsed = ResolvedObjectReferences.model_validate(data)
            _validate_against_prim_tree(parsed.object_references, prim_tree)
            return _merge_resolved_object_references(spec, parsed.object_references)
        except ValidationError as exc:
            traces.extend(format_validation_error(exc))
            return None
        except AssertionError as exc:
            traces.append(str(exc))
            return None

    @staticmethod
    def _user_message(spec: ArenaEnvGraphSpec, prim_tree: list[UsdPrimRecord]) -> str:
        """Format the pass-2 user message for prim_path resolution."""
        return f"{_prim_tree_catalog(prim_tree)}\n\n{_object_reference_context(spec)}"

    @staticmethod
    def _system_prompt() -> str:
        """Return the pass-2 system prompt for prim_path resolution."""
        return """\
You resolve object_reference prim_path values for robot manipulation environment graphs.

GUIDANCE:
- BACKGROUND PRIM TREE lists prims in nested form: each indented line shows a path suffix
  under its parent; join ancestor suffixes with '/' to form the full relative_path for prim_path.
- Pick prim_path only from those full relative_path values.
- prim_path must be a relative suffix under the parent background — never include
  {ENV_REGEX_NS} or the background registry name.
- Match object_type to the prim object_type when possible: base for anchor surfaces,
  articulation for doors and appliances, rigid for manipulable rigid props.
- When an OpenDoorTask targets an articulation object_reference, set params.openable_joint_name
  to one of the joint names listed for that prim.
- Return one object_references entry per unresolved reference from the input, preserving id,
  parent_id, and object_type. Only change prim_path and params.
- Do not invent prim paths absent from BACKGROUND PRIM TREE.
"""


def _validate_against_prim_tree(
    object_references: list[ObjectReferenceSpec],
    prim_tree: list[UsdPrimRecord],
) -> None:
    """Validate resolved object references against the background USD prim tree."""
    records_by_path = {record.relative_path: record for record in prim_tree}
    for ref in object_references:
        assert ref.prim_path is not None, f"Object reference '{ref.id}' requires a prim_path"
        prim_path = ref.prim_path.lstrip("/")
        record = records_by_path.get(prim_path)
        assert (
            record is not None
        ), f"Object reference '{ref.id}' prim_path {prim_path!r} is not in the background prim tree"
        assert ref.object_type == record.object_type, (
            f"Object reference '{ref.id}' object_type {ref.object_type!r} does not match prim tree "
            f"object_type {record.object_type!r} for {prim_path!r}"
        )


def _merge_resolved_object_references(
    spec: ArenaEnvGraphSpec,
    resolved: list[ObjectReferenceSpec],
) -> ArenaEnvGraphSpec:
    """Merge pass-2 prim_path and params into an environment graph spec."""
    resolved_by_id = {ref.id: ref for ref in resolved}
    assert len(resolved_by_id) == len(resolved), "resolve_usd_prim returned duplicate object_reference ids"
    merged_refs: list[ObjectReferenceSpec] = []
    for ref in spec.object_references or []:
        assert ref.id in resolved_by_id, f"resolve_usd_prim missing object_reference id {ref.id!r}"
        patch = resolved_by_id[ref.id]
        assert (
            patch.object_type == ref.object_type
        ), f"object_reference {ref.id!r} object_type mismatch: {ref.object_type!r} != {patch.object_type!r}"
        merged_refs.append(
            ref.model_copy(
                update={
                    "prim_path": patch.prim_path,
                    "params": dict(ref.params).update(patch.params),
                }
            )
        )
    return spec.model_copy(update={"object_references": merged_refs})


def _prim_tree_catalog(prim_tree: list[UsdPrimRecord]) -> str:
    """Format the background USD prim tree for the pass-2 user message."""
    records = sorted(prim_tree, key=lambda record: record.relative_path)
    lines = ["BACKGROUND PRIM TREE:"]
    stack: list[str] = []
    for record in records:
        path = record.relative_path
        while stack and not path.startswith(stack[-1] + "/"):
            stack.pop()
        parent = stack[-1] if stack else ""
        suffix = path[len(parent) + 1 :] if parent else path
        indent = "  " * len(stack)
        tag = record.object_type.value
        if record.joint_names:
            tag += " " + ",".join(record.joint_names)
        lines.append(f"{indent}{suffix} ({tag})")
        stack.append(path)
    return "\n".join(lines)


def _object_reference_context(spec: ArenaEnvGraphSpec) -> str:
    """Format object-reference context (refs, relations, tasks) for pass-2."""
    ref_ids = {ref.id for ref in spec.object_references or []}
    refs_json = json.dumps(
        [ref.model_dump(mode="json") for ref in (spec.object_references or [])],
        indent=2,
    )
    relations = [
        rel.model_dump(mode="json") for rel in spec.relations if rel.subject in ref_ids or rel.reference in ref_ids
    ]
    tasks: list[dict[str, Any]] = []
    for task in spec.task.subtasks:
        if any(isinstance(value, str) and value in ref_ids for value in task.params.values()):
            tasks.append(task.model_dump(mode="json"))
    return (
        f"OBJECT REFERENCES:\n{refs_json}\n\n"
        f"RELATIONS INVOLVING OBJECT REFERENCES:\n{json.dumps(relations, indent=2)}\n\n"
        f"TASKS INVOLVING OBJECT REFERENCES:\n{json.dumps(tasks, indent=2)}"
    )
