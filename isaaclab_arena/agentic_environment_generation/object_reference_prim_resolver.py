# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pass-2 LLM resolver for object_reference prim_path values."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from isaaclab_arena.agentic_environment_generation.agent_utils import build_strict_schema
from isaaclab_arena.agentic_environment_generation.query_backend import QueryBackend, StructuredOutputRequest
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ObjectReferenceSpec
from isaaclab_arena.utils.asset_usd import resolve_asset_usd_path
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord, load_usd_prim_tree


class ResolvedObjectReferences(BaseModel):
    """Pass-2 resolver output: resolved prim_path values for object_reference nodes."""

    object_references: list[ObjectReferenceSpec] = Field(
        description="Resolved object references with prim_path set to a relative suffix under the parent background.",
    )

    @model_validator(mode="after")
    def _validate_against_prim_tree(self, info: ValidationInfo) -> ResolvedObjectReferences:
        prim_tree: list[UsdPrimRecord] | None = info.context.get("prim_tree") if info.context else None
        assert prim_tree is not None, "ResolvedObjectReferences validation requires prim_tree context"
        records_by_path = {record.relative_path: record for record in prim_tree}
        for ref in self.object_references:
            assert ref.prim_path is not None, f"Object reference '{ref.id}' requires a prim_path"
            record = records_by_path.get(ref.prim_path)
            assert (
                record is not None
            ), f"Object reference '{ref.id}' prim_path {ref.prim_path!r} is not in the background prim tree"
            assert ref.object_type == record.object_type, (
                f"Object reference '{ref.id}' object_type {ref.object_type!r} does not match prim tree "
                f"object_type {record.object_type!r} for {ref.prim_path!r}"
            )
        return self


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
        merged_params = dict(ref.params)
        merged_params.update(patch.params)
        merged_refs.append(
            ref.model_copy(
                update={
                    "prim_path": patch.prim_path,
                    "params": merged_params,
                }
            )
        )
    return spec.model_copy(update={"object_references": merged_refs})


def _prim_tree_catalog(prim_tree: list[UsdPrimRecord]) -> str:
    """Format the background USD prim tree for the pass-2 user message."""
    lines = []
    for record in prim_tree:
        joints = f" joints={list(record.joint_names)}" if record.joint_names else ""
        lines.append(f"- {record.relative_path}  object_type={record.object_type}{joints}")
    return "BACKGROUND PRIM TREE:\n" + "\n".join(lines)


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
    for task in spec.tasks:
        if any(isinstance(value, str) and value in ref_ids for value in task.params.values()):
            tasks.append(task.model_dump(mode="json"))
    return (
        f"OBJECT REFERENCES:\n{refs_json}\n\n"
        f"RELATIONS INVOLVING OBJECT REFERENCES:\n{json.dumps(relations, indent=2)}\n\n"
        f"TASKS INVOLVING OBJECT REFERENCES:\n{json.dumps(tasks, indent=2)}"
    )


def _resolve_system_prompt() -> str:
    """Return the pass-2 system prompt for prim_path resolution."""
    return """\
You resolve object_reference prim_path values for robot manipulation environment graphs.

GUIDANCE:
- Pick prim_path only from relative_path values listed in BACKGROUND PRIM TREE.
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


def _build_resolve_user_message(spec: ArenaEnvGraphSpec, prim_tree: list[UsdPrimRecord]) -> str:
    """Format the pass-2 user message for prim_path resolution."""
    return f"{_prim_tree_catalog(prim_tree)}\n\n{_object_reference_context(spec)}"


class ObjectReferencePrimResolver:
    """Pass 2: resolve object_reference prim_path values against background USD."""

    def __init__(self, query_backend: QueryBackend):
        self._query_backend = query_backend
        self._schema = build_strict_schema(ResolvedObjectReferences)

    def infer(
        self,
        spec: ArenaEnvGraphSpec,
        traces: list[str],
    ) -> ArenaEnvGraphSpec:
        """Always resolve object_reference prim_path values against the background USD prim tree."""
        usd_path = resolve_asset_usd_path(spec.background)
        prim_tree = load_usd_prim_tree(usd_path)
        data = self._query_backend.run_json(
            StructuredOutputRequest(
                schema_name="ResolvedObjectReferences",
                schema=self._schema,
                system=_resolve_system_prompt(),
                user=_build_resolve_user_message(spec, prim_tree),
                retry_label="resolve_usd_prim",
            )
        )
        resolved = ResolvedObjectReferences.model_validate(
            data,
            context={"prim_tree": prim_tree},
        ).object_references
        spec = _merge_resolved_object_references(spec, resolved)
        return spec
