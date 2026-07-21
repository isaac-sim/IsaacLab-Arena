# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LLM inference for object_reference prim_path values."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    InferenceBackend,
    StructuredOutputRequest,
    build_strict_schema,
)
from isaaclab_arena.agentic_environment_generation.spec_validation import format_validation_error
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import ObjectReferenceSpec

if TYPE_CHECKING:
    from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord


class PrimPathInference:
    """Identify object_reference prim_path from the background USD."""

    def __init__(self, inference_backend: InferenceBackend):
        self._inference_backend = inference_backend
        self._schema = build_strict_schema(ResolvedObjectReferences)

    def infer(
        self,
        spec: ArenaEnvGraphSpec,
        traces: list[str],
    ) -> ArenaEnvGraphSpec | None:
        """Resolve the background USD prim_path for object references using semantic/physical hints.

        The input spec carries object_references inferred from the natural-language
        prompt, each with semantic hints and an object_type but no prim_path.
        This step maps them to prim paths drawn from the background prim tree.

        Args:
            spec: Spec whose object references have unresolved ``prim_path`` values.
            traces: Accumulator for validation error lines, extended in place when the
                model output fails schema or prim-tree validation.

        Returns:
            A copy of ``spec`` with resolved prim paths on success, otherwise ``None``
            (with error lines appended to ``traces``).
        """
        # Defer pxr import until call time to avoid conflict with SimulationApp.
        from isaaclab_arena.utils.usd_prim_tree import load_usd_prim_tree

        usd_path = spec.background.resolve_usd_path()
        prim_tree = load_usd_prim_tree(usd_path)
        data = self._inference_backend.run_json(
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
        return f"{_prim_tree_catalog(prim_tree)}\n\n{_object_reference_context(spec)}"

    @staticmethod
    def _system_prompt() -> str:
        return """\
You resolve object_reference prim_path values for an ArenaEnvGraphSpec.

GUIDANCE:
- BACKGROUND PRIM TREE lists each prim on its own line as ``<relative_path> (<object_type> ...)``.
  The ``relative_path`` before the parenthesis is the exact prim_path to return.
- Pick prim_path only from those relative_path values (copy verbatim; do not shorten to leaf names).
- prim_path must be a relative suffix under the parent background — never include
  {ENV_REGEX_NS} or the background registry name.
- Respect each input object_reference object_type: pick a prim_path whose object_type in
  BACKGROUND PRIM TREE matches the object_type from the input spec. Do not change object_type.
- When the input object_type is articulation, pick the articulation root prim (the line that
  lists joint names), not a rigid child mesh or collision prim under it.
- When the input object_type is base, pick an anchor-surface prim (counter top, shelf, etc.).
- When an OpenDoorTask targets an articulation object_reference, set params.openable_joint_name
  to one of the joint names listed for that prim.
- Return one object_references entry per unresolved reference from the input, preserving id,
  parent_id, and object_type. Only change prim_path and params.
- Do not invent prim paths absent from BACKGROUND PRIM TREE.
"""


def _prim_tree_catalog(prim_tree: list[UsdPrimRecord]) -> str:
    """Format the background USD prim tree for the user message.

    Each line shows the full default-prim-relative ``relative_path`` (the prim_path
    to return), indented under ancestors for readability, followed by ``object_type``
    and optional joint names. Example output::

        BACKGROUND PRIM TREE:
        counter_right_main_group/top_geometry (base)
        fridge_main_group (articulation fridge_door_joint)
        cab_1_main_group (articulation right_door_joint)
          cab_1_main_group/corpus (rigid)
            cab_1_main_group/corpus/top (base)
            cab_1_main_group/corpus/shelf_1 (base)
          cab_1_main_group/right_door (rigid)
    """
    records = sorted(prim_tree, key=lambda record: record.relative_path)
    lines = ["BACKGROUND PRIM TREE:"]
    stack: list[str] = []
    for record in records:
        path = record.relative_path
        while stack and not path.startswith(stack[-1] + "/"):
            stack.pop()
        indent = "  " * len(stack)
        tag = record.object_type.value
        if record.joint_names:
            tag += " " + ",".join(record.joint_names)
        lines.append(f"{indent}{path} ({tag})")
        stack.append(path)
    return "\n".join(lines)


def _object_reference_context(spec: ArenaEnvGraphSpec) -> str:
    """Format object-reference context (refs, relations, tasks)."""
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


class ResolvedObjectReferences(BaseModel):
    """Resolver output: resolved prim_path values for object_reference nodes."""

    object_references: list[ObjectReferenceSpec] = Field(
        description="Resolved object references with prim_path set to a relative suffix under the parent background.",
    )


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
    """Merge resolved prim_path and params into an environment graph spec."""
    resolved_by_id = {ref.id: ref for ref in resolved}
    assert len(resolved_by_id) == len(resolved), "resolve_usd_prim returned duplicate object_reference ids"
    merged_refs: list[ObjectReferenceSpec] = []
    for ref in spec.object_references or []:
        assert ref.id in resolved_by_id, f"resolve_usd_prim missing object_reference id {ref.id!r}"
        patch = resolved_by_id[ref.id]
        assert (
            patch.object_type == ref.object_type
        ), f"object_reference {ref.id!r} object_type mismatch: {ref.object_type!r} != {patch.object_type!r}"
        assert patch.prim_path is not None, f"object_reference {ref.id!r} requires a prim_path"
        prim_path = patch.prim_path.lstrip("/")
        merged_params = dict(ref.params)
        merged_params.update(patch.params)
        merged_refs.append(
            ref.model_copy(
                update={
                    "prim_path": prim_path,
                    "params": merged_params,
                }
            )
        )
    return spec.model_copy(update={"object_references": merged_refs})
