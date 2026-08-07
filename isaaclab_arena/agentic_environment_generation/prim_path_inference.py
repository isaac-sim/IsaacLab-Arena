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
from isaaclab_arena.agentic_environment_generation.spec_validation import (
    format_validation_error,
    openable_object_reference_ids,
)
from isaaclab_arena.assets.object_type import ObjectType
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
        prompt, each with semantic hints and an object_type but no prim_path. This step
        maps them to prim paths drawn from the background prim tree, and takes each
        reference's object_type from the prim it resolves to.

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
            return _merge_resolved_object_references(spec, parsed.object_references, prim_tree)
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
- BACKGROUND PRIM TREE lists prims in nested form: each indented line shows a path suffix
  under its parent; join ancestor suffixes with '/' to form the full relative_path for prim_path.
- Pick prim_path only from those full relative_path values.
- prim_path must be a relative suffix under the parent background — never include
  {ENV_REGEX_NS} or the background registry name.
- Read each input object_reference object_type as a hint about the prim's role, not as a
  constraint on the prim's own type. The prim tree decides the type; you only pick the prim.
- When a task in TASKS INVOLVING OBJECT REFERENCES names a reference as its openable_object,
  pick the articulation root prim (the line that lists joint names) and set openable_joint_name
  to one of the joint names listed for it, whatever object_type the input reference claims.
- Otherwise set openable_joint_name to null, and when the input object_type is articulation pick
  the articulation root prim, not a rigid child mesh or collision prim under it; when it is base
  or rigid pick a surface or fixture prim (counter top, shelf, floor, etc.), never an
  articulation root.
- Return one object_references entry per unresolved reference from the input, preserving id.
- Do not invent prim paths absent from BACKGROUND PRIM TREE.
"""


def _prim_tree_catalog(prim_tree: list[UsdPrimRecord]) -> str:
    """Format the background USD prim tree for the user message.

    Each line shows a parent-relative path suffix, indented under its nearest
    retained ancestor, followed by ``object_type`` and optional joint names.
    Join ancestor suffixes with ``/`` to recover the full ``relative_path`` for
    ``prim_path``. Example output::

        BACKGROUND PRIM TREE:
        counter_right_main_group/top_geometry (base)
        fridge_main_group (articulation fridge_door_joint)
        cab_1_main_group (articulation right_door_joint)
          corpus (rigid)
            top (base)
            shelf_1 (base)
          right_door (rigid)
    """
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


class ResolvedObjectReference(BaseModel):
    """Resolver output for one object_reference: the prim it names and the joint a task drives."""

    id: str = Field(description="Id of the input object_reference, copied unchanged.")
    prim_path: str = Field(
        description="Full relative_path from BACKGROUND PRIM TREE, a relative suffix under the parent background.",
    )
    openable_joint_name: str | None = Field(
        description=(
            "When a task opens this reference, the joint name BACKGROUND PRIM TREE lists for prim_path; "
            "null when no task opens it."
        ),
    )


class ResolvedObjectReferences(BaseModel):
    """Resolver output: resolved prim_path values for object_reference nodes."""

    object_references: list[ResolvedObjectReference] = Field(
        description="One entry per input object_reference, in the same order.",
    )


def _resolved_prim_record(
    ref: ObjectReferenceSpec,
    patch: ResolvedObjectReference,
    records_by_path: dict[str, UsdPrimRecord],
    *,
    opened: bool,
) -> UsdPrimRecord:
    """Return the prim tree record the resolver picked for the unresolved object reference ``ref``.

    Args:
        ref: Unresolved object reference from the spec, whose ``object_type`` is the prompt's
            reading of the prim's role.
        patch: Resolver output for that reference, naming the prim it picked.
        records_by_path: Background prim tree records keyed by relative path.
        opened: Whether a subtask opens this reference.

    Returns:
        The record for the picked prim.
    """
    prim_path = patch.prim_path.lstrip("/")
    record = records_by_path.get(prim_path)
    assert record is not None, f"Object reference '{ref.id}' prim_path {prim_path!r} is not in the background prim tree"
    # A prim's object_type follows from its own physics schemas, which the prompt cannot describe,
    # so the record decides it and the reference's own object_type is only a hint. What the prompt
    # does decide is the use: a reference a task opens needs the joints only an articulation root has.
    if not opened:
        return record
    assert record.object_type == ObjectType.ARTICULATION, (
        f"Object reference '{ref.id}' is opened by a task, so prim_path {prim_path!r} must be an "
        f"articulation root, got object_type {record.object_type.value!r}"
    )
    assert patch.openable_joint_name in record.joint_names, (
        f"Object reference '{ref.id}' is opened by a task, so it needs an openable_joint_name from "
        f"{record.joint_names}, got {patch.openable_joint_name!r}"
    )
    return record


def _merge_resolved_object_references(
    spec: ArenaEnvGraphSpec,
    resolved: list[ResolvedObjectReference],
    prim_tree: list[UsdPrimRecord],
) -> ArenaEnvGraphSpec:
    """Merge the resolved prim_path, its prim tree object_type, and openable joint into a graph spec."""
    records_by_path = {record.relative_path: record for record in prim_tree}
    resolved_by_id = {ref.id: ref for ref in resolved}
    assert len(resolved_by_id) == len(resolved), "resolve_usd_prim returned duplicate object_reference ids"
    opened_ids = openable_object_reference_ids(spec)
    merged_refs: list[ObjectReferenceSpec] = []
    for ref in spec.object_references or []:
        assert ref.id in resolved_by_id, f"resolve_usd_prim missing object_reference id {ref.id!r}"
        patch = resolved_by_id[ref.id]
        opened = ref.id in opened_ids
        record = _resolved_prim_record(ref, patch, records_by_path, opened=opened)
        if ref.object_type != record.object_type:
            print(
                f"[resolve_usd_prim] expanded object_reference {ref.id!r} object_type"
                f" {ref.object_type.value!r} into the prim tree's {record.object_type.value!r}"
                f" for {record.relative_path!r}",
                flush=True,
            )
        merged_params = dict(ref.params)
        if opened:
            merged_params["openable_joint_name"] = patch.openable_joint_name
        merged_refs.append(
            ref.model_copy(
                update={
                    "prim_path": record.relative_path,
                    "object_type": record.object_type,
                    "params": merged_params,
                }
            )
        )
    return spec.model_copy(update={"object_references": merged_refs})
