# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pass-2 LLM resolver for object_reference prim_path values."""

from __future__ import annotations

import json
from typing import Any

from isaaclab_arena.agentic_environment_generation.agent_utils import build_strict_schema, extract_response_text
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ObjectReferenceSpec, ResolveObjectReferencePrimPathsResult
from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord


def merge_resolved_object_references(
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
    lines = []
    for record in prim_tree:
        joints = f" joints={list(record.joint_names)}" if record.joint_names else ""
        lines.append(f"- {record.relative_path}  object_type={record.object_type}{joints}")
    return "BACKGROUND PRIM TREE:\n" + "\n".join(lines)


def _object_reference_context(spec: ArenaEnvGraphSpec) -> str:
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


def build_resolve_user_message(spec: ArenaEnvGraphSpec, prim_tree: list[UsdPrimRecord]) -> str:
    """Format the pass-2 user message for prim_path resolution."""
    return f"{_prim_tree_catalog(prim_tree)}\n\n{_object_reference_context(spec)}"


def resolve_object_reference_prim_paths_with_client(
    client: Any,
    model: str,
    spec: ArenaEnvGraphSpec,
    *,
    prim_tree: list[UsdPrimRecord],
    schema: dict[str, Any],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> tuple[list[ObjectReferenceSpec], str]:
    """Call the model to resolve object_reference prim_path values.

    Args:
        client: OpenAI-compatible client.
        model: Model identifier.
        spec: Pass-1 graph spec with unresolved object_references.
        prim_tree: Background USD prim catalog.
        schema: Strict JSON schema for :class:`ResolveObjectReferencePrimPathsResult`.
        temperature: Sampling temperature.
        max_tokens: Response token cap.
        max_retries: Retries after recoverable failures.

    Returns:
        A ``(resolved_refs, raw_response)`` tuple.
    """
    messages = [
        {"role": "system", "content": _resolve_system_prompt()},
        {"role": "user", "content": build_resolve_user_message(spec, prim_tree)},
    ]
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        if attempt > 0:
            print(f"[resolve_usd_prim] retry {attempt}/{max_retries} after: {last_exc}", flush=True)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ResolveObjectReferencePrimPathsResult",
                        "strict": True,
                        "schema": schema,
                    },
                },
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choices = getattr(resp, "choices", None) or []
            assert choices, f"Model {model!r} returned HTTP 200 with no choices"
            text, route = extract_response_text(choices[0].message)
            assert route != "empty", f"Model {model!r} returned an empty structured-outputs envelope"
            data = json.loads(text, strict=False)
            result = ResolveObjectReferencePrimPathsResult.model_validate(data)
            return result.object_references, text
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"Model {model!r} failed prim resolution after {1 + max_retries} attempts. Last error: {last_exc}"
    ) from last_exc


def build_resolve_schema() -> dict[str, Any]:
    """Return strict JSON schema for pass-2 prim resolution."""
    return build_strict_schema(ResolveObjectReferencePrimPathsResult)
