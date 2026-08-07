# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LLM inference for environment graph specs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    InferenceBackend,
    StructuredOutputRequest,
    build_strict_schema,
)
from isaaclab_arena.agentic_environment_generation.spec_validation import (
    collect_agent_ready_validation_trace,
    format_validation_error,
)
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

MAX_SPEC_INFERENCE_CALLS = 3
"""Maximum generation calls, including critic retries after validation failures."""


class SpecInference:
    """Infers ArenaEnvGraphSpec from a natural-language prompt."""

    def __init__(self, inference_backend: InferenceBackend):
        self._inference_backend = inference_backend
        self._schema = build_strict_schema(ArenaEnvGraphSpec)

    def infer(
        self,
        prompt: str,
        traces: list[str],
        asset_catalog: Any,
        relation_catalog: Any,
        task_catalog: Any,
    ) -> tuple[ArenaEnvGraphSpec | None, dict[str, Any]]:
        """Generate an ArenaEnvGraphSpec from a natural-language prompt.

        Args:
            prompt: End-user environment description.
            traces: Accumulator for validation error lines, extended in place on failure.
            asset_catalog: Embodiment, background, and object vocabulary for the user message.
            relation_catalog: Relation vocabulary for the user message.
            task_catalog: Task vocabulary for the user message.

        Returns:
            A ``(spec, data)`` tuple. On success, ``spec`` is validated and ``data`` is the
            parsed model JSON. On failure, ``spec`` is ``None`` and ``data`` is the raw
            response object.
        """
        base_user_message = self._user_message(
            prompt,
            asset_catalog,
            relation_catalog,
            task_catalog,
        )
        user_message = base_user_message
        data: dict[str, Any] = {}
        for call_index in range(MAX_SPEC_INFERENCE_CALLS):
            data = self._inference_backend.run_json(
                StructuredOutputRequest(
                    schema_name="ArenaEnvGraphSpec",
                    schema=self._schema,
                    system=self._system_prompt(),
                    user=user_message,
                    retry_label="generate_spec",
                )
            )
            try:
                spec = ArenaEnvGraphSpec.model_validate(data)
                validation_traces = collect_agent_ready_validation_trace(
                    spec,
                    asset_catalog=asset_catalog,
                    task_catalog=task_catalog,
                    relation_catalog=relation_catalog,
                )
            except ValidationError as exc:
                spec = None
                validation_traces = format_validation_error(exc)
            if spec is not None and not validation_traces:
                return spec, data
            if call_index + 1 < MAX_SPEC_INFERENCE_CALLS:
                print(
                    f"[generate_spec] critic retry {call_index + 1}/{MAX_SPEC_INFERENCE_CALLS - 1} "
                    f"after validation failed: {'; '.join(validation_traces)}",
                    flush=True,
                )
                user_message = self._critic_user_message(base_user_message, data, validation_traces)
                continue
            traces.extend(validation_traces)
        return None, data

    @staticmethod
    def _critic_user_message(
        previous_user_message: str,
        rejected_data: dict[str, Any],
        validation_traces: list[str],
    ) -> str:
        """Append rejected output and validation feedback for another complete generation."""
        errors = "\n".join(f"- {trace}" for trace in validation_traces)
        rejected = json.dumps(rejected_data, indent=2, default=str)
        return f"""\
{previous_user_message}

CRITIC FEEDBACK:
The previous response failed validation. Regenerate the complete ArenaEnvGraphSpec, correcting
every error below. Return only the corrected structured response.

VALIDATION ERRORS:
{errors}

REJECTED RESPONSE:
{rejected}
"""

    @staticmethod
    def _user_message(
        prompt: str,
        asset_catalog: Any,
        relation_catalog: Any,
        task_catalog: Any,
    ) -> str:
        vocabulary = (
            f"{asset_catalog.to_catalog_string()}\n\n"
            f"{relation_catalog.to_catalog_string()}\n\n"
            f"{task_catalog.to_catalog_string()}"
        )
        return f"{vocabulary}\n\nUSER PROMPT:\n{prompt}"

    @staticmethod
    def _system_prompt() -> str:
        return """\
You are an environment-generator for robot manipulation tasks.
Convert a natural-language prompt into an ArenaEnvGraphSpec.

GUIDANCE:
- Follow the per-field ``description`` strings in the schema.
- REQUIRED: leave ``placement_validators`` and ``cli_override_specs`` null.
- Use only exact names from the catalog for ``registry_name``:
  EMBODIMENTS for ``embodiment``, BACKGROUNDS for ``background``, and OBJECTS for ``objects``.
- Do NOT hallucinate asset names — every ``registry_name`` must appear verbatim in the catalog.
  If the prompt includes the exact registry name, use it.
  If no reasonable match can be found, return empty string.
  If multiple reasonable matches are found, return the closest match or the one with the most specific name.
- For embodiment, if the prompt only mention the robot family (driod/franka) and there are multiple
  variance of that family in EMBODIMENTS, pick the one with the default tag.
- For multiple instances of the same registry asset, use semantic (left/right) or numerical (1/2/3)
  suffixes in ``id``.
- Use ``object_sets`` only when one object varies across environments; list its variants as ``members``.
  Every member must be an OBJECTS entry marked ``type=rigid``.
- An ``object_reference`` names a prim inside the background. Add one for every surface or appliance
  the prompt names that the background merely contains — the floor the robot stands on, a counter top,
  a sink, a fridge, a microwave. Name it after the appliance or surface itself, never after the moving
  part the task acts on — a prompt opening "the fridge door" still names the reference ``fridge``.
- REQUIRED: never add an ``object_reference`` for a surface that IS the background, such as "the table"
  when ``registry_name`` names a table. Use the background id for it. Leave ``object_references`` unset
  when the background itself is every surface the prompt names.
- REQUIRED: a task or relation param naming part of the background must use that
  ``object_reference`` id, never the background id.
- For each ``object_reference``, leave ``prim_path`` empty.
- REQUIRED: pick ``task.composition`` from the number of subtasks you emit. One subtask is
  ``atomic``. Two or more is ``parallel`` when the prompt says the order does not matter, and
  ``sequential`` when the prompt fixes an order. ``atomic`` never has more than one subtask.
- Task parameters referring to an asest (object, background or object reference) must use the ID (NOT registry_name).
- Relation subject and references must use the asset ID (NOT registry_name).
- REQUIRED: include an ``is_anchor`` relation on the resting surface. That is the
  ``object_reference`` for the surface the prompt names inside the background; use the background id
  only when the prompt names no surface inside it.
- REQUIRED: every ``object_reference`` must have an ``is_anchor`` relation.
- For every relation, include all parameters marked ``required`` in the RELATIONS catalog in
  that relation's ``params``.
- All objects need an ``on`` relation with that anchor as ``reference``.
"""
