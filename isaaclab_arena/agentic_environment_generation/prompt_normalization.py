# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LLM inference to normalize user prompts before spec generation."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    InferenceBackend,
    StructuredOutputRequest,
    build_strict_schema,
)
from isaaclab_arena.agentic_environment_generation.spec_validation import format_validation_error


class NormalizedPromptDescriptions(BaseModel):
    """Structured descriptions for each ArenaEnvGraphSpec section."""

    env_name: str = Field(
        min_length=1,
        description="Short snake_case label summarizing the scene and tasks.",
    )
    embodiment: str = Field(
        min_length=1,
        description="Robot/embodiment intent in plain language (family, control mode, cameras).",
    )
    background: str = Field(
        min_length=1,
        description="Static scene/background intent in plain language.",
    )
    object_references: str = Field(
        default="",
        description=(
            "Optional surfaces or appliances inside the background that should become "
            "object_reference nodes; empty when none are needed."
        ),
    )
    objects: list[str] = Field(
        default_factory=list,
        description=(
            "One short search phrase per manipulable object or distractor, e.g. 'red hammer' or 'ceramic bowl'."
        ),
    )
    relations: str = Field(
        default="",
        description="Spatial layout intent in plain language (on/next_to/anchor).",
    )
    task: str = Field(
        min_length=1,
        description="Overall manipulation task intent in plain language.",
    )


def format_normalized_prompt_block(normalized: NormalizedPromptDescriptions) -> str:
    """Format normalized descriptions for downstream LLM prompts."""
    object_lines = "\n".join(f"- {phrase}" for phrase in normalized.objects) or "- (none)"
    refs = normalized.object_references.strip() or "(none)"
    relations = normalized.relations.strip() or "(none)"
    return (
        "NORMALIZED PROMPT:\n"
        f"env_name: {normalized.env_name}\n"
        f"embodiment: {normalized.embodiment}\n"
        f"background: {normalized.background}\n"
        f"object_references: {refs}\n"
        f"objects:\n{object_lines}\n"
        f"relations: {relations}\n"
        f"task: {normalized.task}"
    )


class PromptNormalizationInference:
    """Natural-language prompt -> normalized section descriptions."""

    def __init__(self, inference_backend: InferenceBackend):
        """Wire prompt normalization to a structured-output backend.

        Args:
            inference_backend: Shared LLM client for JSON-schema completion requests.
        """
        self._inference_backend = inference_backend
        self._schema = build_strict_schema(NormalizedPromptDescriptions)

    def infer(self, prompt: str, traces: list[str]) -> NormalizedPromptDescriptions | None:
        """Normalize a user prompt into section descriptions for later inference passes.

        Args:
            prompt: End-user environment description.
            traces: Accumulator for validation error lines, extended in place on failure.

        Returns:
            Validated normalized descriptions on success, otherwise ``None``.
        """
        data = self._inference_backend.run_json(
            StructuredOutputRequest(
                schema_name="NormalizedPromptDescriptions",
                schema=self._schema,
                system=self._system_prompt(),
                user=f"USER PROMPT:\n{prompt.strip()}",
                retry_label="normalize_prompt",
            )
        )
        try:
            return NormalizedPromptDescriptions.model_validate(data)
        except ValidationError as exc:
            traces.extend(format_validation_error(exc))
            return None

    @staticmethod
    def _system_prompt() -> str:
        return """\
You normalize robot environment-generation prompts into structured descriptions.
Do not choose registry asset names. Describe intent only.

GUIDANCE:
- ``env_name`` should be short snake_case summarizing the scene and task.
- ``objects`` must list one short search phrase per distinct manipulable object or distractor.
  Keep phrases compact and visual, e.g. "red hammer", "blue bowl", "spring clamp".
- Leave ``object_references`` empty unless the prompt explicitly mentions a surface or appliance
  inside the background (counter top, fridge door, table surface).
- ``relations`` should capture placement intent (what sits on what, next_to sides, anchor surface).
- ``task`` should summarize the robot's goal in one or two sentences.
"""
