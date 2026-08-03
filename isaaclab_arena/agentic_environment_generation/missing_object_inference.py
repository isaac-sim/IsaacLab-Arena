# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ask an LLM which objects a prompt needs that the asset catalogue does not have."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from isaaclab_arena.agentic_environment_generation.inference_backend import (
    InferenceBackend,
    StructuredOutputRequest,
    build_strict_schema,
)

_logger = logging.getLogger(__name__)

MAX_SEARCH_PHRASES = 8
"""The most objects one prompt can send to the asset search. Every phrase means another remote
search, and a prompt that needs more than this is asking for a scene, not a manipulation task."""


class MissingObjects(BaseModel):
    """Search phrases for the objects a prompt needs but the catalogue does not have."""

    search_phrases: list[str] = Field(
        default_factory=list,
        description=(
            "One search phrase per object the prompt needs that no catalogue entry covers. "
            "Empty when the catalogue covers everything the prompt asks for."
        ),
    )


class MissingObjectInference:
    """Finds the objects a prompt needs that the registered asset catalogue cannot supply."""

    def __init__(self, inference_backend: InferenceBackend):
        self._inference_backend = inference_backend
        self._schema = build_strict_schema(MissingObjects)

    def infer(self, prompt: str, asset_catalog: Any) -> list[str]:
        """Return search phrases for the objects the catalogue does not have.

        This runs only when the asset search is on, and it does not change spec inference: whatever
        the search finds is registered as an ordinary catalogue entry before the spec is inferred.

        Args:
            prompt: End-user environment description.
            asset_catalog: The registered assets the prompt has to be satisfied from.

        Returns:
            Search phrases in the order the model gave them, at most ``MAX_SEARCH_PHRASES`` of
            them. Empty when the catalogue already covers the prompt.
        """
        data = self._inference_backend.run_json(
            StructuredOutputRequest(
                schema_name="MissingObjects",
                schema=self._schema,
                system=self._system_prompt(),
                user=f"{asset_catalog.to_catalog_string()}\n\nUSER PROMPT:\n{prompt}",
                retry_label="find_missing_objects",
            )
        )
        phrases = [phrase.strip() for phrase in (data.get("search_phrases") or []) if phrase.strip()]
        if len(phrases) > MAX_SEARCH_PHRASES:
            _logger.warning(
                "%d objects to search for; only the first %d are searched", len(phrases), MAX_SEARCH_PHRASES
            )
            phrases = phrases[:MAX_SEARCH_PHRASES]
        if phrases:
            _logger.info("objects the catalogue does not cover: %s", ", ".join(phrases))
        else:
            _logger.info("the catalogue covers every object in the prompt; nothing to search for")
        return phrases

    @staticmethod
    def _system_prompt() -> str:
        return f"""\
You decide which objects a robot-manipulation prompt needs that a fixed asset catalogue lacks.

You are given the catalogue and the prompt. Return a search phrase for each object the prompt
needs that the catalogue cannot supply, so it can be looked up in an external asset library.

GUIDANCE:
- Prefer a catalogue entry whenever one reasonably matches, even loosely. Only name an object the
  catalogue has nothing suitable for. An empty list is the common and correct answer.
- Judge against the OBJECTS block. Embodiments, backgrounds, relations and tasks are not your
  concern, and a resting surface such as a table is a background, not a missing object.
- Write each phrase as adjectives followed by the object noun, not as a clause describing it and
  not as a placeholder (e.g. "green trash can", not "trash can that is green" or "object_1").
- Name the object once however many of it the prompt asks for; the count is not your concern.
- Return at most {MAX_SEARCH_PHRASES} phrases, most important first.
"""
