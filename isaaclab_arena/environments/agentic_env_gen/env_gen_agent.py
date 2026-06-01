# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent for parsing natural-language env-generation prompts into an EnvIntentSpec."""

from __future__ import annotations

import json
import os

from openai import OpenAI

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.agentic_env_gen.env_intent_spec import EnvIntentSpec
from isaaclab_arena.environments.agentic_env_gen.structured_output_utils import (
    build_strict_schema,
    check_structured_output_support,
    extract_response_text,
    ping,
)

DEFAULT_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash"


def build_catalog_text() -> str:
    """Build the vocabulary the agent is allowed to use from AssetRegistry."""
    registry = AssetRegistry()
    backgrounds: list[str] = []
    objects: list[dict] = []
    embodiments: list[str] = []
    # TODO(qianl): handle optional lights and hdr images.
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        if issubclass(cls, EmbodimentBase):
            embodiments.append(name)
        elif issubclass(cls, Background):
            backgrounds.append(name)
        elif issubclass(cls, LibraryObject) and cls.tags and "object" in cls.tags:
            tags = [t for t in cls.tags if t != "object"]
            objects.append({"name": name, "tags": tags})

    object_lines = "\n".join(f"- {o['name']}  tags={o['tags']}" for o in sorted(objects, key=lambda o: o["name"]))
    return (
        f"EMBODIMENTS: {', '.join(sorted(embodiments))}\n\n"
        f"BACKGROUNDS: {', '.join(sorted(backgrounds))}\n\n"
        f"OBJECTS ({len(objects)}):\n{object_lines}"
    )


class EnvGenAgent:
    """Parses a natural-language env-generation prompt into an EnvIntentSpec."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        """Configure the OpenAI-compatible client and validate the model.

        Args:
            api_key: API token for the inference endpoint. Falls back
                to the ``NV_API_KEY`` environment variable.
            model: Model identifier at the inference endpoint.
                Must support OpenAI-compatible structured outputs.
            base_url: OpenAI-compatible inference endpoint.
        """
        self.api_key = api_key or os.getenv("NV_API_KEY")
        assert self.api_key, "API key required: set NV_API_KEY or pass api_key."
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        # Validate basic connection and key authentication.
        ping(self.client, self.model)

        # Validate model can produce structured outputs.
        self._spec_schema = build_strict_schema(EnvIntentSpec)
        check_structured_output_support(self.client, self.model, EnvIntentSpec)

    def generate_spec(
        self,
        prompt: str,
        catalog_text: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> tuple[EnvIntentSpec, str]:
        """Call the model with user prompt and return the parsed EnvIntentSpec.

        Args:
            prompt: Natural-language env description from the end user.
            catalog_text: Pre-built asset vocabulary. When ``None``, the catalog is
                built from the live ``AssetRegistry``.
            temperature: Sampling temperature forwarded to the model. Kept
                low by default (0.2) because EnvIntentSpec generation is a
                deterministic-ish translation task — high temperature
                yields creative but invalid schemas.
            max_tokens: Hard cap on the response length.

        Returns:
            A ``(EnvIntentSpec, raw_response)`` tuple. The raw text is
            useful for debugging.
        """
        catalog_text = catalog_text or build_catalog_text()
        system = self._system_prompt()
        user = f"{catalog_text}\n\nUSER PROMPT:\n{prompt}"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "EnvIntentSpec", "strict": True, "schema": self._spec_schema},
            },
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text, route = extract_response_text(resp.choices[0].message)
        if route == "empty":
            raise RuntimeError(
                f"Model {self.model!r} returned an empty structured-outputs envelope. "
                "Run check_structured_output_support() to verify the endpoint/model "
                "actually honours response_format=json_schema."
            )
        # ``strict=False`` lets json.loads accept unescaped control characters
        # (e.g. literal tabs) inside JSON strings — DeepSeek-v4-flash is known
        # to emit these.
        data = json.loads(text, strict=False)
        spec = EnvIntentSpec.model_validate(data)
        return spec, text

    def _system_prompt(self) -> str:
        return (
            "You are an env-generation parser for robot manipulation tasks.\n"
            "Convert a natural-language prompt into an EnvIntentSpec.\n\n"
            "GUIDANCE:\n"
            "- Follow the per-field ``description`` strings in the schema for what each field expects.\n"
            "- If the prompt does not specify a value for an optional field, output null.\n"
            "  Do NOT hallucinate values — the resolver tolerates nulls; it cannot fix invented data.\n"
            "- Articulated objects (microwave, fridge, cabinet) still need a spatial\n"
            "  'on(<object>, background)' relation in initial_state_graph to anchor them; their\n"
            "  open/close behaviour is expressed via tasks, not via relations.\n"
            "- Distractor items around the appliance need 'on(distractor, background)' relations\n"
            "  in initial_state_graph as well.\n"
            "- Task examples (showing kind + subject + target + description shape):\n"
            '    * Pick-and-place: {"kind": "pick_and_place", "subject": "avocado", "target": "bowl",\n'
            '                       "description": "pick up the avocado and place it in the bowl"}\n'
            '    * Open door: {"kind": "open_door", "subject": "microwave", "target": null,\n'
            '                  "description": "open the microwave door"}\n'
            '    * Close door: {"kind": "close_door", "subject": "microwave", "target": null,\n'
            '                   "description": "close the microwave door"}\n'
        )
