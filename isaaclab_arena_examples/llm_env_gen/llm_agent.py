# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LLM agent for parsing natural-language scene prompts into a SceneSpec.

Uses Claude via NVIDIA's OpenAI-compatible inference API. Modeled on
isaaclab_arena/scene_gen/llm_agent.py from the dev/stark branch, adapted
to emit our SceneSpec (Pydantic) so asset resolution stays deterministic.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from .schema import SceneSpec

DEFAULT_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_MODEL = "aws/anthropic/bedrock-claude-opus-4-6"


def build_catalog_text() -> str:
    """Introspect AssetRegistry and build the vocabulary the LLM is allowed to use."""
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    backgrounds: list[str] = []
    objects: list[dict] = []
    embodiments: list[str] = []
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        tags = list(getattr(cls, "tags", []))
        if "embodiment" in tags:
            embodiments.append(name)
        elif "background" in tags:
            backgrounds.append(name)
        elif "object" in tags:
            objects.append({"name": name, "tags": [t for t in tags if t != "object"]})

    obj_lines = "\n".join(
        f"- {o['name']}  tags={o['tags']}" for o in sorted(objects, key=lambda o: o["name"])
    )
    return (
        f"EMBODIMENTS: {', '.join(sorted(embodiments))}\n\n"
        f"BACKGROUNDS: {', '.join(sorted(backgrounds))}\n\n"
        f"OBJECTS ({len(objects)}):\n{obj_lines}"
    )


class LLMAgent:
    """Parses a natural-language prompt into a SceneSpec."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.api_key = api_key or os.getenv("NV_API_KEY")
        assert self.api_key, "API key required: set NV_API_KEY or pass api_key."
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate_spec(
        self,
        prompt: str,
        catalog_text: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> tuple[SceneSpec, str]:
        """Return (validated SceneSpec, raw LLM response)."""
        catalog_text = catalog_text or build_catalog_text()
        system = self._system_prompt()
        user = (
            f"{catalog_text}\n\n"
            f"USER PROMPT:\n{prompt}\n\n"
            "Return ONLY a JSON object matching the SceneSpec schema."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content
        data = self._extract_json(raw)
        spec = SceneSpec.model_validate(data)
        return spec, raw

    def _system_prompt(self) -> str:
        schema = json.dumps(SceneSpec.model_json_schema(), indent=2)
        return (
            "You are a scene-generation parser for robot manipulation tasks.\n"
            "Convert a natural-language prompt into a SceneSpec JSON object that matches the schema below.\n\n"
            "RULES:\n"
            "- item.query: short human name from the prompt (e.g. 'avocado', 'bowl'). The resolver will fuzzy-match\n"
            "  it against the OBJECTS catalog; you do NOT need to emit the exact registered name.\n"
            "- item.role: 'foreground' for task-relevant objects named in the prompt; 'distractor' for extras;\n"
            "  'anchor' for reference surfaces (rare — usually the background handles this).\n"
            "- item.category_tags: constrain the asset pool. Use ONLY tags that appear in the OBJECTS catalog\n"
            "  (e.g. ['vegetable'], ['fruit'], ['graspable']). For distractors where the prompt says 'veggies',\n"
            "  emit one Item per distractor slot with category_tags=['vegetable'].\n"
            "- relation.kind ∈ {on, next_to, at_position, is_anchor}. subject/target reference items by their\n"
            "  query string, or the background name.\n"
            "- Default embodiment is 'franka_ik' unless the prompt specifies a robot with a different control mode.\n"
            "- Emit ONLY the JSON object. No prose, no markdown fences.\n\n"
            f"SCHEMA:\n{schema}"
        )

    @staticmethod
    def _extract_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        start = content.find("{")
        assert start != -1, f"No JSON object in LLM response: {content!r}"
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(content[start : i + 1])
        raise AssertionError(f"Unbalanced JSON in LLM response: {content!r}")
