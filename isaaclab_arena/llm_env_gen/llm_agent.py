# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LLM agent for parsing natural-language scene prompts into a SceneSpec.

Uses Claude via NVIDIA's OpenAI-compatible inference API. Emits the
SceneSpec Pydantic bundle so asset resolution stays deterministic.
"""

from __future__ import annotations

import contextlib
import json
import os

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

    obj_lines = "\n".join(f"- {o['name']}  tags={o['tags']}" for o in sorted(objects, key=lambda o: o["name"]))
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
        from openai import OpenAI

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
        user = f"{catalog_text}\n\nUSER PROMPT:\n{prompt}\n\nReturn ONLY a JSON object matching the SceneSpec schema."

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
            "- item.query: the short human name as it appears in the prompt (e.g. 'avocado', 'bowl').\n"
            "  The resolver fuzzy-matches this against the OBJECTS catalog; you do NOT need to emit the\n"
            "  exact registered name.\n"
            "- item.role: 'foreground' for objects the task acts on; 'distractor' for extras mentioned as\n"
            "  clutter; 'anchor' for reference surfaces (rare — the background usually covers this).\n"
            "- item.category_tags: tags that semantically narrow the query, preferring assets with those\n"
            "  tags. This is a PREFERENCE, not a hard filter — the resolver will fall back to the full\n"
            "  catalog if the tag pool is empty or yields no close match. Err toward emitting useful tags;\n"
            "  the trace will report what was relaxed.\n"
            "- relation.kind ∈ {on, in, next_to, at_position, is_anchor}. subject/target reference items\n"
            "  by their query string, or the background name.\n"
            "- Emit TWO scene graphs:\n"
            "    * initial_scene_graph — every relation that holds when the env resets (starting state).\n"
            "    * final_scene_graph   — every relation that must hold for the task to succeed.\n"
            "  Both lists are FULL snapshots, not diffs. Unchanged relations appear in BOTH (e.g. the\n"
            "  bowl and distractors stay on the table). Relations invalidated by task completion must\n"
            "  be omitted from the final graph — for 'place avocado into bowl', the avocado's initial\n"
            "  `on(avocado, table)` is dropped from the final graph and replaced by `in(avocado, bowl)`.\n"
            "- embodiment: use a bare robot family name ('franka', 'droid', 'g1', 'gr1') when the prompt\n"
            "  does not specify a control mode — the resolver defaults each to its IK variant. Use a\n"
            "  full registered name (e.g. 'franka_joint_pos') only when the prompt requests joint control.\n"
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

        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(content)

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
