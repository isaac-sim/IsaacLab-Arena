# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent for parsing natural-language env-generation prompts into an EnvironmentIntentSpec."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec
from isaaclab_arena.agentic_environment_generation.structured_output_utils import (
    build_strict_schema,
    check_structured_output_support,
    extract_response_text,
    ping,
)
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.relations.relations import RelationBase

DEFAULT_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash"


@dataclass
class AssetCatalogue:
    """Registered asset vocabulary grouped for the env-gen agent prompt."""

    # A list of embodiment names for agent to choose from.
    embodiments: list[str] = field(default_factory=list)
    # A list of background names for agent to choose from.
    backgrounds: list[str] = field(default_factory=list)
    # A list of object names and their tags for agent to choose from.
    objects: list[dict[str, Any]] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message vocabulary block."""
        object_lines = "\n".join(
            f"- {o['name']}  tags={o['tags']}" for o in sorted(self.objects, key=lambda o: o["name"])
        )
        return (
            f"EMBODIMENTS: {', '.join(sorted(self.embodiments))}\n\n"
            f"BACKGROUNDS: {', '.join(sorted(self.backgrounds))}\n\n"
            f"OBJECTS ({len(self.objects)}):\n{object_lines}"
        )


def build_asset_catalogue(registry: AssetRegistry | None = None) -> AssetCatalogue:
    """Collect registered embodiments, backgrounds, and pick-up objects from ``AssetRegistry``."""

    registry = registry or AssetRegistry()
    catalogue = AssetCatalogue()
    # TODO(qianl): handle optional lights and hdr images.
    # TODO(qianl): add tag to filter out validated/agent-ready assets only.
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        if issubclass(cls, EmbodimentBase):
            catalogue.embodiments.append(name)
        elif issubclass(cls, Background):
            catalogue.backgrounds.append(name)
        elif issubclass(cls, LibraryObject) and cls.tags and "object" in cls.tags:
            tags = [t for t in cls.tags if t != "object"]
            catalogue.objects.append({"name": name, "tags": tags})
    return catalogue


@dataclass
class RelationCatalogueEntry:
    """One registered spatial relation exposed to the env-gen agent."""

    name: str
    unary: bool
    summary: str


@dataclass
class RelationCatalogue:
    """Registered object-relation vocabulary for the env-gen agent prompt."""

    relations: list[RelationCatalogueEntry] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message RELATIONS block."""
        lines = []
        for entry in sorted(self.relations, key=lambda r: r.name):
            arity = "unary" if entry.unary else "binary"
            lines.append(f"- {entry.name} ({arity}): {entry.summary}")
        return f"RELATIONS ({len(self.relations)}):\n" + "\n".join(lines)


def _first_docstring_line(cls: type) -> str:
    doc = cls.__doc__ or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def build_relation_catalogue(
    registry: ObjectRelationLibraryRegistry | None = None,
) -> RelationCatalogue:
    """Collect registered object relations from ``ObjectRelationLibraryRegistry``."""
    registry = registry or ObjectRelationLibraryRegistry()
    catalogue = RelationCatalogue()
    for name in registry.get_all_keys():
        relation_cls = registry.get_object_relation_by_name(name)
        assert issubclass(relation_cls, RelationBase), f"{name!r} is not a RelationBase subclass"
        catalogue.relations.append(
            RelationCatalogueEntry(
                name=name,
                unary=relation_cls.is_unary(),
                summary=_first_docstring_line(relation_cls),
            )
        )
    return catalogue


class EnvironmentGenerationAgent:
    """Parses a natural-language env-generation prompt into an EnvironmentIntentSpec."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
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
        self.model = model or DEFAULT_MODEL
        base_url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        # Validate basic connection and key authentication.
        ping(self.client, self.model)

        # Validate model can produce structured outputs.
        check_structured_output_support(self.client, self.model, EnvironmentIntentSpec)
        self._spec_schema = build_strict_schema(EnvironmentIntentSpec)

    def generate_spec(
        self,
        prompt: str,
        catalog: AssetCatalogue | None = None,
        relation_catalog: RelationCatalogue | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> tuple[EnvironmentIntentSpec, str]:
        """Call the model with user prompt and return the parsed EnvironmentIntentSpec.

        Args:
            prompt: Natural-language env description from the end user.
            catalog: Pre-built asset vocabulary. When ``None``, the catalog is
                built from the live ``AssetRegistry``.
            relation_catalog: Pre-built relation vocabulary. When ``None``, built
                from the live ``ObjectRelationLibraryRegistry``.
            temperature: Sampling temperature forwarded to the model. Kept
                low by default (0.2) because EnvironmentIntentSpec generation is a
                deterministic-ish translation task — high temperature
                yields creative but invalid schemas.
            max_tokens: Hard cap on the response length.

        Returns:
            A ``(EnvironmentIntentSpec, raw_response)`` tuple. The raw text is
            useful for debugging.
        """
        catalog = catalog or build_asset_catalogue()
        relation_catalog = relation_catalog or build_relation_catalogue()
        vocabulary = f"{catalog.to_catalog_string()}\n\n{relation_catalog.to_catalog_string()}"
        system = self._system_prompt()
        user = f"{vocabulary}\n\nUSER PROMPT:\n{prompt}"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "EnvironmentIntentSpec", "strict": True, "schema": self._spec_schema},
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
        spec = EnvironmentIntentSpec.model_validate(data)
        return spec, text

    def _system_prompt(self) -> str:
        return (
            "You are an env-generation parser for robot manipulation tasks.\n"
            "Convert a natural-language prompt into an EnvironmentIntentSpec.\n\n"
            "GUIDANCE:\n"
            "- Follow the per-field ``description`` strings in the schema for what each field expects.\n"
            "- Use only asset names from EMBODIMENTS / BACKGROUNDS / OBJECTS and only "
            "relation kinds from RELATIONS in the user message.\n"
            "- If the prompt does not specify a value for an optional field, output null.\n"
            "  Do NOT hallucinate values — the resolver tolerates nulls; it cannot fix invented data.\n"
            "- For binary relations (e.g. on), subject is the child object and target is "
            "the parent surface (typically the background name).\n"
            "- Articulated objects (microwave, fridge, cabinet) still need an "
            "'on' relation in initial_state_graph (subject=object, target=background) "
            "to anchor them; open/close is expressed via tasks, not relations.\n"
            "- Distractor items around the appliance need the same 'on' pattern in "
            "initial_state_graph.\n"
            "- Do not invent relation kinds (e.g. at_pose, in) that are absent from RELATIONS.\n"
            "- Task examples (showing kind + subject + target + description shape):\n"
            '    * Pick-and-place: {"kind": "pick_and_place", "subject": "avocado", "target": "bowl",\n'
            '                       "description": "pick up the avocado and place it in the bowl"}\n'
            '    * Open door: {"kind": "open_door", "subject": "microwave", "target": null,\n'
            '                  "description": "open the microwave door"}\n'
            '    * Close door: {"kind": "close_door", "subject": "microwave", "target": null,\n'
            '                   "description": "close the microwave door"}\n'
        )
