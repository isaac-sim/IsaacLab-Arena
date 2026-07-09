# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent for parsing natural-language env-generation prompts into an ArenaEnvGraphSpec."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from isaaclab_arena.agentic_environment_generation.agent_utils import build_strict_schema, extract_response_text, ping
from isaaclab_arena.agentic_environment_generation.spec_validation import (
    collect_agent_ready_task_validation_traces,
    required_task_init_param_names,
    try_parse_env_graph_spec,
)
from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry, TaskRegistry
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.relations.relations import RelationBase

# TODO(qianl): This is currently Nvidia internal. Switch to public endpoint.
DEFAULT_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# Asset catalogue (AssetRegistry → user-prompt blocks)
# ---------------------------------------------------------------------------


@dataclass
class AssetCatalogue:
    """Registered asset vocabulary grouped for the agent prompt."""

    # A list of embodiment names and their tags for agent to choose from.
    embodiments: list[dict[str, Any]] = field(default_factory=list)
    # A list of background names and their tags for agent to choose from.
    backgrounds: list[dict[str, Any]] = field(default_factory=list)
    # A list of object names and their tags for agent to choose from.
    objects: list[dict[str, Any]] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message vocabulary block."""
        embodiment_lines = "\n".join(
            f"- {e['name']}  tags={e['tags']}" for e in sorted(self.embodiments, key=lambda e: e["name"])
        )
        background_lines = "\n".join(
            f"- {b['name']}  tags={b['tags']}" for b in sorted(self.backgrounds, key=lambda b: b["name"])
        )
        object_lines = "\n".join(
            f"- {o['name']}  tags={o['tags']}" for o in sorted(self.objects, key=lambda o: o["name"])
        )
        return (
            f"EMBODIMENTS ({len(self.embodiments)}):\n{embodiment_lines}\n\n"
            f"BACKGROUNDS ({len(self.backgrounds)}):\n{background_lines}\n\n"
            f"OBJECTS ({len(self.objects)}):\n{object_lines}"
        )


def build_asset_catalogue(registry: AssetRegistry | None = None) -> AssetCatalogue:
    """Collect registered embodiments, backgrounds, and pick-up objects from ``AssetRegistry``."""
    registry = registry or AssetRegistry()
    catalogue = AssetCatalogue()
    # TODO(qianl): handle optional lights and hdr images.
    # TODO(qianl): add tag to filter out validated/agent-ready assets only.
    # Classify by registry tags, not issubclass(Background/Object/EmbodimentBase): importing those
    # types pulls in pxr before SimulationApp and breaks unit tests.
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        tags = getattr(cls, "tags", None) or []
        if "embodiment" in tags:
            catalogue.embodiments.append({"name": name, "tags": [t for t in tags if t != "embodiment"]})
        elif "background" in tags:
            catalogue.backgrounds.append({"name": name, "tags": [t for t in tags if t != "background"]})
        elif "object" in tags:
            catalogue.objects.append({"name": name, "tags": [t for t in tags if t != "object"]})
    return catalogue


# ---------------------------------------------------------------------------
# Relation catalogue (ObjectRelationLibraryRegistry → user-prompt blocks)
# ---------------------------------------------------------------------------


def _first_docstring_line(cls: type) -> str:
    doc = cls.__doc__ or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass
class RelationCatalogueEntry:
    """One registered spatial relation exposed to the agent."""

    name: str
    unary: bool
    summary: str


@dataclass
class RelationCatalogue:
    """Registered object-relation vocabulary for the agent prompt."""

    relations: list[RelationCatalogueEntry] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message RELATIONS block."""
        lines = []
        for entry in sorted(self.relations, key=lambda r: r.name):
            arity = "unary" if entry.unary else "binary"
            lines.append(f"- {entry.name} ({arity}): {entry.summary}")
        return f"RELATIONS ({len(self.relations)}):\n" + "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Task catalogue (TaskRegistry → user-prompt blocks)
# ---------------------------------------------------------------------------


@dataclass
class TaskCatalogueEntry:
    """One agent_ready task exposed to the agent."""

    name: str
    required_params: list[str]
    summary: str


@dataclass
class TaskCatalogue:
    """Agent-ready task vocabulary for the agent prompt."""

    tasks: list[TaskCatalogueEntry] = field(default_factory=list)

    def to_catalog_string(self) -> str:
        """Format this catalogue as the user-message TASKS block."""
        lines = []
        for entry in sorted(self.tasks, key=lambda t: t.name):
            params = ", ".join(entry.required_params)
            lines.append(f"- {entry.name} ({params}): {entry.summary}")
        return f"TASKS ({len(self.tasks)}):\n" + "\n".join(lines)


def agent_ready_task_names(registry: TaskRegistry | None = None) -> frozenset[str]:
    """Return ``TaskRegistry`` keys for tasks marked with ``@agent_ready``."""
    registry = registry or TaskRegistry()
    return frozenset(
        name for name in registry.get_all_keys() if getattr(registry.get_task_by_name(name), "agent_ready", False)
    )


def build_task_catalogue(registry: TaskRegistry | None = None) -> TaskCatalogue:
    """Collect agent_ready tasks from ``TaskRegistry``."""
    registry = registry or TaskRegistry()
    catalogue = TaskCatalogue()
    for name in sorted(agent_ready_task_names(registry)):
        task_cls = registry.get_task_by_name(name)
        catalogue.tasks.append(
            TaskCatalogueEntry(
                name=name,
                required_params=required_task_init_param_names(task_cls),
                summary=_first_docstring_line(task_cls),
            )
        )
    return catalogue


# ---------------------------------------------------------------------------
# Environment generation agent
# ---------------------------------------------------------------------------


class EnvironmentGenerationAgent:
    """Parses a natural-language env-generation prompt into an ArenaEnvGraphSpec."""

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
        self._spec_schema = build_strict_schema(ArenaEnvGraphSpec)
        self.last_validation_traces: list[str] = []

    def generate_spec(
        self,
        prompt: str,
        asset_catalog: AssetCatalogue | None = None,
        relation_catalog: RelationCatalogue | None = None,
        task_catalog: TaskCatalogue | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> tuple[ArenaEnvGraphSpec | None, dict[str, Any]]:
        """Call the model with user prompt and return the parsed ArenaEnvGraphSpec.

        Args:
            prompt: Natural-language env description from the end user.
            asset_catalog: Pre-built asset vocabulary. When ``None``, built
                from the live ``AssetRegistry``.
            relation_catalog: Pre-built relation vocabulary. When ``None``, built
                from the live ``ObjectRelationLibraryRegistry``.
            task_catalog: Pre-built task vocabulary. When ``None``, built from
                ``TaskRegistry`` tasks marked ``@agent_ready``.
            temperature: Sampling temperature forwarded to the model. Kept
                low by default (0.2) because spec generation is a
                deterministic-ish translation task — high temperature
                yields creative but invalid schemas.
            max_tokens: Hard cap on the response length.
            max_retries: Number of additional attempts after a recoverable failure
                (network errors, timeouts, empty responses, malformed JSON). Each
                retry is a fresh API call.

        Returns:
            A ``(ArenaEnvGraphSpec | None, data)`` tuple. ``data`` is the parsed
            JSON object from the model. When schema validation fails, ``spec`` is
            ``None`` and ``agent.last_validation_traces`` holds the error trace.
        """
        asset_catalog = asset_catalog or build_asset_catalogue()
        relation_catalog = relation_catalog or build_relation_catalogue()
        task_catalog = task_catalog or build_task_catalogue()
        vocabulary = (
            f"{asset_catalog.to_catalog_string()}\n\n"
            f"{relation_catalog.to_catalog_string()}\n\n"
            f"{task_catalog.to_catalog_string()}"
        )
        system = self._system_prompt()
        user = f"{vocabulary}\n\nUSER PROMPT:\n{prompt}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_exc: Exception | None = None
        for attempt in range(1 + max_retries):
            if attempt > 0:
                print(f"[generate_spec] retry {attempt}/{max_retries} after: {last_exc}", flush=True)

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ArenaEnvGraphSpec",
                            "strict": True,
                            "schema": self._spec_schema,
                        },
                    },
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choices = getattr(resp, "choices", None) or []
                assert choices, (
                    f"Model {self.model!r} returned HTTP 200 with no choices "
                    "(content filter / guardrail / rate-limit response with empty body)."
                )
                text, route = extract_response_text(choices[0].message)
                assert route != "empty", (
                    f"Model {self.model!r} returned an empty structured-outputs envelope. "
                    "Verify the endpoint/model supports response_format=json_schema."
                )
                # ``strict=False`` lets json.loads accept unescaped control characters
                # (e.g. literal tabs) inside JSON strings — DeepSeek-v4-flash is known
                # to emit these.
                data = json.loads(text, strict=False)
                # TODO(qianl): add fuzzy-match support for registry_name matching.
                spec, traces = try_parse_env_graph_spec(data)
                if spec is not None:
                    traces = [*traces, *collect_agent_ready_task_validation_traces(spec)]
                self.last_validation_traces = traces
                return spec, data
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            f"Model {self.model!r} failed after {1 + max_retries} attempts. Last error: {last_exc}"
        ) from last_exc

    def _system_prompt(self) -> str:
        return """\
You are an environment-generator for robot manipulation tasks.
Convert a natural-language prompt into an ArenaEnvGraphSpec.

GUIDANCE:
- Follow the per-field ``description`` strings in the schema.
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
- Only populate ``object_references`` when the prompt explicitly mentions surfaces or appliances
  inside the background; otherwise leave it unset.
- For each ``object_reference``, leave ``prim_path`` empty.
- REQUIRED: include an ``is_anchor`` relation on the resting surface (background or an
  ``object_reference`` within it).
- All objects need an ``on`` relation with that anchor as ``reference``.
- For repeated objects (e.g. five bananas into one bin), create separate object ids and one
  ``PickAndPlaceTask`` leaf per object under a ``parallel`` root task.
"""
