# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent for parsing natural-language env-generation prompts into an ArenaEnvGraphSpec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaaclab_arena.agentic_environment_generation.inference_backend import InferenceBackend
from isaaclab_arena.agentic_environment_generation.prim_path_inference import PrimPathInference
from isaaclab_arena.agentic_environment_generation.prompt_normalization import (
    PromptNormalizationInference,
    format_normalized_prompt_block,
)
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadyCandidateCatalogue,
    SimReadySearchConfig,
    search_simready_objects,
)
from isaaclab_arena.agentic_environment_generation.spec_inference import SpecInference
from isaaclab_arena.agentic_environment_generation.spec_validation import required_task_init_param_names
from isaaclab_arena.assets.registries import AssetRegistry, ObjectRelationLibraryRegistry, TaskRegistry
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.relations.relations import RelationBase

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
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
        *,
        enable_simready_search: bool = False,
        simready_config: SimReadySearchConfig | None = None,
    ):
        """Configure the OpenAI-compatible client and validate the model.

        Args:
            api_key: API token for the inference endpoint. Falls back
                to the ``NV_API_KEY`` environment variable.
            model: Model identifier at the inference endpoint.
                Must support OpenAI-compatible structured outputs.
            base_url: OpenAI-compatible inference endpoint.
            temperature: Sampling temperature forwarded to the model. Kept
                low by default (0.2) because spec generation is a
                deterministic-ish translation task — high temperature
                yields creative but invalid schemas.
            max_tokens: Hard cap on the response length.
            max_retries: Number of additional attempts after a recoverable failure
                (network errors, timeouts, empty responses, malformed JSON). Each
                retry is a fresh API call.
            enable_simready_search: When ``True``, run SimReady search on normalized object phrases.
            simready_config: Optional SimReady search configuration.
        """
        inference_backend = InferenceBackend(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        self.prompt_normalization = PromptNormalizationInference(inference_backend)
        self.spec_inference = SpecInference(inference_backend)
        self.prim_path_inference = PrimPathInference(inference_backend)
        self.enable_simready_search = enable_simready_search
        self.simready_config = simready_config or SimReadySearchConfig(enabled=enable_simready_search)
        self._traces: list[str] = []

    @property
    def traces(self) -> tuple[str, ...]:
        """Diagnostic lines from the most recent :meth:`generate_spec` call."""
        return tuple(self._traces)

    def generate_spec(
        self,
        prompt: str,
        asset_catalog: AssetCatalogue | None = None,
        relation_catalog: RelationCatalogue | None = None,
        task_catalog: TaskCatalogue | None = None,
        *,
        enable_simready_search: bool | None = None,
    ) -> tuple[ArenaEnvGraphSpec | None, dict[str, Any] | None]:
        """Call the model with user prompt and return the parsed ArenaEnvGraphSpec.

        Args:
            prompt: Natural-language env description from the end user.
            asset_catalog: Pre-built asset vocabulary. When ``None``, built
                from the live ``AssetRegistry``.
            relation_catalog: Pre-built relation vocabulary. When ``None``, built
                from the live ``ObjectRelationLibraryRegistry``.
            task_catalog: Pre-built task vocabulary. When ``None``, built from
                ``TaskRegistry`` tasks marked ``@agent_ready``.
            enable_simready_search: Override the agent-level SimReady search flag.

        Returns:
            A ``(spec, data)`` tuple. On success, ``spec`` is validated and
            ``data`` is None. On failure, ``spec`` is None and ``data`` is the corresponding JSON dict.
            When validation fails, ``agent.traces`` holds the diagnostic trace.
        """
        self._traces = []
        normalized = self.prompt_normalization.infer(prompt, self._traces)
        if normalized is None:
            return None, None

        normalized_block = format_normalized_prompt_block(normalized)
        self._traces.append(normalized_block)

        use_simready = self.enable_simready_search if enable_simready_search is None else enable_simready_search
        simready_catalog: SimReadyCandidateCatalogue | None = None
        if use_simready:
            simready_config = SimReadySearchConfig(
                enabled=True,
                source=self.simready_config.source,
                s3_url=self.simready_config.s3_url,
                service_url=self.simready_config.service_url,
                project_config_path=self.simready_config.project_config_path,
                indexed_path=self.simready_config.indexed_path,
                indexed_directory_type=self.simready_config.indexed_directory_type,
                max_results_per_object=self.simready_config.max_results_per_object,
                use_service_fallback=self.simready_config.use_service_fallback,
            )
            simready_catalog = search_simready_objects(normalized.objects, simready_config, self._traces)
            simready_block = simready_catalog.to_catalog_string()
            if simready_block:
                self._traces.append(simready_block)

        asset_catalog = asset_catalog or build_asset_catalogue()
        relation_catalog = relation_catalog or build_relation_catalogue()
        task_catalog = task_catalog or build_task_catalogue()
        spec, data = self.spec_inference.infer(
            prompt,
            self._traces,
            asset_catalog=asset_catalog,
            relation_catalog=relation_catalog,
            task_catalog=task_catalog,
            normalized_prompt_block=normalized_block,
            simready_candidate_catalog=simready_catalog,
        )
        if spec is None:
            return None, data
        if spec.object_references:
            resolved = self.prim_path_inference.infer(spec, self._traces)
            if resolved is None:
                return None, spec.to_dict()
            spec = resolved
        return spec, None


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
    # A list of object names, object types, and tags for agent to choose from.
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
            f"- {o['name']}  type={o['object_type']}  tags={o['tags']}"
            for o in sorted(self.objects, key=lambda o: o["name"])
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
            # Exposed so the agent can honour type constraints, e.g. object-set members must be rigid.
            object_type = getattr(cls, "object_type", None)
            catalogue.objects.append({
                "name": name,
                "tags": [t for t in tags if t != "object"],
                "object_type": object_type.value if object_type else "unknown",
            })
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
