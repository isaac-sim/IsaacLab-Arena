# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent for parsing natural-language env-generation prompts into an ArenaEnvGraphSpec."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from isaaclab_arena.agentic_environment_generation.catalogues import (
    AssetCatalogue,
    RelationCatalogue,
    TaskCatalogue,
    build_asset_catalogue,
    build_relation_catalogue,
    build_task_catalogue,
)
from isaaclab_arena.agentic_environment_generation.inference_backend import InferenceBackend
from isaaclab_arena.agentic_environment_generation.missing_object_inference import MissingObjectInference
from isaaclab_arena.agentic_environment_generation.prim_path_inference import PrimPathInference
from isaaclab_arena.agentic_environment_generation.simready_asset_search import (
    SimReadySearchConfig,
    search_simready_objects,
)
from isaaclab_arena.agentic_environment_generation.spec_inference import SpecInference
from isaaclab_arena.assets.simready_constants import SIMREADY_USD_OBJECT_REGISTRY_NAME
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

_logger = logging.getLogger(__name__)


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
        endpoint: str | None = None,
        *,
        enable_simready_search: bool = False,
        simready_config: SimReadySearchConfig | None = None,
    ):
        """Configure the OpenAI-compatible client and validate the model.

        Args:
            api_key: API token for the inference endpoint. Falls back to the environment
                variable the selected endpoint reads.
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
            endpoint: Inference endpoint name, ``internal``, ``public``, or ``openai``.
                Falls back to the ``ARENA_INFERENCE_ENDPOINT`` environment variable.
            enable_simready_search: When ``True``, search SimReady for objects the asset catalog
                does not cover.
            simready_config: Optional SimReady search configuration.
        """
        inference_backend = InferenceBackend(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            endpoint=endpoint,
        )
        self.spec_inference = SpecInference(inference_backend)
        self.missing_object_inference = MissingObjectInference(inference_backend)
        self.prim_path_inference = PrimPathInference(inference_backend)
        self.enable_simready_search = enable_simready_search
        self.simready_config = simready_config or SimReadySearchConfig()
        self._traces: list[str] = []
        self._unavailable_objects: list[str] = []
        self._simready_usd_paths: dict[str, str] = {}

    @property
    def traces(self) -> tuple[str, ...]:
        """Why the most recent :meth:`generate_spec` call failed, empty when it succeeded.

        Callers surface these as errors, so only what defeated the generation belongs here.
        Progress worth reading but not acting on is logged instead.
        """
        return tuple(self._traces)

    @property
    def unavailable_objects(self) -> tuple[str, ...]:
        """Objects the most recent ``generate_spec`` call needed but found no asset for.

        Only the SimReady search reports these. A catalog object always names a registry entry that
        exists, so it can be the wrong choice but never a missing one. The generated spec is still
        valid: these objects were never offered to spec inference, so it built the scene without
        them.
        """
        return tuple(self._unavailable_objects)

    def generate_spec(
        self,
        prompt: str,
        asset_catalog: AssetCatalogue | None = None,
        relation_catalog: RelationCatalogue | None = None,
        task_catalog: TaskCatalogue | None = None,
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

        Returns:
            A ``(spec, data)`` tuple. On success, ``spec`` is validated and
            ``data`` is None. On failure, ``spec`` is None and ``data`` is the corresponding JSON dict.
            When validation fails, ``agent.traces`` holds the diagnostic trace. ``agent.unavailable_objects``
            names any object the search found nothing for; the spec is built without it.
        """
        self._traces = []
        self._unavailable_objects = []
        self._simready_usd_paths = {}
        asset_catalog = asset_catalog or build_asset_catalogue()
        relation_catalog = relation_catalog or build_relation_catalogue()
        task_catalog = task_catalog or build_task_catalogue()
        if self.enable_simready_search:
            asset_catalog = self._extend_catalogue_with_simready(prompt, asset_catalog)
        spec, data = self.spec_inference.infer(
            prompt,
            self._traces,
            asset_catalog=asset_catalog,
            relation_catalog=relation_catalog,
            task_catalog=task_catalog,
        )
        if spec is None:
            return None, data
        if spec.object_references:
            resolved = self.prim_path_inference.infer(spec, self._traces)
            if resolved is None:
                return None, spec.to_dict()
            spec = resolved
        unusable = self._add_simready_usd_path_to_searched_objects(spec)
        if unusable is not None:
            self._traces.append(unusable)
            return None, spec.to_dict()
        return spec, None

    def _add_simready_usd_path_to_searched_objects(self, spec: ArenaEnvGraphSpec) -> str | None:
        """Point the spec's searched objects at the generic SimReady asset, USD path in params.

        A search name only exists in the process that searched, so a spec keeping it loads nowhere else.

        Args:
            spec: Generated spec, rewritten in place.

        Returns:
            An error message, or None when every searched object was rewritten.
        """
        # TODO(xinjieyao, 2026-08-03): Lift this once ObjectSetSpec.members can carry a usd_path.
        for object_set in spec.object_sets or []:
            searched_members = [name for name in object_set.members if name in self._simready_usd_paths]
            if searched_members:
                return (
                    f"Object set '{object_set.id}' has searched SimReady members {searched_members}."
                    " Members have nowhere to carry a usd_path; use them as objects instead."
                )
        for obj in spec.objects:
            usd_path = self._simready_usd_paths.get(obj.registry_name)
            if usd_path is not None:
                obj.registry_name = SIMREADY_USD_OBJECT_REGISTRY_NAME
                obj.params = {**obj.params, "usd_path": usd_path}
        return None

    def _extend_catalogue_with_simready(self, prompt: str, asset_catalog: AssetCatalogue) -> AssetCatalogue:
        """Search SimReady for the objects the catalog misses, and add what it finds to the catalog.

        Args:
            prompt: Natural-language env description, used to work out what the catalog misses.
            asset_catalog: Registered asset vocabulary to extend.

        Returns:
            A copy of the catalog with one added entry per asset found, or the argument itself
            when nothing was searched for or nothing was found.
        """
        # Imported here rather than at module scope: it pulls in the asset base classes, which
        # import pxr, and a pxr import before SimulationApp starts breaks the unit tests.
        from isaaclab_arena.assets.simready_object_library import register_searched_simready_object

        phrases = self.missing_object_inference.infer(prompt, asset_catalog)
        if not phrases:
            return asset_catalog
        search_result = search_simready_objects(phrases, self.simready_config)
        self._unavailable_objects = list(search_result.unmatched_phrases)
        if not search_result.candidates:
            return asset_catalog
        extended = replace(asset_catalog, objects=list(asset_catalog.objects))
        for candidate in search_result.candidates:
            asset_cls = register_searched_simready_object(candidate.search_phrase, candidate.usd_path, candidate.tags)
            tags = [tag for tag in asset_cls.tags if tag != "object"]
            extended.objects.append({
                "name": asset_cls.name,
                "tags": tags,
                "object_type": asset_cls.object_type.value,
            })
            self._simready_usd_paths[asset_cls.name] = candidate.usd_path
            _logger.info("catalogued %r for %r: %s", asset_cls.name, candidate.search_phrase, candidate.usd_path)
        return extended
