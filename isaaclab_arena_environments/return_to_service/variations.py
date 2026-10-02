# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recorded episode conditions applied before the service runtime initializes."""

from __future__ import annotations

import torch
from dataclasses import field

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.agentic_environment_generation.authoring_metadata import AuthoringMetadata, ParameterMetadata
from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

from .scenarios import SCENARIOS


@configclass
class ServiceScenarioVariationCfg(VariationBaseCfg):
    """Select the allowed fault assignments; fixed scenarios remain the default when disabled."""

    scenario_names: list[str] = field(default_factory=lambda: list(SCENARIOS))
    """Named fault assignments sampled uniformly for each resetting environment."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Recorded categorical draws using Arena's seed and replay context."""


def apply_service_scenario(env, env_ids, sampler: ChoiceSampler, scenario_names: tuple[str, ...]) -> None:
    """Set selected conditions before TaskRuntime.reset establishes their physical state."""
    runtime = env.task_runtime
    assert runtime is not None and len(runtime.scenario_names) == env.num_envs, "Service runtime is not initialized."
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    selected = sampler.sample(len(env_ids), choices=scenario_names, env_ids=env_ids)
    for env_id, name in zip(env_ids, selected, strict=True):
        runtime.scenario_names[int(env_id)] = name


class ServiceScenarioVariation(RunTimeVariationBase):
    """Sample hidden fault assignments and record them without changing the public work order."""

    cfg: ServiceScenarioVariationCfg

    authoring_metadata = AuthoringMetadata(
        configuration={
            "scenario_names": ParameterMetadata(description="Allowed evaluator-only faults; not policy observations."),
        },
        constraints=(
            "All choices must be registered service scenarios with the same public work order.",
            "Recorded fault assignments belong to evaluation artifacts and must not be exposed to the policy.",
        ),
        reset_semantics="Select conditions after scene reset; service runtime applies them after all variation events.",
    )

    def __init__(self, cfg: ServiceScenarioVariationCfg | None = None):
        super().__init__(cfg or ServiceScenarioVariationCfg(), name="scenario")

    def validate_cfg(self) -> None:
        names = self.cfg.scenario_names
        assert names and all(name in SCENARIOS for name in names), "Select known service scenario names."
        assert len(set(names)) == len(names), "Scenario choices must not repeat."
        instructions = {SCENARIOS[name].work_order for name in names}
        assert len(instructions) == 1, "Scenario variation must preserve one public work order."

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        self.validate_cfg()
        return "return_to_service_scenario", EventTermCfg(
            func=apply_service_scenario,
            mode="reset",
            params={"sampler": self.sampler, "scenario_names": tuple(self.cfg.scenario_names)},
        )
