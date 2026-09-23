# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Solve placements, filter them with physics, and record complete settled poses."""

from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from omegaconf import MISSING

from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams


@dataclass
class PlacementRecordingCfg:
    """Source scene, candidate pool and offline recording settings."""

    env_spec: str = MISSING
    """Environment graph YAML path."""
    output: str = MISSING
    """Placement JSONL output path; must not exist."""
    num_envs: int = 1
    """Number of parallel simulation environments."""
    layouts_per_env: int = 5
    """Minimum solved candidates per environment before physics filtering."""
    seed: int = 42
    """Seed for candidate placement."""
    presets: str | None = None
    """Optional physics backend override: physx or newton."""
    register: list[str] = field(default_factory=list)
    """Component registration entry points, each module:function."""
    render: bool = False
    """Render physics steps when a visualizer is enabled."""
    settle: PlacementRecordingParams = field(default_factory=PlacementRecordingParams)
    """Physics duration, configured validators and minimum accepted count."""


def record_placements(cfg: PlacementRecordingCfg, device: str = "cuda:0") -> Path:
    """Write physics-filtered final poses for a graph scene and return the output path.

    Args:
        cfg: Source, candidate count and filtering configuration.
        device: Simulation device.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_pool_layouts
    from isaaclab_arena.relations.placement_events import get_placement_pool

    output = Path(cfg.output)
    assert not output.exists(), f"Output already exists: {output}"
    assert cfg.num_envs > 0 and cfg.layouts_per_env > 0, "Environment and layout counts must be positive"
    for entry in cfg.register:
        module, separator, function = entry.partition(":")
        assert separator and module and function, "register entries must be module:function"
        getattr(importlib.import_module(module), function)()
    spec = ArenaEnvGraphSpec.from_yaml(cfg.env_spec)
    assert not spec.object_sets, "Resolve object sets before recording reusable layouts"
    arena_env = spec.to_arena_env()
    arena_env.placer_params = replace(
        arena_env.placer_params,
        placement_seed=cfg.seed,
        min_unique_layouts_per_env=cfg.layouts_per_env,
        resolve_on_reset=True,
    )
    builder = ArenaEnvBuilder(
        arena_env, ArenaEnvBuilderCfg(num_envs=cfg.num_envs, seed=cfg.seed, device=device, presets=cfg.presets)
    )
    env = builder.make_registered()
    try:
        env.reset()
        pool = get_placement_pool(env)
        assert pool is not None, "Scene must contain placement relations"
        result = collect_settled_pool_layouts(
            env,
            pool,
            cfg.settle,
            render=cfg.render,
            scene_assets=arena_env.get_placement_assets(),
        )
    finally:
        env.close()
    result.layouts.write_episode_jsonl(output, source="settled", validation=result.validation)
    print(f"Saved {result.layouts.num_layouts}/{result.attempted} accepted layouts: {output}")
    for reason, count in Counter(result.rejections.values()).items():
        print(f"  Rejected {count}: {reason}")
    return output
