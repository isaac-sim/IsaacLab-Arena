# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record root poses that pass post-physics placement checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import MISSING

from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams

if TYPE_CHECKING:
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams


@dataclass
class PlacementRecordingCfg:
    """Source scene, reset sampling and offline recording settings."""

    env_spec: str = MISSING
    """Environment YAML path."""
    output: str = MISSING
    """Placement JSONL output path; must not exist."""
    num_envs: int = 1
    """Number of parallel simulation environments."""
    env_spacing: float = 30.0
    """Distance between parallel environment origins, in metres."""
    viewer_eye: tuple[float, float, float] | None = None
    """Optional viewer position in simulation-world metres; requires viewer_lookat."""
    viewer_lookat: tuple[float, float, float] | None = None
    """Optional viewer target in simulation-world metres; requires viewer_eye."""
    layouts_per_env: int = 5
    """Number of reset placements sampled per environment before physics filtering."""
    seed: int = 42
    """Seed for placement solving and reset randomization."""
    presets: str | None = None
    """Optional physics backend override: physx or newton."""
    render: bool = False
    """Render physics steps when a visualizer is enabled."""
    settle: PlacementRecordingParams = field(default_factory=PlacementRecordingParams)
    """Physics duration, configured validators and minimum accepted count."""


def replace_placer_params(placer_params: ObjectPlacerParams, cfg: PlacementRecordingCfg) -> ObjectPlacerParams:
    """Copy solver settings with the recording seed and pool requirements.

    Args:
        placer_params: Source settings, left unchanged.
        cfg: Recording seed and minimum candidate count per environment.
    """
    return replace(
        placer_params,
        placement_seed=cfg.seed,
        min_unique_layouts_per_env=cfg.layouts_per_env,
        resolve_on_reset=True,
    )


def record_placements_to_jsonl(cfg: PlacementRecordingCfg, device: str = "cuda:0") -> Path:
    """Write accepted post-physics placement poses and return the output JSONL path.

    Args:
        cfg: Source, candidate count and filtering configuration.
        device: Simulation device.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements

    output = Path(cfg.output)
    assert not output.exists(), f"Output already exists: {output}"
    assert cfg.num_envs > 0 and cfg.layouts_per_env > 0, "Environment and layout counts must be positive"
    assert (cfg.viewer_eye is None) == (cfg.viewer_lookat is None), "Set viewer_eye and viewer_lookat together"
    spec = ArenaEnvGraphSpec.from_yaml(cfg.env_spec)
    assert not spec.object_sets, "Resolve object sets before recording reusable layouts"
    arena_env = spec.to_arena_env()
    arena_env.placer_params = replace_placer_params(arena_env.placer_params, cfg)
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=cfg.num_envs, env_spacing=cfg.env_spacing, seed=cfg.seed, device=device, presets=cfg.presets
        ),
    )
    print(f"[recording] Solving placements for {cfg.num_envs} environments...", flush=True)
    env = builder.make_registered()
    try:
        if cfg.viewer_eye is not None:
            env.unwrapped.sim.set_camera_view(cfg.viewer_eye, cfg.viewer_lookat)
        result = collect_settled_placements(
            env,
            cfg.layouts_per_env,
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
