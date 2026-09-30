# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter generation and placement recording."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import MISSING

from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ClutterGenerationCfg:
    """Offline scene source, output and sampling settings."""

    env_spec: str | None = None
    """Environment graph YAML path for the command line."""
    output: str = MISSING
    """Output placement JSONL path; must not already exist."""
    num_envs: int = 1
    """Number of parallel physics environments."""
    num_layouts: int | None = None
    """Exact number of accepted layouts to save; None uses num_envs."""
    seed: int = 42
    """Seed for placement solving and reset randomization."""
    max_batches: int = 5
    """Maximum resets to sample before requiring the requested accepted count."""
    presets: str | None = None
    """Physics backend override: physx or newton."""
    render: bool = False
    """Render the offline physics steps when a visualizer is enabled."""
    settle: SettledPlacementParams = field(
        default_factory=lambda: SettledPlacementParams(validators=default_clutter_validators())
    )
    """Environment-step duration and named post-physics validator configurations."""


def load_generation_config(overrides: list[str]) -> ClutterGenerationCfg:
    """Load typed generation settings from Hydra key=value overrides."""
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    ConfigStore.instance().store(name="clutter_generation", node=ClutterGenerationCfg)
    with initialize(version_base=None, config_path=None):
        return OmegaConf.to_object(compose(config_name="clutter_generation", overrides=overrides))


def generate_clutter_layouts(
    arena_env: IsaacLabArenaEnvironment, cfg: ClutterGenerationCfg, device: str = "cuda:0"
) -> Path:
    """Build, sample and close an environment, writing exactly the requested accepted layouts.

    Call after SimulationApp startup. Uses ordinary pooled placement resets and leaves
    no output file if the batch budget is exhausted before enough candidates pass.
    Sampling replaces the description's placement seed, pool size and reset selection.

    Args:
        arena_env: Environment description containing ClutterOn relations.
        cfg: Output, batch budget and post-physics check settings.
        device: Simulation device, such as cuda:0 or cpu.

    Returns:
        The written placement JSONL path.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.recording import validate_recording_assets, write_settled_layouts
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import ClutterOn, get_relation

    output = Path(cfg.output)
    assert not output.exists(), f"Output already exists: {output}"
    assert arena_env.placement_layouts is None, "Remove cached placement layouts before generating clutter"
    assets = arena_env.get_placement_assets()
    assert not any(isinstance(asset, RigidObjectSet) for asset in assets), "Resolve object sets before generation"
    assert any(get_relation(asset, ClutterOn) is not None for asset in assets), "Generation requires ClutterOn objects"
    num_layouts = cfg.num_envs if cfg.num_layouts is None else cfg.num_layouts
    assert cfg.num_envs > 0 and num_layouts > 0 and cfg.max_batches > 0, "Counts and batch budget must be positive"
    assert cfg.max_batches * cfg.num_envs >= num_layouts, "Batch budget cannot supply the requested layout count"
    arena_env.placer_params = replace(
        arena_env.placer_params or ObjectPlacerParams(),
        placement_seed=cfg.seed,
        # Refill on reset instead of solving the entire output quota at startup.
        min_unique_layouts_per_env=1,
        resolve_on_reset=True,
    )
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=cfg.num_envs, seed=cfg.seed, device=device, presets=cfg.presets),
    )
    env = builder.make_registered()
    try:
        validate_recording_assets(env, assets)
        poses = {}
        outcomes = []
        rejections = {}
        attempted = 0
        for batch_index in range(cfg.max_batches):
            result = collect_settled_placements(
                env, 1, cfg.settle, render=cfg.render, scene_assets=assets, log_progress=True
            )
            attempted += result.attempted
            for (env_id, _), reason in result.rejections.items():
                rejections[env_id, batch_index] = reason
                print(f"  Rejected env {env_id}, batch {batch_index + 1}: {reason}")
            remaining = num_layouts - len(outcomes)
            for key, values in result.poses.items():
                poses.setdefault(key, []).extend(values[:remaining])
            outcomes.extend(result.validation[:remaining])
            print(f"[generation] batch {batch_index + 1}/{cfg.max_batches}: {len(outcomes)}/{num_layouts} collected")
            if len(outcomes) == num_layouts:
                break
        assert (
            len(outcomes) == num_layouts
        ), f"Accepted {len(outcomes)} layouts; need {num_layouts}. Rejections: {rejections}"
        write_settled_layouts(env, output, assets, poses, outcomes, cfg.settle.num_steps)
        print(f"Saved {len(outcomes)} accepted layouts from {attempted} candidates: {output}")
        return output
    finally:
        env.close()
