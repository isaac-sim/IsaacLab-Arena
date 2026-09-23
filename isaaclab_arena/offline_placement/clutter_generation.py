# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter generation and placement recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import MISSING

from isaaclab_arena.offline_placement.settle_params import ClutterSettleParams
from isaaclab_arena.offline_placement.validators import default_post_physics_validators

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
    """Total layouts to generate; None generates one per environment."""
    seed: int = 42
    """Initial release-sampling seed."""
    attempts: int = 5
    """Maximum settling trials per environment."""
    presets: str | None = None
    """Physics backend override: physx or newton."""
    register: list[str] = field(default_factory=list)
    """Optional component registration entry points, each module:function."""
    settle: ClutterSettleParams = field(default_factory=ClutterSettleParams)
    """Physics time budget and sampling interval."""
    post_physics: dict[str, dict] = field(default_factory=default_post_physics_validators)
    """Named Hydra validator configurations; every enabled check must pass."""


def generate_clutter_layouts(
    arena_env: IsaacLabArenaEnvironment, cfg: ClutterGenerationCfg, device: str = "cuda:0"
) -> Path:
    """Generate accepted placement records from offline physics and return the output path.

    Args:
        arena_env: Environment description with ClutterOn relations.
        cfg: Output and settling settings.
        device: Simulation device, such as cuda:0 or cpu.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.geometry import dynamic_rigid_object_keys
    from isaaclab_arena.offline_placement.settle import groups_from_assets, settle_clutter
    from isaaclab_arena.offline_placement.validators import build_post_physics_validators
    from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts

    assert arena_env.placement_layouts is None, "Remove cached placement layouts before generating clutter"

    validators = build_post_physics_validators(cfg.post_physics)
    output = Path(cfg.output)
    assert not output.exists(), f"Output already exists: {output}"
    assets = arena_env.get_placement_assets()
    assert not has_heterogeneous_objects(assets), "Resolve object sets to concrete assets before offline settling"
    groups_from_assets(assets)
    placer_params = arena_env.placer_params or ObjectPlacerParams()
    num_layouts = cfg.num_layouts if cfg.num_layouts is not None else cfg.num_envs
    assert (
        cfg.num_envs > 0 and num_layouts > 0 and cfg.attempts > 0
    ), "num_envs, num_layouts and attempts must be positive"
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=cfg.num_envs,
            seed=cfg.seed,
            device=device,
            presets=cfg.presets,
            solve_relations=False,
        ),
    )
    env = builder.make_registered()
    try:
        keys = dynamic_rigid_object_keys(env.unwrapped.scene)
        env.reset()
        layouts = []
        for start in range(0, num_layouts, cfg.num_envs):
            batch = settle_clutter(
                env,
                assets,
                seed=cfg.seed + start * cfg.attempts * placer_params.max_placement_attempts,
                attempts=cfg.attempts,
                params=cfg.settle,
                placer_params=placer_params,
                validators=validators,
            )
            layouts.extend(batch[: num_layouts - start])
        cache = PlacementLayouts({key: [layout.poses[key] for layout in layouts] for key in keys})
        source = (
            "settled" if any(validator.check == "rest" and validator.enabled for validator in validators) else "physics"
        )
        cache.write_episode_jsonl(output, source=source, validation=[layout.validation for layout in layouts])
        print(f"Saved {cache.num_layouts} accepted layouts: {output}")
        return output
    finally:
        env.close()
