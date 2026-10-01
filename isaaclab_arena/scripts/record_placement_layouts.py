# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record root poses that pass post-physics placement checks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.recording import PlacementRecordingSummary
from isaaclab_arena.offline_placement.recording_config import (
    PlacementRecordingCfg,
    load_recording_config,
    resolved_num_layouts,
)

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def replace_placer_params(placer_params: ObjectPlacerParams, cfg: PlacementRecordingCfg) -> ObjectPlacerParams:
    """Copy solver settings with the recording seed and pool requirements.

    Args:
        placer_params: Source settings, left unchanged.
        cfg: Recording seed and refill tranche size per environment.
    """
    return replace(
        placer_params,
        placement_seed=cfg.seed,
        min_unique_layouts_per_env=cfg.layouts_per_env,
        resolve_on_reset=True,
    )


def _assert_recording_cfg(cfg: PlacementRecordingCfg) -> int:
    """Validate recording counts and return the resolved layout target."""
    num_layouts = resolved_num_layouts(cfg)
    assert cfg.num_envs > 0 and cfg.layouts_per_env > 0, "Environment and layout counts must be positive"
    assert num_layouts > 0 and cfg.max_batches > 0, "Layout target and batch budget must be positive"
    assert (
        cfg.max_batches * cfg.num_envs >= num_layouts
    ), "Batch budget cannot supply the requested accepted layout count"
    assert (cfg.viewer_eye is None) == (cfg.viewer_lookat is None), "Set viewer_eye and viewer_lookat together"
    return num_layouts


def _load_recording_arena_env(cfg: PlacementRecordingCfg) -> IsaacLabArenaEnvironment:
    """Build an environment description from the recording YAML path."""
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    spec = ArenaEnvGraphSpec.from_yaml(cfg.env_spec)
    assert not spec.object_sets, "Resolve object sets before recording reusable layouts"
    return spec.to_arena_env()


def _validate_recording_arena_env(arena_env: IsaacLabArenaEnvironment) -> None:
    """Require a fresh, concrete scene description before building the sim."""
    from isaaclab_arena.assets.object_set import RigidObjectSet

    assert arena_env.placement_layouts is None, "Remove cached placement layouts before recording"
    assets = arena_env.get_placement_assets()
    assert not any(isinstance(asset, RigidObjectSet) for asset in assets), "Resolve object sets before recording"


def record_placements_to_jsonl(
    env: gym.Env,
    output: str | Path,
    *,
    num_layouts: int,
    max_batches: int,
    params: PlacementRecordingParams | None = None,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
) -> PlacementRecordingSummary:
    """Collect poses iteratively, write JSONL when the layout target is met.

    Each outer batch resets every environment once and runs one settle pass.
    The caller owns the environment; it stays open at its final state on success
    or failure. If the batch budget is exhausted before ``num_layouts`` accepts,
    returns a summary with output=None and leaves the destination unwritten.

    Args:
        env: Built environment with a pooled placement reset event.
        output: JSONL destination; must not exist.
        num_layouts: Accepted layouts required before writing.
        max_batches: Maximum reset-and-settle rounds.
        params: Simulation duration and post-physics validators.
        render: Render the offline physics steps.
        scene_assets: Asset definitions for scene roots outside the placement pool.
    """
    from isaaclab_arena.offline_placement.recording import (
        collect_layouts_until_count,
        resolve_settle_params,
        validate_recording_assets,
        write_settled_layouts,
    )
    from isaaclab_arena.relations.placement_events import get_placement_pool

    output = Path(output)
    assert not output.exists(), f"Output already exists: {output}"
    pool = get_placement_pool(env)
    assert pool is not None, "Recording requires a pooled placement reset event"
    assets = list(pool.objects)
    for asset in scene_assets or []:
        if asset not in assets:
            assets.append(asset)
    validate_recording_assets(env, assets)
    settle = resolve_settle_params(assets, params)
    poses, validation, attempted, rejections = collect_layouts_until_count(
        env,
        num_layouts,
        max_batches,
        settle,
        render=render,
        scene_assets=assets,
    )
    summary = PlacementRecordingSummary(
        output=None,
        accepted=len(validation),
        attempted=attempted,
        rejections=rejections,
    )
    if summary.accepted < num_layouts:
        return summary
    write_settled_layouts(env, output, assets, poses, validation, settle.num_steps)
    summary.output = output
    return summary


def record_settled_placement_layouts(
    cfg: PlacementRecordingCfg,
    *,
    device: str = "cuda:0",
    arena_env: IsaacLabArenaEnvironment | None = None,
) -> PlacementRecordingSummary:
    """Build an environment, collect accepted poses, and close the simulation.

    Call after starting SimulationApp. Pass ``arena_env`` to record from an
    in-memory description; otherwise ``cfg.env_spec`` loads the scene YAML.
    A summary with output=None means the layout target was not reached.

    Args:
        cfg: Sampling, validation and output settings.
        device: Simulation device.
        arena_env: Optional pre-built environment description.

    Returns:
        Output path, acceptance counts and per-candidate rejection reasons.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    num_layouts = _assert_recording_cfg(cfg)
    if arena_env is None:
        arena_env = _load_recording_arena_env(cfg)
    _validate_recording_arena_env(arena_env)
    arena_env.placer_params = replace_placer_params(arena_env.placer_params, cfg)
    scene_assets = arena_env.get_placement_assets()
    print(f"[recording] Solving placements for {cfg.num_envs} environments...", flush=True)
    env = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=cfg.num_envs, env_spacing=cfg.env_spacing, seed=cfg.seed, device=device, presets=cfg.presets
        ),
    ).make_registered()
    try:
        if cfg.viewer_eye is not None:
            env.unwrapped.sim.set_camera_view(cfg.viewer_eye, cfg.viewer_lookat)
        return record_placements_to_jsonl(
            env,
            cfg.output,
            num_layouts=num_layouts,
            max_batches=cfg.max_batches,
            params=cfg.settle,
            render=cfg.render,
            scene_assets=scene_assets,
        )
    finally:
        env.close()


def main() -> None:
    """Run offline recording with Hydra settings and Isaac Lab launcher flags."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args, overrides = parser.parse_known_args()
    assert_hydra_overrides(overrides, parser)
    cfg = load_recording_config(overrides)
    target = resolved_num_layouts(cfg)
    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    with SimulationAppContext(launcher_args):
        summary = record_settled_placement_layouts(cfg, device=launcher_args.device)
        for reason, count in Counter(summary.rejections.values()).items():
            print(f"  Rejected {count}: {reason}")
        assert (
            summary.output is not None
        ), f"Accepted {summary.accepted} layouts; need {target}. Rejections: {summary.rejections}"
        print(f"Saved {summary.accepted}/{summary.attempted} accepted layouts: {summary.output}")


if __name__ == "__main__":
    main()
