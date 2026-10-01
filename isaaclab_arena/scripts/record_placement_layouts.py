# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record root poses that pass post-physics placement checks."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.recording import PlacementRecordingSummary
from isaaclab_arena.offline_placement.recording_config import PlacementRecordingCfg, load_recording_config

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def record_placements_to_jsonl(
    env: gym.Env,
    output: str | Path,
    *,
    min_layouts: int,
    max_batches: int,
    params: SettledPlacementParams | None = None,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
) -> PlacementRecordingSummary:
    """Collect poses up to the layout target or batch budget and write accepted layouts.

    Each outer batch resets every environment once and runs one settle pass.
    The caller owns the environment; it stays open at its final state on success
    or failure. If the batch budget is exhausted, every accepted layout is
    written. The destination remains unwritten only when no layouts are accepted.

    Args:
        env: Built environment with a pooled placement reset event.
        output: JSONL destination; must not exist.
        min_layouts: Target number of accepted layouts to collect.
        max_batches: Maximum reset-and-settle rounds.
        params: Simulation duration and post-physics validators.
        render: Render the offline physics steps.
        scene_assets: Asset definitions for scene roots outside the placement pool.
    """
    from isaaclab_arena.offline_placement.recording import (
        collect_layouts_until_count,
        validate_recording_assets,
        write_settled_layouts,
    )
    from isaaclab_arena.offline_placement.settled_placement import resolve_settle_params
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
        min_layouts,
        max_batches,
        settle,
        render=render,
        scene_assets=assets,
    )
    accepted = len(validation)
    summary = PlacementRecordingSummary(output=None, accepted=accepted, attempted=attempted, rejections=rejections)
    if accepted == 0:
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
    A summary with output=None means no layouts were accepted.

    Args:
        cfg: Sampling, validation and output settings.
        device: Simulation device.
        arena_env: Optional pre-built environment description.

    Returns:
        Output path, acceptance counts and per-candidate rejection reasons.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environment_spec.placer_params_cfg_override import build_placer_params_from_override
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    assert cfg.num_envs > 0 and cfg.layouts_per_env > 0, "Environment and layout counts must be positive"
    assert cfg.min_layouts > 0 and cfg.max_batches > 0, "Layout target and batch budget must be positive"
    assert (
        cfg.max_batches * cfg.num_envs >= cfg.min_layouts
    ), "Batch budget cannot supply the requested accepted layout count"
    assert (cfg.viewer_eye is None) == (cfg.viewer_lookat is None), "Set viewer_eye and viewer_lookat together"

    if arena_env is None:
        spec = ArenaEnvGraphSpec.from_yaml(cfg.env_spec)
        assert not spec.object_sets, "Resolve object sets before recording reusable layouts"
        arena_env = spec.to_arena_env()
    assert arena_env.placement_layouts is None, "Remove cached placement layouts before recording"
    scene_assets = arena_env.get_placement_assets()
    assert not any(isinstance(asset, RigidObjectSet) for asset in scene_assets), "Resolve object sets before recording"
    placer_params = arena_env.placer_params
    if placer_params is None:
        placer_params = build_placer_params_from_override(None)
    arena_env.placer_params = replace(
        placer_params,
        placement_seed=cfg.seed,
        min_unique_layouts_per_env=cfg.layouts_per_env,
        resolve_on_reset=True,
    )

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
            min_layouts=cfg.min_layouts,
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
    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    with SimulationAppContext(launcher_args):
        summary = record_settled_placement_layouts(cfg, device=launcher_args.device)
        for reason, count in Counter(summary.rejections.values()).items():
            print(f"  Rejected {count}: {reason}")
        if summary.accepted < cfg.min_layouts:
            if summary.output is None:
                logging.error(
                    "No layouts were accepted; no recording was written (requested %d).",
                    cfg.min_layouts,
                )
            else:
                logging.error(
                    "Only %d of %d requested layouts were recorded: %s",
                    summary.accepted,
                    cfg.min_layouts,
                    summary.output,
                )
            return
        print(f"Saved {summary.accepted}/{summary.attempted} accepted layouts: {summary.output}")


if __name__ == "__main__":
    main()
