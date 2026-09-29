# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record root poses that pass post-physics placement checks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.offline_placement.recording import PlacementRecordingSummary
from isaaclab_arena.offline_placement.recording_config import PlacementRecordingCfg, load_recording_config

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


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


def create_env(cfg: PlacementRecordingCfg, device: str = "cuda:0") -> tuple[gym.Env, list[PlaceableAsset]]:
    """Build a recording environment and configure its view after SimulationApp startup.

    Args:
        cfg: Source scene, placement sampling and environment settings.
        device: Simulation device.

    Returns:
        Configured environment owned by the caller and its scene asset definitions.
        The environment is closed if viewer setup fails.
    """
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

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
    scene_assets = arena_env.get_placement_assets()
    print(f"[recording] Solving placements for {cfg.num_envs} environments...", flush=True)
    env = builder.make_registered()
    try:
        if cfg.viewer_eye is not None:
            env.unwrapped.sim.set_camera_view(cfg.viewer_eye, cfg.viewer_lookat)
    except BaseException:
        env.close()
        raise
    return env, scene_assets


def record_placements_to_jsonl(
    env: gym.Env,
    output: str | Path,
    num_batches: int,
    params: PlacementRecordingParams | None = None,
    *,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
) -> PlacementRecordingSummary:
    """Collect poses, write JSONL when the minimum yield is met, and return a summary.

    Each batch resets the environment and consumes one placement per environment.
    The caller owns the environment; it stays open at its final state on success
    or failure. Insufficient acceptance returns a summary with output=None and
    rejection reasons, leaving the destination unwritten.

    Args:
        env: Built environment with a pooled placement reset event.
        output: JSONL destination; must not exist.
        num_batches: Number of resets to sample, independent of pool refills.
        params: Simulation duration, post-physics validators and minimum yield.
        render: Render the offline physics steps.
        scene_assets: Asset definitions for scene roots outside the placement pool.
    """
    from isaaclab_arena.offline_placement.recording import validate_recording_assets
    from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
    from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts

    output = Path(output)
    assert not output.exists(), f"Output already exists: {output}"
    if params is None:
        params = PlacementRecordingParams()
    pool = get_placement_pool(env)
    assert pool is not None, "Recording requires a pooled placement reset event"
    assets = list(pool.objects)
    for asset in scene_assets or []:
        if asset not in assets:
            assets.append(asset)
    validate_recording_assets(env, assets)
    result = collect_settled_placements(env, num_batches, params, render=render, scene_assets=assets, log_progress=True)
    summary = PlacementRecordingSummary(
        output=None,
        accepted=len(result.accepted_indices),
        attempted=result.attempted,
        rejections=result.rejections,
    )
    if summary.accepted < params.min_layouts:
        return summary
    layouts = PlacementLayouts(result.poses)
    layouts.validate_assets(assets)
    embodiment_keys = []
    for asset in assets:
        if asset.tags and "embodiment" in asset.tags:
            embodiment_keys.extend(asset.get_scene_root_keys())
    sampling = {
        "num_steps": params.num_steps,
        "decimation": env.unwrapped.cfg.decimation,
        "physics_dt_s": env.unwrapped.sim.get_physics_dt(),
        "embodiment_keys": embodiment_keys,
    }
    validation = []
    for outcome in result.validation:
        validation.append({
            "pre_physics": outcome.pre_physics,
            "post_physics": [asdict(report) for report in outcome.post_physics],
            "sampling": sampling,
        })
    layouts.write_episode_jsonl(output, source="settled", validation=validation)
    summary.output = output
    return summary


def record_settled_placement_layouts(
    cfg: PlacementRecordingCfg, *, device: str = "cuda:0"
) -> PlacementRecordingSummary:
    """Build an environment, collect and save poses, then close the environment.

    Call after starting SimulationApp. A summary with output=None means the
    minimum accepted count was not reached and no file was written.

    Args:
        cfg: Environment, sampling, validation and output settings.
        device: Simulation device.

    Returns:
        Output path, acceptance counts and per-candidate rejection reasons.
    """
    assert not Path(cfg.output).exists(), f"Output already exists: {cfg.output}"
    env, scene_assets = create_env(cfg, device=device)
    try:
        return record_placements_to_jsonl(
            env,
            cfg.output,
            cfg.layouts_per_env,
            cfg.settle,
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
        assert (
            summary.output is not None
        ), f"Accepted {summary.accepted} layouts; need {cfg.settle.min_layouts}. Rejections: {summary.rejections}"
        print(f"Saved {summary.accepted}/{summary.attempted} accepted layouts: {summary.output}")


if __name__ == "__main__":
    main()
