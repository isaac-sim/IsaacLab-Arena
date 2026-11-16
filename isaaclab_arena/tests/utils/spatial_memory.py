# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks shared by the spatial-memory suite tests; every function runs inside a SimulationApp."""

from __future__ import annotations

import torch
from typing import Any

NUM_SEEDS = 50
"""Episodes (one per parallel environment, each with its own layout seed) per check."""

ORACLE_MIN_SUCCESS_RATE = 0.9
BLIND_MAX_SUCCESS_RATE = 0.3
STATIC_TOLERANCE_M = 0.005
"""Largest displacement of any static part while the task-relevant geometry is hidden."""

SEED = 1234


def _rollout(env_name: str, mode: str, num_seeds: int, difficulty: float | None, seed: int) -> list[dict[str, Any]]:
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env, rollout_episodes
    from isaaclab_arena_environments.spatial_memory.policies import SpatialMemoryBlindPolicy, SpatialMemoryOraclePolicy

    env = make_spatial_memory_env(env_name, num_envs=num_seeds, seed=seed, difficulty=difficulty)
    policy = SpatialMemoryOraclePolicy() if mode == "oracle" else SpatialMemoryBlindPolicy()
    try:
        records = rollout_episodes(env, policy, num_seeds)
    finally:
        env.close()
    assert len(records) == num_seeds, f"Expected {num_seeds} finished episodes, got {len(records)}."
    return records


def _describe_failures(records: list[dict[str, Any]], limit: int = 5) -> str:
    lines = []
    for record in records:
        if record.get("success"):
            continue
        sm = record["spatial_memory"]
        lines.append(
            f"env {record['env_id']}: len={record['episode_length']} collision={sm['first_collision']} "
            f"searched_again={sm['searched_again']} layout={sm['layout']}"
        )
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def check_oracle_solves_with_hidden_geometry(
    env_name: str,
    num_seeds: int = NUM_SEEDS,
    difficulty: float | None = None,
    check_static: bool = True,
    seed: int = SEED,
    seen_earlier_targets: list[str] | None = None,
    seen_earlier_fraction: float = 0.9,
) -> dict[str, float]:
    """Assert the oracle solves >= 90% of seeds and the hidden targets stay hidden (and static) in every action phase.

    Also asserts that each target in ``seen_earlier_targets`` was visible before the action phase in at least
    ``seen_earlier_fraction`` of the episodes, i.e. that the hidden information was observable earlier.

    Returns:
        The rollout summary (success, collision and searched-again rates).
    """
    from isaaclab_arena_environments.spatial_memory.evaluation import summarise

    records = _rollout(env_name, "oracle", num_seeds, difficulty, seed)
    summary = summarise(records)
    print(f"[spatial_memory] {env_name} oracle: {summary}")
    assert summary["success_rate"] >= ORACLE_MIN_SUCCESS_RATE, (
        f"Oracle success rate {summary['success_rate']:.2f} < {ORACLE_MIN_SUCCESS_RATE} on {env_name}.\n"
        + _describe_failures(records)
    )
    # The hidden-geometry checks describe the task, so they are made on the episodes the oracle solved.
    for record in records:
        if not record["success"]:
            continue
        sm = record["spatial_memory"]
        assert sm["action_phase_s"] > 0.0, f"env {record['env_id']}: succeeded without an action phase."
        seen = sm["hidden_targets_seen_in_action_phase"]
        assert not seen, f"env {record['env_id']}: hidden targets {seen} were visible during the action phase."
        if check_static:
            assert (
                sm["static_max_displacement_m"] < STATIC_TOLERANCE_M
            ), f"env {record['env_id']}: static geometry moved {sm['static_max_displacement_m']:.4f} m while hidden."
    for target in seen_earlier_targets or []:
        seen = [target in record["spatial_memory"]["targets_seen_before_action_phase"] for record in records]
        fraction = sum(seen) / len(seen)
        print(f"[spatial_memory] {env_name}: '{target}' seen before the action phase in {fraction:.2f} of episodes")
        assert fraction >= seen_earlier_fraction, f"'{target}' was seen earlier in only {fraction:.2f} of episodes."
    return summary


def check_blind_controller_fails(
    env_name: str, num_seeds: int = NUM_SEEDS, difficulty: float | None = None, seed: int = SEED
) -> dict[str, float]:
    """Assert the blind controller fails on at least 70% of seeds."""
    from isaaclab_arena_environments.spatial_memory.evaluation import summarise

    records = _rollout(env_name, "blind", num_seeds, difficulty, seed)
    summary = summarise(records)
    print(f"[spatial_memory] {env_name} blind: {summary}")
    assert (
        summary["success_rate"] <= BLIND_MAX_SUCCESS_RATE
    ), f"Blind controller succeeds on {summary['success_rate']:.2f} of seeds on {env_name}; the task is too easy."
    return summary


def _reset_and_read_layout(env_name: str, seed: int, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env
    from isaaclab_arena_environments.spatial_memory.runtime import get_runtime

    env = make_spatial_memory_env(env_name, num_envs=num_envs, seed=seed)
    try:
        env.reset()
        base_env = env.unwrapped
        runtime = get_runtime(base_env)
        live_poses = runtime.get_part_poses_w(base_env).clone()
        live_poses[..., :3] -= base_env.scene.env_origins.unsqueeze(1)
        return runtime.layout_poses_e.clone().cpu(), live_poses.cpu()
    finally:
        env.close()


def check_deterministic_reset(env_name: str, num_envs: int = 4) -> None:
    """Assert that the same seed reproduces the layout exactly and that another seed changes it."""
    layout_a, live_a = _reset_and_read_layout(env_name, seed=7, num_envs=num_envs)
    layout_b, live_b = _reset_and_read_layout(env_name, seed=7, num_envs=num_envs)
    layout_c, _ = _reset_and_read_layout(env_name, seed=8, num_envs=num_envs)
    assert torch.equal(layout_a, layout_b), "The same seed produced different layouts."
    assert torch.allclose(live_a, live_b, atol=1e-4), "The same seed produced different simulated part poses."
    assert torch.allclose(live_a[..., :3], layout_a[..., :3], atol=2e-3), "Simulated parts do not match the layout."
    assert not torch.equal(layout_a, layout_c), "Different seeds produced identical layouts."
    # Environments of one build must not share a layout.
    assert not torch.equal(layout_a[0], layout_a[1]), "Two environments received the same layout."


def check_observation_variants(env_name: str, enable_cameras: bool = False, num_envs: int = 2) -> None:
    """Build every observation variant and check the groups and shapes it adds."""
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env
    from isaaclab_arena_environments.spatial_memory.observations import ONLINE_MAP_MAX_POINTS, default_history_length

    for variant in ("current", "history_k", "oracle_map", "mapped"):
        env = make_spatial_memory_env(
            env_name, num_envs=num_envs, seed=3, observation_variant=variant, enable_cameras=enable_cameras
        )
        try:
            obs, _ = env.reset()
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for _ in range(2):
                obs, *_ = env.step(action)
            assert "policy" in obs, f"{variant}: missing proprioception group."
            if enable_cameras:
                assert "camera_obs" in obs, f"{variant}: missing camera group."
            if variant == "history_k":
                k = default_history_length(env.unwrapped.step_dt)
                assert obs["policy"]["joint_pos"].shape[:2] == (num_envs, k), obs["policy"]["joint_pos"].shape
                if enable_cameras:
                    image = next(iter(obs["camera_obs"].values()))
                    assert image.shape[:2] == (num_envs, k), image.shape
            else:
                assert obs["policy"]["joint_pos"].dim() == 2
            if variant == "oracle_map":
                points = obs["oracle_map"]["points"]
                boxes = obs["oracle_map"]["boxes"]
                assert points.dim() == 3 and points.shape[0] == num_envs and points.shape[-1] == 4
                assert boxes.shape[0] == num_envs and boxes.shape[-1] == 10
                assert torch.isfinite(points).all() and torch.isfinite(boxes).all()
            else:
                assert "oracle_map" not in obs
            if variant == "mapped":
                assert obs["mapped"]["online_map"].shape == (num_envs, ONLINE_MAP_MAX_POINTS, 4)
            else:
                assert "mapped" not in obs
        finally:
            env.close()


def _segmentation_pixel_counts(camera, part_names: list[str], num_envs: int) -> torch.Tensor:
    """Return (N, P) counts of pixels whose instance belongs to each part in each environment."""
    segmentation = camera.data.output["instance_id_segmentation_fast"]
    if hasattr(segmentation, "torch"):
        segmentation = segmentation.torch
    id_to_label = camera.data.info["instance_id_segmentation_fast"]["idToLabels"]
    counts = torch.zeros(num_envs, len(part_names), dtype=torch.long)
    segmentation = segmentation.reshape(num_envs, -1).cpu()
    for env_id in range(num_envs):
        ids, pixel_counts = torch.unique(segmentation[env_id], return_counts=True)
        for instance_id, pixel_count in zip(ids.tolist(), pixel_counts.tolist()):
            label = id_to_label.get(instance_id, id_to_label.get(str(instance_id), ""))
            for part_index, part_name in enumerate(part_names):
                if f"/env_{env_id}/{part_name}/" in f"{label}/":
                    counts[env_id, part_index] += pixel_count
    return counts


def check_visibility_matches_rendering(env_name: str, num_envs: int = 4, max_steps: int = 300) -> dict[str, float]:
    """Compare the analytic visibility of whole parts with the wrist camera's rendered instance segmentation.

    Every step of an oracle rollout, each task part is either analytically hidden (no sample point on any face in
    the frustum and unoccluded) or visible. The check asserts that analytically hidden parts almost never show up in
    the rendered image (``false_hidden``), which is the property the suite's hidden-geometry tests rely on.

    Returns:
        Rates of agreement, false-hidden and false-visible samples.
    """
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env
    from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
    from isaaclab_arena_environments.spatial_memory.policies import SpatialMemoryOraclePolicy
    from isaaclab_arena_environments.spatial_memory.runtime import get_runtime
    from isaaclab_arena_environments.spatial_memory.visibility import VisibilityModel, VisibilityTarget

    env = make_spatial_memory_env(env_name, num_envs=num_envs, seed=5, enable_cameras=True, with_segmentation=True)
    base_env = env.unwrapped
    policy = SpatialMemoryOraclePolicy()
    try:
        obs, _ = env.reset()
        policy.reset()
        runtime = get_runtime(base_env)
        logic = runtime.logic
        part_names = sorted({target.part_name for target in logic.visibility_targets()})
        whole_part_model = VisibilityModel(
            camera=logic.embodiment.camera_model(),
            targets=[VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in part_names],
            part_sizes={spec.name: spec.size_xyz for spec in logic.part_specs()},
            part_names=runtime.part_names,
            robot_occluders=logic.embodiment.robot_occluders(),
            device=base_env.device,
        )
        camera = next(sensor for name, sensor in base_env.scene.sensors.items() if "cam" in name)
        samples = agree = false_hidden = false_visible = 0
        with torch.inference_mode():
            for _ in range(max_steps):
                obs, *_ = env.step(policy.get_action(env, obs))
                if base_env.reset_env_ids.numel() > 0:
                    policy.reset(env_ids=base_env.reset_env_ids)
                mount = runtime.get_robot_body_poses_w(base_env, [whole_part_model.camera.mount_body])[:, 0]
                occluders = runtime.get_robot_body_poses_w(
                    base_env, [occluder.body_name for occluder in whole_part_model.robot_occluders]
                )
                analytic = whole_part_model.compute(mount, runtime.get_part_poses_w(base_env), occluders).cpu()
                rendered = _segmentation_pixel_counts(camera, part_names, num_envs)
                # Ignore boundary cases: a handful of points or pixels at the image or occluder edge.
                analytic_visible = analytic >= 3
                analytic_hidden = analytic == 0
                rendered_visible = rendered >= 4
                rendered_hidden = rendered == 0
                decided = (analytic_visible | analytic_hidden) & (rendered_visible | rendered_hidden)
                samples += int(decided.sum())
                agree += int((decided & (analytic_visible == rendered_visible)).sum())
                false_hidden += int((analytic_hidden & rendered_visible).sum())
                false_visible += int((analytic_visible & rendered_hidden).sum())
    finally:
        env.close()
    rates = {
        "samples": samples,
        "agreement": agree / max(samples, 1),
        "false_hidden": false_hidden / max(samples, 1),
        "false_visible": false_visible / max(samples, 1),
    }
    print(f"[spatial_memory] {env_name} visibility vs rendering: {rates}")
    assert samples > 100, "Too few decided samples to compare the visibility model with rendering."
    assert rates["false_hidden"] <= 0.02, f"Analytically hidden parts were rendered: {rates}"
    assert rates["agreement"] >= 0.9, f"The visibility model disagrees with rendering: {rates}"
    return rates
