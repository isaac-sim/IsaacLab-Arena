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

SEED = 1234
"""Environment seed of the oracle and blind checks."""

ORACLE_MIN_SUCCESS_RATE = 0.9
"""The oracle must solve at least this fraction of seeds."""

BLIND_MAX_SUCCESS_RATE = 0.3
"""The blind controller must solve at most this fraction of seeds."""

REMEMBERED_TARGET_MIN_SEEN_FRACTION = 0.9
"""Each remembered target must be seen before the hidden phase in at least this fraction of episodes."""

STATIC_TOLERANCE_M = 0.005
"""Largest displacement of any static part while the task-relevant geometry is hidden."""

RENDERING_MIN_SAMPLES = 100
"""Fewest decided (part, step) samples for a meaningful comparison of the visibility model with rendering."""

RENDERING_MAX_FALSE_HIDDEN_RATE = 0.02
"""Largest fraction of samples in which an analytically hidden part is rendered."""

RENDERING_MIN_AGREEMENT = 0.9
"""Smallest fraction of samples in which the visibility model and rendering agree."""

ANALYTIC_VISIBLE_MIN_POINTS = 3
"""Visible sample points above which a part counts as visible to the model (fewer is a boundary case)."""

RENDERED_VISIBLE_MIN_PIXELS = 4
"""Rendered pixels above which a part counts as visible in the image (fewer is a boundary case)."""


def _roll_out(env_name: str, mode: str, num_seeds: int, seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the episode records of a scripted rollout and the task's remembered targets."""
    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env, rollout_episodes
    from isaaclab_arena_environments.spatial_memory.policies import SpatialMemoryBlindPolicy, SpatialMemoryOraclePolicy

    env = make_spatial_memory_env(env_name, num_envs=num_seeds, seed=seed)
    policy = SpatialMemoryOraclePolicy() if mode == "oracle" else SpatialMemoryBlindPolicy()
    try:
        records = rollout_episodes(env, policy, num_seeds)
        remembered_targets = get_episode_state(env).logic.remembered_target_names()
    finally:
        env.close()
    assert len(records) == num_seeds, f"Expected {num_seeds} finished episodes, got {len(records)}."
    return records, remembered_targets


def _describe_failures(records: list[dict[str, Any]], limit: int = 5) -> str:
    """Return one line per failed episode (at most ``limit``): length, first collision and layout."""
    lines = []
    for record in records:
        if record["success"]:
            continue
        episode = record["spatial_memory"]
        lines.append(
            f"env {record['env_id']}: len={record['episode_length']} collision={episode['first_collision']} "
            f"searched_again={episode['searched_again']} layout={episode['layout']}"
        )
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def check_oracle_solves_with_hidden_geometry(env_name: str, num_seeds: int = NUM_SEEDS, seed: int = SEED) -> None:
    """Assert the oracle solves enough seeds, the hidden targets stay hidden and static, and were seen earlier.

    The hidden-geometry checks describe the task, so they are made on the episodes the oracle solved. The task's
    remembered targets must have been visible before the hidden phase, so that what is hidden was observable.
    """
    from isaaclab_arena_environments.spatial_memory.evaluation import summarize_episode_records

    records, remembered_targets = _roll_out(env_name, "oracle", num_seeds, seed)
    summary = summarize_episode_records(records)
    print(f"[spatial_memory] {env_name} oracle: {summary}")
    assert summary["success_rate"] >= ORACLE_MIN_SUCCESS_RATE, (
        f"Oracle success rate {summary['success_rate']:.2f} < {ORACLE_MIN_SUCCESS_RATE} on {env_name}.\n"
        + _describe_failures(records)
    )
    for record in records:
        if not record["success"]:
            continue
        episode = record["spatial_memory"]
        assert episode["hidden_phase_s"] > 0.0, f"env {record['env_id']}: succeeded without a hidden phase."
        seen = episode["hidden_targets_seen_in_hidden_phase"]
        assert not seen, f"env {record['env_id']}: hidden targets {seen} were visible during the hidden phase."
        displacement_m = episode["static_max_displacement_m"]
        assert (
            displacement_m < STATIC_TOLERANCE_M
        ), f"env {record['env_id']}: static geometry moved {displacement_m:.4f} m while hidden."
    for target in remembered_targets:
        seen = [target in record["spatial_memory"]["targets_seen_before_hidden_phase"] for record in records]
        fraction = sum(seen) / len(seen)
        print(f"[spatial_memory] {env_name}: '{target}' seen before the hidden phase in {fraction:.2f} of episodes")
        assert (
            fraction >= REMEMBERED_TARGET_MIN_SEEN_FRACTION
        ), f"'{target}' was seen before the hidden phase in only {fraction:.2f} of episodes."


def check_blind_controller_fails(env_name: str, num_seeds: int = NUM_SEEDS, seed: int = SEED) -> None:
    """Assert the blind controller solves at most ``BLIND_MAX_SUCCESS_RATE`` of seeds."""
    from isaaclab_arena_environments.spatial_memory.evaluation import summarize_episode_records

    records, _ = _roll_out(env_name, "blind", num_seeds, seed)
    summary = summarize_episode_records(records)
    print(f"[spatial_memory] {env_name} blind: {summary}")
    assert (
        summary["success_rate"] <= BLIND_MAX_SUCCESS_RATE
    ), f"Blind controller succeeds on {summary['success_rate']:.2f} of seeds on {env_name}; the task is too easy."


def _reset_and_read_layout(env_name: str, seed: int, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the sampled layout (N, K, 7) and the simulated part poses (N, K, 7) after one reset."""
    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env

    env = make_spatial_memory_env(env_name, num_envs=num_envs, seed=seed)
    try:
        env.reset()
        base_env = env.unwrapped
        episode = get_episode_state(base_env)
        T_E_parts = episode.get_part_poses_w(base_env).clone()
        T_E_parts[..., :3] -= base_env.scene.env_origins.unsqueeze(1)
        return episode.layout_poses_e.clone().cpu(), T_E_parts.cpu()
    finally:
        env.close()


def check_deterministic_reset(env_name: str, num_envs: int = 4) -> None:
    """Assert that the same seed reproduces the layout exactly and that another seed changes it."""
    layout_a, T_E_parts_a = _reset_and_read_layout(env_name, seed=7, num_envs=num_envs)
    layout_b, T_E_parts_b = _reset_and_read_layout(env_name, seed=7, num_envs=num_envs)
    layout_c, _ = _reset_and_read_layout(env_name, seed=8, num_envs=num_envs)
    assert torch.equal(layout_a, layout_b), "The same seed produced different layouts."
    assert torch.allclose(T_E_parts_a, T_E_parts_b, atol=1e-4), "The same seed produced different part poses."
    assert torch.allclose(T_E_parts_a[..., :3], layout_a[..., :3], atol=2e-3), "Parts do not match the layout."
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
            # The embodiment's first proprioceptive term (its name depends on the embodiment).
            proprioception = next(iter(obs["policy"].values()))
            if variant == "history_k":
                history_length = default_history_length(env.unwrapped.step_dt)
                assert proprioception.shape[:2] == (num_envs, history_length), proprioception.shape
                if enable_cameras:
                    image = next(iter(obs["camera_obs"].values()))
                    assert image.shape[:2] == (num_envs, history_length), image.shape
            else:
                assert proprioception.dim() == 2
            if variant == "oracle_map":
                points = obs["oracle_map"]["points"]
                boxes = obs["oracle_map"]["boxes"]
                assert points.dim() == 3 and points.shape[0] == num_envs and points.shape[-1] == 4
                assert boxes.shape[0] == num_envs and boxes.shape[-1] == 10
                assert torch.isfinite(points).all() and torch.isfinite(boxes).all()
            else:
                assert "oracle_map" not in obs
            if variant == "mapped":
                cloud = obs["mapped"]["online_map"]
                assert cloud.shape == (num_envs, ONLINE_MAP_MAX_POINTS, 4)
                assert torch.isfinite(cloud).all()
                assert ((cloud[..., 3] == 0) | (cloud[..., 3] == 1)).all()
                if enable_cameras:
                    assert cloud[..., 3].any(dim=1).all(), "Every camera must contribute measured points."
                else:
                    assert not cloud.any(), "Camera-free runs must have no measured points."
            else:
                assert "mapped" not in obs
        finally:
            env.close()


def _segmentation_pixel_counts(camera, part_names: list[str], num_envs: int) -> torch.Tensor:
    """Return (N, P) counts of pixels whose instance belongs to each part in each environment."""
    segmentation = camera.data.output["instance_id_segmentation_fast"]
    # Camera outputs are proxy arrays with a ``torch`` view on some backends and plain tensors on others.
    segmentation = segmentation.torch if hasattr(segmentation, "torch") else segmentation
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


def check_visibility_matches_rendering(env_name: str, num_envs: int = 4, max_steps: int = 300) -> None:
    """Assert the analytic visibility of whole parts agrees with the egocentric camera's instance segmentation.

    Every step of an oracle rollout, each task part is either analytically hidden (no sample point on any face in the
    frustum and unoccluded) or visible. Analytically hidden parts must almost never show up in the rendered image,
    which is the property the suite's hidden-geometry checks rely on.
    """
    from isaaclab_arena_environments.spatial_memory.episode_state import get_episode_state
    from isaaclab_arena_environments.spatial_memory.evaluation import make_spatial_memory_env
    from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
    from isaaclab_arena_environments.spatial_memory.policies import SpatialMemoryOraclePolicy
    from isaaclab_arena_environments.spatial_memory.visibility import VisibilityModel, VisibilityTarget

    env = make_spatial_memory_env(env_name, num_envs=num_envs, seed=5, enable_cameras=True, with_segmentation=True)
    base_env = env.unwrapped
    policy = SpatialMemoryOraclePolicy()
    try:
        obs, _ = env.reset()
        policy.reset()
        episode = get_episode_state(base_env)
        logic = episode.logic
        part_names = sorted({target.part_name for target in logic.visibility_targets()})
        whole_part_model = VisibilityModel(
            camera=logic.embodiment.camera_model(),
            targets=[VisibilityTarget(name, name, ALL_FACES, spacing_m=0.02) for name in part_names],
            part_sizes={spec.name: spec.size_xyz for spec in logic.part_specs()},
            part_names=episode.part_names,
            robot_occluders=logic.embodiment.robot_occluders(),
            device=base_env.device,
        )
        occluder_bodies = [occluder.body_name for occluder in whole_part_model.robot_occluders]
        camera = next(sensor for name, sensor in base_env.scene.sensors.items() if "cam" in name)
        num_samples = num_agreeing = num_false_hidden = 0
        with torch.inference_mode():
            for _ in range(max_steps):
                obs, *_ = env.step(policy.get_action(env, obs))
                if base_env.reset_env_ids.numel() > 0:
                    policy.reset(env_ids=base_env.reset_env_ids)
                T_W_mount = episode.get_robot_body_poses_w(base_env, [whole_part_model.camera.mount_body])[:, 0]
                T_W_occluders = episode.get_robot_body_poses_w(base_env, occluder_bodies)
                T_W_parts = episode.get_part_poses_w(base_env)
                analytic = whole_part_model.count_visible_points(T_W_mount, T_W_parts, T_W_occluders).cpu()
                rendered = _segmentation_pixel_counts(camera, part_names, num_envs)
                # Boundary cases (a handful of points or pixels at the image or occluder edge) are left undecided.
                analytic_visible = analytic >= ANALYTIC_VISIBLE_MIN_POINTS
                analytic_hidden = analytic == 0
                rendered_visible = rendered >= RENDERED_VISIBLE_MIN_PIXELS
                rendered_hidden = rendered == 0
                decided = (analytic_visible | analytic_hidden) & (rendered_visible | rendered_hidden)
                num_samples += int(decided.sum())
                num_agreeing += int((decided & (analytic_visible == rendered_visible)).sum())
                num_false_hidden += int((analytic_hidden & rendered_visible).sum())
    finally:
        env.close()
    agreement = num_agreeing / max(num_samples, 1)
    false_hidden_rate = num_false_hidden / max(num_samples, 1)
    print(
        f"[spatial_memory] {env_name} visibility vs rendering: agreement {agreement:.3f}, false hidden "
        f"{false_hidden_rate:.3f} over {num_samples} samples"
    )
    assert num_samples > RENDERING_MIN_SAMPLES, "Too few decided samples to compare the model with rendering."
    assert false_hidden_rate <= RENDERING_MAX_FALSE_HIDDEN_RATE, "Analytically hidden parts were rendered."
    assert agreement >= RENDERING_MIN_AGREEMENT, "The visibility model disagrees with rendering."
