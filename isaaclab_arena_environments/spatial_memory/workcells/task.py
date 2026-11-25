# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination, episode-relative fixture motion and optional head-depth observations."""

import torch
from functools import partial

from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena_environments.spatial_memory.observations import get_depth_map_provider, reset_online_map
from isaaclab_arena_environments.spatial_memory.workcells.layouts import FORBIDDEN_CONTACT, LAYOUTS, REQUIRED_SUPPORT
from isaaclab_arena_environments.spatial_memory.workcells.outcomes import (
    CROSSINGS,
    at_seam,
    crossing_opening,
    pose_e,
    reached_goal,
    scoop_tipped,
)


def measured_map(env):
    return get_depth_map_provider(env, "robot_head_cam", env.scene["robot_head_cam"].cfg.spawn.clipping_range[1]).query(
        env
    )


@configclass
class MapGroup(ObservationGroupCfg):
    online_map = ObservationTermCfg(func=measured_map)

    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False


@configclass
class MapObservations:
    mapped = MapGroup()


def move_fixture(env, env_ids, times, poses, repeat):
    """Sample the authored animation using the individual episode clocks."""
    frame_time = (env.episode_length_buf[env_ids] + 1) * env.step_dt
    times = frame_time.new_tensor(times)
    poses = frame_time.new_tensor(poses)
    t = frame_time.remainder(times[-1]) if repeat else frame_time.clamp(max=times[-1])
    high = torch.searchsorted(times, t).clamp(1, len(times) - 1)
    fraction = (t - times[high - 1]) / (times[high] - times[high - 1])
    sample = torch.lerp(poses[high - 1], poses[high], fraction[:, None])
    mover = env.scene["mover"]
    target = sample.new_zeros((len(env_ids), 7))
    target[:, :3] = sample[:, :3] + env.scene.env_origins[env_ids]
    half_yaw = sample[:, 3].deg2rad() * 0.5
    target[:, 5] = half_yaw.sin()
    target[:, 6] = half_yaw.cos()
    # GPU PhysX supports pose writes for kinematic bodies. Its target API is CPU-only.
    displacement_xy = target[:, :2] - mover.data.root_pose_w.torch[env_ids, :2]
    mover.write_root_pose_to_sim_index(root_pose=target, env_ids=env_ids)
    if "deck_contact" in env.scene.sensors:
        # Apply the driven deck's tangential motion only while it supports the loose part.
        load = env.scene["part"]
        contact = env.scene["deck_contact"].data.force_matrix_w.torch[env_ids, 0, 0]
        supported = contact.norm(dim=-1) > 0.15
        carried_ids = env_ids[supported]
        if len(carried_ids):
            carried_pose = load.data.root_pose_w.torch[carried_ids].clone()
            carried_pose[:, :2] += displacement_xy[supported]
            load.write_root_pose_to_sim_index(root_pose=carried_pose, env_ids=carried_ids)


def reset_adhesion(env, env_ids):
    """Reattach the two peel strips in just the resetting environments."""
    if not hasattr(env, "cover_attached"):
        env.cover_attached = torch.ones((env.num_envs, 2), device=env.device, dtype=torch.bool)
    env.cover_attached[env_ids] = True
    env.scene["backing"].permanent_wrench_composer.reset(env_ids=env_ids)


def resist_peel(env, env_ids):
    """Hold each backing strip at its original surface until tension pulls it free."""
    from isaaclab.utils.math import quat_apply

    backing = env.scene["backing"]
    composer = backing.permanent_wrench_composer
    composer.reset(env_ids=env_ids)
    pose = backing.data.root_pose_w.torch[env_ids]
    initial = backing.data.default_root_pose.torch[env_ids, :3] + env.scene.env_origins[env_ids]
    for i, offset_y in enumerate((-0.062, 0.062)):
        offset = pose.new_tensor((0.0, offset_y, 0.0)).expand(len(env_ids), 3)
        point = pose[:, :3] + quat_apply(pose[:, 3:], offset)
        extension = initial + offset - point
        attached = env.cover_attached[env_ids, i] & (extension.norm(dim=1) < 0.026)
        env.cover_attached[env_ids, i] = attached
        force = (320.0 * extension) * attached[:, None]
        composer.add_forces_and_torques_index(
            forces=force[:, None], positions=point[:, None], env_ids=env_ids, is_global=True
        )


def robot_fell(env):
    return pose_e(env, "robot")[:, 2] < 0.38


def guard_contact(env):
    from isaaclab_arena_environments.spatial_memory.collision import CONTACT_FORCE_THRESHOLD_N

    return env.scene["guard_sensor"].data.net_forces_w.torch.norm(dim=-1).amax(dim=1) > CONTACT_FORCE_THRESHOLD_N


class MotionTask(NoTask):
    """Run one short guide motion with a measured workpiece outcome."""

    def __init__(self, scene, layout, mapped, animation, seam_points):
        super().__init__(episode_length_s=30, task_description=LAYOUTS[scene].instruction)
        self.scene = scene
        self.layout = layout
        self.mapped = mapped
        self.animation = animation
        self.seam_points = seam_points

    def get_scene_cfg(self):
        fields = []
        if self.scene in FORBIDDEN_CONTACT:
            sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/workcell/Guard")
            fields.append(("guard_sensor", ContactSensorCfg, sensor))
        if self.scene == "moving_carriage":
            sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/workcell/Body_part",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/workcell/Body_mover"],
            )
            fields.append(("deck_contact", ContactSensorCfg, sensor))
        for surface, body in REQUIRED_SUPPORT.get(self.scene, ()):
            sensor = ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/workcell/Support_{surface}",
                filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/workcell/Body_{body}"],
            )
            fields.append(("support_" + surface, ContactSensorCfg, sensor))
        return make_configclass("WorkcellContactSensors", fields)() if fields else None

    def get_events_cfg(self):
        fields = []
        if self.scene == "display_cover":
            fields.append(("attach_cover", EventTermCfg, EventTermCfg(func=reset_adhesion, mode="reset")))
            fields.append((
                "peel_resistance",
                EventTermCfg,
                EventTermCfg(func=resist_peel, mode="interval", interval_range_s=(0.02, 0.02)),
            ))
        if self.mapped:
            fields.append(("clear_map", EventTermCfg, EventTermCfg(func=reset_online_map, mode="reset")))
        if self.animation:
            event = EventTermCfg(
                func=move_fixture, mode="interval", interval_range_s=(0.02, 0.02), params=self.animation
            )
            fields.append(("move_fixture", EventTermCfg, event))
        return make_configclass("MotionEvents", fields)() if fields else None

    def get_observation_cfg(self):
        return MapObservations() if self.mapped else None

    def get_termination_cfg(self):
        if self.seam_points:
            sequence = [partial(at_seam, target=point) for point in self.seam_points]
        else:
            sequence = [
                TrueForConsecutiveStepsCfg(
                    predicate=partial(reached_goal, scene=self.scene, layout=self.layout), required_steps=5
                )
            ]
        if self.scene == "scoop_pour":
            sequence.insert(0, scoop_tipped)
        if self.scene in CROSSINGS:
            sequence.insert(0, partial(crossing_opening, scene=self.scene))
        failures = {"fell": TerminationTermCfg(func=robot_fell)}
        if self.scene in FORBIDDEN_CONTACT:
            failures["guard_contact"] = TerminationTermCfg(func=guard_contact)
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[CompletionCriteria(name=self.scene, predicate_sequence=sequence)],
            failures=failures,
        )
