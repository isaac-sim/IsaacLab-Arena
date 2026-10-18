# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Five ordered insertions with current attachment and settled-table conditions."""

import math
import torch
from functools import partial

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.object_settling import objects_below_velocity_thresholds
from isaaclab_arena.tasks.predicates.spatial import contact_force_is_upward_support
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena_environments.burvik_assembly.connections import attachment_enabled, attachment_errors
from isaaclab_arena_environments.burvik_assembly.geometry import BENCH_HEIGHT, PART_NAMES


class BurvikMotionMeasurement:
    """Measure every part's translation and rotation between consecutive control steps."""

    def __init__(self, env):
        self.poses = torch.zeros((env.num_envs, len(PART_NAMES), 7), device=env.device)
        self.steps = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.settled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.steps[ids] = -1
        self.settled[ids] = False

    def measure(self, env):
        """Return motion below 20 mm/s and 0.05 rad/s; repeated calls in one step are cached."""
        step = env.episode_length_buf
        changed = step != self.steps
        if not bool(changed.any()):
            return self.settled
        current = torch.stack([env.arena_world.get_pose_w(name) for name in PART_NAMES], dim=1)
        consecutive = (self.steps >= 0) & (step == self.steps + 1)
        linear = torch.linalg.vector_norm(current[..., :3] - self.poses[..., :3], dim=-1) / env.step_dt
        # atan2 avoids acos's loss of precision for small rotations in float32.
        delta = quat_mul(current[..., 3:], quat_inv(self.poses[..., 3:]))
        angular = 2 * torch.atan2(torch.linalg.vector_norm(delta[..., :3], dim=-1), delta[..., 3].abs())
        measured = consecutive & ((linear < 0.02) & (angular / env.step_dt < 0.05)).all(dim=1)
        self.settled[changed] = measured[changed]
        self.poses[changed] = current[changed]
        self.steps[changed] = step[changed]
        return self.settled


def parts_settled(env):
    """Check observed part motion and reported linear speed despite TGS angular-velocity noise."""
    if not hasattr(env, "burvik_motion_measurement"):
        env.burvik_motion_measurement = BurvikMotionMeasurement(env)
    observed = env.burvik_motion_measurement.measure(env)
    return observed & objects_below_velocity_thresholds(env, list(PART_NAMES), 0.02, float("inf"))


def installed_part(env, part_name, position, quaternion, require_standing=False):
    """Require an enabled connection and measured seating; optionally a supported upright table."""
    distance, angle = attachment_errors(env, part_name, position, quaternion)
    installed = attachment_enabled(env, part_name) & (distance < 0.0015) & (angle < math.radians(1.0))
    if require_standing:
        T_W_T = env.arena_world.get_pose_w("tray")
        z = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(env.num_envs, -1)
        upright = quat_apply(T_W_T[:, 3:], z)[:, 2] > math.cos(math.radians(2.0))
        height = env.arena_world.get_pose_e("tray")[:, 2]
        supported = (height - (BENCH_HEIGHT + 0.42)).abs() < 0.003
        still = parts_settled(env)
        bench_force = sum(
            env.scene[f"leg_{i}_bench_contact"].data.normal_force_matrix_w.torch[:, 0, 0] for i in range(1, 5)
        )
        support = contact_force_is_upward_support(bench_force, force_threshold=20.0, support_cone_half_angle_rad=0.3)
        installed &= upright & supported & still & support
    return installed


class InstallBurvikPartTask(TaskBase):
    """A seated attachment milestone, retaining existing Arena temporal/progress machinery."""

    def __init__(self, part_name, transform, final=False):
        super().__init__(episode_length_s=45.0, task_description=f"Push {part_name} into the tray until seated.")
        self.part_name = part_name
        self.transform = transform
        self.final = final

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_termination_cfg(self):
        predicate = partial(installed_part, part_name=self.part_name, require_standing=self.final, **self.transform)
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name=f"install_{self.part_name}",
                    predicate_sequence=[
                        TrueForConsecutiveStepsCfg(predicate, required_steps=120 if self.final else 24)
                    ],
                )
            ],
            failures={
                "part_dropped": TerminationTermCfg(
                    func=mdp.root_height_below_minimum,
                    params={"minimum_height": BENCH_HEIGHT - 0.12, "asset_cfg": SceneEntityCfg(self.part_name)},
                )
            },
        )

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("The displacement demo does not define robot Mimic grasp segments.")
