# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 workcells with a surveyed entrance and ordered physical operations."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass

from isaaclab_arena_environments.spatial_memory.collision import CONTACT_FORCE_THRESHOLD_N, filtered_sensor_name
from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board, box_item
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget


@dataclass(frozen=True)
class Fixture:
    """Box center measured from the workcell entrance, with z measured above the floor."""

    name: str
    xyz: tuple[float, float, float]
    size: tuple[float, float, float]
    color: str = "gray"
    yaw: float = 0.0
    movable: bool = False


@dataclass(frozen=True)
class Motion:
    """Sinusoidal or rotary motion relative to a fixture's own sampled reset pose."""

    fixture: str
    axis: str
    amplitude: float
    period_s: float
    pivot_radius: float = 0.0


@dataclass(frozen=True)
class Milestone:
    """A physical condition held consecutively in a named fixture's local frame."""

    label: str
    body: str
    fixture: str
    offset: tuple[float, float, float]
    radius: float = 0.17
    yaw: float | None = None
    symmetric_heading: bool = False
    hold_s: float = 0.06
    released: bool = False
    support: str | None = None
    contact_parts: tuple[str, ...] = ()
    max_pelvis_height: float | None = None
    retreat_m: float = 0.0
    motion_window: tuple[str, str, float, float] | None = None


def heading(q):
    """Return world yaw from an xyzw quaternion."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


class SurveyedWorkcell(SpatialMemoryTaskLogic):
    """Survey fixed geometry, back through its opening, then perform the workcell's operation."""

    robot = "g1"
    minimum_env_spacing_m = 15.0
    episode_length_s = 210.0
    forbids_searching_again = True
    reobservation_budget_s = 0.0
    fixtures: tuple[Fixture, ...]
    stages: tuple[Milestone, ...]
    survey_targets: tuple[str, ...]
    approach: tuple[str, float, float]
    """Static reference fixture and x/y offsets to the first approach, measured in its local frame."""
    example: str
    approach_by_bearing = False
    """Latch the direction of the first half-metre of travel toward a remembered station."""

    preset_hand: tuple[str, str] | None = None
    """Palm and surveyed fixture whose heights must match at approach commitment."""

    motions: tuple[Motion, ...] = ()
    touchable: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()

    def build_part_specs(self):
        parts = []
        for f in self.fixtures:
            parts.append(box_item(f.name, f.size, f.color, 0.35) if f.movable else board(f.name, f.size, f.color))
        return parts

    def sample_layout(self, generator):
        lane = int(torch.randint(4, (), generator=generator))
        entry_y = (lane - 1.5) * 0.80 + uniform(generator, -0.055, 0.055)
        mirror = 1 if int(torch.randint(2, (), generator=generator)) else -1
        depth = uniform(generator, -0.18, 0.18)
        angle = uniform(generator, -0.18, 0.18)
        poses = {}
        for f in self.fixtures:
            x, y, z = f.xyz
            y *= mirror
            x_E = math.cos(angle) * x - math.sin(angle) * y + depth
            y_E = math.sin(angle) * x + math.cos(angle) * y + entry_y
            poses[f.name] = pose(x_E, y_E, G1_FLOOR_Z_M + z, angle + mirror * f.yaw)
        return LayoutSample(
            self.part_poses_tensor(poses),
            {
                "entry_y": torch.tensor(entry_y),
                "mirror": torch.tensor(mirror),
                "depth_offset": torch.tensor(depth),
                "cell_yaw": torch.tensor(angle),
                "start_xy": 0.18 * torch.rand(2, generator=generator) - 0.09,
                "motion_direction": 2 * torch.randint(2, (len(self.motions),), generator=generator) - 1,
                "motion_speed": 0.85 + 0.30 * torch.rand(len(self.motions), generator=generator),
            },
        )

    def on_reset(self, env, episode, env_ids):
        robot = env.scene["robot"]
        T_W_R = robot.data.root_pose_w.torch[env_ids].clone()
        T_W_R[:, :2] = env.scene.env_origins[env_ids, :2] + episode.layout_params["start_xy"][env_ids]
        T_W_R[:, 3:] = T_W_R.new_tensor((0, 0, 1, 0))
        robot.write_root_pose_to_sim(T_W_R, env_ids=env_ids)

    def hidden_target_names(self):
        return list(self.survey_targets)

    def memory_target_names(self):
        return self.hidden_target_names()

    def visibility_targets(self):
        return [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.10) for name in self.hidden_target_names()]

    def manipulated_item_names(self):
        return self.item_names()

    def item_support_names(self):
        return list(self.supports)

    def hand_touchable_part_names(self):
        return [*self.touchable, *self.supports]

    def instruction_dim(self):
        return 1

    def instruction(self, env, episode):
        return episode.instruction_revealed[:, None].float()

    def commitment_made(self, env, episode, active):
        return episode.task_state_buffer("entered", False, torch.bool)

    def hidden_phase(self, env, episode):
        return episode.instruction_revealed & ~self.commitment_made(env, episode, None)

    def static_part_mask(self, env, episode):
        mask = super().static_part_mask(env, episode).clone()
        for motion in self.motions:
            mask[:, episode.part_index[motion.fixture]] = False
        return mask

    def _move_fixtures(self, env, episode, active):
        if not self.motions:
            return
        ids = active.nonzero().flatten()
        if not len(ids):
            return
        t = env.episode_length_buf[ids] * env.step_dt
        mirror = episode.layout_params["mirror"][ids]
        for i, motion in enumerate(self.motions):
            index = episode.part_index[motion.fixture]
            T_E_F = episode.layout_poses_e[ids, index].clone()
            phase = 2 * math.pi * t * episode.layout_params["motion_speed"][ids, i] / motion.period_s
            delta = motion.amplitude * phase.sin() * episode.layout_params["motion_direction"][ids, i]
            if motion.axis == "yaw":
                initial = heading(T_E_F[:, 3:])
                angle = initial + mirror * delta
                T_E_F[:, 0] += motion.pivot_radius * (angle.cos() - initial.cos())
                T_E_F[:, 1] += motion.pivot_radius * (angle.sin() - initial.sin())
                T_E_F[:, 5] = (angle / 2).sin()
                T_E_F[:, 6] = (angle / 2).cos()
            else:
                axis = "xyz".index(motion.axis)
                if axis == 2:
                    T_E_F[:, 2] += delta
                else:
                    angle = episode.layout_params["cell_yaw"][ids] + axis * math.pi / 2
                    distance = delta * (mirror if axis == 1 else 1)
                    T_E_F[:, 0] += distance * angle.cos()
                    T_E_F[:, 1] += distance * angle.sin()
            T_E_F[:, :3] += env.scene.env_origins[ids]
            env.scene[motion.fixture].write_root_pose_to_sim(T_E_F, env_ids=ids)

    @staticmethod
    def _goal(T_E_F, mirror, offset):
        x, y, z = offset
        angle = heading(T_E_F[:, 3:])
        goal_E = T_E_F[:, :3].clone()
        goal_E[:, 0] += angle.cos() * x - angle.sin() * mirror * y
        goal_E[:, 1] += angle.sin() * x + angle.cos() * mirror * y
        goal_E[:, 2] += z
        return goal_E

    def reference_pose(self, env, name):
        """Return the measured reference frame used for an approach or operation."""
        return env.arena_world.get_pose_e(name)

    def _body_pose(self, env, episode, body, pelvis):
        if body == "pelvis":
            return pelvis
        if body in ("left", "right"):
            T_E_body = episode.get_robot_body_poses_w(env, [f"{body}_hand_palm_link"])[:, 0].clone()
            T_E_body[:, :3] -= env.scene.env_origins
            return T_E_body
        return env.arena_world.get_pose_e(body)

    @staticmethod
    def _touching_parts(env, body):
        """Return a per-fixture contact mask from the body's filtered force sensor."""
        sensor = env.scene[filtered_sensor_name(body)]
        return sensor.data.force_matrix_w.torch[:, 0].norm(dim=-1) > CONTACT_FORCE_THRESHOLD_N

    def _at_stage(self, env, episode, stage, pelvis):
        mirror = episode.layout_params["mirror"]
        fixture = self.reference_pose(env, stage.fixture)
        goal = self._goal(fixture, mirror, stage.offset)
        body = self._body_pose(env, episode, stage.body, pelvis)
        dimensions = 2 if stage.body == "pelvis" else 3
        reached = (body[:, :dimensions] - goal[:, :dimensions]).norm(dim=1) <= stage.radius
        if stage.yaw is not None:
            yaw_error = heading(body[:, 3:]) - heading(fixture[:, 3:]) - mirror * stage.yaw
            cosine = yaw_error.cos()
            reached &= (cosine.abs() if stage.symmetric_heading else cosine) > math.cos(0.28)
        if len(stage.contact_parts) == 2:
            a, b = stage.contact_parts
            left = self._touching_parts(env, "left_hand_palm_link")
            right = self._touching_parts(env, "right_hand_palm_link")
            i, j = episode.part_index[a], episode.part_index[b]
            reached = (left[:, i] & right[:, j]) | (left[:, j] & right[:, i])
        elif stage.contact_parts:
            contacts = self._touching_parts(env, f"{stage.body}_hand_palm_link")
            reached &= contacts[:, episode.part_index[stage.contact_parts[0]]]
        if stage.max_pelvis_height is not None:
            reached &= pelvis[:, 2] - G1_FLOOR_Z_M <= stage.max_pelvis_height
        if stage.released:
            velocity_W = env.arena_world.get_root_linear_velocity_w(stage.body)
            support = episode.part_index[stage.support or stage.fixture]
            touching = self._touching_parts(env, stage.body)[:, support]
            reached &= ~self.gripper_closed(env) & (velocity_W.norm(dim=1) < 0.08)
            reached &= ((body[:, 2] - goal[:, 2]).abs() <= 0.035) & touching
        if stage.motion_window is not None:
            name, axis, low, high = stage.motion_window
            value = self._fixture_displacement(env, episode, name, axis)
            reached &= (value >= low) & (value <= high)
        return reached

    def _fixture_displacement(self, env, episode, name, axis):
        current = env.arena_world.get_pose_e(name)
        home = episode.layout_poses_e[:, episode.part_index[name]]
        mirror = episode.layout_params["mirror"]
        if axis == "yaw":
            delta = heading(current[:, 3:]) - heading(home[:, 3:])
            return mirror * torch.atan2(delta.sin(), delta.cos())
        if axis == "z":
            return current[:, 2] - home[:, 2]
        angle = episode.layout_params["cell_yaw"] + (math.pi / 2 if axis == "y" else 0)
        direction = torch.stack((angle.cos(), angle.sin()), dim=1)
        displacement = ((current[:, :2] - home[:, :2]) * direction).sum(dim=1)
        return displacement * mirror if axis == "y" else displacement

    def on_step(self, env, episode, visible, active):
        pelvis = self.embodiment.tracked_pose_w(env)
        pelvis[:, :3] -= env.scene.env_origins
        surveyed = episode.visible_before_hidden_phase.all(dim=1)
        at_start = pelvis[:, :2].norm(dim=1) < 0.27
        self.update_memory_probe(env, episode, visible, active, at_start & surveyed)
        self._check_approach(env, episode, active, pelvis)
        self._advance_operation(env, episode, active, pelvis)
        self._move_fixtures(env, episode, active)

    def _check_approach(self, env, episode, active, pelvis):
        entered = self.commitment_made(env, episode, active)
        reference, dx, dy = self.approach
        fixture = self.reference_pose(env, reference)
        goal = self._goal(fixture, episode.layout_params["mirror"], (dx, dy, 0))
        pending = active & episode.instruction_revealed & ~entered
        if self.approach_by_bearing:
            # Measure from the common reveal pose, so the first departure indicates the chosen direction.
            anchor = episode.task_state_buffer("reveal_xy", 0.0, torch.float, (2,))
            start = episode.instruction_reveal_step == env.episode_length_buf
            anchor[start] = pelvis[start, :2]
            displacement = pelvis[:, :2] - anchor
            crossing = pending & (displacement.norm(dim=1) >= 0.55)
            toward_goal = torch.nn.functional.normalize(goal[:, :2] - anchor, dim=1)
            bearing = torch.nn.functional.normalize(displacement, dim=1)
            wrong = (bearing * toward_goal).sum(dim=1) < math.cos(0.13)
        else:
            delta_xy = pelvis[:, :2] - goal[:, :2]
            angle = heading(fixture[:, 3:])
            longitudinal = delta_xy[:, 0] * angle.cos() + delta_xy[:, 1] * angle.sin()
            lateral = -delta_xy[:, 0] * angle.sin() + delta_xy[:, 1] * angle.cos()
            crossing = pending & (longitudinal <= 0)
            facing_away = (heading(pelvis[:, 3:]) - angle).cos() > math.cos(0.55)
            wrong = (lateral.abs() > 0.30) | ~facing_away
        if self.preset_hand:
            hand, target = self.preset_hand
            palm = self._body_pose(env, episode, hand, pelvis)
            height = env.arena_world.get_pose_e(target)[:, 2]
            wrong |= (palm[:, 2] - height).abs() > 0.085
        episode.task_state_buffer("wrong_entrance", False, torch.bool).logical_or_(crossing & wrong)
        entered |= crossing

    def _advance_operation(self, env, episode, active, pelvis):
        entered = self.commitment_made(env, episode, active)
        step = episode.task_state_buffer("operation", 0, torch.long)
        held = episode.task_state_buffer("held_steps", 0, torch.long)
        previous_xy = episode.task_state_buffer("previous_xy", 0.0, torch.float, (2,))
        retreat = episode.task_state_buffer("backward_distance", 0.0, torch.float)
        yaw = heading(pelvis[:, 3:])
        displacement = pelvis[:, :2] - previous_xy
        previous_xy[active] = pelvis[active, :2]
        initial_step = step.clone()
        for i, stage in enumerate(self.stages):
            current = active & entered & (initial_step == i)
            if not current.any():
                continue
            reached = self._at_stage(env, episode, stage, pelvis)
            if stage.retreat_m:
                fixture = self.reference_pose(env, stage.fixture)
                required_yaw = heading(fixture[:, 3:]) + episode.layout_params["mirror"] * stage.yaw
                facing = (yaw - required_yaw).cos() > math.cos(0.55)
                axis = torch.stack((required_yaw.cos(), required_yaw.sin()), dim=1)
                progress = -(displacement * axis).sum(dim=1)
                retreat[current] += torch.where(facing, progress, progress.clamp(max=0))[current]
                wrong_way = episode.task_state_buffer("forward_exit_distance", 0.0, torch.float)
                wrong_way[current & ~facing] += progress.clamp(min=0)[current & ~facing]
                reached &= retreat >= stage.retreat_m
            held[current] = torch.where(reached[current], held[current] + 1, 0)
            complete = current & (held >= math.ceil(stage.hold_s / env.step_dt))
            step[complete] += 1
            held[complete] = 0
            retreat[complete] = 0

    def success_terms(self, env, episode):
        step = episode.task_state_buffer("operation", 0, torch.long)
        return {stage.label: step > i for i, stage in enumerate(self.stages)}

    def extra_failures(self, env, episode):
        pelvis_z = self.embodiment.tracked_pose_w(env)[:, 2] - env.scene.env_origins[:, 2]
        wrong_entrance = episode.task_state_buffer("wrong_entrance", False, torch.bool)
        forward_exit = episode.task_state_buffer("forward_exit_distance", 0.0, torch.float) > 0.10
        return wrong_entrance | forward_exit | (pelvis_z < G1_FLOOR_Z_M + 0.40)

    def plan(self, env, episode, env_id, mode):
        raise NotImplementedError(f"{self.name} accepts policy actions; no scripted {mode} controller is provided.")
