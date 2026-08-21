# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
    InitialStateRecorderCfg,
    PostStepProcessedActionsRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
)
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.configclass import combine_configclass_instances

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


class PostStepFlatPolicyActionObservationRecorder(RecorderTerm):
    """Recorder term that records the ``action`` observation group at the end of each step.

    Mirrors the locomanip mimic patch's post-step action recorder, but no-ops on envs
    whose policy does not expose an ``action`` observation group so it can be safely
    enabled for any task.
    """

    def record_post_step(self):
        obs_buf = getattr(self._env, "obs_buf", None)
        if not isinstance(obs_buf, dict) or "action" not in obs_buf:
            return None, None
        return "action", obs_buf["action"]


@configclass
class PostStepFlatPolicyActionObservationRecorderCfg(RecorderTermCfg):
    """Configuration for the post-step ``action`` observation recorder term."""

    class_type: type[RecorderTerm] = PostStepFlatPolicyActionObservationRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder manager extended with arena-specific recorder terms."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()
    record_post_step_flat_policy_action_observations = PostStepFlatPolicyActionObservationRecorderCfg()


class EpisodeIdentityRecorder(RecorderTerm):
    """Recorder term that stamps each exported demo with the episode it came from.

    Demos are named ``demo_0``, ``demo_1``, ... in export order, which carries no reference back to
    the ``(env_id, episode_in_env)`` pair the episode recorder writes to its JSONL. Recording the
    pair alongside the trajectory makes that join explicit.
    """

    def __init__(self, cfg: RecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._first_reset = True

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        # The initial reset touches every env before any episode has run; there is nothing to stamp.
        if self._first_reset:
            self._first_reset = False
            return None, None
        env_ids = list(range(self._env.num_envs)) if env_ids is None else [int(env_id) for env_id in env_ids]
        # Runs before the env advances its counters, so this is still the finishing episode's index.
        episode_indices = [self._env.get_episode_index(env_id) for env_id in env_ids]
        return "episode_id", {
            "env_id": torch.tensor(env_ids, dtype=torch.int64, device=self._env.device),
            "episode_in_env": torch.tensor(episode_indices, dtype=torch.int64, device=self._env.device),
        }


@configclass
class EpisodeIdentityRecorderCfg(RecorderTermCfg):
    """Configuration for the episode identity recorder term."""

    class_type: type[RecorderTerm] = EpisodeIdentityRecorder


class EndEffectorPosesRecorder(RecorderTerm):
    """Recorder term that exports end-effector (and fingertip) poses and velocities from ``ee_frame``.

    Pose and velocity share the same env-aligned frame as scene ``root_pose`` / ``root_velocity``.
    No-ops when the scene has no sensor named by :attr:`EndEffectorPosesRecorderCfg.frame_transformer_name`.
    """

    def __init__(self, cfg: RecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        # Cached once: each tracked frame (EE, fingertips, ...) may sit on a different robot link.
        self._body_ids_by_frame: dict[str, int] | None = None

    def _resolve_body_ids(self, sensor) -> dict[str, int]:
        """Map each tracked frame name to its parent link index on the robot.

        Args:
            sensor: The end-effector frame transformer sensor.
        """
        if self._body_ids_by_frame is not None:
            return self._body_ids_by_frame

        robot: Articulation = self._env.scene[self.cfg.asset_name]
        cfg_by_name = {frame.name: frame for frame in sensor.cfg.target_frames}
        body_ids_by_frame: dict[str, int] = {}
        for frame_name in sensor.data.target_frame_names:
            frame_cfg = cfg_by_name[frame_name]
            # Leaf of the USD prim path is the articulation link name (e.g. panda_hand).
            body_name = frame_cfg.prim_path.rstrip("/").rsplit("/", 1)[-1]
            body_ids, matched_names = robot.find_bodies(body_name)
            assert (
                len(body_ids) == 1
            ), f"Expected exactly one robot body matching '{body_name}' for frame '{frame_name}', got {matched_names}"
            body_ids_by_frame[frame_name] = body_ids[0]
        self._body_ids_by_frame = body_ids_by_frame
        return body_ids_by_frame

    def _end_effector_poses(self, env_ids: Sequence[int] | None) -> dict | None:
        """Return per-frame poses and velocities, or None when the scene has no such sensor.

        Args:
            env_ids: Environments to report, or None for all of them.
        """
        sensor = self._env.scene.sensors.get(self.cfg.frame_transformer_name)
        if sensor is None:
            return None

        body_ids_by_frame = self._resolve_body_ids(sensor)
        robot: Articulation = self._env.scene[self.cfg.asset_name]

        # sensor.data already includes embodiment offsets (e.g. the grasp point beyond the link origin).
        target_pos_w = sensor.data.target_pos_w.torch
        # Positions: subtract env origins so they match scene root_pose. Velocities: unchanged
        # (env origins are fixed translations, so world and env velocities are the same).
        positions = target_pos_w - self._env.scene.env_origins.unsqueeze(1)
        orientations = sensor.data.target_quat_w.torch

        link_pos_w = robot.data.body_link_pos_w.torch
        link_vel_w = robot.data.body_link_vel_w.torch

        frames: dict = {}
        for frame_index, frame_name in enumerate(sensor.data.target_frame_names):
            body_id = body_ids_by_frame[frame_name]
            link_lin_vel = link_vel_w[:, body_id, :3]
            link_ang_vel = link_vel_w[:, body_id, 3:]
            # Point velocity on a rigid link: v_point = v_link + ω × r, with r the
            # world-space offset from the link origin to the recorded frame point.
            offset_w = target_pos_w[:, frame_index] - link_pos_w[:, body_id]
            linear_velocity = link_lin_vel + torch.cross(link_ang_vel, offset_w, dim=-1)
            # As the offset r is fixed, the angular velocity matches the parent link.
            angular_velocity = link_ang_vel

            # env_ids is set on partial reset; None means record every parallel env.
            if env_ids is not None:
                frames[frame_name] = {
                    "position": positions[env_ids, frame_index],
                    "orientation": orientations[env_ids, frame_index],
                    "linear_velocity": linear_velocity[env_ids],
                    "angular_velocity": angular_velocity[env_ids],
                }
            else:
                frames[frame_name] = {
                    "position": positions[:, frame_index],
                    "orientation": orientations[:, frame_index],
                    "linear_velocity": linear_velocity,
                    "angular_velocity": angular_velocity,
                }
        return frames

    def record_post_reset(self, env_ids: Sequence[int] | None):
        # Paired with record_post_step: without this, the pose right after reset (before the
        # first action) would never be recorded.
        poses = self._end_effector_poses(env_ids)
        return (None, None) if poses is None else ("initial_state/kinematics", poses)

    def record_post_step(self):
        poses = self._end_effector_poses(env_ids=None)
        return (None, None) if poses is None else ("states/kinematics", poses)


@configclass
class EndEffectorPosesRecorderCfg(RecorderTermCfg):
    """Configuration for the end-effector pose recorder term."""

    class_type: type[RecorderTerm] = EndEffectorPosesRecorder

    frame_transformer_name: str = "ee_frame"
    """Name of the scene's end-effector frame transformer; the term no-ops when it is absent."""

    asset_name: str = "robot"
    """Scene entity name of the articulation that owns the end-effector frames."""


class GripperStateRecorder(RecorderTerm):
    """Recorder term that records how far each binary gripper has closed, alongside the scene states.

    The opening is normalised against the gripper action term's own open and close commands, so it
    reads 0 fully open and 1 fully closed for every embodiment without per-robot constants. Recording
    it spares consumers from locating the driver joint among the articulation's joints and rescaling
    it by limits the dataset does not carry.
    """

    def _gripper_states(self, env_ids: Sequence[int] | None, include_command: bool) -> dict | None:
        """Return per-gripper openings, or None when the embodiment has no binary gripper.

        Args:
            env_ids: Environments to report, or None for all of them.
            include_command: Whether to also report which command the gripper was last given.
        """
        # Imported here because the module pulls in USD, which must not load before SimulationApp
        # starts: a second copy of USD alongside Kit's own crashes the process during startup.
        from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointAction

        action_manager = self._env.action_manager
        states = {}
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            if not isinstance(term, BinaryJointAction):
                continue
            # BinaryJointAction exposes no public accessor for its joints or its commands.
            asset: Articulation = term._asset  # noqa: SLF001
            joint_ids = term._joint_ids  # noqa: SLF001
            open_command = term._open_command  # noqa: SLF001
            close_command = term._close_command  # noqa: SLF001

            travel = close_command - open_command
            assert (travel != 0).all(), "Binary gripper open and close commands must be distinct"
            joint_positions = wp.to_torch(asset.data.joint_pos)[:, joint_ids]
            opening = (joint_positions - open_command) / travel
            gripper_state = {"position": opening[env_ids] if env_ids is not None else opening}

            if include_command:
                # process_actions() assigns the command tensors verbatim, so this comparison is exact.
                is_commanded_open = (term.processed_actions == open_command).all(dim=-1)
                gripper_state["is_commanded_open"] = (
                    is_commanded_open[env_ids] if env_ids is not None else is_commanded_open
                )

            states[term_name] = gripper_state
        return states or None

    def record_post_reset(self, env_ids: Sequence[int] | None):
        # No action has been processed yet, so only the measured opening is meaningful here.
        states = self._gripper_states(env_ids, include_command=False)
        return (None, None) if states is None else ("initial_state/kinematics", states)

    def record_post_step(self):
        states = self._gripper_states(None, include_command=True)
        return (None, None) if states is None else ("states/kinematics", states)


@configclass
class GripperStateRecorderCfg(RecorderTermCfg):
    """Configuration for the gripper state recorder term."""

    class_type: type[RecorderTerm] = GripperStateRecorder


@configclass
class TrajectoryRecorderTermsCfg:
    """Recorder terms capturing per-step robot and object trajectories."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
    record_episode_id = EpisodeIdentityRecorderCfg()
    record_end_effector_poses = EndEffectorPosesRecorderCfg()
    record_gripper_state = GripperStateRecorderCfg()


def add_trajectory_recorder_terms(recorder_cfg: RecorderManagerBaseCfg) -> RecorderManagerBaseCfg:
    """Return ``recorder_cfg`` extended with the per-step trajectory recorder terms.

    The metric terms already on ``recorder_cfg`` are preserved, because metrics are computed by
    reading their terms back out of the same exported dataset.

    Args:
        recorder_cfg: The composed recorder manager config to extend.
    """
    return combine_configclass_instances(
        "TrajectoryRecorderManagerCfg",
        recorder_cfg,
        TrajectoryRecorderTermsCfg(),
        bases=(RecorderManagerBaseCfg,),
    )
