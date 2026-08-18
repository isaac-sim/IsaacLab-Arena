# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-step recorder terms for raw physical state.

These record *raw* channels only. Every derived quantity — stage boundaries, approach yaw, minimum
end-effector distance, contact duration, in-hand slip — is computed offline from what is written
here. Recomputing a margin costs seconds; re-running the episodes that produced it costs GPU-hours,
so nothing is thresholded or reduced online.

Attach by overriding ``TaskBase.get_recorder_term_cfg()``; the terms land alongside the existing
metric recorders in the episode HDF5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import torch
import warp as wp
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.configclass import make_configclass


class RawStateRecorder(RecorderTerm):
    """Records one raw per-step channel, produced by ``cfg.extract``."""

    def __init__(self, cfg: RawStateRecorderCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.name = cfg.name
        self._extract = cfg.extract

    def record_post_step(self):
        value = self._extract(self._env)
        assert value.shape[0] == self._env.num_envs, f"{self.name}: expected leading dim {self._env.num_envs}"
        return self.name, value


@configclass
class RawStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = RawStateRecorder
    name: str = MISSING
    extract: Callable[[ManagerBasedEnv], torch.Tensor] = MISSING


# --- extractors -------------------------------------------------------------------------------
# Each returns (num_envs, D). Bound with functools.partial-style closures rather than params so the
# recorder stays a single class.


def object_pose(object_name: str) -> Callable[[ManagerBasedEnv], torch.Tensor]:
    """World pose of a rigid object as ``[x, y, z, qx, qy, qz, qw]``.

    The quaternion is **(x, y, z, w)** — Isaac Lab's ``root_link_pose_w`` convention, which is the
    opposite of the ``(w, x, y, z)`` used by ``ee_quat`` and by Isaac Lab's math helpers. Offline
    consumers must not assume a convention; read it from here.

    ``root_link_state_w`` is deprecated upstream and raises ``NotImplementedError``, so pose and
    velocity are read from their separate properties.
    """

    def _f(env: ManagerBasedEnv) -> torch.Tensor:
        return wp.to_torch(env.scene[object_name].data.root_link_pose_w)

    return _f


def object_velocity(object_name: str) -> Callable[[ManagerBasedEnv], torch.Tensor]:
    """Linear and angular velocity of a rigid object as ``[vx, vy, vz, wx, wy, wz]``."""

    def _f(env: ManagerBasedEnv) -> torch.Tensor:
        return wp.to_torch(env.scene[object_name].data.root_link_vel_w)

    return _f


def fingertip_positions(
    frame_name: str = "ee_frame",
    targets: tuple[str, str] = ("tool_leftfinger", "tool_rightfinger"),
) -> Callable[[ManagerBasedEnv], torch.Tensor]:
    """World positions of both fingertips as [lx, ly, lz, rx, ry, rz].

    Target frames are resolved **by name**. On the DROID embodiment the frame at index 0 is
    ``end_effector``, which resolves to ``panda_link0`` — the robot base, not the gripper — so
    positional indexing silently measures from the wrong body.
    """

    def _f(env: ManagerBasedEnv) -> torch.Tensor:
        frame = env.scene[frame_name]
        names = list(frame.data.target_frame_names)
        idx = [names.index(t) for t in targets]
        pos = wp.to_torch(frame.data.target_pos_w)[:, idx, :]
        return pos.reshape(pos.shape[0], -1)

    return _f


def ee_body_pose(body_name: str = "base_link", asset_name: str = "robot") -> Callable[[ManagerBasedEnv], torch.Tensor]:
    """Gripper-body pose as ``[x, y, z, qx, qy, qz, qw]`` — quaternion is **(x, y, z, w)**.

    This is the body Arena's own ``ee_pos`` / ``ee_quat`` observations use, so it is the same
    reference the policy sees. It exists because the ``ee_frame`` target frames are not usable for
    orientation: they are Robotiq four-bar *linkage* frames whose separation grows as the jaws close
    and which are 0.3 mm apart when fully open, so the direction between them is numerically
    meaningless exactly when the gripper is open.

    CORRECTION (2026-08-14): this docstring previously claimed ``body_quat_w`` was ``(w, x, y, z)``
    and that two conventions coexisted. That was wrong. Isaac Lab documents ``body_link_quat_w`` as
    "Orientation (x, y, z, w)" (``assets/articulation/base_articulation_data.py:986``), and it
    measures that way: decoded as ``(x, y, z, w)`` the fixed ``panda_link8 -> base_link`` mount is
    constant across arm configurations to 2.1e-08 m, whereas ``(w, x, y, z)`` makes it drift
    2.9e-02 m. **Both this channel and ``root_link_pose_w`` are ``(x, y, z, w)``.** See EXP-011.
    """

    def _f(env: ManagerBasedEnv) -> torch.Tensor:
        robot = env.scene[asset_name]
        idx = robot.data.body_names.index(body_name)
        pos = wp.to_torch(robot.data.body_pos_w)[:, idx, :]
        quat = wp.to_torch(robot.data.body_quat_w)[:, idx, :]
        return torch.cat([pos, quat], dim=-1)

    return _f


def joint_positions(joint_names: tuple[str, ...], asset_name: str = "robot") -> Callable[[ManagerBasedEnv], torch.Tensor]:
    """Joint positions in the order given, resolved by name."""

    def _f(env: ManagerBasedEnv) -> torch.Tensor:
        robot = env.scene[asset_name]
        order = list(robot.data.joint_names)
        idx = [order.index(n) for n in joint_names]
        return wp.to_torch(robot.data.joint_pos)[:, idx]

    return _f


# --- assembly ---------------------------------------------------------------------------------

DROID_ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


def make_raw_state_recorder_cfg(
    object_name: str,
    destination_name: str | None = None,
    arm_joints: tuple[str, ...] = DROID_ARM_JOINTS,
    gripper_joint: str = "finger_joint",
    ee_frame_name: str = "ee_frame",
    sim_state_stride: int | None = None,
):
    """Build the raw-channel recorder set for a single-arm pick-and-place task.

    ``sim_state_stride`` adds restorable scene-state capture as a *field* of the returned configclass.
    It has to be a field, not an attribute set afterwards: ``RecorderManager`` enumerates the config's
    dataclass fields, so anything attached with ``setattr`` is silently ignored.
    """
    terms: list[tuple[str, type, RawStateRecorderCfg]] = [
        ("raw_object_pose", RecorderTermCfg, RawStateRecorderCfg(name="raw_object_pose", extract=object_pose(object_name))),
        ("raw_object_velocity", RecorderTermCfg, RawStateRecorderCfg(name="raw_object_velocity", extract=object_velocity(object_name))),
        ("raw_fingertip_pos", RecorderTermCfg, RawStateRecorderCfg(name="raw_fingertip_pos", extract=fingertip_positions(ee_frame_name))),
        ("raw_ee_body_pose", RecorderTermCfg, RawStateRecorderCfg(name="raw_ee_body_pose", extract=ee_body_pose())),
        ("raw_arm_joint_pos", RecorderTermCfg, RawStateRecorderCfg(name="raw_arm_joint_pos", extract=joint_positions(arm_joints))),
        ("raw_gripper_joint_pos", RecorderTermCfg, RawStateRecorderCfg(name="raw_gripper_joint_pos", extract=joint_positions((gripper_joint,)))),
    ]
    if destination_name is not None:
        terms.append(
            ("raw_destination_pose", RecorderTermCfg, RawStateRecorderCfg(name="raw_destination_pose", extract=object_pose(destination_name)))
        )
    if sim_state_stride is not None:
        from isaaclab_arena.recording.sim_state_terms import make_sim_state_recorder_cfg

        terms.append(("sim_state", RecorderTermCfg, make_sim_state_recorder_cfg(stride=sim_state_stride)))
    return make_configclass("RawStateRecorderManagerCfg", terms, bases=(ActionStateRecorderManagerCfg,))()
