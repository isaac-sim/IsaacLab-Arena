# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free cuRobo IK-reachability prototype.

Builds a standalone cuRobo solver (``MotionGen`` by default, or the lighter ``IKSolver``) straight
from an embodiment's registered cuRobo config, with a collision world assembled from object bounding
boxes instead of live USD meshes. No ``SimulationApp``, no Isaac Lab env, no ``env.reset`` — only
cuRobo, a CUDA GPU, and pure-math frame transforms. This is the sim-free counterpart to the
env-coupled path in ``curobo_planner_utils`` + ``ik_utils`` and is intended to gate placement inside
the pool solver's build-time solve->validate->refill loop.

Frame convention: cuRobo IK goals and the collision world are expressed in the robot base frame. All
inputs here are world-frame poses plus a config-supplied robot base pose; everything is transformed
into the robot base frame before it reaches cuRobo.
"""

from __future__ import annotations

import logging
import os
import torch
import yaml
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg

# See ik_utils: cuRobo captures a CUDA graph on the IK solver's first solve and errors on a later
# "goal type" change (MotionGen warms up single-goal, then we issue a batched solve). Opt in to a
# graph reset on CUDA >= 12 instead of erroring. ``setdefault`` keeps any explicit user override.
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")

# Top-down grasp orientation (gripper approach axis pointing -Z) in the robot base frame, (x, y, z, w).
DOWN_FACING_QUAT_XYZW = (0.0, 1.0, 0.0, 0.0)


@dataclass
class SimFreeCuboid:
    """A collision obstacle described by an axis-aligned bounding box in the world frame.

    ``dims_xyz`` are full extents (edge lengths), matching cuRobo's ``Cuboid.dims``. ``quat_wxyz`` is
    the box orientation in the world frame; for an axis-aligned box built from a layout it is identity.
    """

    name: str
    center_xyz: tuple[float, float, float]
    dims_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Pick an explicit CUDA device, defaulting to the current one."""
    if device is not None:
        return torch.device(device)
    assert torch.cuda.is_available(), "Sim-free cuRobo IK requires a CUDA GPU."
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def build_robot_cfg_dict(curobo_cfg: CuroboEmbodimentCfg) -> dict:
    """Load an embodiment's cuRobo robot config and patch in its downloaded URDF, sim-free.

    In-memory equivalent of the robot-config half of ``make_planner_cfg``: pull the ``.yml`` + URDF
    from the asset server, splice the local URDF path into the kinematics block, and return the
    ``robot_cfg`` sub-dict ready for ``MotionGenConfig``/``IKSolverConfig.load_from_robot_config``.
    Kept import-light (only ``retrieve_file_path``) so it never pulls in the env-coupled planner.
    """
    from isaaclab.utils.assets import retrieve_file_path

    robot_cfg_path = retrieve_file_path(curobo_cfg.robot_cfg_template)
    robot_urdf_path = retrieve_file_path(curobo_cfg.robot_urdf)
    with open(robot_cfg_path) as f:
        robot_yaml = yaml.safe_load(f)
    robot_yaml["robot_cfg"]["kinematics"]["urdf_path"] = robot_urdf_path
    return robot_yaml["robot_cfg"]


def world_pose_to_robot_frame(
    pos_w: torch.Tensor,
    quat_w_xyzw: torch.Tensor,
    robot_base_pos_w: torch.Tensor,
    robot_base_quat_w_xyzw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-express a world-frame pose in the robot base frame.

    Frames: W = world, R = robot base, O = object. Returns ``(t_R_O, q_R_O_xyzw)``. This is the
    pure-math equivalent of the live-sim transform in ``sync_object_poses_in_robot_base_frame``.
    """
    R_R_W = math_utils.matrix_from_quat(robot_base_quat_w_xyzw.unsqueeze(0))[0].T
    q_R_W_xyzw = math_utils.quat_inv(robot_base_quat_w_xyzw.unsqueeze(0))[0]
    t_R_O = (R_R_W @ (pos_w - robot_base_pos_w).unsqueeze(-1)).squeeze(-1)
    q_R_O_xyzw = math_utils.quat_mul(q_R_W_xyzw.unsqueeze(0), quat_w_xyzw.unsqueeze(0))[0]
    return t_R_O, q_R_O_xyzw


def world_config_from_cuboids(
    cuboids: list[SimFreeCuboid],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
    device: str | torch.device | None = None,
):
    """Build a cuRobo ``WorldConfig`` of cuboids expressed in the robot base frame.

    Each obstacle's world pose is transformed into the robot base frame (the frame cuRobo's collision
    world and IK goals live in). Include anchor objects (e.g. a table) here as static cuboids.
    """
    from curobo.geom.types import Cuboid, WorldConfig

    dev = _resolve_device(device)
    robot_pos = torch.tensor(robot_base_pos_w, dtype=torch.float32, device=dev)
    robot_quat = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float32, device=dev)

    curobo_cuboids = []
    for c in cuboids:
        pos_w = torch.tensor(c.center_xyz, dtype=torch.float32, device=dev)
        # SimFreeCuboid holds a wxyz world quat; the transform math works in xyzw.
        quat_w_xyzw = math_utils.convert_quat(torch.tensor(c.quat_wxyz, dtype=torch.float32, device=dev), to="xyzw")
        t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(pos_w, quat_w_xyzw, robot_pos, robot_quat)
        q_R_O_wxyz = math_utils.convert_quat(q_R_O_xyzw, to="wxyz")
        # cuRobo Cuboid pose is [x, y, z, qw, qx, qy, qz].
        pose = t_R_O.tolist() + q_R_O_wxyz.tolist()
        curobo_cuboids.append(Cuboid(name=c.name, pose=pose, dims=list(c.dims_xyz)))
    return WorldConfig(cuboid=curobo_cuboids)


def top_down_grasp_pose_simfree(
    object_pos_w: tuple[float, float, float],
    object_quat_w_xyzw: tuple[float, float, float, float],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
    grasp_z_offset: float = 0.02,
    align_yaw_to_object: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Top-down grasp pose at an object's center, in the robot base frame, as a 4x4 transform.

    Pure-math counterpart of ``top_down_grasp_pose_in_robot_frame`` that reads inputs from a layout
    (object world pose + config-supplied robot base pose) rather than a live sim.
    """
    dev = _resolve_device(device)
    t_W_O = torch.tensor(object_pos_w, dtype=torch.float32, device=dev)
    q_W_O_xyzw = torch.tensor(object_quat_w_xyzw, dtype=torch.float32, device=dev)
    t_W_R = torch.tensor(robot_base_pos_w, dtype=torch.float32, device=dev)
    q_W_R_xyzw = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float32, device=dev)

    t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(t_W_O, q_W_O_xyzw, t_W_R, q_W_R_xyzw)
    t_R_O = t_R_O.clone()
    t_R_O[2] += grasp_z_offset

    R_down = math_utils.matrix_from_quat(
        torch.tensor(DOWN_FACING_QUAT_XYZW, dtype=torch.float32, device=dev).unsqueeze(0)
    )[0]
    if align_yaw_to_object:
        R_R_O = math_utils.matrix_from_quat(q_R_O_xyzw.unsqueeze(0))[0]
        yaw = torch.atan2(R_R_O[1, 0], R_R_O[0, 0])
        # A parallel-jaw grasp is symmetric under a 180 deg spin about the approach axis; fold the
        # yaw into [-pi/2, pi/2] to stay away from the wrist joint limit while matching the object.
        yaw = torch.remainder(yaw + torch.pi / 2, torch.pi) - torch.pi / 2
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        R_yaw = torch.eye(3, device=dev, dtype=t_R_O.dtype)
        R_yaw[0, 0], R_yaw[0, 1] = cos_y, -sin_y
        R_yaw[1, 0], R_yaw[1, 1] = sin_y, cos_y
        R_grasp = R_yaw @ R_down
    else:
        R_grasp = R_down

    return math_utils.make_pose(t_R_O, R_grasp)


class SimFreeIKReachability:
    """Standalone cuRobo IK-reachability oracle with no Isaac Sim / Isaac Lab env.

    Constructs a cuRobo solver from an embodiment's registered cuRobo config on an explicit CUDA
    device, holds a bounding-box collision world, and answers per-pose IK feasibility via a single
    batched solve. Mirrors the members the env-coupled ``check_ik_feasibility_batch_goal_poses``
    relies on (``_to_curobo_device``, ``_make_pose``, ``ik_solver``, ``logger``).
    """

    def __init__(
        self,
        curobo_cfg: CuroboEmbodimentCfg,
        device: str | torch.device | None = None,
        use_motion_gen: bool = True,
        collision_activation_distance: float = 0.01,
        num_seeds: int = 12,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        collision_cache_size: dict[str, int] | None = None,
        debug: bool = False,
    ) -> None:
        """Build and warm up the solver from a cuRobo embodiment config, sim-free.

        Args:
            curobo_cfg: The embodiment's registered ``CuroboEmbodimentCfg`` (robot yaml + URDF paths).
            device: Explicit CUDA device (e.g. ``"cuda:0"``); defaults to the current device.
            use_motion_gen: Build a full ``MotionGen`` (max fidelity with the env-coupled path, reuses
                its warmup) and use its ``ik_solver``. If False, build the lighter ``IKSolver`` alone.
            collision_activation_distance: Distance (m) at which cuRobo starts penalizing collisions.
            num_seeds: IK seeds optimized in parallel per pose.
            position_threshold: cuRobo internal IK position convergence threshold (m).
            rotation_threshold: cuRobo internal IK rotation convergence threshold.
            collision_cache_size: Collision cache sizes; defaults to ``{"obb": 150, "mesh": 150}``.
            debug: Enable cuRobo/planner debug logging.
        """
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.util.logger import setup_curobo_logger

        self.logger = logging.getLogger("SimFreeIKReachability")
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        setup_curobo_logger("info" if debug else "warn")

        self.device = _resolve_device(device)
        self.tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)
        collision_cache = collision_cache_size or {"obb": 150, "mesh": 150}

        self.robot_cfg = build_robot_cfg_dict(curobo_cfg)
        # Start with an empty collision world; update_world() fills it per layout.
        world_cfg = WorldConfig(cuboid=[])

        self.motion_gen = None
        if use_motion_gen:
            from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg,
                tensor_args=self.tensor_args,
                num_trajopt_seeds=num_seeds,
                collision_cache=collision_cache,
                collision_activation_distance=collision_activation_distance,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
            )
            self.motion_gen = MotionGen(motion_gen_config)
            self.logger.info("Warming up MotionGen (standalone, sim-free)...")
            self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
            self.ik_solver = self.motion_gen.ik_solver
        else:
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

            ik_config = IKSolverConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg,
                tensor_args=self.tensor_args,
                num_seeds=num_seeds,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                collision_cache=collision_cache,
                collision_activation_distance=collision_activation_distance,
                # Variable batch sizes across layouts; disable the fixed-batch CUDA graph.
                use_cuda_graph=False,
            )
            self.ik_solver = IKSolver(ik_config)

    @classmethod
    def from_embodiment(cls, embodiment, **kwargs) -> SimFreeIKReachability:
        """Build from an embodiment by looking up its registered cuRobo config.

        Convenience wrapper for callers that already hold an embodiment (importing the embodiment may
        pull in heavier Isaac Lab modules than passing a ``CuroboEmbodimentCfg`` directly).
        """
        from isaaclab_arena_curobo.embodiment_curobo_registry import get_curobo_cfg_for

        return cls(get_curobo_cfg_for(embodiment), **kwargs)

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor onto the cuRobo CUDA device / dtype (device isolation)."""
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _make_pose(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        *,
        quat_is_xyzw: bool = True,
    ):
        """Create a cuRobo ``Pose`` on the cuRobo device, converting xyzw->wxyz when needed."""
        from curobo.types.math import Pose

        position = self._to_curobo_device(position)
        quaternion = self._to_curobo_device(quaternion)
        quaternion_wxyz = torch.roll(quaternion, shifts=1, dims=-1) if quat_is_xyzw else quaternion
        return Pose(position=position, quaternion=quaternion_wxyz)

    def update_world(
        self,
        cuboids: list[SimFreeCuboid],
        robot_base_pos_w: tuple[float, float, float],
        robot_base_quat_w_xyzw: tuple[float, float, float, float],
    ) -> None:
        """Replace the collision world with cuboids derived from a layout's bounding boxes.

        ``load_collision_model`` (via cuRobo's ``update_world``) reloads the model rather than only
        moving obstacle poses, so this handles both moved objects and a changed object set per layout,
        as long as the obstacle count stays within the collision cache.
        """
        world_cfg = world_config_from_cuboids(cuboids, robot_base_pos_w, robot_base_quat_w_xyzw, self.device)
        if self.motion_gen is not None:
            self.motion_gen.update_world(world_cfg)
        else:
            self.ik_solver.update_world(world_cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def check_ik_feasibility_simfree(
    solver: SimFreeIKReachability,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched IK feasibility of all ``target_poses`` against a standalone sim-free solver.

    Variant of ``check_ik_feasibility_batch_goal_poses`` that runs against a ``SimFreeIKReachability``
    instead of a ``CuroboPlanner``. Same math and thresholds; the only difference is the solver source.

    Args:
        solver: A built ``SimFreeIKReachability``.
        target_poses: ``(b, 4, 4)`` end-effector goal transforms in the robot base frame.
        seed_config: Optional joint seed tensor.
        position_threshold: Max position error (m) to count as feasible.
        rotation_threshold: Max rotation error (rad) to count as feasible.

    Returns:
        ``(feasible, position_error, rotation_error)``, each length ``b`` and aligned with the input;
        errors are the best-seed values per pose.
    """
    target_poses = solver._to_curobo_device(target_poses)
    positions, rotations = math_utils.unmake_pose(target_poses)
    goal_pose = solver._make_pose(
        position=positions,
        quaternion=math_utils.quat_from_matrix(rotations),  # xyzw
        quat_is_xyzw=True,
    )

    ik_seed = None
    if seed_config is not None:
        ik_seed = solver._to_curobo_device(seed_config)
        while ik_seed.dim() < 3:
            ik_seed = ik_seed.unsqueeze(0)

    ik_result = solver.ik_solver.solve_batch(goal_pose, seed_config=ik_seed)

    num_poses = positions.shape[0]
    pos_err = ik_result.position_error.view(num_poses, -1)
    rot_err = ik_result.rotation_error.view(num_poses, -1)
    best_idx = pos_err.argmin(dim=1, keepdim=True)
    best_pos_err = pos_err.gather(1, best_idx).squeeze(1)
    best_rot_err = rot_err.gather(1, best_idx).squeeze(1)
    feasible = (best_pos_err < position_threshold) & (best_rot_err < rotation_threshold)

    solver.logger.debug(f"Batch IK feasibility: {int(feasible.sum().item())}/{num_poses} feasible")
    return feasible, best_pos_err, best_rot_err
