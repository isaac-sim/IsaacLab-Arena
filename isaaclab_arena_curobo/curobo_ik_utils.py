# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone IK feasibility utilities that operate on a CuroboPlanner instance.

Kept outside the IsaacLab submodule so changes can be committed directly
to IsaacLab-Arena without forking the upstream planner. The batched-solve core lives in
``curobo_frame_utils.solve_ik_feasibility`` and is shared with the standalone path; importing it also
sets the CUDA-graph-reset env var before any solve.
"""

from __future__ import annotations

import logging
import torch
from dataclasses import dataclass

import isaaclab.utils.math as math_utils
from curobo.geom.types import Cuboid, WorldConfig

from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.utils.device import resolve_cuda_device
from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg
from isaaclab_arena_curobo.curobo_frame_utils import (
    load_patched_robot_yaml,
    solve_ik_feasibility,
    top_down_grasp_matrix,
    world_pose_to_robot_frame,
)


@dataclass
class AABBCollisionCuboid:
    """A collision obstacle described by an axis-aligned bounding box in the world frame.

    ``dims_xyz`` are full extents (edge lengths), matching cuRobo's ``Cuboid.dims``. ``quat_wxyz`` is
    the box orientation in the world frame; for an axis-aligned box built from a layout it is identity.
    """

    name: str
    center_xyz: tuple[float, float, float]
    dims_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def get_aabb_collision_cuboid_for_object(
    obj: ObjectBase, pos_w: tuple[float, float, float], quat_w_xyzw: tuple[float, ...]
) -> AABBCollisionCuboid:
    """Axis-aligned bounding-box collision cuboid for an object at its layout pose (world frame).

    The bounding box is object-local, so its center offset is rotated by the object's world orientation
    and added to the root position -- placing e.g. a table box at its true mid-height rather than at the
    root.
    """
    bbox = obj.get_bounding_box()
    dims = tuple(float(v) for v in bbox.size[0].tolist())
    quat_t = torch.tensor(quat_w_xyzw, dtype=torch.float32)
    rotation = math_utils.matrix_from_quat(quat_t.unsqueeze(0))[0]
    center_world = torch.tensor(pos_w, dtype=torch.float32) + rotation @ bbox.center[0].to(torch.float32)
    quat_wxyz = tuple(float(v) for v in math_utils.convert_quat(quat_t, to="wxyz").tolist())
    return AABBCollisionCuboid(
        name=obj.name,
        center_xyz=tuple(float(v) for v in center_world.tolist()),
        dims_xyz=dims,
        quat_wxyz=quat_wxyz,
    )


def world_config_from_cuboids(
    cuboids: list[AABBCollisionCuboid],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
    device: str | torch.device | None = None,
):
    """Build a cuRobo ``WorldConfig`` of cuboids expressed in the robot base frame.

    Each obstacle's world pose is transformed into the robot base frame. Include anchor objects (e.g. a table) here as static cuboids.
    """

    dev = resolve_cuda_device(device)
    robot_pos = torch.tensor(robot_base_pos_w, dtype=torch.float32, device=dev)
    robot_quat = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float32, device=dev)

    curobo_cuboids = []
    for c in cuboids:
        pos_w = torch.tensor(c.center_xyz, dtype=torch.float32, device=dev)
        # AABBCollisionCuboid holds a wxyz world quat; the transform math works in xyzw.
        quat_w_xyzw = math_utils.convert_quat(torch.tensor(c.quat_wxyz, dtype=torch.float32, device=dev), to="xyzw")
        t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(pos_w, quat_w_xyzw, robot_pos, robot_quat)
        q_R_O_wxyz = math_utils.convert_quat(q_R_O_xyzw, to="wxyz")
        # cuRobo Cuboid pose is [x, y, z, qw, qx, qy, qz].
        pose = t_R_O.tolist() + q_R_O_wxyz.tolist()
        curobo_cuboids.append(Cuboid(name=c.name, pose=pose, dims=list(c.dims_xyz)))
    return WorldConfig(cuboid=curobo_cuboids)


def top_down_grasp_pose_from_world_poses(
    object_pos_w: tuple[float, float, float],
    object_quat_w_xyzw: tuple[float, float, float, float],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
    grasp_z_offset: float = 0.02,
    align_yaw_to_object: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Top-down grasp pose at an object's center, in the robot base frame, as a 4x4 transform."""
    dev = resolve_cuda_device(device)
    # Pure-math computation of the object pose in the robot base frame
    t_W_O = torch.tensor(object_pos_w, dtype=torch.float32, device=dev)
    q_W_O_xyzw = torch.tensor(object_quat_w_xyzw, dtype=torch.float32, device=dev)
    t_W_R = torch.tensor(robot_base_pos_w, dtype=torch.float32, device=dev)
    q_W_R_xyzw = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float32, device=dev)

    t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(t_W_O, q_W_O_xyzw, t_W_R, q_W_R_xyzw)
    return top_down_grasp_matrix(t_R_O, q_R_O_xyzw, grasp_z_offset, align_yaw_to_object)


class StandaloneIKReachability:
    """Standalone cuRobo IK-reachability oracle with no Isaac Sim / Isaac Lab env.

    Constructs a cuRobo solver from an embodiment's registered cuRobo config on an explicit CUDA
    device, holds a bounding-box collision world, and answers per-pose IK feasibility via a single
    batched solve. Feasibility is pose reachability (position/rotation convergence) only; the
    collision-free check is not wired up yet (see ``solve_ik_feasibility``'s ``require_collision_free``).
    """

    def __init__(
        self,
        curobo_cfg: CuroboEmbodimentCfg,
        device: str | torch.device | None = None,
        collision_activation_distance: float = 0.01,
        num_seeds: int = 12,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        collision_cache_size: dict[str, int] | None = None,
        debug: bool = False,
    ) -> None:
        """Build and warm up the solver from a cuRobo embodiment config, standalone.

        Args:
            curobo_cfg: The embodiment's registered ``CuroboEmbodimentCfg`` (robot yaml + URDF paths).
            device: Explicit CUDA device (e.g. ``"cuda:0"``); defaults to the current device.
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
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        self.logger = logging.getLogger("StandaloneIKReachability")
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        setup_curobo_logger("info" if debug else "warn")

        self.device = resolve_cuda_device(device)
        self.tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)
        collision_cache = collision_cache_size or {"obb": 150, "mesh": 150}

        self.robot_cfg = load_patched_robot_yaml(curobo_cfg)["robot_cfg"]
        # Start with an empty collision world; update_world() fills it per layout.
        world_cfg = WorldConfig(cuboid=[])

        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            world_cfg,
            tensor_args=self.tensor_args,
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            collision_cache=collision_cache,
            collision_activation_distance=collision_activation_distance,
            use_cuda_graph=False,
        )
        self.ik_solver = IKSolver(ik_config)

    @classmethod
    def from_embodiment(cls, embodiment, **kwargs) -> StandaloneIKReachability:
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
        cuboids: list[AABBCollisionCuboid],
        robot_base_pos_w: tuple[float, float, float],
        robot_base_quat_w_xyzw: tuple[float, float, float, float],
    ) -> None:
        """Replace the collision world with cuboids derived from a layout's bounding boxes.

        ``load_collision_model`` (via cuRobo's ``update_world``) reloads the model rather than only
        moving obstacle poses, so this handles both moved objects and a changed object set per layout,
        as long as the obstacle count stays within the collision cache.
        """
        world_cfg = world_config_from_cuboids(cuboids, robot_base_pos_w, robot_base_quat_w_xyzw, self.device)
        self.ik_solver.update_world(world_cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def check_ik_feasibility(
    pose_ctx,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched IK feasibility of all ``target_poses`` against a context's cuRobo IK solver.

    Serves both paths with the same math and thresholds; the only difference is where the solver comes
    from. Pass a ``StandaloneIKReachability`` (build-time, exposes ``ik_solver``) or a ``CuroboPlanner``
    (env-coupled, exposes ``motion_gen.ik_solver``).

    Args:
        pose_ctx: A ``StandaloneIKReachability`` or ``CuroboPlanner`` supplying the pose plumbing and IK solver.
        target_poses: ``(b, 4, 4)`` end-effector goal transforms in the robot base frame.
        seed_config: Optional joint seed tensor.
        position_threshold: Max position error (m) to count as feasible.
        rotation_threshold: Max rotation error (rad) to count as feasible.

    Returns:
        ``(feasible, position_error, rotation_error)``, each length ``b`` and aligned with the input;
        errors are the best-seed values per pose.
    """
    ik_solver = getattr(pose_ctx, "ik_solver", None)
    if ik_solver is None:
        ik_solver = pose_ctx.motion_gen.ik_solver
    return solve_ik_feasibility(
        pose_ctx,
        ik_solver,
        target_poses,
        seed_config=seed_config,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
        # Future hook: a context may opt into the (not-yet-supported) collision-free check by exposing
        # ``require_collision_free``; no current context does, so this stays False.
        require_collision_free=getattr(pose_ctx, "require_collision_free", False),
    )
