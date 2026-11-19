# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free cuRobo IK-reachability oracle (no Isaac Sim / Isaac Lab env)."""

from __future__ import annotations

import logging
import torch

import isaaclab.utils.math as math_utils
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.logging import setup_logger
from curobo.types import DeviceCfg, GoalToolPose
from curobo.types import Pose as CuroboPose

from isaaclab_arena.utils.device import resolve_cuda_device
from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg
from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
from isaaclab_arena_curobo.utils.ik_solver_utils import AABBCollisionCuboid
from isaaclab_arena_curobo.utils.robot_cfg_utils import load_patched_robot_yaml


class CuroboIKSolver:
    """Sim-free cuRobo IK-reachability oracle with no Isaac Sim / Isaac Lab env.

    Constructs a cuRobo solver from an embodiment's registered cuRobo config on an explicit CUDA
    device, holds a bounding-box collision world, and answers per-pose IK feasibility via a single
    batched solve. Feasibility is pose reachability (position/rotation convergence), optionally also
    collision-freeness against that world (see ``solve_ik_feasibility``'s ``require_collision_free``).
    """

    # Note(xinjieyao, 2026-07-23): When validating params like EventTermCfg.params, Isaac Lab's configclass recursively
    # scans through every object's instance dict (__dict__) with no cycle guard.
    # Curobo's IKSolver has a deep reference trees with circular deps and raw CUDA/C++ handles, so
    # it results in an infinite loop traversing Curobo's cyclic objects.
    # By defining __slots__, python suppresses the creation of __dict__ and stops before traversing into Curobo's objects.
    # Args:
    #     logger: The logger for the solver.
    #     device: The cuda device for the solver.
    #     device_cfg: The cuRobo device and dtype for the solver's tensors.
    #     robot_cfg: The curobo robot configuration for the solver.
    #     ik_solver: The IK solver for the solver.
    #     hand_link_names: The robot's hand links, muted during a collision-free solve.
    #     max_batch_size: Most problems one solve may hold.
    __slots__ = (
        "logger",
        "device",
        "device_cfg",
        "robot_cfg",
        "ik_solver",
        "hand_link_names",
        "max_batch_size",
    )

    def __init__(
        self,
        curobo_cfg: CuroboEmbodimentCfg,
        device: str | torch.device | None = None,
        collision_activation_distance: float = 0.0,
        num_seeds: int = 12,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.1,
        collision_cache_size: dict[str, int] | None = None,
        max_batch_size: int = 1,
        use_cuda_graph: bool = True,
        debug: bool = False,
    ) -> None:
        """Build the solver from a cuRobo embodiment config, sim-free.

        Args:
            curobo_cfg: The embodiment's registered ``CuroboEmbodimentCfg`` (robot yaml + URDF paths).
            device: Explicit CUDA device (e.g. ``"cuda:0"``); defaults to the current device.
            collision_activation_distance: Distance (m) at which cuRobo starts penalizing collisions. A positive margin
                holds the arm that far off every obstacle, so grasps near a support surface stall about that far from
                the goal and read as unreachable even though a collision-free solution exists.
            num_seeds: IK seeds optimized in parallel per pose.
            position_threshold: cuRobo internal IK position convergence threshold (m).
            rotation_threshold: cuRobo internal IK rotation convergence threshold (rad, geodesic angle).
            collision_cache_size: Obstacle slots per type; defaults to ``{"cuboid": 150}``. A world with more
                obstacles than its slots fails to load.
            max_batch_size: Most poses one ``solve_ik_feasibility`` call may pass; smaller batches are padded to it.
            use_cuda_graph: Replay captured CUDA graphs for the solve, which cuRobo v2 recommends for speed.
            debug: Enable cuRobo/planner debug logging.
        """
        self.logger = logging.getLogger("CuroboIKSolver")
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        setup_logger("info" if debug else "warning")

        self.device = resolve_cuda_device(device)
        self.device_cfg = DeviceCfg(device=self.device, dtype=torch.float32)

        self.robot_cfg = load_patched_robot_yaml(curobo_cfg)["robot_cfg"]
        self.hand_link_names = list(curobo_cfg.hand_link_names)
        self.max_batch_size = max_batch_size

        ik_config = InverseKinematicsCfg.create(
            robot=self.robot_cfg,
            # cuRobo only builds a collision checker when given a scene; update_world() fills it per layout.
            # Every problem of a batched solve has its own collision world, so each needs an (empty) world up
            # front to allocate its obstacle storage.
            scene_model=[{} for _ in range(max_batch_size)],
            collision_cache=collision_cache_size or {"cuboid": 150},
            device_cfg=self.device_cfg,
            num_seeds=num_seeds,
            position_tolerance=position_threshold,
            orientation_tolerance=rotation_threshold,
            optimizer_collision_activation_distance=collision_activation_distance,
            max_batch_size=max_batch_size,
            multi_env=True,
            use_cuda_graph=use_cuda_graph,
        )
        self.ik_solver = InverseKinematics(ik_config)

    def __deepcopy__(self, memory_map: dict[int, object]) -> CuroboIKSolver:
        """Return the live solver for ``copy.deepcopy`` instead of duplicating it.

        Isaac Lab's configclass deep-copies the ``EventTermCfg.params`` this solver rides in, but cuRobo's
        ``IKSolver`` holds un-copyable CUDA/ctypes handles and is a shared, effectively-immutable oracle,
        so the copy can only be the same instance.

        Args:
            memory_map: ``copy.deepcopy``'s ``id(original) -> copy`` cache; registering ``self`` in it makes every
                other reference to this solver resolve to the same shared instance.
        """
        memory_map[id(self)] = self
        return self

    @classmethod
    def from_embodiment(cls, embodiment, **kwargs) -> CuroboIKSolver:
        """Build from an embodiment by looking up its registered cuRobo config.

        Convenience wrapper for callers that already hold an embodiment (importing the embodiment may
        pull in heavier Isaac Lab modules than passing a ``CuroboEmbodimentCfg`` directly).
        """
        return cls(get_embodiment_curobo_cfg(embodiment), **kwargs)

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor onto the cuRobo CUDA device / dtype (device isolation)."""
        return tensor.to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)

    def _make_pose(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        *,
        quat_is_xyzw: bool = True,
    ) -> CuroboPose:
        """Create a cuRobo ``Pose`` on the cuRobo device, converting xyzw->wxyz for cuRobo."""
        position = self._to_curobo_device(position).contiguous()
        quaternion = self._to_curobo_device(quaternion)
        quaternion_wxyz = torch.roll(quaternion, shifts=1, dims=-1) if quat_is_xyzw else quaternion
        return CuroboPose(position=position, quaternion=quaternion_wxyz.contiguous())

    def _make_goal_tool_pose(self, pose: CuroboPose) -> GoalToolPose:
        """Wrap a batch of end-effector goal poses as cuRobo's per-tool-frame IK goal."""
        return GoalToolPose.from_poses({self.ik_solver.tool_frames[0]: pose})

    def update_world(
        self,
        cuboids: list[AABBCollisionCuboid],
        robot_base_pos_w: tuple[float, float, float],
        robot_base_quat_w_xyzw: tuple[float, float, float, float],
    ) -> None:
        """Replace the collision world of every pose of a batched solve with the same cuboids.

        The cuboids are expressed in the robot base frame. Use ``load_collision_worlds`` to give each pose its own
        world.
        """
        self.load_collision_worlds(
            [cuboids] * self.max_batch_size,
            [robot_base_pos_w] * self.max_batch_size,
            [robot_base_quat_w_xyzw] * self.max_batch_size,
        )

    def load_collision_worlds(
        self,
        cuboids_per_problem: list[list[AABBCollisionCuboid]],
        robot_base_pos_w_per_problem: list[tuple[float, float, float]],
        robot_base_quat_w_xyzw_per_problem: list[tuple[float, float, float, float]],
    ) -> None:
        """Load one collision world per pose of the next batched solve; the i-th pose is solved in the i-th world.

        Each world is the cuboids expressed in that problem's robot base frame. The worlds are loaded by writing
        the obstacle storage directly, so any set of obstacles fits as long as it stays within the collision cache.

        Args:
            cuboids_per_problem: Obstacle cuboids of each problem's world, at most ``max_batch_size`` problems.
            robot_base_pos_w_per_problem: Robot base position of each problem.
            robot_base_quat_w_xyzw_per_problem: Robot base orientation of each problem.
        """
        num_problems = len(cuboids_per_problem)
        assert (
            num_problems <= self.max_batch_size
        ), f"{num_problems} problems exceed max_batch_size={self.max_batch_size}"
        assert (
            len(robot_base_pos_w_per_problem) == num_problems
            and len(robot_base_quat_w_xyzw_per_problem) == num_problems
        )
        cuboid_storage = self.ik_solver.scene_collision_checker.data.cuboids
        counts = [len(cuboids) for cuboids in cuboids_per_problem]
        assert (
            max(counts, default=0) <= cuboid_storage.max_n
        ), f"a world has {max(counts)} cuboids, more than the {cuboid_storage.max_n} obstacle slots"
        # Express every obstacle of every world in its robot base frame with one batched computation, then
        # write them all into the solver's obstacle storage at once.
        cuboids = [cuboid for world in cuboids_per_problem for cuboid in world]
        cuboid_storage.enable[:num_problems] = 0
        cuboid_storage.count[:num_problems] = torch.tensor(counts, dtype=torch.int32, device=self.device)
        if cuboids:
            # Index of the world and of the slot inside it that each cuboid of the flat list is written to.
            env_idx_list: list[int] = []
            slot_idx_list: list[int] = []
            for problem_idx, num_cuboids in enumerate(counts):
                env_idx_list.extend([problem_idx] * num_cuboids)
                slot_idx_list.extend(range(num_cuboids))
            env_idx = torch.tensor(env_idx_list, dtype=torch.long, device=self.device)
            slot_idx = torch.tensor(slot_idx_list, dtype=torch.long, device=self.device)
            # Stack the cuboids, and the base pose of the world each belongs to, into one tensor per quantity.
            pos_w = self._to_curobo_device(torch.tensor([c.pose_W_O.position_xyz for c in cuboids]))
            quat_w_xyzw = self._to_curobo_device(torch.tensor([c.pose_W_O.rotation_xyzw for c in cuboids]))
            dims = self._to_curobo_device(torch.tensor([c.dims_xyz for c in cuboids]))
            base_pos_w = self._to_curobo_device(torch.tensor(robot_base_pos_w_per_problem))[env_idx]
            base_quat_w_xyzw = self._to_curobo_device(torch.tensor(robot_base_quat_w_xyzw_per_problem))[env_idx]

            # Express every cuboid pose in its robot base frame R: T_R_O = T_R_W * T_W_O.
            q_R_W_xyzw = math_utils.quat_inv(base_quat_w_xyzw)
            t_R_O = math_utils.quat_apply(q_R_W_xyzw, pos_w - base_pos_w)
            q_R_O_xyzw = math_utils.quat_mul(q_R_W_xyzw, quat_w_xyzw)
            # cuRobo stores each obstacle's inverse pose, as [x, y, z, qw, qx, qy, qz].
            inv_pose = self._make_pose(t_R_O, q_R_O_xyzw).inverse().get_pose_vector()
            # Write each cuboid into its world's slot and mark the slot as in use.
            cuboid_storage.dims[env_idx, slot_idx, :3] = dims
            cuboid_storage.inv_pose[env_idx, slot_idx, :7] = inv_pose
            cuboid_storage.enable[env_idx, slot_idx] = 1
        # cuRobo keeps a Python list of cuboid names per world; fill the unused slots with None.
        next_cuboid = 0
        for env, num_cuboids in enumerate(counts):
            names = [cuboid.name for cuboid in cuboids[next_cuboid : next_cuboid + num_cuboids]]
            cuboid_storage.names[env] = names + [None] * (cuboid_storage.max_n - num_cuboids)
            next_cuboid += num_cuboids
