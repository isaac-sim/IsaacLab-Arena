# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free cuRobo IK-reachability oracle (no Isaac Sim / Isaac Lab env)."""

from __future__ import annotations

import logging
import torch

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.util.logger import setup_curobo_logger
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from isaaclab_arena.utils.device import resolve_cuda_device
from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg
from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
from isaaclab_arena_curobo.utils.ik_solver_utils import AABBCollisionCuboid, world_config_from_cuboids
from isaaclab_arena_curobo.utils.robot_cfg_utils import load_patched_robot_yaml


class CuroboIKSolver:
    """Sim-free cuRobo IK-reachability oracle with no Isaac Sim / Isaac Lab env.

    Constructs a cuRobo solver from an embodiment's registered cuRobo config on an explicit CUDA
    device, holds a bounding-box collision world, and answers per-pose IK feasibility via a single
    batched solve. Feasibility is pose reachability (position/rotation convergence) only; the
    collision-free check is not wired up yet (see ``solve_ik_feasibility``'s ``require_collision_free``).
    """

    # Note(xinjieyao, 2026-07-23): When validating params like EventTermCfg.params, Isaac Lab's configclass recursively
    # scans through every object's instance dict (__dict__) with no cycle guard.
    # Curobo's IKSolver has a deep reference trees with circular deps and raw CUDA/C++ handles, so
    # it results in an infinite loop traversing Curobo's cyclic objects.
    # By defining __slots__, python suppresses the creation of __dict__ and stop before traversing into Curobo's objects.
    # Args:
    #     logger: The logger for the solver.
    #     device: The cuda device for the solver.
    #     tensor_args: The tensor arguments for the solver.
    #     robot_cfg: The curobo robot configuration for the solver.
    #     ik_solver: The IK solver for the solver.
    __slots__ = ("logger", "device", "tensor_args", "robot_cfg", "ik_solver")

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
        """Build and warm up the solver from a cuRobo embodiment config, sim-free.

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
        self.logger = logging.getLogger("CuroboIKSolver")
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
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _make_pose(
        self,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        *,
        quat_is_xyzw: bool = True,
    ) -> CuroboPose:
        """Create a cuRobo ``Pose`` on the cuRobo device, converting xyzw->wxyz for cuRobo."""
        position = self._to_curobo_device(position)
        quaternion = self._to_curobo_device(quaternion)
        quaternion_wxyz = torch.roll(quaternion, shifts=1, dims=-1) if quat_is_xyzw else quaternion
        return CuroboPose(position=position, quaternion=quaternion_wxyz)

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
