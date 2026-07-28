# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import FaceTo, get_anchor_objects, get_relation
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import MINIMUM_FACING_DIRECTION_XY_M

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class RelationSolverState:
    """Batched object poses and bounding boxes used by the relation solver."""

    def __init__(
        self,
        objects: list[PlaceableAsset],
        initial_positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        device: torch.device | None = None,
        env_bboxes: dict[PlaceableAsset, OrientedBoundingBox] | None = None,
        rotations: list[dict[PlaceableAsset, tuple[float, float, float, float]]] | None = None,
        collision_objects: list[CollisionObject] | None = None,
    ):
        """Initialize optimization state.

        Args:
            objects: Placement assets to track. Must include at least one
                object marked with IsAnchor() which serves as a fixed reference.
            initial_positions: List of dicts (one per env). Length 1 = single-env,
                length > 1 = batched.
            device: Torch device for all tensors. Defaults to CPU.
            env_bboxes: Optional per-env bounding boxes keyed by object.
            rotations: Fixed candidate rotations in xyzw order. FaceTo subjects
                derive their effective world-Z rotation from current subject/target positions.
            collision_objects: Optional fixed background obstacles that participate in
                no-overlap collision only (never in relation constraints). They keep a
                constant world bounding box and are not optimized. Must be disjoint from objects.
        """
        assert len(initial_positions) >= 1, "initial_positions must contain at least one dict."
        anchor_objects = get_anchor_objects(objects)
        assert len(anchor_objects) > 0, "No anchor object found in objects list."

        self._all_objects = objects
        self._anchor_objects: set[PlaceableAsset] = set(anchor_objects)
        self._optimizable_objects = [obj for obj in objects if obj not in self._anchor_objects]
        self._collision_objects: list[CollisionObject] = list(collision_objects) if collision_objects else []
        assert not (set(self._collision_objects) & set(objects)), (
            "collision_objects must be disjoint from placed objects; an object cannot be "
            "both optimized and a fixed collision obstacle."
        )

        self._obj_to_idx: dict[PlaceableAsset, int] = {obj: i for i, obj in enumerate(objects)}

        self._device = device or torch.device("cpu")
        self._batch_size = len(initial_positions)

        for d in initial_positions:
            for obj in objects:
                assert obj in d, f"Missing initial position for {obj.name}"

        pos_nested = [[d[obj] for obj in objects] for d in initial_positions]
        all_positions = torch.tensor(pos_nested, dtype=torch.float32, device=self._device)

        self._anchor_indices: set[int] = {self._obj_to_idx[obj] for obj in self._anchor_objects}
        for idx in self._anchor_indices:
            pose = objects[idx].get_initial_pose()
            assert isinstance(pose, Pose), f"Anchor '{objects[idx].name}' must have a fixed Pose."
            fixed_position = torch.tensor(pose.position_xyz, dtype=torch.float32, device=self._device)
            assert torch.allclose(all_positions[:, idx], fixed_position.expand(self._batch_size, 3)), (
                f"Anchor '{objects[idx].name}' supplied positions must match its fixed Pose position "
                f"{pose.position_xyz}, got {all_positions[:, idx].tolist()}."
            )
        self._anchor_positions: dict[int, torch.Tensor] = {
            idx: all_positions[0, idx].clone() for idx in self._anchor_indices
        }

        self._anchor_pos_tensor = torch.zeros(1, len(objects), 3, dtype=torch.float32, device=self._device)
        for idx, pos in self._anchor_positions.items():
            self._anchor_pos_tensor[0, idx, :] = pos

        self._optimizable_indices = [i for i in range(len(objects)) if i not in self._anchor_indices]
        self._global_to_opt_idx: dict[int, int] = {
            global_idx: opt_idx for opt_idx, global_idx in enumerate(self._optimizable_indices)
        }
        if self._optimizable_indices:
            self._opt_idx_tensor = torch.tensor(self._optimizable_indices, dtype=torch.long, device=self._device)
            self._optimizable_positions = all_positions[:, self._opt_idx_tensor, :].clone()
            self._optimizable_positions.requires_grad = True
        else:
            self._opt_idx_tensor = None
            self._optimizable_positions = None

        self._env_bboxes = env_bboxes
        rotations = rotations or [{} for _ in range(self._batch_size)]
        assert (
            len(rotations) == self._batch_size
        ), f"rotations must contain one dictionary per candidate, got {len(rotations)} for N={self._batch_size}."
        identity = (0.0, 0.0, 0.0, 1.0)
        supplied_rotation_objects = {obj for candidate in rotations for obj in candidate}
        invalid_rotation_objects = supplied_rotation_objects - set(self._optimizable_objects)
        assert not invalid_rotation_objects, (
            "Rotation keys must belong to optimizable objects, got "
            f"{sorted(obj.name for obj in invalid_rotation_objects)}."
        )
        candidate_rotations: dict[PlaceableAsset, torch.Tensor] = {}
        for obj in self._optimizable_objects:
            values = [
                torch.as_tensor(candidate.get(obj, identity), dtype=torch.float32, device=self._device)
                for candidate in rotations
            ]
            assert all(
                value.shape == (4,) for value in values
            ), f"Candidate rotations for '{obj.name}' must each have shape (4,)."
            rotation_tensor = torch.stack(values)
            assert rotation_tensor.shape == (
                self._batch_size,
                4,
            ), f"Candidate rotations for '{obj.name}' must have shape (N, 4)."
            assert torch.isfinite(rotation_tensor).all(), f"Candidate rotations for '{obj.name}' must be finite."
            norms = torch.linalg.vector_norm(rotation_tensor, dim=-1)
            assert torch.allclose(
                norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5
            ), f"Candidate rotations for '{obj.name}' must be unit quaternions."
            candidate_rotations[obj] = rotation_tensor
        self._base_rotations = {obj: candidate_rotations[obj] for obj in self._optimizable_objects}

        # Fixed world boxes do not depend on optimized positions.
        self._fixed_obstacle_world_bboxes: dict[PlaceableAsset | CollisionObject, OrientedBoundingBox] = {}
        for obj in self._anchor_objects:
            pose = obj.get_initial_pose()
            assert isinstance(pose, Pose), f"Anchor '{obj.name}' must have a fixed Pose."
            local_bbox = env_bboxes[obj] if env_bboxes is not None and obj in env_bboxes else obj.get_bounding_box()
            self._fixed_obstacle_world_bboxes[obj] = local_bbox.to(self._device).transformed(
                pose.position_xyz, pose.rotation_xyzw
            )
        for obj in self._collision_objects:
            self._fixed_obstacle_world_bboxes[obj] = obj.get_world_bounding_box().to(self._device)

    @property
    def device(self) -> torch.device:
        """Torch device for all position tensors."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Number of independent position sets (leading dimension of position tensors)."""
        return self._batch_size

    @property
    def optimizable_positions(self) -> torch.Tensor | None:
        """Tensor of optimizable positions (batch_size, num_optimizable, 3), or None if all objects are anchors.

        This is the tensor that should be passed to the optimizer.
        """
        return self._optimizable_positions

    @property
    def optimizable_objects(self) -> list[PlaceableAsset]:
        """List of optimizable objects (excludes anchors)."""
        return self._optimizable_objects

    @property
    def anchor_objects(self) -> set[PlaceableAsset]:
        """Set of anchor objects (fixed during optimization)."""
        return self._anchor_objects

    @property
    def collision_objects(self) -> list[CollisionObject]:
        """Copy of the collision-only fixed obstacles (constant world pose, no relation constraints)."""
        return list(self._collision_objects)

    def get_position(self, obj: PlaceableAsset) -> torch.Tensor:
        """Get current position for an object.

        Args:
            obj: The object to get position for.

        Returns:
            Position tensor of shape (batch_size, 3).
        """
        idx = self._obj_to_idx[obj]
        if idx in self._anchor_indices:
            return self._anchor_positions[idx].unsqueeze(0).expand(self._batch_size, 3)
        assert self._optimizable_positions is not None, f"No optimizable position for '{obj.name}'."
        opt_idx = self._global_to_opt_idx[idx]
        return self._optimizable_positions[:, opt_idx, :]

    def get_fixed_obstacle_world_bbox(self, obj: PlaceableAsset | CollisionObject) -> OrientedBoundingBox:
        """Return the cached constant world bounding box for an anchor or collision object."""
        assert (
            obj in self._fixed_obstacle_world_bboxes
        ), f"'{obj.name}' is not a fixed obstacle (anchor or collision object) tracked by this state."
        return self._fixed_obstacle_world_bboxes[obj]

    def get_rotation(self, obj: PlaceableAsset) -> torch.Tensor:
        """Return effective candidate rotations with shape (N, 4), in xyzw order.

        Fixed candidate rotations come from RotateAroundSolution/random_yaw_init.
        FaceTo is the one dynamic case: its world-Z quaternion is recomputed from
        current positions on every loss evaluation, while positions remain the
        solver's only optimized parameters. This position-derived rotation remains
        differentiable, so geometry losses can propagate through it to both the
        FaceTo subject and target positions.
        """
        base = self._base_rotations[obj]
        face_to = get_relation(obj, FaceTo)
        if face_to is None:
            return base
        delta = self.get_position(face_to.parent)[:, :2] - self.get_position(obj)[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=1)
        valid = distance > MINIMUM_FACING_DIRECTION_XY_M
        safe_delta = torch.where(valid.unsqueeze(1), delta, delta.new_tensor([1.0, 0.0]))
        yaw = torch.atan2(safe_delta[:, 1], safe_delta[:, 0])
        half_yaw = yaw * 0.5
        facing = torch.stack(
            [
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(half_yaw),
                torch.cos(half_yaw),
            ],
            dim=1,
        )
        return torch.where(valid.unsqueeze(1), facing, base)

    def get_base_bbox(self, obj: PlaceableAsset) -> OrientedBoundingBox:
        """Return the asset-local bounding box for an object."""
        if self._env_bboxes is not None and obj in self._env_bboxes:
            return self._env_bboxes[obj].to(self._device)
        return obj.get_bounding_box().to(self._device)

    def get_bbox(self, obj: PlaceableAsset) -> OrientedBoundingBox:
        """Return the candidate-oriented local bounding box for an object."""
        bbox = self.get_base_bbox(obj)
        if obj in self._anchor_objects:
            return bbox
        return bbox.rotated_by_quat_unchecked(self.get_rotation(obj))

    def get_all_positions_snapshot(self) -> list[tuple[float, float, float]]:
        """Get detached copy of all positions for history tracking.

        Returns:
            List of (x, y, z) positions for each object (in original order). Uses env 0.
        """
        return [tuple(self.get_position(obj)[0].detach().tolist()) for obj in self._all_objects]

    def get_final_positions(self) -> list[dict[PlaceableAsset, tuple[float, float, float]]]:
        """Get final positions as a list of dicts, one per env.

        Returns:
            List of dictionaries with object instances as keys and (x, y, z) tuples as values.
        """
        # Reconstruct the full (N, num_objects, 3) tensor and transfer to CPU in one call.
        full = self._reconstruct_all_positions()
        pos_list = full.detach().cpu().tolist()
        return [
            {obj: tuple(pos_list[env_idx][obj_idx]) for obj_idx, obj in enumerate(self._all_objects)}
            for env_idx in range(self._batch_size)
        ]

    def _reconstruct_all_positions(self) -> torch.Tensor:
        """Reconstruct a full (batch_size, num_objects, 3) tensor from anchor and optimizable parts."""
        full = self._anchor_pos_tensor.expand(self._batch_size, -1, -1).clone()
        if self._optimizable_positions is not None:
            full[:, self._opt_idx_tensor, :] = self._optimizable_positions
        return full
