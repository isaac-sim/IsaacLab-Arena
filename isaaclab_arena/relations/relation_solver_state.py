# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab_arena.assets.dummy_object import DummyObject


class RelationSolverState:
    """Encapsulates position state during optimization.

    This class manages the mapping between objects and their positions,
    keeping anchor (fixed) and optimizable positions separate internally
    while providing an interface for position lookups.
    """

    def __init__(
        self,
        objects: list[DummyObject],
        anchor_object: DummyObject,
    ):
        """Initialize optimization state from objects.

        Args:
            objects: List of all DummyObject instances to track.
            anchor_object: The fixed reference object (won't be optimized).
        """
        self._objects = objects
        self._anchor_object = anchor_object

        # Build object-to-index mapping
        self._obj_to_idx: dict[DummyObject, int] = {obj: i for i, obj in enumerate(objects)}

        # Extract initial positions from all objects
        positions = []
        for obj in objects:
            pose = obj.get_initial_pose()
            assert pose is not None, f"Pose is None for {obj.name}"
            positions.append(torch.tensor(pose.position_xyz, dtype=torch.float32))

        # Separate anchor from optimizable positions
        self._anchor_idx = self._obj_to_idx[anchor_object]
        self._anchor_position = positions[self._anchor_idx].clone()

        # Build optimizable positions tensor (excludes anchor)
        self._optimizable_indices = [i for i in range(len(objects)) if i != self._anchor_idx]
        self._optimizable_positions = torch.stack([positions[i] for i in self._optimizable_indices])
        self._optimizable_positions.requires_grad = True

    @property
    def optimizable_positions(self) -> torch.Tensor:
        """Tensor of optimizable positions (shape: [N-1, 3]).

        This is the tensor that should be passed to the optimizer.
        """
        return self._optimizable_positions

    @property
    def objects(self) -> list[DummyObject]:
        """List of all tracked objects."""
        return self._objects

    def get_position(self, obj: DummyObject) -> torch.Tensor:
        """Get current position for an object.

        Args:
            obj: The object to get position for.

        Returns:
            Position tensor (x, y, z).

        Raises:
            KeyError: If object is not tracked by this state.
        """
        idx = self._obj_to_idx[obj]
        if idx == self._anchor_idx:
            return self._anchor_position
        opt_idx = self._optimizable_indices.index(idx)
        return self._optimizable_positions[opt_idx]

    def get_all_positions_snapshot(self) -> list[list[float]]:
        """Get detached copy of all positions for history tracking.

        Returns:
            List of [x, y, z] positions for each object (in original order).
        """
        return [self.get_position(obj).detach().tolist() for obj in self._objects]

    def get_final_positions_dict(self) -> dict[DummyObject, tuple[float, float, float]]:
        """Get final positions as a dictionary mapping objects to positions.

        Returns:
            Dictionary with object instances as keys and (x, y, z) tuples as values.
        """
        result: dict[DummyObject, tuple[float, float, float]] = {}
        for obj in self._objects:
            pos = self.get_position(obj).detach().tolist()
            result[obj] = (pos[0], pos[1], pos[2])
        return result
