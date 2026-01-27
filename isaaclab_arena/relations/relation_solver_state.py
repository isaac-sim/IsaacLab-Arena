# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import find_anchor_object

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference


class RelationSolverState:
    """Encapsulates position state during optimization.

    This class manages the mapping between objects and their positions,
    keeping anchor (fixed) and optimizable positions separate internally
    while providing an interface for position lookups.
    """

    def __init__(
        self,
        objects: list[Object | ObjectReference],
        initial_positions: dict[Object | ObjectReference, tuple[float, float, float]],
    ):
        """Initialize optimization state.

        Args:
            objects: List of all Object or ObjectReference instances to track. Must include
                exactly one object marked with IsAnchor() which serves as the fixed reference.
            initial_positions: Starting positions for all objects (including anchor).
        """
        anchor_object = find_anchor_object(objects)
        assert anchor_object is not None, "No anchor object found in objects list."

        self._all_objects = objects
        self._anchor_object = anchor_object
        self._optimizable_objects = [obj for obj in objects if obj is not anchor_object]

        # Build object-to-index mapping
        self._obj_to_idx: dict[Object | ObjectReference, int] = {obj: i for i, obj in enumerate(objects)}

        # Extract positions from the provided dict
        positions = []
        for obj in objects:
            assert obj in initial_positions, f"Missing initial position for {obj.name}"
            positions.append(torch.tensor(initial_positions[obj], dtype=torch.float32))

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
    def optimizable_objects(self) -> list[Object | ObjectReference]:
        """List of optimizable objects (excludes anchor)."""
        return self._optimizable_objects

    def get_position(self, obj: Object | ObjectReference) -> torch.Tensor:
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

    def get_all_positions_snapshot(self) -> list[tuple[float, float, float]]:
        """Get detached copy of all positions for history tracking.

        Returns:
            List of (x, y, z) positions for each object (in original order).
        """
        return [tuple(self.get_position(obj).detach().tolist()) for obj in self._all_objects]

    def get_final_positions_dict(self) -> dict[Object | ObjectReference, tuple[float, float, float]]:
        """Get final positions as a dictionary mapping objects to positions.

        Returns:
            Dictionary with object instances as keys and (x, y, z) tuples as values.
        """
        result: dict[Object | ObjectReference, tuple[float, float, float]] = {}
        for obj in self._all_objects:
            pos = self.get_position(obj).detach().tolist()
            result[obj] = (pos[0], pos[1], pos[2])
        return result
