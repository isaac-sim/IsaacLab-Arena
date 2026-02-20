# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import Relation, get_anchor_objects

if TYPE_CHECKING:
    from isaaclab_arena.relations.placeable_entity import PlaceableEntity


class RelationSolverState:
    """Encapsulates position state during optimization.

    This class manages the mapping between objects and their positions,
    keeping anchor (fixed) and optimizable positions separate internally
    while providing an interface for position lookups.
    """

    def __init__(
        self,
        objects: list[PlaceableEntity],
        initial_positions: dict[PlaceableEntity, tuple[float, float, float]],
    ):
        """Initialize optimization state.

        Args:
            objects: List of all PlaceableEntity instances to track (Object, ObjectReference,
                EmbodimentBase, etc.). Must include at least one object marked with
                IsAnchor() which serves as a fixed reference.
            initial_positions: Starting positions for all objects (including anchors).
        """
        anchor_objects = get_anchor_objects(objects)
        assert len(anchor_objects) > 0, "No anchor object found in objects list."

        self._all_objects = objects
        self._anchor_objects: set[PlaceableEntity] = set(anchor_objects)
        self._optimizable_objects = [obj for obj in objects if obj not in self._anchor_objects]

        # Build object-to-index mapping
        self._obj_to_idx: dict[PlaceableEntity, int] = {obj: i for i, obj in enumerate(objects)}

        # Validate that all relation parents are in the objects list
        objects_set = set(objects)
        for obj in objects:
            for relation in obj.get_spatial_relations():
                if isinstance(relation, Relation) and relation.parent not in objects_set:
                    tracked_names = [o.name for o in objects]
                    raise ValueError(
                        f"Object '{obj.name}' has a {type(relation).__name__}() relation with parent "
                        f"'{relation.parent.name}', but '{relation.parent.name}' is not in the objects "
                        "list passed to the solver.\n"
                        f"  Tracked objects: {tracked_names}\n"
                        f"  Hint: Make sure '{relation.parent.name}' is added to the Scene or otherwise "
                        "included in the placeables list."
                    )

        # Extract positions from the provided dict
        positions = []
        for obj in objects:
            assert obj in initial_positions, f"Missing initial position for {obj.name}"
            positions.append(torch.tensor(initial_positions[obj], dtype=torch.float32))

        # Separate anchor positions from optimizable positions
        self._anchor_indices: set[int] = {self._obj_to_idx[obj] for obj in self._anchor_objects}
        self._anchor_positions: dict[int, torch.Tensor] = {idx: positions[idx].clone() for idx in self._anchor_indices}

        # Build optimizable positions tensor (excludes all anchors)
        self._optimizable_indices = [i for i in range(len(objects)) if i not in self._anchor_indices]
        if self._optimizable_indices:
            self._optimizable_positions = torch.stack([positions[i] for i in self._optimizable_indices])
            self._optimizable_positions.requires_grad = True
        else:
            self._optimizable_positions = None

    @property
    def optimizable_positions(self) -> torch.Tensor | None:
        """Tensor of optimizable positions (shape: [N-num_anchors, 3]), or None if all objects are anchors.

        This is the tensor that should be passed to the optimizer.
        """
        return self._optimizable_positions

    @property
    def optimizable_objects(self) -> list[PlaceableEntity]:
        """List of optimizable objects (excludes anchors)."""
        return self._optimizable_objects

    @property
    def anchor_objects(self) -> set[PlaceableEntity]:
        """Set of anchor objects (fixed during optimization)."""
        return self._anchor_objects

    def get_position(self, obj: PlaceableEntity) -> torch.Tensor:
        """Get current position for an object.

        Args:
            obj: The object to get position for.

        Returns:
            Position tensor (x, y, z).

        Raises:
            KeyError: If object is not tracked by this state.
            RuntimeError: If requesting position for optimizable object when none exist.
        """
        idx = self._obj_to_idx[obj]
        if idx in self._anchor_indices:
            return self._anchor_positions[idx]
        if self._optimizable_positions is None:
            raise RuntimeError(f"No optimizable positions available for object '{obj.name}'")
        opt_idx = self._optimizable_indices.index(idx)
        return self._optimizable_positions[opt_idx]

    def get_all_positions_snapshot(self) -> list[tuple[float, float, float]]:
        """Get detached copy of all positions for history tracking.

        Returns:
            List of (x, y, z) positions for each object (in original order).
        """
        return [tuple(self.get_position(obj).detach().tolist()) for obj in self._all_objects]

    def get_final_positions_dict(self) -> dict[PlaceableEntity, tuple[float, float, float]]:
        """Get final positions as a dictionary mapping objects to positions.

        Returns:
            Dictionary with PlaceableEntity instances as keys and (x, y, z) tuples as values.
        """
        result: dict[PlaceableEntity, tuple[float, float, float]] = {}
        for obj in self._all_objects:
            pos = self.get_position(obj).detach().tolist()
            result[obj] = (pos[0], pos[1], pos[2])
        return result
