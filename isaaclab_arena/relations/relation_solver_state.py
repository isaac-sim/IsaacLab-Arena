# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import get_anchor_objects

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference


class RelationSolverState:
    """Encapsulates position state during optimization.

    This class manages the mapping between objects and their positions,
    keeping anchor (fixed) and optimizable positions separate internally
    while providing an interface for position lookups.

    Positions are always stored as (N, num_objects, 3) where N = num_envs
    (N=1 for single-env).
    """

    def __init__(
        self,
        objects: list[Object | ObjectReference],
        initial_positions: list[dict[Object | ObjectReference, tuple[float, float, float]]],
    ):
        """Initialize optimization state.

        Args:
            objects: List of all Object or ObjectReference instances to track. Must include at least one
                object marked with IsAnchor() which serves as a fixed reference.
            initial_positions: One dict of starting positions per env. Length 1 = single-env,
                length > 1 = batched (one layout per env).
        """
        assert len(initial_positions) >= 1, "initial_positions must contain at least one dict."
        anchor_objects = get_anchor_objects(objects)
        assert len(anchor_objects) > 0, "No anchor object found in objects list."

        self._all_objects = objects
        self._anchor_objects: set[Object] = set(anchor_objects)
        self._optimizable_objects = [obj for obj in objects if obj not in self._anchor_objects]

        # Build object-to-index mapping
        self._obj_to_idx: dict[Object | ObjectReference, int] = {obj: i for i, obj in enumerate(objects)}

        # Extract positions from each env's dict
        self._num_envs = len(initial_positions)
        positions_per_env = []
        for d in initial_positions:
            positions = []
            for obj in objects:
                assert obj in d, f"Missing initial position for {obj.name}"
                positions.append(torch.tensor(d[obj], dtype=torch.float32))
            positions_per_env.append(positions)

        # Separate anchor positions from optimizable positions
        self._anchor_indices: set[int] = {self._obj_to_idx[obj] for obj in self._anchor_objects}
        self._anchor_positions: dict[int, torch.Tensor] = {
            idx: positions_per_env[0][idx].clone() for idx in self._anchor_indices
        }

        # Build optimizable positions tensor (excludes all anchors)
        # Always stored as (N, num_opt, 3) where N = num_envs
        self._optimizable_indices = [i for i in range(len(objects)) if i not in self._anchor_indices]
        if self._optimizable_indices:
            opt_tensors = [
                torch.stack([positions_per_env[e][i] for e in range(self._num_envs)])
                for i in self._optimizable_indices
            ]
            self._optimizable_positions = torch.stack(opt_tensors, dim=1)  # (N, num_opt, 3)
            self._optimizable_positions.requires_grad = True
        else:
            self._optimizable_positions = None

    @property
    def num_envs(self) -> int:
        """Number of environments (leading dimension N)."""
        return self._num_envs

    @property
    def optimizable_positions(self) -> torch.Tensor | None:
        """Tensor of optimizable positions (N, num_opt, 3), or None if all objects are anchors.

        This is the tensor that should be passed to the optimizer.
        """
        return self._optimizable_positions

    @property
    def optimizable_objects(self) -> list[Object]:
        """List of optimizable objects (excludes anchors)."""
        return self._optimizable_objects

    @property
    def anchor_objects(self) -> set[Object]:
        """Set of anchor objects (fixed during optimization)."""
        return self._anchor_objects

    def get_position(self, obj: Object | ObjectReference) -> torch.Tensor:
        """Get current position for an object.

        Args:
            obj: The object to get position for.

        Returns:
            Position tensor of shape (N, 3).

        Raises:
            KeyError: If object is not tracked by this state.
            RuntimeError: If requesting position for optimizable object when none exist.
        """
        idx = self._obj_to_idx[obj]
        if idx in self._anchor_indices:
            return self._anchor_positions[idx].unsqueeze(0).expand(self._num_envs, 3)
        if self._optimizable_positions is None:
            raise RuntimeError(f"No optimizable positions available for object '{obj.name}'")
        opt_idx = self._optimizable_indices.index(idx)
        return self._optimizable_positions[:, opt_idx, :]

    def get_all_positions_snapshot(self) -> list[tuple[float, float, float]]:
        """Get detached copy of all positions for history tracking.

        Returns:
            List of (x, y, z) positions for each object (in original order). Uses env 0.
        """
        return [tuple(self.get_position(obj)[0].detach().tolist()) for obj in self._all_objects]

    def get_final_positions(self) -> list[dict[Object | ObjectReference, tuple[float, float, float]]]:
        """Get final positions as a list of dicts, one per env.

        Returns:
            List of dictionaries with object instances as keys and (x, y, z) tuples as values.
        """
        out = []
        for e in range(self._num_envs):
            d: dict[Object | ObjectReference, tuple[float, float, float]] = {}
            for obj in self._all_objects:
                pos = self.get_position(obj)[e].detach().tolist()
                d[obj] = (pos[0], pos[1], pos[2])
            out.append(d)
        return out
