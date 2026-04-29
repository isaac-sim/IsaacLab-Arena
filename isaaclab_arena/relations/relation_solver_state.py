# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import get_anchor_objects

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class RelationSolverState:
    """Encapsulates position state during optimization.

    This class manages the mapping between objects and their positions,
    keeping anchor (fixed) and optimizable positions separate internally
    while providing an interface for position lookups.

    Positions are always stored as (batch_size, num_objects, 3).
    """

    def __init__(
        self,
        objects: list[ObjectBase],
        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]],
        device: torch.device | None = None,
    ):
        """Initialize optimization state.

        Args:
            objects: List of all ObjectBase instances to track. Must include at least one
                object marked with IsAnchor() which serves as a fixed reference.
            initial_positions: List of dicts (one per env). Length 1 = single-env,
                length > 1 = batched.
            device: Torch device for all tensors. Defaults to CPU.
        """
        assert len(initial_positions) >= 1, "initial_positions must contain at least one dict."
        anchor_objects = get_anchor_objects(objects)
        assert len(anchor_objects) > 0, "No anchor object found in objects list."

        self._all_objects = objects
        self._anchor_objects: set[ObjectBase] = set(anchor_objects)
        self._optimizable_objects = [obj for obj in objects if obj not in self._anchor_objects]

        # Build object-to-index mapping
        self._obj_to_idx: dict[ObjectBase, int] = {obj: i for i, obj in enumerate(objects)}

        self._device = device or torch.device("cpu")
        self._batch_size = len(initial_positions)

        # Validate that every dict contains all objects before building the tensor.
        for d in initial_positions:
            for obj in objects:
                assert obj in d, f"Missing initial position for {obj.name}"

        # Build all positions as a single (N, num_objects, 3) tensor in one call.
        pos_nested = [[d[obj] for obj in objects] for d in initial_positions]
        all_positions = torch.tensor(pos_nested, dtype=torch.float32, device=self._device)

        # Separate anchor positions from optimizable positions
        self._anchor_indices: set[int] = {self._obj_to_idx[obj] for obj in self._anchor_objects}
        # Anchors must be identical across envs (they are fixed reference points).
        for idx in self._anchor_indices:
            for env_idx in range(1, self._batch_size):
                assert torch.allclose(all_positions[0, idx], all_positions[env_idx, idx]), (
                    f"Anchor '{objects[idx].name}' has different positions across envs "
                    f"(env 0: {all_positions[0, idx].tolist()}, env {env_idx}: {all_positions[env_idx, idx].tolist()})"
                )
        self._anchor_positions: dict[int, torch.Tensor] = {
            idx: all_positions[0, idx].clone() for idx in self._anchor_indices
        }

        # Pre-build anchor positions as (1, num_objects, 3) for fast _reconstruct_all_positions.
        self._anchor_pos_tensor = torch.zeros(1, len(objects), 3, dtype=torch.float32, device=self._device)
        for idx, pos in self._anchor_positions.items():
            self._anchor_pos_tensor[0, idx, :] = pos

        # Build optimizable positions tensor by slicing from the full tensor.
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
    def optimizable_objects(self) -> list[ObjectBase]:
        """List of optimizable objects (excludes anchors)."""
        return self._optimizable_objects

    @property
    def anchor_objects(self) -> set[ObjectBase]:
        """Set of anchor objects (fixed during optimization)."""
        return self._anchor_objects

    def get_position(self, obj: ObjectBase) -> torch.Tensor:
        """Get current position for an object.

        Args:
            obj: The object to get position for.

        Returns:
            Position tensor of shape (batch_size, 3).

        Raises:
            KeyError: If object is not tracked by this state.
            RuntimeError: If requesting position for optimizable object when none exist.
        """
        idx = self._obj_to_idx[obj]
        if idx in self._anchor_indices:
            return self._anchor_positions[idx].unsqueeze(0).expand(self._batch_size, 3)
        if self._optimizable_positions is None:
            raise RuntimeError(f"No optimizable positions available for object '{obj.name}'")
        opt_idx = self._global_to_opt_idx[idx]
        return self._optimizable_positions[:, opt_idx, :]

    def get_all_positions_snapshot(self) -> list[tuple[float, float, float]]:
        """Get detached copy of all positions for history tracking.

        Returns:
            List of (x, y, z) positions for each object (in original order). Uses env 0.
        """
        return [tuple(self.get_position(obj)[0].detach().tolist()) for obj in self._all_objects]

    def get_final_positions(self) -> list[dict[ObjectBase, tuple[float, float, float]]]:
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
