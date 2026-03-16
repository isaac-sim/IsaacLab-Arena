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
    while providing an interface for position lookups. Supports single-env
    and batched (num_envs) modes.
    """

    def __init__(
        self,
        objects: list[Object | ObjectReference],
        initial_positions: dict[Object | ObjectReference, tuple[float, float, float]] | None = None,
        initial_positions_per_env: list[dict[Object | ObjectReference, tuple[float, float, float]]] | None = None,
    ):
        """Initialize optimization state.

        Args:
            objects: List of all Object or ObjectReference instances to track. Must include at least one
                object marked with IsAnchor() which serves as a fixed reference.
            initial_positions: Starting positions for all objects (single-env). Use when
                initial_positions_per_env is None.
            initial_positions_per_env: When provided, batched mode: one dict per env. Length must
                match num_envs; initial_positions is ignored.
        """
        anchor_objects = get_anchor_objects(objects)
        assert len(anchor_objects) > 0, "No anchor object found in objects list."

        self._all_objects = objects
        self._anchor_objects: set[Object] = set(anchor_objects)
        self._optimizable_objects = [obj for obj in objects if obj not in self._anchor_objects]
        self._obj_to_idx: dict[Object | ObjectReference, int] = {obj: i for i, obj in enumerate(objects)}
        self._anchor_indices: set[int] = {self._obj_to_idx[obj] for obj in self._anchor_objects}
        self._optimizable_indices = [i for i in range(len(objects)) if i not in self._anchor_indices]

        self._num_envs = 1
        if initial_positions_per_env is not None:
            self._num_envs = len(initial_positions_per_env)
            # Batched: (num_envs, num_optimizable, 3)
            positions_per_env = []
            for d in initial_positions_per_env:
                pos_list = [torch.tensor(d[obj], dtype=torch.float32) for obj in objects]
                positions_per_env.append(pos_list)
            # Stack per-object across envs: list of (num_envs, 3), then stack optimizable only
            opt_tensors = []
            for opt_idx in self._optimizable_indices:
                opt_tensors.append(torch.stack([positions_per_env[e][opt_idx] for e in range(self._num_envs)]))
            self._optimizable_positions = torch.stack(opt_tensors, dim=1)  # (num_envs, num_opt, 3)
            self._optimizable_positions.requires_grad = True
            # Anchors: same for all envs from first env
            self._anchor_positions = {
                idx: torch.tensor(initial_positions_per_env[0][objects[idx]], dtype=torch.float32)
                for idx in self._anchor_indices
            }
        else:
            assert initial_positions is not None, "Provide initial_positions or initial_positions_per_env"
            positions = [torch.tensor(initial_positions[obj], dtype=torch.float32) for obj in objects]
            self._anchor_positions = {idx: positions[idx].clone() for idx in self._anchor_indices}
            if self._optimizable_indices:
                self._optimizable_positions = torch.stack([positions[i] for i in self._optimizable_indices])
                self._optimizable_positions.requires_grad = True
            else:
                self._optimizable_positions = None

    @property
    def num_envs(self) -> int:
        """Number of environments (1 for single-env, >1 for batched)."""
        return self._num_envs

    @property
    def optimizable_positions(self) -> torch.Tensor | None:
        """Tensor of optimizable positions (shape: [N-num_anchors, 3]), or None if all objects are anchors.

        This is the tensor that should be passed to the optimizer.
        In batched mode shape is [num_envs, N_opt, 3].
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

    def get_position(self, obj: Object | ObjectReference, env_index: int | None = None) -> torch.Tensor:
        """Get current position for an object.

        Args:
            obj: The object to get position for.
            env_index: In batched mode, env index (0..num_envs-1) or None for all envs.

        Returns:
            Position tensor (x, y, z). In batched mode with env_index None, shape (num_envs, 3).

        Raises:
            KeyError: If object is not tracked by this state.
            RuntimeError: If requesting position for optimizable object when none exist.
        """
        idx = self._obj_to_idx[obj]
        if idx in self._anchor_indices:
            anchor_pos = self._anchor_positions[idx]
            if self._num_envs > 1 and env_index is None:
                return anchor_pos.unsqueeze(0).expand(self._num_envs, 3)
            return anchor_pos
        if self._optimizable_positions is None:
            raise RuntimeError(f"No optimizable positions available for object '{obj.name}'")
        opt_idx = self._optimizable_indices.index(idx)
        if self._num_envs > 1:
            if env_index is None:
                return self._optimizable_positions[:, opt_idx, :]
            return self._optimizable_positions[env_index, opt_idx, :]
        return self._optimizable_positions[opt_idx]

    def get_all_positions_snapshot(self) -> list[tuple[float, float, float]]:
        """Get detached copy of all positions for history tracking.

        Returns:
            List of (x, y, z) positions for each object (in original order). In batched mode uses env 0.
        """
        if self._num_envs > 1:
            return [tuple(self.get_position(obj, env_index=0).detach().tolist()) for obj in self._all_objects]
        return [tuple(self.get_position(obj).detach().tolist()) for obj in self._all_objects]

    def get_final_positions_dict(self) -> dict[Object | ObjectReference, tuple[float, float, float]]:
        """Get final positions as a dictionary mapping objects to positions.

        Returns:
            Dictionary with object instances as keys and (x, y, z) tuples as values. In batched mode returns env 0.
        """
        result: dict[Object | ObjectReference, tuple[float, float, float]] = {}
        for obj in self._all_objects:
            pos_t = self.get_position(obj, env_index=0 if self._num_envs > 1 else None)
            pos = pos_t.detach().tolist()
            result[obj] = (pos[0], pos[1], pos[2])
        return result

    def get_final_positions_per_env(self) -> list[dict[Object | ObjectReference, tuple[float, float, float]]]:
        """Return one position dict per env. Single-env returns a list of one dict."""
        if self._num_envs <= 1:
            return [self.get_final_positions_dict()]
        out = []
        for e in range(self._num_envs):
            d: dict[Object | ObjectReference, tuple[float, float, float]] = {}
            for obj in self._all_objects:
                pos = self.get_position(obj, env_index=e).detach().tolist()
                d[obj] = (pos[0], pos[1], pos[2])
            out.append(d)
        return out
