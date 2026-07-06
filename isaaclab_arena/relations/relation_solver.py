# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import torch
from typing import TYPE_CHECKING, cast

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.no_overlap_aabb import compute_no_overlap_loss_aabb
from isaaclab_arena.relations.no_overlap_mesh import compute_no_overlap_loss_mesh, prepare_mesh_collision_cache
from isaaclab_arena.relations.relation_loss_strategies import (
    NoCollisionLossStrategy,
    RelationLossStrategy,
    UnaryRelationLossStrategy,
)
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On, Relation, RelationBase, UnaryRelation
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.mesh_pair_cache import MeshPairCache
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class RelationSolver:
    """Differentiable solver for 3D spatial relations of IsaacLab Arena Objects.

    Uses the Strategy pattern for loss computation: each Relation type has a
    corresponding RelationLossStrategy that handles the actual loss calculation.
    """

    POSITION_HISTORY_SAVE_INTERVAL = 10
    """Save position snapshot every N iterations (when save_position_history is enabled)."""

    def __init__(
        self,
        params: RelationSolverParams | None = None,
    ):
        """
        Args:
            params: Solver configuration parameters. If None, uses defaults.
        """
        self.params = params or RelationSolverParams()
        # High slope (vs 10-100 for relation strategies) so overlap avoidance dominates.
        self._no_collision_strategy = NoCollisionLossStrategy(slope=10000.0)
        self._last_loss_history: list[float] = []
        self._last_position_history: list = []
        self._last_loss_per_env: torch.Tensor | None = None
        self._last_no_overlap_pair_count: int = 0
        self._mesh_orientations: list[dict[ObjectBase, float]] | None = None
        self._warned_no_mesh: set[str] = set()
        self._mesh_manager: WarpMeshAndSphereCache | None = None
        self._mesh_cache: MeshPairCache | None = None

    def _get_strategy(self, relation: RelationBase) -> RelationLossStrategy | UnaryRelationLossStrategy:
        """Look up the loss strategy for a relation type.

        Args:
            relation: The relation to find a strategy for.
        """
        strategy = self.params.strategies.get(type(relation))
        if strategy is None:
            raise ValueError(
                f"No loss strategy registered for {type(relation).__name__}. "
                f"Available strategies: {list(self.params.strategies.keys())}"
            )
        return strategy

    def _compute_total_loss(
        self,
        state: RelationSolverState,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute total loss from all relations using registered strategies.

        Args:
            state: Current optimization state with object positions and
                optional per-env bounding boxes (accessed via state.get_bbox).
            debug: If True, print detailed loss breakdown.

        Returns:
            Scalar loss tensor (mean over environments).
        """
        batch_size = state.batch_size
        device = state.device
        total_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)

        for obj in state.optimizable_objects:
            for relation in obj.get_spatial_relations():
                child_pos = state.get_position(obj)
                strategy = self._get_strategy(relation)
                child_bbox = state.get_bbox(obj)

                if isinstance(relation, UnaryRelation):
                    unary_strategy = cast(UnaryRelationLossStrategy, strategy)
                    loss = unary_strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=child_bbox,
                    )
                    if debug:
                        _print_unary_relation_debug(obj, relation, child_pos[0], loss.mean())
                # Binary relation (On, NextTo, etc.)
                elif isinstance(relation, Relation):
                    relation_strategy = cast(RelationLossStrategy, strategy)
                    parent = relation.parent
                    if parent in state.anchor_objects:
                        parent_world_bbox = parent.get_world_bounding_box().to(device)
                    else:
                        parent_pos = state.get_position(parent)
                        parent_bbox = state.get_bbox(parent)
                        parent_world_bbox = parent_bbox.translated(parent_pos)
                    loss = relation_strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=child_bbox,
                        parent_world_bbox=parent_world_bbox,
                    )
                    if debug:
                        parent_pos = state.get_position(parent)
                        _print_relation_debug(obj, relation, child_pos[0], parent_pos[0], loss.mean())
                else:
                    raise ValueError(f"Unknown relation type: {type(relation).__name__}")

                total_loss = total_loss + loss

        # Add built-in no-overlap loss between all object pairs
        total_loss = total_loss + self._compute_no_overlap_loss(state, debug)

        self._last_loss_per_env = total_loss.detach().clone()
        return total_loss.mean()

    def _compute_no_overlap_loss(
        self,
        state: RelationSolverState,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute pairwise no-overlap loss, skipping On-linked pairs."""
        if self.params.collision_mode == CollisionMode.MESH:
            mesh_loss = compute_no_overlap_loss_mesh(
                state,
                self._mesh_cache,
                self._mesh_manager,
                self._mesh_orientations,
                self.params.clearance_m,
                self._no_collision_strategy.slope,
                debug,
            )
            aabb_loss, n = compute_no_overlap_loss_aabb(
                state,
                self._no_collision_strategy,
                self.params.clearance_m,
                self._mesh_manager,
                skip_mesh_pairs=True,
                debug=debug,
            )
            self._last_no_overlap_pair_count = n
            return mesh_loss + aabb_loss
        else:
            loss, n = compute_no_overlap_loss_aabb(
                state,
                self._no_collision_strategy,
                self.params.clearance_m,
                self._mesh_manager,
                debug=debug,
            )
            self._last_no_overlap_pair_count = n
            return loss

    def solve(
        self,
        objects: list[ObjectBase],
        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox] | None = None,
        orientations: list[dict[ObjectBase, float]] | None = None,
    ) -> list[dict[ObjectBase, tuple[float, float, float]]]:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of ObjectBase instances. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            initial_positions: List of dicts (one per env). Use a single-element list
                for single-env placement.
            env_bboxes: Optional per-env bounding boxes keyed by object.
                ObjectPlacer always supplies these, with each
                AxisAlignedBoundingBox shaped (batch, 3). Direct solver calls
                may omit them to use each object's default get_bounding_box().
            orientations: Optional per-env yaw angles (radians about Z) per object.
                Used in MESH mode to rotate sphere centers before collision queries.

        Returns:
            List of dicts (one per env) mapping objects to their solved (x, y, z) positions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = RelationSolverState(objects, initial_positions, device=device, env_bboxes=env_bboxes)

        if self.params.verbose:
            anchor_names = [obj.name for obj in state.anchor_objects]
            optimizable_names = [obj.name for obj in state.optimizable_objects]
            print("=== RelationSolver ===")
            print(f"Anchors (fixed): {anchor_names}")
            print(f"Optimizable: {optimizable_names}")

        # Early return if nothing to optimize (all objects are anchors)
        if len(state.optimizable_objects) == 0:
            if self.params.verbose:
                print("No optimizable objects, skipping solver.")
            self._last_loss_history = [0.0]
            self._last_loss_per_env = torch.zeros(state.batch_size)
            self._last_position_history = [state.get_all_positions_snapshot()]
            return state.get_final_positions()

        if self.params.profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        solve_start = time.perf_counter()

        # Precompute mesh collision cache (once per solve, before opt loop)
        if self.params.collision_mode == CollisionMode.MESH:
            non_anchor_objects = state.optimizable_objects
            anchor_objects = list(state.anchor_objects)
            on_pairs: set[tuple[int, int]] = set()
            for obj in [*non_anchor_objects, *anchor_objects]:
                for rel in obj.get_relations():
                    if isinstance(rel, On):
                        on_pairs.add((id(obj), id(rel.parent)))
                        on_pairs.add((id(rel.parent), id(obj)))
            self._mesh_orientations = orientations
            device_str = str(state.device)
            if self._mesh_manager is None or self._mesh_manager.device != device_str:
                self._mesh_manager = WarpMeshAndSphereCache(num_spheres=self.params.num_spheres, device=device_str)
            self._mesh_cache = prepare_mesh_collision_cache(state, self._mesh_manager, on_pairs, self._warned_no_mesh)
            self._mesh_manager.reset_sentinel_warning()

        # Setup optimizer (only for optimizable positions)
        optimizer = torch.optim.Adam([state.optimizable_positions], lr=self.params.lr)

        # Compute initial loss so _last_loss_per_env is always populated, even when max_iters=0.
        with torch.no_grad():
            self._compute_total_loss(state)

        # Optimization loop
        loss_history = []
        position_history = []  # Track positions for visualization

        for iter in range(self.params.max_iters):
            optimizer.zero_grad()

            if self.params.save_position_history and iter % self.POSITION_HISTORY_SAVE_INTERVAL == 0:
                position_history.append(state.get_all_positions_snapshot())

            loss = self._compute_total_loss(state)
            loss_history.append(loss.item())

            # Constant-zero loss has no grad_fn — skip backward when overlap filter culls all pairs.
            if loss.grad_fn is not None:
                loss.backward()
                optimizer.step()

            if self.params.verbose and iter % 100 == 0:
                print(f"Iter {iter}: loss = {loss.item():.6f}")

            # Check convergence
            if loss.item() < self.params.convergence_threshold:
                if self.params.verbose:
                    print(f"Converged at iteration {iter}")
                break

        if self.params.profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        solve_elapsed_ms = (time.perf_counter() - solve_start) * 1e3

        if self.params.save_position_history:
            position_history.append(state.get_all_positions_snapshot())

        if self.params.verbose and loss_history:
            print(f"\nFinal loss: {loss_history[-1]:.6f}")
            print(f"Total iterations: {len(loss_history)}")

        if self.params.profile and loss_history:
            iters_run = len(loss_history)
            print(
                f"[RelationSolver] solve: {solve_elapsed_ms:.1f} ms"
                f" | batch={state.batch_size}"
                f" | objects={len(state.optimizable_objects)} optimizable + {len(state.anchor_objects)} anchors"
                f" | no-overlap pairs={self._last_no_overlap_pair_count}"
                f" | iters={iters_run} ({solve_elapsed_ms / iters_run:.2f} ms/iter)"
            )

        self._last_loss_history = loss_history
        self._last_position_history = position_history

        return state.get_final_positions()

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent solve() call."""
        return self._last_loss_history

    @property
    def last_loss_per_env(self) -> torch.Tensor | None:
        """Per-candidate loss tensor of shape (batch_size,) from the last solve() call."""
        return self._last_loss_per_env

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent solve() call."""
        return self._last_position_history

    def debug_losses(self, objects: list[ObjectBase]) -> None:
        """Print detailed loss breakdown for all relations using final positions.

        Call this after solve() to inspect why objects may not be correctly positioned.

        Args:
            objects: The same list of objects passed to solve().
        """
        print("\n" + "=" * 60)
        print("DEBUG: Final Loss Breakdown")
        print("=" * 60)

        final_positions_list = self.last_position_history[-1] if self.last_position_history else None
        if final_positions_list is None:
            print("No position history available. Run solve() first.")
            return

        final_positions = {obj: (pos[0], pos[1], pos[2]) for obj, pos in zip(objects, final_positions_list)}

        state = RelationSolverState(objects, [final_positions])
        self._compute_total_loss(state, debug=True)
        print("\n" + "=" * 60)


def _print_relation_debug(
    obj: ObjectBase,
    relation: Relation,
    child_pos: torch.Tensor,
    parent_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a single binary relation."""
    child_bbox = obj.get_bounding_box()
    parent_world_bbox = relation.parent.get_world_bounding_box()

    print(f"\n=== {obj.name} -> {type(relation).__name__}({relation.parent.name}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(
        f"  Child bbox: min={child_bbox.min_point[0].tolist()}, max={child_bbox.max_point[0].tolist()},"
        f" size={child_bbox.size[0].tolist()}"
    )
    print(f"  Parent pos: ({parent_pos[0].item():.4f}, {parent_pos[1].item():.4f}, {parent_pos[2].item():.4f})")
    print(
        f"  Parent world bbox: min={parent_world_bbox.min_point[0].tolist()},"
        f" max={parent_world_bbox.max_point[0].tolist()}, size={parent_world_bbox.size[0].tolist()}"
    )

    # Child world extents
    child_x_range = (
        child_pos[0].item() + child_bbox.min_point[0, 0].item(),
        child_pos[0].item() + child_bbox.max_point[0, 0].item(),
    )
    child_y_range = (
        child_pos[1].item() + child_bbox.min_point[0, 1].item(),
        child_pos[1].item() + child_bbox.max_point[0, 1].item(),
    )

    print(f"  Child world X: [{child_x_range[0]:.4f}, {child_x_range[1]:.4f}]")
    print(f"  Child world Y: [{child_y_range[0]:.4f}, {child_y_range[1]:.4f}]")
    print(
        f"  Parent world X: [{parent_world_bbox.min_point[0, 0].item():.4f},"
        f" {parent_world_bbox.max_point[0, 0].item():.4f}]"
    )
    print(
        f"  Parent world Y: [{parent_world_bbox.min_point[0, 1].item():.4f},"
        f" {parent_world_bbox.max_point[0, 1].item():.4f}]"
    )
    print(f"  Loss: {loss.item():.6f}")


def _print_unary_relation_debug(
    obj: ObjectBase,
    relation: RelationBase,
    child_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a unary relation (no parent)."""
    child_bbox = obj.get_bounding_box()

    params = {k: v for k, v in relation.__dict__.items() if v is not None and k != "relation_loss_weight"}
    param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
    print(f"\n=== {obj.name} -> {type(relation).__name__}({param_str}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(
        f"  Child bbox: min={child_bbox.min_point[0].tolist()}, max={child_bbox.max_point[0].tolist()},"
        f" size={child_bbox.size[0].tolist()}"
    )
    print(f"  Loss: {loss.item():.6f}")
