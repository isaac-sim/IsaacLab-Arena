# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_candidate_pool import PlacementCandidatePool
from isaaclab_arena.relations.placement_events import get_rotation_xyzw, solve_and_place_objects
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.scene.scene import Scene


@dataclass
class PlacementProblem:
    """Placement problem assembled from the Arena Env components."""

    objects: list[ObjectBase]
    """Objects with spatial predicates that should be relation-solved."""

    object_placer_params: ObjectPlacerParams
    """Parameters for the object-only placement implementation."""

    num_envs: int
    """Number of environments that the solver must prepare."""

    @property
    def pool_size(self) -> int:
        """Number of object-layout candidates to pre-solve."""
        return self.num_envs * self.object_placer_params.min_unique_layouts_per_env


@dataclass
class ArenaRelationSolveResult:
    """Result returned by :class:`ArenaRelationSolver.solve`."""

    objects: list[ObjectBase]
    """Objects considered by relation placement."""

    object_placer_params: ObjectPlacerParams | None = None
    """Object placer params used for solving, or None when no solve was needed."""

    placement_candidate_pool: PlacementCandidatePool | None = None
    """Placement candidate pool used by reset-time placement, when one exists."""

    placement_event_cfg: EventTermCfg | None = None
    """Reset event config to attach to the Arena environment, if needed."""


class ObjectRelationSolver:
    """Solve and validate object-object spatial relations using the v0.2 object placer."""

    def solve(self, problem: PlacementProblem) -> PlacementCandidatePool:
        """Solve object-only relation placement and return a layout pool."""
        placement_candidate_pool = PlacementCandidatePool(
            objects=problem.objects,
            placer_params=problem.object_placer_params,
            pool_size=problem.pool_size,
            candidate_validator=self.validate_layout,
        )
        self.validate_placement_candidate_pool(placement_candidate_pool)
        return placement_candidate_pool

    def validate_placement_candidate_pool(self, placement_candidate_pool: PlacementCandidatePool) -> None:
        """Hook for subclasses that need to validate the solved placement candidate pool."""

    def validate_layout(self, layout: PlacementResult) -> None:
        """Hook for subclasses that need to validate one sampled layout."""


class ArenaRelationSolver:
    """Arena-facing relation placement orchestration.

    The solver owns the API boundary for collecting relation-bearing objects,
    building a placement problem, selecting layouts, and applying the selected
    result to Arena reset/static placement state.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        scene: Scene | None = None,
        objects: list[ObjectBase] | None = None,
        placement_seed: int | None = None,
        resolve_on_reset: bool | None = None,
        object_solver: ObjectRelationSolver | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.scene = scene
        self.objects = list(objects) if objects is not None else None
        self.object_solver = object_solver or ObjectRelationSolver()
        self.object_placer_params = ObjectPlacerParams(
            placement_seed=placement_seed,
            apply_positions_to_objects=False,
            solver_params=RelationSolverParams(save_position_history=False, verbose=False),
        )
        if resolve_on_reset is not None:
            self.object_placer_params.resolve_on_reset = resolve_on_reset

    def solve(self) -> ArenaRelationSolveResult:
        """Solve Arena relation placement and apply the selected result."""
        objects = self._collect_related_objects()
        if not objects:
            print("No objects with relations found in scene. Skipping relation solving.")
            return ArenaRelationSolveResult(objects=[])

        problem = self._build_problem(objects)

        # TODO(xinjieyao, 2026-05-22): Add joint object/embodiment placement once task-dependent
        # reachability constraints are available. For now this always uses the object-only placer.
        placement_candidate_pool = self.object_solver.solve(problem)
        return self._apply_result(problem, placement_candidate_pool)

    def _collect_related_objects(self) -> list[ObjectBase]:
        """Collect objects with spatial predicates from explicit input or the scene."""
        if self.objects is not None:
            return list(self.objects)
        if self.scene is None:
            return []
        return self.scene.get_objects_with_relations()

    def _build_problem(self, objects: list[ObjectBase]) -> PlacementProblem:
        """Build placement inputs from scene objects and solver params."""
        return PlacementProblem(
            objects=objects,
            object_placer_params=deepcopy(self.object_placer_params),
            num_envs=self.num_envs,
        )

    def _apply_result(
        self,
        problem: PlacementProblem,
        placement_candidate_pool: PlacementCandidatePool,
    ) -> ArenaRelationSolveResult:
        """Apply selected candidates to object spawn state and build reset event config."""
        anchor_objects_set = set(get_anchor_objects(problem.objects))
        # Prevent external pose-reset events from conflicting with relation-solved objects.
        self._validate_no_conflicting_pose_reset_events(problem.objects, anchor_objects_set)

        placement_event_cfg = self._apply_object_placement_result(
            problem,
            placement_candidate_pool,
            anchor_objects_set,
        )

        return ArenaRelationSolveResult(
            objects=problem.objects,
            object_placer_params=problem.object_placer_params,
            placement_candidate_pool=placement_candidate_pool,
            placement_event_cfg=placement_event_cfg,
        )

    def _apply_object_placement_result(
        self,
        problem: PlacementProblem,
        placement_candidate_pool: PlacementCandidatePool,
        anchor_objects_set: set[ObjectBase],
    ) -> EventTermCfg | None:
        """Apply object placement poses and return a reset event config when needed."""
        # Anchor objects do not move, so no need to apply reset event.
        if anchor_objects_set == set(problem.objects):
            return None

        # Apply reset event to spawn new poses for each environment.
        if problem.object_placer_params.resolve_on_reset:
            return self._apply_dynamic_spawn_pose(
                problem.objects,
                placement_candidate_pool,
                anchor_objects_set,
            )

        # Static initial poses for each environment.
        self._apply_static_initial_poses(
            problem.objects,
            placement_candidate_pool,
            anchor_objects_set,
        )
        return None

    def _apply_dynamic_spawn_pose(
        self,
        objects: list[ObjectBase],
        placement_candidate_pool: PlacementCandidatePool,
        anchor_objects_set: set[ObjectBase],
    ) -> EventTermCfg:
        """Set initial spawn pose from one layout and return the reset placement event."""
        from isaaclab.managers import EventTermCfg

        layout = placement_candidate_pool.sample_with_replacement(1)[0]
        for obj in objects:
            if obj in anchor_objects_set:
                continue
            pos = layout.positions.get(obj)
            if pos is None:
                raise RuntimeError(f"Placement candidate is missing a solved position for '{obj.name}'.")
            object_cfg = getattr(obj, "object_cfg", None)
            if object_cfg is None:
                raise RuntimeError(f"Object '{obj.name}' must have object_cfg initialized before placement.")
            object_cfg.init_state.pos = pos
            object_cfg.init_state.rot = get_rotation_xyzw(obj)

        return EventTermCfg(
            func=solve_and_place_objects,
            mode="reset",
            params={
                "objects": objects,
                "placement_candidate_pool": placement_candidate_pool,
            },
        )

    def _apply_static_initial_poses(
        self,
        objects: list[ObjectBase],
        placement_candidate_pool: PlacementCandidatePool,
        anchor_objects_set: set[ObjectBase],
    ) -> None:
        """Apply fixed per-environment poses for ``resolve_on_reset=False``."""
        layouts = placement_candidate_pool.sample_with_replacement(self.num_envs)
        for obj in objects:
            if obj in anchor_objects_set:
                continue
            rotation_xyzw = get_rotation_xyzw(obj)
            poses = []
            for env_idx in range(self.num_envs):
                pos = layouts[env_idx].positions.get(obj)
                if pos is None:
                    raise RuntimeError(f"Placement candidate is missing a solved position for '{obj.name}'.")
                poses.append(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
            obj.set_initial_pose(PosePerEnv(poses=poses))

    def _validate_no_conflicting_pose_reset_events(
        self,
        objects: list[ObjectBase],
        anchor_objects_set: set[ObjectBase],
    ) -> None:
        """Reject conflicting explicit pose-reset events on relation-solved objects."""
        for obj in objects:
            if obj not in anchor_objects_set and getattr(obj, "event_cfg", None) is not None:
                raise RuntimeError(
                    f"Non-anchor object '{obj.name}' has an explicit pose-reset event. "
                    "Relational solving should not be combined with explicit setting of "
                    "poses on non-anchor objects."
                )
