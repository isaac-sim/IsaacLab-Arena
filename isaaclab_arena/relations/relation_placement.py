# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import get_rotation_xyzw, solve_and_place_objects
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


@dataclass
class ObjectRelationSolveResult:
    """Result from solving object-object spatial relations."""

    object_placement_pool: PooledObjectPlacer
    """Pool of object layouts produced by ObjectRelationSolver."""

    object_placer_params: ObjectPlacerParams
    """Object placement parameters used to build the layout pool."""


@dataclass
class RobotRelationSolveResult:
    """Result from solving robot/task relation constraints."""

    embodiment: EmbodimentBase | None
    """Embodiment used for robot/task-aware checks, if any."""


@dataclass
class RelationSolveResult:
    """Combined relation solve results for scene objects and embodiment."""

    object_result: ObjectRelationSolveResult
    """Object-object relation solve result."""

    robot_result: RobotRelationSolveResult
    """Robot/task relation solve result."""


@dataclass
class PlacementSolution:
    """Prepared placement artifacts and informational scene snapshot."""

    objects: list[ObjectBase]
    embodiment: EmbodimentBase | None
    placement_event_cfg: EventTermCfg | None = None


class ObjectRelationSolver:
    """Solve and validate object-object spatial relations.

    Args:
        num_envs: Number of environment instances to prepare placements for.
        placement_seed: Optional seed forwarded to the object placer.
        resolve_on_reset: Whether to sample a new layout on each reset. ``None``
            keeps the default from :class:`ObjectPlacerParams`.
    """

    def __init__(self, num_envs: int, *, placement_seed: int | None = None, resolve_on_reset: bool | None = None):
        self.num_envs = num_envs
        self.placement_seed = placement_seed
        self.resolve_on_reset = resolve_on_reset

    def solve(self, objects: list[ObjectBase]) -> ObjectRelationSolveResult:
        """Solve object relations and return the object layout pool plus params used."""
        object_placer_params = self._create_object_placer_params()
        object_placement_pool = self._create_object_placement_pool(objects, object_placer_params)
        self.validate_object_placement_pool(object_placement_pool)
        return ObjectRelationSolveResult(
            object_placement_pool=object_placement_pool,
            object_placer_params=object_placer_params,
        )

    def _create_object_placer_params(self) -> ObjectPlacerParams:
        placer_params = ObjectPlacerParams(
            placement_seed=self.placement_seed,
            apply_positions_to_objects=False,
            solver_params=RelationSolverParams(save_position_history=False, verbose=False),
        )
        if self.resolve_on_reset is not None:
            placer_params.resolve_on_reset = self.resolve_on_reset
        return placer_params

    def _create_object_placement_pool(
        self,
        objects: list[ObjectBase],
        placer_params: ObjectPlacerParams,
    ) -> PooledObjectPlacer:
        pool_size = self.num_envs * placer_params.min_unique_layouts_per_env
        return PooledObjectPlacer(
            objects=objects,
            placer_params=placer_params,
            pool_size=pool_size,
        )

    def validate_object_placement_pool(self, object_placement_pool: PooledObjectPlacer) -> None:
        """Hook for subclasses that need to validate or wrap the solved object layout pool."""
        return None

    def validate_layout(self, layout: PlacementResult) -> None:
        """Validate one object-relation placement result."""
        if not self.check_objects_valid(layout):
            raise RuntimeError("Relation placement failed object validation.")

    def check_objects_valid(self, layout: PlacementResult) -> bool:
        return layout.success


class RobotRelationSolver:
    """Embodiment-aware robot/task relation solving extension point.

    The current implementation preserves existing object-only behavior. Future
    robot solvers should subclass this and implement ``solve`` and
    ``validate_layout`` using the embodiment and solved object layouts.
    """

    def __init__(self, embodiment: EmbodimentBase | None = None) -> None:
        self.embodiment = embodiment
        self._warned_base_validation_skip = False

    def solve(
        self,
        objects: list[ObjectBase],
        object_result: ObjectRelationSolveResult,
    ) -> RobotRelationSolveResult:
        """Prepare embodiment-specific solve state for robot-aware validation."""
        # TODO: pass task-specific reachability context here once SceneGraph YAML
        # carries grasp, dropoff, handle, and other task-dependent target poses.
        return RobotRelationSolveResult(embodiment=self.embodiment)

    def validate_layout(
        self,
        layout: PlacementResult,
        objects: list[ObjectBase],
        robot_result: RobotRelationSolveResult,
    ) -> None:
        """Validate robot constraints for one solved layout."""
        if type(self) is not RobotRelationSolver and robot_result.embodiment is not None:
            # Subclasses signal intent to validate robot constraints; require
            # them to implement the hook instead of inheriting a silent pass.
            raise NotImplementedError("Robot layout validation is unimplemented.")
        if robot_result.embodiment is not None and not self._warned_base_validation_skip:
            print("Robot relation validation is not implemented; skipping IK validation.")
            self._warned_base_validation_skip = True
        return None

    def check_IK_reachable(self, objects: list[ObjectBase], embodiment: EmbodimentBase) -> bool:
        """Return whether scene objects are IK-reachable for the embodiment.

        TODO: use the solved object poses from ``objects`` and the embodiment's
        robot API to run the actual IK/reachability query.
        """
        raise NotImplementedError("IK reachability check is unimplemented.")


class ValidatedPlacementPool:
    """Pooled placement adapter that validates every sampled layout.

    The reset event samples layouts after ``ArenaRelationSolver.prepare()``
    returns, so the validation context (objects, solvers, and robot result) is
    captured here instead of depending on later mutable solver state.
    ``EventTermCfg`` deep-copies params during construction; ``__deepcopy__``
    snapshots solver state while preserving object identity for validation.
    """

    def __init__(
        self,
        placement_pool: PooledObjectPlacer,
        objects: list[ObjectBase],
        object_solver: ObjectRelationSolver,
        robot_solver: RobotRelationSolver,
        robot_result: RobotRelationSolveResult,
    ) -> None:
        self._placement_pool = placement_pool
        self._objects = list(objects)
        self._object_solver = object_solver
        self._robot_solver = robot_solver
        self._robot_result = robot_result

    def __deepcopy__(self, memo):
        copied_pool = deepcopy(self._placement_pool, memo)
        copied_object_solver = deepcopy(self._object_solver, memo)
        copied_robot_solver = deepcopy(self._robot_solver, memo)
        copied_robot_result = deepcopy(self._robot_result, memo)
        return type(self)(
            placement_pool=copied_pool,
            objects=self._objects,
            object_solver=copied_object_solver,
            robot_solver=copied_robot_solver,
            robot_result=copied_robot_result,
        )

    def sample_without_replacement(self, count: int) -> list[PlacementResult]:
        layouts = self._placement_pool.sample_without_replacement(count)
        self._validate_layouts(layouts)
        return layouts

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        layouts = self._placement_pool.sample_with_replacement(count)
        self._validate_layouts(layouts)
        return layouts

    @property
    def remaining(self) -> int:
        return self._placement_pool.remaining

    @property
    def pool_size(self) -> int:
        return self._placement_pool.pool_size

    def _validate_layouts(self, layouts: list[PlacementResult]) -> None:
        for layout in layouts:
            self._object_solver.validate_layout(layout)
            self._robot_solver.validate_layout(layout, self._objects, self._robot_result)


class ArenaRelationSolver:
    """Arena-facing relation placement orchestration.

    This class owns the Arena API boundary for scene objects, optional
    embodiment context, reset events, and high-level relation placement calls.
    Object and robot solvers provide their own solving and validation
    independently.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        placement_seed: int | None = None,
        resolve_on_reset: bool | None = None,
        embodiment: EmbodimentBase | None = None,
        object_solver: ObjectRelationSolver | None = None,
        robot_solver: RobotRelationSolver | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.objects: list[ObjectBase] = []
        self._robot_solver_supplied = robot_solver is not None
        if object_solver is not None:
            if object_solver.num_envs != num_envs:
                raise ValueError("object_solver.num_envs must match ArenaRelationSolver.num_envs.")
            if placement_seed is not None or resolve_on_reset is not None:
                raise ValueError(
                    "placement_seed and resolve_on_reset are owned by object_solver when object_solver is provided."
                )
        robot_solver_embodiment = robot_solver.embodiment if robot_solver is not None else None
        if embodiment is not None and robot_solver_embodiment is not None and embodiment is not robot_solver_embodiment:
            raise ValueError("embodiment must match robot_solver.embodiment when both are provided.")
        self.embodiment = embodiment if embodiment is not None else robot_solver_embodiment
        self.object_solver = object_solver or ObjectRelationSolver(
            num_envs=num_envs,
            placement_seed=placement_seed,
            resolve_on_reset=resolve_on_reset,
        )
        self.robot_solver = robot_solver or RobotRelationSolver(embodiment=self.embodiment)
        self._reconcile_robot_solver_embodiment()
        self.robot_result: RobotRelationSolveResult | None = None

    def prepare(
        self,
        objects: list[ObjectBase],
        embodiment: EmbodimentBase | None = None,
    ) -> PlacementSolution | None:
        """Solve scene-object and embodiment relations, then prepare Arena placement."""
        self.robot_result = None
        if not objects:
            print("No objects with relations found in scene. Skipping relation solving.")
            return None
        self.objects = list(objects)
        self._set_embodiment(embodiment)

        relation_result = self.solve_relations()
        anchor_objects_set = set(get_anchor_objects(self.objects))
        _validate_no_explicit_non_anchor_pose_events(self.objects, anchor_objects_set)

        placement_event_cfg = self._apply_relation_result(relation_result, anchor_objects_set)

        return PlacementSolution(
            objects=list(self.objects),
            embodiment=self.embodiment,
            placement_event_cfg=placement_event_cfg,
        )

    def _set_embodiment(self, embodiment: EmbodimentBase | None) -> None:
        """Update placement and robot-solver embodiment context for this prepare call."""
        if embodiment is not None:
            if (
                self._robot_solver_supplied
                and self.robot_solver.embodiment is not None
                and self.robot_solver.embodiment is not embodiment
            ):
                raise ValueError("embodiment must match robot_solver.embodiment when both are provided.")
            self.embodiment = embodiment
        self._reconcile_robot_solver_embodiment()

    def _reconcile_robot_solver_embodiment(self) -> None:
        if not self._robot_solver_supplied and self.robot_solver.embodiment is None:
            self.robot_solver.embodiment = self.embodiment
            return
        if self.robot_solver.embodiment is not self.embodiment:
            raise ValueError("embodiment must match robot_solver.embodiment when both are provided.")

    def solve_relations(self) -> RelationSolveResult:
        object_result = self.object_solver.solve(self.objects)
        robot_result = self.robot_solver.solve(self.objects, object_result)
        self.robot_result = robot_result
        return RelationSolveResult(object_result=object_result, robot_result=robot_result)

    def _apply_relation_result(
        self,
        relation_result: RelationSolveResult,
        anchor_objects_set: set[ObjectBase],
    ) -> EventTermCfg | None:
        """Apply solved relation state to Arena objects and reset events."""
        object_result = relation_result.object_result
        placement_pool = ValidatedPlacementPool(
            placement_pool=object_result.object_placement_pool,
            objects=self.objects,
            object_solver=self.object_solver,
            robot_solver=self.robot_solver,
            robot_result=relation_result.robot_result,
        )
        if object_result.object_placer_params.resolve_on_reset:
            # Dynamic reset keeps the pool in the reset event so each reset can
            # draw a newly validated layout.
            self._apply_dynamic_spawn_pose(placement_pool, anchor_objects_set)
            return EventTermCfg(
                func=solve_and_place_objects,
                mode="reset",
                params={
                    "objects": list(self.objects),
                    "placement_pool": placement_pool,
                },
            )

        self._apply_static_initial_poses(placement_pool, anchor_objects_set)
        return None

    def validate_layout(self, layout: PlacementResult) -> None:
        if self.robot_result is None:
            raise RuntimeError("Robot relation solve result must be available before validation.")
        self.object_solver.validate_layout(layout)
        self.robot_solver.validate_layout(layout, self.objects, self.robot_result)

    def _apply_dynamic_spawn_pose(
        self,
        object_placement_pool: ValidatedPlacementPool,
        anchor_objects_set: set[ObjectBase],
    ) -> None:
        """Set ``object_cfg.init_state`` from a pool layout so objects spawn at valid positions.

        A single sampled layout keeps initial spawn valid before the reset event
        takes over. Anchors stay fixed at their user-provided initial poses.
        """
        layout = object_placement_pool.sample_with_replacement(1)[0]
        for obj in self.objects:
            if obj in anchor_objects_set:
                continue
            pos = layout.positions.get(obj)
            if pos is None:
                raise RuntimeError(f"Placement pool layout is missing a solved position for '{obj.name}'.")
            rotation_xyzw = get_rotation_xyzw(obj)
            object_cfg = obj.object_cfg
            if object_cfg is None:
                raise RuntimeError(f"Object '{obj.name}' must have object_cfg initialized before placement.")
            object_cfg.init_state.pos = pos
            object_cfg.init_state.rot = rotation_xyzw

    def _apply_static_initial_poses(
        self,
        object_placement_pool: ValidatedPlacementPool,
        anchor_objects_set: set[ObjectBase],
    ) -> None:
        """Apply fixed per-environment poses for ``resolve_on_reset=False``."""
        layouts = object_placement_pool.sample_with_replacement(self.num_envs)
        for obj in self.objects:
            if obj in anchor_objects_set:
                continue
            rotation_xyzw = get_rotation_xyzw(obj)
            poses = []
            for env_idx in range(self.num_envs):
                pos = layouts[env_idx].positions.get(obj)
                if pos is None:
                    raise RuntimeError(f"Placement pool layout is missing a solved position for '{obj.name}'.")
                poses.append(Pose(position_xyz=pos, rotation_xyzw=rotation_xyzw))
            obj.set_initial_pose(PosePerEnv(poses=poses))


def prepare_relation_placement(
    objects: list[ObjectBase],
    num_envs: int,
    *,
    placement_seed: int | None = None,
    resolve_on_reset: bool | None = None,
    embodiment: EmbodimentBase | None = None,
    solver: ArenaRelationSolver | None = None,
) -> PlacementSolution | None:
    """Prepare relation placement with the default or caller-provided solver instance."""
    if solver is not None and (placement_seed is not None or resolve_on_reset is not None):
        raise ValueError("placement_seed and resolve_on_reset are owned by solver when solver is provided.")
    if solver is not None and solver.num_envs != num_envs:
        raise ValueError("solver.num_envs must match num_envs.")
    placement_solver = solver or ArenaRelationSolver(
        num_envs=num_envs,
        placement_seed=placement_seed,
        resolve_on_reset=resolve_on_reset,
        embodiment=embodiment,
    )
    return placement_solver.prepare(objects, embodiment=embodiment)


def _validate_no_explicit_non_anchor_pose_events(objects: list[ObjectBase], anchor_objects_set: set[ObjectBase]) -> None:
    """Reject conflicting explicit pose-reset events on relation-solved objects."""
    for obj in objects:
        if obj not in anchor_objects_set and obj.event_cfg is not None:
            raise RuntimeError(
                f"Non-anchor object '{obj.name}' has an explicit pose-reset event. "
                "Relational solving should not be combined with explicit setting of "
                "poses on non-anchor objects."
            )
