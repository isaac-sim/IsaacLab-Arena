"""GoalSpec goal_relations → concrete (x, y, z) GOAL positions.

Scope: goal-state only. Initial poses are already baked into the scene USD
and goal_spec.initial_state — this module does NOT re-solve them.

Pipeline:
    predicate (LLM vocab) → Arena Relation → RelationSolver → xyz

Step 1 is `predicate_to_relation.predicate_to_relations()` (already exists).
Step 2 is Arena's differentiable `RelationSolver`. This module glues them
with lightweight handles so it works at env-build/runtime without the heavy
`isaaclab_arena.assets.object.Object` machinery (which wants USD paths and
Isaac Lab configs).

Consumers:
- CuRobo target poses (plan-ahead motion planning)
- `dropped(obj)` termination: check obj.pos ≈ resolved goal xyz

Adding a new LLM predicate does NOT require changes here — only in
predicate_to_relation.py — because downstream everything speaks the fixed
Arena Relation vocabulary (On/Inside/NextTo/AtPosition).
"""

from __future__ import annotations

from isaaclab_arena.env_gen.predicate_to_relation import predicate_to_relations
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.env_gen.predicate_vocab import normalize_predicate
from isaaclab_arena.relations.relations import (
    AtPosition, IsAnchor, Relation, RelationBase,
)
from isaaclab_arena.task_gen.goal_spec import GoalSpec
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


# Defaults for non-office tables (bamboo/black). Office_table callers should
# pass table_pose=(0.55, 0, -0.697), table_dims=(1.26, 0.8, 0.697).
_DEFAULT_TABLE_POS = (0.547, 0.0, -0.35)
_DEFAULT_TABLE_DIMS = (0.7, 1.0, 0.35)


class _Handle:
    """Minimal object handle satisfying RelationSolver's duck-typed interface.

    Arena's `Object` class requires USD paths, spawner/contact configs, etc.
    The solver only needs: `.name`, `.get_relations()`,
    `.get_spatial_relations()`, `.get_bounding_box()`, plus identity hashing.
    """

    __slots__ = ("name", "position", "bbox", "relations")

    def __init__(
        self,
        name: str,
        position: tuple[float, float, float],
        dims: tuple[float, float, float],
    ):
        self.name = name
        self.position = position
        dx, dy, dz = dims
        # Arena asset convention: origin at base, centered in XY.
        self.bbox = AxisAlignedBoundingBox(
            min_point=(-dx / 2, -dy / 2, 0.0),
            max_point=(dx / 2, dy / 2, dz),
        )
        self.relations: list[RelationBase] = []

    def get_relations(self) -> list[RelationBase]:
        return self.relations

    def get_spatial_relations(self) -> list[RelationBase]:
        return [r for r in self.relations if isinstance(r, (Relation, AtPosition))]

    def add_relation(self, rel: RelationBase) -> None:
        self.relations.append(rel)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        return self.bbox

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        # Anchors only — the solver uses this for fixed-pose parents. Non-anchor
        # parents go through get_bounding_box().translated(current_pos) instead,
        # so we don't need to track rotation here.
        return self.bbox.translated(self.position)

    # Identity hashing — the solver uses objects as dict keys.
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other


def resolve_goal_xyz(
    goal_spec: GoalSpec,
    table_pose: tuple[float, float, float] = _DEFAULT_TABLE_POS,
    table_dims: tuple[float, float, float] = _DEFAULT_TABLE_DIMS,
    scene_positions: dict[str, tuple[float, float, float]] | None = None,
    scene_dims: dict[str, tuple[float, float, float]] | None = None,
    verbose: bool = False,
) -> dict[str, tuple[float, float, float]]:
    """Resolve each `goal_relation` to a concrete (x, y, z) for its `object`.

    Args:
        goal_spec: the task definition. Uses `initial_state` for starting
            positions/dims and `goal_relations` for the constraints.
        table_pose: (x, y, z) of the table anchor.
        table_dims: (dx, dy, dz) of the table.
        verbose: forward to the solver.

    Returns:
        {object_name: (x, y, z)} for each object named in
        `goal_relations.object` that was resolvable. Objects without
        initial_state dims or an unknown target are skipped.
    """
    handles: dict[str, _Handle] = {}
    initial_positions: dict[_Handle, tuple[float, float, float]] = {}

    # Table anchor
    table = _Handle("table", table_pose, table_dims)
    table.add_relation(IsAnchor())
    handles["table"] = table
    initial_positions[table] = table_pose

    # Only the objects named as goal_relations.object are optimizable; every
    # other scene object (including target objects like sugar_box that appear
    # in goal_relations.target) is an anchor so the solver can't satisfy a
    # relation by sliding the wrong object.
    movable_names = {gr.object for gr in goal_spec.goal_relations}

    for name, state in goal_spec.initial_state.items():
        if state.dims is None:
            continue
        h = _Handle(name, state.position, state.dims)
        handles[name] = h
        initial_positions[h] = state.position
        if name not in movable_names:
            h.add_relation(IsAnchor())

    # Scene-wide fallback: task_gen filters initial_state to contact_objects,
    # so target-only objects (e.g. desk_caddy_001 as the target of every
    # goal_relation) may be missing. Fill those in from the live scene.
    if scene_positions:
        default_dims = (0.1, 0.1, 0.1)  # approximate bbox if dims unknown
        target_names = {gr.target for gr in goal_spec.goal_relations}
        for name in target_names:
            if name in handles or name not in scene_positions:
                continue
            pos = scene_positions[name]
            dims = (scene_dims or {}).get(name, default_dims)
            h = _Handle(name, pos, dims)
            h.add_relation(IsAnchor())
            handles[name] = h
            initial_positions[h] = pos

    # Attach goal_relations as Arena Relations
    attached = 0
    for gr in goal_spec.goal_relations:
        obj = handles.get(gr.object)
        target = handles.get(gr.target)
        if obj is None or target is None:
            continue
        for r in predicate_to_relations(gr.predicate, obj, target):
            obj.add_relation(r)
            attached += 1
        # `center_of` maps to bare On(target); the Arena docstring notes the
        # caller must add AtPosition(target.x, target.y) to actually CENTER.
        # predicate_to_relation.py is asset-agnostic, so we do it here where
        # target.position is known.
        if normalize_predicate(gr.predicate) == "center_of":
            tx, ty, _ = target.position
            obj.add_relation(AtPosition(x=tx, y=ty))
            attached += 1

    if attached == 0:
        return {}

    solver = RelationSolver(RelationSolverParams(verbose=verbose))
    final = solver.solve(list(handles.values()), initial_positions)

    result: dict[str, tuple[float, float, float]] = {}
    for gr in goal_spec.goal_relations:
        h = handles.get(gr.object)
        if h is not None and h in final:
            x, y, z = final[h]
            result[gr.object] = (float(x), float(y), float(z))
    return result
