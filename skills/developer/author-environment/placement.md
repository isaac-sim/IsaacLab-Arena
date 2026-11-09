# Placing objects and the robot

Read this when laying out an environment's objects and robot with spatial relations. For the
relations themselves, see [relations](../../../docs/pages/concepts/object_placement/relations.rst),
and for reading rejected layouts, see
[placement diagnostics](../../../docs/pages/concepts/object_placement/placement_diagnostics.rst).

## A simple layout

Most table-top scenes need two relations: anchor the support, then place each object `on` it. From
the RoboLab [banana and bowl scene](../../../isaaclab_arena_environments/robolab/scenes/banana_bowl.yaml):

```yaml
relations:
- kind: is_anchor
  subject: table_oak_robolab
- kind: 'on'
  subject: bowl
  reference: table_oak_robolab
- kind: 'on'
  subject: banana
  reference: table_oak_robolab
```

Add `next_to` or `align_with` only when objects must be arranged relative to each other.

## A layout in a larger background

A scene in a larger background, such as a kitchen, also needs references to its support surfaces, a
placed robot, and objects arranged relative to each other. The KitchenBench
[banana and orange in bowl](../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml)
environment lays them out as follows:

- **Supports:** the kitchen uses mesh collision, so it has no relations. Object references to its
  floor and corner counter are the anchors instead; see [background collision](#background-collision).
- **Task objects:** the bowl sits `on` the counter, somewhere within a `position_limits_box`. The
  banana is aligned to the bowl on both axes and the orange to the banana, so the three move
  together, and each `tolerance_m` lets them vary around their offsets.
- **Robot:** the robot stands `on` the floor, at a fixed distance in front of the counter
  (`align_with` on `y`), turned 90 degrees by `rotate_around_solution`. Aligning it with the bowl
  on `x` makes it follow the task objects along the counter; see
  [robot placement](#robot-placement).

## Background collision

Give a complex background with many built-in objects, such as a kitchen with trays and utensils on
its counters, `collision_mode: mesh` so placement avoids those objects. A simple background, such as
a RoboLab table, can keep the default bounding-box mode.

A mesh-mode background cannot have relations, not even `is_anchor`; placement fails with an error
if it does. Anchor an object reference to each support prim instead, such as the countertop and
floor, found with the scene inspector, and place objects `on` those references. See
[anchors](../../../docs/pages/concepts/object_placement/relations.rst) in the relations page.

## Robot placement

To stand a robot at a work surface, follow
[robot embodiment placement](../../../docs/pages/concepts/object_placement/solver.rst#robot-embodiment-placement).

## Reachability

Keep task objects within the robot's reach. Tasks such as `PickAndPlaceTask` mark their objects
`requires_reachability`, and the `ik_reachable` check rejects layouts where the arm cannot reach
them, so you need not check reach yourself. Its pass counts are in the placement diagnostics.

## Diverse layouts

Prefer relative relations (`next_to`, `align_with`, `on`) with loose tolerances over absolute
positions, and make the bounds as loose as the arrangement allows, so layouts vary as much as
possible between resets. An object varies along a relation's axis by about that relation's
`tolerance_m`. For large variation:

- Place objects that belong together relative to each other, such as the robot relative to the
  task objects, so the group moves as one, rather than pinning each to the support separately.
- Give the group's anchor a wide band along the support. Do not fit the bounds to a clear strip
  found with the scene inspector; let `no_overlap` reject layouts that hit the scene's own objects, and
  `ik_reachable` reject layouts the robot cannot reach.
- When the side does not matter, use `align_with` on both axes rather than `next_to` with a fixed
  side.

Wide bounds lower the share of candidates that pass, so the pool fills more slowly; the placement
diagnostics' per-check pass counts show which relation or check rejects them. Keep the pool settings
(`min_unique_layouts_per_env`, `max_placement_attempts`) at their defaults: every extra layout adds
solving time to each build.
