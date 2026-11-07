# Placing objects and the robot

Read this when laying out an environment's objects and robot with spatial relations. For the
relations themselves, see [relations](../../../docs/pages/concepts/object_placement/relations.rst),
and for reading rejected layouts, see
[placement diagnostics](../../../docs/pages/concepts/object_placement/placement_diagnostics.rst).

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

## Reset variation

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
solving time to each build. Do not run episodes to see how resets vary; a zero-action episode only
resets at its timeout, which takes minutes.
