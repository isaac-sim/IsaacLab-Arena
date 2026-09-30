Offline Clutter Settling
========================

``ClutterOn`` defines collision-checked release poses. Offline settling uses the
same reset, physics stepping and validation workflow as
:doc:`recording`. Each reset selects a solved layout from the placement pool.
Physics drops the objects, and the configured validators decide whether to keep
their final poses.

Collect layouts
---------------

Build the environment with relation solving enabled. For an environment definition
containing ``ClutterOn`` relations:

.. code-block:: python

   from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
   from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
   from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
   from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
   from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams

   params = SettledPlacementParams(
       num_steps=480,
       validators=default_clutter_validators(),
   )
   env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=4)).make_registered()
   try:
       result = collect_settled_placements(
           env, num_batches=2, params=params,
           scene_assets=arena_env.get_placement_assets(),
           render=True, log_progress=True,
       )
       print(result.accepted_indices)
       print(result.rejections)
   finally:
       env.close()

Run this after ``SimulationApp`` starts. Do not reset before collecting: collection
resets once per batch. Two batches of four environments sample eight candidates.
``num_steps`` counts environment steps, each containing the configured number of
physics substeps; it is independent of the number of batches.

``collect_settled_placements`` detects ``ClutterOn`` relations and checks their
scene prerequisites before resetting or stepping physics. Pass the complete
``scene_assets`` list so it can check passive objects as well as placement objects.
It returns ``SettledPlacementResult``:

* ``poses`` contains accepted rigid-object and articulation root poses, keyed by
  runtime scene name. Positions are environment-local metres; rotations are xyzw.
* ``accepted_indices`` identifies each accepted environment and reset batch.
* ``validation`` retains solver verdicts and post-physics settings and outcomes.
* ``rejections`` explains failed candidates. All-rejected runs return empty pose
  lists. Increase ``num_batches`` to sample more candidates.

The caller owns the environment. It remains at its final state after collection
or failure.

Acceptance checks
-----------------

When ``params`` is omitted, collection selects clutter defaults for ``ClutterOn``
scenes and ordinary recording defaults otherwise. The shared default duration is
short; the example above explicitly sets a longer drop window. Explicit ``params``
are used unchanged, including disabled or custom validators. Use
``default_clutter_validators()`` when configuring a longer clutter run.

The shared validator builder and evaluator run these clutter defaults:

* ``physics_settled`` checks final root velocities for all measured objects.
* ``pose_shift`` limits other objects to 2 mm translation and 2 degrees rotation.
  The shared ``PoseShiftValidator`` excludes intentional ``ClutterOn`` drops.
* ``articulation_link_shift`` checks articulated task objects using the existing
  root-relative link check. Robot links are excluded, as in ordinary recording.
* ``support_containment`` checks that clutter stays within its support footprint
  and does not fall through it. It also requires successful release
  ``no_overlap`` and ``clutter_on_relation`` checks.

All enabled, applicable checks must pass. Disabled and inapplicable checks retain
their skip reasons. For example, set
``params.validators["support_containment"]["containment_margin_m"] = 0.005``
to permit 5 mm overhang. Physics runs for the configured duration; final velocity
limits determine whether the objects are still moving.
Validators read captured measurements, without stepping physics or reading the
live environment themselves.

Scope and limitations
---------------------

Supports must be fixed anchors with a flat rectangular collision surface covering
the top face of their bounds. This is checked before the first reset. A whole tray
with a rim or a table with rails does not satisfy that assumption: its highest
point is not the resting surface. Use an ``ObjectReference`` to the Xform containing
only the tray floor or tabletop collider as the ``ClutterOn`` parent, and mark that
reference ``IsAnchor``. Author the reference's translate, orient and scale operations
before building the environment; rewriting a collider's transforms after physics
initialization can invalidate its physics view. Both release placement and settled
containment then use that surface's bounds. Cube colliders and connected planar
mesh facets are supported; curved surfaces and surfaces assembled from separate
coplanar colliders are not.

Clutter members must be dynamic rigid bodies with gravity enabled in every
selected object-set variant. Assign object-set variants
before scene construction. Other placement must already be resolved to fixed
anchors. Anchors, backgrounds and passive obstacles must match their configured
poses; pose-changing reset variations on this fixed geometry are unsupported.

Release generation uses normal placement's collision discovery, including MESH
background fixtures, per-asset collision modes, and anchored support exclusions.
Tilted clutter requires BBOX because the solver's mesh checks use yaw only.
Post-physics checks do not rerun collision or IK validation.

Assets marked ``RequiresReachability`` are rejected. Use a generation scene
without reachability requirements. Retain ``no_overlap`` and
``clutter_on_relation`` in its solver checks. Do not use the pre-physics
``physics_settled`` check on intentional release poses; the post-physics velocity
validator checks the dropped objects instead.

Physics also advances the robot. The recorder does not immobilize its joints,
and moving robot links can affect the objects. Only root poses are collected;
joint states are not recorded. Root speed and displacement remain checked.

The offline package depends on the solver and shared records. Online placement
and runtime replay do not import offline modules.
