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

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Check
     - Measures
     - Default limit
   * - ``physics_settled``
     - Final linear and angular speed of all measured roots.
     - 0.1 m/s and 0.1 rad/s
   * - ``pose_shift``
     - Root displacement and rotation, excluding intentional ``ClutterOn`` drops.
     - 2 mm and 2 degrees
   * - ``articulation_link_shift``
     - Task-object link motion relative to its root; excludes robot embodiments.
     - 2 mm and 2 degrees
   * - ``support_containment``
     - Clutter bounds relative to the support footprint and minimum resting height.
       Requires successful release ``no_overlap`` and ``clutter_on_relation`` checks.
     - No overhang; 1 cm below the minimum height

All enabled, applicable checks must pass. Disabled and inapplicable checks retain
their skip reasons. For example, set
``params.validators["support_containment"]["containment_margin_m"] = 0.005``
to permit 5 mm overhang. Physics runs for the configured duration; final velocity
limits determine whether the objects are still moving. Post-physics checks use captured
measurements without stepping physics or reading the live environment.

Containers
----------

Release placement stays above the full support bounds, including a container's
rim. For settling inside a bin or bowl, configure the minimum accepted height of
an object's bottom:

.. code-block:: python

   params.validators["support_containment"]["minimum_resting_heights_m"] = {
       "bowl": -0.025,
   }

``bowl`` is the support's runtime scene key. The value is in metres along the
support's local Z axis, after asset scaling, before its world translation. For
the default-scale ``bowl_ycb_robolab`` asset, the inner floor is approximately
-0.025 m and the rim is +0.0275 m in that frame. Choose a height appropriate to
your asset; these values do not apply to all bowls. The configured height must
lie within the support's local Z bounds, including either endpoint. Unknown support keys and invalid heights
are rejected before sampling.

Supports without an override still require a verified flat top surface. An
override changes only the post-physics minimum height; release clearance,
footprint checks and velocity checks remain unchanged. The height is saved with
the validator configuration in each report.

This is a bounding-box footprint and height check. It does not establish exact
containment inside curved walls or detect every object-wall penetration.

Scope and limitations
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Scene feature
     - Support and limitations
   * - Clutter objects
     - Dynamic rigid bodies with gravity enabled. Object sets are rejected;
       resolve them to individual objects before collection, as for recording.
   * - Supports
     - Fixed anchors with an upright quarter-turn orientation. Without a height
       override, containment requires a flat rectangular top covered by a Cube
       collider or connected planar mesh facet.
   * - Rails and rims
     - Set a minimum resting height for the full container, or use an
       ``ObjectReference`` to its flat floor as the ``ClutterOn`` parent and mark
       that reference ``IsAnchor``.
   * - Support references
     - Author translate, orient and scale operations before building the scene.
       Rewriting collider transforms after physics initialization can invalidate
       physics views.
   * - Other placement relations
     - Resolve them to fixed anchors before clutter collection. Keep anchors,
       backgrounds and passive obstacles at their configured poses after building
       the placement pool; runtime edits and pose-changing variations are unsupported.
   * - Reachability
     - ``ClutterOn`` objects cannot require reachability: dropping changes the
       poses checked by the solver. Non-clutter fixed targets may retain
       ``RequiresReachability``. Their solver checks and the displacement limits
       still apply; collection does not rerun IK after physics.
   * - Collision checks
     - Release generation uses normal placement collision discovery, including
       MESH background fixtures and anchored support exclusions. Tilted clutter
       requires BBOX. Post-physics checks do not rerun collision validation.
   * - Robot motion
     - Physics also advances the robot; joints are not immobilized. Moving links
       can affect objects. Root speed and displacement remain checked, but joint
       states are not recorded.

Retain ``no_overlap`` and ``clutter_on_relation`` in the solver checks. Do not use
pre-physics ``physics_settled`` on intentional release poses: post-physics velocity
validation checks the dropped objects instead.

The offline package depends on the solver and shared records. Online placement
and runtime replay do not import offline modules.
