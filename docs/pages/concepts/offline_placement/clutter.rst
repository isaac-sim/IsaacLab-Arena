Offline Clutter Placement
=========================

``ClutterOn`` defines collision-checked release poses. Offline settling uses the
same reset, physics stepping and validation workflow as
:doc:`recording`. Each reset selects a solved layout from the placement pool.
Physics drops the objects, and the configured validators decide whether to keep
their final poses.

Generate reusable layouts
-------------------------

Complete :doc:`../../quickstart/installation` and run the following command from
the repository root in your native environment or Arena container. The maintained
``franka_three_hammers_and_clamp_no_task`` environment drops three hammers and a
clamp onto a table. This command opens a viewer and requires a display. For a
headless run, use ``render=false --viz none`` instead of ``render=true --viz kit``:

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml \
       output=outputs/clutter/episodes.jsonl num_envs=4 num_layouts=10 layouts_per_env=4 \
       seed=42 max_batches=15 settle.num_steps=480 render=true \
       +settle.validators.support_containment.minimum_resting_heights_m.office_table_background=0.5306 \
       --device cpu --viz kit

The table has a beveled top, so the command sets its minimum resting height
explicitly to 0.5306 m along its scaled local Z axis.

Success writes exactly 10 accepted layouts to ``outputs/clutter/episodes.jsonl``
and prints the saved count and path. The acceptance count per batch can vary;
rejected candidates are reported and later batches supply replacements. If the
requested count is not reached within the batch budget, generation raises an
error with rejection reasons and writes no file. Existing files are never
overwritten; choose a new output path when rerunning.

On PhysX, resetting a fixed support can print ``Body must be non-kinematic``
when the background reset writes its velocity. This diagnostic also occurs in
the normal reset path. Check the validator reports and saved-layout count to
determine whether generation succeeded.

``num_envs`` controls parallel environments. ``num_layouts`` is the total number
of accepted layouts to save (default 1).
``layouts_per_env`` sets how many solver layouts each environment receives when
the placement pool refills. ``max_batches`` caps reset-and-settle rounds; the
example permits up to 15. The placement pool pre-solves ``layouts_per_env`` layouts
per environment and refills in the same tranche size when depleted. Each batch
resets every environment,
then advances ``settle.num_steps`` environment steps if any candidate passed
solver validation. Extra accepted layouts in the final batch are omitted.

Generation settings use Hydra ``key=value`` syntax; launcher options use
``--flag`` syntax. Configure post-physics checks through ``settle.validators``;
for example, append ``settle.validators.physics_settled.lin_vel_thresh=0.05``
to lower the final linear-speed limit. See `Acceptance checks`_ for all defaults.

These captures show one environment with seed 42, before and after the same
480-step settling pass. The four-environment command can produce different layouts:

.. figure:: ../../../images/clutter/release.png
   :width: 640px
   :alt: Three hammers and a clamp suspended above a table before settling.

   Release poses selected by the placement solver.

.. figure:: ../../../images/clutter/settled.png
   :width: 640px
   :alt: The hammers and clamp resting on the table after settling.

   Final poses after physics settling.

Each JSONL line stores a layout under ``variations["scene.relation_placement"]``:
root poses keyed by runtime scene name, the source label and validator reports.
Positions are environment-local metres; rotations are xyzw. Inspect one record
as a format example:

.. code-block:: bash

   head -n 1 outputs/clutter/episodes.jsonl | python -m json.tool

For interactive replay with a display available:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_environments/clutter/franka_three_hammers_and_clamp_no_task.yaml \
       --placement_layouts outputs/clutter/episodes.jsonl --num_envs 1 \
       --device cpu --viz kit

This command starts from the first saved layout. The ``NoTask`` environment has
no episode resets; close the viewer to exit. For replay during policy evaluation,
see :doc:`recording`; for queue and reset semantics, see
:doc:`../object_placement/relations`.

After starting ``SimulationApp``, call
``record_settled_placement_layouts(cfg, arena_env=arena_env)`` in
``isaaclab_arena.scripts.record_placement_layouts`` with a
``PlacementRecordingCfg`` from ``isaaclab_arena.offline_placement.recording_config``.
The helper builds and closes its simulation environment. To collect poses from a
simulation environment you already own, use ``record_placements_to_jsonl`` from the
same module. Clutter
scenes merge ``support_containment`` into settle validators automatically when
``ClutterOn`` is present. Collection checks clutter prerequisites before resetting
or stepping physics.


Acceptance checks
-----------------

Recording merges clutter validators for ``ClutterOn`` scenes when settle settings
are omitted or partially overridden. The shared default duration is short. For an
explicit drop window, keep the clutter checks when creating params:

.. code-block:: python

   from isaaclab_arena.offline_placement.clutter_validators import default_clutter_validators
   from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams

   params = SettledPlacementParams(num_steps=480, validators=default_clutter_validators())

Explicit ``params`` are used unchanged, including disabled or custom validators.
``num_steps`` counts environment steps per batch, including their configured
physics substeps.

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

These captures use the library's ``bowl_ycb_robolab`` asset with three 2 cm
cubes, seed 42 and 480 environment steps. The bowl is fixed at Z = 0.56 m;
the cubes use ``ClutterOn(bowl, spread=0.3, clearance_m=0.01, random_yaw=False)``.
Configure a fixed bowl before building the environment:

.. code-block:: python

   from isaaclab.sim import RigidBodyBaseCfg

   bowl.object_cfg.spawn.rigid_props = RigidBodyBaseCfg(kinematic_enabled=True)

.. figure:: ../../../images/clutter/bowl_release.png
   :width: 640px
   :alt: Three cubes above the bowl before physics settling.

   Release poses above the rim, seed 42.

.. figure:: ../../../images/clutter/bowl_settled.png
   :width: 640px
   :alt: The cubes resting inside the bowl after physics settling.

   The same layout after 480 environment steps and acceptance checks.

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
