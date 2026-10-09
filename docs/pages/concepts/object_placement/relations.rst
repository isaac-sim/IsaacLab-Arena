Relations and Strategies
========================

Relations describe where a placeable asset should be positioned or oriented.
Attach them to an asset with ``add_relation()``. Arena considers all relations
on that asset together. Arena applies solved and recorded layouts through the
:doc:`relation placement variation <../variations/relation_placement>`.

Positional relations use solver strategies that convert the requested
arrangement into optimization objectives. Orientation relations and placement
modifiers are handled separately. Most users keep the default strategies;
advanced users can replace entries in ``RelationSolverParams.strategies``.

.. code-block:: python

   from isaaclab_arena.relations.relations import IsAnchor, NextTo, On

   table.add_relation(IsAnchor())
   mug.add_relation(On(table))
   bowl.add_relation(On(table))
   bowl.add_relation(NextTo(mug))

This describes the intended arrangement without requiring coordinates derived
from the dimensions of the table, mug, and bowl.

Anchors
-------

An anchor is a fixed reference in the relation graph. Mark it with
``IsAnchor()``; the solver does not move it. This marker does not make an asset
static or kinematic in physics. A standalone anchor needs a fixed
initial pose; in YAML, an omitted pose defaults to identity. An
``ObjectReference`` instead derives its pose from the referenced prim within
its parent asset. A tabletop or counter reference is a common anchor.

An anchor's fixed root rotation must be a multiple of 90 degrees about world Z,
with no tilt. For an ``ObjectReference``, this restriction applies to its parent
asset's pose; the referenced prim's authored rotation is already included in its
bounds. The solver rotates these bounds into world-aligned bounds.

When the support surface is part of a larger background, use an
``ObjectReference`` to identify that surface:

.. code-block:: python

   from isaaclab_arena.assets.object_reference import ObjectReference
   from isaaclab_arena.assets.object_type import ObjectType

   table_reference = ObjectReference(
       name="table",
       prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
       parent_asset=background,
       object_type=ObjectType.RIGID,
   )
   table_reference.add_relation(IsAnchor())
   mug.add_relation(On(table_reference))

Anchor the background asset directly when its complete bounds represent the
support. Use an ``ObjectReference`` when only an internal tabletop, counter, or
similar prim should support placement.

Common Relations
----------------

Most environments can be described with a small set of relations:

``On(parent)``
   Places an object on a support surface and keeps its footprint within the
   support bounds. Use ``clearance_m`` to leave a vertical gap and
   ``edge_margin_m`` to keep the object away from the support edges.

   Set ``overlap=True`` to allow the object to extend beyond the support:

   .. code-block:: python

      box.add_relation(On(table, overlap=True))

   This requires overlap in both X and Y (edge contact counts), ignores
   ``edge_margin_m``, and keeps the same height constraint. It does not guarantee
   stable support: the object may tip or fall. The default is ``overlap=False``.

   ``On`` uses the top and horizontal footprint of the parent's axis-aligned
   bounding box. For L-shaped, hollow, or concave supports, anchor an
   ``ObjectReference`` that identifies the valid support surface.

   During initial sampling, the default initializer follows the object's ``On``
   chain and uses the nearest ``IsAnchor`` ancestor's bounds as a proxy. If the
   chain has no anchor or loops, it falls back to the first anchor collected by
   ``ObjectPlacer``. This affects only the starting pose; final solving and
   validation use each relation's actual parent.

.. _clutter-on-relation:

``ClutterOn(parent)``
   Defines a **release pose** above an ``IsAnchor`` support, before physics.
   ``ObjectPlacer`` samples a central release region and raises objects above
   overlapping footprints. The support must be upright, with a fixed yaw that
   is a multiple of 90 degrees.

   .. list-table::
      :header-rows: 1
      :widths: 22 13 65

      * - Parameter
        - Default
        - Meaning
      * - ``spread``
        - ``0.2``
        - Fraction of the support's width and depth, in ``(0, 1]``. A value of
          0.2 selects the central 20% of each axis, or 4% of the XY area.
      * - ``clearance_m``
        - ``0.01``
        - Minimum height of the object's bottom above the support top, in metres.
      * - ``gap_m``
        - ``0.03``
        - Initial gap to neighboring release bounds, in metres. Sampling uses
          the larger of this value and the solver's collision clearance;
          subsequent solving uses the shared collision clearance.
      * - ``edge_margin_m``
        - ``0.0``
        - Inward margin within the release region, in metres. The rotated object
          footprint must fit inside the remaining region.
      * - ``random_yaw``
        - ``True``
        - Sample world-Z yaw in addition to ``RotateAroundSolution``. This setting
          controls clutter independently of ``ObjectPlacerParams.random_yaw_init``.

   ``ClutterOn`` must be the object's only spatial relation and cannot use
   ``RandomAroundSolution``. ``RotateAroundSolution`` sets its base rotation;
   random yaw preserves that rotation's tilt. Both ``bbox`` and ``mesh`` collision
   modes support yaw-only clutter. Roll or pitch requires ``collision_mode="bbox"``
   on the object, whose bounds enclose the full rotation.

   The ``clutter_on_relation`` check enforces the release footprint and minimum
   height without requiring contact or an upper height limit. It allows
   1 micrometre of numerical slack on ``clearance_m``, but never penetration below
   the support top. ``on_relation_z_tolerance_m`` does not apply to clutter.
   With explicit ``enabled_checks`` or ``required_checks``, include
   ``clutter_on_relation`` for clutter and ``on_relation`` for ordinary ``On``
   objects. Both checks are enabled by default.

   A **settled pose** is the final pose after the configured physics interval.
   An **accepted layout** passes the required pre-physics and enabled, applicable
   :ref:`post-physics checks <recording-post-physics-checks>` for the complete
   candidate. Settled objects may use the full support footprint, beyond the
   smaller release region. Release validation alone does not certify the final pile.
   For pooled placement, disable ``ObjectPlacerParams.allow_best_loss_fallbacks``
   to reject invalid layouts. Direct ``ObjectPlacer.place()`` callers must check
   each result's ``success`` before using it.

.. _next-to-relation:

``NextTo(parent)``
   Places an object beside another object. A side and distance can be specified
   when needed. Geometric validation rejects candidates that are not on the
   requested side, or whose gap to the parent differs from ``distance_m`` by
   more than ``tolerance_m`` (0.01 m by default). Placing the object closer than
   requested also fails.

   With no additional arguments, ``NextTo(parent)`` places the subject on the
   parent's positive X side at a distance of 0.05 m.

   ``side`` accepts ``Side.POSITIVE_X``, ``Side.NEGATIVE_X``,
   ``Side.POSITIVE_Y``, or ``Side.NEGATIVE_Y``.

``NotNextTo(parent)``
   Defines a side-specific keep-out region next to the parent. The region
   extends outward from the selected side and spans the parent's footprint
   along the perpendicular axis. Validation rejects candidates inside that
   region. The keep-out margin defaults to 0.1 m. Advanced users can change it
   by providing a ``NotNextToLossStrategy`` for ``NotNextTo`` in
   ``RelationSolverParams.strategies``.

``AtPosition(...)``
   Constrains selected world-coordinate axes. It can be combined with ``On`` so
   the relation determines height while coordinates determine horizontal
   position.

``PositionLimitsBox`` and ``PositionLimitsCylindrical``
   ``PositionLimitsBox`` constrains selected world X, Y, or Z coordinates
   between optional minimum and maximum values.
   ``PositionLimitsCylindrical`` constrains the XY distance from a chosen
   center using a minimum radius, maximum radius, or both; it does not
   constrain Z.

``FaceTo(target)``
   Rotates an object around world Z so that its local +X heading points toward
   another object.

.. code-block:: python

   from isaaclab_arena.relations.relations import FaceTo

   target.add_relation(On(table))
   camera_prop.add_relation(On(table))
   camera_prop.add_relation(FaceTo(target))

``FaceTo`` determines the heading after position solving:

- It cannot be combined with ``RotateAroundSolution``.
- When random yaw initialization is enabled, it replaces the random heading.
- The target must also participate in relation placement.
- A movable subject can have only one ``FaceTo`` relation.
- Neither the subject nor target can use ``RandomAroundSolution`` with nonzero
  XY offsets.
- The subject and target must have different XY positions.

Combining Relations
-------------------

Relations are most useful in small combinations:

- ``On`` alone means "somewhere on this surface."
- ``On`` with ``NextTo`` means "on this surface, beside that object."
- ``On`` with ``AtPosition`` means "at this horizontal location on the
  surface."
- A positional relation with ``FaceTo`` controls both location and
  orientation.

Avoid specifying more relations than the environment needs. Extra constraints
can make the intended layout harder or impossible to satisfy.

Placement Modifiers
-------------------

``RandomAroundSolution`` and ``RotateAroundSolution`` are pose modifiers applied
after solving. They change how a solved pose is used rather than adding spatial
constraints:

- ``RandomAroundSolution`` creates a range of positions and orientations around
  the solved pose. It is intended for direct, single-environment
  ``ObjectPlacer`` use; the default builder does not apply it as a continuous
  reset range.
- ``RotateAroundSolution`` adds a fixed roll, pitch, or yaw to the solved pose.
  The robot-placement example later in this sequence uses it to set the
  robot's final heading.

Relations in Environment Specifications
---------------------------------------

YAML environment specifications use the same model:

.. code-block:: yaml

   relations:
     - kind: is_anchor
       subject: table
     - kind: 'on'
       subject: mug
       reference: table
     - kind: next_to
       subject: bowl
       reference: mug

Each entry identifies the relation, its subject, and—when needed—the object it
references. Add parameters only when the default relation does not express the
intended arrangement. Quote ``'on'`` so YAML treats it as a string rather than
a Boolean value.

Collision handling is integrated into placement and is not expressed as a
relation.

.. _recorded-layouts:

Recorded Layouts
----------------

Set the companion file in an Experiment Definition. This maintained example
replays the settled clamp recording:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/settled_placement_replay_experiment.yaml
   :language: yaml
   :start-at: runs:

Run it through the Experiment Runner:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
       --experiment_config isaaclab_arena_environments/experiment_configs/settled_placement_replay_experiment.yaml \
       --device cpu --viz kit \
       --output_base_dir outputs/placements/evaluation

Recorded variation replay is not exposed by the Policy Runner CLI. Python
callers can set
``ArenaEnvBuilderCfg(recorded_variation_samples_path="layouts.jsonl")``
directly. All file paths are relative to the working directory.

A ten-layout example for ``isaaclab_arena/tests/test_data/placement_replay.yaml``
is available in ``isaaclab_arena/tests/test_data/placement_replay.jsonl``.

Each JSONL line contains one complete layout under
``variations["scene.relation_placement"]["poses"]``. Poses use runtime scene keys
for both YAML and Python environments. Use ``asset.get_scene_root_keys()`` to
identify all owned physics roots. Ordinary objects use their instance names;
single-root embodiments commonly use ``"robot"``. Compound embodiments expose
each owned root, whose runtime name can differ from the YAML node ID.
Positions are environment-local, in metres; rotations are xyzw quaternions.
Every nonblank line must contain the placement block with the same object set.
Additional episode fields are ignored; episodes without placement records cannot
be loaded. Set the replay path before ``compose_manager_cfg()`` or
``make_registered()``.

Replay validates scene-root names but does not currently verify that they refer
to the same concrete objects that produced the recording. When an environment
supports object selection, use the same objects for recording and replay.

``PlacementLayouts.write_episode_jsonl(path, source=...)`` writes the same format.
The caller supplies the source label, such as ``"solver"`` or ``"settled"``;
the writer does not solve or simulate the poses.

Replay Order
~~~~~~~~~~~~

Resetting environments draw consecutive layouts from one shared queue, in reset
request order. The queue wraps after its last layout. For four layouts and three
environments, successive full resets select ``[0, 1, 2]``, then ``[3, 0, 1]``.
A partial reset consumes only the layouts needed by those environments; other
poses remain unchanged. Layouts can repeat across active environments after the
queue wraps. If the environment count is a multiple of the layout count,
repeated full resets assign the same layout to each environment. The queue covers
all layouts across the batch; it does not guarantee that each environment visits
every layout. Partial-reset order determines later assignments, so different
policies may receive different per-environment sequences.

.. _placement-replay-configuration:

Replay Configuration
~~~~~~~~~~~~~~~~~~~~

Replay validates finite poses, unit quaternions, consistent object coverage and
reset ownership. Recorded objects share one reset writer, which zeros their root
velocities. All non-anchor objects with spatial relations must be included,
as must a non-anchor embodiment carrying any placement relation or marker.

Before replaying a recording:

- Use concrete assets rather than object sets, and include every owned root of
  each recorded asset.
- Enable pose resets and use fixed initial poses for assets with pose-reset
  events. Remove ``RandomAroundSolution`` from recorded assets and keep their
  initial root velocities zero.
- ``scene.relation_placement.resample_on_reset`` controls live placement only;
  replay always applies its
  scheduled recorded layout.

Placement validator settings apply only when solving; they do not revalidate a
recorded layout or open the solver's debug viewer.

Replay seeds construction from the recording and does not build or consume a
relation-placement pool. It does not rerun geometry, reachability or settling
checks, and is compatible with ``--no_solve_relations``. Preserve the scene
geometry, robot initialization and physics settings used to record the layouts.
The file contains root poses, not joint states or other randomized properties;
their normal reset initialization still applies. Disable pose-changing
variations and callbacks when exact root replay is required.

Next Steps
----------

See :doc:`../offline_placement/recording` to settle layouts and save poses
for reuse. For table and container examples, see
:doc:`../offline_placement/clutter`.

Continue to :doc:`./collision_handling` to learn how Arena checks placed assets
against one another and against fixed geometry.
