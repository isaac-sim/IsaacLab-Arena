Relations and Strategies
========================

Relations describe where a placeable asset should be positioned or oriented.
Attach them to an asset with ``add_relation()``. Arena considers all relations
on that asset together.

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
``IsAnchor()``; the solver does not move it. A standalone anchor needs a fixed
initial pose; in YAML, an omitted pose defaults to identity. An
``ObjectReference`` instead derives its pose from the referenced prim within
its parent asset. A tabletop or counter reference is a common anchor.

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

   ``On`` uses the top and horizontal footprint of the parent's axis-aligned
   bounding box. For L-shaped, hollow, or concave supports, anchor an
   ``ObjectReference`` that identifies the valid support surface.

   During initial sampling, a movable parent directly on an anchor uses that
   anchor's bounds as a proxy. For a deeper chain, such as a spoon ``On`` a cup
   ``On`` a tray ``On`` a table, initialization of the spoon uses the first
   anchor collected by ``ObjectPlacer`` as a proxy. This affects only the
   starting pose; final solving and validation use each relation's actual
   parent.

``ClutterOn(parent)``
   Defines a clutter release above a fixed ``IsAnchor`` support. ``ObjectPlacer``
   samples within a centered fraction of its footprint (``spread``, default
   0.2) and lowers objects into free vertical space in asset order, leaving
   ``clearance_m`` above the surface and at least ``gap_m`` between overlapping bounds.
   Initialization also honors the solver collision clearance when it is larger.
   The registered loss and normal placement validators enforce that scaled
   release footprint and minimum surface clearance. Offline settled-pile
   validation uses the whole support footprint. ``gap_m`` controls initialization; subsequent
   solving uses the shared collision clearance.

   This is a release arrangement, not a settled pile. Physics makes the objects
   fall when simulation starts. ``ClutterOn`` can combine with other spatial
   relations, such as ``AtPosition``, but cannot use ``RandomAroundSolution``.
   ``RotateAroundSolution`` sets its base rotation; ``random_yaw`` (default True) adds world-Z yaw while
   preserving that rotation's tilt. Failure handling uses the same
   ``ObjectPlacerParams.allow_best_loss_fallbacks`` option as other relations;
   set it to False when only validated layouts are acceptable.

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

Cached Layouts
--------------

An environment can load a companion pose YAML through ``placement_layouts``
in its graph specification or the ``--placement_layouts`` runtime flag. The
YAML field is relative to the environment file; the CLI override is relative
to the working directory and takes precedence.

The companion file maps graph object IDs to equal-length lists of poses:

.. code-block:: yaml

   mug:
   - position_xyz: [0.1, 0.2, 0.8]
     rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
   - position_xyz: [-0.1, 0.2, 0.8]
     rotation_xyzw: [0.0, 0.0, 0.0, 1.0]

Positions are in the local environment frame, in metres. One index selects a
complete layout across all objects. Environment ``i`` starts at index
``i % num_layouts`` and cycles on its own resets. Reusing a cache bypasses
relation solving; it does not run physics settling. All non-anchor objects
with spatial relations must be included, and object sets are unsupported.
Assets must expose writable physics roots. Disable pose-changing variations
and callbacks when exact replay is required.

The CAP offline generator in
``isaaclab_arena_environments/isaac_cap/clutter/generate_clutter_scene.py``
creates such files from ``ClutterOn`` relations. Its adjacent ``README.rst``
describes generation controls and validation limits.

Cached placement validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Companion pose files validate finite values, unit quaternions, object coverage
and layout counts. Loading does not rerun geometric, relation or reachability
checks. ``placement_validators`` configure generated placements and are not
rerun on cached replay. Keep those settings in the source environment YAML;
loading its companion file does not require removing them. Cached poses must
match the scene used to generate them; successful loading does not certify
physical validity or reachability.


Next Steps
----------

Continue to :doc:`./collision_handling` to learn how Arena checks placed assets
against one another and against fixed geometry.
