Relations and Strategies
========================

Relations describe where a placeable asset should be positioned or oriented.
Attach them to an asset with ``add_relation()``. Arena considers all relations
on that asset together.

Positional relations use solver strategies that convert the requested
arrangement into optimization objectives. Orientation relations and placement
markers are handled separately. Most users keep the default strategies;
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
initial pose, while an ``ObjectReference`` derives its pose from the referenced
prim in its parent asset. A tabletop or counter reference is a common anchor.

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

Most scenes can be described with a small set of relations:

``On(parent)``
   Places an object on a support surface and keeps its footprint within the
   support bounds. Use ``clearance_m`` to leave a vertical gap and
   ``edge_margin_m`` to keep the object away from the support edges.

   ``On`` uses the top and horizontal footprint of the parent's axis-aligned
   bounding box. For L-shaped, hollow, or concave supports, anchor an
   ``ObjectReference`` that identifies the valid support surface. During
   initial sampling, a movable parent directly on an anchor uses that anchor's
   bounds as a proxy; deeper ``On`` chains fall back to the first anchor's
   bounds. Final solving and validation still use each relation's actual parent.

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
   region. The keep-out margin defaults to 0.1 m. To change it, provide a
   ``NotNextToLossStrategy`` for ``NotNextTo`` in
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

``FaceTo`` determines the heading after position solving. It cannot be combined
with ``RotateAroundSolution``; when random yaw initialization is enabled,
``FaceTo`` subjects use the relation-derived heading instead. The target must
also participate in relation placement. A movable subject may have only one
``FaceTo`` relation. Neither the subject nor its target can use
``RandomAroundSolution`` with nonzero XY variation. Their XY positions must
differ so that the facing direction is defined.

Combining Relations
-------------------

Relations are most useful in small combinations:

- ``On`` alone means "somewhere on this surface."
- ``On`` with ``NextTo`` means "on this surface, beside that object."
- ``On`` with ``AtPosition`` means "at this horizontal location on the
  surface."
- A positional relation with ``FaceTo`` controls both location and
  orientation.

Avoid specifying more relations than the scene needs. Extra constraints can
make the intended layout harder or impossible to satisfy.

Placement Modifiers
-------------------

``RandomAroundSolution`` and ``RotateAroundSolution`` are pose modifiers applied
after solving. They change how a solved pose is used rather than adding spatial
constraints. In particular, ``RandomAroundSolution`` is intended for direct,
single-environment ``ObjectPlacer`` use; the default builder does not apply it
as a continuous reset range. See
:doc:`./pooled_placement` for variation across environments and resets.

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
intended arrangement.

Collision avoidance is automatic and is not expressed as a relation. See
:doc:`./collision_handling` for collision representations.
