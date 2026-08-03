Placement Relations
===================

Relations describe where a placeable asset should be positioned or oriented.
Attach them to an asset with ``add_relation()``. Arena considers all relations
on that asset together.

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

   from isaaclab_arena.assets.object_base import ObjectType
   from isaaclab_arena.assets.object_reference import ObjectReference

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
   ``ObjectReference`` that isolates the valid support surface. Initial
   sampling considers at most one intermediate movable parent; deeper ``On``
   chains fall back to the anchor bounds.

``NextTo(parent)``
   Places an object beside another object. A side and distance can be specified
   when the arrangement requires them. Geometric validation checks that the
   selected candidate remains on the requested side and at the requested
   distance and marks violations as invalid.

   ``side`` accepts ``Side.POSITIVE_X``, ``Side.NEGATIVE_X``,
   ``Side.POSITIVE_Y``, or ``Side.NEGATIVE_Y``.

``NotNextTo(parent)``
   Defines a side-specific keep-out region next to the parent. The region
   extends outward from the selected side and spans the parent's footprint
   along the perpendicular axis. Validation marks candidates inside that
   region as invalid.

``AtPosition(...)``
   Constrains selected world-coordinate axes. It can be combined with ``On`` so
   the relation determines height while coordinates determine horizontal
   position.

``PositionLimitsBox`` and ``PositionLimitsCylindrical``
   Restrict placement to a rectangular or radial region.

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
``RandomAroundSolution`` with nonzero XY variation.

Combining Relations
-------------------

Relations are most useful in small combinations:

- ``On`` alone means “somewhere on this surface.”
- ``On`` with ``NextTo`` means “on this surface, beside that object.”
- ``On`` with ``AtPosition`` means “at this horizontal location on the
  surface.”
- A positional relation with ``FaceTo`` controls both location and
  orientation.

Avoid specifying more relations than the scene needs. Extra constraints can
make the intended layout harder or impossible to satisfy.

Placement Variation
-------------------

The default builder creates several checked layouts and stores them in a pool.
This is the preferred way to vary positions and orientations across
environments and resets.

``RandomAroundSolution`` and ``RotateAroundSolution`` are advanced post-solve
modifiers. They change how a solved pose is applied rather than adding spatial
constraints. In particular, ``RandomAroundSolution`` is intended for direct,
single-environment ``ObjectPlacer`` use; the default pooled builder path does
not apply it as a continuous reset range.

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
:doc:`./collision_and_validation` for collision representations and validation.
