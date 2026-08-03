Relations and Placement Modifiers
=================================

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
``IsAnchor()`` and set its initial pose before placement. The solver does not
move anchors. A tabletop or counter reference is a common anchor.

Common Relations
----------------

Most scenes can be described with a small set of relations:

``On(parent)``
   Places an object on a support surface and keeps its footprint within the
   support bounds.

``NextTo(parent)``
   Places an object beside another object. A side and distance can be specified
   when the arrangement requires them.

``NotNextTo(parent)``
   Excludes an adjacent region without prescribing the object's final
   location.

``AtPosition(...)``
   Constrains selected world-coordinate axes. It can be combined with ``On`` so
   the relation determines height while coordinates determine horizontal
   position.

``PositionLimitsBox`` and ``PositionLimitsCylindrical``
   Restrict placement to a rectangular or radial region.

``FaceTo(target)``
   Rotates an object around world Z so that its local forward direction points
   toward another object.

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
     - kind: on
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
