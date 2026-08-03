Collision and Placement Validation
==================================

Spatial relations describe the intended arrangement, while collision
constraints and validation determine whether a candidate layout is acceptable.
Arena applies both automatically during relation placement.

What Collision Avoidance Covers
-------------------------------

Every pair of placed, non-anchor assets receives a no-overlap constraint unless
the pair has a support relationship such as ``On``. Placed objects are also
checked against fixed obstacles, including nearby furniture or appliances
discovered in a background scene.

Collision avoidance is automatic. Users describe where objects belong with
relations; they do not add a separate no-collision relation.

Choosing a Collision Representation
-----------------------------------

Arena supports two collision modes:

``CollisionMode.BBOX``
   Uses axis-aligned bounding boxes. It is fast and works well when the box is
   a reasonable approximation of the object.

``CollisionMode.MESH``
   Follows the collision geometry more closely. It is useful when bounding
   boxes reject valid placements around irregular or concave shapes, but it
   requires more computation.

.. figure:: ../../../images/mesh_vs_bbox_collision.png
   :width: 100%
   :alt: Comparison of bounding-box and mesh collision modes
   :align: center

   For the same requested placement, overlapping bounding boxes reject the
   layout, while mesh collision recognizes that the object surfaces do not
   overlap.

Start with ``BBOX``. Use ``MESH`` only when bounding boxes exclude space that
the actual objects can safely occupy. Collision mode changes overlap checking;
it does not change the meaning of relations such as ``On`` or ``NextTo``.

Background and Passive Obstacles
--------------------------------

Objects do not need placement relations to act as obstacles. A table can be an
anchor that supports an ``On`` relation, while a nearby appliance can remain a
fixed passive obstacle. This lets Arena place objects on a surface while
avoiding the rest of a complex scene.

Two Levels of Validation
------------------------

Arena checks placement at two different stages:

1. **Geometric validation** checks collisions and whether relations such as
   ``On``, ``NextTo``, and ``FaceTo`` are satisfied.
2. **Physics validation** can test whether a completed layout remains stable
   after simulation starts.

The distinction matters because a geometrically valid object can still topple
or slide under physics. Use geometric validation for normal placement
generation and physics validation when stability is part of the requirement.

When Placement Fails
--------------------

Check the intent before tuning solver parameters:

1. Confirm that the requested relations are compatible.
2. Check that anchors and support surfaces have the expected poses and bounds.
3. Check whether a bounding box is rejecting otherwise valid free space.
4. Inspect physics behavior if geometry passes but objects move or topple.

Only after identifying the failure should you adjust clearances, collision
mode, placement attempts, or solver settings.
