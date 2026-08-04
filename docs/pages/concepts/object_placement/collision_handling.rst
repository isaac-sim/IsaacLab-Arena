Collision Handling
==================

Collision constraints guide the solver away from overlapping layouts. Users
describe where assets belong with spatial relations; they do not add a separate
no-collision relation.

What Collision Avoidance Covers
-------------------------------

Every pair of placed, non-anchor assets receives a no-overlap constraint unless
the pair has a support relationship such as ``On``. Fixed scene assets can also
act as passive obstacles when Arena can obtain their collision bounds. When a
background uses mesh collision, its collision geometry can act as a passive
obstacle, allowing placement to avoid furniture and appliances within a complex
scene.

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

Configure mesh collision when defining the environment:

.. code-block:: python

   from isaaclab_arena.environments.isaaclab_arena_environment import (
       IsaacLabArenaEnvironment,
   )
   from isaaclab_arena.relations.collision_mode import CollisionMode
   from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
   from isaaclab_arena.relations.relation_solver_params import (
       RelationSolverParams,
   )

   placer_params = ObjectPlacerParams(
       solver_params=RelationSolverParams(
           collision_mode=CollisionMode.MESH,
           clearance_m=0.01,
       )
   )
   environment = IsaacLabArenaEnvironment(
       name="mesh_placement",
       scene=scene,
       placer_params=placer_params,
   )

``RelationSolverParams.clearance_m`` sets the minimum separation used when
checking collisions between assets. ``On.clearance_m`` is separate and controls
only the vertical gap above a support surface.

Start with ``BBOX``. Use ``MESH`` only when bounding boxes exclude space that
the actual objects can safely occupy. Collision mode changes overlap checking;
it does not change the meaning of relations such as ``On`` or ``NextTo``.

An individual asset can override the solver default. In an environment graph,
set the mode in that asset's parameters:

.. code-block:: yaml

   background:
     id: kitchen
     registry_name: lightwheel_robocasa_kitchen
     params:
       collision_mode: mesh

When an asset has no extractable collision mesh, Arena uses its bounding box as
a proxy. If neither asset in a pair provides a mesh, the pair falls back to
bounding-box checking and logs the fallback.

Background and Passive Obstacles
--------------------------------

Objects do not need placement relations to act as obstacles. A table can be an
anchor that supports an ``On`` relation, while a nearby appliance can remain a
fixed passive obstacle. This lets Arena place objects on a surface while
avoiding the rest of a complex scene.

The included kitchen example shows objects placed on a counter while avoiding
the background mesh. It requires a graphical display; see
:ref:`the display setup note <placement-viewer-display>` for remote and
container setup:

.. code-block:: bash

   python \
     isaaclab_arena_examples/relations/isaac_sim_kitchen_background_collision_notebook.py \
     --viz kit \
     --view_steps 0

``--viz kit`` opens the viewer, and ``--view_steps 0`` keeps it open until you
close the application.

Validation checks determine whether a solved layout is accepted. See
:doc:`./validation` for geometric, reachability, and physics validation.
