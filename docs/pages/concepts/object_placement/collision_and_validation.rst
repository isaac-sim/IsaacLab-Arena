Collision and Placement Validation
==================================

Spatial relations describe the intended arrangement, while collision
constraints and validation determine whether a candidate layout is acceptable.
Arena applies collision constraints and geometric validation during relation
placement. Physics validation is an optional, separate step.

What Collision Avoidance Covers
-------------------------------

Every pair of placed, non-anchor assets receives a no-overlap constraint unless
the pair has a support relationship such as ``On``. Eligible fixed scene assets
can act as passive obstacles. When a background uses mesh collision, its
collision geometry can also act as a passive obstacle, allowing placement to
avoid furniture and appliances within a composite scene.

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

``RelationSolverParams.clearance_m`` sets the pairwise collision margin.
``On.clearance_m`` is separate and controls only the vertical gap above a
support surface.

Start with ``BBOX``. Use ``MESH`` only when bounding boxes exclude space that
the actual objects can safely occupy. Collision mode changes overlap checking;
it does not change the meaning of relations such as ``On`` or ``NextTo``.

An individual asset can override the solver default, for example with
``background.collision_mode = CollisionMode.MESH``. When an asset has no
extractable collision mesh, Arena uses its bounding box as a proxy; if neither
asset in a pair provides a mesh, the pair falls back to AABB checking and logs
the fallback.

Background and Passive Obstacles
--------------------------------

Objects do not need placement relations to act as obstacles. A table can be an
anchor that supports an ``On`` relation, while a nearby appliance can remain a
fixed passive obstacle. This lets Arena place objects on a surface while
avoiding the rest of a complex scene.

Run the included kitchen example to see objects placed on a counter while
avoiding the background mesh:

.. code-block:: bash

   python \
     isaaclab_arena_examples/relations/isaac_sim_kitchen_background_collision_notebook.py \
     --viz kit \
     --view_steps 0

``--viz kit`` opens the viewer, and ``--view_steps 0`` keeps it open until you
close the application.

Validation Stages
-----------------

Arena can check placement at three stages:

1. **Geometric build-time validation** checks collisions and whether relations
   such as ``On``, ``NextTo``, and ``FaceTo`` are satisfied.
2. **Task-aware build-time validation** can reject layouts whose task targets
   are unreachable. When the cuRobo extension and a compatible embodiment
   configuration are available, the ``ik_reachable`` check runs after the
   inexpensive geometric checks. It currently evaluates reachability from the
   embodiment's configured initial base pose, not a relation-solved base pose.
3. **Physics validation** can test whether a completed layout remains stable
   after simulation starts.

The distinction matters because a geometrically valid object can still topple
or slide under physics, and geometric proximity alone does not guarantee that a
robot can reach a task target.

Tasks can mark an object for the optional reachability check with
``RequiresReachability``:

.. code-block:: python

   from isaaclab_arena.relations.relations import RequiresReachability

   target_object.add_relation(RequiresReachability())

Selecting Geometric Checks
--------------------------

By default, Arena runs all available build-time checks and requires each
executed check to pass. Environment graph specifications can select a subset:

.. code-block:: yaml

   placement_validators:
     enabled_checks:
       - no_overlap
       - on_relation
       - next_to
     required_checks:
       - no_overlap
       - on_relation

Built-in geometric check names are ``no_overlap``, ``on_relation``,
``next_to``, ``not_next_to``, and ``face_to``. Extension packages may register
additional checks such as ``ik_reachable``. Every required check must also be
enabled. Only registered and available checks execute; naming an unavailable
check does not make it gate placement.

Physics Stability
-----------------

Run the optional physics check on a registered environment with:

.. code-block:: bash

   python isaaclab_arena/scripts/run_placement_pool_validation.py \
     --num_envs 4 \
     pick_and_place_maple_table

When Placement Fails
--------------------

Check the intent before tuning solver parameters:

1. Confirm that the requested relations are compatible.
2. Check that anchors and support surfaces have the expected poses and bounds.
3. Check whether a bounding box is rejecting otherwise valid free space.
4. Inspect physics behavior if geometry passes but objects move or topple.

Only after identifying the failure should you adjust clearances, collision
mode, placement attempts, or solver settings. If fallback layouts are enabled
and no candidate passes every required check, Arena warns and may retain
best-loss layouts for the affected environments. Set
``allow_best_loss_fallbacks=False`` when every applied layout must pass; if no
valid layout can be produced, construction or pool refill raises
``RuntimeError``.
