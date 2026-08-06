Object Placement
================

The Motivation
--------------

The traditional approach to placing objects in a simulation environment is to set each object's
pose manually. Suppose you want to place a microwave on a table, with a cracker box sitting next to it.
You look up the table surface height, measure the object dimensions, and compute the coordinates by hand:

.. code-block:: python

   # Table surface is at z = 0.42 m in world frame.
   # Microwave is 0.50 m wide (x) and 0.30 m tall (z).
   # Cracker box is 0.064 m wide (x) and 0.212 m tall (z).

   CLEARANCE = 0.01  # m

   microwave_z = 0.42 + 0.30 / 2 + CLEARANCE             # = 0.58
   microwave.set_initial_pose(Pose(position_xyz=(0.0, 0.0, microwave_z)))

   cracker_box_x = 0.0 + 0.50 / 2 + CLEARANCE + 0.064 / 2   # = 0.292
   cracker_box_z = 0.42 + 0.212 / 2 + CLEARANCE              # = 0.536
   cracker_box.set_initial_pose(Pose(position_xyz=(cracker_box_x, 0.0, cracker_box_z)))

This works, but it is brittle. If you swap the microwave for a larger appliance, every downstream
coordinate that depended on its size must be recalculated. If you use a different table, the surface
height changes and all Z values are wrong. If you add an apple next to the cracker box, you need to
chain yet another calculation on top of the previous ones.

The relations system eliminates this problem. Instead of computing coordinates, you declare *constraints*:

.. code-block:: python

   from isaaclab_arena.relations.relations import IsAnchor, On, NextTo, Side

   packing_table.add_relation(IsAnchor())

   microwave.add_relation(On(packing_table))

   cracker_box.add_relation(On(packing_table))
   cracker_box.add_relation(NextTo(microwave, side=Side.POSITIVE_X, distance_m=0.01))

   apple.add_relation(On(packing_table))
   apple.add_relation(NextTo(cracker_box, side=Side.POSITIVE_X, distance_m=0.01))

A constraint solver reads the actual bounding boxes of each object and computes positions that satisfy all
the constraints together. Swap the microwave for any other object and the solver re-derives everything
automatically — the cracker box and apple will still end up next to it at the right height, regardless
of the new object's dimensions.

.. figure:: ../../images/relations_highlevel.png
   :width: 100%
   :alt: High-level visualization of spatial relations between objects on a table
   :align: center

   Example relations: a microwave and a cracker box both with ``On(packing_table)``,
   and an apple with ``NextTo(microwave)``. The solver reads these constraints and
   computes a valid, collision-free layout automatically.

Try It Out
----------

The ``pick_and_place_maple_table`` environment is a good place to experiment with object placement.
By default it places one pick-up object and one destination on the table. You can add any number of
extra objects via ``--additional_table_objects`` — each one receives an ``On(table_reference)``
relation and the solver places them all without collisions:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 100 \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --hdr home_office_robolab \
     --additional_table_objects cracker_box mug tomato_soup_can

Swap any of those names for other registered objects to see the solver automatically adapt the
layout to the new sizes and footprints.

.. figure:: ../../images/object_placement_variety.gif
   :width: 100%
   :alt: Object placement example with multiple objects on a table
   :align: center

   Five sequential runs on the maple table environment, each adding more objects (0 to 5 extra).
   The solver automatically computes a valid, collision-free layout for each configuration —
   no manual pose adjustments needed when the object set changes.

Core Concepts
-------------

The three core concepts you work with directly are **Spatial Relations**, **Anchors**, and
**Placement Modifiers**. Under the hood, ``ArenaEnvBuilder`` collects these from your scene,
runs the solver, and applies the results automatically — you rarely need to interact with the
solver directly.

Relations are attached to objects via ``add_relation()``:

.. code-block:: python

   from isaaclab_arena.relations.relations import IsAnchor, On, AtPosition

   table_reference.add_relation(IsAnchor())
   cracker_box.add_relation(On(table_reference))
   cracker_box.add_relation(AtPosition(x=0.4, y=0.0))

Multiple relations on the same object are all satisfied simultaneously.

Spatial Relations
~~~~~~~~~~~~~~~~~

Spatial relations constrain where an object may be relative to a parent object or world position.

**On**

Places a child object on top of a parent object. The child's footprint must lie within the
parent's horizontal extents, and the child's bottom surface rests at the top of the parent
with an optional clearance gap:

.. code-block:: python

   from isaaclab_arena.relations.relations import On

   cracker_box.add_relation(On(table_reference))

Parameters:

- ``parent``: The reference object to place on top of
- ``clearance_m`` (default ``0.01``): Extra gap above the parent's top surface, in meters
- ``relation_loss_weight`` (default ``1.0``): Weight in the solver's loss function

.. note::

   ``On`` positions the child at the top of the parent's **axis-aligned bounding box**, not
   its physical surface. For concave objects such as bowls, the bounding box top is at the
   rim, so the child may end up inside the bowl rather than resting on it. Avoid using
   ``On`` with concave objects as the parent.

**NextTo**

Places a child object adjacent to a parent object at a specified distance along a given axis:

.. code-block:: python

   from isaaclab_arena.relations.relations import NextTo, Side

   mug.add_relation(On(table_reference))
   mug.add_relation(NextTo(cracker_box, side=Side.POSITIVE_X, distance_m=0.1))

The ``Side`` enum selects the direction: ``POSITIVE_X``, ``NEGATIVE_X``, ``POSITIVE_Y``, or ``NEGATIVE_Y``.

The ``cross_position_ratio`` parameter (default ``0.0``) controls alignment along the perpendicular axis:
``-1.0`` aligns with the parent's near edge, ``0.0`` centers the child, and ``1.0`` aligns with
the parent's far edge.

Parameters:

- ``parent``: The reference object to place next to
- ``side``: Which side to place on (``Side`` enum)
- ``distance_m`` (default ``0.05``): Distance from the parent's edge, in meters
- ``cross_position_ratio`` (default ``0.0``): Alignment along the perpendicular axis, in ``[-1, 1]``
- ``relation_loss_weight`` (default ``1.0``): Weight in the solver's loss function

**AtPosition**

Pins an object to specific world-frame coordinates. You can constrain any subset of axes —
unconstrained axes are determined by other relations (for example, ``On`` controls the Z axis):

.. code-block:: python

   from isaaclab_arena.relations.relations import AtPosition

   cracker_box.add_relation(On(table_reference))
   cracker_box.add_relation(AtPosition(x=0.4, y=0.0))  # z left free for On to control

Parameters:

- ``x``, ``y``, ``z``: Target world coordinates (any can be ``None`` to leave unconstrained)
- ``relation_loss_weight`` (default ``1.0``): Weight in the solver's loss function

**PositionLimitsBox and PositionLimitsCylindrical**

``PositionLimitsBox`` constrains an object's world-frame placement origin with
axis-aligned ``x_min``/``x_max``, ``y_min``/``y_max``, and ``z_min``/``z_max``
bounds. ``PositionLimitsCylindrical`` constrains the origin's world-XY distance
from ``(center_x, center_y)``: ``radius_max`` keeps it inside a cylinder,
``radius_min`` excludes the inner cylinder, and both bounds create a cylindrical
annulus. These relations constrain the placement origin only; they do not account
for the object's bounding-box footprint.

Attach both relations when an object needs the intersection of box and cylindrical
limits. For example, this keeps an object's origin inside a rectangular tabletop
region and within 0.25 m of a world-XY point:

.. code-block:: python

   from isaaclab_arena.relations.relations import (
       PositionLimitsBox,
       PositionLimitsCylindrical,
   )

   object.add_relation(
       PositionLimitsBox(x_min=0.2, x_max=0.6, y_min=-0.4, y_max=0.0)
   )
   object.add_relation(
       PositionLimitsCylindrical(center_x=0.4, center_y=-0.2, radius_max=0.25)
   )

The equivalent YAML relations are:

.. code-block:: yaml

   relations:
     - kind: position_limits_box
       subject: object
       params:
         x_min: 0.2
         x_max: 0.6
         y_min: -0.4
         y_max: 0.0
     - kind: position_limits_cylindrical
       subject: object
       params:
         center_x: 0.4
         center_y: -0.2
         radius_max: 0.25

Parameters:

- ``PositionLimitsBox`` requires at least one optional world-coordinate axis bound:
  ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``, or ``z_max``
- ``PositionLimitsCylindrical`` requires ``center_x`` and ``center_y`` plus at
  least one nonnegative ``radius_min`` or ``radius_max``; when both are present,
  ``radius_min`` must be strictly less than ``radius_max``
- Both relations accept ``relation_loss_weight`` (default ``1.0``)

For backward compatibility, ``PositionLimits`` and YAML kind ``position_limits``
remain aliases for ``PositionLimitsBox``.

**ClutteredOn**

Places a group of objects as a settled pile on a support surface. Unlike the relations
above, members are not positioned by the solver: they are released above the support and
allowed to fall, and where they come to rest is the placement.

.. code-block:: python

   table.add_relation(IsAnchor())
   for tool in (mug, cracker_box, power_drill):
       tool.add_relation(ClutteredOn(table, group="tools", spread=0.8))

The equivalent YAML. Each member declares its own relation, and the ones naming the same
``reference`` and ``group`` are poured together as a single pile:

.. code-block:: yaml

   relations:
     - kind: cluttered_on
       subject: mug
       reference: table
       params: {group: tools, spread: 0.8}
     - kind: cluttered_on
       subject: cracker_box
       reference: table
       params: {group: tools, spread: 0.8}
     - kind: cluttered_on
       subject: power_drill
       reference: table
       params: {group: tools, spread: 0.8}

A support may hold several groups, and each is poured separately. Note that ``spread``
repeats identically across the three: it describes the region they share, so members that
disagree on it are rejected.

Parameters describing the pile, which every member must declare identically:

- ``group`` (default ``"clutter"``) — ties members of one pile together
- ``spread`` (default ``1.0``) — fraction of the support footprint to use, shrunk about
  its centre; must be in ``(0, 1]``. A pile is poured into one region, so members
  disagreeing on it is rejected.
- ``drop_order`` (default ``as_listed``) — which members end up underneath; see below

Parameters describing a single member, which may differ between them:

- ``gap_m`` (default ``0.03``) — vertical gap this object leaves above whatever it is
  released over
- ``clearance_m`` (default ``0.01``) — height above the surface at which this object starts
- ``random_yaw`` (default ``True``) — sample a yaw for this object before dropping. Turn it
  off for one member to drop it axis-aligned while the rest are turned.

Only the yaw is sampled. Any other release orientation is authored, by giving the member a
``RotateAroundSolution`` marker that the sampled yaw is then composed onto:

.. code-block:: python

    mug.add_relation(ClutteredOn(table, group="tools"))
    mug.add_relation(RotateAroundSolution(roll_rad=math.pi))  # released upside down

The drop planner refits the member's bounding box to that rotation, so a flipped object still
starts with its true lowest point at ``clearance_m`` above the surface. What cannot be expressed
is a *distribution* over orientations: every member carrying a marker is released at exactly that
rotation, in every environment and on every reset, and only its yaw varies. The marker is per
member, so half a pile can be marked upside down and half left alone; what cannot be expressed is
sampling that split per layout.

Resting orientations are another matter — the pile tumbles as it settles, so members come to rest
at whatever roll and pitch physics gives them, which is why a layout records full quaternions
rather than a yaw.

``drop_order`` decides the sequence members are planned in, which is how you choose which of
them end up underneath -- each member is stacked above whatever earlier ones its footprint
overlaps:

- ``as_listed`` (default) — the order the members appear in the asset list, so listing an object
  first puts it at the bottom
- ``flattest_first`` — shortest first, so flat objects reach the surface before it gets lumpy
  and lie flat rather than coming to rest on an edge
- ``shuffle`` — randomised per layout, so no one member sits at the bottom of every pile

It is pile-wide, so every member of a group must declare the same value. And it decides the
planned stack only: once physics runs, a pile landing on the first object can still shove it
aside, so it is a strong influence rather than a guarantee.

Not yet reachable from ``cluttered_on``, and worth knowing before designing around it:

- **No pile offset.** A group's region is always concentric with its support, so two groups on
  one support are poured into the same rectangle. They avoid each other's footprints, but
  "one pile on the left, one on the right" cannot be expressed; use two supports instead.
- **Orientation beyond yaw is authored, not sampled**, as above.

Because a pile is produced by simulation rather than optimisation, it differs from the
other relations in ways worth knowing:

- **The support must be provably immovable.** Its spawner must set
  ``rigid_props.kinematic_enabled``. ``IsAnchor`` is not enough -- it fixes an asset for the
  solver's arithmetic and says nothing about simulation -- and neither is an absent
  ``rigid_props``, since a background asset can spawn a rigid body through ``spawn_cfg_addon``.
  A pile is captured relative to where its support stood while settling, so a support that
  physics can shove replays the pile against a surface that has moved out from under it.
- **The support must be square to the world axes.** A rotation off a quarter turn is refused:
  the pour region and the region a settled pile is judged against are both axis-aligned boxes,
  so neither would describe a turned support's surface.
- **Nothing may be related to a member, and a member may declare little.** A relation naming a
  member as its parent is refused, and a member may carry only ``cluttered_on`` and a single
  ``RotateAroundSolution``. Members are held out of the solve, so anything else would be
  discarded -- a parent reference deep in the solver, a carried relation in silence.
- **A support the layout does not place must declare a concrete Pose.** A ``PoseRange`` or
  ``PosePerEnv`` offers no single surface to pour onto. A solver-placed support may declare
  either, since its pose comes from the layout rather than the declaration.
- **The support must have a readable USD.** The drop region is derived from its bounding
  box, so a procedurally spawned support without geometry cannot be used.
- **A placement seed is required.** Clutter refuses to pour without one, since a pile that
  cannot be reproduced defeats the purpose of seeding a layout.
- **Layouts must resolve on reset.** A pile is settled after the environment is built (the
  default; ``settle_clutter_on_build`` turns it off for callers that settle themselves) and its
  resting poses are written back into the pool each reset draws from. Static placement keeps
  no such pool, so the combination is rejected rather than spawning a pile in mid-air.
- **Members never reach the solver or the build-time checks.** A pour assigns their poses,
  so any pose the optimiser gave them would be discarded; leaving them in would make them
  phantom obstacles that push the genuinely constrained objects around. They are therefore
  held out of the solver and out of the generic build-time validators, including overlap. The
  clutter-specific checks -- what a member may declare, whether the pile settled, whether it
  stayed on its support -- do run.
- **A pile that always spills is a configuration error.** Layouts are settled once and the ones
  whose members left the support are discarded. An environment left with none fails at build
  time rather than running episodes against a pile lying beside its support: reduce the members,
  lower the group's ``spread``, or raise ``clutter_containment_margin_m``.
- **The pour avoids what is already on the surface instead.** Objects the solver placed on
  the same support, and members of earlier groups, are treated as occupied footprints and
  clutter is released above them rather than into them. A settled pile is in contact by
  definition, so checking it for overlap afterwards would reject correct arrangements.

Anchors
~~~~~~~

Every scene must have at least one **anchor** — an object whose position is fixed during
optimization and serves as the reference frame for all other objects. Anchors are marked
with ``IsAnchor()`` and are not moved by the solver.

Any object in the scene can be an anchor. A common case is using an ``ObjectReference``
that points to a specific surface within a background asset, such as a tabletop or counter,
whose pose is derived automatically from the USD scene:

.. code-block:: python

   from isaaclab_arena.assets.object_reference import ObjectReference
   from isaaclab_arena.relations.relations import IsAnchor

   table_reference = ObjectReference(
       name="table",
       prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
       parent_asset=table_background,
   )
   table_reference.add_relation(IsAnchor())

You can have multiple independent anchors in a scene — for example, a table and a separate
counter. Objects are then placed relative to whichever anchor they reference.

Placement Modifiers
~~~~~~~~~~~~~~~~~~~

Placement modifiers are attached alongside spatial relations and affect how the solved position
is applied to the object. They are not processed by the solver itself.

**RandomAroundSolution**

After solving, converts the fixed solved pose into a ``PoseRange`` so the object's position
is randomized within a box around the solution at each environment reset:

.. code-block:: python

   from isaaclab_arena.relations.relations import On, RandomAroundSolution

   cracker_box.add_relation(On(table_reference))
   cracker_box.add_relation(RandomAroundSolution(x_half_m=0.1, y_half_m=0.1))

At each reset, the object spawns at a random position uniformly sampled within
``[solved_x ± x_half_m, solved_y ± y_half_m, solved_z ± z_half_m]``.

The solver only validates the center of this range — the solved position itself.
No placement checking is performed at reset time, so positions near the edges of the
range may be physically invalid (e.g. the object partially off the surface, or
overlapping a neighbour).

The safest approach is to combine ``RandomAroundSolution`` with ``AtPosition``.
Because ``AtPosition`` pins the object to a known world coordinate, you can reason
about the available margin purely from the surface geometry and the object's bounding
box — no need to inspect solver output. For example, if a surface spans ``x ∈ [0.0, 1.0]``
and you place the object at ``x = 0.5`` with a footprint of ``0.1 m``, there is
``0.5 - 0.05 = 0.45 m`` of margin on each side, so ``x_half_m = 0.1`` is safely
within bounds:

.. code-block:: python

   obj.add_relation(On(table_reference))
   obj.add_relation(AtPosition(x=0.5, y=0.0))
   obj.add_relation(RandomAroundSolution(x_half_m=0.1, y_half_m=0.05))

Parameters: ``x_half_m``, ``y_half_m``, ``z_half_m`` — half-extents in meters (default ``0.0``);
``roll_half_rad``, ``pitch_half_rad``, ``yaw_half_rad`` — half-extents in radians (default ``0.0``).

**RotateAroundSolution**

Applies an explicit rotation (in Euler angles) on top of the solver's solution:

.. code-block:: python

   import math
   from isaaclab_arena.relations.relations import On, RotateAroundSolution

   cracker_box.add_relation(On(table_reference))
   cracker_box.add_relation(RotateAroundSolution(yaw_rad=math.pi / 4))  # 45° rotation

When combined with ``RandomAroundSolution``, the rotation is applied to the PoseRange center.

Parameters: ``roll_rad``, ``pitch_rad``, ``yaw_rad`` — rotation in radians (default ``0.0``).

How the Solver Works
--------------------

The solver (``RelationSolver``) treats the X, Y, Z position of each non-anchor object as
learnable parameters and minimizes a total loss derived from all attached spatial relations.
Each relation type contributes a differentiable loss component:

- **On**: Band loss on X/Y (child footprint within parent), point loss on Z (child bottom at parent top)
- **NextTo**: Half-plane loss (correct side), band loss (perpendicular alignment), distance loss (target gap)
- **AtPosition**: Point loss on each constrained axis
- **PositionLimitsBox**: Band or one-sided loss for each axis-aligned rectangular bound
- **PositionLimitsCylindrical**: Band or one-sided loss on the world-XY radius from its center

Losses use linear ReLU-style functions (zero inside the valid region, linearly growing outside),
which provide constant gradients that work well with the Adam optimizer.

``ObjectPlacer`` wraps the solver with retry logic and geometric validation:

1. Random initial positions are generated within a bounding box centered on the anchor
2. The solver runs up to ``max_iters`` gradient steps (default 600)
3. The result is validated: no pair of objects may overlap, and all ``On`` constraints must be satisfied
4. If validation fails, the process repeats up to ``max_placement_attempts`` times (default 5)

ArenaEnvBuilder Integration
----------------------------

When you build an environment with ``ArenaEnvBuilder``, object placement runs automatically.
You do not need to call ``ObjectPlacer`` directly — just attach relations to your objects
and add them to the scene:

.. code-block:: python

   from isaaclab_arena.assets.object_reference import ObjectReference
   from isaaclab_arena.relations.relations import IsAnchor, On, AtPosition
   from isaaclab_arena.scene.scene import Scene
   from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
   from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg

   # Define anchor
   table_reference = ObjectReference(
       name="table",
       prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
       parent_asset=table_background,
   )
   table_reference.add_relation(IsAnchor())

   # Define placeable objects
   cracker_box = asset_registry.get_asset_by_name("cracker_box")()
   cracker_box.add_relation(On(table_reference))
   cracker_box.add_relation(AtPosition(x=0.0, y=0.0))

   mug = asset_registry.get_asset_by_name("mug")()
   mug.add_relation(On(table_reference))

   # Build scene and environment
   scene = Scene(assets=[table_background, table_reference, cracker_box, mug, light])
   env = IsaacLabArenaEnvironment(name="demo", scene=scene, ...)

   env_builder = ArenaEnvBuilder(env, ArenaEnvBuilderCfg())
   gym_env = env_builder.make_registered()

The builder automatically:

- Collects all objects with at least one relation from the scene
- Enforces no-overlap between all object pairs automatically (built-in solver behavior, controlled by ``clearance_m``)
- Creates an ``ObjectPlacer`` and runs placement
- Sets the solved poses on the objects before handing them to Isaac Lab

Usage Patterns
--------------

**Single object on a surface**

The most common pattern: an anchor marks a surface, and one or more objects are placed on it.

.. code-block:: python

   table_reference.add_relation(IsAnchor())

   pick_object.add_relation(On(table_reference))
   destination.add_relation(On(table_reference))

**Precise XY placement**

Combine ``On`` with ``AtPosition`` when you need the object at a known world location:

.. code-block:: python

   pick_object.add_relation(On(counter_top))
   pick_object.add_relation(AtPosition(x=0.4, y=0.0))

**Multiple objects arranged side-by-side**

Use ``NextTo`` to arrange objects in a row:

.. code-block:: python

   cracker_box.add_relation(On(table_reference))
   cracker_box.add_relation(AtPosition(x=0.0, y=0.0))

   mug.add_relation(On(table_reference))
   mug.add_relation(NextTo(cracker_box, side=Side.POSITIVE_X, distance_m=0.1))

   tomato_soup_can.add_relation(On(table_reference))
   tomato_soup_can.add_relation(NextTo(cracker_box, side=Side.NEGATIVE_X, distance_m=0.1))

**Episode-level randomization**

Add ``RandomAroundSolution`` to vary object positions across resets while respecting the spatial
constraints established by the solver:

.. code-block:: python

   pick_object.add_relation(On(counter_top))
   pick_object.add_relation(AtPosition(x=4.05, y=-0.58))
   pick_object.add_relation(RotateAroundSolution(yaw_rad=math.pi / 2))
   pick_object.add_relation(RandomAroundSolution(x_half_m=0.05, y_half_m=0.05))

**Multiple anchors**

Scenes with multiple fixed reference surfaces use one ``IsAnchor`` per surface:

.. code-block:: python

   table.add_relation(IsAnchor())
   counter.add_relation(IsAnchor())

   mug.add_relation(On(table))
   bowl.add_relation(On(counter))

Configuration
-------------

``ObjectPlacerParams`` controls the placer's behavior:

.. code-block:: python

   from isaaclab_arena.relations.object_placer import ObjectPlacer
   from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

   placer = ObjectPlacer(params=ObjectPlacerParams(
       max_placement_attempts=10,
       placement_seed=42,
       verbose=True,
   ))

Key parameters:

- ``max_placement_attempts`` (default ``10``): Number of solver restarts before giving up
- ``placement_seed`` (default ``None``): Random seed for reproducible placements
- ``on_relation_z_tolerance_m`` (default ``5e-3``): Tolerance for Z validation of ``On`` relations

The underlying solver can also be tuned via ``RelationSolverParams`` nested inside ``ObjectPlacerParams``:

.. code-block:: python

   from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

   placer = ObjectPlacer(params=ObjectPlacerParams(
       solver_params=RelationSolverParams(
           max_iters=1000,
           lr=0.005,
           convergence_threshold=1e-5,
       )
   ))

Key solver parameters:

- ``max_iters`` (default ``600``): Maximum optimization iterations per attempt
- ``lr`` (default ``0.01``): Adam optimizer learning rate
- ``convergence_threshold`` (default ``1e-4``): Stop early when loss falls below this value
