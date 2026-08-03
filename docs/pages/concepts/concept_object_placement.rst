Placement
=========

Placement determines the initial poses of objects in an Arena scene. For a
fixed scene, you can set every pose manually. For scenes whose assets,
dimensions, or number of environments may change, relation-based placement is
more adaptable: describe the intended layout and let Arena compute poses that
satisfy it.

For example, manual placement requires coordinates derived from the current
table and object dimensions:

.. code-block:: python

   microwave.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.58)))
   cracker_box.set_initial_pose(Pose(position_xyz=(0.292, 0.0, 0.536)))

The equivalent relation-based layout describes intent instead:

.. code-block:: python

   from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side

   table.add_relation(IsAnchor())
   microwave.add_relation(On(table))
   cracker_box.add_relation(On(table))
   cracker_box.add_relation(
       NextTo(microwave, side=Side.POSITIVE_X, distance_m=0.01)
   )

``ArenaEnvBuilder`` collects these relations, solves candidate layouts, checks
them, and applies a selected layout while compiling the environment. Replacing
an asset or changing its dimensions does not require recalculating downstream
coordinates.

.. figure:: ../../images/relations_highlevel.png
   :width: 100%
   :alt: High-level visualization of spatial relations between objects on a table
   :align: center

   Spatial relations describe the intended layout. Arena computes poses that
   satisfy the relations and collision constraints together.

When to Use Placement Relations
-------------------------------

Use explicit poses when a scene has one fixed, known configuration. Use
placement relations when you need one or more of the following:

- layouts that adapt to different assets or asset dimensions;
- diverse but reproducible layouts across environments and resets;
- automatic collision avoidance between placed assets, discovered fixed
  obstacles, and configured background mesh geometry;
- a shared layout description across scene variants; or
- placement checks such as relation validity and physical stability.

.. _homogeneous-and-heterogeneous-placement:

Homogeneous and Heterogeneous Placement
---------------------------------------

Placement supports both kinds of parallel environment:

- In **homogeneous placement**, an asset has the same geometry in every
  environment, although its solved pose can differ between environments.
- In **heterogeneous placement**, a ``RigidObjectSet`` can select a different
  object variant in each environment. Arena solves and validates each layout
  using the selected variant's dimensions.

Both cases use the same per-environment placement path. Heterogeneous therefore
describes variation in the objects themselves, not merely different positions
of the same objects.

.. figure:: ../../images/heterogeneous_placement.gif
   :width: 100%
   :alt: Heterogeneous object variants placed across parallel environments
   :align: center

   A fixed banana and orange appear with four heterogeneous distractors. Across
   resets, the assigned objects reappear while their positions and orientations
   vary.

Mental Model
------------

Relation-based placement works with four roles:

- A **placeable asset** is an object, object set, or object reference whose root
  pose can participate in placement.
- An **anchor** has ``IsAnchor()`` and remains fixed while other assets are
  placed relative to it. A tabletop or counter reference is a common anchor.
- An **optimized asset** has a positional relation and is moved by the solver.
- A **passive obstacle** has no placement relation but has fixed collision
  geometry that placed assets must avoid.

Relations, modifiers, and checks serve different purposes:

- **Spatial relations** constrain positions or orientations, such as ``On``,
  ``NextTo``, or ``FaceTo``.
- **Placement modifiers** change how a solved pose is applied, such as adding
  reset-time randomization.
- **Collision constraints** prevent placed assets from overlapping each other
  or passive scene geometry.
- **Validation checks** reject candidate layouts that do not meet required
  geometric or physical conditions.

The relation-placement ``PlaceableAsset`` model is separate from the
``Placeable`` task affordance, which describes whether an object can be placed
upright during a task.

This guide focuses on object placement. The shared ``PlaceableAsset``
abstraction leaves room for additional scene entities without changing the
placement model described here.

How Placement Fits Together
---------------------------

Environment compilation performs the following high-level steps:

1. Collect placeable assets, anchors, relations, and passive obstacles.
2. Generate candidate layouts for every simulation environment.
3. Optimize the candidate poses against spatial and collision constraints.
4. Validate and rank the candidates.
5. Store selected layouts and apply them during construction or reset.

The default builder path maintains a pool of layouts for each environment, so
early resets can receive fresh configurations from pre-solved candidates.
Exhausting an environment's queue triggers a synchronous refill during reset.
Seeds make both variant assignment and placement repeatable.

Try It Out
----------

The ``pick_and_place_maple_table`` environment is a convenient placement
example. Additional objects receive an ``On(table_reference)`` relation and are
placed without manual pose adjustments:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 100 \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --hdr home_office_robolab \
     --additional_table_objects cracker_box mug tomato_soup_can

Swap the registered object names to see placement adapt to different dimensions
and footprints.

Learn More
----------

Use the following guides according to the question you are trying to answer:

.. toctree::
   :maxdepth: 1

   object_placement/relations
   object_placement/collision_and_validation
   object_placement/placement_lifecycle
