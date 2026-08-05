Object and Robot Placement
==========================

Placement determines the initial poses of objects and, when configured, the
robot embodiment in an Arena environment. For a fixed environment, you can set
every pose manually. This becomes brittle when assets or their dimensions
change.

Suppose you want a microwave on a table with a cracker box next to it. Manual
placement requires looking up the table height, measuring both objects, and
chaining those dimensions into world coordinates:

.. code-block:: python

   # Table surface: z = 0.42 m.
   # Microwave: 0.50 m wide and 0.30 m tall.
   # Cracker box: 0.064 m wide and 0.212 m tall.
   clearance_m = 0.01

   microwave_z = 0.42 + 0.30 / 2 + clearance_m
   microwave.set_initial_pose(Pose(position_xyz=(0.0, 0.0, microwave_z)))

   cracker_box_x = 0.50 / 2 + clearance_m + 0.064 / 2
   cracker_box_z = 0.42 + 0.212 / 2 + clearance_m
   cracker_box.set_initial_pose(
       Pose(position_xyz=(cracker_box_x, 0.0, cracker_box_z))
   )

If the table height or either object changes, every dependent coordinate must
be recalculated. Relation-based placement describes the intended arrangement
instead:

.. code-block:: python

   from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, Side

   table.add_relation(IsAnchor())
   microwave.add_relation(On(table))
   cracker_box.add_relation(On(table))
   cracker_box.add_relation(
       NextTo(microwave, side=Side.POSITIVE_X, distance_m=0.01)
   )

``ArenaEnvBuilder`` collects these relations, solves and validates candidate
layouts, and applies a selected layout during environment compilation.
Replacing an asset or changing its dimensions does not require manually
recalculating coordinates for related assets. See
`pick_and_place_maple_table_environment.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/pick_and_place_maple_table_environment.py>`_
for a complete Python environment definition.

.. figure:: ../../images/placement_pipeline.png
   :width: 100%
   :alt: ArenaEnvBuilder collects assets, roles, spatial relationships, and
      placement configuration for the relation solver.
   :align: center

   ``ArenaEnvBuilder`` turns an Arena environment definition into a placement
   problem, and the relation solver generates collision-aware candidate layouts.

.. figure:: ../../images/relations_highlevel.png
   :width: 100%
   :alt: High-level visualization of spatial relations between objects on a table
   :align: center

   Spatial relations describe the intended layout. Arena computes poses that
   satisfy the relations and collision constraints together.

When to Use Placement Relations
-------------------------------

Use explicit poses when an Arena environment has one fixed, known layout. Use
placement relations when you need one or more of the following:

- layouts that adapt to different assets or asset dimensions;
- diverse but reproducible layouts across environments and resets;
- automatic collision avoidance between placed assets and background assets or
  other fixed collision geometry that the solver treats as passive obstacles;
- a shared layout description across Arena environments; or
- geometric relation checks and optional physics stability testing.

See :doc:`object_placement/collision_handling` for passive obstacles and
collision representations.

.. _same-and-different-objects-across-environments:

Same and Different Objects Across Environments
----------------------------------------------

Across parallel environments, placement supports two patterns:

- The same registered objects can appear in every environment while their
  solved poses differ.
- Different registered objects can appear in each environment. Arena solves and
  validates each layout using the selected object's dimensions.

Both patterns use the same placement process. See
:doc:`object_placement/pooled_placement` for how layouts and object selections
behave across environments and resets.

.. figure:: ../../images/heterogeneous_placement.gif
   :width: 100%
   :alt: Different objects placed across four parallel environments
   :align: center

   The orange and banana stay the same in every environment, while object sets
   select bottles, cans, tools, and packages. Resets keep those selections while
   applying different pooled layouts.

Asset Roles During Placement
----------------------------

Objects, object sets, object references, and supported robot embodiments share
the ``PlaceableAsset`` interface. Their placement behavior depends on their
role:

.. list-table::
   :header-rows: 1
   :widths: 20 30 15 35

   * - Role
     - Purpose
     - Moved by solver
     - Initial pose
   * - Anchor
     - Fixed reference marked with ``IsAnchor()``
     - No
     - Required, or derived through an ``ObjectReference``
   * - Placed asset
     - Layout member with a positional relation
     - Yes
     - Do not set one; the builder owns its construction and reset poses
   * - Passive obstacle
     - Fixed collision geometry that placed assets must avoid
     - No
     - Required

Relations, Modifiers, and Checks
--------------------------------

Relations, modifiers, and checks serve different purposes:

- **Spatial relations** constrain positions or orientations, such as ``On``,
  ``NextTo``, or ``FaceTo``.
- **Placement modifiers** alter a solved pose after solving. The builder applies
  ``RotateAroundSolution`` to pooled layouts; ``RandomAroundSolution`` is
  limited to direct, single-environment placement.
- **Collision constraints** prevent placed assets from overlapping each other
  or passive background geometry.
- **Validation checks** evaluate whether candidate layouts meet required
  geometric conditions. Candidates that pass are preferred.

Try It Out
----------

.. _placement-viewer-display:

.. note::

   Commands using ``--viz kit`` require an available graphical display. In a
   remote or container session, configure display forwarding and set
   ``DISPLAY`` to the active X display, for example ``export DISPLAY=:1``.

The ``pick_and_place_maple_table`` environment provides a simple placement
example. Additional objects receive an ``On(table_reference)`` relation and are
placed without manual pose adjustments:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_steps 100 \
     pick_and_place_maple_table \
     --additional_table_objects cracker_box mug tomato_soup_can

Swap the registered object names to see placement adapt to different dimensions
and footprints.

.. figure:: ../../images/adaptive_object_placement.gif
   :width: 100%
   :alt: Relation-based placement adapting as objects are added
   :align: center

   Each run adds registered objects with different dimensions. The solver
   recomputes a collision-free layout without manual pose changes.

To see different fruit objects placed across four environments, run the
included object-set example:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_envs 4 \
     --num_steps 100 \
     --env_graph_spec_yaml \
       isaaclab_arena_environments/droid_pick_fruit_into_bowl_maple_table.yaml

Each environment selects one fruit and solves a layout using that fruit's
dimensions.

In both examples, ``--viz kit`` opens the viewer, ``zero_action`` lets you
inspect the environment without commanded motion, and ``--num_steps`` limits
the run.
The first command adds registered table objects with
``--additional_table_objects``. The second uses ``--num_envs`` to create four
environments from the graph passed to ``--env_graph_spec_yaml``.

Runner flags precede a registered environment name. Environment-specific flags,
such as ``--additional_table_objects``, follow it.

Learn More
----------

The following guides cover authoring relations, collision handling, the
placement solver, pooled placement, and validation:

.. toctree::
   :maxdepth: 1

   object_placement/relations
   object_placement/collision_handling
   object_placement/solver
   object_placement/pooled_placement
   object_placement/validation
