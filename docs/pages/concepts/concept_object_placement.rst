Object and Robot Placement
==========================

Placement determines the initial poses of objects and, when configured, the
robot embodiment in an Arena scene. For a fixed scene, you can set every pose
manually. For scenes whose assets, dimensions, or number of environments may
change, relation-based placement is more adaptable: describe the intended
layout and let Arena compute poses that satisfy it.

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

``ArenaEnvBuilder`` collects these relations, solves and validates candidate
layouts, and applies a selected layout during environment compilation.
Replacing an asset or changing its dimensions does not require manually
recalculating coordinates for related assets. See
`pick_and_place_maple_table_environment.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/pick_and_place_maple_table_environment.py>`_
for a complete Python environment definition.

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
- automatic collision avoidance between placed assets and fixed scene geometry
  that Arena can use as passive obstacles;
- a shared layout description across scene variants; or
- geometric relation checks and optional physics stability testing.

.. _homogeneous-and-heterogeneous-placement:

Homogeneous and Heterogeneous Object Placement
----------------------------------------------

Across parallel environments, placement supports two asset patterns:

- In **homogeneous placement**, an asset has the same geometry in every
  environment, although its solved pose can differ between environments.
- In **heterogeneous placement**, a ``RigidObjectSet`` can select a different
  object variant in each environment. Arena solves and validates each layout
  using the selected variant's dimensions.

Both cases follow the same placement process. Here, heterogeneous describes
variation in the objects themselves, not merely different positions of the
same objects. See
:doc:`object_placement/pooled_placement` for how object and layout variation
behave across environments and resets.

.. figure:: ../../images/heterogeneous_placement.gif
   :width: 100%
   :alt: Heterogeneous object variants placed across parallel environments
   :align: center

   Some object roles use the same asset in every environment, while others use
   environment-specific variants. Resets preserve those assignments while
   varying object positions.

Core Placement Concepts
-----------------------

Objects, object sets, object references, and supported robot embodiments share
the ``PlaceableAsset`` interface. Objects, object sets, and embodiments can be
solved for new poses; object references participate only as fixed anchors.

During placement, assets have three roles:

- An **anchor** has ``IsAnchor()`` and remains fixed while other assets are
  placed relative to it.
- A **placed asset** has a positional relation and is moved by the solver.
- A **passive obstacle** has no placement relation but has fixed collision
  geometry that placed assets must avoid.

Relations, modifiers, and checks serve different purposes:

- **Spatial relations** constrain positions or orientations, such as ``On``,
  ``NextTo``, or ``FaceTo``.
- **Placement modifiers** alter a solved pose after solving. The builder applies
  ``RotateAroundSolution`` to pooled layouts; ``RandomAroundSolution`` is
  limited to direct, single-environment placement.
- **Collision constraints** prevent placed assets from overlapping each other
  or passive scene geometry.
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

To see heterogeneous placement across four environments, run the included
fruit object-set example:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_envs 4 \
     --num_steps 100 \
     --env_graph_spec_yaml \
       isaaclab_arena_environments/droid_pick_fruit_into_bowl_maple_table.yaml

Each environment assigns one fruit variant and solves a layout using that
variant's dimensions.

In both examples, ``--viz kit`` opens the viewer, ``zero_action`` lets you
inspect the scene without commanded motion, and ``--num_steps`` limits the run.
The first command adds registered table objects with
``--additional_table_objects``. The second uses ``--num_envs`` to create four
environments from the graph passed to ``--env_graph_spec_yaml``.

Runner flags precede a registered environment name. Environment-specific flags,
such as ``--additional_table_objects``, follow it.

Learn More
----------

The following guides cover the placement solver, pooled placement, validation,
and what to check when placement fails:

.. toctree::
   :maxdepth: 1

   object_placement/solver
   object_placement/pooled_placement
   object_placement/validation
