Placement
=========

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

``ArenaEnvBuilder`` collects these relations, solves candidate layouts, checks
them, and applies a selected layout while compiling the environment. Replacing
an asset or changing its dimensions does not require recalculating downstream
coordinates. See
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
- automatic collision avoidance between placed assets, discovered fixed
  obstacles, and configured background mesh geometry;
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

Both cases use the same per-environment placement path. Heterogeneous therefore
describes variation in the objects themselves, not merely different positions
of the same objects. See :doc:`object_placement/placement_lifecycle` for how
object and layout variation behave across environments and resets.

.. figure:: ../../images/heterogeneous_placement.gif
   :width: 100%
   :alt: Heterogeneous object variants placed across parallel environments
   :align: center

   Every environment contains the same banana and orange asset types, while
   four distractor roles use environment-specific variants. Resets preserve
   those assignments while varying object positions.

Mental Model
------------

Relation-based placement works with four roles:

- A **placeable asset** is an object, object set, object reference, or robot
  embodiment whose root pose can participate in placement.
- An **anchor** has ``IsAnchor()`` and remains fixed while other assets are
  placed relative to it. A tabletop or counter reference is a common anchor.
- An **optimized asset** has a positional relation and is moved by the solver.
- A **passive obstacle** has no placement relation but has fixed collision
  geometry that placed assets must avoid.

Relations, modifiers, and checks serve different purposes:

- **Spatial relations** constrain positions or orientations, such as ``On``,
  ``NextTo``, or ``FaceTo``.
- **Placement modifiers** adjust how a solved pose is applied in direct
  placement workflows. The default builder uses checked layout pools for reset
  variation.
- **Collision constraints** prevent placed assets from overlapping each other
  or passive scene geometry.
- **Validation checks** evaluate whether candidate layouts meet required
  geometric conditions. Candidates that pass are preferred.

Objects, object sets, object references, and supported robot embodiments share
the ``PlaceableAsset`` interface. Objects, object sets, and embodiments can be
solved for new poses; object references participate only as fixed anchors.

Try It Out
----------

The ``pick_and_place_maple_table`` environment is a convenient placement
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

The example flags serve the following purposes:

- ``--viz kit`` opens the interactive Kit viewer.
- ``--policy_type zero_action`` keeps policy actions at zero so you can inspect
  the generated scene.
- ``--num_steps`` controls how long the runner advances the simulation.
- ``--additional_table_objects`` adds registered objects to the selected
  environment; each receives an ``On(table_reference)`` relation.
- ``--num_envs`` selects the number of parallel environments.
- ``--env_graph_spec_yaml`` builds an environment from the specified graph
  rather than from a registered environment name.

Runner flags appear before a registered environment name. Environment-specific
flags, such as ``--additional_table_objects``, appear after it.

Learn More
----------

Use the following guides according to the question you are trying to answer:

.. toctree::
   :maxdepth: 1

   object_placement/relations
   object_placement/collision_and_validation
   object_placement/placement_lifecycle
