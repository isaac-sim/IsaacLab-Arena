Placement Across Environments and Resets
========================================

``ArenaEnvBuilder`` manages relation placement as part of environment
compilation. Users describe the intended arrangement; the builder generates,
checks, and applies layouts for the requested simulation environments.

How a Layout Is Produced
------------------------

Placement follows four steps:

1. Collect objects, relations, anchors, and fixed obstacles.
2. Generate and optimize several candidate layouts.
3. Validate the candidates and prefer those that satisfy all checks.
4. Store selected layouts for construction and reset.

This process happens per environment, so parallel environments do not need to
share one pose configuration. Arena solves candidates for the environments in
batched passes; users do not place each environment separately.

.. important::

   Do not set an initial pose on an object whose pose is determined by
   placement relations. The builder owns that object's construction and reset
   poses. Anchors remain fixed and therefore still need a known pose.

Object and Layout Variation
---------------------------

Two independent kinds of variation are supported:

- **Layout variation:** the same objects receive different positions or
  orientations.
- **Object variation:** a ``RigidObjectSet`` selects different object variants
  across environments.

Homogeneous placement uses the same object geometry in every environment.
Heterogeneous placement uses the dimensions of each environment's selected
variant when solving and checking collisions. Both use the same placement
flow. See :ref:`Homogeneous and Heterogeneous Object Placement
<homogeneous-and-heterogeneous-placement>` for a visual example.

Within a complete environment graph, a heterogeneous object role and its
placement relation look like this:

.. code-block:: yaml

   object_sets:
     - id: fruit
       members:
         - apple_01_objaverse_robolab
         - orange_01_fruits_veggies_robolab
       random_choice: true

   relations:
     - kind: 'on'
       subject: fruit
       reference: maple_table

See
``isaaclab_arena_environments/droid_pick_fruit_into_bowl_maple_table.yaml``
for the complete runnable example, including the table, anchor, embodiment, and
task declarations.

Orientation Variation
---------------------

Enable random yaw initialization when objects may begin with arbitrary
horizontal orientations. In ``BBOX`` mode, collision checking uses the
conservative axis-aligned box enclosing each rotated object; ``MESH`` mode
follows the collision geometry more closely:

.. code-block:: python

   from isaaclab_arena.environments.isaaclab_arena_environment import (
       IsaacLabArenaEnvironment,
   )
   from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

   environment = IsaacLabArenaEnvironment(
       name="orientation_variation",
       scene=scene,
       placer_params=ObjectPlacerParams(random_yaw_init=True),
   )

Objects with a ``FaceTo`` relation use their relation-derived heading instead
of a random yaw.

Robot Embodiment Placement
--------------------------

Robot embodiments that provide placement bounds can use the same relation
workflow as objects. The builder includes an embodiment in placement when it
has relations. A typical mobile-manipulation layout places the robot on the
floor, offsets it from a work surface, and fixes its heading:

.. code-block:: yaml

   relations:
     - kind: is_anchor
       subject: floor
     - kind: is_anchor
       subject: right_counter_top
     - kind: 'on'
       subject: droid
       reference: floor
     - kind: next_to
       subject: droid
       reference: right_counter_top
       params:
         side: negative_y
         distance_m: 0.15
     - kind: rotate_around_solution
       subject: droid
       params:
         yaw_rad: 1.57

The shipped
``isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml``
contains the complete example. Its ``placement_bbox_stand_only: true`` option
uses only the Droid stand footprint for placement; the robot arm is excluded
from those placement bounds.

Embodiment placement establishes a geometric base pose. Whether the resulting
task targets are reachable is a separate validation concern. The current IK
validator uses the embodiment's configured initial base pose and does not
follow a relation-solved embodiment pose.

Layouts Across Resets
---------------------

The builder prepares a pool of checked layouts for each environment.
By default, resets consume layouts from these pools, producing scene variation
without solving every reset. If a pool becomes empty, placement refills it
synchronously.

Set ``resolve_on_reset=False`` when each environment should reuse its initial
solved layout. Keep it enabled when layouts should vary across episodes.

Reproducibility
---------------

Set ``placement_seed`` when placement must be repeatable. With the same scene,
seed, and environment count, Arena reproduces both object-set assignments and
layout generation.

For example, this command uses a repeatable placement and reuses the selected
layout on every reset:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --placement_seed 42 \
     --no-resolve_on_reset \
     --num_steps 100 \
     pick_and_place_maple_table

Omit ``--no-resolve_on_reset`` to vary layouts across resets while keeping the
sequence repeatable. The simulation seed controls the broader experiment;
``placement_seed`` controls placement. Set both for a fully repeatable run.

What Most Users Need to Choose
------------------------------

Start with the defaults and make only these decisions:

1. Use ``RigidObjectSet`` if object identity should vary across environments.
2. Set ``placement_seed`` if results must be repeatable.
3. Choose whether layouts should change on reset with ``resolve_on_reset``.
4. Choose a collision representation only when bounding boxes are too
   conservative. See :doc:`./collision_and_validation`.

Solver iteration counts, optimizer settings, pool sizes, and debugging options
are advanced tuning controls. Change them only after identifying a specific
placement failure or performance problem.

The complete configuration fields are defined by
`ObjectPlacerParams <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/relations/object_placer_params.py>`_
and
`RelationSolverParams <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/relations/relation_solver_params.py>`_.
Relation-specific parameters are defined in
`relations.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/relations/relations.py>`_.
