Pooled Placement
================

``ArenaEnvBuilder`` manages relation placement through a pool of solved and
validated layouts. Users describe the intended arrangement; the builder
generates, stores, and applies layouts for the requested simulation
environments.

How the Pool Is Built
---------------------

``PooledObjectPlacer`` repeatedly uses :doc:`the placement solver <./solver>`
to generate candidate layouts. Validators rank the candidates, and the selected
layouts are stored per environment for construction and reset. Arena processes
candidates across environments together; users do not place each environment
separately.

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

For object variation, object-variant assignments remain fixed across resets
while layouts can change. Each layout is solved and checked using the assigned
variant's dimensions. See :ref:`Homogeneous and Heterogeneous Object Placement
<homogeneous-and-heterogeneous-placement>` for the distinction and a visual
example.

Within a complete environment graph, a heterogeneous object set and its
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

Layouts Across Resets
---------------------

The builder prepares a pool of validated layouts for each environment.
By default, resets consume layouts from these pools, producing scene variation
without solving every reset. If a pool becomes empty, placement generates more
layouts during the reset.

Set ``ObjectPlacerParams.resolve_on_reset=False`` when each environment should
reuse its initial solved layout. From the command line, use
``--no-resolve_on_reset``. Keep the default when layouts should vary across
episodes.

Reproducibility
---------------

Set ``placement_seed`` when placement must be repeatable. With the same scene,
seed, and environment count, Arena reproduces both object-set assignments and
layout generation.

This command produces a repeatable sequence of layouts across resets:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --placement_seed 42 \
     --num_steps 100 \
     pick_and_place_maple_table

The simulation seed controls other simulation randomness, while
``placement_seed`` controls placement. Set both for a fully repeatable run.

Recommended Starting Points
---------------------------

Start with the defaults. Most users only need to decide:

1. Use ``RigidObjectSet`` if object identity should vary across environments.
2. Set ``placement_seed`` if results must be repeatable.
3. Choose whether layouts should change on reset with ``resolve_on_reset``.
4. Choose a collision representation only when bounding boxes are too
   conservative. See :doc:`./collision_handling`.

Solver iteration counts, optimizer settings, pool sizes, and debugging options
are advanced tuning controls. Change them only after identifying a specific
placement failure or performance problem.

The complete configuration fields are defined by
`ObjectPlacerParams <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/relations/object_placer_params.py>`_
and
`RelationSolverParams <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/relations/relation_solver_params.py>`_.
Relation-specific parameters are defined in
`relations.py <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena/relations/relations.py>`_.
