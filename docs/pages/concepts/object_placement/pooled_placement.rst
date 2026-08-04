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

.. figure:: ../../../images/pooled_placement_flow.png
   :width: 100%
   :alt: Placement relations are optimized in batches into a solution pool that
      supplies layouts to parallel Isaac Lab environments.
   :align: center

   The optimizer generates more solutions than there are environments. During
   evaluation, each environment draws a placement solution from the pool.

Different Layouts and Objects
-----------------------------

Arena can independently change a layout and the objects in it:

- **Different layouts:** the same objects receive different positions or
  orientations across environments or resets.
- **Different objects:** a ``RigidObjectSet`` selects one member for each
  environment.

Object selections remain fixed across resets while layouts can change. Each
layout is solved and checked using the selected object's dimensions. See
:ref:`Same and Different Objects Across Environments
<same-and-different-objects-across-environments>` for a visual example.

.. figure:: ../../../images/same_objects_different_layouts.gif
   :width: 100%
   :alt: The same objects placed in different layouts across four environments
   :align: center

   The same six objects appear in every environment, but each environment
   receives a different solved layout.

.. figure:: ../../../images/heterogeneous_placement.gif
   :width: 100%
   :alt: Different objects placed across four parallel environments
   :align: center

   The orange and banana stay the same in every environment. Object sets select
   bottles, cans, tools, and packages, and the solver uses each selected
   object's dimensions.

Within a complete environment graph, an object set and its placement relation
look like this:

.. tab-set::

   .. tab-item:: Python
      :selected:

      .. code-block:: python

         from isaaclab_arena.assets.object_set import RigidObjectSet
         from isaaclab_arena.relations.relations import IsAnchor, On

         maple_table = asset_registry.get_asset_by_name(
             "maple_table_robolab"
         )()
         maple_table.add_relation(IsAnchor())

         fruit = RigidObjectSet(
             name="fruit",
             objects=[
                 asset_registry.get_asset_by_name(
                     "apple_01_objaverse_robolab"
                 )(),
                 asset_registry.get_asset_by_name(
                     "orange_01_fruits_veggies_robolab"
                 )(),
             ],
             random_choice=True,
         )
         fruit.add_relation(On(maple_table))

   .. tab-item:: YAML

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

Run the complete graph example, including its table, anchor, embodiment, and
task declarations, with:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_envs 4 \
     --num_steps 100 \
     --env_graph_spec_yaml \
       isaaclab_arena_environments/droid_pick_fruit_into_bowl_maple_table.yaml

.. important::

   Do not set an initial pose on an object whose pose is determined by
   placement relations. The builder owns that object's construction and reset
   poses. Anchors remain fixed and therefore still need a known pose.

Layouts Across Resets
---------------------

The builder prepares a pool of validated layouts for each environment.
By default, resets consume layouts from these pools, producing different layouts
without solving every reset. If a pool becomes empty, placement generates more
layouts during the reset.

Set ``ObjectPlacerParams.resolve_on_reset=False`` when each environment should
reuse its initial solved layout. From the command line, use
``--no-resolve_on_reset``. Keep the default when layouts should vary across
episodes.

Reproducibility
---------------

Set ``placement_seed`` when placement must be repeatable. With the same Arena
environment definition, placement seed, and environment count, layout
generation is repeatable.

This command produces a repeatable sequence of layouts across resets:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --seed 42 \
     --placement_seed 42 \
     --num_steps 100 \
     pick_and_place_maple_table

``--seed`` controls general simulation randomness, while ``--placement_seed``
controls placement-specific randomness.

Recommended Starting Points
---------------------------

Start with the defaults. Most users only need to decide:

1. Use ``RigidObjectSet`` if environments should contain different objects.
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
