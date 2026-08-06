Pooled Placement
================

``ArenaEnvBuilder`` manages relational placement through a logical pool of solved
layouts partitioned by environment. Users describe the intended arrangement;
the builder generates, stores, and applies layouts for the requested simulation
environments.

How the Pool Is Built
---------------------

``PooledObjectPlacer`` repeatedly uses :doc:`the placement solver <./solver>`
to generate candidate layouts. Inexpensive validators check every candidate;
expensive validators check candidates that pass the required inexpensive
checks. The placer ranks candidates by required-check failures, optional-check
failures, and then solver loss. Selected layouts are stored in the pool
partition for their environment. Arena processes candidates across environments
together; users do not place each environment separately.

.. figure:: ../../../images/pooled_placement_flow.png
   :width: 100%
   :alt: Placement relations are optimized in batches into a solution pool
      partitioned across parallel Isaac Lab environments.
   :align: center

   The diagram shows the logical solution pool. Arena keeps its layouts
   partitioned by environment, and each reset draws from the corresponding
   partition.

.. note::

   Layouts that pass every required check are preferred. By default,
   ``allow_best_loss_fallbacks=True`` permits the final refill batch to store
   one or more highest-ranked candidates for an environment when none passes
   every required check. Arena logs a warning when it uses such a fallback. Set
   this option to ``False`` when placement must fail instead.

Object Identity and Layouts
---------------------------

Object identity and layout selection are independent. Homogeneous placement
uses the same registered objects while their layouts can differ. Heterogeneous
placement keeps each object's per-environment member assignment fixed while
layouts can change across resets. Every layout is solved and checked using the
object geometry assigned to its environment.

See :doc:`./homogeneous_and_heterogeneous_placement` for the implementation
differences, visual examples, and runnable commands.

Layouts Across Resets
---------------------

The builder prepares a pool of ranked layouts for each environment. By default,
each reset assigns the next queued layout from that environment's pool, allowing
layouts to vary without solving every reset. Layout uniqueness is not
guaranteed. If a pool becomes empty, placement generates more layouts during
the reset.

Set ``ObjectPlacerParams.resolve_on_reset=False`` when each environment should
reuse the layout sampled for it during environment creation. From the command
line, use ``--no-resolve_on_reset``. Keep the default when layouts should vary
across episodes.

Reproducibility
---------------

Set ``placement_seed`` when placement must be repeatable. With the same Arena
environment definition, placement seed, and environment count, layout
generation and random object-set assignment are repeatable.

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
