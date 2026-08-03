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
share one pose configuration.

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
flow. See :ref:`Homogeneous and Heterogeneous Placement
<homogeneous-and-heterogeneous-placement>` for a visual example.

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

.. code-block:: python

   placer_params = ObjectPlacerParams(
       placement_seed=42,
       resolve_on_reset=True,
   )

The simulation seed controls the broader experiment, while
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
