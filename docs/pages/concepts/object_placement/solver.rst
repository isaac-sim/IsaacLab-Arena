Placement Solver
================

The placement solver computes poses for objects and supported robot embodiments.
Spatial relations express the desired relative layout. Collision constraints
penalize disallowed overlaps with other placed assets, fixed anchors, and
passive obstacles.

How Candidates Are Solved
-------------------------

The solver produces candidate layouts in four steps:

1. ``ObjectPlacer`` collects the placeable objects, any supported robot
   embodiment, their relations, the anchors, and the fixed obstacles.
2. It initializes poses for several candidate layouts.
3. ``RelationSolver`` optimizes those poses against the spatial relations and
   collision constraints.
4. ``ObjectPlacer`` applies post-solve ``FaceTo`` headings and sends the
   candidates to :doc:`validators <./validation>`, which check geometric and
   task-specific conditions at build time.

Example Walkthrough
~~~~~~~~~~~~~~~~~~~

The maintained ``pick_and_place_maple_table`` environment defines a table
anchor and gives every object an ``On`` relation. The following code builds
that environment with three additional objects. Calling ``make_registered()``
creates the environment and triggers relation solving automatically; users do
not call ``RelationSolver`` directly:

.. code-block:: python

   from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
   from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
   from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
       PickAndPlaceMapleTableEnvironment,
       PickAndPlaceMapleTableEnvironmentCfg,
   )

   arena_environment = PickAndPlaceMapleTableEnvironment().build(
       PickAndPlaceMapleTableEnvironmentCfg(
           additional_table_objects=["cracker_box", "mug", "tomato_soup_can"],
       )
   )
   builder = ArenaEnvBuilder(arena_environment, ArenaEnvBuilderCfg())
   env = builder.make_registered()
   env.reset()

The four solving steps apply to this example as follows:

1. **Collection:** the environment factory creates the table reference as a
   fixed anchor and adds the registered objects with their ``On`` relations.
   ``ObjectPlacer`` collects these assets and relations.
2. **Initialization:** it creates several candidates with different initial
   poses for the objects.
3. **Optimization:** ``RelationSolver`` adjusts those poses to satisfy the
   ``On`` relations while penalizing collisions. The table remains fixed.
4. **Post-processing and validation:** this example has no ``FaceTo`` relation,
   so no relation-derived heading is applied. ``ObjectPlacer`` sends the
   resulting candidates to the configured :doc:`validators <./validation>`.

Orientation Handling
--------------------

Random Yaw Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~

The walkthrough uses the default yaw initialization. To initialize the same
environment's candidates with arbitrary yaw angles, set its placement
parameters before creating the builder:

.. code-block:: python

   from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams

   arena_environment.placer_params = ObjectPlacerParams(random_yaw_init=True)
   builder = ArenaEnvBuilder(arena_environment, ArenaEnvBuilderCfg())
   env = builder.make_registered()

In ``BBOX`` mode, collision checking uses the conservative axis-aligned box
enclosing each rotated object. ``MESH`` mode follows the collision geometry
more closely. Objects with a ``FaceTo`` relation use their relation-derived
heading instead of a random yaw.

Robot Embodiment Placement
--------------------------

Object placement does not require the robot itself to be relation-placed. A
robot embodiment can also use the solver if it provides placement bounds and
has spatial relations. A typical mobile-manipulation layout places the robot on
the floor, offsets it from a work surface, and sets its heading.

YAML Specification
~~~~~~~~~~~~~~~~~~

The following excerpt applies this pattern to a Droid and a kitchen counter:

.. important::

   Do not set an explicit initial-pose override on a relation-placed
   embodiment; the builder supplies its creation and reset poses.

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

The floor and countertop are fixed anchors. The ``On`` relation places the
Droid on the floor, ``NextTo`` offsets it from the counter, and
``rotate_around_solution`` sets its final heading.

The complete example is
``isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml``.
Its ``placement_bbox_stand_only: true`` option uses only the Droid stand
footprint for placement; the robot arm is excluded from those bounds.

Running the Example
~~~~~~~~~~~~~~~~~~~

Run the example with:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_steps 100 \
     --env_graph_spec_yaml \
       isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml

This example uses the Droid stand footprint for geometric placement.
Reachability, when configured by a task, is checked separately during
candidate validation — see the :ref:`ik-reachable-check` check.

Next Steps
----------

Continue to :doc:`./validation` to see how candidates from these four steps
are checked and either kept or rejected before they enter the placement pool.
