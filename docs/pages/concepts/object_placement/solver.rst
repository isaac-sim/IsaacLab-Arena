Placement Solver
================

The placement solver computes asset poses from spatial relations and collision
constraints. Relations describe the intended arrangement, while collision
handling keeps assets from overlapping.

How a Candidate Is Solved
-------------------------

Solving a candidate involves four steps:

1. ``ObjectPlacer`` collects placeable assets, relations, anchors, and fixed
   obstacles.
2. It initializes several candidate layouts.
3. ``RelationSolver`` optimizes positions using relation strategies and
   collision constraints.
4. ``ObjectPlacer`` resolves orientation relations and sends the candidates to
   validators, which check required geometric or task-specific conditions.

Random Orientation
------------------

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
       name="random_orientation",
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

The complete example is
``isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml``.
Its ``placement_bbox_stand_only: true`` option uses only the Droid stand
footprint for placement; the robot arm is excluded from those placement bounds.
Run it from the repository root with:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type zero_action \
     --num_steps 100 \
     --env_graph_spec_yaml \
       isaaclab_arena_environments/kitchen_bench/droid_pick_and_place_lightwheel_kitchen.yaml

Robot-base placement and task reachability are checked separately.
Documentation for validation, including reachability, is forthcoming in
:doc:`./validation`.

See :doc:`./relations` to define spatial intent and
:doc:`./collision_handling` to choose a collision representation. Next, see
:doc:`./pooled_placement` for how solved layouts are stored and applied across
environments and resets.
