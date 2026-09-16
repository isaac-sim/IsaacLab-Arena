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

In ``BBOX`` mode, collision checking preserves each box's full rotation.
``MESH`` mode follows the collision geometry more closely. Objects with a ``FaceTo`` relation use their relation-derived
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
``isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_and_place.yaml``.
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
     --env_spec \
       isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_and_place.yaml

This example uses the Droid stand footprint for geometric placement.
Reachability, when configured by a task, is checked separately during
candidate validation — see the :ref:`ik-reachable-check` check.

Next Steps
----------

Continue to :doc:`./validation` to see how candidates from these four steps
are checked and either kept or rejected before they enter the placement pool.


Bounding-box and Rotation API
-----------------------------

``OrientedBoundingBox`` stores a center, nonnegative half-extents, and a unit
``rotation_xyzw`` quaternion. Tensors have shapes ``(N, 3)``, ``(N, 3)``, and
``(N, 4)``; a single row broadcasts over a batch. The quaternion maps the box's
own axes into the frame containing its center. It is distinct from an asset's
placement rotation when the asset-local box itself is oriented.

.. code-block:: python

   from isaaclab_arena.utils.bounding_box import OrientedBoundingBox

   box_O = OrientedBoundingBox.from_min_max((-0.2, -0.1, 0.0), (0.2, 0.1, 0.3))
   box_W = box_O.transformed(position_xyz, rotation_xyzw)
   minimum_W, maximum_W = box_W.get_axis_aligned_bounds()
   lower_Z, upper_Z = box_W.get_bounds_along_axis((0.0, 0.0, 1.0))

``rotated_by_quat`` rotates both the center and box axes about the containing
frame's origin. ``transformed`` then translates the result. Neither operation
refits the rotated box into an AABB. USD bounds exclude the default prim's
authored transform, matching spawning, and include spawn scale once. Sub-prim
bounds retain their rigid orientation and conservatively enclose geometry
when nonuniform scale introduces shear.

Direct ``RelationSolver.solve`` calls accept ``rotations``, one dictionary of
complete xyzw rotations per candidate, keyed by movable asset. ``env_bboxes``
contains asset-local boxes without candidate rotation. The old yaw-valued
``orientations`` and ``env_bboxes_include_yaw`` arguments are removed.
``PlacementResult.rotations`` carries those complete quaternions through
validation, initial placement, and pooled resets. Anchors keep their fixed
``Pose`` and must not appear in the rotations dictionary.

The solver optimizes positions only. ``RotateAroundSolution`` contributes a
fixed quaternion; optional random yaw is composed about placement-frame Z.
``FaceTo`` remains a horizontal facing relation: its heading is derived from
current positions and its geometry is reevaluated during optimization.
Relations such as ``On`` and ``NextTo`` continue to use projections along
placement-frame axes. An OBB does not turn ``On`` into a general contact solver
for inclined surfaces or concave supports.

Runtime ``ArenaWorld.get_aabb_in_local_frame`` retains its explicit frame-aligned
contract and now returns an identity-oriented ``OrientedBoundingBox``. Use
``get_axis_aligned_bounds()`` in place of ``min_point``/``max_point``,
``2 * half_extents`` for local box size, and ``get_corners()`` for corners.
