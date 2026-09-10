Physics-Settled Clutter
=======================

Clutter describes a scene arrangement. Objects in that arrangement remain
ordinary Arena assets and may be task targets, distractors, or both. Existing
clutter scenes can use ordinary ``On`` relations without physics-based generation.

``ClutteredOn`` requests one specific placement method: sample release poses for
a group above a support and let physics determine their resting poses. Members
may rest directly on the support or on other members. This method does not
require every member to touch another, and does not guarantee any particular
density, occlusion, graspability, or task difficulty.

Arena solves ordinary placement relations first, plans noninterpenetrating
release poses, and saves the resulting positions and full quaternions after
settling. It then revalidates the final poses with Arena's enabled placement
validators, including task-added reachability when its validator is available.

Generate a fixed scene offline
------------------------------

Run the standalone example inside the Arena development container, from
``/workspaces/isaaclab_arena``:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena_examples/relations/generate_clutter_scene.py \
       --env_spec isaaclab_arena_examples/relations/clutter_scene.yaml \
       --output outputs/clutter/scene.yaml --seed 42 --viz none

The output is a normal Arena environment graph. It preserves the asset and task
specifications, replaces placement relations with ``params.initial_pose``, and
restores these poses through the assets' reset events. Positions are in metres
in the environment-local frame; quaternions use ``[x, y, z, w]``. Saving a layout
does not make its rigid objects kinematic.

.. code-block:: yaml

   objects:
   - id: cube_0
     registry_name: dex_cube
     params:
       initial_pose:
         position_xyz: [0.1, 0.2, 0.8]
         rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
   relations: []

Inspect a generated scene with the ordinary environment runner:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
       --env_spec outputs/clutter/scene.yaml

``--num_envs 4`` prepares independent piles in parallel and writes
``scene_env_0.yaml`` through ``scene_env_3.yaml``. Each file stores a single
layout that can itself be replayed in any number of environments.
``--layouts_per_env`` controls the candidate pool before rejection; the default
is five. The exporter refuses existing output files and refuses an environment
whose entire pool failed settling, final validation, or containment. It validates every requested
output before writing files.

For downstream asset and task packages, install the package in the same runtime
and pass ``--register package.module:register_components``. Registration runs
after simulation startup and before YAML loading. ``--presets`` selects an
Arena physics preset. Reuse the asset versions and physics configuration used
during generation when evaluating the exported scene. Exact saved poses remove
the need to repeat a stochastic pour; they do not promise identical subsequent
physics trajectories across different backends or simulator versions.

Declare clutter in Python or YAML
---------------------------------

.. code-block:: python

   from isaaclab_arena.relations.relations import ClutteredOn, IsAnchor, RotateAroundSolution

   table.add_relation(IsAnchor())
   for tool in tools:
       tool.add_relation(ClutteredOn(table, group="tools", spread=0.7))
   tools[0].add_relation(RotateAroundSolution(roll_rad=1.57079632679))

.. code-block:: yaml

   relations:
   - kind: is_anchor
     subject: table
   - kind: cluttered_on
     subject: tool_0
     reference: table
     params:
       group: tools
       spread: 0.7
       clearance_m: 0.01
       gap_m: 0.03
       random_yaw: true
       drop_order: shuffle

Members on the same support with the same ``group`` form one pile. They must
agree on ``spread`` and ``drop_order``. ``spread`` scales the release region
about its centre, with values in ``(0, 1]``. Containment is checked against the
whole support after settling, so a tight pile may relax outward.

Each member independently specifies its initial floor clearance, vertical gap
above overlapping footprints, and whether to sample yaw. World-Z yaw is composed
on top of ``RotateAroundSolution``, preserving authored roll and pitch. Drop
order may be ``as_listed``, ``flattest_first``, or ``shuffle``.

Online preparation and resets
-----------------------------

Set ``ArenaEnvBuilderCfg.placement_seed`` explicitly. Online clutter requires
``resolve_on_reset=True``. Arena prepares the pool once after simulator
construction, checks consecutive quiet pose windows and support containment,
and then recycles the accepted layouts. Exhausting one environment's queue
rewinds only that queue; it never generates unvalidated release poses during a
reset. Pool size therefore bounds the available diversity.

The builder invokes an explicit preparation operation after simulator construction;
the environment constructor itself does not settle layouts. Applications using
``build_registered()`` and ``gym.make()`` directly should call
``builder.prepare_placement(env)`` before the first reset. An unprepared clutter
pool cannot be used by a reset event.

Configure preparation before constructing the simulator. The offline example uses
``builder.make_registered()``, which prepares the pool automatically:

.. code-block:: python

   from isaaclab_arena.relations.clutter_validation import ClutterSettleParams

   cfg, kwargs = builder.compose_manager_cfg()
   cfg.clutter_settle_params = ClutterSettleParams(timeout_s=15.0)
   env = builder.make_registered(env_cfg=cfg, env_kwargs=kwargs)

To defer preparation, set ``cfg.settle_clutter_on_build=False`` before construction,
then call ``builder.prepare_placement(env)`` once before the first reset.
Calling preparation on an already-prepared pool raises; it cannot apply new
thresholds retroactively.

Configure ``clutter_settle_params`` to change the timeout, poll interval, quiet
thresholds, or ``containment_margin_m``. ``passive_move_thresh_m`` and
``passive_turn_thresh_deg`` bound total passive-body drift from reset poses,
independently of the consecutive-sample rest thresholds. Time budgets use
simulated seconds and are independent of control decimation. Preparation starts
from configured scene defaults, including robot joints, before testing candidate
layouts. It restores the caller's poses, velocities, joint state and actuator
targets on exit, including when a validator raises. Candidate trials are isolated.

Final checks use full object rotations. Ordinary neighbors are rechecked against
their placement relations; a pile that pushes a passive fixture away from its reset
pose is rejected. Extensions that supply placement validators must implement
``PlacementValidator.validate_poses`` to participate in physics preparation.

Export uses the exact graph-node-to-asset mapping returned by
``build_arena_env_with_assets_from_graph_spec``. Registry names and runtime names
need not match graph IDs.

Scope and constraints
---------------------

- Supports must be horizontal, aligned to the world axes or turned by a multiple
  of 90 degrees, and have static or kinematic spawned geometry. ``IsAnchor`` alone only fixes the solver
  pose. A base ``ObjectReference`` inside a kinematic fixture may identify its
  floor; relative prim paths resolve under the named parent asset.
- Members must be dynamic rigid objects with gravity enabled. Articulated and
  deformable member state cannot be saved as a single pose. Preparation also
  rejects articulation link motion exceeding the passive drift thresholds, since a
  pose-only layout cannot replay changed joint configurations.
- Members may carry ``ClutteredOn``, ``RotateAroundSolution`` and
  ``RequiresReachability``. Reachability uses the final captured pose and the
  normal registered Arena validator. Other placement relations on members, or
  relations targeting a member, are rejected because pouring controls their poses.
- Release planning uses conservative bounding boxes. Concave supports need an
  explicit floor reference and mesh collision checking for their enclosing
  geometry; their outer bounds do not describe a usable interior.
  Collision with the support and other objects is resolved by physics. Settled
  contacts within a pile and with its support are intentional. Other object pairs
  are checked by the normal collision mode using full final rotations.
- Offline export currently requires concrete registered objects. Object sets
  must first be resolved to concrete asset variants. Runtime randomization or
  external callbacks that move saved objects must be disabled for fixed replay.
- Reachability is checked only when its validator is enabled and available, as
  with ordinary Arena placement. A valid initial layout does not guarantee grasp
  or task success.
