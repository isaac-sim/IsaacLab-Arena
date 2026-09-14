CAP Offline Clutter Placement
=============================

The CAP offline tool creates physics-settled layouts for ordinary Arena rigid
objects. It uses Arena's ``ClutterOn`` relation and ``ObjectPlacer`` for release
poses, then steps physics and validates rest and support containment. Settling
runs in the CAP extension; relation placement and cached replay are shared
Arena capabilities.

Declare clutter in the environment
----------------------------------

Add relations to the environment YAML:

.. code-block:: yaml

   relations:
   - kind: is_anchor
     subject: table
   - kind: clutter_on
     subject: cube_0
     reference: table
     params:
       spread: 0.2
       clearance_m: 0.01
       gap_m: 0.03
       random_yaw: true

``spread`` scales the release footprint about the support center. Release
solving and validation use that smaller region. After physics settling,
containment uses the whole support. ``clearance_m`` is the initial surface gap;
``gap_m`` separates overlapping bounds during initialization. Objects start in
free space in asset order, then the solver applies their relations and the
shared collision clearance. ``ClutterOn`` can combine with ``AtPosition`` or
other spatial constraints. Use Arena's ``RotateAroundSolution`` relation to
author a base rotation; sampled world-Z yaw preserves its roll and pitch. Set
``random_yaw: false`` to retain that rotation.

Without a cache, the same environment uses ``ObjectPlacer`` to initialize the
release column. The objects fall when simulation starts. No online settling or
settled-layout validation runs during environment construction or reset.

Generate a companion cache
--------------------------

Run inside the Arena development container, from ``/workspaces/isaaclab_arena``:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena_environments/isaac_cap/clutter/generate_clutter_scene.py \
       --env_spec isaaclab_arena_environments/isaac_cap/clutter/clutter_scene.yaml \
       --output outputs/clutter/placements.yaml --num_envs 4 --num_layouts 100 \
       --seed 42 --viz none

One environment file and one companion pose file describe all layouts.
``--num_envs`` controls parallel generation; ``--num_layouts`` controls the total
(default: one per environment). ``--attempts`` limits retries per environment;
``--timeout_s`` limits simulated time per trial. Only rejected environments are
retried. Generation fails if any requested layout cannot be validated, and
writes output only after all layouts pass. Only release candidates that pass
placement validation are simulated. Existing output files
are not overwritten.

Other placement must already be resolved to fixed anchors. Object sets are not
supported by this cache format. Downstream packages can register assets and
tasks with ``--register package.module:register_components``. ``--presets``
selects the physics backend.

Load at runtime
---------------

Pass the companion file explicitly:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_environments/isaac_cap/clutter/clutter_scene.yaml \
       --placement_layouts outputs/clutter/placements.yaml

Alternatively, add ``placement_layouts: placements.yaml`` to the environment
YAML. That path is relative to the environment file; a command-line override is
relative to the working directory and takes precedence.

The companion file maps graph object IDs to lists of environment-local poses,
with positions in metres and quaternions in ``[x, y, z, w]`` order:

.. code-block:: yaml

   cube_0:
   - position_xyz: [0.1, 0.2, 0.8]
     rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
   - position_xyz: [-0.1, 0.2, 0.8]
     rotation_xyzw: [0.0, 0.0, 0.0, 1.0]

Every object has the same number of poses. An index selects a complete layout
across all objects. Environment ``i`` starts at index ``i % num_layouts``; each
reset advances that environment's index and wraps at the end. Partial resets
leave other environments unchanged. Replay restores exact root poses and zero
velocities, bypassing relation solving and settling. Pose-changing variations
or custom reset callbacks can alter the layout and should be disabled for exact
replay. Cached assets must not have their own explicit pose-reset events;
the complete-layout event owns their pose resets.

Use from Python
---------------

.. code-block:: python

   from isaaclab_arena_environments.isaac_cap.clutter.settle import settle_clutter
   from isaaclab_arena_environments.isaac_cap.clutter.validation import ClutterSettleParams

   env.reset()
   layouts = settle_clutter(
       env,
       list(arena_env.scene.assets.values()),
       seed=42,
       params=ClutterSettleParams(timeout_s=15.0),
   )

Assets must carry their ``ClutterOn`` and ``IsAnchor`` relations. Their fixed
poses must match the constructed scene. ``layouts[env_id]`` maps dynamic rigid
object scene keys to ``Pose`` values. The helper restores scene state and
actuator targets on success and failure.

Checks and limits
-----------------

- Supports must be horizontal, axis aligned or turned by a multiple of 90
  degrees, with static or kinematic spawned geometry.
- Clutter members must be dynamic rigid objects with gravity enabled. Passive
  rigid neighbors need fixed poses and collision geometry. All dynamic rigid
  objects are monitored and cached; the YAML exporter rejects unmapped bodies.
- Rest requires consecutive quiet pose windows. Full rotated bounds must stay
  above and inside the support. Moving, non-finite, or escaped objects reject
  the trial. Rest thresholds use ``--move_thresh_m`` and ``--turn_thresh_deg``.
- Neighbors, supports and robot links must remain within
  ``--passive_move_thresh_m`` and ``--passive_turn_thresh_deg`` of their initial
  poses. Articulation configurations are not stored in the pose cache.
- Conservative bounding boxes do not describe concave interiors. Use a fixed
  reference identifying the usable support surface inside a fixture.
- Release placement uses Arena's geometry validators. Settled layouts are
  checked for rest and containment; task reachability is not certified.
- Replay assumes the same assets and physics configuration. Exact starting
  poses do not guarantee identical trajectories across simulator versions.

Run the tests
-------------

.. code-block:: bash

   /isaac-sim/python.sh -m pytest -sv \
       isaaclab_arena/tests/isaac_cap/test_settled_scene.py \
       isaaclab_arena/tests/isaac_cap/test_clutter_validation.py \
       isaaclab_arena/tests/test_clutter_on.py \
       isaaclab_arena/tests/test_placement_layouts.py

   /isaac-sim/python.sh -m pytest -sv isaaclab_arena/tests/isaac_cap/test_clutter_cli.py
