Offline Clutter Placement
=========================

Clutter objects are ordinary Arena assets. The offline example samples release
poses, steps physics until the objects rest, and saves their exact poses in a
normal scene YAML. Runtime environments load that file and restore those poses
through existing asset reset events.

Generate a scene cache
----------------------

Run inside the Arena development container, from ``/workspaces/isaaclab_arena``:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena_examples/relations/generate_clutter_scene.py \
       --env_spec isaaclab_arena_examples/relations/clutter_scene.yaml \
       --support table --objects cube_0 cube_1 cube_2 cube_3 --spread 0.2 \
       --output outputs/clutter/scene.yaml --seed 42 --viz none

``--support`` and ``--objects`` identify graph nodes. The input scene must use
concrete asset poses and have no unresolved placement relations or object sets.
Place the surrounding scene before running offline generation. Assets without
``params.initial_pose`` retain their registered defaults.

``--num_envs 4`` generates independent layouts and writes ``scene_env_0.yaml``
through ``scene_env_3.yaml``. ``--attempts`` limits offline retries per environment;
``--timeout_s`` limits each trial's simulated time. A failed trial is retried
only offline. If any environment has no accepted layout, generation fails before
writing output files. Each output is published as a complete YAML file; existing
output files are never overwritten. The output filesystem must support hard links;
its write and link permissions are checked before scene construction and settling.
If the check fails, choose an output directory that supports hard links.

``--spread`` scales the release region; final containment uses the whole support.
Sampled world-Z yaw preserves each object's initial roll and pitch. Use
``--keep_rotation`` to retain the entire initial rotation, or ``--drop_order`` to
choose ``as_listed``, ``flattest_first``, or ``shuffle``. ``--gap_m`` controls the
vertical gap above overlapping footprints; ``--clearance_m`` sets the initial
clearance above the support.

``--poll_interval_s``, ``--required_quiet_windows``, ``--move_thresh_m`` and
``--turn_thresh_deg`` control rest detection. ``--fall_through_tolerance_m`` and
``--containment_margin_m`` control support containment. ``--passive_move_thresh_m``
and ``--passive_turn_thresh_deg`` bound total neighbor, support and robot-link
movement during a trial, independently of the rest-detection thresholds.

Downstream packages can register their assets and tasks with
``--register package.module:register_components``. Registration runs after Isaac
Sim starts and before graph loading. ``--presets`` selects the physics backend.
Reuse the asset versions and physics configuration when replaying cached scenes.

Load at runtime
---------------

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
       --env_spec outputs/clutter/scene.yaml

The cache stores environment-local positions in metres and full quaternions in
``[x, y, z, w]`` order:

.. code-block:: yaml

   objects:
   - id: cube_0
     registry_name: dex_cube
     params:
       initial_pose:
         position_xyz: [0.1, 0.2, 0.8]
         rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
   relations: []

Each generated file represents one fixed layout, reusable across any number of
runtime environments. Choose a different cached file for another layout. Loading
a cache does not run settling, regenerate clutter, or change the objects' physics
properties. Tasks and callbacks must preserve these pose reset events for fixed
replay.

Use from Python
---------------

The helper operates on an already constructed scene, independent of how the
application placed it. Pass scene keys rather than graph IDs:

.. code-block:: python

   from isaaclab_arena_examples.relations.clutter.settle import ClutterGroup, settle_clutter
   from isaaclab_arena_examples.relations.clutter.validation import ClutterSettleParams

   env.reset()
   layouts = settle_clutter(
       env,
       [ClutterGroup(support="table", objects=("tool_0", "tool_1"), spread=0.7)],
       seed=42,
       params=ClutterSettleParams(timeout_s=15.0),
   )

``layouts[env_id]`` maps each dynamic rigid object's scene key to an Arena
``Pose``. Multiple groups may use different supports. The helper restores the
caller's scene state and actuator targets on success and failure. Applications
can save these poses through their own cache format or use the example's
``scene_with_cached_poses`` function with an exact graph-node-to-asset mapping.

Checks and limits
-----------------

- Supports must be horizontal, axis aligned or turned by a multiple of 90 degrees,
  with static or kinematic spawned geometry. A fixed floor reference may identify
  the usable surface inside a fixture.
- Members must be dynamic rigid objects with gravity enabled. All dynamic rigid
  objects are monitored and cached. Undeclared neighbors and supports must stay
  within the passive drift limits of their original poses; put objects that may
  move into an explicit clutter group. The YAML exporter rejects rigid bodies
  without a corresponding graph asset before settling.
- Rest requires consecutive quiet pose windows. Full rotated object bounds must
  remain above and inside the support footprint. Failed trials name moving or
  non-finite objects, displaced neighbors, and members outside their support.
- Articulation configurations cannot be represented by rigid-object poses.
  Changes beyond ``passive_move_thresh_m`` or ``passive_turn_thresh_deg`` reject
  the trial; these limits are separate from consecutive-sample rest thresholds.
- Release bounds are conservative. Concave fixtures need an explicit usable floor
  reference; outer bounds alone do not describe a container interior.
- Physics resolves contact during offline settling. The script checks rest and
  containment; it does not invoke Arena's relation or reachability validators.
  Task feasibility and collision fidelity depend on the authored scene and physics.
- Exact initial poses do not promise identical subsequent trajectories across
  different assets, backends, or simulator versions.
