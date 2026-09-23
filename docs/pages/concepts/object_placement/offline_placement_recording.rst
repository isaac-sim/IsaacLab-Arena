Record Initial Placement Poses
==============================

Use ``record_placement_layouts.py`` to prepare reusable initial poses before
running a policy. It solves existing placement relations, advances physics, and
saves the final poses of layouts that remain close to the solved arrangement.
The saved poses can then be loaded on reset without solving again.

.. code-block:: text

   Environment YAML -> solver and required validation -> physics
                          -> velocity and pose-shift checks -> poses.jsonl
   Environment YAML + poses.jsonl -> restore poses on reset -> run policy

Record a scene
--------------

Run these commands from the repository root inside the Arena container.
The included scene has four cubes ``On`` a table and a separate Franka arm.
It uses ``clearance_m: 0.001`` so each cube begins close to contact:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/placement_recording/scene.yaml \
       output=outputs/placements/poses.jsonl \
       num_envs=2 layouts_per_env=5 seed=42 \
       settle.num_steps=120 --device cpu --viz none

The command prints each validator's effective settings and enabled or skipped
status, followed by accepted counts and rejection reasons. It writes one JSONL
record per accepted layout. An existing output file is never overwritten; choose
a new path for another run.

Recording settings use Hydra ``key=value`` syntax. Isaac Lab launcher settings
retain their ``--flag`` syntax. Use ``presets=newton`` for a Newton scene, and
``render=true --viz kit`` to watch the recording pass. Every enabled source
relation remains part of the same solver problem; no special relation is required.

Inspect the result
------------------

Each line contains poses keyed by runtime scene name under
``variations["scene.relation_placement"]["poses"]``. To inspect the first layout:

.. code-block:: bash

   head -n 1 outputs/placements/poses.jsonl | python -m json.tool

Expect finite ``position_xyz`` and ``rotation_xyzw`` values for the cubes and robot
root. Positions are in metres in the local environment frame; quaternions are
xyzw. The ``validation`` entry contains source solver verdicts under ``pre_physics``
and a list of reports under ``post_physics``. Each report includes the check name,
implementation path, effective settings, ``passed`` and ``reason``. A skipped check
has ``passed: null`` and a reason, such as "scene has no articulations". Simulation
duration is recorded under ``sampling``. Solver verdicts refer to the initial
candidate; post-physics reports describe the recorded poses.

Open the saved scene using the existing interactive runner:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
       --env_spec isaaclab_arena_environments/placement_recording/scene.yaml \
       --placement_layouts outputs/placements/poses.jsonl \
       --num_envs 1 --device cpu --viz kit

The cubes should start resting on the table, without a release drop. Close and
reopen the command to inspect the same first layout. The interactive runner uses
CPU PhysX; use the same backend and device as recording for comparisons of
subsequent physics. Exact reset poses do not imply identical future trajectories.
See :doc:`relations` for replay selection, cycling and partial resets.

Acceptance checks
-----------------

Each recorded candidate must pass all required solver checks and every enabled,
applicable post-physics validator. Candidates with missing explicitly required
solver results are rejected before simulation, including unavailable IK checks.
All validators share one physics pass. Disabled
or inapplicable validators produce skipped reports, not successful verdicts.
All rigid and articulation roots are recorded, including fixed rigid roots.

``settle.num_steps`` controls duration in environment steps (default 5), each
containing ``decimation`` physics substeps. The example uses 120 to give the scene
time to settle. ``settle.min_layouts`` sets the minimum accepted count needed to
write output (default 1).

The default validators are:

* ``physics_settled``: existing final-velocity check, with ``lin_vel_thresh=0.1`` m/s
  and ``ang_vel_thresh=0.1`` rad/s.
* ``pose_shift``: maximum root displacement of ``max_translation_m=0.002`` metres
  and rotation of ``max_rotation_deg=2`` degrees from the initial pose.
* ``articulation_link_shift``: the same displacement and rotation limits for links
  relative to their root. Skipped when the scene has no articulations.

Settings live under ``settle.validators.<check>``. For example, append
``settle.validators.pose_shift.max_translation_m=0.001`` to tighten the root limit.
To disable a check explicitly, use
``settle.validators.pose_shift.enabled=false``. Its skipped report remains in the
record; that record no longer certifies the disabled condition. At least one
applicable validator must remain enabled. Keep the defaults for the SQA example.

With the default checks, a layout that comes to rest after a large drop is rejected. ``On`` defaults to
1 cm of release clearance, which exceeds the recording shift limit. For an
initial arrangement intended to remain still, use a smaller relation clearance,
as in the sample, rather than weakening the recording limit. The accepted final
pose, including a small permitted adjustment, is what gets recorded. Joint
states are not saved, so excessive link motion also rejects the layout.

The solver's IK and geometric checks are not repeated after physics. The default
post-physics checks certify the velocity and shift limits above, not exact
preservation of every relation or a new IK solution at the measured poses. If recording fails, inspect
the rejection reason and scene before changing acceptance limits.

Python use and scope
--------------------

For an initialized environment with a placement pool:

.. code-block:: python

   from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
   from isaaclab_arena.offline_placement.settled_placement import collect_settled_pool_layouts
   from isaaclab_arena.relations.placement_events import get_placement_pool

   result = collect_settled_pool_layouts(
       env, get_placement_pool(env), PlacementRecordingParams(num_steps=120),
       scene_assets=arena_env.get_placement_assets(),
   )
   result.layouts.write_episode_jsonl(
       "poses.jsonl", source="settled", validation=result.validation,
   )

Collection does not consume the pool or change its validation results. It restores
scene roots, joints and actuator targets on completion or failure. The recorder
and ``run_placement_pool_validation.py`` share the same physics loop; a separate
validation run is unnecessary. Pool validation lives in
``isaaclab_arena.offline_placement.pool_validation``. The velocity-only pool
validator also supports deformables; root-pose recording does not.

Use concrete assets with writable rigid or articulation roots. Object sets and
``RandomAroundSolution`` are unsupported. Replay requires the same assets and
robot joint reset configuration. Disable pose-changing variations and callbacks
when exact pose restoration is required. This tool records object and robot root
poses, not general variations or joint states.

Custom checks
-------------

Custom post-physics checks subclass ``PostPhysicsPlacementValidator`` in
``isaaclab_arena.offline_placement.validators``. Define a dataclass with a unique
``check`` name and implement ``validate(PostPhysicsState)`` to return one
``PlacementValidatorReport`` per ``env_ids`` entry, in order. Use ``self.report``
to retain the effective settings. Enabled, applicable checks must return pass/fail;
``skip_reason`` describes scene-level inapplicability.

For an importable ``my_project.validators.SupportValidator`` with
``check = "support"``, add it to the existing checks with:

.. code-block:: bash

   +settle.validators.support._target_=my_project.validators.SupportValidator

The shared ``PlacementValidator`` base lives in ``relations.placement_validation``.
Existing solver validators keep their batch API; post-physics implementations
live under ``offline_placement``. The online placement path does not import the
offline package.
