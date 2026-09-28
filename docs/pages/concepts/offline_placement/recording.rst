Record and Replay Placement Poses
=================================

Use ``record_placement_layouts.py`` to prepare reusable initial poses before
policy evaluation. It solves placement relations, advances physics, and records
the final poses of accepted layouts. Replay loads those poses on reset without
solving or settling them again.

Each recording batch calls ``env.reset()`` to select one solved layout per
parallel environment. It then steps physics, checks the measured poses, and saves
accepted layouts. ``layouts_per_env`` is the number of reset batches; four
environments and four batches produce 16 attempts.

The examples below use Robolab tasks with visualization enabled. The clamp task
demonstrates recording and replay; the smartphone task demonstrates rejection.

Sampling Workflow
-----------------

1. Record the Clamp Scene
~~~~~~~~~~~~~~~~~~~~~~~~~

Use a current standard Arena Docker image with a workstation display. See
:doc:`../../quickstart/installation` for setup. From the host repository root:

.. code-block:: bash

   ./docker/run_docker.sh -s record-replay

Run the remaining commands from the repository root inside that container, with a
workstation display available. Recording and replay use the same container.
The examples use CPU PhysX and show the Kit viewport.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml \
       output=outputs/placements/clamp.jsonl \
       num_envs=4 env_spacing=2 layouts_per_env=4 seed=42 \
       'viewer_eye=[4.0,4.0,6.3]' 'viewer_lookat=[0.6,0.6,0.3]' \
       settle.num_steps=120 \
       settle.validators.pose_shift.max_translation_m=0.015 \
       render=true --device cpu --viz kit

The existing scenes use about 1 cm of ``On`` release clearance. This exceeds the
recorder's default 2 mm shift limit. The commands explicitly allow 15 mm of root
translation. Rotation limits remain 2 degrees and root-speed limits remain
0.1 m/s and 0.1 rad/s. These are example settings, not changed defaults;
use limits appropriate to the accuracy needed by your evaluation.

Physics also advances the robot, which may move or contact objects. The checks
measure all root speeds and shifts, plus link shifts for articulated task objects.
Robot joints are excluded; their motion does not cause a link-shift rejection.
Only root poses are saved. See
:ref:`recording_robot_motion` before relying on replay to match robot contact geometry.

``viewer_eye`` and ``viewer_lookat`` set the camera in simulation-world coordinates
before the first batch. These values produce the four-environment overview shown
below; omitting them keeps the task's default camera. Set both together.

The viewport shows four tables and robots. Each batch resets the environments to
the next solved layouts, then advances 960 physics steps (120 environment steps
with decimation 8).
The clamp scene varies positions without randomizing tool orientations and has a
small initial drop. Reset also runs the task's joint and other reset events.
The console prints enabled checks
and their settings, physics-step progress, and acceptance counts per batch.

.. image:: ../../../images/offline_placement/clamp_recording.gif
   :alt: Four batches of clamp layouts in four parallel environments.
   :width: 100%

This eight-second GIF shows short excerpts from four batches: the first second
of settling and a brief final view of each layout. Each batch still runs all
960 physics steps. Use the console results below to check acceptance.

The reference run accepted 15/16 layouts. One hammer moved 15.7 mm, exceeding
the configured 15 mm root-shift limit. Console excerpt:

.. code-block:: text

   [recording] batch 1/4: 480/960 physics steps
   [recording] batch 1/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 4/16 validated, 4 accepted
   [recording] batch 2/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 8/16 validated, 8 accepted
   [recording] batch 3/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 12/16 validated, 12 accepted
   [recording] batch 4/4: 4 solutions, 4 passed solver validation, 3 passed post-physics validation; overall 16/16 validated, 15 accepted
   Saved 15/16 accepted layouts: outputs/placements/clamp.jsonl

Both full and partial acceptance are valid outcomes. Counts can vary across
machines, simulator versions, assets and solver configurations; 15/16 is a
reference result, not a required count. Every saved layout must pass all enabled,
applicable checks, and rejected layouts must have reported reasons. If no layouts
pass, the recorder reports failure and writes no file.

An existing output file is never overwritten; choose a new path for each recording.

2. Inspect the Saved Poses
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   wc -l outputs/placements/clamp.jsonl
   head -n 1 outputs/placements/clamp.jsonl | /isaac-sim/python.sh -m json.tool

The line count must match the recorder's accepted count. Each line contains one
complete layout. ``source``, ``poses`` and ``validation`` are fields inside
``variations["scene.relation_placement"]``. Check that:

* ``source`` is ``"settled"``;
* ``poses`` contains scene names, including ``spring_clamp``, the bins and ``robot``;
* positions and quaternions contain finite numbers;
* ``physics_settled`` and ``pose_shift`` in ``validation.post_physics`` have
  ``passed: true`` and the settings shown above;
* ``articulation_link_shift`` has ``passed: null`` because no articulated task
  objects were selected. Robot joints are excluded from this check.

Positions are in metres in the local environment frame; quaternions are XYZW.
``validation.pre_physics`` holds the solver verdicts for the initial candidate.
``validation.sampling`` records the physics duration and ``embodiment_keys`` excluded
from link-shift checks. Only root poses are saved,
not robot joint states.

3. Replay the Accepted Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same environment YAML and the saved file:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
       --env_spec isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml \
       --placement_layouts outputs/placements/clamp.jsonl \
       --policy_type zero_action --num_episodes 3 \
       --num_envs 1 --device cpu --viz kit \
       --output_base_dir outputs/placements/evaluation

The clamp and bins should start at the saved poses without the original release
drop. This task uses absolute joint-position actions, so zero actions can move the
robot arm; they do not perform the pick-and-place task. Check the object poses
at each reset before attributing later motion to replay. With one environment,
each reset loads the next record and wraps after the last one. Restarting the
command starts from the first record.
The command exits after three episodes; each runs for up to 70 seconds of
simulation, which may differ from wall-clock time.

Open ``outputs/placements/evaluation/<timestamp>/index.html`` for the results.
The adjacent ``episode_results_rank0.jsonl`` contains per-episode metrics. Zero
task success is expected with this policy. Verify that the report contains three
completed episodes and inspect any reported object movement. The reference replay
reported ``object_moved_rate: 0.0``. This check does not measure manipulation
performance.
Include the command, console log, input JSONL and evaluation results when reporting
a discrepancy.

Exact root reset poses do not guarantee matching joint geometry or future
trajectories. Matching joint geometry requires deterministic joint initialization
as well as the same scene, backend and device.
See :doc:`../object_placement/relations` for parallel and partial-reset selection.

4. Check Rejection with the Smartphone Scene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same settings on the existing smartphone task:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/robolab/tasks/smartphone_in_bin.yaml \
       output=outputs/placements/smartphone.jsonl \
       num_envs=4 env_spacing=2 layouts_per_env=4 seed=42 \
       'viewer_eye=[4.0,4.0,6.3]' 'viewer_lookat=[0.6,0.6,0.3]' \
       settle.num_steps=120 \
       settle.validators.pose_shift.max_translation_m=0.015 \
       render=true --device cpu --viz kit

.. image:: ../../../images/offline_placement/smartphone_recording.gif
   :alt: Four batches of smartphone-scene layouts in four parallel environments, showing object motion.
   :width: 100%

The reference run rejected all 16 candidates and wrote no file. Objects moved
or tumbled beyond the shift limits; some also exceeded final-velocity limits. Console excerpt:

.. code-block:: text

   [recording] batch 1/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 4/16 validated, 0 accepted
   [recording] batch 4/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 16/16 validated, 0 accepted
   AssertionError: Accepted 0 layouts; need 1. Rejections: {...}

For example, one mouse moved 53 mm and rotated 57 degrees, exceeding the configured
15 mm and 2 degree limits.

A rejected run exits with an error and writes no file when fewer than
``settle.min_layouts`` candidates pass (default 1). Check the named object and
reason in the rejection summary:

* ``physics_settled``: final velocity exceeds the limits. Inspect contacts and the
  source arrangement; increase ``settle.num_steps`` only if the motion is transient.
* ``pose_shift``: a root moved or rotated too far from its solved pose.
* ``articulation_link_shift``: an articulated task object's link moved too far relative
  to its root. Root-only replay cannot reproduce an altered joint configuration.
* ``missing required solver checks``: make the named check available or fix the
  source configuration. Do not remove required checks merely to obtain a file.

Other runtimes can produce different rejection counts. Inspect the reported
conditions instead of assuming that a task name guarantees acceptance or rejection.

Run Without a Display
~~~~~~~~~~~~~~~~~~~~~

For recording, replace ``render=true --viz kit`` with
``render=false --viz none``. For replay, replace ``--viz kit`` with ``--viz none``.
The output and checks remain the same. Recording settings use Hydra ``key=value``
syntax; launcher settings retain ``--flag`` syntax.

The reference runs used CPU PhysX without optional cuRobo IK validation.

Acceptance and Limitations
--------------------------

The recorder requires all required solver checks and all enabled, applicable
post-physics checks to pass. Missing required solver results are rejected.
All enabled post-physics checks share one physics pass:

* ``physics_settled`` checks final root speeds, with defaults of 0.1 m/s and 0.1 rad/s.
* ``pose_shift`` limits initial-to-final root motion to 2 mm and 2 degrees by default.
* ``articulation_link_shift`` applies the same limits to articulated task objects'
  links relative to their root. Robot embodiments are excluded. The check is skipped
  when there are no articulated task objects.

Root-speed and root-shift checks still include the robot, since its root pose is
recorded. Robot joints use the normal reset behavior during replay; their settled
configuration is outside this recording contract.

Configure checks under ``settle.validators.<check>``. For example,
``settle.validators.pose_shift.max_translation_m=0.001`` tightens the root limit.
Setting a check's ``enabled=false`` records a skipped result; it does not certify
that condition. At least one applicable check must remain enabled.
``settle.min_layouts`` controls the minimum accepted count needed to write a file.

.. _recording_robot_motion:

Reset and Robot Behavior
~~~~~~~~~~~~~~~~~~~~~~~~

Recording uses the environment's normal reset events, including joint
randomization and configured variations. Records contain poses, not sampled
variation settings such as mass or visibility. Each reset consumes one placement
per environment; the pool solves more candidates if it runs out. Collection runs
a fixed number of batches and leaves the environment at the final state, including on failure. It does not
restore the caller's state. Use an environment whose reset events reset the roots,
joints and actuator targets needed by the task. Physics-only stepping does not
advance episode-step counters, so reset events with a minimum step interval may
not run on every batch.

When at least one candidate passes required solver checks, physics advances the
whole batch, including any solver-failed candidates applied by reset. Failed
candidates are never recorded. The robot is not frozen and may move or push
objects, affecting both rejected and accepted layouts. Robot joint motion is not
an acceptance check. Task-object link checks compare initial and final poses;
they do not measure joint speed or prove that joints stayed still throughout settling.

Records save measured root poses, not joint states. Replay uses the environment's
joint reset behavior; the Droid examples randomize joint positions at each reset.
Matching robot contact geometry requires matching joint initialization. Solver
geometry and IK checks are not repeated after physics. Acceptance certifies the
configured checks, not a complete articulated state or deterministic trajectory.

Use concrete assets with writable rigid or articulation roots. Object sets and
``RandomAroundSolution`` are unsupported. Recorded assets must allow pose resets,
have zero initial velocity and have no randomized or per-environment root-pose
reset policy. Disable additional pose-changing variations and callbacks when
exact root replay is required.

Recording and Replay Flow
-------------------------

.. image:: ../../../images/offline_placement/recording_pipeline.svg
   :alt: Solve layouts, reset to one per environment, step physics, validate and save accepted poses.
   :width: 100%

Replay loads the JSONL during environment construction. It bypasses solving and
pool creation, then writes saved root poses and zero root velocities on reset.
Physics runs normally during policy evaluation.

.. image:: ../../../images/offline_placement/replay_pipeline.svg
   :alt: Load the scene and saved poses, select layouts on reset, then run the policy.
   :width: 100%

Python API
----------

For a constructed environment with pooled placement enabled:

.. code-block:: python

   from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
   from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements

   result = collect_settled_placements(
       env, num_batches=4, params=PlacementRecordingParams(num_steps=120),
       scene_assets=arena_env.get_placement_assets(),
   )
   result.layouts.write_episode_jsonl(
       "poses.jsonl", source="settled", validation=result.validation,
   )

The collector owns the sampling resets; an initial ``env.reset()`` is unnecessary.
Result indices identify the environment and reset batch, not an index into a
stored pool. ``validate_pool_layouts()`` remains a separate tool for grading every
stored candidate without consuming the pool.

Custom checks subclass ``PostPhysicsPlacementValidator`` in
``isaaclab_arena.offline_placement.post_physics_validation``. Implement
``validate(PostPhysicsState)`` with one report per requested environment, using
``self.report`` to retain settings and results. The state contains all root poses
and only task-object link poses. To add an importable check:

.. code-block:: bash

   +settle.validators.support._target_=my_project.validators.SupportValidator
