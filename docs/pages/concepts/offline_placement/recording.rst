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

The examples use the existing Robolab environments with visualization enabled:
``clamp_in_right_bin`` for recording and replay, and ``smartphone_in_bin`` for rejection.

Sampling Workflow
-----------------

1. Record ``clamp_in_right_bin``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete :doc:`../../quickstart/installation` using native ``uv`` or Docker.
For an installed native ``uv`` environment, activate it from the repository root:

.. code-block:: bash

   source .venv/bin/activate

For Docker, use the Arena container shell prepared during installation.
Run the following commands from the repository root in your chosen environment;
``python`` uses the configured Arena interpreter in either workflow. A display is
only needed for ``--viz kit``; see `Run Without a Display`_ for headless commands.

Recordings are portable JSONL files. You can copy them to another native or Docker
installation with compatible assets, physics-root names and reset settings.
Point the replay command at the copied file; the original container is unnecessary.

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml \
       output=outputs/placements/clamp.jsonl \
       num_envs=4 env_spacing=2 layouts_per_env=4 seed=42 \
       'viewer_eye=[4.0,4.0,6.3]' 'viewer_lookat=[0.6,0.6,0.3]' \
       settle.num_steps=120 \
       settle.validators.pose_shift.max_translation_m=0.015 \
       render=true --device cpu --viz kit

These environments use about 1 cm of ``On`` release clearance. This exceeds the
recorder's default 2 mm shift limit. The commands explicitly allow 15 mm of root
translation. Rotation limits remain 2 degrees and root-speed limits remain
0.1 m/s and 0.1 rad/s. These are example settings, not changed defaults;
use limits appropriate to the accuracy needed by your evaluation.

See :ref:`recording_robot_motion` for robot-motion and joint-state limitations.

The default viewer is already part of the environment's task configuration.
For these ``PickAndPlaceTask`` environments, it looks at the pickup object's
initial position in the first environment, with the eye offset by
``[-1.5, -1.5, 1.5]`` metres. The optional ``viewer_eye`` and ``viewer_lookat``
values above instead frame all four environments together in simulation-world
coordinates. Set both to override the camera, or omit both to keep the task's view.

Each batch resets the four environments to the next solved layouts. The command
sets ``settle.num_steps=120`` environment steps; Arena's default environment
configuration supplies ``decimation=8`` physics substeps per environment step.
The stepping code multiplies them to advance 960 physics steps per batch. With
that configuration's ``sim.dt=1/120`` seconds, this is eight seconds of simulation;
the step count and duration depend on these settings.

The ``clamp_in_right_bin`` environment varies object positions without randomizing
tool orientations and has a small initial drop. Each reset also runs the Droid
embodiment's configured robot-joint reset event. The console prints enabled checks
and their settings, physics-step progress, and acceptance counts per batch.

.. image:: ../../../images/offline_placement/clamp_recording.gif
   :alt: Four batches of clamp_in_right_bin layouts in four parallel environments.
   :width: 100%

This eight-second GIF shows short excerpts from four batches: the first second
of settling and a brief final view of each layout. Each batch still runs all
960 physics steps. Use the console results below to check acceptance.

The reference run accepted 15/16 layouts. The pose-shift validator computes the
distance between each root's measured initial and final positions. For one
``black_hammer`` candidate it logged 0.015724 m (15.724 mm), exceeding the configured
15 mm limit. The rejection diagnostic appears in the console:

.. code-block:: text

   [placement] 480/960 physics steps
   [placement] batch 1/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 4/16 validated, 4 accepted
   [placement] batch 2/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 8/16 validated, 8 accepted
   [placement] batch 3/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 12/16 validated, 12 accepted
   [placement] batch 4/4: 4 solutions, 4 passed solver validation, 3 passed post-physics validation; overall 16/16 validated, 15 accepted
   Rejected 1: pose_shift: black_hammer: moved 0.015724 m and rotated 1.112 deg; limits 0.015 m, 2 deg
   Saved 15/16 accepted layouts: outputs/placements/clamp.jsonl

Both full and partial acceptance are valid outcomes. Two reference runs with
``seed=42`` on the same machine and CPU PhysX runtime, using identical assets and
settings apart from the output path, both accepted 15/16 and produced byte-identical
JSONL. This is evidence from two runs, not a guarantee that a fixed seed always
produces identical physics results. Regenerating placements samples and simulates
again; replay reuses the saved root poses.

No acceptance-count tolerance has been established across hardware, backends,
simulator versions, assets or solver settings; 15/16 is a reference result, not a
required pass count. For this command, a successful recording must:

* exit successfully and save at least ``settle.min_layouts`` layouts (default 1);
* write exactly the reported accepted count as JSONL records;
* save only layouts that passed all required solver checks and all enabled,
  applicable post-physics checks at the configured limits;
* report reasons for every rejected layout.

Rejection by a configured check is an expected outcome. If fewer than
``settle.min_layouts`` layouts pass, the command must exit with an error and write
no file. A different accepted count alone does not establish a recorder bug;
missing reasons, saved invalid layouts, mismatched counts or incorrect output
and exit behavior require investigation.

An existing output file is never overwritten; choose a new path for each recording.

2. Inspect the Saved Poses
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   wc -l outputs/placements/clamp.jsonl
   head -n 1 outputs/placements/clamp.jsonl | python -m json.tool

The line count must match the recorder's accepted count. Each line contains one
complete layout. ``source``, ``poses`` and ``validation`` are fields inside
``variations["scene.relation_placement"]``. Check that:

* ``source`` is ``"settled"``;
* ``poses`` contains physics-root names, including ``spring_clamp``, the bins and ``robot``;
* positions and quaternions contain finite numbers;
* ``physics_settled`` and ``pose_shift`` in ``validation.post_physics`` have
  ``passed: true`` and the settings shown above;
* ``articulation_link_shift`` has ``passed: null`` because no articulated task
  objects were selected. Robot joints are excluded from this check.

Positions are in metres in the local environment frame; quaternions are XYZW.
``validation.pre_physics`` holds the solver verdicts for the initial candidate.
``validation.sampling`` records the physics duration and ``embodiment_keys`` excluded
from link-shift checks.

3. Replay the Accepted Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same environment YAML and the saved file:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
       --env_spec isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml \
       --placement_layouts outputs/placements/clamp.jsonl \
       --policy_type zero_action --num_episodes 3 \
       --num_envs 1 --device cpu --viz kit \
       --output_base_dir outputs/placements/evaluation

The clamp and bins should start at the saved poses without the original release
drop. This zero-action run checks placement replay; task success is not expected.
Check the object poses at each reset before attributing later motion to replay.
With one environment, each reset loads the next record and wraps after the last
one. Restarting the command starts from the first record.

Open ``outputs/placements/evaluation/<timestamp>/index.html`` for the results.
The adjacent ``episode_results_rank0.jsonl`` contains per-episode metrics. The
command should finish without errors and report three completed episodes.
Missing episodes or incorrect root poses at reset fail this replay check.
The reference replay reported ``object_moved_rate: 0.0``. If this metric is higher,
inspect reset poses and subsequent object motion; the metric alone does not
identify a replay defect. This check does not measure manipulation performance.
Include the command, console log, input JSONL and evaluation results when reporting
a discrepancy.

See :doc:`../object_placement/relations` for parallel and partial-reset selection.

4. Check Rejection with ``smartphone_in_bin``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same settings on the existing ``smartphone_in_bin`` environment:

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/robolab/tasks/smartphone_in_bin.yaml \
       output=outputs/placements/smartphone.jsonl \
       num_envs=4 env_spacing=2 layouts_per_env=4 seed=42 \
       'viewer_eye=[4.0,4.0,6.3]' 'viewer_lookat=[0.6,0.6,0.3]' \
       settle.num_steps=120 \
       settle.validators.pose_shift.max_translation_m=0.015 \
       render=true --device cpu --viz kit

.. image:: ../../../images/offline_placement/smartphone_recording.gif
   :alt: Four batches of smartphone_in_bin layouts in four parallel environments, showing object motion.
   :width: 100%

The reference run rejected all 16 candidates and wrote no file. Objects moved
or tumbled beyond the shift limits; some also exceeded final-velocity limits. Console excerpt:

.. code-block:: text

   [placement] batch 1/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 4/16 validated, 0 accepted
   [placement] batch 4/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 16/16 validated, 0 accepted
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
Headless recording uses the same validation checks and JSONL format. Recording
settings use Hydra ``key=value`` syntax; launcher settings retain ``--flag`` syntax.

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
configured joint resets. In these Droid environments, the embodiment's
``randomize_franka_joint_state`` event randomizes joint positions around their
defaults and restores default joint velocities on reset. Matching robot contact
geometry requires matching joint initialization. Future trajectories also depend
on environment settings, backend and device. Solver geometry and IK checks are
not repeated after physics. Acceptance certifies the configured checks, not a
complete articulated state or deterministic trajectory.

Recording requires concrete assets with writable rigid or articulation roots.
Object sets and ``RandomAroundSolution`` are unsupported for recording.
Recorded assets must allow pose resets,
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

With ``SimulationApp`` already running, use the recording workflow to build an
environment, collect accepted poses and write JSONL:

.. code-block:: python

   from isaaclab_arena.offline_placement.recording_config import PlacementRecordingCfg
   from isaaclab_arena.offline_placement.recording_params import PlacementRecordingParams
   from isaaclab_arena.scripts.record_placement_layouts import record_settled_placement_layouts

   settling = PlacementRecordingParams(num_steps=120)
   settling.validators["pose_shift"]["max_translation_m"] = 0.015
   summary = record_settled_placement_layouts(
       PlacementRecordingCfg(
           env_spec="isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml",
           output="outputs/placements/clamp_python.jsonl",
           num_envs=4, env_spacing=2.0, layouts_per_env=4, settle=settling,
       ),
       device="cpu",
   )

The workflow owns and closes the environment. Inspect ``summary.accepted``,
``summary.attempted`` and ``summary.rejections`` to see the outcome. When fewer
than ``settle.min_layouts`` candidates pass, ``summary.output`` is ``None`` and
no file is written; otherwise it contains the output path. The command-line
entry point reports insufficient acceptance as an error.

For an environment you already own, ``record_placements_to_jsonl`` in the same
script accepts ``env``, an output path and ``num_batches`` and returns the same
summary. This helper leaves environment cleanup to its caller.

For the reusable library API, call ``collect_settled_placements`` to inspect
poses and rejection reasons in memory. It measures the scene's rigid and
articulation roots without enforcing recording or replay policies:

.. code-block:: python

   from isaaclab_arena.offline_placement.settled_placement import collect_settled_placements
   from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams

   try:
       result = collect_settled_placements(
           env, num_batches=4, params=SettledPlacementParams(num_steps=120),
           scene_assets=arena_env.get_placement_assets(),
       )
   finally:
       env.close()

``result.poses`` maps runtime scene names to accepted ``Pose`` lists in matching
order. ``result.validation`` contains the accepted ``PlacementOutcome`` values:
each has ``pre_physics`` solver verdicts and ``post_physics`` validator reports
with ``check``, ``passed``, ``reason`` and ``configuration`` attributes.
``result.accepted_indices`` and ``result.rejections`` identify the source
environment and reset batch, not an index into a stored pool.

The collector returns empty pose lists and rejection reasons when no candidates
pass. The recording wrapper separately enforces ``min_layouts``, validates replay
compatibility and adds sampling metadata to the saved JSONL. Asset definitions
are optional for collection; ``scene_assets`` supplies metadata used to distinguish
embodiments from articulated task objects, not a required definition for each root.

Collection omits progress messages by default; set ``log_progress=True`` to
print validator settings, physics-step progress and batch results. The recording
script enables these messages. Both APIs perform sampling resets; an initial
``env.reset()`` is unnecessary. ``validate_pool_layouts()`` remains a separate tool
for grading every stored candidate without consuming the pool.

Each collection batch calls ``sample_and_settle_batch`` to capture source solver
results, initial and final root/link poses, and final root velocities in a
``SettledBatch``. ``evaluate_settled_batch`` evaluates these captured values and
returns a ``PlacementOutcome`` for every environment, including solver failures.
Evaluation does not read a live environment, so a captured batch can be evaluated
after further simulation.

Custom checks subclass ``PostPhysicsPlacementValidator`` in
``isaaclab_arena.offline_placement.post_physics_validation``. Implement
``validate(batch: SettledBatch)`` with one report per ID in ``batch.env_ids``, using
``self.report`` to retain settings and results. The batch includes all measured
root poses and velocities, and only the selected task-object link poses.
To add an importable check:

.. code-block:: bash

   +settle.validators.support._target_=my_project.validators.SupportValidator
