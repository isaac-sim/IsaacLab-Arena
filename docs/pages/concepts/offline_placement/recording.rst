Record Settled Placement Poses
==============================

Use ``record_placement_layouts.py`` to prepare reusable initial poses before
policy evaluation. It solves placement relations, advances physics, and records
the final poses of accepted layouts. Replay loads those poses on reset without
solving or settling them again.

This walkthrough uses two existing Robolab tasks: a clamp scene that passes and
a smartphone scene that demonstrates rejection. Both use four parallel environments
and four batches. For acceptance rules and the Python API, see
:doc:`recording_design`.

.. toctree::
   :hidden:

   recording_design

1. Record the clamp scene
-------------------------

Use a standard Docker image built from this checkout, as described in
:doc:`../../quickstart/installation`. The reference runtime uses Isaac Sim 6.1.0,
Newton 1.5.2 and Warp 1.16.0, without optional cuRobo IK validation.
The launcher reuses existing images, so an older local image may have incompatible
dependencies. To build and launch a separate image, run from the host repository root:

.. code-block:: bash

   ./docker/run_docker.sh -n isaaclab_arena_sqa -s record-replay

Run the remaining commands from the repository root inside that container, with a
workstation display available. Recording and replay use the same container.
The examples use CPU PhysX and show the Kit viewport.

The existing scenes use about 1 cm of ``On`` release clearance. This exceeds the
recorder's default 2 mm shift limit. The commands explicitly allow 15 mm of root
translation and 5 mm of articulation-link translation: the reference Droid run
moved its links about 4 mm. Rotation limits remain 2 degrees and velocity limits
remain 0.1 m/s and 0.1 rad/s. These are example settings, not changed defaults;
use limits appropriate to the accuracy needed by your evaluation.

Physics also advances the robot, which may move or contact objects. The checks
measure final root speeds and initial-to-final root/link shifts; they do not prove
that every joint has stopped. Only root poses are saved. See
:ref:`recording_robot_motion` before relying on replay to match robot contact geometry.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml \
       output=outputs/placements/clamp.jsonl \
       num_envs=4 env_spacing=2 layouts_per_env=4 seed=42 \
       'viewer_eye=[4.0,4.0,6.3]' 'viewer_lookat=[0.6,0.6,0.3]' \
       settle.num_steps=120 \
       settle.validators.pose_shift.max_translation_m=0.015 \
       settle.validators.articulation_link_shift.max_translation_m=0.005 \
       render=true --device cpu --viz kit

``viewer_eye`` and ``viewer_lookat`` set the camera in simulation-world coordinates
before the first batch. These values produce the four-environment overview shown
below; omitting them keeps the task's default camera. Set both together.

The viewport shows four tables and robots. Each batch applies a different solved
layout, then advances 960 physics steps (120 environment steps with decimation 8).
The clamp scene varies positions without randomizing tool orientations and has a
small initial drop. The console prints enabled checks
and their settings, physics-step progress, and acceptance counts per batch.

.. image:: ../../../images/offline_placement/clamp_recording.gif
   :alt: Four batches of clamp layouts in four parallel environments.
   :width: 100%

This eight-second GIF shows short excerpts from four batches: the first second
of settling and a brief final view of each layout. Each batch still runs all
960 physics steps. Use the console results below to check acceptance.

The reference run accepted 15 of 16 candidates. One black hammer moved 15.7 mm,
exceeding the 15 mm root-shift limit, so that layout was excluded. Console excerpt:

.. code-block:: text

   [recording] batch 1/4: 480/960 physics steps
   [recording] batch 1/4: 960/960 physics steps
   [recording] batch 1/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 4/16 validated, 4 accepted
   [recording] batch 2/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 8/16 validated, 8 accepted
   [recording] batch 3/4: 4 solutions, 4 passed solver validation, 4 passed post-physics validation; overall 12/16 validated, 12 accepted
   [recording] batch 4/4: 4 solutions, 4 passed solver validation, 3 passed post-physics validation; overall 16/16 validated, 15 accepted
   Saved 15/16 accepted layouts: outputs/placements/clamp.jsonl
     Rejected 1: pose_shift: black_hammer: moved 0.015724 m and rotated 1.112 deg; limits 0.015 m, 2 deg

Counts can vary with the simulator, assets and solver configuration. Every saved
layout must pass all enabled, applicable checks. An existing output file is never
overwritten; choose a new path for each recording.

2. Inspect the saved poses
--------------------------

.. code-block:: bash

   wc -l outputs/placements/clamp.jsonl
   head -n 1 outputs/placements/clamp.jsonl | /isaac-sim/python.sh -m json.tool

The line count must match the recorder's accepted count. Each line contains one
complete layout. ``source``, ``poses`` and ``validation`` are fields inside
``variations["scene.relation_placement"]``. Check that:

* ``source`` is ``"settled"``;
* ``poses`` contains scene names, including ``spring_clamp``, the bins and ``robot``;
* positions and quaternions contain finite numbers;
* all three entries in ``validation.post_physics`` have ``passed: true`` and the
  settings shown in the command above.

Positions are in metres in the local environment frame; quaternions are XYZW.
``validation.pre_physics`` holds the solver verdicts for the initial candidate.
``validation.sampling`` records the physics duration. Only root poses are saved,
not robot joint states.

3. Replay the accepted layouts
------------------------------

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
at each reset before attributing later motion to replay. With one environment, each reset loads the next record
and wraps after the last one. Restarting the command starts from the first record.
The command exits after three episodes; each runs for up to 70 seconds of
simulation, which may differ from wall-clock time.

Open ``outputs/placements/evaluation/<timestamp>/index.html`` for the results.
The adjacent ``episode_results_rank0.jsonl`` contains per-episode metrics. Zero
task success is expected with this policy. Inspect the viewport and any
reported object movement; this check does not measure manipulation performance.
Include the command, console log, input JSONL and evaluation results when reporting
a discrepancy.

Exact root reset poses do not guarantee matching joint geometry or future
trajectories. Matching joint geometry requires deterministic joint initialization
as well as the same scene, backend and device.
See :doc:`../object_placement/relations` for parallel and partial-reset selection.

4. Check rejection with the smartphone scene
--------------------------------------------

Use the same settings on the existing smartphone task:

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/robolab/tasks/smartphone_in_bin.yaml \
       output=outputs/placements/smartphone.jsonl \
       num_envs=4 env_spacing=2 layouts_per_env=4 seed=42 \
       'viewer_eye=[4.0,4.0,6.3]' 'viewer_lookat=[0.6,0.6,0.3]' \
       settle.num_steps=120 \
       settle.validators.pose_shift.max_translation_m=0.015 \
       settle.validators.articulation_link_shift.max_translation_m=0.005 \
       render=true --device cpu --viz kit

.. image:: ../../../images/offline_placement/smartphone_recording.gif
   :alt: Four batches of smartphone-scene layouts in four parallel environments, showing object motion.
   :width: 100%

All 16 candidates passed solver validation but failed post-physics validation,
so no recording was written. The mouse moved or tumbled beyond the pose-shift
limits. Several layouts also exceeded the separate final-velocity limits.
The GIF uses the same short excerpts as the clamp example.

Console excerpt:

.. code-block:: text

   [recording] batch 1/4: 480/960 physics steps
   [recording] batch 1/4: 960/960 physics steps
   [recording] batch 1/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 4/16 validated, 0 accepted
   [recording] batch 2/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 8/16 validated, 0 accepted
   [recording] batch 3/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 12/16 validated, 0 accepted
   [recording] batch 4/4: 4 solutions, 4 passed solver validation, 0 passed post-physics validation; overall 16/16 validated, 0 accepted
   AssertionError: Accepted 0 layouts; need 1. Rejections: {...}

One of the reported reasons was:

.. code-block:: text

   pose_shift: computer_mouse: moved 0.023071 m and rotated 14.295 deg; limits 0.015 m, 2 deg

A rejected run exits with an error and writes no file when fewer than
``settle.min_layouts`` candidates pass (default 1). Check the named object and
reason in the rejection summary:

* ``physics_settled``: final velocity exceeds the limits. Inspect contacts and the
  source arrangement; increase ``settle.num_steps`` only if the motion is transient.
* ``pose_shift``: a root moved or rotated too far from its solved pose.
* ``articulation_link_shift``: a link moved too far relative to its root. Root-only
  replay cannot reproduce an altered joint configuration.
* ``missing required solver checks``: make the named check available or fix the
  source configuration. Do not remove required checks merely to obtain a file.

Other runtimes can produce different rejection counts. Inspect the reported
conditions instead of assuming that a task name guarantees acceptance or rejection.

Without a display
-----------------

For recording, replace ``render=true --viz kit`` with
``render=false --viz none``. For replay, replace ``--viz kit`` with ``--viz none``.
The output and checks remain the same. Recording settings use Hydra ``key=value``
syntax; launcher settings retain ``--flag`` syntax.
