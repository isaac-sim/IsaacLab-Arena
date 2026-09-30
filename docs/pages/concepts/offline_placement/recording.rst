Record and Replay Placement Poses
=================================

Use ``record_placement_layouts.py`` to save reusable initial poses for policy
evaluation. It solves placement relations, advances physics, and writes accepted
layouts to JSONL. Replay restores those poses on reset without solving or settling
them again.

To run without a viewer, use ``render=false --viz none`` when recording and
``--viz none`` when replaying.

Record Placement Layouts
------------------------

Run this command from the repository root to record layouts from
``clamp_in_right_bin`` in ``outputs/placements/clamp.jsonl``:

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml \
       output=outputs/placements/clamp.jsonl \
       num_envs=4 env_spacing=2 layouts_per_env=4 seed=42 \
       'viewer_eye=[4.0,4.0,6.3]' 'viewer_lookat=[0.6,0.6,0.3]' \
       settle.num_steps=120 \
       settle.validators.pose_shift.max_translation_m=0.015 \
       render=true --device cpu --viz kit

Objects start about 1 cm above their supporting surfaces. The command allows
15 mm of displacement during settling to accommodate this drop.

The example records four batches in four parallel environments:

.. image:: ../../../images/offline_placement/clamp_recording.gif
   :alt: Four batches of clamp_in_right_bin layouts in four parallel environments.
   :width: 100%

Choose a new output path for each recording. Existing files are not overwritten.
See :doc:`../object_placement/validation` for acceptance checks and their settings.

Replay Placement Layouts
------------------------

Set ``environment_builder.placement_layouts_path`` in an
:doc:`Experiment Definition <../concept_arena_experiments>` to load the recording:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/settled_placement_replay_experiment.yaml
   :language: yaml
   :start-at: runs:

Run this command to replay the recording with the experiment definition above:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
       --experiment_config isaaclab_arena_environments/experiment_configs/settled_placement_replay_experiment.yaml \
       --device cpu --viz kit \
       --output_base_dir outputs/placements/evaluation

This example uses a zero-action policy to replay the initial poses.
Recordings restore object and robot root poses, while robot joints use their
configured resets, which randomize the arm configuration in this Droid example.
See :doc:`../object_placement/relations` for the file format and replay behavior.

Record Layouts with Rejections
------------------------------

Run this command to record layouts from ``smartphone_in_bin``, where objects can
move beyond the allowed limits during settling:

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

Only accepted layouts are saved. The console reports rejection reasons, explained
in :ref:`recording_rejection_summary`.

Supported Use Cases and Limits
-------------------------------

The table summarizes the requirements for recording and the state that replay
can restore.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Use case or setting
     - Support and limitations
   * - Rigid and articulation roots
     - Require writable roots, enabled pose resets, fixed root-reset poses and
       zero initial velocity. Replay restores root poses and zeros root velocities.
   * - Object sets and ``RandomAroundSolution``
     - Unsupported for recording. Resolve sets to concrete assets and remove
       ``RandomAroundSolution`` before recording.
   * - Randomized or per-environment root-reset poses
     - Unsupported. Disable other pose-changing variations and callbacks when
       exact root replay is required.
   * - Robot motion and joint state
     - The robot can move and push objects during settling. Robot joints are
       neither recorded nor checked for acceptance and use their configured resets
       during replay.
   * - Articulated task objects
     - Link checks compare initial and final poses relative to the root.
       They do not check joint speeds or motion between those times. Replay
       cannot restore joint changes that occurred during settling.
   * - Geometry and IK after settling
     - Geometry and reachability are not rechecked after settling. Accepted
       layouts do not guarantee valid contacts, reachability or deterministic
       motion during evaluation.
   * - Other variations, such as mass or visibility
     - Sampled normally but not stored. Match these settings and joint
       initialization when comparing recording and replay.
   * - Reusing an environment for collection
     - Collection consumes pool entries and leaves the final state, even on
       failure. Reset events must restore roots, joints and actuator targets.
       Physics-only steps do not advance counters for reset events with a
       minimum step interval.
   * - Batches with solver failures
     - If any candidate passes the required solver checks, physics advances the
       whole batch. Candidates that failed those checks remain rejected.
