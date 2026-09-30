Variation record and replay
===========================

Arena writes every enabled variation sample into the per-rebuild episode-result
JSONL.  A later run can consume that JSONL directly through
``episode_conditions_path`` so the same build-time and per-episode samples are
applied instead of being drawn again.

The comparison below uses one environment, five episodes, and a one-second
episode timeout.  The left side records live draws and the right side replays
the resulting JSONL.  Each row resets at the same time; the bowl's
present/absent sequence is identical.

.. image:: ../../../images/variations/variation_record_replay.gif
   :width: 100%
   :alt: Side-by-side recording and replay of five identical RoboLab variation conditions
   :align: center

The demonstration shortens the task timeout to make the five resets visible.
The runnable example below keeps the original timeout from RoboLab's
``banana_in_bowl.yaml`` environment definition.

Record five conditions
----------------------

The example Experiment uses the existing RoboLab environment graph at
``isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml`` with a zero-action
policy.  It enables one build-time variation (the HDR image) and two run-time
variations (bowl visibility and wrist-camera extrinsics):

.. literalinclude:: ../../../../isaaclab_arena_environments/robolab/experiment_configs/banana_in_bowl_variation_record_replay.yaml
   :language: yaml

Run the Experiment into an exact, initially empty output directory:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --record_camera_video \
     --experiment_output_directory outputs/variation_record \
     --experiment_config \
       isaaclab_arena_environments/robolab/experiment_configs/banana_in_bowl_variation_record_replay.yaml

The five records are written to
``outputs/variation_record/banana_in_bowl_variations/episode_results_rebuild0.jsonl``.
Each complete JSON object occupies one line.  An abridged pair looks like:

.. code-block:: json

   {"episode_in_env":0,"variations":{"light.hdr_image":"home_office_robolab","bowl.disappear":true,"droid_abs_joint_pos.camera_extrinsics_wrist_camera":[-0.012,0.021,0.006]}}
   {"episode_in_env":1,"variations":{"light.hdr_image":"home_office_robolab","bowl.disappear":false,"droid_abs_joint_pos.camera_extrinsics_wrist_camera":[0.027,-0.008,-0.019]}}

Build-time samples repeat on every line because they describe the shared
environment build.  Run-time samples contain the value drawn for that episode
and environment slot.

Replay the JSONL directly
-------------------------

Point the same Experiment at the recorded file and clear its configured
episode limit:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --record_camera_video \
     --experiment_output_directory outputs/variation_replay \
     --experiment_config \
       isaaclab_arena_environments/robolab/experiment_configs/banana_in_bowl_variation_record_replay.yaml \
     runs.banana_in_bowl_variations.environment_builder.episode_conditions_path=outputs/variation_record/banana_in_bowl_variations/episode_results_rebuild0.jsonl \
     runs.banana_in_bowl_variations.rollout_limit.num_episodes=null

The JSONL supplies the episode budget, so replay runs exactly five conditions.
Explicit ``num_steps`` or ``num_episodes`` limits are rejected, and replay
requires ``num_rebuilds: 1``.  Keep the environment definition and enabled
variation set the same as the recording run.

For multiple parallel environments, Arena assigns recorded conditions through
a global FIFO queue as slots reset.  This preserves the recorded condition set
without requiring the same condition to return to the same environment index.
Replay output includes ``replay_condition_id`` and
``replay_source_episode_results`` for traceability.

Scope
-----

Variation replay covers values owned by Arena variations, including lighting,
camera, mass, and object-visibility samples.  Relation-solver placement layouts
are deliberately excluded from this path.  Use
:doc:`../offline_placement/recording` when exact object poses must be recorded
and replayed; the comparison above fixes the placement seed only to isolate the
variation behavior.

The extraction utility remains available when an editable YAML overlay is
useful:

.. code-block:: bash

   python isaaclab_arena/scripts/extract_episode_conditions.py \
     --episode-results outputs/variation_record/banana_in_bowl_variations/episode_results_rebuild0.jsonl \
     --output outputs/variation_record/conditions.yaml

``episode_conditions_path`` accepts either the original JSONL or the extracted
YAML overlay.
