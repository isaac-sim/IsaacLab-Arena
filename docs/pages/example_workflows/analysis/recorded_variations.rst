Run an Evaluation with Recorded Variations
==========================================

Recorded variations let an evaluation reproduce the same placement, lighting,
object-presence, and camera conditions with a different number of parallel
environments. This workflow uses the ``clamp_in_right_bin`` graph environment,
whose episode timeout is set to one second, and the local zero-action policy.

The workflow has three stages:

1. record eight settled placement layouts;
2. run 64 episodes in four environments while recording all enabled variations;
3. replay those 64 records in five environments and compare the variation
   payloads line by line.

The generated JSONL files stay below ``outputs/recorded_variations_workflow``
and are not source files.


Record Settled Placement Layouts
--------------------------------

From the repository root, record eight layouts that pass the configured
post-physics checks:

.. code-block:: bash

   python isaaclab_arena/scripts/record_placement_layouts.py \
     env_spec=isaaclab_arena_environments/robolab/tasks/clamp_in_right_bin.yaml \
     output=outputs/recorded_variations_workflow/placements.jsonl \
     num_envs=4 env_spacing=2 layouts_per_env=2 \
     min_layouts=8 max_batches=5 seed=42 presets=physx \
     settle.num_steps=120 \
     settle.validators.pose_shift.max_translation_m=0.015 \
     --device cuda:0

The verified run accepted all eight requested layouts. Each output row records
the complete scene-root poses under the top-level ``placement`` key.


Record a 64-Episode Evaluation
------------------------------

The maintained Experiment Definition configures four parallel environments,
the zero-action policy, and 64 episodes:

.. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/recorded_variations_workflow_experiment.yaml
   :language: yaml

It enables two build-time variations:

* ``light.hdr_image``
* ``light.color``

It also enables two run-time variations:

* ``red_hammer.disappear`` for a distractor that is not part of the task;
* ``droid_abs_joint_pos.camera_extrinsics_wrist_camera``.

The placement recording supplies the scene-level relation-placement samples.
Other enabled variations are sampled live and recorded in each episode result.

Run the Experiment:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/experiment_configs/recorded_variations_workflow_experiment.yaml \
     --viz none \
     --experiment_output_directory \
       outputs/recorded_variations_workflow/record

The canonical recording is written to:

.. code-block:: text

   outputs/recorded_variations_workflow/record/recorded_variations/episode_results_rebuild0.jsonl

The verified run completed and wrote exactly 64 episode rows. Zero action does
not solve the task; task failure is expected and does not indicate an execution
failure.


Replay with Five Parallel Environments
--------------------------------------

Run the same Experiment again, overriding only the parallel environment count,
the replay source, and the output directory:

.. code-block:: bash

   export RECORDED_VARIATIONS="outputs/recorded_variations_workflow/record/recorded_variations/episode_results_rebuild0.jsonl"

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config \
       isaaclab_arena_environments/experiment_configs/recorded_variations_workflow_experiment.yaml \
     --viz none \
     --experiment_output_directory \
       outputs/recorded_variations_workflow/replay \
     runs.recorded_variations.environment_builder.num_envs=5 \
     "runs.recorded_variations.recorded_variation_samples_paths=[${RECORDED_VARIATIONS}]"

Verify the episode count, source order, and variation payloads:

.. code-block:: bash

   python - <<'PY'
   import json
   from pathlib import Path

   root = Path("outputs/recorded_variations_workflow")
   source_path = root / "record/recorded_variations/episode_results_rebuild0.jsonl"
   replay_path = root / "replay/recorded_variations/episode_results_rebuild0.jsonl"
   source = [json.loads(line) for line in source_path.read_text().splitlines()]
   replay = [json.loads(line) for line in replay_path.read_text().splitlines()]

   assert len(source) == len(replay) == 64
   for index, (source_row, replay_row) in enumerate(zip(source, replay)):
       assert replay_row["replay_source_record_index"] == index
       assert replay_row["placement"] == source_row["placement"]
       assert replay_row["variations"] == source_row["variations"]
   print("Matched all 64 recorded placement and variation rows.")
   PY

The verified replay completed and wrote exactly 64 rows. Its
``replay_source_record_index`` values were ``0`` through ``63`` in line order,
and every replayed row's ``placement`` and ``variations`` mappings exactly
matched the corresponding recorded row. Episode metadata such as ``env_id`` and
``timestamp`` can differ because the replay uses a different parallelization.

The matching replay payload contains:

* the relation-placement scene-root poses under ``placement``;
* HDR selection, light color, red-hammer presence, and wrist-camera extrinsics
  under ``variations``.

This verifies that recorded variation order is independent of the number of
parallel simulation environments.
