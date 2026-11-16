Workbench Cleanup Catalog
=========================

The workbench cleanup examples contain four DROID atomic tasks and a sequential
full-cleanup task defined as environment graph YAML specs under
``isaaclab_arena_environments/workbench_cleanup/``. Each task has a VLM
Experiment Definition under ``experiment_configs/``.

Run an Experiment
-----------------

Run from the repository root in the Arena runtime. The workbench assets
currently require NVIDIA Storage API access; follow the prerequisites in
:doc:`agentic_authoring/workbench` to configure it. Set ``NV_API_KEY`` for the
configured internal endpoint and ``openai/openai/gpt-6-astra`` model, then run
one drawer-closing episode:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/workbench_cleanup/experiment_configs/close_drawer_vlm_goal_experiment.yaml \
     --viz none --device cuda:0 \
     --record_camera_video \
     --output_base_dir artifacts/workbench_vlm_results \
     runs.close_drawer.rollout_limit.num_episodes=1

Replace both ``close_drawer`` occurrences with ``close_toolcase``,
``turn_off_scope``, ``bin_shelf``, or ``full_cleanup`` to run another task.

These experiments use ``VLMGoalPolicy`` with the DROID embodiment adapter,
following the gear-insertion VLM example. The policy receives camera images,
robot state, and **privileged simulator object poses**. All five experiments
use the standard policy. Selected articulation joint positions are supplied
through ``joint_positions``; the toolcase and full cleanup additionally use
``grasp_points`` to report the lid's live contact point, surface normal, and
closing tangent. These points are configured in the experiment YAML; see
:ref:`vlm-grasp-point-observations` for the reusable observation format.

Task Catalog
------------

Each row links to the task and experiment YAMLs and shows one recorded VLM
episode. Each GIF shows two synchronized external robot cameras. Playback
follows simulation time, so time spent waiting for model responses does not
appear as pauses. The full-cleanup recording uses 9x playback. These recordings
predate the general grasp-point adapter refactor; they are not new policy-success
measurements for that refactor.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45
   :class: workbench-catalog

   * - Task
     - Configuration
     - Validation outcome
     - VLM policy execution
   * - Push the top-left drawer closed.
     - `Task YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/task_close_drawer.yaml>`__

       `Experiment YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/experiment_configs/close_drawer_vlm_goal_experiment.yaml>`__
     - Success in 346 control steps.
     - .. image:: ../../images/workbench_cleanup_assets/vlm/close_drawer_success.gif
          :alt: DROID closes the top-left drawer, shown from two robot cameras
          :width: 100%
   * - Close the socket case lid.
     - `Task YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/task_close_toolcase.yaml>`__

       `Experiment YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/experiment_configs/close_toolcase_vlm_goal_experiment.yaml>`__
     - Success in 105 control steps.
     - .. image:: ../../images/workbench_cleanup_assets/vlm/close_toolcase_success.gif
          :alt: DROID pushes the socket case lid closed, shown from two robot cameras
          :width: 100%
   * - Press the oscilloscope power button.
     - `Task YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/task_turn_off_scope.yaml>`__

       `Experiment YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/experiment_configs/turn_off_scope_vlm_goal_experiment.yaml>`__
     - Success in 74 control steps.
     - .. image:: ../../images/workbench_cleanup_assets/vlm/turn_off_scope_success.gif
          :alt: DROID presses the oscilloscope power button, shown from two robot cameras
          :width: 100%
   * - Pick up the white bin and place it on the overhead shelf.
     - `Task YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/task_bin_shelf.yaml>`__

       `Experiment YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/experiment_configs/bin_shelf_vlm_goal_experiment.yaml>`__
     - Failure after 151 control steps.
     - .. image:: ../../images/workbench_cleanup_assets/vlm/bin_shelf_failure.gif
          :alt: DROID lifts the white bin but drops it before placing it on the overhead shelf
          :width: 100%
   * - Full cleanup: close the drawer, close the toolcase, place the bin on the
       shelf, then turn off the oscilloscope.
     - `Task YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml>`__

       `Experiment YAML <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/workbench_cleanup/experiment_configs/full_cleanup_vlm_goal_experiment.yaml>`__
     - Timeout after 4,500 control steps. The drawer completed at step 283;
       the other three subtasks remained incomplete.
     - .. image:: ../../images/workbench_cleanup_assets/vlm/full_cleanup_failure.gif
          :alt: DROID closes the drawer, then attempts the bin while the toolcase remains open; the full task times out
          :width: 100%
