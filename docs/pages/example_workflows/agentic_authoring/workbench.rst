Author a Workbench Environment with Codex
=========================================

This workflow uses a coding agent to author a reviewable environment graph and
run the result. Codex edits environment YAML files, reuses existing Arena
features, and runs task and environment validation.

The workflow:

* Previews a checked-in agent-authored environment and its force-based demonstration.
* Prompts Codex to author a related workbench cleanup task.
* Evaluates the checked-in atomic tasks with a VLM policy and DROID.

Prerequisites
-------------

Complete the :doc:`Arena installation <../../quickstart/installation>`, then
set up either the native uv environment or the Arena Docker container:

:runtime_setup_tabs:

Run the commands below from the repository root in that runtime. Install and
authenticate the Codex CLI on the host, then start it from the repository root
so it can inspect Arena's source and use the repository's environment authoring
skills.

.. note::

   Running this checked-in example currently requires NVIDIA employee access
   because some assets are hosted on the internal NVIDIA Storage API. The same
   authoring workflow works with public registered assets without this access.

   Authenticate as the user running Arena using the
   `Storage API device-login quickstart
   <https://gitlab-master.nvidia.com/omniverse/usd-storage/ops/sapi-deployments/-/blob/master/doc/user/kit/kit-vm-quickstart.md>`_.
   The runtime needs ``omni.client`` 2.74.0 or newer and the credentials in
   ``~/.nvidia-omniverse/config/auth.toml``; for Docker, these must be available
   inside the container.

   Enable Storage discovery in the shell where you run the commands below:

   .. code-block:: bash

      export OMNI_STORAGE_DISCOVERY=https://rc.storage.omni.nvidia.com


Step 1: Viewing an Existing Agent-Authored Environment
------------------------------------------------------

Arena includes an agent-generated workbench scene and task:

* ``isaaclab_arena_environments/workbench_cleanup/scene_tool_bin_scope.yaml``
* ``isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml``

The scene contains two tools in the open top drawer, an open socket set, an
oscilloscope, and a white bin. Preview it in Kit:

.. code-block:: bash

   python -m isaaclab_arena.scripts.environment_runner \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --viz kit \
     --device cpu

The checked-in object-control recipe validates all four task steps:

.. code-block:: bash

   python -m isaaclab_arena.agentic_tools.object_control_runner \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --commands isaaclab_arena_environments/workbench_cleanup/object_control/full_cleanup.jsonl \
     --output $HOME/eval/workbench/full_cleanup_demo \
     --num_envs 1 \
     --seed 42 \
     --placement_seed 42 \
     --visualizer kit

The recipe closes the drawer and toolcase, places the bin on the overhead
shelf, and presses the oscilloscope power button. The system checks for
composite task success.

.. video:: ../../../images/workbench_cleanup_assets/full_cleanup.mp4
   :width: 100%
   :alt: Force-based object control completing the workbench cleanup task
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :caption: Complete all four workbench cleanup subtasks.

Step 2: Authoring a New Workbench Environment
---------------------------------------------

Start Codex from the repository root:

.. code-block:: bash

   codex

Enter a prompt like the one below:

.. code-block:: text

   Set up an environment for evaluation of a workbench cleanup task.
   Use the simready_central workbench assets registered in
   isaaclab_arena/assets/background_library.py and
   isaaclab_arena/assets/object_library.py.

   On the tabletop of the workbench, there is a clutter of five tools, such as
   wrenches and screwdrivers, at the center. The top drawer of the workbench is
   fully open.

   Task: put the tools into the drawer one by one, then close the drawer.

   Validate task success correctly with a force-based demo script.

After Codex finishes (expect 10~20 minutes), it should return a successful force-based demonstration video like the one below:

.. video:: ../../../images/workbench_cleanup_assets/tools_in_drawer.mp4
   :width: 100%
   :alt: Force-based demonstration placing five tools into the top workbench drawer and closing it
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :caption: Successful five-tool drawer cleanup using bounded forces and joint efforts.

Step 3: Evaluating the Atomic Tasks with a VLM
-----------------------------------------------

The checked-in workbench includes four Experiment Definitions under
``isaaclab_arena_environments/workbench_cleanup/experiment_config/``:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Experiment filename
     - Task
   * - ``close_drawer_vlm_goal_experiment.yaml``
     - Push the top-left drawer closed.
   * - ``close_toolcase_vlm_goal_experiment.yaml``
     - Close the socket case lid.
   * - ``turn_off_scope_vlm_goal_experiment.yaml``
     - Press the oscilloscope power button.
   * - ``bin_shelf_vlm_goal_experiment.yaml``
     - Pick up the white bin and place it on the overhead shelf.

These experiments use ``VLMGoalPolicy`` with the DROID embodiment adapter,
following the gear-insertion VLM example. The policy receives camera images,
robot state, and **privileged simulator object poses**. The drawer and bin use
the standard policy. The toolcase and oscilloscope use
``WorkbenchVLMGoalPolicy``.

Set ``NV_API_KEY`` in the runtime for the configured internal endpoint and
``openai/openai/gpt-6-astra`` model. Start with the drawer task:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/workbench_cleanup/experiment_config/close_drawer_vlm_goal_experiment.yaml \
     --viz none --device cuda:0 \
     --record_camera_video \
     --output_base_dir artifacts/workbench_vlm_results \
     runs.close_drawer.rollout_limit.num_episodes=1

Replace both ``close_drawer`` occurrences with ``close_toolcase``,
``turn_off_scope``, or ``bin_shelf`` to run another task.

Each GIF shows two synchronized external robot cameras from one episode.
Playback follows simulation time, so time spent waiting for model responses
does not appear as pauses.

.. figure:: ../../../images/workbench_cleanup_assets/vlm/close_drawer_success.gif
   :width: 100%
   :alt: DROID closes the top-left drawer, shown from two robot cameras

   Close drawer: success in 346 control steps.

.. figure:: ../../../images/workbench_cleanup_assets/vlm/close_toolcase_success.gif
   :width: 100%
   :alt: DROID pushes the socket case lid closed, shown from two robot cameras

   Close toolcase: success in 105 control steps.

.. figure:: ../../../images/workbench_cleanup_assets/vlm/turn_off_scope_success.gif
   :width: 100%
   :alt: DROID presses the oscilloscope power button, shown from two robot cameras

   Press oscilloscope button: success in 74 control steps.

.. figure:: ../../../images/workbench_cleanup_assets/vlm/bin_shelf_failure.gif
   :width: 100%
   :alt: DROID lifts the white bin but drops it before placing it on the overhead shelf

   White bin to shelf: failure after 151 control steps.
