Author a Workbench Environment with a Coding Agent
==================================================

This workflow uses a coding agent to author and validate a workbench cleanup task.

Prerequisites
-------------

Complete the :doc:`Arena installation <../../quickstart/installation>`, then
set up either the native uv environment or the Arena Docker container:

:runtime_setup_tabs:

Install and authenticate your favourite coding agent on the host, then start it from the repository root.
In our example, we use Codex with GPT-6 Astra and high reasoning effort.

.. include:: ../../_includes/codex_setup.rst.inc

Step 1: Viewing an Existing Agent-Authored Environment
------------------------------------------------------

Run this step in the Arena runtime (uv environment or inside the Docker container).

Arena includes an agent-generated workbench scene and task:

* ``isaaclab_arena_environments/workbench_cleanup/scene_tool_bin_scope.yaml``
* ``isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml``

The scene contains two tools in the open top drawer, an open socket set, an
oscilloscope, and a white bin. Preview it in Kit:

.. code-block:: bash
   :caption: Arena runtime (activated uv environment or Arena Docker shell)

   python isaaclab_arena/evaluation/policy_runner.py \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --robot_control droid_differential_ik \
     --policy_type zero_action \
     --num_steps 1000 \
     --viz kit

You should see the workbench scene with objects and the DROID robot holding its pose. The preview
exits after 1,000 steps. This preview does not attempt the task.

Next, run the recipe with the :doc:`Object Control Runner <../../agentic_tools/object_control>`
to see a force-based demonstration of all four subtasks:

.. code-block:: bash
   :caption: Arena runtime (activated uv environment or Arena Docker shell)

   python isaaclab_arena/agentic_tools/object_control_runner.py \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --commands isaaclab_arena_environments/workbench_cleanup/object_control/full_cleanup.jsonl \
     --output $HOME/eval/workbench/full_cleanup_demo \
     --seed 42 \
     --placement_seed 42 \
     --visualizer kit

On success, the runner prints the following summary:

.. code-block:: text

   TASK_SUCCESS {"success": [true], "report": "/home/<user>/eval/workbench/full_cleanup_demo/validation.json"}

The recipe applies bounded forces and joint efforts to close the drawer and toolcase, place the bin on the overhead
shelf, and press the oscilloscope power button. The agentic authoring pipeline leverages this tool to validate the
object physics and checks for task success without invoking a robot policy.

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

Start Codex from the repository root on the host:

.. code-block:: bash
   :caption: Host terminal (repository root)

   codex --model gpt-6-astra -c 'model_reasoning_effort="high"' --yolo

.. note::

   The command above will start Codex in ``yolo`` mode, which means that the agent
   runs commands without asking for approval and outside Codex's sandbox.
   Do this at your own risk.

Enter a prompt like the one below:

.. code-block:: text
   :caption: Prompt in the host Codex session

   Set up an environment for evaluation of a workbench cleanup task.
   Use the simready_central workbench assets registered in
   isaaclab_arena/assets/background_library.py and
   isaaclab_arena/assets/object_library.py.

   On the tabletop of the workbench, there is a clutter of five tools, such as
   wrenches and screwdrivers, at the center. The top drawer of the workbench is
   fully open.

   Task: put the tools into the drawer one by one, then close the drawer.

   Validate task success correctly with a force-based demo script.

The agent uses Arena's ``author-environment`` skill to author and validate the requested environment.
After Codex finishes (expect 10~20 minutes), it should return a successful force-based demonstration video like the one below:

.. video:: ../../../images/workbench_cleanup_assets/tools_in_drawer.mp4
   :width: 100%
   :alt: Force-based demonstration placing five tools into the top workbench drawer and closing it
   :autoplay:
   :loop:
   :muted:
   :playsinline:
   :caption: Successful five-tool drawer cleanup using bounded forces and joint efforts.
