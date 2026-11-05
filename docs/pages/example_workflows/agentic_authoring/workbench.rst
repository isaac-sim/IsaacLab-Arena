Author a Workbench Environment with Codex
=========================================

This workflow uses a coding agent to inspect Arena's current APIs, author a
reviewable environment graph, and run the result. Codex edits normal YAML files
in the checkout, reuses existing Arena features, and runs focused validation
before handing the changes back for review.

The example first previews a checked-in agent-authored environment and its
force-based demonstration, then prompts Codex to author a related workbench
cleanup task.

Prerequisites
-------------

Complete the :doc:`Arena installation <../../quickstart/installation>`, then
set up either the native uv environment or the Arena Docker container:

:runtime_setup_tabs:

Run the commands below from the repository root in that runtime. Install and
authenticate the Codex CLI on the host, then start it from the repository root
so it can inspect Arena's source and project guidance.

Step 1: Prepare the Workbench Assets
------------------------------------

Download and process the registered workbench assets into the runtime cache:

.. code-block:: bash

   python isaaclab_arena_environments/workbench_cleanup/prebuild_assets.py

The command prepares all assets used by the checked-in workbench scene and
prints their cached USD paths. Subsequent authoring, previews, and demos reuse
these outputs.

Step 2: Viewing an Existing Agent-Authored Environment
------------------------------------------------------

Arena includes a reviewed workbench scene and full cleanup task:

* ``isaaclab_arena_environments/workbench_cleanup/scene_tool_bin_scope.yaml``
* ``isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml``

The scene contains two tools in the open top drawer, an open socket set, an
oscilloscope, and a white bin. Preview it in Kit:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --viz kit \
     --device cpu

The checked-in force-based recipe validates all four task steps:

.. code-block:: bash

   python isaaclab_arena/agentic_tools/object_control_runner.py \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --commands isaaclab_arena_environments/workbench_cleanup/object_control/full_cleanup.jsonl \
     --output /eval/workbench/full_cleanup_demo \
     --seed 42 \
     --placement_seed 42 \
     --visualizer kit

The recipe closes the drawer and toolcase, places the bin on the overhead
shelf, and presses the oscilloscope power button. The runner applies bounded
forces and joint efforts directly to scene objects, then checks each native
task predicate and the final sequential-task success condition.

.. figure:: ../../../images/workbench_cleanup_assets/full_cleanup.gif
   :align: center
   :width: 100%
   :alt: Force-based object control completing the workbench cleanup task

   Complete all four workbench cleanup subtasks.

Step 3: Authoring a New Workbench Environment
---------------------------------------------

Start Codex from the repository root:

.. code-block:: bash

   codex

Paste a concise task description like the prompt used for Arena's agentic
authoring examples:

.. code-block:: text

   Set up an environment for evaluation of a workbench cleanup task using
   PhysX simulation. Use the prepared workbench assets registered in
   isaaclab_arena_environments/workbench_cleanup/assets.py.

   On the tabletop of the workbench, there is a clutter of five tools, such as
   wrenches and screwdrivers, at the center. The top drawer of the workbench is
   fully open.

   Task: put the tools into the drawer one by one, then close the drawer.

   Validate task success correctly with a force-based demo script.

Review the resulting environment graph, task definition, and demo before
running them. Verify asset registrations, prim paths, joint names, SI units,
placement clearances, reset behavior, and each task success condition.
