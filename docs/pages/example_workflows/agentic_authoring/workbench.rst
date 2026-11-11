Author a Workbench Environment with Codex
=========================================

This workflow uses a coding agent to author a reviewable environment graph and
run the result. Codex edits environment YAML files, reuses existing Arena
features, and runs task and environment validation.

The workflow:

* Previews a checked-in agent-authored environment and its force-based demonstration.
* Prompts Codex to author a related workbench cleanup task.

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
