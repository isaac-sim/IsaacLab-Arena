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

Step 1: Access the Workbench Assets
-----------------------------------

The registered workbench, socket set, and oscilloscope load prepared USD layers
from ``ovstorage://isaac/Isaac/IsaacLab/Arena/assets/object_library/workbench``.
Their source geometry and textures reference public Isaac Sim S3 assets. The
bin, screwdriver, and wrench use pinned Hugging Face packages through Isaac
Lab's download cache. No asset-processing command is required.

The uploaded layers currently require internal NVIDIA Storage API access.
Authenticate as the user running Arena using the
`Storage API device-login quickstart
<https://gitlab-master.nvidia.com/omniverse/usd-storage/ops/sapi-deployments/-/blob/master/doc/user/kit/kit-vm-quickstart.md>`_.
The runtime needs ``omni.client`` 2.74.0 or newer and the credentials in
``~/.nvidia-omniverse/config/auth.toml``; for Docker, these must be available
inside the container.

Standalone Python entrypoints also need a live Storage discovery registration.
Save the following launcher as ``/tmp/with_workbench_storage.py`` in the runtime.
It retains that registration while running a standard Arena module:

.. code-block:: python

   import os
   import runpy
   import sys
   from pathlib import Path

   import isaacsim  # Initializes ISAAC_PATH for this runtime.

   # Locate the bundled client in either a native install or the Arena container.
   sys.path.insert(0, str(Path(os.environ["ISAAC_PATH"]) / "kit/extscore/omni.client.lib"))
   import omni.client as client

   module = sys.argv.pop(1)
   initialized = client.initialize()
   assert initialized, "Could not initialize the Storage client"
   registration = client.register_storage_from_discovery("https://rc.storage.omni.nvidia.com")
   assert registration is not None and registration.result == client.Result.OK
   try:
       runpy.run_module(module, run_name="__main__")
   finally:
       registration.stop()
       client.shutdown()

Step 2: Viewing an Existing Agent-Authored Environment
------------------------------------------------------

Arena includes a reviewed workbench scene and full cleanup task:

* ``isaaclab_arena_environments/workbench_cleanup/scene_tool_bin_scope.yaml``
* ``isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml``

The scene contains two tools in the open top drawer, an open socket set, an
oscilloscope, and a white bin. Preview it in Kit:

.. code-block:: bash

   python /tmp/with_workbench_storage.py isaaclab_arena.scripts.environment_runner \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --viz kit \
     --device cpu

The checked-in force-based recipe validates all four task steps:

.. code-block:: bash

   python /tmp/with_workbench_storage.py isaaclab_arena.agentic_tools.object_control_runner \
     --env_spec isaaclab_arena_environments/workbench_cleanup/task_full_cleanup.yaml \
     --commands isaaclab_arena_environments/workbench_cleanup/object_control/full_cleanup.jsonl \
     --output /eval/workbench/full_cleanup_demo \
     --num_envs 1 \
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
