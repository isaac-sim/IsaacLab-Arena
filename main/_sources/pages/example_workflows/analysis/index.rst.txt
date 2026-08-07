Analysis
========

Which tested environment conditions are associated with a policy's success? Arena's analysis
workflows turn controlled evaluation sweeps into an answer. Vary factors such as camera pose,
lighting, object mass, or table material, record the outcome of each episode, then estimate a joint
posterior that highlights which factor values are associated with success or failure.

This walkthrough uses an OpenPI policy and varies its wrist-camera position. The Variations
workflow collects the episode results, and the Sensitivity Analysis workflow turns them into a
posterior visualization.

Start or enter the Base Docker container from the repository root:

:docker_run_default:

In another terminal on the host, start the OpenPI server from the repository root:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

Leave the server running while you complete the Variations workflow. You can stop it before
starting Sensitivity Analysis, which reads the saved episode results directly. For installation,
model variants, and server options, see
:doc:`../../quickstart/first_experiments/running_a_real_policy/openpi`.

In the Base Docker container, create the output directory used throughout this walkthrough:

.. code-block:: bash

   export CAMERA_SENSITIVITY_OUTPUT_DIR="/eval/camera_sensitivity_workflow"
   mkdir -p "${CAMERA_SENSITIVITY_OUTPUT_DIR}"

The ``/eval`` directory is mounted from ``$HOME/eval`` on the host by default. Arena requires the
workflow directory to be empty.

.. toctree::
   :maxdepth: 1

   variations
   Sensitivity Analysis <../sensitivity_analysis/sensitivity_analysis>
