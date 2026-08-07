Analysis
========

Which environment conditions drive a policy's success, and where does it break down? Arena's
analysis workflows turn controlled evaluation sweeps into an answer. Vary factors such as camera
pose, lighting, object mass, or table material, record the outcome of each episode, then fit a joint
posterior that highlights which factor values are associated with success or failure.

This walkthrough uses an OpenPI policy and varies its wrist-camera position. The Variations
workflow collects the episode results, and the Sensitivity Analysis workflow turns them into a
visual report.

Start or enter the Base Docker container from the repository root:

:docker_run_default:

In another terminal on the host, start the OpenPI server from the repository root:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

Leave the server running. For installation, model variants, and server options, see
:doc:`../../quickstart/first_experiments/running_a_real_policy/openpi`.

.. toctree::
   :maxdepth: 1

   variations
   Sensitivity Analysis <../sensitivity_analysis/index>
