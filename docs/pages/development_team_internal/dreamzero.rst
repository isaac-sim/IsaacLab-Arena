DreamZero
=========

DreamZero is a World Action Model with a checkpoint fine-tuned on DROID
(``GEAR-Dreams/DreamZero-DROID``). Arena ships a thin WebSocket client
(``DreamZeroRemotePolicy``) that talks to a DreamZero inference server running remotely.

.. note::

   DreamZero inference requires an H100-class GPU. This guide starts the server
   on an OSMO cluster and runs Arena locally. Before continuing, follow the
   :ref:`OSMO setup instructions <osmo-setup>` and identify an H100-capable
   pool. Replace ``isaac-dev-h100-01`` below with the name of that pool.

The setup uses two terminals: the **DreamZero server** (terminal 1, hosts the model remotely on OSMO)
and the **Arena Experiment Runner** (terminal 2, runs the simulation and exchanges observations
and actions with the server).

Terminal 1 — DreamZero server
------------------------------

**Start the prebuilt server on OSMO**

Arena provides a prebuilt server image with the DreamZero code and public
``GEAR-Dreams/DreamZero-DROID`` checkpoint. Submit the server workflow to the H100 pool:

.. code-block:: bash

   osmo workflow submit isaaclab_arena_dreamzero/docker/dreamzero_inference_server.yaml \
       --pool isaac-dev-h100-01 \
       --set port=5000

The command prints the workflow ID. Once its ``serve`` task is running, forward the
server port to your machine:

.. code-block:: bash

   osmo workflow port-forward <WORKFLOW_ID> serve --port 5000

Leave the port-forward command running. The workflow uses a single H100 GPU and serves
the baked checkpoint from ``/workspace/dreamzero/checkpoints/DreamZero-DROID``.

Terminal 2 — Experiment Runner
------------------------------

Arena includes a one-Run YAML configuration. It selects the environment, the DreamZero policy,
the language instruction, and a three-episode rollout:

.. dropdown:: Configuration file (``droid_pnp_dreamzero_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../isaaclab_arena_environments/experiment_configs/droid_pnp_dreamzero_experiment.yaml
      :language: yaml

The policy configuration uses ``localhost:5000`` by default, matching the port forward
from terminal 1.

**Run DreamZero closed-loop**

Open a second terminal, enter the Arena container with ``./docker/run_docker.sh``, and
start the rollout:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_dreamzero_experiment.yaml

The runner reads the other values from YAML and records the Run under the name
``droid_pnp_dreamzero``. Run headless by replacing ``--viz kit`` with ``--headless``.
