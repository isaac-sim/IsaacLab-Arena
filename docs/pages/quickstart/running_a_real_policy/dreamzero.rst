DreamZero
=========

DreamZero is a World Action Model with a checkpoint fine-tuned on DROID
(``GEAR-Dreams/DreamZero-DROID``). Arena ships a thin WebSocket client
(``DreamZeroRemotePolicy``) that talks to a DreamZero inference server running remotely.

.. note::
  DreamZero requires quite a large amount of GPU memory and therefore we provide tools to run this model remotely using OSMO.

The setup uses two terminals: the **DreamZero server** (terminal 1, hosts the model remotely on OSMO)
and the **Arena Experiment Runner** (terminal 2, runs the simulation and exchanges observations
and actions with the server).

Terminal 1 — DreamZero server
------------------------------

**Build and push the server image**

Arena ships everything needed to build the DreamZero inference server image and run
it as an OSMO job. Log in to the NGC registry once:

.. code-block:: bash

   docker login nvcr.io -u '$oauthtoken' -p <YOUR_NGC_API_KEY>

Then build and push (bakes the public ``GEAR-Dreams/DreamZero-DROID`` checkpoint into the image):

.. code-block:: bash

   ./isaaclab_arena_dreamzero/docker/push_to_ngc.sh -p
   # Optional overrides:
   #   -t <tag>  Image tag (default: latest)
   #   -n <name> Override image name (default: dreamzero_inference_server)
   #   -R        Build without cache

This produces ``nvcr.io/nvidian/dreamzero_inference_server:<tag>`` with the
``GEAR-Dreams/DreamZero-DROID`` checkpoint baked in at
``/workspace/dreamzero/checkpoints/DreamZero-DROID``.

**Submit the OSMO job**

.. code-block:: bash

   osmo workflow submit isaaclab_arena_dreamzero/docker/dreamzero_inference_server.yaml \
       --set port=5000

The job starts the WebSocket inference server on the requested port using a single H100
GPU. Once the job is running, find its IP in the OSMO job logs. You will pass it to the
runner below.

Terminal 2 — Experiment Runner
------------------------------

Arena includes a one-Run YAML configuration. It selects the environment, the DreamZero policy,
the language instruction, and a three-episode rollout:

.. dropdown:: Configuration file (``droid_pnp_dreamzero_experiment.yaml``)
   :animate: fade-in

   .. literalinclude:: ../../../../isaaclab_arena_environments/experiment_configs/droid_pnp_dreamzero_experiment.yaml
      :language: yaml

The policy configuration uses ``localhost:5000`` by default. This works when the DreamZero server
is local or its port is forwarded to your machine.

**Run DreamZero closed-loop**

Open a second terminal and enter the Arena container with ``./docker/run_docker.sh``. Replace
``OSMO_JOB_IP`` with the address from the server job, then start the rollout:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_dreamzero_experiment.yaml \
     runs.droid_pnp_dreamzero.policy.remote_host=OSMO_JOB_IP

The runner reads the other values from YAML and records the Run under the name
``droid_pnp_dreamzero``. Omit the final override when the server is available on ``localhost``.
Run headless by replacing ``--viz kit`` with ``--headless``.
