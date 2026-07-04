DreamZero
=========

DreamZero is a World Action Model with a checkpoint fine-tuned on DROID
(``GEAR-Dreams/DreamZero-DROID``). Arena ships a thin WebSocket client
(``DreamZeroRemotePolicy``) that talks to a DreamZero inference server running remotely.

.. note::
  DreamZero requires quite a large amount of GPU memory and therefore we provide tools to run this model remotely using OSMO.

The setup uses two terminals: the **DreamZero server** (terminal 1, hosts the model remotely on OSMO)
and the **arena policy runner** (terminal 2, runs the simulation and sends
observations / receives actions over WebSocket + MessagePack).

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

The job starts the WebSocket inference server on the requested port using 2 H100
GPUs. Once the job is running, find its IP from the OSMO job logs; you will pass it to the
policy as ``--dreamzero_host`` below.

Terminal 2 — arena policy runner
----------------------------------

**Run DreamZero closed-loop**

Open a second terminal, enter the Arena container with ``./docker/run_docker.sh``, and
point the arena policy runner at the server. All global and policy-specific flags must
appear **before** the environment name (subcommand); flags like ``--embodiment`` that are
specific to the environment go after it.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy \
     --dreamzero_host <OSMO_JOB_IP> \
     --dreamzero_port 5000 \
     --enable_cameras \
     --num_episodes 3 \
     --language_instruction "Pick up the Rubik's cube and place it in the bowl." \
     pick_and_place_maple_table \
       --embodiment droid_abs_joint_pos \
       --pick_up_object rubiks_cube_hot3d_robolab \
       --destination_location bowl_ycb_robolab \
       --hdr home_office_robolab

Defaults: ``--dreamzero_host localhost``, ``--dreamzero_port 5000``,
``--dreamzero_embodiment_adapter droid`` (the only embodiment adapter the checkpoint
currently supports). Run headless by swapping ``--viz kit`` for ``--headless``.
