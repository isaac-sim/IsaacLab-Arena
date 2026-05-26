Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the finetuned GR00T N1.7 policy in closed-loop and evaluating it
in the Arena Unitree G1 Static Apple-to-Plate Task environment using Arena's **server-client
(remote-policy) architecture**. The GR00T policy server, which hosts the finetuned checkpoint, runs
outside the Arena container. The Arena container itself runs only the simulation and a thin GR00T
client that queries the server for actions.

This tutorial uses the dataset you collected in :doc:`step_2_teleoperation` and the model
you trained in :doc:`step_3_policy_training`, or the released checkpoint downloaded below.

Step 1: Start the GR00T policy server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The server runs GR00T's stock ``run_gr00t_server.py`` from the standalone Isaac-GR00T (N1.7) checkout.
Start it **before** launching the client; the client will connect on first inference. Run the
server **outside Docker** in the standalone Isaac-GR00T venv created in :doc:`index`.

The server takes all of its configuration from CLI flags (model checkpoint, embodiment tag, the
modality config from Arena's source tree, and bind host/port). Replace
``/path/to/IsaacLab-Arena`` with the absolute path to your Arena clone and ``${MODEL_PATH}`` with
the finetuned checkpoint directory from :doc:`step_3_policy_training`.

.. dropdown:: Download Pre-trained Model (skip policy post-training)
   :animate: fade-in

   These commands can be used to download the pre-trained GR00T N1.7 static apple checkpoint,
   such that the policy post-training step can be skipped. Run them **outside Docker** in the
   standalone Isaac-GR00T venv before starting the server.

   .. code-block:: bash

      export MODELS_DIR=~/models/isaaclab_arena/static_apple_tutorial
      export MODEL_PATH=$MODELS_DIR/gn1x_tuned_static_apple

      mkdir -p "$MODEL_PATH"
      hf download \
         nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace \
         --repo-type model \
         --local-dir $MODEL_PATH

   If you trained your own checkpoint in :doc:`step_3_policy_training`, set ``MODEL_PATH`` to that
   trainer output instead, for example
   ``$MODELS_DIR/static_apple_n17_finetune/checkpoint-20000``.

.. code-block:: bash

   cd $ISAAC_GR00T_DIR

   uv run python gr00t/eval/run_gr00t_server.py \
      --modality-config-path /path/to/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py \
      --model-path ${MODEL_PATH} \
      --embodiment-tag NEW_EMBODIMENT \
      --device cuda \
      --host 0.0.0.0 \
      --port 5555

The server prints ``Server Ready and listening on 0.0.0.0:5555`` once it is ready for clients.


Step 2: Run Single Environment Evaluation (Arena container)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the server from Step 1 running, launch the Arena client. The client side does not need any
GR00T dependencies — it talks to the server over ZeroMQ — so it runs in the standard **Base**
Arena container. ``Gr00tRemoteClosedloopPolicy`` is Arena's client wrapper around the remote GR00T server.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/static_apple_tutorial
    export MODELS_DIR=/models/isaaclab_arena/static_apple_tutorial

.. note::

   If Kit reports permission errors while writing
   ``/isaac-sim/kit/data/Kit/IsaacLab/3.0/user.config.json`` or cache files, start from a clean
   Arena container/cache or rebuild the Docker image. This can happen when stale Isaac Sim / Kit
   state from another setup is reused with incompatible ownership.

We first run the policy in a single environment with visualization via the GUI. Replace
``<SERVER_HOST>`` below with the IP of the host running Step 1 (or ``localhost`` if it is the
same machine).

.. caution::

   Before running, edit ``model_path`` in
   ``isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml`` to point at
   the checkpoint directory you are serving (for example,
   ``/models/isaaclab_arena/static_apple_tutorial/gn1x_tuned_static_apple`` for the Hugging Face
   checkpoint, or
   ``/models/isaaclab_arena/static_apple_tutorial/static_apple_n17_finetune/checkpoint-20000`` for
   a locally trained checkpoint).
   It must match the ``--model-path`` you passed to ``run_gr00t_server.py`` in Step 1.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
      --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml \
      --remote_host <SERVER_HOST> --remote_port 5555 \
      --num_steps 600 \
      --enable_cameras \
      galileo_g1_static_pick_and_place \
      --object apple_01_objaverse_robolab \
      --destination clay_plates_hot3d_robolab \
      --embodiment g1_wbc_agile_joint

Note the lower ``--num_steps`` (600 instead of 1500): with no walking phase, a successful
static apple-to-plate episode runs for roughly half as long as the loco-manipulation variant.

.. note::

   The 600-step command is intended as a quick smoke test. To get a more representative
   success rate, evaluate complete episodes instead of relying on one short rollout: use
   ``--num_episodes 100`` for a quick estimate or ``--num_episodes 1000`` for a stronger
   estimate. If you prefer ``--num_steps``, this task's 6-second timeout comes from the
   task episode length (``episode_length_s=6.0`` in
   ``galileo_g1_static_pick_and_place_environment.py``). At 50 Hz control, that is about
   300 environment steps per episode, so 100 episodes is roughly ``--num_steps 30000``
   and 1000 episodes is roughly ``--num_steps 300000``.

The evaluation should produce the following output on the console at the end of the evaluation.
You should see similar metrics.

Note that all these metrics are computed over the entire evaluation process, and are affected
by the quality of post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. code-block:: text

   [Rank 0/1] Metrics: {'success_rate': 1.0, 'object_moved_rate': 1.0, 'num_episodes': 1}


Run Parallel Environments Evaluation (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple environments is also supported by the policy runner.
The command below assumes the GR00T server from Step 1 is still running.

Test the policy in 5 parallel environments with visualization via the GUI. The 600-step smoke test
gives each environment enough steps to complete at least one full 6-second episode.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
      --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml \
      --remote_host <SERVER_HOST> \
      --remote_port 5555 \
      --num_steps 600 \
      --num_envs 5 \
      --enable_cameras \
      galileo_g1_static_pick_and_place \
      --object apple_01_objaverse_robolab \
      --destination clay_plates_hot3d_robolab \
      --embodiment g1_wbc_agile_joint

During evaluation, the console prints which environments terminated because the task succeeded or
timed out, and which environments were truncated by runner-level limits.

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([3], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of the evaluation, you should see metrics similar to the single-environment run, but
computed over more episodes because multiple environments are stepped in parallel.

.. code-block:: text

   [Rank 0/1] Metrics: {'success_rate': 1.0, 'num_episodes': 5}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``g1_wbc_agile_joint``, which is
   different from ``g1_wbc_agile_pink`` used during teleoperation recording.
   This is because during tele-operation, the upper body is controlled via target end-effector poses,
   which are realized by using the PINK IK controller, and the lower body is controlled via the AGILE
   WBC policy. The GR00T N1.7 policy is trained on upper body joint positions and lower body WBC
   policy inputs, so we use the joint-control twin (``g1_wbc_agile_joint``) for closed-loop policy
   inference -- it shares the AGILE lower-body backend with the recording embodiment, just bypasses
   PinkIK.

.. note::

   The same-shelf placement makes the static variant slightly easier than the loco-manipulation
   apple-to-plate task: the destination plate is always within arm's reach so the policy
   never has to recover from a mistimed approach, and there are no intermediate locomotion
   phases that can drift off-course. The success criterion is the same contact-sensor
   termination used by the loco-manipulation variant (``force_threshold=0.5 N``,
   ``velocity_threshold=0.1 m/s``), filtered to contacts with the ``--destination`` asset.
   Both values are passed to ``PickAndPlaceTask`` from
   ``isaaclab_arena_environments/galileo_g1_static_pick_and_place_environment.py``; edit the
   ``force_threshold`` / ``velocity_threshold`` kwargs there if you need a different success
   criterion for a new pick-up object or destination.

.. note::

   **Common server-client failure modes.**

   - ``ValueError: Invalid action shape, expected: 23, received: 50.`` — the client's embodiment
     expects a 23-D PinkIK action, but the server is returning a 43-DoF joint chunk. Make sure the
     client uses ``--embodiment g1_wbc_agile_joint`` (joint twin), not
     ``g1_wbc_agile_pink`` (PinkIK twin).
   - ``ModuleNotFoundError`` on the client side — check the client's ``--policy_type``. This
     workflow must use the remote client wrapper
     ``isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy``,
     together with ``--policy_config_yaml_path``.
   - Action shape mismatch on the server (e.g., ``Action key 'left_arm''s horizon must be 40.
     Got 50``) — the action modality used to launch the server does not match the checkpoint's
     training horizon. This workflow trains and serves GR00T N1.7 with ``action_horizon: 40``.
     Re-finetune with the same action ``delta_indices`` horizon, or launch
     ``run_gr00t_server.py`` with the same ``--modality-config-path`` used during finetuning. Keep
     the Arena client YAML's ``action_horizon`` and ``action_chunk_length`` in sync as well (see
     the caution in :doc:`step_3_policy_training`).
