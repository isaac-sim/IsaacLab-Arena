Server-Client Mode With GR00T
=============================

Arena supports running the simulation and the policy model in separate containers connected via a
lightweight RPC protocol. This is useful when the policy environment has different dependencies or
needs to run on a different machine. You can see more details in the :doc:`../concepts/policy/concept_remote_policies_design` page.

It requires two containers to run:

- The **Base** Isaac Lab Arena container, started via ``docker/run_docker.sh``.
- A separate **GR00T policy server** container, started via ``docker/run_gr00t_server.sh``.

A typical workflow is:

1. Start the Base container for simulation and evaluation:

.. code-block:: bash

  bash docker/run_docker.sh

2. In a second terminal, start the GR00T policy server container:

.. code-block:: bash

  bash docker/run_gr00t_server.sh \
    # The models directory is the directory that contains the GR00T model checkpoint.
    -m ${MODELS_DIR} \
    --host 127.0.0.1  \
    --port 5555 \
    --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy \
    --policy_config_yaml_path ${POLICY_CONFIG_YAML_PATH} # e.g. isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml

3. Inside the Base container, run the evaluation script with a
   client-side remote policy.

.. code-block:: bash

  python -m isaaclab_arena/evaluation/policy_runner.py \
    --visualizer kit \
    --policy_type isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy \
    --remote_host 127.0.0.1 \
    --remote_port 5555 \
    # for example: 2000
    --num_steps ${NUM_STEPS} \
    # for example: 10
    --num_envs ${NUM_ENVS} \
    --enable_cameras \
    --remote_kill_on_exit \
    # for example: gr1_open_microwave, put_item_in_fridge_and_close_door, etc.
    ${ARENA_ENVIRONMENT_NAME} \
    # for example: gr1_joint, g1_wbc_joint, etc.
    --embodiment ${ROBOT_EMBODIMENT_NAME} \
    # for example: cracker_box, ketchup_bottle_hope_robolab, ranch_dressing_hope_robolab, etc.
    --object ${OBJECT_NAME}

This setup sets the Isaac Lab Arena simulation environment to the specified environment and runs the GR00T policy in a separate container.
The simulation environment and the policy model communicate via the remote policy interface.

If you want to host other policy models as remote servers, you can follow the same pattern: create
a dedicated server Dockerfile and launcher script (similar to ``docker/Dockerfile.gr00t_server``
and ``docker/run_gr00t_server.sh``), and point it to a custom ``ServerSidePolicy``
implementation as described in :doc:`../concepts/policy/concept_remote_policies_design`.
