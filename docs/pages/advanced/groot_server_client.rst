Server-Client Mode With GR00T
=============================

Arena supports running the simulation and the policy model in separate containers connected via a
lightweight RPC protocol. This is useful when the policy environment has different dependencies or
needs to run on a different machine.

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
      --host 127.0.0.1  \
      --port 5555 \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy \
      --policy_config_yaml_path {policy_config_yaml_path} # e.g. isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml

3. Inside the Base container, run the evaluation script with a
   client-side remote policy (refer to :doc:`../example_workflows/static_manipulation/step_5_evaluation` for full command lines).

This setup cleanly separates the Isaac Lab Arena simulation environment from the GR00T policy
server environment.

If you want to host other policy models as remote servers, you can follow the same pattern: create
a dedicated server Dockerfile and launcher script (similar to ``docker/Dockerfile.gr00t_server``
and ``docker/run_gr00t_server.sh``), and point it to a custom ``ServerSidePolicy``
implementation as described in :doc:`../concepts/concept_remote_policies_design`.
