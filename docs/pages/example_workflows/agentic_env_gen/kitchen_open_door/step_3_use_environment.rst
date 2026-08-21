Use the Generated Environment
-----------------------------

Once you are satisfied with the environment, you can use it to evaluate a policy on the environment.

For example, you can use the policy runner to evaluate a PI policy on the
environment. For other policy types, see
:doc:`Running a Real Policy <../../../quickstart/running_a_real_policy/index>`.

Open one terminal and run the following command outside the Arena docker container to launch the PI policy server:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

In the other terminal, run the following command to launch the policy runner. The commands below use the
ready-made spec that ships with Arena; to evaluate a spec you generated yourself, point
``--env_spec`` at ``isaaclab_arena_environments/agent_generated/<env_name>.yaml``.


Complete the shared :ref:`agentic-env-gen-prerequisites` before running this command.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
      --enable_cameras \
      --num_envs 1 \
      --num_episodes 2 \
      --env_spec isaaclab_arena_environments/kitchen_bench/droid_open_fridge_lightwheel_kitchen.yaml

.. figure:: ../../../../images/agentic_environment_generation/droid_kitchen_open_door_pi.gif
   :width: 100%
   :alt: PI policy controlling DROID to open the fridge door in the kitchen
   :align: center

   PI controls DROID to reach the fridge and open its door in the agentically
   generated kitchen environment.
