Use the Generated Environment
-----------------------------

Once you are satisfied with the environment, you can use it to evaluate a policy on the environment.

For example, you can use the policy runner to evaluate a PI policy on the
environment. For other policy types, see
:doc:`Running a Real Policy <../../../quickstart/running_a_real_policy/index>`.

Open one terminal and run the following command outside the Arena docker container to launch the PI policy server:

.. code-block:: bash

   ./isaaclab_arena_policies/isaaclab_arena_openpi/docker/run_openpi_server.sh

In the other terminal, run the following command to launch the policy runner.
Keep ``--num_envs`` above one so the policy is evaluated against a different fruit per environment.
The commands below use the ready-made spec that ships with Arena; to evaluate a spec you generated
yourself, point ``--env_spec`` at ``isaaclab_arena_environments/agent_generated/<env_name>.yaml``.

Complete the shared :ref:`agentic-env-gen-prerequisites` before running this command.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
      --enable_cameras \
      --num_envs 12 \
      --num_episodes 12 \
      --env_spec isaaclab_arena_environments/maple_table_top/droid_pick_fruit_into_bowl_maple_table.yaml

.. figure:: ../../../../images/agentic_environment_generation/agentic_env_droid_pi_fruit_plate_objectset_pnp_run.gif
   :width: 100%
   :alt: Policy evaluation of the generated environment using the OpenPI policy.
   :align: center

   Policy evaluation of the generated environment using the OpenPI policy with reachability validation.
   The robot picks up the fruit and places it into the bowl in each parallel environment.
