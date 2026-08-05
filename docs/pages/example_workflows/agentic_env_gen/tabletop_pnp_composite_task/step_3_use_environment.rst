Use the Generated Environment
-----------------------------

Once you are satisfied with the environment, you can use it to evaluate a policy on the environment.
The base container runs the environment as it was generated. The cuRobo-installed container additionally
gates object placement on whether the robot can reach the target objects.

Open one terminal and run the following command outside the Arena docker container to launch the PI policy server:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

In the other terminal, run the following command to launch the policy runner:

.. tab-set::

   .. tab-item:: Policy evaluation (base)
      :selected:

      **Docker Container**: Base (see :doc:`../../../quickstart/installation` for more details)

      :docker_run_default:

      For example, you can use the policy runner to evaluate PI policy on the environment.

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
            --viz kit \
            --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
            --enable_cameras \
            --num_envs 1 \
            --num_episodes 3 \
            --env_graph_spec_yaml isaaclab_arena_environments/agent_generated/simready_droid_pick_place_cans_hammer_maple_table.yaml

      For other policy types, please refer to the eavluation workflow page.

      .. todo:: add link to policy evaluation workflow page

   .. tab-item:: Policy evaluation with reachability validation (cuRobo)

      If you want to ensure the robot can reach the target objects (i.e. pepsi can, bean can and mini plastic
      basket), you can use this environment in the cuRobo-installed docker container to activate the
      reachability validation.

      **Docker Container**: Curobo-installed Base (see :doc:`../../../quickstart/installation` for more details)

      :docker_run_curobo:

      Now only the layouts the robot can reach are used:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
            --viz kit \
            --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
            --enable_cameras \
            --num_envs 1 \
            --num_episodes 3 \
            --env_graph_spec_yaml isaaclab_arena_environments/agent_generated/simready_droid_pick_place_cans_hammer_maple_table.yaml

      .. todo:: add link to reachability concept page

.. figure:: ../../../../images/agentic_env_droid_pi_pick_place_cans_hammer_maple_table_run1.gif
   :width: 100%
   :alt: Policy evaluation of the generated environment with the PI policy with reachability validation. Showing the robot picking up the pepsi can and bean can and placing them into the mini plastic basket.
   :align: center
