Use the Generated Environment
-----------------------------

Once you are satisfied with the environment, you can use it to evaluate a policy on the environment.

For example, you can use the Experiment Runner to evaluate a PI policy on the
environment. For other policy types, see
:doc:`Running a Real Policy <../../../quickstart/running_a_real_policy/index>`.

Open one terminal and run the following command outside the Arena docker container to launch the PI policy server:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

The Experiment Runner reads the evaluation from an Experiment Definition. Arena includes one with a
single Run for this environment. It uses the ready-made spec that ships with Arena, enables the
cameras, runs one environment, and stops after two episodes:

.. literalinclude:: ../../../../../isaaclab_arena_environments/experiment_configs/agentic_env_gen/kitchen_open_fridge_openpi_experiment.yaml
   :language: yaml

In the other terminal, run the following command to launch the Experiment Runner.
Complete the shared :ref:`agentic-env-gen-prerequisites` before running this command.

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
      --viz kit \
      --experiment_config isaaclab_arena_environments/experiment_configs/agentic_env_gen/kitchen_open_fridge_openpi_experiment.yaml

To evaluate a spec you generated yourself, point ``shared.environment.type`` at
``isaaclab_arena_environments/agent_generated/<env_name>.yaml``. The spec path is declared under
``shared`` in the YAML, so you can change it on the command line:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
      --viz kit \
      --experiment_config isaaclab_arena_environments/experiment_configs/agentic_env_gen/kitchen_open_fridge_openpi_experiment.yaml \
      shared.environment.type=isaaclab_arena_environments/agent_generated/<env_name>.yaml

.. figure:: ../../../../images/agentic_environment_generation/droid_kitchen_open_door_pi.gif
   :width: 100%
   :alt: PI policy controlling DROID to open the fridge door in the kitchen
   :align: center

   PI controls DROID to reach the fridge and open its door in the agentically
   generated kitchen environment.
