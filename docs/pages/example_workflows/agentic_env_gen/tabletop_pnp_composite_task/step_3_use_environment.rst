Use the Generated Environment
-----------------------------

Once you are satisfied with the environment, you can use it to evaluate a policy on the environment.
The base container runs the environment as it was generated. The cuRobo-installed container additionally
gates object placement on whether the robot can reach the target objects.

For example, you can use the Experiment Runner to evaluate a PI policy on the
environment. For other policy types, see
:doc:`Running a Real Policy <../../../quickstart/running_a_real_policy/index>`.

Open one terminal and run the following command outside the Arena docker container to launch the PI policy server:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

The Experiment Runner reads the evaluation from an Experiment Definition. Arena includes one with a
single Run for this environment. It uses the ready-made spec that ships with Arena, enables the
cameras, runs one environment, and stops after three episodes:

.. literalinclude:: ../../../../../isaaclab_arena_environments/experiment_configs/agentic_env_gen/tabletop_cans_into_basket_openpi_experiment.yaml
   :language: yaml

In the other terminal, run the following command to launch the Experiment Runner. To evaluate a spec you
generated yourself, add ``shared.environment.type=isaaclab_arena_environments/agent_generated/<env_name>.yaml``
at the end of the command. The spec path is declared under ``shared`` in the YAML, so you can change it on
the command line.

.. tab-set::

   .. tab-item:: Policy evaluation (without reachability validation)
      :selected:

      Complete the shared :ref:`agentic-env-gen-prerequisites` before running this command.

      .. code-block:: bash

         python isaaclab_arena/evaluation/experiment_runner.py \
            --viz kit \
            --experiment_config isaaclab_arena_environments/experiment_configs/agentic_env_gen/tabletop_cans_into_basket_openpi_experiment.yaml


   .. tab-item:: Policy evaluation (with cuRobo-based reachability validation)

      .. note::
         Reachability validation runs only in the cuRobo-installed Docker container
         (``./docker/run_docker.sh -c``). It is not available with a native ``uv``
         install — see :doc:`../../../quickstart/installation` and
         :ref:`ik-reachable-check`.

      If you want to ensure the robot can reach the target objects (i.e. beverage can, bean can and mini plastic
      basket), you can use this environment in the cuRobo-installed docker container to activate the
      reachability validation.

      **Docker Container**: Curobo-installed Base (see :doc:`../../../quickstart/installation` for more details)

      :docker_run_curobo:

      Now only the layouts the robot can reach are used:

      .. code-block:: bash

         python isaaclab_arena/evaluation/experiment_runner.py \
            --viz kit \
            --experiment_config isaaclab_arena_environments/experiment_configs/agentic_env_gen/tabletop_cans_into_basket_openpi_experiment.yaml

      While the environment builds, every batch of candidate layouts reports how many of them passed each
      check. ``ik_reachable`` is the cuRobo verdict, so its ratio is the rejection rate to watch:

      .. code-block:: text

         [placement] Validated 50 candidate layout(s); passed per check: on_relation=47/50, next_to=39/50, not_next_to=50/50, face_to=50/50, no_overlap=39/50, ik_reachable=6/36

      A low ``ik_reachable`` ratio means most sampled layouts put the beverage can, bean can, or basket outside the
      arm's workspace, and the placer keeps resampling. A composite task rejects more layouts than an atomic
      one, because every pick target and the place target must be reachable in the same layout.
      When an environment finds no reachable layout at all, it falls back to its lowest-loss layout.

      See :ref:`ik-reachable-check` for how this check is registered, what it requires, and how to tune
      or disable it.

.. figure:: ../../../../images/agentic_environment_generation/agentic_env_droid_pi_cans_basket_pnp_run1.gif
   :width: 100%
   :alt: Policy evaluation of the generated environment using the OpenPI policy.
   :align: center

   Policy evaluation of the generated environment using the OpenPI policy with reachability validation.
   The robot picks up the beverage can and bean can and places them into the mini plastic basket.
