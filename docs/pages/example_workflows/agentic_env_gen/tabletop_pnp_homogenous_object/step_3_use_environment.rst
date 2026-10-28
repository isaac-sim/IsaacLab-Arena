Use the Generated Environment
-----------------------------

Once you are satisfied with the environment, you can use it to evaluate a policy on the environment.
Every Arena install includes cuRobo, so you can also keep only the object layouts in which the robot
can reach the target objects (see below).

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
      --num_episodes 3 \
      --env_spec isaaclab_arena_environments/maple_table_top/droid_banana_on_plate_maple_table.yaml

To keep only layouts in which the robot can reach the target objects, turn on the cuRobo
``ik_reachable`` check, which is off by default, in the spec's ``placer_params``:

.. code-block:: yaml

   placer_params:
     reachability_config:
       enabled: true

While the environment builds, every batch of candidate layouts reports how many of them passed each
check. ``ik_reachable`` is the cuRobo verdict, so its ratio is the rejection rate to watch:

.. code-block:: text

   [placement] Validated 50 candidate layout(s); passed per check: on_relation=46/50, clutter_on_relation=50/50, align_with=50/50, next_to=50/50, not_next_to=50/50, face_to=50/50, no_overlap=45/50, ik_reachable=39/44

A low ``ik_reachable`` ratio means most sampled layouts put the banana or the plate outside the arm's
workspace, and the placer keeps resampling.
When an environment finds no reachable layout at all, it falls back to its lowest-loss layout.

See :ref:`ik-reachable-check` for how this check is registered, what it requires, and how to tune
or disable it.

.. figure:: ../../../../images/agentic_environment_generation/agentic_env_droid_pi_banana_plate_pnp_run1.gif
   :width: 100%
   :alt: Policy evaluation of the generated environment using the OpenPI policy.
   :align: center

   Policy evaluation of the generated environment using the OpenPI policy with reachability validation.
   The robot picks up the banana and places it on the plate.
