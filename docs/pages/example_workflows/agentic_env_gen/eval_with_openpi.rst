Evaluate with OpenPI
====================

OpenPI evaluation runs an agentically generated Arena environment graph with a
remote OpenPI policy server. The generated environment is passed as a environment graph spec YAML,
while OpenPI receives camera observations, and returns DROID joint-position actions via WebSocket.

This page assumes you already generated or selected a environment graph spec YAML.

Start the OpenPI Server
-----------------------

In the first terminal, start the OpenPI server:

.. code-block:: bash

   ./isaaclab_arena_openpi/docker/run_openpi_server.sh

Leave this terminal running. The server is ready when it reports:

.. code-block:: text

   INFO:websockets.server:server listening on 0.0.0.0:8000

See :doc:`../../quickstart/first_experiments/running_a_real_policy/openpi` for
details on OpenPI variants, rebuild flags, and server setup.


Run the Generated Environment
-----------------------------

In a second terminal, run the policy runner with the generated environment graph spec YAML:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
      --num_envs 4 \
      --num_steps 1000 \
      --env_spacing 1.5 \
      --enable_cameras \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/larger_object_raisin_box_in_bin.yaml

Important flags:

* ``--policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy``
  selects the OpenPI remote policy client.
* ``--enable_cameras`` is required because OpenPI consumes visual observations.
* ``--env_graph_spec_yaml`` points the runner at the agentically generated
  environment graph spec.

Add a language instruction when you want to make the OpenPI task explicit:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
      --num_envs 4 \
      --num_steps 1000 \
      --env_spacing 1.5 \
      --enable_cameras \
      --language_instruction "Pick up the red raisin box and place it in the grey bin." \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/larger_object_raisin_box_in_bin.yaml

To use with variations, append the variation overrides after the environment source, e.g. to enable camera extrinsics variations:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
      --num_envs 4 \
      --num_steps 1000 \
      --env_spacing 1.5 \
      --enable_cameras \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/larger_object_raisin_box_in_bin.yaml \
      droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true

.. note::
    Variation overrides, such as ``light.hdr_image.enabled=true`` and
    ``droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true``, can be appended
    after the environment source.

Run with Experiment Runner
--------------------------------

For repeatable evaluation, put the generated environment graph spec YAML in an
eval jobs config. The ``environment`` field points at the graph spec YAML, and the
OpenPI connection options go in ``policy_config_dict``:

.. code-block:: json

   {
       "jobs": [
           {
               "name": "agentic_openpi_butter_raisin_box_grey_bin",
               "arena_env_args": {
                   "enable_cameras": true,
                   "environment": "isaaclab_arena_environments/robolab/tasks/larger_object_raisin_box_in_bin.yaml"
               },
               "num_steps": 100,
               "language_instruction": "Pick up the red raisin box and place it in the grey bin.",
               "policy_type": "isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy",
               "policy_config_dict": {
                   "policy_variant": "pi05",
                   "policy_device": "cuda:0",
                   "remote_host": "127.0.0.1",
                   "remote_port": 8000,
                   "openpi_embodiment_adapter": "droid"
               },
               "variations": {
                   "droid_abs_joint_pos": {
                       "camera_extrinsics_wrist_camera": { "enabled": true }
                   }
               }
           }
       ]
   }

Then run the Experiment Runner:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
      --viz kit \
      --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/agentic_openpi_jobs_config.json

This keeps the generated environment, language instruction, OpenPI policy
settings, and variation settings in one reusable config.
