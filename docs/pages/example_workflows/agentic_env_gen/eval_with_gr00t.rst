Evaluate with GR00T
===================

GR00T evaluation runs an agentically generated Arena environment graph with a
remote GR00T policy server. The generated environment is passed as a environment graph spec YAML,
while GR00T receives camera observations, and returns DROID actions via remote policy connection.

This page assumes you already generated or selected a environment graph spec YAML.


Start the GR00T Server
----------------------

In the first terminal, start the GR00T policy server:

.. code-block:: bash

   cd submodules/Isaac-GR00T
   uv run python gr00t/eval/run_gr00t_server.py \
      --model-path nvidia/GR00T-N1.6-DROID \
      --embodiment-tag OXE_DROID \
      --device cuda --host 127.0.0.1 --port 5555

Leave this terminal running. The first launch downloads the ``nvidia/GR00T-N1.6-DROID``
weights from HuggingFace; later launches reuse the local cache.

See :doc:`../../quickstart/first_experiments/running_a_real_policy/gr00t` for
details on GR00T server setup, model downloads, and batch-evaluation examples.


Run the Generated Environment
-----------------------------

In a second terminal, run the policy runner with the generated environment graph spec YAML:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
      --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
      --remote_host 127.0.0.1 \
      --remote_port 5555 \
      --enable_cameras \
      --num_steps 1000 \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/mustard_above_raisin.yaml

The important pieces are:

* ``--policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy``
  selects the GR00T remote closed-loop policy client.
* ``--policy_config_yaml_path`` provides the DROID manipulation policy config.
* ``--remote_host`` and ``--remote_port`` must match the GR00T server.
* ``--enable_cameras`` is required because GR00T consumes visual observations.
* ``--env_graph_spec_yaml`` points the runner at the agentically generated
  environment graph spec.

Add a language instruction when you want to make the GR00T task explicit:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
      --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
      --remote_host 127.0.0.1 \
      --remote_port 5555 \
      --enable_cameras \
      --num_steps 1000 \
      --language_instruction "Pick up the mustard bottle and place it in the raisin box." \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/mustard_above_raisin.yaml

To use with variations, append the variation overrides after the environment source, e.g. to enable camera extrinsics variations:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
      --viz kit \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
      --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
      --remote_host 127.0.0.1 \
      --remote_port 5555 \
      --language_instruction "Pick up the mustard bottle and place it in the raisin box." \
      --enable_cameras \
      --num_steps 1000 \
      --env_graph_spec_yaml isaaclab_arena_environments/robolab/tasks/mustard_above_raisin.yaml \
      droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true

.. note::

    Variation overrides, such as ``light.hdr_image.enabled=true`` and
    ``droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true``, can be appended
    after the environment source.
