Running a Real Policy
======================

The zero-action experiments keep the robot still and success rates at zero.
In this section we will see an actual
model in action, `GR00T N1.6 <https://github.com/NVIDIA/Isaac-GR00T/>`_, a pre-trained
robotic foundation model. No fine-tuning or separate model download is required —
the weights fetch automatically from
`HuggingFace <https://huggingface.co/nvidia/GR00T-N1.6-DROID>`_ on first use.

**Prerequisite: GR00T container**

GR00T requires extra dependencies not included in the base Arena container. Rebuild and restart
with the ``-g`` flag:

.. code-block:: bash

   ./docker/run_docker.sh -g

**Run GR00T closed-loop**

Two things change relative to the zero-action baseline:

- ``--policy_type`` points to the GR00T closed-loop policy class and ``--policy_config_yaml_path``
  provides its config (model ID, action chunk length, camera names, etc.)
- ``--enable_cameras`` turns on the robot's cameras, which GR00T requires for observations

GR00T also requires absolute joint positions, so use ``--embodiment droid_abs_joint_pos``
instead of ``--embodiment droid_rel_joint_pos``. The command uses ``--num_episodes`` rather than
``--num_steps`` so the run terminates on task completion rather than after a fixed number
of simulation steps:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
     --enable_cameras \
     --num_episodes 3 \
     pick_and_place_maple_table \
     --embodiment droid_abs_joint_pos \
     --pick_up_object rubiks_cube_hot3d_robolab \
     --destination_location bowl_ycb_robolab \
     --hdr home_office_robolab

The first run fetches the ``nvidia/GR00T-N1.6-DROID`` weights from HuggingFace and caches them
locally; this can take some time. Subsequent runs start immediately. After each episode Arena prints whether the
pick-and-place succeeded. You can swap ``--pick_up_object`` and ``--hdr`` exactly as in the
zero-action experiments. This functionality can be used to test how the policy adapts to each new
object and lighting condition, as we shall see in the next section.

**Multi-job evaluation across object variations**

To measure success rates across several variations of the environment in a single command:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_gr00t_jobs_config.json

This runs six object variations sequentially and reports a per-variation success rate.
Each evaluation is run without restarting Isaac Sim to save on the startup time.

.. figure:: ../../../images/gr00t_droid_mem.gif
   :width: 100%
   :alt: 5x5 grid of GR00T N1.6 DROID runs across different backgrounds, lighting, and destination objects
   :align: center

   25 closed-loop evaluation runs of GR00T N1.6 on the DROID embodiment, varying background,
   lighting, and pick-up object across the grid.


.. _Next Steps:

Next Steps
----------

To go beyond the pre-trained GR00T N1.6 foundation model — for example, fine-tuning on your own
teleoperation data — see :doc:`../../../pages/example_workflows/imitation_learning/index` for
end-to-end imitation learning workflows.
