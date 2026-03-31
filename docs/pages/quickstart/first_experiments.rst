Running your First Experiments
==============================

Arena lets you evaluate a robot policy across variations of object, lighting, and embodiment
from a single environment definition — no task logic changes, no duplicated configuration. You
swap one argument and get a completely different environment.

The fastest way to see this is with the **policy runner** and a **zero-action policy** — a
placeholder policy that sends zero commands to the robot every step. The robot stays still, but
the environment loads, the scene renders, and you can verify that each variation works. No model
weights needed.

The four experiments below all use the same ``pick_and_place_maple_table`` environment. Each one changes
exactly one argument from the baseline.


Experiment: Baseline
---------------------

Your reference run — rubiks cube on the table, bowl as destination:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object rubiks_cube_hot3d_robolab \
     --destination_location bowl_ycb_robolab \
     --hdr home_office_robolab

.. figure:: ../../images/default_srl_pnp.png
   :width: 100%
   :alt: Default pick_and_place_maple_table environment — rubiks cube and bowl on table
   :align: center


Experiment: Swap Objects
-------------------------

Change ``--pick_up_object`` and ``--destination_location`` to swap what is on the table.
Some options to try:

.. code-block:: text

   --pick_up_object:
     rubiks_cube_hot3d_robolab     mustard_bottle_hot3d_robolab
     mug_hot3d_robolab             soup_can_hot3d_robolab
     ceramic_mug_hot3d_robolab     pitcher_hot3d_robolab
     mustard_ycb_robolab           sugar_box_ycb_robolab
     tomato_soup_can_ycb_robolab   mug_ycb_robolab

   --destination_location:
     bowl_ycb_robolab              wooden_bowl_hot3d_robolab

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object mustard_bottle_hot3d_robolab \
     --destination_location wooden_bowl_hot3d_robolab \
     --hdr home_office_robolab

.. figure:: ../../images/swap_objects.gif
   :width: 100%
   :alt: Swapping pick-up objects in pick_and_place_maple_table
   :align: center


Experiment: Change Background HDR
-----------------------------------

Set ``--hdr`` to any registered HDR environment map to change the background and ambient lighting:

.. code-block:: text

   home_office_robolab            empty_warehouse_robolab
   billiard_hall_robolab          aerodynamics_workshop_robolab
   wooden_lounge_robolab          garage_robolab
   kiara_interior_robolab         brown_photostudio_robolab
   carpentry_shop_robolab

You can also adjust the dome light intensity independently with ``--light_intensity``:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object rubiks_cube_hot3d_robolab \
     --destination_location bowl_ycb_robolab \
     --hdr billiard_hall_robolab \
     --light_intensity 1000.0

.. figure:: ../../images/swap_hdr.gif
   :width: 100%
   :alt: Changing background HDR in pick_and_place_maple_table
   :align: center


Experiment: Scale Up
---------------------

Add ``--num_envs`` to run many environments in parallel on the GPU:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     --num_envs 64 \
     pick_and_place_maple_table \
     --embodiment droid_rel_joint_pos \
     --pick_up_object rubiks_cube_hot3d_robolab \
     --destination_location bowl_ycb_robolab

All 64 environments share the same assets and run simultaneously on the GPU. When you swap in a
real policy, every one of those environments runs inference in parallel — giving you 64 evaluation
episodes at the cost of one. This is how Arena makes it practical to measure policy robustness
across hundreds of object and scene combinations in a single run.

.. figure:: ../../images/scale_up.gif
   :width: 100%
   :alt: Running 64 parallel pick_and_place_maple_table environments
   :align: center


Batch Evaluation
-----------------

The four experiments above run one variation at a time. In practice, Arena is used to evaluate
a policy across hundreds of object, scene, and embodiment combinations in a single run. The
``eval_runner.py`` script reads a JSON job config that lists any number of jobs — each with its
own environment arguments, policy, and step count — and runs them sequentially within one Isaac
Sim process, collecting success metrics for each:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/getting_started_jobs_config.json

.. figure:: ../../images/iterate_getting_started_jobs_config.gif
   :alt: Four evaluation jobs running sequentially: baseline, swapped objects, changed HDR, and 64 parallel environments
   :align: center

At the end of the run you get a per-job summary of success rates. See
:ref:`sequential-batch-eval-runner` for full details on the job config format and available options.


Running GR00T N1.6 (a Real Policy)
------------------------------------

The experiments above use ``zero_action`` — the robot stays still and success rates are zero.
To see an actual model in action, swap in `GR00T N1.6 <https://github.com/NVIDIA/Isaac-GR00T/>`_,
a pre-trained DROID manipulation foundation model. No fine-tuning or separate model download is
required — the weights fetch automatically from
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
instead of ``droid_rel_joint_pos``:

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
locally; subsequent runs start immediately. After each episode Arena prints whether the
pick-and-place succeeded. You can swap ``--pick_up_object`` and ``--hdr`` exactly as in the
zero-action experiments — the policy adapts to each new object and lighting condition without
any further configuration.

**Batch evaluation across object variations**

To measure success rates across multiple objects in one run:

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_gr00t_jobs_config.json

This runs seven object variations sequentially and reports a per-job success rate, all within a
single Isaac Sim process.


.. _Next Steps:

Next Steps
----------

To go beyond the pre-trained GR00T N1.6 foundation model — for example, fine-tuning on your own
teleoperation data — see :doc:`../../pages/example_workflows/imitation_learning/index` for
end-to-end imitation learning workflows.
