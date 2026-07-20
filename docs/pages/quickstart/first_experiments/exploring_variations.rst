Exploring Environment Variations
=================================

Arena lets you evaluate a robot policy across variations of object, lighting, and embodiment
from a single environment definition — no task logic changes, no duplicated configuration. You
swap one argument and get a completely different environment.

The fastest way to see this is with the **Experiment Runner's inline preview mode**. It builds one
registered environment and steps it with a **zero-action policy**, so the robot stays still while
the environment loads and renders. No model weights or Experiment YAML are needed.

The four experiments below all use the same ``pick_and_place_maple_table`` environment. Each one changes
exactly one argument from the baseline.


Experiment: Baseline
---------------------

Your reference run — rubiks cube on the table, bowl as destination:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --environment pick_and_place_maple_table \
     environment.embodiment=droid_rel_joint_pos \
     environment.pick_up_object=rubiks_cube_hot3d_robolab \
     environment.destination_location=bowl_ycb_robolab \
     environment.hdr=home_office_robolab \
     rollout_limit.num_steps=50

.. figure:: ../../../images/default_srl_pnp.png
   :width: 100%
   :alt: Default pick_and_place_maple_table environment — rubiks cube and bowl on table
   :align: center


Experiment: Swap Objects
-------------------------

Change ``environment.pick_up_object`` and ``environment.destination_location`` to swap what is on the table.
Some options to try:

.. code-block:: text

   environment.pick_up_object:
     rubiks_cube_hot3d_robolab     mustard_bottle_hot3d_robolab
     mug_hot3d_robolab             soup_can_hot3d_robolab
     ceramic_mug_hot3d_robolab     pitcher_hot3d_robolab
     mustard_ycb_robolab           sugar_box_ycb_robolab
     tomato_soup_can_ycb_robolab   mug_ycb_robolab

   environment.destination_location:
     bowl_ycb_robolab              wooden_bowl_hot3d_robolab

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --environment pick_and_place_maple_table \
     environment.embodiment=droid_rel_joint_pos \
     environment.pick_up_object=mustard_bottle_hot3d_robolab \
     environment.destination_location=wooden_bowl_hot3d_robolab \
     environment.hdr=home_office_robolab \
     rollout_limit.num_steps=50

.. figure:: ../../../images/swap_objects.gif
   :width: 100%
   :alt: Swapping pick-up objects in pick_and_place_maple_table
   :align: center


Experiment: Change Background HDR
-----------------------------------

Set ``environment.hdr`` to any registered HDR environment map to change the background and ambient lighting:

.. code-block:: text

   home_office_robolab            empty_warehouse_robolab
   billiard_hall_robolab          aerodynamics_workshop_robolab
   wooden_lounge_robolab          garage_robolab
   kiara_interior_robolab         brown_photostudio_robolab
   carpentry_shop_robolab

You can also adjust the dome light intensity independently with ``environment.light_intensity``:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --environment pick_and_place_maple_table \
     environment.embodiment=droid_rel_joint_pos \
     environment.pick_up_object=rubiks_cube_hot3d_robolab \
     environment.destination_location=bowl_ycb_robolab \
     environment.hdr=billiard_hall_robolab \
     environment.light_intensity=1000.0 \
     rollout_limit.num_steps=50

.. figure:: ../../../images/swap_hdr.gif
   :width: 100%
   :alt: Changing background HDR in pick_and_place_maple_table
   :align: center


Experiment: Scale Up
---------------------

Use ``--num_envs`` to run many environments in parallel on the GPU:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --environment pick_and_place_maple_table \
     --num_envs 64 \
     environment.embodiment=droid_rel_joint_pos \
     environment.pick_up_object=rubiks_cube_hot3d_robolab \
     environment.destination_location=bowl_ycb_robolab \
     rollout_limit.num_steps=50

All 64 environments share the same assets and run simultaneously on the GPU. In a configured Run
with a real policy, every one of those environments runs inference in parallel — giving you 64
evaluation episodes at the cost of one. This is how Arena makes it practical to measure policy
robustness across hundreds of object and scene combinations in a single run.

.. figure:: ../../../images/scale_up.gif
   :width: 100%
   :alt: Running 64 parallel pick_and_place_maple_table environments
   :align: center


Sequential Batch Evaluation
---------------------------

The four experiments above run one variation at a time. In practice, Arena is used to evaluate
a policy across hundreds of object, scene, and embodiment combinations in a single run. The
Experiment Runner can also load a typed YAML Experiment Definition containing any number of
named Runs. Each Run declares its environment, policy, and rollout limit, and the Runs execute
sequentially within one Isaac Sim process. ``getting_started_experiment.yaml`` bundles the four
experiments above into a single definition:

.. code-block:: bash

   python isaaclab_arena/evaluation/experiment_runner.py \
     --viz kit \
     --experiment_config isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml

.. figure:: ../../../images/iterate_getting_started_jobs_config.gif
   :alt: Four evaluation Runs executing sequentially: baseline, swapped objects, changed HDR, and 64 parallel environments
   :align: center

At the end of the Experiment you get a per-Run summary of success rates. See
:ref:`sequential-batch-experiment-runner` for details on inline previews and configured Experiments.

All of the above used a zero-action policy — the robot stays still and success rates are zero.
The next page swaps in a real pre-trained model and runs it in closed loop:
:doc:`running_a_real_policy/index`.
