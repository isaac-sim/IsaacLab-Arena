Getting Started
===============

The fastest way to explore Arena is with the eval runner and a **zero-action policy** — no model
weights required. This walks you through four short jobs that each demonstrate a core Arena concept:
swapping objects, changing the background HDR, and running parallel environments.

Run the Getting Started Config
-------------------------------

.. code-block:: bash

   python isaaclab_arena/evaluation/eval_runner.py \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/getting_started_jobs_config.json

All four jobs run in the ``srl_pick_and_place`` environment. This illustrates a core Arena
concept: you define **one** base environment and vary individual axes — object, lighting, scale —
without touching any task logic or duplicating configuration. The same pattern is how Arena is
used in practice: a single environment definition becomes the foundation for evaluating a policy
across dozens of object, scene, and embodiment combinations.

Each job changes exactly one thing from the baseline, making it easy to see what each field
controls:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Job
     - What changes
     - Key fields
   * - ``01_baseline``
     - Nothing — your reference run
     - ``srl_pick_and_place`` + ``rubiks_cube`` → ``bowl``
   * - ``02_swap_objects``
     - Different pick-up object and destination
     - ``pick_up_object``, ``destination_location``
   * - ``03_change_background_hdr``
     - Different background HDR environment map
     - ``hdr``
   * - ``04_parallel_envs``
     - 4 environments running simultaneously
     - ``num_envs``

.. figure:: ../../images/franka_kitchen_pickup.gif
   :width: 100%
   :alt: Getting Started — baseline, swap objects, change HDR, parallel envs
   :align: center

   TODO(cvolk): Add all of the below into one gif

**Expected outcome:** Because this uses a zero-action policy, the robot does not move and no
objects are picked up. Each job will report a success rate of ``0.0``. This is expected — the
purpose here is to verify your setup and explore how each field affects the environment. To get
non-zero success rates, replace ``"policy_type": "zero_action"`` with a trained policy (see
`Next Steps`_ below).


Experiment: Swap Objects
-------------------------

Change ``pick_up_object`` and ``destination_location`` to any registered asset name.
Some options to try:

.. code-block:: text

   pick_up_object:
     rubiks_cube_hot3d_robolab     mustard_bottle_hot3d_robolab
     mug_hot3d_robolab             soup_can_hot3d_robolab
     ceramic_mug_hot3d_robolab     pitcher_hot3d_robolab
     mustard_ycb_robolab           sugar_box_ycb_robolab
     tomato_soup_can_ycb_robolab   mug_ycb_robolab

   destination_location:
     bowl_ycb_robolab              wooden_bowl_hot3d_robolab

**Expected outcome:** The scene respawns with the new object on the table. Success rate remains
``0.0`` — the robot does not move.

.. figure:: ../../images/swap_objects.gif
   :width: 100%
   :alt: Swapping pick-up objects in srl_pick_and_place
   :align: center


Experiment: Change Background HDR
-----------------------------------

Set ``hdr`` to any registered HDR environment map to change the background and ambient lighting:

.. code-block:: text

   home_office_robolab            empty_warehouse_robolab
   billiard_hall_robolab          aerodynamics_workshop_robolab
   wooden_lounge_robolab          garage_robolab
   kiara_interior_robolab         brown_photostudio_robolab
   carpentry_shop_robolab

You can also adjust the dome light intensity independently of the HDR map:

.. code-block:: json

   "hdr": "empty_warehouse_robolab",
   "light_intensity": 1000.0

**Expected outcome:** The background and ambient light color change to match the selected HDR.
Success rate remains ``0.0``.

.. figure:: ../../images/swap_hdr.gif
   :width: 100%
   :alt: Changing background HDR in srl_pick_and_place
   :align: center


Experiment: Scale Up
---------------------

Increase ``num_envs`` to run multiple environments in parallel:

.. code-block:: json

   "num_envs": 64

All environments share the same assets and run simultaneously on the GPU. This is how Arena
is used for large-scale policy evaluation.

**Expected outcome:** Four environment instances appear side-by-side. Success rate remains
``0.0`` across all environments.

.. figure:: ../../images/scale_up.gif
   :width: 100%
   :alt: Running 64 parallel srl_pick_and_place environments
   :align: center


.. _Next Steps:

Next Steps
----------

Once you have verified the setup, replace ``zero_action`` with a trained policy to get non-zero
success rates. The same environment and job config work without modification:

.. code-block:: json

   "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
   "policy_config_dict": {
       "policy_config_yaml_path": "path/to/config.yaml",
       "policy_device": "cuda:0"
   }

See :doc:`../../pages/policy_evaluation/index` for policy evaluation details, or refer to
``isaaclab_arena_environments/eval_jobs_configs/gr00t_jobs_config.json`` for a complete GR00T
example (requires ``./docker/run_docker.sh -g``).

For a deeper look at how environments are composed from scratch using the Python API, see
:doc:`first_arena_env`.
