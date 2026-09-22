Dexsuite Kuka Allegro Lift Task (Newton)
=========================================

This example is an **experimental showcase for the Isaac Lab 3.0 Newton physics
backend**, demonstrating dexterous object lifting with the Kuka Allegro hand.
Training is performed in Isaac Lab and the resulting checkpoint is evaluated in
Arena — both using Newton (MuJoCo-Warp solver) for physically accurate contact
modelling during dexterous manipulation.

.. image:: ../../../images/dexsuite_lift_task.gif
   :align: center
   :height: 400px

.. important::

   **Newton Physics — Experimental**

   The ``dexsuite_lift`` environment defaults to Newton physics. Newton can
   also be selected explicitly for other Arena environments by passing
   ``--presets newton``. However, **Newton support is experimental** — only
   this example has been verified to work with Newton under the current
   simulation settings. Other
   environments may require additional tuning of solver parameters and physics parameters
   to run correctly using Newton physics.


Task Overview
-------------

**Task ID:** ``dexsuite_lift``

**Task Description:** The Kuka arm with an Allegro dexterous hand lifts a procedurally
generated cuboid to a commanded target position using joint-space actions and
contact-rich proprioceptive observations — including fingertip contact forces, hand-tip
body states, object point cloud, and 5-step observation history.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Dexterous manipulation, contact-rich
   * - **Skills**
     - Reach, Grasp, Lift (multi-finger)
   * - **Embodiment**
     - Kuka LBR iiwa + Allegro Hand (7 DOF arm + 16 DOF hand)
   * - **Scene**
     - Procedural table (static background) with ground plane and lighting
   * - **Objects**
     - Procedural lift cuboid (``procedural_cube``)
   * - **Policy**
     - RSL-RL PPO (``KukaAllegroPPORunnerCfg``)
   * - **Training Method**
     - Reinforcement Learning (on-policy PPO) — trained in **Isaac Lab**
   * - **Physics Backend**
     - Newton (default) or PhysX (``--presets physx``)
   * - **Simulation Rate**
     - 200 Hz physics, 50 Hz control (decimation = 4)
   * - **Episode Length**
     - 6 seconds
   * - **Closed-loop**
     - Yes (50 Hz control)
   * - **Command Space**
     - Target position [x, y, z], position-only, resampled every 2–3 s

.. note::

   Arena evaluation defaults to **Newton** for this environment. Pass
   ``--presets physx`` to ``policy_runner.py`` to use PhysX instead. Isaac Lab
   training selects Newton separately with ``physics=newton_mjwarp``.


Workflow
--------

This tutorial covers training in Isaac Lab and evaluating the resulting
checkpoint in Arena.

Prerequisites
^^^^^^^^^^^^^

Start the Isaac Lab Arena docker container:

:docker_run_default:

Workflow Steps
^^^^^^^^^^^^^^

Follow the steps below to complete the workflow:

- :doc:`step_1_environment_setup`
- :doc:`step_2_policy_training`
- :doc:`step_3_evaluation`


.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_environment_setup
   step_2_policy_training
   step_3_evaluation
