Dexsuite Kuka Allegro Lift Task (Newton)
=========================================

This example demonstrates **dexterous object lifting** with the Kuka Allegro hand using
**Newton physics** in Isaac Lab Arena. Training is performed in Isaac Lab with PhysX, and
the resulting checkpoint is evaluated in Arena under Newton — a more physically accurate
solver suitable for contact-rich manipulation.

.. image:: ../../../images/dexsuite_lift_task.gif
   :align: center
   :height: 400px

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
     - Dexsuite kinematic table with ground plane and lighting
   * - **Objects**
     - Procedural lift cuboid (``dexsuite_lift_object``)
   * - **Policy**
     - RSL-RL PPO (``DexsuiteKukaAllegroPPORunnerCfg``)
   * - **Training Method**
     - Reinforcement Learning (on-policy PPO) — trained in **Isaac Lab** with Newton
   * - **Evaluation Physics**
     - **Newton** (MuJoCo-Warp solver)
   * - **Simulation Rate**
     - 120 Hz physics, 60 Hz control (decimation = 2)
   * - **Episode Length**
     - 6 seconds
   * - **Closed-loop**
     - Yes (60 Hz control)
   * - **Command Space**
     - Target position [x, y, z], position-only, resampled every 2–3 s

.. note::

   **Newton physics** uses the MuJoCo-Warp solver, which is more physically accurate for
   contact-rich tasks than PhysX. The Arena environment callback automatically configures
   the Newton backend, reset-mode domain randomization events (gravity scheduling, object/robot
   reset), and simulation parameters. PhysX startup events (material/friction/mass randomization)
   are disabled under Newton.


Workflow
--------

This tutorial covers training in Isaac Lab with Newton physics and evaluating the resulting
checkpoint in Arena (also Newton).

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
