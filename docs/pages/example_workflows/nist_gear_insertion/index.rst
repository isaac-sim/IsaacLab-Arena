NIST Gear Insertion Task
========================

This example demonstrates the complete workflow for **reinforcement learning-based gear insertion**
on the assembled NIST board using the Franka Panda robot, operational-space control, and RL Games,
covering environment setup, policy training, and closed-loop evaluation.

.. image:: ../../../images/nist_gear_insertion_task.gif
   :align: center
   :height: 400px

Task Overview
-------------

**Task ID:** ``nist_assembled_gear_mesh_osc``

**Task Description:** The robot starts with the medium gear in its gripper and learns to align and
insert it onto the target peg on the NIST assembly board using task-space control and contact-rich
observations that include wrist-force feedback.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top assembly
   * - **Skills**
     - Align, insert, contact-aware manipulation
   * - **Embodiment**
     - Franka Panda (7 DOF arm + 2 DOF gripper)
   * - **Controller**
     - Operational-space control (7-D policy action)
   * - **Scene**
     - Table, assembled NIST board, target gear base, held medium gear, dome light
   * - **Observations**
     - 24-D policy observation plus task observations for critic/state
   * - **Policy**
     - RL Games PPO (learned from scratch)
   * - **Training Method**
     - Reinforcement Learning (on-policy PPO)
   * - **Physics**
     - PhysX
   * - **Closed-loop**
     - Yes
   * - **Action Space**
     - 7-D task-space action [3 position, 3 rotation, 1 auxiliary scalar]
   * - **Metric**
     - Success rate
   * - **Episode Length**
     - 15 seconds


Workflow
--------

This tutorial covers the pipeline for creating an RL environment, training a policy using RL Games,
and evaluating the trained policy in closed-loop. A user can follow the whole pipeline, or can start
at any intermediate step after preparing a checkpoint.

Prerequisites
^^^^^^^^^^^^^

Start the Isaac Lab Arena docker container:

:docker_run_default:

You'll need to create folders for logs, checkpoints, and models:

.. code-block:: bash

   export LOG_DIR=logs/rl_games
   mkdir -p $LOG_DIR
   export MODELS_DIR=models/isaaclab_arena/nist_gear_insertion
   mkdir -p $MODELS_DIR

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
