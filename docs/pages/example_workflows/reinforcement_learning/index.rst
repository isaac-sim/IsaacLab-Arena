Franka Lift Object Task
========================

This example demonstrates the complete workflow for **reinforcement learning-based object lifting** using the Franka Panda robot in Isaac Lab - Arena, covering environment setup, policy training with RSL-RL, and closed-loop evaluation.

.. image:: ../../../images/franka_kitchen_pickup.gif
   :align: center
   :height: 400px

Task Overview
-------------

**Task ID:** ``lift_object``

**Task Description:** The Franka Panda robot learns to grasp and lift objects to target positions through reinforcement learning. The task uses a command-based goal specification, where the RL agent learns to reach sampled target poses.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation
   * - **Skills**
     - Reach, Grasp, Lift
   * - **Embodiment**
     - Franka Panda (9 DOF arm + 2 DOF gripper)
   * - **Scene**
     - Table with ground plane and lighting
   * - **Objects**
     - Configurable (default: dex_cube)
   * - **Policy**
     - RSL-RL PPO (learned from scratch)
   * - **Training Method**
     - Reinforcement Learning (on-policy PPO)
   * - **Physics**
     - PhysX (50Hz @ 2 decimation)
   * - **Closed-loop**
     - Yes (50Hz control)
   * - **Command Space**
     - Target position [x, y, z] relative to object initial pose
   * - **Training Time**
     - ~6 hours (12,000 iterations on 512 environments in a A6000)


Workflow
--------

This tutorial covers the pipeline for creating an RL environment, training a policy using RSL-RL,
and evaluating the trained policy in closed-loop. A user can follow the whole pipeline, or can start
at any intermediate step by using the provided checkpoints.

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container

:docker_run_default:

We store models on Hugging Face, so you'll need log in to Hugging Face if you haven't already.

.. code-block:: bash

    hf auth login

You'll need to create folders for logs, checkpoints, and models:

.. code:: bash

    export LOG_DIR=logs/rsl_rl
    mkdir -p $LOG_DIR
    export MODELS_DIR=models/isaaclab_arena/reinforcement_learning
    mkdir -p $MODELS_DIR

Workflow Steps
^^^^^^^^^^^^^^

Follow the following steps to complete the workflow:

- :doc:`step_1_environment_setup`
- :doc:`step_2_policy_training`
- :doc:`step_3_evaluation`


.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_environment_setup
   step_2_policy_training
   step_3_evaluation
