NIST Gear Insertion Task
========================

This example demonstrates **reinforcement learning-based gear insertion** on the assembled NIST board using the Franka Panda robot in Isaac Lab - Arena, covering environment setup and closed-loop evaluation.

.. image:: ../../../images/nist_gear_insertion_task.gif
   :align: center
   :height: 400px

Task Overview
-------------

**Task ID:** ``nist_assembled_gear_mesh_osc``

**Task Description:** The Franka Panda robot starts with the medium gear in its gripper and learns to align and insert it onto the target peg on the NIST assembly board using operational-space control and contact-rich observations that include wrist-force feedback.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top assembly
   * - **Skills**
     - Align, Insert, Contact-aware manipulation
   * - **Embodiment**
     - Franka Panda (7 DOF arm + 2 DOF gripper)
   * - **Scene**
     - Table, assembled NIST board, target gear base, held medium gear, dome light
   * - **Objects**
     - Medium NIST gear (held), gears and base (fixed insertion target)
   * - **Policy**
     - RL Games PPO
   * - **Policy Source**
     - Pre-trained checkpoint or externally trained RL Games PPO policy
   * - **Physics**
     - PhysX (default)
   * - **Closed-loop**
     - Yes (task-space control)
   * - **Action Space**
     - 7-D policy vector: 3 bounded position channels, yaw around the gear axis, and an auxiliary success scalar
   * - **Episode Length**
     - 15 seconds


Workflow
--------

This tutorial covers the pipeline for creating an RL environment and evaluating a trained policy in
closed-loop. Use the provided checkpoint, or replace it with a checkpoint from an external training
launcher.

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container

:docker_run_default:

We store models on Hugging Face, so you'll need log in to Hugging Face if you haven't already.

.. code-block:: bash

    hf auth login

You'll need to create a folder for model checkpoints:

.. code:: bash

    export MODELS_DIR=/models/isaaclab_arena/nist_gear_insertion
    mkdir -p $MODELS_DIR

Workflow Steps
^^^^^^^^^^^^^^

Follow the following steps to complete the workflow:

- :doc:`step_1_environment_setup`
- :doc:`step_2_evaluation`


.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_environment_setup
   step_2_evaluation
