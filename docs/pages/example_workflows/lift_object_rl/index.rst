Franka Lift Object Task
=======================

This example demonstrates the complete workflow for the **Franka lift object task** in Isaac Lab - Arena, covering environment setup, policy training using RSL-RL, and closed-loop evaluation.

.. image:: ../../../images/kitchen_gr1_arena.gif
   :align: center
   :height: 400px

Task Overview
-------------

**Task ID:** ``franka_lift_object``

**Task Description:** The Franka robot lifts an object from the ground to a given location above the table.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Lift object, Table-top manipulation
   * - **Skills**
     - Reach, Grasp, Lift object
   * - **Embodiment**
     - Franka (7 DOF robot)
   * - **Scene**
     - Table environment
   * - **Objects**
     - Dex cube (articulated object)
   * - **Policy**
     - RSL-RL (reinforcement learning policy)
   * - **Training**
     - Reinforcement Learning
   * - **Checkpoint**
     - `GN1x-Tuned-Arena-GR1-Manipulation <https://huggingface.co/nvidia/GN1x-Tuned-Arena-GR1-Manipulation>`_
   * - **Physics**
     - PhysX (200Hz @ 4 decimation)
   * - **Closed-loop**
     - Yes (50Hz control)
   * - **Metrics**
     - Success rate


Workflow
--------

This tutorial covers the pipeline between creating an environment, training a policy using RSL-RL, and evaluating the policy in closed-loop.

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container

:docker_run_default:


We store data on Hugging Face, so you'll need log in to Hugging Face if you haven't already.

.. code-block:: bash

    hf auth login

You'll also need to create the folders for the pre-trained model.
Create the folders for the pre-trained model with:

.. code:: bash

    export MODELS_DIR=/models/isaaclab_arena/lift_object_rl_tutorial
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
