G1 Static Apple-to-Plate Task
=============================

This example demonstrates the complete workflow for the **G1 static (no-navigation) apple-to-plate task** in Isaac Lab - Arena, covering environment setup and validation, teleoperation data collection (OpenXR with Meta Quest 3 or Pico 4 Ultra), data generation, policy post-training, and closed-loop evaluation.

This workflow is the no-locomotion sibling of the :doc:`G1 Loco-Manipulation Box Pick and Place Task <../locomanipulation/index>`. The robot stands in place using the same Whole Body Controller (WBC) for balance, but the destination plate sits on the *same* shelf as the apple — within arm's reach — so the lower body never moves. If you want a tabletop manipulation surface for upper-body data collection without the complexity of full-body locomotion, this is the workflow to use.

Task Overview
-------------

**Task Name:** ``galileo_g1_static_pick_and_place``

**Task Description:** The G1 humanoid robot stands in front of a shelf and uses its arms to pick up
an apple and place it onto a plate sitting on the same shelf, within arm's reach. WBC actively
balances the standing pose (no navigation, no squat), and PinkIK drives the upper body via the same
23-D action layout used by the loco-manipulation variant.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Tabletop manipulation, no locomotion
   * - **Skills**
     - Pick, Place (no walk / squat / turn)
   * - **Embodiment**
     - Unitree G1 (29 DOF humanoid with Whole Body Controller, balance only)
   * - **Interop**
     - Isaac Lab Mimic
   * - **Scene**
     - Galileo Lab Environment (single shelf, no second table)
   * - **Manipulated Object(s)**
     - Apple (rigid body), Clay plate (destination, same-shelf placement)
   * - **Policy**
     - GR00T N1.6 (vision-language-action foundation model)
   * - **Post-training**
     - Imitation Learning
   * - **Dataset**
     - Self-recorded (collect via Step 2)
   * - **Checkpoint**
     - Self-trained (post-train via Step 4)
   * - **Physics**
     - PhysX (200Hz @ 4 decimation)
   * - **Closed-loop**
     - Yes (50Hz control)
   * - **Metrics**
     - Success rate


Workflow
--------

This tutorial covers the pipeline between creating an environment, collecting teleoperation
demonstrations, generating training data, fine-tuning a policy (GR00T N1.6), and evaluating the
policy in closed-loop. Follow the steps in order from teleoperation through closed-loop evaluation;
each step consumes the artifacts produced by the previous one.

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container

:docker_run_default:

Create the folders for the data and models:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/static_apple_tutorial
    mkdir -p $DATASET_DIR
    export MODELS_DIR=/models/isaaclab_arena/static_apple_tutorial
    mkdir -p $MODELS_DIR


Workflow Steps
^^^^^^^^^^^^^^

Follow the following steps to complete the workflow:

- :doc:`step_1_environment_setup`
- :doc:`step_2_teleoperation`
- :doc:`step_3_data_generation`
- :doc:`step_4_policy_training`
- :doc:`step_5_evaluation`


.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_environment_setup
   step_2_teleoperation
   step_3_data_generation
   step_4_policy_training
   step_5_evaluation
