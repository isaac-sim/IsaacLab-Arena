G1 Loco-Manipulation Apple-to-Plate Task
========================================

This example demonstrates the complete workflow for the **G1 loco-manipulation apple-to-plate task** in Isaac Lab - Arena, covering environment setup and validation, teleoperation data collection (OpenXR with Meta Quest 3), data generation, policy post-training, and closed-loop evaluation.

This workflow mirrors the :doc:`G1 Loco-Manipulation Box Pick and Place Task <../locomanipulation/index>`, but replaces the brown box with a smaller, rounder object (an apple) and the blue bin with a flat plate. If you are new to Arena loco-manipulation, we recommend the box/bin workflow first since the manipulated objects are more forgiving.

Task Overview
-------------

**Task Name:** ``galileo_g1_locomanip_pick_and_place``

**Task Description:** The G1 humanoid robot navigates through a lab environment, picks up an apple
from a shelf, and places it onto a plate on the table to the right of the shelf. This task requires
full-body coordination including lower body locomotion, squatting, and bimanual manipulation, with
tighter end-effector control than the box pick-and-place due to the smaller, rounder object and
flatter target.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Room-scale loco-manipulation
   * - **Skills**
     - Squat, Turn, Walk, Pick, Place
   * - **Embodiment**
     - Unitree G1 (29 DOF humanoid with Whole Body Controller)
   * - **Interop**
     - Isaac Lab Mimic
   * - **Scene**
     - Galileo Lab Environment
   * - **Manipulated Object(s)**
     - Apple (rigid body), Clay plate (destination)
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

This tutorial covers the pipeline between creating an environment, collecting teleoperation demonstrations, generating training data,
fine-tuning a policy (GR00T N1.6), and evaluating the policy in closed-loop.
Follow the steps in order from teleoperation through closed-loop evaluation; each step consumes the
artifacts produced by the previous one.

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container

:docker_run_default:

Create the folders for the data and models:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/locomanip_apple_tutorial
    mkdir -p $DATASET_DIR
    export MODELS_DIR=/models/isaaclab_arena/locomanip_apple_tutorial
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
