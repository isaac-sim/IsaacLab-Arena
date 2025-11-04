GR1 Open Microwave Door Task
=============================

This example demonstrates the complete workflow for the **GR1 manipulation task of opening a microwave door** in Isaac Lab - Arena, covering environment setup and validation, teleoperation data collection, data generation with Isaac Lab Mimic, policy post-training, and closed-loop evaluation.

.. image:: ../../../images/kitchen_gr1_arena.gif
   :align: center
   :height: 400px

Task Overview
-------------

**Task ID:** ``gr1_open_microwave``

**Task Description:** The GR1T2 humanoid uses its upper body (arms and hands) to reach toward a microwave, and open the door.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation
   * - **Skills**
     - Reach, Open door
   * - **Embodiment**
     - Fourier GR1T2 (54 DOF humanoid)
   * - **Interop**
     - Isaac Lab Teleop, Isaac Lab Mimic
   * - **Scene**
     - Kitchen environment
   * - **Objects**
     - Microwave (articulated object)
   * - **Policy**
     - GR00T N1.5 (vision-language foundation model)
   * - **Post-training**
     - Imitation Learning
   * - **Dataset**
     - `Arena-GR1-Manipulation-Task <https://huggingface.co/datasets/nvidia/Arena-GR1-Manipulation-Task>`_
   * - **Checkpoint**
     - `GN1x-Tuned-Arena-GR1-Manipulation <https://huggingface.co/nvidia/GN1x-Tuned-Arena-GR1-Manipulation>`_
   * - **Physics**
     - PhysX (200Hz @ 4 decimation)
   * - **Closed-loop**
     - Yes (50Hz control)
   * - **Metrics**
     - Success rate, Door moved rate


Workflow
--------

This tutorial covers the pipeline between creating an environment, generating training data,
fine-tuning a policy (GR00T N1.5), and evaluating the policy in closed-loop.
A user can follow the whole pipeline, or can start at any intermediate step
by downloading the pre-generated output of the preceding step(s), which we provide
(described in the relevant step below).

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container

:docker_run_default_code_block:


We store data on Hugging Face, so you'll need log in to Hugging Face if you haven't already.

.. code-block:: bash

    hf auth login

You'll also need to create the folders for the data and models.
Create the folders for the data and models with:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/static_manipulation_tutorial
    mkdir -p $DATASET_DIR
    export MODELS_DIR=/models/isaaclab_arena/static_manipulation_tutorial
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
