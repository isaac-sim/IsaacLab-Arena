G1 Loco-Manipulation Box Pick and Place Task
============================================

This example demonstrates the complete workflow for the **G1 loco-manipulation box pick and place task** in Isaac Lab - Arena, covering environment setup and validation, data augmentation, policy post-training, and closed-loop evaluation.

.. image:: ../../images/g1_galileo_arena_box_pnp_locomanip.gif
   :align: center
   :height: 400px

Task Overview
-------------

**Task ID:** ``galileo_g1_locomanip_pick_and_place``

**Task Description:** The G1 humanoid robot navigates through a lab environment, picks up a brown box from a shelf, and places it into a blue bin. This task requires full-body coordination including locomotion, squatting, and bimanual manipulation.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Room-scale loco-manipulation
   * - **Skills**
     - Squat, turn, walk, pick, place
   * - **Embodiment**
     - Unitree G1 (29 DOF humanoid with Whole Body Controller)
   * - **Interop**
     - Isaac Lab Mimic
   * - **Scene**
     - Galileo Lab Environment
   * - **Objects**
     - Brown box (rigid body)
   * - **Policy**
     - GR00T N1.5 (vision-language foundation model)
   * - **Post-training**
     - Imitation learning
   * - **Dataset**
     - `Arena-G1-Loco-Manipulation-Task <https://huggingface.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task>`_
   * - **Checkpoint**
     - `GN1x-Tuned-Arena-G1-Loco-Manipulation <https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation>`_
   * - **Physics**
     - PhysX (200Hz @ 4 decimation)
   * - **Closed-loop**
     - Yes (50Hz control)
   * - **Metrics**
     - Success rate


Workflows
---------

The complete pipeline includes the following workflows:

workflow #1: Environment Setup and Validation
workflow #2: Data Augmentation
workflow #3: Policy Post-Training
workflow #4: Closed-Loop Policy Inference and Evaluation

.. note::
   You can skip workflow #2-#3 by using the provided pre-generated datasets or post-trained checkpoints. See `Download Ready-to-Use Data`_ section.

Download Ready-to-Use Data
---------------------------

Download Annotated Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the pre-recorded annotated dataset for quick start:

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-G1-Loco-Manipulation-Task \
       arena_g1_loco_manipulation_dataset_annotated.hdf5 \
       --local-dir /datasets/Arena-G1-Loco-Manipulation-Task

This dataset contains manually annotated demonstrations segmented into subtasks.

Download Augmented Mimic Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the pre-generated dataset augmented with Isaac Lab Mimic (100 demonstrations):

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-G1-Loco-Manipulation-Task \
       arena_g1_loco_manipulation_dataset_generated.hdf5 \
       --local-dir /datasets/Arena-G1-Loco-Manipulation-Task

Download LeRobot Converted Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the pre-converted LeRobot format dataset ready for GR00T training:

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-G1-Loco-Manipulation-Task \
       lerobot \
       --local-dir /datasets/Arena-G1-Loco-Manipulation-Task

Download Trained GR00T Checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the trained GR00T N1.5 policy checkpoint:

.. code-block:: bash

   huggingface-cli download \
       nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
       --local-dir /checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation

Workflow #1: Environment Setup and Validation
----------------------------------------------

This workflow sets up the G1 loco-manipulation environment in Isaac Lab - Arena and validates it by replaying existing demonstrations.

Prerequisites
^^^^^^^^^^^^^

- Isaac Lab - Arena Docker container running
- At least one demonstration dataset (annotated or generated)

Step 1: Start Isaac Lab - Arena
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./docker/run_docker.sh

Step 2: Understand the Environment Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The environment is composed of three main components:

**Embodiment Configuration:**

.. code-block:: python

   from isaac_arena.embodiments.g1 import G1WBCPinkEmbodiment

   embodiment = G1WBCPinkEmbodiment(
       enable_cameras=True
   )

**Key Features:**

- 43 DOF control (legs, torso, arms, hands)
- Head-mounted RGB camera (480x640, FOV 128°x80°)
- Whole Body Controller (WBC) for locomotion
- PINK IK controller for upper body manipulation
- Action space: EEF poses + gripper states + navigation commands (x velocity, y velocity, yaw velocity, base height, torso pose)

**Scene Configuration:**

- Galileo Lab environment
- Brown box (spawned on shelf)
- Blue bin (target location)

**Task Configuration:**

.. code-block:: python

   from isaac_arena.tasks import PickAndPlaceTask
   
   task = PickAndPlaceTask(
       pick_object=brown_box,
       place_target=blue_bin,
       success_threshold=0.1  # Within 10cm of bin center
   )

Step 3: Validate Environment with Demo Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the annotated dataset to verify the environment setup:

.. code-block:: bash

   python isaac_arena/scripts/replay_demos.py \
     --enable_cameras \
     --dataset_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

You should see the G1 robot replaying the recorded trajectories in the simulation.


Workflow #2: Data Augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow covers annotating and augmenting the demonstration dataset using Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>, converting to LeRobot format, and post-training GR00T N1.5.

Prerequisites
^^^^^^^^^^^^^

- Annotated demonstration dataset
- Isaac Lab - Arena Docker container (default)

Step 1: Generate Augmented Dataset with Isaac Lab Mimic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic automatically generates additional demonstrations from a small set of annotated demonstrations by using rigid body transformations to introduce variations.

.. code-block:: bash

   # Start Isaac Lab - Arena container
   ./docker/run_docker.sh

   # Generate 100 demonstrations
   python isaac_arena/scripts/generate_dataset.py \
     --headless \
     --enable_cameras \
     --mimic \
     --input_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
     --output_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_generated.hdf5 \
     --generation_num_trials 100 \
     --device cpu \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

.. note::

   - Remove ``--headless`` to visualize data generation
   - Data generation takes 1-4 hours depending on hardware
   - Object poses are randomized to increase diversity
   - Action noise is added to improve robustness

Step 2: Validate Generated Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the generated dataset to verify quality:

.. code-block:: bash

   python isaac_arena/scripts/replay_demos.py \
     --enable_cameras \
     --dataset_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_generated.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_pink

Workflow #3: Policy Post-Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This workflow covers post-training an example policy using the augmented dataset,  here we use `GR00T N1.5 <https://github.com/NVIDIA/Isaac-GR00T>`_ as an example.

Prerequisites
^^^^^^^^^^^^^

- Augmented demonstration dataset
- Isaac Lab - Arena Docker container with GR00T dependencies

Step 1: Switch to Docker Container with GR00T Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Switch to the Docker container with GR00T dependencies by running the following command:

.. code-block:: bash

   ./docker/run_docker.sh -g -G base

Step 2: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert the HDF5 dataset to LeRobot format for GR00T training:

.. code-block:: bash

   python isaac_arena/policy/data_utils/convert_hdf5_to_lerobot.py \
     --config_yaml_path isaac_arena/policy/config/g1_locomanip_config.yaml

**Configuration file** (``g1_locomanip_config.yaml``):

.. code-block:: yaml

   # Input/Output paths
   data_root: "/datasets/Arena-G1-Loco-Manipulation-Task"
   hdf5_name: "arena_g1_loco_manipulation_dataset_generated.hdf5"

   # Task description
   language_instruction: "Pick up the brown box and place it in the blue bin"
   task_index: 2

   # Data field mappings
   state_name_sim: "robot_joint_pos"
   action_name_sim: "processed_actions"
   pov_cam_name_sim: "robot_head_cam"

   # Output configuration
   fps: 50
   chunks_size: 1000

This creates:

- ``/datasets/Arena-G1-Loco-Manipulation-Task/lerobot/data/`` - Parquet files with states/actions
- ``/datasets/Arena-G1-Loco-Manipulation-Task/lerobot/videos/`` - MP4 camera recordings
- ``/datasets/Arena-G1-Loco-Manipulation-Task/lerobot/meta/`` - Dataset metadata

Step 3: Post-Train Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd submodules/Isaac-GR00T

   python scripts/gr00t_finetune.py \
     --dataset_path=/datasets/Arena-G1-Loco-Manipulation-Task/lerobot/ \
     --output_dir=/checkpoints/my_g1_policy \
     --data_config=unitree_g1_sim_wbc \
     --batch_size=24 \
     --max_steps=20000 \
     --num_gpus=8 \
     --save_steps=5000 \
     --base_model_path=nvidia/GR00T-N1.5-3B \
     --no_tune_llm \
     --tune_visual \
     --tune_projector \
     --tune_diffusion_model \
     --no-resume \
     --dataloader_num_workers=16 \
     --report_to=wandb \
     --embodiment_tag=new_embodiment

**Training Configuration:**

- **Base Model:** GR00T-N1.5-3B (foundation model)
- **Tuned Modules:** Visual backbone, projector, diffusion model
- **Frozen Modules:** LLM (language model)
- **Batch Size:** 24 (adjust based on GPU memory)
- **Training Steps:** 20,000
- **GPUs:** 8 (multi-GPU training)

.. hint::

   For training with fewer GPUs or limited memory, see the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_.

Training takes approximately 4-8 hours on 8x L40s GPUs.

Workflow #4: Closed-Loop Policy Inference and Evaluation
---------------------------------------------------------

This workflow demonstrates running the trained GR00T N1.5 policy in closed-loop and evaluating it across multiple parallel environments.

Prerequisites
^^^^^^^^^^^^^

- Trained GR00T policy checkpoint
- Isaac Lab - Arena Docker container with GR00T dependencies
- GPU with at least 24GB VRAM

Step 1: Configure Closed-Loop Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create or verify the inference configuration file:

**Configuration** (``isaac_arena/arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml``):

.. code-block:: yaml

   # Model configuration
   model_path: /checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation
   embodiment_tag: new_embodiment
   data_config: unitree_g1_sim_wbc

   # Task configuration
   language_instruction: "Pick up the brown box and place it in the blue bin"
   task_mode_name: g1_locomanipulation

   # Inference parameters
   denoising_steps: 10
   policy_device: cuda
   target_image_size: [256, 256, 3]

   # Joint mappings
   gr00t_joints_config_path: isaac_arena/policy/config/g1/gr00t_43dof_joint_space.yaml
   action_joints_config_path: isaac_arena/policy/config/g1/43dof_joint_space.yaml
   state_joints_config_path: isaac_arena/policy/config/g1/43dof_joint_space.yaml

Step 2: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The embodiment using in closed-loop policy inference is ``g1_wbc_joint``, which is different from ``g1_wbc_pink`` used in data generation and policy post-training.
Instead of using the PINK IK controller for upper body manipulation, the ``g1_wbc_joint`` embodiment uses the same WBC policy for locomotion and upper body joint positions for upper body manipulation.

Test the policy in a single environment with visualization using the ``g1_wbc_joint`` embodiment:
.. code-block:: bash

   python isaac_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaac_arena/arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
     --num_steps 1200 \
     --enable_cameras \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

**Expected Output:**

.. code-block:: text

   Metrics: {success_rate: 1.0, num_episodes: 1}

Step 3: Run Parallel Evaluation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate the policy across multiple parallel environments using the ``g1_wbc_joint`` embodiment:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaac_arena/arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
     --num_steps 1200 \
     --num_envs 16 \
     --enable_cameras \
     --headless \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

**Performance Notes:**

.. TODO::
   (xinjie.yao, 2025-10-15): Add performance notes

**Expected Output:**

.. code-block:: text

   Metrics: {success_rate: 0.75, num_episodes: 16}

Step 4: Analyze Results
^^^^^^^^^^^^^^^^^^^^^^^^

The evaluation outputs several metrics:

- **Success Rate:** Percentage of episodes where box was placed in bin
- **Num Episodes:** Total number of completed episodes


Troubleshooting
---------------

Policy Not Loading
^^^^^^^^^^^^^^^^^^

**Error:** ``FileNotFoundError: model_path not found``

**Solution:** Verify the checkpoint path in the config file:

.. code-block:: bash

   ls /checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation

