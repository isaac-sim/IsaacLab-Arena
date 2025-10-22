GR1 Open Microwave Door Task
=============================

This example demonstrates the complete workflow for the **GR1 manipulation task of opening a microwave door** in Isaac Lab - Arena, covering environment setup and validation, teleoperation data collection, data generation with Isaac Lab Mimic, policy post-training, and closed-loop evaluation.

.. image:: ../../images/kitchen_gr1_arena.gif
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


Workflows
---------

The complete pipeline includes the following workflows:

- `Workflow #1: Environment Setup and Validation`_

- `Workflow #2: Teleoperation Data Collection`_

- `Workflow #3: Data Generation`_

- `Workflow #4: Policy Post-Training`_

- `Workflow #5: Closed-Loop Policy Inference and Evaluation`_

.. note::
   You can skip workflow #2-#4 by using the provided pre-generated datasets or post-trained checkpoints. See `Download Ready-to-Use Data`_ section.



Download Ready-to-Use Data
---------------------------

You can start at intermediate steps in the workflow by downloading an (intermediate) dataset or checkpoint.

Prerequisites
^^^^^^^^^^^^^

To download the pre-made datasets and checkpoints you need to
`install <https://huggingface.co/docs/huggingface_hub/installation>`_
and login to Hugging Face CLI.

.. code-block:: bash

   pip install -U "huggingface_hub[cli]"
   huggingface-cli login

Download Annotated Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the pre-recorded annotated dataset for quick start:

Make sure you setup Hugging Face CLI outside the container by following the instructions in the `Hugging Face CLI Installation <https://huggingface.co/docs/huggingface_hub/installation>`_.

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-GR1-Manipulation-Task \
       arena_gr1_manipulation_dataset_annotated.hdf5 \
       --local-dir $YOUR_LOCAL_DATA_DIR   # Make sure this is a directory on your local machine, and virtually mounted to the container.

.. todo:: (clemens.volk, 2025-10-16): upload annotated dataset

This dataset contains manually annotated demonstrations segmented into subtasks. To generated a new dataset with more demonstrations, continue following the steps in `Workflow #3: Data Generation`_.

Download Generated Mimic Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the pre-generated dataset using Isaac Lab Mimic:

Make sure you setup Hugging Face CLI outside the container by following the instructions in the `Hugging Face CLI Installation <https://huggingface.co/docs/huggingface_hub/installation>`_.

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-GR1-Manipulation-Task \
       arena_gr1_manipulation_dataset_generated.hdf5 \
       --local-dir $YOUR_LOCAL_DATA_DIR   # Make sure this is a directory on your local machine, and virtually mounted to the container.

This dataset is generated from the annotated dataset using Isaac Lab Mimic, resulting a new dataset including more demonstrations.
To visually inspect the dataset, you can follow the steps in `Step 3: Validate Environment with Demo Replay`_ in `Workflow #1: Environment Setup and Validation`_.
If you want to post-train a policy using the generated dataset, you can continue following the steps in `Workflow #4: Policy Post-Training`_.


Download LeRobot Converted Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the pre-converted LeRobot format dataset:

Make sure you setup Hugging Face CLI outside the container by following the instructions in the `Hugging Face CLI Installation <https://huggingface.co/docs/huggingface_hub/installation>`_.

.. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-GR1-Manipulation-Task \
       --repo-type dataset \
       --local-dir $YOUR_LOCAL_DATA_DIR   # Make sure this is a directory on your local machine, and virtually mounted to the container.


This dataset is converted from the Mimic generated dataset with `Step 2: Convert to LeRobot Format`_.
To use it for policy post-training, you can continue following `Step 3: Post-Train Policy (Optional)`_ in `Workflow #4: Policy Post-Training`_.


Download Trained GR00T Checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the trained GR00T N1.5 policy checkpoint:

Make sure you setup Hugging Face CLI outside the container by following the instructions in the `Hugging Face CLI Installation <https://huggingface.co/docs/huggingface_hub/installation>`_.

.. code-block:: bash

   huggingface-cli download \
       nvidia/GN1x-Tuned-Arena-GR1-Manipulation \
       --local-dir $YOUR_LOCAL_CKPTS_DIR   # Make sure this is a directory on your local machine, and virtually mounted to the container.

Workflow #1: Environment Setup and Validation
----------------------------------------------

This workflow sets up the GR1 manipulation environment in Isaac Lab - Arena and validates it by replaying existing demonstrations.

Prerequisites
^^^^^^^^^^^^^

- Isaac Lab - Arena Docker container running
- At least one demonstration dataset

Step 1: Start Isaac Lab - Arena
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start the Arena Docker container:

   :docker_run_default:

Step 2: Understand the Environment Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Embodiment Configuration:**

.. code-block:: python

   from isaac_arena.embodiments.gr1t2 import GR1T2PinkEmbodiment

   embodiment = GR1T2PinkEmbodiment(enable_cameras=True)

**Key Features:**

- 36 DOF control (upper body: torso, arms, hands)
- Head-mounted RGB camera (512x512)
- PINK IK controller for upper body manipulation
- Lower body fixed in standing pose, gravity disabled

**Scene Configuration:**

- Kitchen environment
- Microwave with articulated door (revolute joint)
- Task: Open door to 80% (success threshold)

**Task Configuration:**

.. code-block:: python

   from isaac_arena.tasks import OpenDoorTask
   from isaac_arena.affordances import Openable
   # define microwave object reference, and background scene

   microwave = Openable(
       name="microwave",
       articulation_cfg=microwave_cfg,
       joint_name="door_joint"
   )

   task = OpenDoorTask(
       openable_object=microwave,
       openness_threshold=0.8,  # >80% open = success
       background_scene=background_scene
   )

Step 3: Validate Environment with Demo Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the generated dataset to verify the environment setup:

.. code-block:: bash

   python isaac_arena/scripts/replay_demos.py \
     --enable_cameras \
     --dataset_file /datasets/Arena-GR1-Manipulation-Task/arena_gr1_manipulation_dataset_generated.hdf5 \
     gr1_open_microwave \
     --embodiment gr1_pink

You should see the GR1 robot replaying the generated demonstrations, performing the microwave door opening task in the kitchen environment.

Workflow #2: Teleoperation Data Collection
--------------------------------------------

This workflow covers collecting demonstrations using Isaac Lab Teleop with Apple Vision Pro.

Prerequisites
^^^^^^^^^^^^^

- Apple Vision Pro with Isaac XR Teleop Sample Client installed
- CloudXR Runtime Docker container
- Isaac Lab - Arena Docker container
- Local network connection between workstation and Vision Pro

Step 1: Install Isaac XR Teleop App on Vision Pro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the `Isaac Lab CloudXR documentation <https://isaac-sim.github.io/IsaacLab/v2.1.0/source/how-to/cloudxr_teleoperation.html#build-and-install-the-isaac-xr-teleop-sample-client-app-for-apple-vision-pro>`_ to build and install the app on your Vision Pro.

Step 2: Start CloudXR Runtime Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a terminal, start the CloudXR runtime:

.. code-block:: bash

   cd submodules/IsaacLab
   mkdir -p openxr

   docker run -it --rm --name cloudxr-runtime \
     --user $(id -u):$(id -g) \
     --gpus=all \
     -e "ACCEPT_EULA=Y" \
     --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
     -p 48010:48010 \
     -p 47998:47998/udp \
     -p 47999:47999/udp \
     -p 48000:48000/udp \
     -p 48005:48005/udp \
     -p 48008:48008/udp \
     -p 48012:48012/udp \
     nvcr.io/nvidia/cloudxr-runtime:5.0.0

Step 3: Start Isaac Lab - Arena Recording
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a separate terminal, start the recording session:

Start the Arena Docker container:

   :docker_run_default:

Start the recording session:

.. code-block:: bash

   python isaac_arena/scripts/record_demos.py \
     --dataset_file /datasets/my_gr1_demos.hdf5 \
     --num_demos 10 \
     --num_success_steps 2 \
     gr1_open_microwave \
     --teleop_device avp_handtracking

Step 4: Connect Vision Pro and Record
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Launch the Isaac XR Teleop app on Vision Pro
2. Enter your workstation's IP address
3. Wait for connection (you should see the simulation in VR)
4. Use hand tracking to control the GR1 robot:

   - Hand positions control end-effector targets
   - Pinch gestures control grippers

5. Complete the task (open microwave door)
6. Press the "Submit Demo" button in VR

Repeat for all demonstrations. The script will automatically save successful demonstrations to the HDF5 file.

.. hint::

   For best results:

   - Move slowly and smoothly
   - Keep hands within tracking volume
   - Ensure good lighting for hand tracking
   - Complete at least 10 successful demonstrations

Workflow #3: Data Generation
----------------------------

This workflow covers generating a new dataset using `Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.

Prerequisites
^^^^^^^^^^^^^

- Recorded demonstration dataset (10+ demos)
- Isaac Lab - Arena Docker container (default)

Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Segment demonstrations into subtasks (reach, grasp, pull):

.. code-block:: bash

   python isaac_arena/scripts/annotate_demos.py \
     --input_file /datasets/my_gr1_demos.hdf5 \
     --output_file /datasets/my_gr1_demos_annotated.hdf5 \
     --enable_pinocchio \
     --mimic \
     gr1_open_microwave

Follow the on-screen instructions to mark subtask boundaries:

1. **Reach:** Robot reaches toward the microwave door
2. **Open door:** Robot opens the door

Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic automatically generates additional demonstrations from a small set of annotated demonstrations by using rigid body transformations to introduce variations.

Generate the dataset:

.. code-block:: bash

   python isaac_arena/scripts/generate_dataset.py \
     --generation_num_trials 50 \
     --num_envs 10 \
     --input_file /datasets/my_gr1_demos_annotated.hdf5 \
     --output_file /datasets/my_gr1_demos_generated.hdf5 \
     --enable_pinocchio \
     --enable_cameras \
     --headless \
     --mimic \
     gr1_open_microwave

.. note::

   - Remove ``--headless`` to visualize data generation
   - Data generation takes 30-60 minutes depending on hardware
   - Microwave poses are randomized to increase diversity
   - Action noise is added to improve robustness

Step 3: Validate Generated Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python isaac_arena/scripts/replay_demos.py \
     --enable_cameras \
     --dataset_file /datasets/my_gr1_demos_generated.hdf5 \
     gr1_open_microwave \
     --embodiment gr1_joint


Workflow #4: Policy Post-training
----------------------------------

This workflow covers post-training an example policy using the augmented dataset, here we use `GR00T N1.5 <https://github.com/NVIDIA/Isaac-GR00T>`_ as the base model.

Step 1: Switch to Docker Container with GR00T Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Switch to the Docker container **with GR00T dependencies** by running the following command:

:docker_run_gr00t:

Step 2: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert the HDF5 dataset to LeRobot format for policy post-training:

.. code-block:: bash

   python isaac_arena/policy/data_utils/convert_hdf5_to_lerobot.py \
     --config_yaml_path isaac_arena/policy/config/gr1_manip_config.yaml

**Configuration** (``gr1_manip_config.yaml``):

.. code-block:: yaml

   # Input/Output paths
   data_root: "/datasets/Arena-GR1-Manipulation-Task"
   hdf5_name: "my_gr1_demos_generated.hdf5"

   # Task description
   language_instruction: "Reach out to the microwave and open it."
   task_index: 0

   # Data field mappings
   state_name_sim: "robot_joint_pos"
   action_name_sim: "processed_actions"
   pov_cam_name_sim: "robot_pov_cam"

   # Output configuration
   fps: 50
   chunks_size: 1000

Step 3: Post-train Policy (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo:: (xinjie.yao, 2025-10-16): include instruction about how to retrieve the training ckpt inside or outside the container

Depending on your GPU memory, you can adjust training config parameters to post-train the policy. We provide an example below for reference.

.. code-block:: bash

   cd submodules/Isaac-GR00T

   python scripts/gr00t_finetune.py \
     --dataset_path=/datasets/Arena-GR1-Manipulation-Task/lerobot/ \
     --output_dir=/checkpoints/my_gr1_policy \
     --data_config=gr1_arms_only \
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

- **Base Model:** GR00T-N1.5-3B
- **Data Config:** ``gr1_arms_only`` (upper body control)
- **Tuned Modules:** Visual, projector, diffusion
- **Training Time:** ~4-6 hours on 8x L40s GPUs

.. hint::

   For training with fewer GPUs or limited memory, see the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_.

Workflow #5: Closed-Loop Policy Inference and Evaluation
--------------------------------------------------------

This workflow demonstrates running the trained GR00T policy in closed-loop and evaluating across multiple parallel environments.

Prerequisites
^^^^^^^^^^^^^

- Trained GR00T policy checkpoint
- Isaac Lab - Arena Docker container **with GR00T dependencies**

Step 1: Configure Closed-Loop Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Configuration** (``isaac_arena/arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml``):

.. code-block:: yaml

   # Model configuration
   model_path: /checkpoints/GN1x-Tuned-Arena-GR1-Manipulation
   embodiment_tag: gr1
   data_config: gr1_arms_only

   # Task configuration
   language_instruction: "Reach out to the microwave and open it."
   task_mode_name: gr1_manipulation

   # Inference parameters
   denoising_steps: 10
   policy_device: cuda
   target_image_size: [256, 256, 3]

   # Joint mappings
   gr00t_joints_config_path: isaac_arena/policy/config/gr1/gr00t_26dof_joint_space.yaml
   action_joints_config_path: isaac_arena/policy/config/gr1/36dof_joint_space.yaml
   state_joints_config_path: isaac_arena/policy/config/gr1/54dof_joint_space.yaml

Step 2: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The embodiment using in closed-loop policy inference is ``gr1_joint``, which is different from ``gr1_pink`` used in data generation.
Instead of using the target end-effector poses with PINK IK controller for upper body manipulation, the ``gr1_joint`` uses upper body joint positions instead.

Test the policy in a single environment with visualization using the ``gr1_joint`` embodiment:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaac_arena/arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml \
     --num_steps 400 \
     --enable_cameras \
     gr1_open_microwave \
     --embodiment gr1_joint

**Expected Output:**

Depending on how many episodes the simulation runs (num_steps = 400 steps in this case), you will see a similar metrics output as below:

.. code-block:: text

   Metrics: {success_rate: 1.0, door_moved_rate: 1.0, num_episodes: 2}

Step 3: Run Parallel Evaluation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate the policy across multiple parallel environments:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaac_arena/arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml \
     --num_steps 400 \
     --num_envs 16 \
     --enable_cameras \
     --headless \
     gr1_open_microwave \
     --embodiment gr1_joint

**Performance Notes:**

.. todo:: (xinjie.yao, 2025-10-15): Add performance notes

**Expected Output:**

.. code-block:: text

   Metrics: {success_rate: 0.9375, door_moved_rate: 0.9375, num_episodes: 32}

Step 4: Analyze Results
^^^^^^^^^^^^^^^^^^^^^^^^

The evaluation outputs several metrics:

- **Success Rate:** Percentage of episodes where door was opened ≥ 80%
- **Door Moved Rate:** Percentage of episodes where door moved > 10%
- **Num Episodes:** Total number of completed episodes

Troubleshooting
---------------

Policy Not Found During Closed-Loop Policy Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Error:** ``FileNotFoundError: model_path not found`` or ``OSError: [Errno 2] No such file or directory: '/checkpoints/GN1x-Tuned-Arena-GR1-Manipulation'``

**Solution:** Verify the checkpoint path in the config file:

.. code-block:: bash

   # If the checkpoint is inside the container, you can list it:
   ls /checkpoints/GN1x-Tuned-Arena-GR1-Manipulation
   # If the checkpoint is not found, you can download it from Hugging Face:
   huggingface-cli download \
       nvidia/GN1x-Tuned-Arena-GR1-Manipulation \
       --local-dir /checkpoints/GN1x-Tuned-Arena-GR1-Manipulation
