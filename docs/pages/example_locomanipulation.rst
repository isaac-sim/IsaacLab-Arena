Example: Loco-Manipulation
==========================

This example demonstrates how to work with loco-manipulation tasks in Isaac Arena, covering the complete workflow from recording demonstrations to running closed-loop policy inference.

Overview
--------

The loco-manipulation pipeline includes:

1. **Recording Demonstrations**: Collecting human demonstrations via teleoperation
2. **Data Conversion**: Converting recorded data to LeRobot format for GR00T training
3. **Replay Recorded Data**: Replaying original HDF5 demonstrations for validation
4. **Replay LeRobot Data**: Replaying converted LeRobot dataset trajectories
5. **Closed-loop Policy Inference**: Running trained GR00T policies in closed-loop

Recording Demonstrations
------------------------

Use the teleoperation interface to record human demonstrations:

# TODO(xinjie.yao, 2025-10-03): add record_demos command

Mimic Generated Data
---------------------

Use Isaac Lab Mimic with added Mimic supports for loco-manipulation to generate demonstrations:

# TODO(xinjie.yao, 2025-10-03): add mimic command

Download Mimic Generated Data (Optional)
----------------------------------------

Download the recorded HDF5 data from `here <https://huggingface.co/nvidia/Arena-G1-Loco-Manipulation-Task>`_.

.. code-block:: bash

    huggingface-cli download \
        nvidia/Arena-G1-Loco-Manipulation-Task \
        --local-dir /datasets/arena_g1_loco_manipulation_task


Converting to LeRobot Format
----------------------------

Make sure you are running the docker container with GR00T dependencies. You can do this by running the following command:

.. code-block:: bash

    ./docker/run_docker.sh -g -G base # Include other docker arguments if needed

Convert HDF5 demonstrations to LeRobot format for GR00T training:

.. code-block:: bash

   python isaac_arena/policy/data_utils/convert_hdf5_to_lerobot.py \
     --config_yaml_path isaac_arena/policy/config/g1_locomanip_config.yaml

Configuration file (``g1_locomanip_config.yaml``):

.. code-block:: yaml

   # Input/Output paths
   data_root: "/datasets/"
   hdf5_name: "my_g1_demos.hdf5"    # Modify this to the name of the HDF5 file you want to convert

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

The conversion process:

1. Loads HDF5 demonstrations
2. Extracts robot states, actions, and camera data
3. Applies joint remapping for GR00T compatibility
4. Generates MP4 videos from camera observations
5. Creates LeRobot-compatible dataset structure

Downloaded converted LeRobot Data (Optional)
--------------------------------------------

Download the converted LeRobot data from `here <https://huggingface.co/nvidia/Arena-G1-Loco-Manipulation-Task>`_.

.. code-block:: bash

    huggingface-cli download \
        nvidia/Arena-G1-Loco-Manipulation-Task \
        --local-dir /datasets/Arena-G1-Loco-Manipulation-Task

Replaying Recorded Data
-----------------------

Replay original HDF5 demonstrations to validate data quality:

# TODO(xinjie.yao, 2025-10-03): verify this command

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py \
     --policy_type replay \
     --replay_file_path /datasets/my_g1_demos.hdf5 \
     --episode_name episode_0 \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

Options:

- ``--episode_name``: Specific episode to replay (optional, defaults to first)
- ``--validate_states``: Enable state validation during replay
- ``--headless``: Run without GUI for automated validation

Replaying LeRobot Converted Data
--------------------------------

Make sure you are running the docker container with GR00T dependencies. You can do this by running the following command:

.. code-block:: bash

    ./docker/run_docker.sh -g -G base # Include other docker arguments if needed

Replay trajectories from the converted LeRobot dataset:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py \
     --policy_type replay_lerobot \
     --config_yaml_path isaac_arena/policy/gr00t/g1_locomanip_replay_action_config.yaml \
     --trajectory_index 0 \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

Configuration file (``g1_locomanip_replay_action_config.yaml``):

.. code-block:: yaml

   # Dataset path (LeRobot format)
   dataset_path: /datasets/Arena-G1-Loco-Manipulation-Task
   
   # Action chunking parameters
   action_horizon: 16
   num_feedback_actions: 1
   
   # Robot configuration
   embodiment_tag: new_embodiment
   data_config: unitree_g1_sim_wbc
   
   # Joint mappings
   gr00t_joints_config_path: isaac_arena/policy/config/g1/gr00t_43dof_joint_space.yaml
   action_joints_config_path: isaac_arena/policy/config/g1/43dof_joint_space.yaml
   
   # Task mode
   task_mode: g1_locomanipulation

Key features:

- **Trajectory Selection**: Choose specific trajectories with ``--trajectory_index``
- **Action Chunking**: Executes 1 per step as replaying from the converted LeRobot dataset
- **Joint Remapping**: Converts between GR00T and Isaac Lab joint orders
- **Partial Replay**: Use ``--max_steps`` to replay only part of a trajectory

Post-training Policy
--------------------

To post-train the GR00T N1.x policy on the converted LeRobot dataset, you can use the following command:
# TODO(xinjie.yao, 2025-10-03): add post-training policy command

Download the trained GR00T N1.x checkpoints
-------------------------------------------

Download the trained GR00T N1.x policy checkpoints from `here <https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation>`_.

.. code-block:: bash

    huggingface-cli download \
        nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
        --local-dir /checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation

Closed-loop Policy Inference and Evaluation
-------------------------------------------

Make sure you are running the docker container with GR00T dependencies. You can do this by running the following command:

.. code-block:: bash

    ./docker/run_docker.sh -g -G base # Include other docker arguments if needed

# TODO(xinjie.yao, 2025-10-03): add evaluation command

Run trained GR00T policies in closed-loop:

.. code-block:: bash

   python isaac_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaac_arena/policy/gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
     --num_steps 1000 \
     --enable_cameras \
     galileo_g1_locomanip_pick_and_place \
     --object brown_box \
     --embodiment g1_wbc_joint

Configuration file (``g1_locomanip_gr00t_closedloop_config.yaml``):

.. code-block:: yaml

   # Model configuration
   model_path: /checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation
   embodiment_tag: new_embodiment
   data_config: unitree_g1_sim_wbc
   
   # Task configuration
   language_instruction: "Pick up the brown box and place it in the blue bin"
   task_mode: g1_locomanipulation
   
   # Inference parameters
   denoising_steps: 10
   policy_device: cuda
   target_image_size: [256, 256, 3]
   
   # Joint mappings
   gr00t_joints_config_path: isaac_arena/policy/config/g1/gr00t_43dof_joint_space.yaml
   action_joints_config_path: isaac_arena/policy/config/g1/43dof_joint_space.yaml

Policy features:

- **Vision-Language**: Processes RGB camera input and language instructions
- **Action Chunking**: Predicts multiple future actions for smooth control
- **Joint Space Control**: Outputs joint position targets
- **Real-time Inference**: Runs at simulation frequency (50Hz)
- **Whole Body Control**: Uses Whole Body Control (WBC) for robot control

Advanced Usage
--------------

Bring your own environment & policy

# TODO(xinjie.yao, 2025-10-03): add advanced usage command
