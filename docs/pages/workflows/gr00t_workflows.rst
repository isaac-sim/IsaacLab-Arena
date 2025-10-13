Gr00t Workflows
===============

This example demonstrates how to work with **GR1 manipulation** and **G1 loco-manipulation** tasks in Isaac Lab - Arena, covering the complete workflow from generating demonstrations to running closed-loop policy (GR00T N1.5) inference and evaluation.


 .. list-table::
    :class: gallery
    :widths: auto

    * - **G1 Locomanipulation**

        .. image:: ../../images/g1_galileo_arena_box_pnp_locomanip.gif
          :height: 400px

    * - **GR1 Manipulation**

        .. image:: ../../images/kitchen_gr1_arena.gif
          :height: 400px

Overview
--------

The pipeline includes the following steps for both **GR1 manipulation** and **G1 loco-manipulation** tasks:

0. **Human Demonstration Collection** (Optional): Collect task-specific demonstrations
1. **Annotation and Augmentation using Isaac Lab Mimic** (Optional): Multiplying demonstrations using Isaac Lab Mimic
2. **Converting To LeRobot Format** (Optional): Converting recorded data to LeRobot format for GR00T training
3. **Replaying Recorded Data**: Replaying original HDF5 demonstrations for validation
4. **Replaying LeRobot Converted Data**: Replaying converted LeRobot dataset trajectories for validation
5. **Post-training and Closed-loop Policy Inference**: Taining and running GR00T N1.5 policies in closed-loop

.. note::
   Steps 0-2 are optional. You can skip one or more of these steps by using our ready-to-use datasets for each step by downloading them from the `Download Ready-To-Use Data`_ section.



Human Demonstration Collection
------------------------------

You can either record your own demonstrations by teleoperating a robot in Isaac Lab or use the provided datasets.

.. tabs::

    .. tab:: G1 Loco-Manipulation

      Isaac Lab Arena does not support collecting data for G1 loco-manipulation task yet. Please use the provided datasets in the `Download Ready-To-Use Annotated Dataset`_ section below.


    .. tab:: GR1 Manipulation

      See the following sections on how to record upper body manipulation demonstrations for the GR1 humanoid.
      For recording demonstrations for the GR1 humanoid, we support the `Apple Vision Pro <https://www.apple.com/vision-pro/>`_ as a teleoperation device.
      Find additional details on recording demonstrations for the GR1 humanoid in the `Isaac Lab humanoid data generation documentation <https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/teleop_imitation.html#demo-data-generation-and-policy-training-for-a-humanoid-robot>`_.

      To record demonstrations using the Apple Vision Pro and generate additional demonstrations using Isaac Lab Mimic, you need to:

      1. Install the `Isaac XR Teleop Sample Client App <https://isaac-sim.github.io/IsaacLab/v2.1.0/source/how-to/cloudxr_teleoperation.html#build-and-install-the-isaac-xr-teleop-sample-client-app-for-apple-vision-pro>`_ on your Apple Vision Pro.

      2. Start the CloudXR Runtime Docker container on your Isaac Lab workstation (corresponds to option 2 from the `CloudXR Runtime documentation <https://isaac-sim.github.io/IsaacLab/v2.1.0/source/how-to/cloudxr_teleoperation.html#run-isaac-lab-with-the-cloudxr-runtime>`_):

      .. code:: bash

         cd <isaac_arena_workspace>/submodules/IsaacLab

      .. code:: bash

         mkdir openxr

      .. code:: bash

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

      3. In a separate terminal, start the Isaac Lab - Arena docker container:

      .. code:: bash

        ./docker/run_docker.sh

      4. From within this docker, start the record script launching IsaacLab with the AR Client enabled:

      .. code:: bash

         python isaac_arena/scripts/record_demos.py \
             --dataset_file /datasets/<output_file>.hdf5 \
             --num_demos 10 \
             --num_success_steps 2 \
             gr1_open_microwave \
             --teleop_device avp_handtracking

Annotation and Augmentation using Isaac Lab Mimic
-------------------------------------------------

.. note::
   If desired, you may skip the `Annotation and Augmentation using Isaac Lab Mimic`_ step of this example and download a pre-generated dataset for use.
   To do so, skip to the `Download Ready-To-Use Annotated Dataset`_ section below.

Use Isaac Lab Mimic to generate an HDF5 dataset for GR00T training. Isaac Lab Mimic automatically generates additional robot demonstrations
from a small set of annotated demonstrations by using rigid body transformations to introduce variations to the dataset. During data generation,
object initial poses are randomized to introduce variation and boost dataset diversity. Noise is added to the robot's actions to further
increase diversity. For more information about Isaac Lab Mimic, please visit the Isaac Lab documentation `here <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#teleoperation-and-imitation-learning-with-isaac-lab-mimic>`_.


.. tabs::

    .. tab:: G1 Loco-Manipulation

      Use the annotated dataset (from `Download Ready-To-Use Annotated Dataset`_ section) and Isaac Lab Mimic to generate a new dataset of 100 demonstrations for GR00T training .

      Before proceeding, make sure you are inside the Isaac Lab - Arena docker container:

      .. code:: bash

         ./docker/run_docker.sh

      Then run the following command to generate the dataset:

      .. code:: bash

         python isaac_arena/scripts/generate_dataset.py \
         --headless \
         --enable_cameras \
         --mimic \
         --input_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
         --output_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_generated.hdf5 \
         --generation_num_trials 100 \
         --device cpu galileo_g1_locomanip_pick_and_place \
         --object brown_box \
         --embodiment g1_wbc_pink

      .. note::

         The above command runs data generation in headless mode for faster execution. If you want to see the data generation process,
         remove the ``--headless`` flag.

      Data generation times may vary from ~1 hour to multiple hours depending on your compute hardware.
      After the generation is complete, the dataset can be visualized using the ``replay_demos.py`` script.

      .. code:: bash

         python isaac_arena/scripts/replay_demos.py \
         --enable_cameras \
         --dataset_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_generated.hdf5 \
         --device cpu galileo_g1_locomanip_pick_and_place \
         --object brown_box \
         --embodiment g1_wbc_pink

    .. tab:: GR1 Manipulation

      1. The recorded demonstrations need to be segmented into subtasks before the next step. This is done by manual annotation, following the instructions printed to the terminal after running the following command:

      .. code:: bash

         python isaac_arena/scripts/annotate_demos.py
             --input_file /datasets/<recorded_demos>.hdf5
             --output_file /datasets/<recorded_demos_annotated>.hdf5
             --enable_pinocchio
             --mimic
             gr1_open_microwave

      2. Generate additional demonstrations using Isaac Lab Mimic from the annotated demonstrations (find more detail in the `Generating additional demonstrations <https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/teleop_imitation.html#generating-additional-demonstrations>`_ section of the Isaac Lab Mimic documentation).

      .. code:: bash

         python isaac_arena/scripts/generate_dataset.py
             --generation_num_trials 50
             --num_envs 10
             --input_file /datasets/<recorded_demos_annotated>.hdf5
             --output_file /datasets/<recorded_demos_annotated_generated>.hdf5
             --enable_pinocchio
             --enable_cameras
             --headless
             --mimic
             gr1_open_microwave



Converting To LeRobot Format
----------------------------
Next, we need to convert the HDF5 dataset generated by Isaac Lab Mimic to LeRobot format for GR00T training.
To do this, we provide a script that performs the following steps:

1. Loads HDF5 demonstrations
2. Extracts robot states, actions, and camera data
3. Applies joint remapping for GR00T compatibility
4. Generates MP4 videos from camera observations
5. Creates LeRobot-compatible dataset structure

Make sure you are running the docker container with GR00T dependencies. You can do this by running the following command:

.. code-block:: bash

    ./docker/run_docker.sh -g -G base # Include other docker arguments if needed


Convert the HDF5 demonstrations to LeRobot format for GR00T training:

.. tabs::

    .. tab:: G1 Loco-Manipulation

       .. code-block:: bash

          python isaac_arena/policy/data_utils/convert_hdf5_to_lerobot.py \
            --config_yaml_path isaac_arena/policy/config/g1_locomanip_config.yaml

       .. note::
          Configuration file (``g1_locomanip_config.yaml``):

          .. code-block:: yaml

             # Input/Output paths
             data_root: "/datasets/Arena-G1-Loco-Manipulation-Task"
             hdf5_name: "g1_galileo_box_npnp_mimic_generated_100_v2_action_noise_003.hdf5"    # Modify this to the name of the HDF5 file you want to convert

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


    .. tab:: GR1 Manipulation

       .. code-block:: bash

          python isaac_arena/policy/data_utils/convert_hdf5_to_lerobot.py \
            --config_yaml_path isaac_arena/policy/config/gr1_manip_config.yaml

       .. note::
          Make sure to point to the correct data_root and hdf5_name in the configuration file (``gr1_manip_config.yaml``):

          .. code-block:: yaml

             # Input/Output paths
             data_root: "/datasets/datasets/gr1_open_microwave/lerobot_gr1/lerobot_gr1/"
             hdf5_name: "gr1_open_microwave_50_generated.hdf5"    # Modify this to the name of the HDF5 file you want to convert

             # Task description
             language_instruction: "Reach out to the microwave and open it."
             task_index: 0



Replaying Recorded Data
-----------------------

Replay original HDF5 demonstrations to validate data quality:

.. tabs::
    .. tab:: G1 Loco-Manipulation

       .. code-block:: bash

          python isaac_arena/examples/policy_runner.py \
            --policy_type replay \
            --replay_file_path /datasets/my_g1_demos.hdf5 \
            --episode_name episode_0 \
            galileo_g1_locomanip_pick_and_place \
            --object brown_box \
            --embodiment g1_wbc_joint

    .. tab:: GR1 Manipulation


       .. code-block:: bash

          python isaac_arena/examples/policy_runner.py \
            --policy_type replay \
            --replay_file_path /datasets/my_gr1_demos.hdf5 \
            --episode_name episode_0 \
            gr1_open_microwave \
            --embodiment gr1_joint


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

.. tabs::
    .. tab:: G1 Loco-Manipulation

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
          dataset_path: /datasets/Arena-G1-Loco-Manipulation-Task/lerobot/

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

    .. tab:: GR1 Manipulation

       .. code-block:: bash

          python isaac_arena/examples/policy_runner.py \
            --policy_type replay_lerobot \
            --config_yaml_path isaac_arena/policy/gr00t/gr1_manip_replay_action_config.yaml \
            gr1_open_microwave \
            --embodiment gr1_joint


Key features:

- **Trajectory Selection**: Choose specific trajectories with ``--trajectory_index``
- **Action Chunking**: Executes 1 per step as replaying from the converted LeRobot dataset
- **Joint Remapping**: Converts between GR00T and Isaac Lab joint orders
- **Partial Replay**: Use ``--max_steps`` to replay only part of a trajectory

Post-training and Closed-loop Policy Inference
----------------------------------------------

[GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#nvidia-isaac-gr00t) is a foundation model for generalized humanoid robot reasoning and skills, trained on an extensive multimodal dataset that includes real-world, synthetic, and internet-scale data. The model is designed for cross-embodiment generalization and can be efficiently adapted to new robot embodiments, tasks, and environments through post training.

Make sure you are running the docker container with GR00T dependencies. You can do this by running the following command:

.. code-block:: bash

    ./docker/run_docker.sh -g -G base # Include other docker arguments if needed

To post-train the GR00T N1.5 policy on the converted LeRobot dataset, you can use the following command:

.. tabs::
    .. tab:: G1 Loco-Manipulation

       .. code-block:: bash

         cd submodules/Isaac-GR00T
         # Provide the directory where the GR00T-Lerobot data is stored as DATASET_PATH
         # Please use full path, instead of relative path

         python scripts/gr00t_finetune.py \
            --dataset_path=${DATASET_PATH} \
            --output_dir=${OUTPUT_DIR} \
            --data_config=unitree_g1_sim_wbc \
            --batch_size=24 \ # Adjust this to adapt your GPU memory
            --max_steps=20000 \
            --num_gpus=8 \
            --save_steps=5000 \
            --base_model_path=nvidia/GR00T-N1.5-3B \
            --no_tune_llm  \
            --tune_visual \
            --tune_projector \
            --tune_diffusion_model \
            --no-resume \
            --dataloader_num_workers=16 \
            --report_to=wandb \
            --embodiment_tag=new_embodiment

    .. tab:: GR1 Manipulation

       .. code-block:: bash

         cd submodules/Isaac-GR00T
         # Provide the directory where the GR00T-Lerobot data is stored as DATASET_PATH
         # Please use full path, instead of relative path

         python scripts/gr00t_finetune.py \
            --dataset_path=${DATASET_PATH} \
            --output_dir=${OUTPUT_DIR} \
            --data_config=gr1_arms_only \
            --batch_size=24 \    # Adjust this to adapt your GPU memory
            --max_steps=20000 \
            --num_gpus=8 \
            --save_steps=5000 \
            --base_model_path=nvidia/GR00T-N1.5-3B \
            --no_tune_llm  \
            --tune_visual \
            --tune_projector \
            --tune_diffusion_model \
            --no-resume \
            --dataloader_num_workers=16 \
            --report_to=wandb \
            --embodiment_tag=new_embodiment

.. hint::

1. Tuning with visual backend, action projector and diffusion model generally yields smaller trajectories errors (MSE), and higher closed-loop success rates.

2. If you prefer tuning with less powerful GPUs, please follow the [reference guidelines](https://github.com/NVIDIA/Isaac-GR00T/tree/n1-release?tab=readme-ov-file#3-fine-tuning) about other finetuning options.

Download the trained GR00T N1.5 checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. tabs::
    .. tab:: G1 Loco-Manipulation

       Download the trained GR00T N1.5 policy checkpoints from `huggingface: GN1x-Tuned-Arena-G1-Loco-Manipulation <https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation>`_.

       .. code-block:: bash

         huggingface-cli download \
            nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
            --local-dir /checkpoints/GN1x-Tuned-Arena-G1-Loco-Manipulation

    .. tab:: GR1 Manipulation

       Download the trained GR00T N1.5 policy checkpoints from `huggingface: GN1x-Tuned-Arena-GR1-Manipulation <https://huggingface.co/nvidia/GN1x-Tuned-Arena-GR1-Manipulation>`_.

       .. code-block:: bash

         huggingface-cli download \
            nvidia/GN1x-Tuned-Arena-GR1-Manipulation \
            --local-dir /checkpoints/GN1x-Tuned-Arena-GR1-Manipulation

Closed-loop Policy Inference and Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you are running the docker container with GR00T dependencies. You can do this by running the following command:

.. code-block:: bash

    ./docker/run_docker.sh -g -G base # Include other docker arguments if needed


Run trained GR00T policies in closed-loop:

.. tabs::
    .. tab:: G1 Loco-Manipulation

       .. code-block:: bash

         python isaac_arena/examples/policy_runner.py \
         --policy_type gr00t_closedloop \
         --policy_config_yaml_path isaac_arena/arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
         --num_steps 1000 \
         --enable_cameras \
         galileo_g1_locomanip_pick_and_place \
         --object brown_box \
         --embodiment g1_wbc_joint

    .. tab:: GR1 Manipulation

       .. code-block:: bash

         python isaac_arena/examples/policy_runner.py \
         --policy_type gr00t_closedloop \
         --policy_config_yaml_path isaac_arena/arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml \
         --num_steps 1000 \
         --enable_cameras \
         gr1_open_microwave \
         --embodiment gr1_joint

Configuration file for closed-loop inference of GR00T N1.5 policy

.. tabs::
    .. tab:: G1 Loco-Manipulation

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

    .. tab:: GR1 Manipulation

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

When the simulation app closes, you will see related metrics in the terminal.

.. tabs::
    .. tab:: G1 Loco-Manipulation

       Metrics: {success_rate: 0.5, num_episodes: 2}

    .. tab:: GR1 Manipulation

       Metrics: {success_rate: 1.0, door_moved_rate: 1.0, num_episodes: 2}

.. todo::
   (xinjie.yao, 2025-10-13): Add parallel env metrics example.

Policy features:

- **Vision-Language**: Processes RGB camera input and language instructions
- **Action Chunking**: Predicts multiple future actions for smooth control
- **Joint Space Control**: Outputs joint position targets
- **Real-time Inference**: Runs at simulation frequency (50Hz)
- **Whole Body Control(G1 Loco-Manipulation Only)**: Uses Whole Body Control (WBC) for robot control


Download Ready-To-Use Data
--------------------------

Download Ready-To-Use Annotated Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: G1 Loco-Manipulation

        Download the pre-recorded annotated dataset from `TODO(xyao, 2025-10-10): add link here <https://>`_ and place it
        under ``isaac_arena/datasets/Arena-G1-Loco-Manipulation-Task/``.

        .. hint::
           The annotated dataset can be visualized using the ``replay_demos.py`` script.
           Enter the Isaac Lab - Arena docker container:

        .. code:: bash

           ./docker/run_docker.sh

        Run the following command to play back the dataset:

        .. code:: bash

           python isaac_arena/scripts/replay_demos.py \
           --enable_cameras \
           --dataset_file /datasets/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
           --device cpu galileo_g1_locomanip_pick_and_place \
           --object brown_box \
           --embodiment g1_wbc_pink


    .. tab:: GR1 Manipulation

        Download the pre-recorded annotated dataset from `TODO(cvolk, 2025-10-10): add link here <https://>`_ and place it
        under ``isaac_arena/datasets/Arena-GR1-Manipulation-Task/``.

        .. hint::
           The annotated dataset can be visualized using the ``replay_demos.py`` script.
           Enter the Isaac Lab - Arena docker container:

        .. code:: bash

           ./docker/run_docker.sh

        Run the following command to play back the dataset:

        .. code:: bash

           python isaac_arena/scripts/replay_demos.py \
           --enable_cameras \
           --dataset_file /datasets/Arena-GR1-Manipulation-Task/arena_gr1_manipulation_dataset_annotated.hdf5 \
           --device cpu gr1_open_microwave \
           --embodiment gr1_joint


Download Ready-To-Use Augmented Mimic Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you skipped the `Annotation and Augmentation using Isaac Lab Mimic`_ step of this example, you can download a pre-generated dataset for use.

.. tabs::

    .. tab:: G1 Loco-Manipulation
        Download a pre-generated HDF5 dataset from `Arena-G1-Loco-Manipulation-Task <https://huggingface.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task>`_.

        .. code-block:: bash

            huggingface-cli download \
                nvidia/Arena-G1-Loco-Manipulation-Task \
                --local-dir /datasets/Arena-G1-Loco-Manipulation-Task

    .. tab:: GR1 Manipulation
        Download a pre-generated HDF5 dataset from TODO(cvolk, 2025-10-10): add link here <>.

        .. code-block:: bash

            TODO(cvolk, 2025-10-10): add link here.


Download Ready-To-Use Converted LeRobot Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: G1 Loco-Manipulation

       Download the converted LeRobot data from `huggingface: Arena-G1-Loco-Manipulation-Task <https://huggingface.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task>`_.

       .. code-block:: bash

           huggingface-cli download \
               nvidia/Arena-G1-Loco-Manipulation-Task \
               --local-dir /datasets/Arena-G1-Loco-Manipulation-Task

    .. tab:: GR1 Manipulation

      Download the converted LeRobot data from `huggingface: Arena-GR1-Manipulation-Task <https://huggingface.co/datasets/nvidia/Arena-GR1-Manipulation-Task>`_.

      .. code-block:: bash

          huggingface-cli download \
              nvidia/Arena-GR1-Manipulation-Task \
              --local-dir /datasets/Arena-GR1-Manipulation-Task
