Policy Post-training
--------------------

This workflow covers post-training an example policy using the generated dataset,
here we use `GR00T N1.6 <https://github.com/NVIDIA/Isaac-GR00T>`_ as the base model.


**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/sequential_static_manipulation_tutorial
    export MODELS_DIR=/models/isaaclab_arena/sequential_static_manipulation_tutorial

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Data Generation) <step_3_data_generation>` or downloaded the
pre-generated dataset from Hugging Face as described below.

.. dropdown:: Download Pre-generated Dataset (skip preceding steps)
   :animate: fade-in

   These commands can be used to download the Mimic-generated HDF5 dataset ready for policy post-training,
   such that the preceding steps can be skipped.

   To download run:

   .. code-block:: bash

      hf download \
         nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task \
         ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR


Step 1: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GR00T N1.6 requires the dataset to be in LeRobot format.
We provide a script to convert from the IsaacLab Mimic generated HDF5 dataset to LeRobot format.
Note that this conversion step can be skipped by downloading the pre-converted LeRobot format dataset.

.. dropdown:: Download Pre-converted LeRobot Dataset (skip conversion step)
   :animate: fade-in

   These commands can be used to download the pre-converted LeRobot format dataset,
   such that the conversion step can be skipped.

   .. code-block:: bash

      hf download \
         nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task \
         --include "ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot/*" \
         --repo-type dataset \
         --local-dir $DATASET_DIR

   If you download this dataset, you can skip the conversion step below and continue to the next step.


Convert the HDF5 dataset to LeRobot format for policy post-training:

.. code-block:: bash

   python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
     --yaml_file isaaclab_arena_gr00t/lerobot/config/gr1_manip_ranch_bottle_config.yaml

This creates a folder ``$DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot`` containing parquet files with states/actions, MP4 camera recordings, and dataset metadata. The converter is controlled by a config file at
``isaaclab_arena_gr00t/lerobot/config/gr1_manip_ranch_bottle_config.yaml``.

.. dropdown:: Configuration file (``gr1_manip_ranch_bottle_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      # Input/Output paths
      data_root: /datasets/isaaclab_arena/sequential_static_manipulation_tutorial/ranch_bottle_into_fridge
      hdf5_name: "ranch_bottle_into_fridge_generated_100.hdf5"

      # Task description
      language_instruction: "Place the ranch dressing bottle on the top shelf of the fridge, and close the fridge door."
      task_index: 0

      # Data field mappings
      state_name_sim: "robot_joint_pos"
      action_name_sim: "processed_actions"
      pov_cam_name_sim: "robot_pov_cam_rgb"


      # Output configuration
      fps: 50
      chunks_size: 1000

Step 2: Verify Generated LeRobot Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To verify the generated LeRobot dataset, you first need to compute statistics information about the dataset.
This can be done by running the following command:

.. code-block:: bash

   python submodules/Isaac-GR00T/gr00t/data/stats.py $DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot gr1_joint

.. todo::
   There is a bug in ISAAC-GR00T submodule gr00t/data/stats.py. PR is submmited and in review from Gear.
   Use single gpu finetune command in the last step for now.

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type isaaclab_arena_gr00t.policy.replay_lerobot_action_policy.ReplayLerobotActionPolicy \
     --config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_replay_action_config.yaml \
     --enable_cameras \
     --device cpu \
     put_item_in_fridge_and_close_door \
     --embodiment gr1_joint

.. figure:: ../../../images/gr1_sequential_static_manip_mimic_datagen.gif
   :width: 100%
:alt: GR1 picking up and placing an object in a refrigerator and closing the door
   :align: center

   IsaacLab Arena GR1 picking up and placing an object in a refrigerator and closing the door (with action noise)

.. note::

   You should see the robot perform the manipulation task. Note that the robot's arms shake due to the action noise
   added during data generation, which is expected. If you observe inconsistent behavior, please check the data generation and conversion steps.

Step 3: Upload Dataset to OSMO (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to upload the generated LeRobot dataset to OSMO for it to be used in the next step, you can run the following command:

.. code-block:: bash

   osmo dataset upload gr1_ranch_bottle_into_fridge $DATASET_DIR/ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100


Step 4: Post-train Policy
^^^^^^^^^^^^^^^^^^^^^^^^^

We post-train the GR00T N1.6 policy on the task.

The GR00T N1.6 policy has 3 billion parameters so post training is an an expensive operation.
We provide three post-training options:

* Best Quality: 8 GPUs with 48GB memory (local)
* Best Quality: 8 GPUs with 48GB memory (OSMO)
* Low Hardware Requirements: 1 GPU with 24GB memory


.. tabs::

   .. tab:: Best Quality (local)

      Training takes approximately 4-8 hours on 8x L40s GPUs.

      Training Configuration:

      - **Base Model:** GR00T-N1.6-3B (foundation model)
      - **Tuned Modules:** Visual backbone, projector, diffusion model
      - **Frozen Modules:** LLM (language model)
      - **Global Batch Size:** 96 (adjust based on GPU memory)
      - **Training Steps:** 20,000
      - **GPUs:** 8 (multi-GPU training)

      To post-train the policy, run the following command

      .. code-block:: bash

         python -m torch.distributed.run --nproc_per_node=8 --standalone submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
         --dataset_path=$DATASET_DIR/arena_gr1_manipulation_dataset_generated/lerobot \
         --output_dir=$MODELS_DIR \
         --modality_config_path=isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
         --global_batch_size=96 \
         --max_steps=20000 \
         --num_gpus=8 \
         --save_steps=5000 \
         --save_total_limit=5 \
         --base_model_path=nvidia/GR00T-N1.6-3B \
         --no_tune_llm \
         --tune_visual \
         --tune_projector \
         --tune_diffusion_model \
         --no-resume \
         --dataloader_num_workers=16 \
         --use-wandb \
         --embodiment_tag=GR1 \
         --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

   .. tab:: Best Quality (OSMO)

      Training takes approximately 4-8 hours on 8x L40 GPUs.

      Training Configuration:

      - **Base Model:** GR00T-N1.6-3B (foundation model)
      - **Tuned Modules:** Visual backbone, projector, diffusion model
      - **Frozen Modules:** LLM (language model)
      - **Global Batch Size:** 96 (adjust based on GPU memory)
      - **Training Steps:** 20,000
      - **GPUs:** 8 (multi-GPU training)

      To post-train the policy, after you set the OSMO infrastructure following the OSMO instructions `<https://github.com/NVIDIA/OSMO/tree/main?tab=readme-ov-file#documentation>`
      you can submit the workflow to OSMO by running the following command:

      .. code-block:: bash

         osmo workflow submit osmo/finetune.yaml --pool ${POOL_NAME}


   .. tab:: Low Hardware Requirements

      Training takes approximately 2-3 hours on 1x Ada6000 GPU.

      Training Configuration:

      - **Base Model:** GR00T-N1.6-3B (foundation model)
      - **Tuned Modules:** Visual backbone, projector, diffusion model
      - **Frozen Modules:** LLM (language model)
      - **Global Batch Size:** 16 (adjust based on GPU memory)
      - **Training Steps:** 30,000
      - **GPUs:** 1 (single-GPU training)

      To post-train the policy, run the following command

      .. code-block:: bash

         CUDA_VISIBLE_DEVICES=0 python submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
         --dataset_path=$DATASET_DIR/arena_gr1_manipulation_dataset_generated/lerobot \
         --output_dir=$MODELS_DIR \
         --modality_config_path=isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
         --global_batch_size=16 \
         --max_steps=30000 \
         --num_gpus=1 \
         --save_steps=5000 \
         --base_model_path=nvidia/GR00T-N1.6-3B \
         --no_tune_llm \
         --tune_visual \
         --tune_projector \
         --tune_diffusion_model \
         --dataloader_num_workers=16 \
         --use-wandb \
         --embodiment_tag=GR1 \
         --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
         --save_total_limit=5

.. todo::
   Verify training locally

see the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_
for information on how to adjust the training configuration to your hardware, to achieve
the best results.

Step 5: Download Post-trained Model (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you post-train the model on OSMO, you can download the checkpoint to your local machine.
Note down the workflow name and output directory from the OSMO workflow job,
and add them to the following command:

.. code-block:: bash

   osmo data download swift://pdx.s8k.io/AUTH_team-isaac/mimic/datasets/${WORKFLOW_NAME}/checkpoint-20000 $MODELS_DIR/ranch_bottle_into_fridge/
