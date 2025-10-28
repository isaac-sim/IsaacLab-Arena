Policy Post-training
--------------------

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

This workflow covers post-training an example policy using the generated dataset,
here we use `GR00T N1.5 <https://github.com/NVIDIA/Isaac-GR00T>`_ as the base model.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Data Generation) <step_3_data_generation>` or downloaded the
pre-generated dataset from Hugging Face as described below.

.. dropdown:: Download Pre-generated Dataset (skip preceding steps)
   :animate: fade-in

   These commands can be used to download the LeRobot-formatted dataset ready for policy post-training,
   such that the preceding steps can be skipped.

   To download run (replacing ``<INPUT_DATASET_PATH>`` with the actual path):

   .. code-block:: bash

   huggingface-cli download \
       nvidia/Arena-GR1-Manipulation-Task \
       arena_gr1_manipulation_dataset_generated.hdf5 \
       --repo-type dataset \
       --local-dir <INPUT_DATASET_PATH>


Step 2: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GR00T N1.5 requires the dataset to be in LeRobot format.
We provide a script to convert from the IsaacLab Mimic generated HDF5 dataset to LeRobot format.
Note that this conversion step can be skipped by downloading the pre-converted LeRobot format dataset.

.. dropdown:: Download Pre-converted LeRobot Dataset (skip conversion step)
   :animate: fade-in

   These commands can be used to download the pre-converted LeRobot format dataset,
   such that the conversion step can be skipped.

   To download run (replacing ``<LEROBOT_DATASET_PATH>`` with the actual path):

   .. code-block:: bash

      huggingface-cli download \
         nvidia/Arena-G1-Loco-Manipulation-Task \
         lerobot \
         --repo-type dataset \
         --local-dir <LEROBOT_DATASET_PATH>

   If you download this dataset, you can skip the conversion step below and continue to the next step.


We first need to modify the configuration file to point to the correct input/output paths.
In the config file at ``isaaclab_arena/policy/config/gr1_manip_config.yaml``,
Replace ``<INPUT_DATASET_DIR>`` with the actual path.

.. todo:: (alexmillane, 2025-10-23): We should move the input/output paths out of the config file
   and onto the command line. Then change the statement above.


**Configuration file** (``gr1_manip_config.yaml``):

.. code-block:: yaml

   # Input/Output paths
   data_root: <INPUT_DATASET_DIR>
   hdf5_name: "arena_g1_loco_manipulation_dataset_generated.hdf5"

   # Task description
   language_instruction: "Pick up the brown box and place it in the blue bin"
   task_index: 0

   # Data field mappings
   state_name_sim: "robot_joint_pos"
   action_name_sim: "processed_actions"
   pov_cam_name_sim: "robot_head_cam"

   # Output configuration
   fps: 50
   chunks_size: 1000

Convert the HDF5 dataset to LeRobot format for policy post-training:

.. code-block:: bash

   python isaaclab_arena/policy/data_utils/convert_hdf5_to_lerobot.py \
     --config_yaml_path isaaclab_arena/policy/config/gr1_manip_config.yaml

This creates:

- ``<INPUT_DATASET_DIR>/lerobot/data/`` - Parquet files with states/actions
- ``<INPUT_DATASET_DIR>/lerobot/videos/`` - MP4 camera recordings
- ``<INPUT_DATASET_DIR>/lerobot/meta/`` - Dataset metadata



Step 3: Post-train Policy
^^^^^^^^^^^^^^^^^^^^^^^^^

We post-train the GR00T N1.5 policy on the task.

The GR00T N1.5 policy has 3 billion parameters so post training is an an expensive operation.
We provide two post-training options:
* Best Quality: 8 GPUs with 48GB memory
* Low Hardware Requirements: 1 GPU with 24GB memory


.. tabs::

   .. tab:: Best Quality

      Training takes approximately 4-8 hours on 8x L40s GPUs.

      Training Configuration:

      - **Base Model:** GR00T-N1.5-3B (foundation model)
      - **Tuned Modules:** Visual backbone, projector, diffusion model
      - **Frozen Modules:** LLM (language model)
      - **Batch Size:** 24 (adjust based on GPU memory)
      - **Training Steps:** 20,000
      - **GPUs:** 8 (multi-GPU training)

      To post-train the policy, run the following command
      (replacing ``<LEROBOT_DATASET_PATH>`` and ``<OUTPUT_DIR>`` with the actual paths):

      .. code-block:: bash

         cd submodules/Isaac-GR00T

         python scripts/gr00t_finetune.py \
         --dataset_path=<LEROBOT_DATASET_PATH> \
         --output_dir=<OUTPUT_DIR> \
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

   .. tab:: Low Hardware Requirements

      TBD

see the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_
for information on how to adjust the training configuration to your hardware, to achieve
the best results.
