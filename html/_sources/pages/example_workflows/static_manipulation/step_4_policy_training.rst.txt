Policy Post-training
--------------------

This workflow covers post-training an example policy using the generated dataset,
here we use `GR00T N1.5 <https://github.com/NVIDIA/Isaac-GR00T>`_ as the base model.

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
         nvidia/Arena-GR1-Manipulation-Task \
         arena_gr1_manipulation_dataset_generated.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR


**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/static_manipulation_tutorial
    export MODELS_DIR=/models/isaaclab_arena/static_manipulation_tutorial

.. note::
    The GR00T N1.5 codebase does not support running on Blackwell architecture by default. There are
    instructions `here <https://github.com/NVIDIA/Isaac-GR00T?tab=readme-ov-file#faq>`_ to building certain packages from source to support running on these architectures.
    We have not tested these instructions, and therefore we do not recommend using
    the **Base + GR00T** container for policy post-training and evaluation on
    Blackwell architecture, like RTX 50 series, RTX Pro 6000 or DGX Spark.


Step 1: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GR00T N1.5 requires the dataset to be in LeRobot format.
We provide a script to convert from the IsaacLab Mimic generated HDF5 dataset to LeRobot format.
Note that this conversion step can be skipped by downloading the pre-converted LeRobot format dataset.

.. dropdown:: Download Pre-converted LeRobot Dataset (skip conversion step)
   :animate: fade-in

   These commands can be used to download the pre-converted LeRobot format dataset,
   such that the conversion step can be skipped.

   .. code-block:: bash

      hf download \
         nvidia/Arena-GR1-Manipulation-Task \
         --include lerobot/* \
         --repo-type dataset \
         --local-dir $DATASET_DIR/arena_gr1_manipulation_dataset_generated

   If you download this dataset, you can skip the conversion step below and continue to the next step.


Convert the HDF5 dataset to LeRobot format for policy post-training:

.. code-block:: bash

   python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot.py \
     --yaml_file isaaclab_arena_gr00t/config/gr1_manip_config.yaml

This creates a folder ``$DATASET_DIR/arena_gr1_manipulation_dataset_generated/lerobot`` containing parquet files with states/actions, MP4 camera recordings, and dataset metadata. The converter is controlled by a config file at
``isaaclab_arena_gr00t/config/gr1_manip_config.yaml``.

.. dropdown:: Configuration file (``gr1_manip_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      # Input/Output paths
      data_root: /datasets/isaaclab_arena/static_manipulation_tutorial
      hdf5_name: "arena_gr1_manipulation_dataset_generated.hdf5"

      # Task description
      language_instruction: "Reach out to the microwave and open it."
      task_index: 0

      # Data field mappings
      state_name_sim: "robot_joint_pos"
      action_name_sim: "processed_actions"
      pov_cam_name_sim: "robot_pov_cam_rgb"


      # Output configuration
      fps: 50
      chunks_size: 1000


Step 2: Post-train Policy
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

      .. code-block:: bash

         cd submodules/Isaac-GR00T

         python scripts/gr00t_finetune.py \
         --dataset_path=$DATASET_DIR/arena_gr1_manipulation_dataset_generated/lerobot \
         --output_dir=$MODELS_DIR \
         --data_config=fourier_gr1_arms_only \
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
         --embodiment_tag=gr1

   .. tab:: Low Hardware Requirements

      Training takes approximately 2-3 hours on 1x Ada6000 GPU.

      Training Configuration:

      - **Base Model:** GR00T-N1.5-3B (foundation model)
      - **Tuned Modules:** Visual backbone, projector, diffusion model
      - **Frozen Modules:** LLM (language model)
      - **Batch Size:** 24 (adjust based on GPU memory)
      - **Training Steps:** 20,000
      - **GPUs:** 1 (single-GPU training)
      - **LoRA Fine-tuning:** Enabled
      - **LoRA Rank:** 128

      To post-train the policy, run the following command

      .. code-block:: bash

         cd submodules/Isaac-GR00T

         python scripts/gr00t_finetune.py \
         --dataset_path=$DATASET_DIR/arena_gr1_manipulation_dataset_generated/lerobot \
         --output_dir=$MODELS_DIR \
         --data_config=fourier_gr1_arms_only \
         --batch_size=24 \
         --max_steps=20000 \
         --num_gpus=1 \
         --save_steps=5000 \
         --base_model_path=nvidia/GR00T-N1.5-3B \
         --no_tune_llm \
         --tune_visual \
         --tune_projector \
         --tune_diffusion_model \
         --no-resume \
         --dataloader_num_workers=16 \
         --report_to=wandb \
         --embodiment_tag=gr1 \
         --lora_rank=128


see the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_
for information on how to adjust the training configuration to your hardware, to achieve
the best results.

.. note::

   If you encounter the following error when running the post-training command:

   .. code-block:: none

      FileNotFoundError: [Errno 2] No such file or directory: 'torchrun'

   Please add the following in your terminal before running the post-training command:

   .. code-block:: bash

      export PATH=/isaac-sim/kit/python/bin:${PATH}
