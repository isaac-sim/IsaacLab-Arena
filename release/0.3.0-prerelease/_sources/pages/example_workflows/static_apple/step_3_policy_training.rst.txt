Policy Post-Training (GR00T N1.7)
---------------------------------

This workflow covers post-training a
`GR00T N1.7 <https://github.com/NVIDIA/Isaac-GR00T/tree/4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e>`_
policy
directly on the **teleoperated demonstrations** exported in HDF5 from :doc:`step_2_teleoperation`.
The recorded HDF5 is converted to LeRobot format inside the Arena container, then handed off to a
**standalone Isaac-GR00T checkout** (``$ISAAC_GR00T_DIR`` from :doc:`index`) for finetuning. The
finetuned checkpoint is later served back to Arena over the server-client architecture in
:doc:`step_4_evaluation`.

The N1.7 finetune script lives in the standalone Isaac-GR00T repo, not in Arena's pinned
``submodules/Isaac-GR00T``. This lets you train on the latest GR00T release without bumping the
Arena submodule.

This page assumes you have either a successful recording at
``$DATASET_DIR/arena_g1_static_apple_dataset_recorded.hdf5`` from
:doc:`step_2_teleoperation` or the pre-generated HDF5 dataset downloaded below.


Step 1: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GR00T N1.7 consumes datasets in LeRobot format. The conversion runs inside the standard
**Base** Arena container.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the dataset directory:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/static_apple_tutorial

.. dropdown:: Download Pre-generated Dataset (skip teleoperation)
   :animate: fade-in

   These commands can be used to download the pre-recorded static apple HDF5 dataset ready for
   LeRobot conversion, such that the teleoperation step can be skipped.
   Run them where ``$DATASET_DIR`` points to the target directory; use the ``/datasets/...`` path
   inside Docker, or the matching host path outside Docker. The Hugging Face CLI from
   ``huggingface_hub`` must be installed in that environment.

   To download run:

   .. code-block:: bash

      hf download \
         nvidia/Arena-G1-Static-PickNPlace-Task \
         arena_g1_static_apple_dataset_recorded_200_demos.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR

      mv "$DATASET_DIR/arena_g1_static_apple_dataset_recorded_200_demos.hdf5" \
         "$DATASET_DIR/arena_g1_static_apple_dataset_recorded.hdf5"

.. caution::

   ``isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py`` expects each episode to include
   ego RGB under ``observations/camera_obs/robot_head_cam_rgb`` (see ``pov_cam_name_sim`` in the
   conversion config). Before bulk collection, run the conversion once on a short recording to
   confirm the layout matches.

Edit ``isaaclab_arena_gr00t/lerobot/config/g1_static_apple_config.yaml`` so ``hdf5_name`` matches
your recorded file (``arena_g1_static_apple_dataset_recorded.hdf5``) and ``data_root`` matches
``$DATASET_DIR``. The Hugging Face download command above renames the released HDF5 to this local
tutorial filename so the default config can be used unchanged.

Convert the HDF5 dataset to LeRobot format for policy post-training:

.. code-block:: bash

   python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
     --yaml_file isaaclab_arena_gr00t/lerobot/config/g1_static_apple_config.yaml


This creates a folder ``$DATASET_DIR/arena_g1_static_apple_dataset_recorded/lerobot`` containing
parquet files with states/actions, MP4 camera recordings, and dataset metadata.

The converter is controlled by a config file at
``isaaclab_arena_gr00t/lerobot/config/g1_static_apple_config.yaml``.

.. dropdown:: Configuration file (``g1_static_apple_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      # Input/Output paths
      data_root: /datasets/isaaclab_arena/static_apple_tutorial
      hdf5_name: "arena_g1_static_apple_dataset_recorded.hdf5"

      # Task description
      language_instruction: "move the apple to the plate"
      task_index: 3

      # Data field mappings
      state_name_sim: "robot_joint_pos"
      action_name_sim: "processed_actions"
      pov_cam_name_sim: "robot_head_cam_rgb"

      # Output configuration
      fps: 50
      chunks_size: 1000

The main differences from the loco-manipulation box config (``g1_locomanip_config.yaml``) are the
``data_root`` / ``hdf5_name`` pointing at the static apple-to-plate dataset and the
``language_instruction`` describing the same-shelf placement (no walking, no second table).
The 43-DoF action layout, embodiment tag, modality template and joint-space configurations are all
shared with the loco-manipulation variant — the static workflow does not need its own GR00T embodiment
config because the upper-body action channels and observation modalities are identical; only the
recorded body channel happens to stay at zero throughout each demo.

.. note::

   The recorder's ``processed_actions`` field already contains the 43-DoF joint-space targets
   that PinkIK produced during teleoperation. That is why the doc tells you to record with
   ``g1_wbc_agile_pink`` and evaluate with ``g1_wbc_agile_joint`` — the policy never sees the
   end-effector pose targets PinkIK consumed; it only sees the joint targets PinkIK *produced*.


Step 2: Post-train Policy (standalone Isaac-GR00T venv)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We post-train the GR00T N1.7 policy on the task using the **standalone Isaac-GR00T checkout** from
:doc:`index` (referenced as ``$ISAAC_GR00T_DIR``). This step runs **outside the Arena container** so
GR00T's dependencies do not have to coexist with the Arena/Isaac Sim ones.

The GR00T N1.7 policy has 3 billion parameters so post-training is an expensive operation.
The command below is the tested single-GPU configuration for this workflow.

Training takes approximately 2-3 hours for 20,000 steps on a single NVIDIA RTX 6000 Ada GPU.

Compute Requirements:

- **GPUs:** 1x NVIDIA RTX 6000 Ada or another GPU with at least 48 GB VRAM
- **System RAM:** 128 GB or more recommended

Training Configuration:

- **Base Model:** GR00T-N1.7-3B (foundation model, downloaded from Hugging Face on first run)
- **Tuned Modules:** Visual backbone, projector, diffusion model
- **Frozen Modules:** LLM (language model)
- **Batch Size:** 12 (adjust based on GPU memory)
- **Training Steps:** 20,000
- **Action horizon:** 40 (must match the diffusion head value used at evaluation; see note below)
- **Embodiment tag:** ``new_embodiment`` (case-insensitive; resolved to
  ``EmbodimentTag.NEW_EMBODIMENT`` by ``gr00t``)

To post-train the policy, open another terminal **outside** the Arena Base Docker container, set up
GR00T's native ``uv`` environment by following the
`GR00T installation guide
<https://github.com/NVIDIA/Isaac-GR00T/blob/4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e/README.md#installation-guide>`_,
and ``cd`` to ``$ISAAC_GR00T_DIR``. The launcher runs inside the standalone repo's ``uv``-managed venv. Replace
``/path/to/IsaacLab-Arena`` with the absolute path to your Arena clone so the
``--modality-config-path`` argument can register the WBC modality from Arena's source tree.

Because finetuning runs outside the Arena Docker container, set ``DATASET_DIR`` and ``MODELS_DIR``
in the standalone GR00T terminal to the host paths that correspond to the Docker mounts. With the
default Arena Docker mounts, these are usually:

.. code-block:: bash

   export DATASET_DIR=~/datasets/isaaclab_arena/static_apple_tutorial
   export MODELS_DIR=~/models/isaaclab_arena/static_apple_tutorial

.. code-block:: bash

   cd $ISAAC_GR00T_DIR

   uv run python -m torch.distributed.run --nproc_per_node=1 --standalone \
     gr00t/experiment/launch_finetune.py \
     --base-model-path nvidia/GR00T-N1.7-3B \
     --dataset-path $DATASET_DIR/arena_g1_static_apple_dataset_recorded/lerobot \
     --output-dir $MODELS_DIR/static_apple_n17_finetune \
     --modality-config-path /path/to/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py \
     --embodiment-tag new_embodiment \
     --global-batch-size 12 \
     --max-steps 20000 \
     --num-gpus 1 \
     --save-steps 5000 \
     --save-total-limit 5 \
     --no-tune-llm \
     --tune-visual \
     --tune-projector \
     --tune-diffusion-model \
     --dataloader-num-workers 8 \
     --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

.. note::

   **N1.7 CLI vs N1.6 CLI.** The N1.7 ``launch_finetune.py`` is built on `tyro
   <https://brentyi.github.io/tyro/>`_, so flags are kebab-case (``--base-model-path``, not
   ``--base_model_path``) and booleans use the ``--flag`` / ``--no-flag`` pair (``--no-tune-llm``).
   ``--color-jitter-params`` takes alternating ``key value`` pairs, not a JSON string. Run
   ``uv run python gr00t/experiment/launch_finetune.py --help`` from ``$ISAAC_GR00T_DIR`` to
   inspect the full argument set for the version you have checked out.

.. note::

   The ``--modality-config-path`` argument points to Arena's
   ``isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`` so that
   ``register_modality_config(...)`` runs and ``new_embodiment`` resolves to the WBC modality
   layout (5 state keys + 7 action keys). This is the **same file** the server consumes at
   evaluation time, so it is the single source of truth for the modality layout.

.. caution::

   **Action horizon must match between training and serving.** ``action_horizon`` is baked into the
   diffusion head at training time and cannot be changed at inference. The Arena server YAML
   used in :doc:`step_4_evaluation` ships with ``action_horizon: 40`` to match the value that this
   step trains. If you want a different horizon, change **both**:

   1. ``delta_indices=list(range(N))`` in
      ``isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`` for the action
      modality (controls what the LeRobot loader feeds the model during training).
   2. ``action_horizon: N`` and ``action_chunk_length: N`` (≤ ``action_horizon``) in the
      server-side YAML at
      ``isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml``.

If you have more powerful GPUs, please see the
`GR00T fine-tuning guidelines
<https://github.com/NVIDIA/Isaac-GR00T/blob/4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e/README.md#3-fine-tuning>`_ for
information on how to adjust the training configuration to your hardware. We recommend fine-tuning
the visual backbone, projector, and diffusion model for better results.


Recommendations for finetuning that works with AGILE on apple-to-plate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The static apple-to-plate environment runs the AGILE Whole Body Controller (lower-body) and either
PinkIK (recording) or direct joint control (evaluation). The following choices materially affect
whether the finetuned policy actually works in the AGILE-joint runtime:

#. **Record with AGILE, not HOMIE.** Keep the default ``--embodiment g1_wbc_agile_pink`` during
   teleoperation. AGILE's WBC is a single end-to-end velocity policy; HOMIE is a stand+walk pair.
   The lower-body joint targets PinkIK plus the WBC produce are systematically different between
   the two, so a HOMIE-trained policy will be off-distribution when served against the AGILE-joint
   eval embodiment.

#. **Don't record with the joint embodiment.** Use ``g1_wbc_agile_pink`` (PinkIK on top of AGILE),
   not ``g1_wbc_agile_joint``. The recorder writes the **joint-space output of PinkIK** as
   ``processed_actions``, which is what the policy needs to learn. Recording with
   ``g1_wbc_agile_joint`` would force the human teleoperator to drive 43 joint targets directly,
   which is impractical and not what eval uses anyway.

#. **Use Arena's** ``g1_sim_wbc_data_gr00t_n_1_7_config.py`` **for** ``--modality-config-path``\ **.**
   This registers the modality with ``NEW_EMBODIMENT`` (40-step action horizon for N1.7) and is the
   same file the Arena server consumes at eval. Keeping a single source of truth prevents skew
   between training and serving.

#. **Pick** ``action_horizon`` **deliberately.** The default (40) gives an 800 ms inference chunk at
   50 Hz, which trades responsiveness against compute, and is the maximum supported by the released
   GR00T N1.7 base model (``max_action_horizon = 40`` is baked into the checkpoint). For static
   apple-to-plate (~600 step episodes) 40 is a good default. You can go lower (e.g., 20) for more
   responsive closed-loop control at the cost of more frequent policy queries; you cannot go higher
   without retraining the base model. Whichever value you pick, **keep the modality config and the
   server YAML in sync** (see the caution above).

If you adjust any of these and the resulting checkpoint behaves badly at evaluation, the most
common culprits in order are: (i) too few or low-quality demonstrations, (ii) modality config /
``action_horizon`` mismatch between training and server YAML, (iii) recording with the wrong
embodiment.
