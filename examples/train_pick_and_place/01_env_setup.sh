#!/usr/bin/env bash

# 数据集和模型的保存位置
export DATASET_DIR="$HOME/IsaacLab-Arena/datasets/isaaclab_arena/static_apple_tutorial"
export MODELS_DIR="$HOME/IsaacLab-Arena/models/isaaclab_arena/static_apple_tutorial"

# Isaac-GR00T 的保存位置
export ISAAC_GR00T_DIR="$HOME/IsaacLab-Arena/submodules/Isaac-GR00T-N1.7"

# 下载 Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git "$ISAAC_GR00T_DIR"

### ----------------------------------------------------------------------------
# 下载训练数据集
# hf download \
#    nvidia/Arena-G1-Static-PickNPlace-Task \
#    arena_g1_static_apple_dataset_recorded_200_demos.hdf5 \
#    --repo-type dataset \
#    --local-dir $DATASET_DIR
# mv "$DATASET_DIR/arena_g1_static_apple_dataset_recorded_200_demos.hdf5" \
#    "$DATASET_DIR/arena_g1_static_apple_dataset_recorded.hdf5"

# 数据集转成lerobot格式
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/g1_static_apple_config.yaml

python tools/inspect_parquet.py "$DATASET_DIR/arena_g1_static_apple_dataset_recorded_200_demos/lerobot/data/chunk-000/episode_000000.parquet"

### ----------------------------------------------------------------------------
# 开始训练
cd $ISAAC_GR00T_DIR
# 运行训练脚本，训练全部参数，显存占用大，官方文档说，要48G*8显存
uv run python -m torch.distributed.run --nproc_per_node=1 --standalone \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $DATASET_DIR/arena_g1_static_apple_dataset_recorded_200_demos/lerobot \
  --output-dir $MODELS_DIR/static_apple_n17_finetune \
  --modality-config-path $HOME/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py \
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

# 训练脚本2,只训练最后的projector
# 32G显存，只能训练projector。要训练DiT和Projector,需要48G显存
uv run python -m torch.distributed.run --nproc_per_node=1 --standalone \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $DATASET_DIR/arena_g1_static_apple_dataset_recorded_200_demos/lerobot \
  --output-dir $MODELS_DIR/static_apple_n17_finetune_diffusion_only \
  --modality-config-path $HOME/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py \
  --embodiment-tag new_embodiment \
  --global-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-steps 2000 \
  --num-gpus 1 \
  --save-steps 500 \
  --save-total-limit 5 \
  --shard-size 64 \
  --no-tune-llm \
  --no-tune-visual \
  --no-tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers 1 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
