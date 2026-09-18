1. 配置环境
#!/usr/bin/env bash

# 数据集和模型的保存位置
export DATASET_DIR="~/IsaacLab-Arena/datasets/isaaclab_arena/static_apple_tutorial"
export MODELS_DIR="~/IsaacLab-Arena/models/isaaclab_arena/static_apple_tutorial"

# Isaac-GR00T 的保存位置
export ISAAC_GR00T_DIR="~/IsaacLab-Arena/submodules/Isaac-GR00T-N1.7"

# 下载 Isaac-GR00T
# git clone https://github.com/NVIDIA/Isaac-GR00T.git "$ISAAC_GR00T_DIR"

# 安装虚拟环境
因为不同版本的虚拟环境依赖不同，需要给N1.7在自己的目录下安装独立的虚拟环境
cd $ISAAC_GR00T_DIR
uv sync
注意，针对不同的平台，还需要手工修改uv配置文件的内容。
例如在5090上，需要改成torch==2.8.1，以及flash-attn版本


### ----------------------------------------------------------------------------
# 下载训练数据集
hf download \
   nvidia/Arena-G1-Static-PickNPlace-Task \
   arena_g1_static_apple_dataset_recorded_200_demos.hdf5 \
   --repo-type dataset \
   --local-dir $DATASET_DIR
mv "$DATASET_DIR/arena_g1_static_apple_dataset_recorded_200_demos.hdf5" \
   "$DATASET_DIR/arena_g1_static_apple_dataset_recorded.hdf5"

# 数据集转成lerobot格式
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/g1_static_apple_config.yaml

python tools/inspect_parquet.py "$DATASET_DIR/arena_g1_static_apple_dataset_recorded_200_demos/lerobot/data/chunk-000/episode_000000.parquet"


### ----------------------------------------------------------------------------
# 开始训练
uv run \
python -m torch.distributed.run --nproc_per_node=1 --standalone \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $DATASET_DIR/arena_g1_static_apple_dataset_recorded/lerobot \
  --output-dir $MODELS_DIR/static_apple_n17_finetune \
  --modality-config-path /path/to/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py \
  --embodiment-tag NEW_EMBODIMENT \
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
