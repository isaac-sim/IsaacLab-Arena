# lerobot+smolVLA配置流程

### 先安装好 IsaacLab-Arena

官方教程使用的是旧版本的 IsaacLab-Arena，这里使用当前版本的
IsaacLab-Arena，并使用项目根目录下的 `.venv`。


### 用IsaacLab-Arena评估
```bash
.venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
    --headless \
    --enable_cameras \
    --device cuda:0 \
    --num_envs 1 \
    --policy_type zero_action \
    --num_episodes 1 \
    --record_viewport_video \
    --record_camera_video \
    --output_base_dir outputs/arena_open_microwave_check \
    gr1_open_microwave \
    --embodiment gr1_pink \
    --object mustard_bottle
````


### 安装 LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
.venv/bin/pip install -e "./lerobot[evaluation,datasets,smol_vla]"
```

### 运行 evaluation

在 IsaacLab-Arena 项目根目录下执行：

```bash
.venv/bin/lerobot-eval \
    --policy.path=nvidia/smolvla-arena-gr1-microwave \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=gr1_open_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1
```

新版环境注册名是 `gr1_open_microwave`。LeRobot 会自动录制前 10 个 evaluation episode，
视频画面来自策略使用的 `robot_pov_cam_rgb` 相机观测，输出位于本次评估目录的
`videos/` 子目录。

`gr1_open_microwave` 在当前 IsaacLab-Arena 中的单个 episode 时长固定为 5 秒。
`--env.video_length` 和 `--env.video_interval` 是旧 Arena 录像器的参数，不会改变 LeRobot evaluation
视频的时长，因此新启动命令不再传入它们。
