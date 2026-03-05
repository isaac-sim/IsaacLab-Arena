# NaVILA VLN Benchmark for IsaacLab Arena

**Vision-Language Navigation (VLN)** benchmark using the [NaVILA](https://github.com/AnjieCheng/NaVILA) VLM on the H1 humanoid robot in Matterport 3D indoor scenes.

## Architecture

```
 ISAAC SIM CONTAINER                       NAVILA SERVER CONTAINER
 ══════════════════                        ═══════════════════════

 ┌──────────────┐  ┌───────────────────┐
 │ Matterport   │  │ VLN-CE R2R Data   │
 │ Scene (.usd) │  │ (.json.gz)        │
 └──────┬───────┘  └───────┬───────────┘
        │                  │
        ▼                  ▼
 ┌──────────────┐  ┌───────────────────┐
 │ H1 Robot     │  │VlnR2rMatterport   │
 │ + Head Cam   │  │Task               │
 │ + Follow Cam │  │ scene_filter      │
 └──────┬───────┘  │ termination       │
        │          │ metrics (SPL...)  │
        │          └───────┬───────────┘
        │                  │
        ▼                  ▼
 ┌─────────────────────────────────┐
 │  VlnVlmLocomotionPolicy        │
 │                                 │
 │  ┌─ High-level ──────────────┐  │
 │  │  Send 1 RGB frame ────────┼──┼──► ZeroMQ ──►┐
 │  │  Receive vel_cmd ◄────────┼──┼──◄ ZeroMQ ◄──┤
 │  └────────────┬──────────────┘  │              │
 │               │                 │              │
 │  ┌────────────▼──────────────┐  │              │
 │  │  Low-level: RSL-RL        │  │              │
 │  │  Inject vel_cmd → obs     │  │              │
 │  │  NN forward → joint act   │  │              │
 │  └────────────┬──────────────┘  │              │
 └───────────────┼─────────────────┘              │
                 │                                │
                 ▼                                │
 ┌─────────────────────────────┐    ┌─────────────▼────────────┐
 │  PhysX GPU Simulation       │    │  NaVilaServerPolicy      │
 │  joint act → robot moves    │    │                          │
 │  → new camera RGB           │    │  1. Receive 1 RGB frame  │
 └─────────────────────────────┘    │  2. Append to history    │
                                    │  3. Uniform sample 8     │
                                    │     from full history    │
                                    │  4. VLM inference        │
                                    │  5. Parse → vel_cmd      │
                                    │  6. Return vel_cmd + dur │
                                    └──────────────────────────┘
```

**Key**: Each `get_action()` call sends **1 RGB frame** to the server. The server
accumulates all frames into an episode history and **uniformly samples 8** for
VLM inference. This gives the VLM global temporal context to determine when
to output "stop".

## Quick Start

### 1. Start the NaVILA VLM Server

```bash
# Option A: Use the launch script (rebuilds Docker if needed)
bash docker/run_vln_server.sh -m ~/models/navila/navila-llama3-8b-8f --port 5555

# Option B: Manual start with code mounted (no rebuild)
docker run --rm -d --gpus all --net host \
    --name vln_policy_server_container \
    -v ~/models:/models \
    -v ~/IsaacLab-Arena/isaaclab_arena_navila:/workspace/isaaclab_arena_navila \
    -v ~/IsaacLab-Arena/isaaclab_arena/remote_policy:/workspace/isaaclab_arena/remote_policy \
    vln_policy_server:latest \
    --host 0.0.0.0 --port 5555 --timeout_ms 15000 \
    --policy_type isaaclab_arena_navila.navila_server_policy.NaVilaServerPolicy \
    --model_path /models/navila/navila-llama3-8b-8f
```

Wait for: `listening on tcp://0.0.0.0:5555`

### 2. Run VLN Evaluation (Client)

```bash
# Enter Isaac Sim container
bash docker/run_docker.sh

# Inside the container (headless mode for batch evaluation):
/isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
    --enable_cameras --num_envs 1 \
    --policy_type isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy \
    --remote_host localhost --remote_port 5555 \
    --ll_checkpoint_path isaaclab_arena/policy/vln/pretrained/h1_navila_locomotion.pt \
    --ll_agent_cfg isaaclab_arena/policy/vln/pretrained/h1_navila_agent.yaml \
    --num_episodes 10 \
    h1_vln_matterport \
    --usd_path /datasets/VLN-CE-Isaac/matterport_usd/zsNo4HB9uLZ/zsNo4HB9uLZ.usd \
    --r2r_dataset_path /datasets/VLN-CE-Isaac/vln_ce_isaac_v1.json.gz
```

Add `--headless` for batch evaluation without GUI. Omit it to see the viewport.

## CLI Parameters

### Client Parameters (before `h1_vln_matterport`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_envs` | 1 | **Must be 1.** Multi-env not yet supported. |
| `--policy_type` | (required) | `isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy` |
| `--remote_host` | localhost | VLM server hostname |
| `--remote_port` | 5555 | VLM server port |
| `--ll_checkpoint_path` | (required) | RSL-RL locomotion checkpoint (.pt) |
| `--ll_agent_cfg` | (required) | RSL-RL agent config (.yaml) |
| `--warmup_steps` | 200 | Low-level policy warmup steps |
| `--debug_vln` | false | Enable VLM query logging and frame saving |
| `--num_episodes` | (required) | Number of episodes to evaluate |

### Environment Parameters (after `h1_vln_matterport`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--usd_path` | (required) | Path to Matterport USD scene file |
| `--r2r_dataset_path` | (required) | Path to VLN-CE R2R dataset (.json.gz) |
| `--episode_start` | None | Start index within filtered episodes |
| `--episode_end` | None | End index (exclusive) within filtered episodes |
| `--episode_length_s` | 60 | Max episode duration in seconds (3000 steps) |
| `--success_radius` | 3.0 | Success distance threshold (meters) |
| `--camera_resolution` | 512 | Camera resolution (512 or 1024 for demo) |
| `--no_follow_camera` | false | Disable third-person follow camera |

### Server Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | (required) | Path to NaVILA/LLaVA model checkpoint |
| `--port` | 5555 | Server listening port |
| `--num_video_frames` | 8 | Number of frames sent to VLM per query |
| `--conv_mode` | llama_3 | LLaVA conversation template |

## Available Scenes

The VLN-CE-Isaac dataset contains 1077 episodes across 11 Matterport scenes. Each run loads one scene:

| Scene ID | Episodes | USD Path |
|----------|----------|----------|
| zsNo4HB9uLZ | 270 | `matterport_usd/zsNo4HB9uLZ/zsNo4HB9uLZ.usd` |
| QUCTc6BB5sX | 240 | `matterport_usd/QUCTc6BB5sX/QUCTc6BB5sX.usd` |
| 2azQ1b91cZZ | 204 | `matterport_usd/2azQ1b91cZZ/2azQ1b91cZZ.usd` |
| TbHJrupSAjP | 108 | `matterport_usd/TbHJrupSAjP/TbHJrupSAjP.usd` |
| X7HyMhZNoso | 69 | `matterport_usd/X7HyMhZNoso/X7HyMhZNoso.usd` |

Episode selection is automatic: the scene ID from `--usd_path` filters episodes.

To evaluate all scenes, run each scene sequentially:

```bash
for scene in zsNo4HB9uLZ QUCTc6BB5sX 2azQ1b91cZZ TbHJrupSAjP X7HyMhZNoso; do
    /isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
        ... \
        --usd_path /datasets/VLN-CE-Isaac/matterport_usd/$scene/$scene.usd \
        ...
done
```

## Low-Level Locomotion Policy

The VLN pipeline uses a two-level architecture. The **high-level** VLM outputs velocity commands `[vx, vy, yaw_rate]`, and the **low-level** RSL-RL locomotion policy converts them into joint-position actions. The low-level policy is robot-specific — different robots need different checkpoints.

### Using the Pre-trained H1 Checkpoint

A pre-trained H1 checkpoint is included in the repo for quick start:

```
isaaclab_arena/policy/vln/pretrained/
├── h1_navila_locomotion.pt     # RSL-RL PPO checkpoint (4.7MB)
├── h1_navila_agent.yaml        # PPO agent config
└── h1_navila_env.yaml          # Training env config (reference)
```

This checkpoint was trained for the Unitree H1 humanoid and is compatible with the `H1VlnEmbodiment`. For early access users, this checkpoint will be available in the repo. For production use, consider training your own checkpoint.

### Training Your Own Low-Level Policy

The low-level policy is trained using IsaacLab's standard RL pipeline. The example below is for the H1 robot, but you can train for any robot that IsaacLab supports:

```bash
# Example: Train H1 velocity tracking policy
cd submodules/IsaacLab
python -m isaaclab_rl.rsl_rl.train \
    --task Isaac-Velocity-Flat-H1-v0 \
    --headless --num_envs 4096

# For other robots (e.g., Go2):
# python -m isaaclab_rl.rsl_rl.train --task Isaac-Velocity-Flat-Go2-v0 ...
```

Key requirements for the trained policy to work with the VLN pipeline:

- **Observation must include `velocity_commands`**: The VLM output is injected into the observation vector at a specific index range (e.g., indices 9:12 for H1)
- **The index range must match** `vel_cmd_obs_indices` in `VlnVlmLocomotionPolicy`
- **Network architecture and action scale** must match the agent config YAML

The included H1 checkpoint uses: network `[512, 256, 128]`, observation 69-dim, velocity commands at indices 9:12, action scale 0.5.

## Camera Configuration

### Default Camera Positions

| Camera | Position (pelvis frame) | FOV | Purpose |
|--------|------------------------|-----|---------|
| Head camera | (0.1, 0.0, 0.5) | 54° | VLM input (first-person view) |
| Follow camera | (-1.0, 0.0, 0.57) | 100° | Visualization (third-person) |

### Adjusting Camera Position

Camera positions are not exposed as CLI parameters. To modify them, edit the source code:

```python
# isaaclab_arena/embodiments/h1/h1.py

_DEFAULT_H1_CAMERA_OFFSET = Pose(
    position_xyz=(0.1, 0.0, 0.5),         # (forward, left, up) from pelvis
    rotation_wxyz=(-0.5, 0.5, -0.5, 0.5), # facing forward in ROS convention
)

_DEFAULT_H1_FOLLOW_CAMERA_OFFSET = Pose(
    position_xyz=(-1.0, 0.0, 0.57),       # 1m behind, 0.57m above pelvis
    rotation_wxyz=(-0.5, 0.5, -0.5, 0.5),
)
```

Or pass custom offsets programmatically:

```python
from isaaclab_arena.utils.pose import Pose

embodiment = H1VlnEmbodiment(
    camera_offset=Pose(position_xyz=(0.2, 0.0, 0.6), rotation_wxyz=(-0.5, 0.5, -0.5, 0.5)),
    follow_camera_offset=Pose(position_xyz=(-2.0, 0.0, 1.0), rotation_wxyz=(-0.5, 0.5, -0.5, 0.5)),
)
```

## Integrating a New VLM

To replace NaVILA with another VLM (e.g., GR00T, GPT-4V):

1. Create a new server policy in `isaaclab_arena_<your_model>/`:

```python
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy

class YourVlmServerPolicy(ServerSidePolicy):
    def get_action(self, observation, options=None):
        # Extract RGB from observation
        # Run your VLM inference
        # Return {"action": [vx, vy, yaw_rate], "duration": seconds}
        pass
```

2. Launch with `--policy_type your_package.YourVlmServerPolicy`
3. The client side (`VlnVlmLocomotionPolicy`) remains unchanged.

## Using a Different Dataset

The default dataset is VLN-CE R2R (`--r2r_dataset_path`). You can use a different dataset by either:

### Option A: Same Format, Different Data

If your dataset follows the same JSON format, just pass a different file:

```bash
--r2r_dataset_path /path/to/your_custom_dataset.json.gz
```

Required JSON format (gzipped):

```json
{"episodes": [
    {
        "episode_id": 1,
        "scene_id": "mp3d/SCENE_ID/SCENE_ID.glb",
        "start_position": [x, y, z],
        "start_rotation": [w, x, y, z],
        "instruction": {"instruction_text": "Walk to the kitchen..."},
        "reference_path": [[x1,y1,z1], [x2,y2,z2], ...]
    }
]}
```

### Option B: Different Format (New Task)

If your dataset has a different structure (e.g., RxR, REVERIE):

1. Create a new task class in `isaaclab_arena/tasks/` using `VlnR2rMatterportTask` as a template
2. Register a new environment in `isaaclab_arena_environments/`
3. The VLM server and client policy remain unchanged

## Metrics

| Metric | Description |
|--------|-------------|
| **Success** | Fraction of episodes where final distance-to-goal < `success_radius` |
| **SPL** | Success weighted by path efficiency: `S × (shortest_path / actual_path)` |
| **Distance-to-Goal** | Geodesic distance from final position to goal (XY only) |
| **Path Length** | Total distance traversed (XY only) |

Metrics use XY (horizontal) distance because the robot pelvis height (~0.9m) differs from dataset waypoint heights (~0.17m floor level). For future multi-floor support, 3D geodesic with height offset correction will be needed.

## Known Limitations

- **`num_envs` must be 1**: The VLM server tracks a single instruction and image history. Multi-env requires per-env instruction tracking, which is planned for a future release.
- **Scene switching requires restart**: Matterport USD is loaded at initialization. To evaluate different scenes, run each as a separate process.
- **Collision**: Uses an invisible ground plane at z=0 instead of Matterport mesh collision (GPU physics limitation with referenced USD). The robot may fall at some positions.
- **VLM stop accuracy**: The VLM determines task completion from visual context. Stop accuracy depends on the VLM model quality and the full image history sampling.

---

# NaVILA VLN Benchmark（中文）

基于 [NaVILA](https://github.com/AnjieCheng/NaVILA) 视觉语言模型的 **视觉语言导航（VLN）** 基准测试，使用 H1 人形机器人在 Matterport 3D 室内场景中评估。

## 架构概述

系统分为两个容器：

**Isaac Sim 容器（客户端）**：加载 Matterport 场景 + H1 机器人，运行物理仿真，采集相机图像，执行低层运控策略。

**NaVILA 容器（服务端）**：接收 RGB 图像，运行 VLM 推理，输出速度命令（前进/转向/停止）。

两层控制：
1. **高层**：VLM 根据 8 帧均匀采样的图像历史 + 导航指令，生成速度命令 `[vx, vy, yaw_rate]`
2. **低层**：RSL-RL 运控策略将速度命令转换为 H1 的 19 个关节位置动作

关键特性：**VLM 使用完整 episode 图像历史的均匀采样**（不仅仅是最近几帧），使其能够判断任务是否完成并输出 "stop"。

## 快速开始

### 1. 启动 VLM 服务器

```bash
# 方式 A：使用启动脚本
bash docker/run_vln_server.sh -m ~/models/navila/navila-llama3-8b-8f --port 5555

# 方式 B：手动启动（挂载代码，无需重建镜像）
docker run --rm -d --gpus all --net host \
    --name vln_policy_server_container \
    -v ~/models:/models \
    -v ~/IsaacLab-Arena/isaaclab_arena_navila:/workspace/isaaclab_arena_navila \
    -v ~/IsaacLab-Arena/isaaclab_arena/remote_policy:/workspace/isaaclab_arena/remote_policy \
    vln_policy_server:latest \
    --host 0.0.0.0 --port 5555 --timeout_ms 15000 \
    --policy_type isaaclab_arena_navila.navila_server_policy.NaVilaServerPolicy \
    --model_path /models/navila/navila-llama3-8b-8f
```

### 2. 运行评估

```bash
# 进入 Isaac Sim 容器
bash docker/run_docker.sh

# 容器内执行（不加 --headless 可以看到 GUI 窗口）
/isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
    --enable_cameras --num_envs 1 \
    --policy_type isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy \
    --remote_host localhost --remote_port 5555 \
    --ll_checkpoint_path isaaclab_arena/policy/vln/pretrained/h1_navila_locomotion.pt \
    --ll_agent_cfg isaaclab_arena/policy/vln/pretrained/h1_navila_agent.yaml \
    --num_episodes 10 \
    h1_vln_matterport \
    --usd_path /datasets/VLN-CE-Isaac/matterport_usd/zsNo4HB9uLZ/zsNo4HB9uLZ.usd \
    --r2r_dataset_path /datasets/VLN-CE-Isaac/vln_ce_isaac_v1.json.gz
```

添加 `--headless` 可在无 GUI 模式下批量评估。

## CLI 参数

### 客户端参数（`h1_vln_matterport` 之前）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_envs` | 1 | **必须为 1**，多环境暂不支持 |
| `--policy_type` | （必填） | `isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy` |
| `--remote_host` | localhost | VLM 服务器主机名 |
| `--remote_port` | 5555 | VLM 服务器端口 |
| `--ll_checkpoint_path` | （必填） | RSL-RL 运控 checkpoint (.pt) |
| `--ll_agent_cfg` | （必填） | RSL-RL agent 配置 (.yaml) |
| `--warmup_steps` | 200 | 低层策略预热步数 |
| `--debug_vln` | 关 | 开启 VLM 查询日志和帧保存 |
| `--num_episodes` | （必填） | 评估的 episode 数量 |
| `--headless` | 关 | 无 GUI 模式（批量评估时使用） |

### 环境参数（`h1_vln_matterport` 之后）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--usd_path` | （必填） | Matterport USD 场景文件路径 |
| `--r2r_dataset_path` | （必填） | VLN-CE R2R 数据集路径 (.json.gz) |
| `--episode_start` | 无 | 场景过滤后的起始 episode 索引 |
| `--episode_end` | 无 | 结束索引（不包含） |
| `--episode_length_s` | 60 | 最大 episode 时长（秒，3000 步） |
| `--success_radius` | 3.0 | 成功判定距离（米） |
| `--camera_resolution` | 512 | 相机分辨率（1024 用于高清 demo） |
| `--no_follow_camera` | 关 | 关闭第三人称跟随相机 |

### 服务端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | （必填） | NaVILA/LLaVA 模型路径 |
| `--port` | 5555 | 服务器监听端口 |
| `--num_video_frames` | 8 | 每次 VLM 查询发送的帧数 |
| `--conv_mode` | llama_3 | LLaVA 对话模板 |

## 可用场景

VLN-CE-Isaac 数据集包含 11 个 Matterport 场景共 1077 个 episode。每次运行加载一个场景：

| 场景 ID | Episode 数 | USD 路径 |
|---------|-----------|----------|
| zsNo4HB9uLZ | 270 | `matterport_usd/zsNo4HB9uLZ/zsNo4HB9uLZ.usd` |
| QUCTc6BB5sX | 240 | `matterport_usd/QUCTc6BB5sX/QUCTc6BB5sX.usd` |
| 2azQ1b91cZZ | 204 | `matterport_usd/2azQ1b91cZZ/2azQ1b91cZZ.usd` |
| TbHJrupSAjP | 108 | `matterport_usd/TbHJrupSAjP/TbHJrupSAjP.usd` |
| X7HyMhZNoso | 69 | `matterport_usd/X7HyMhZNoso/X7HyMhZNoso.usd` |

Episode 自动过滤：从 `--usd_path` 提取场景 ID，只选匹配的 episode。

切换场景只需更换 `--usd_path`：

```bash
# 场景 A（270 个 episode）
--usd_path .../zsNo4HB9uLZ/zsNo4HB9uLZ.usd

# 场景 B（240 个 episode）
--usd_path .../QUCTc6BB5sX/QUCTc6BB5sX.usd
```

评估所有场景需要依次运行每个场景（每个场景一个进程）：

```bash
for scene in zsNo4HB9uLZ QUCTc6BB5sX 2azQ1b91cZZ TbHJrupSAjP X7HyMhZNoso; do
    /isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
        ... \
        --usd_path /datasets/VLN-CE-Isaac/matterport_usd/$scene/$scene.usd \
        ...
done
```

## 低层运控策略

VLN 使用两层架构：**高层** VLM 输出速度命令 `[vx, vy, yaw_rate]`，**低层** RSL-RL 运控策略将其转换为关节动作。低层策略与具体机器人绑定——不同机器人需要不同的 checkpoint。

### 使用预训练 H1 模型

仓库包含预训练的 H1 运控 checkpoint，可直接使用：

```
isaaclab_arena/policy/vln/pretrained/
├── h1_navila_locomotion.pt     # RSL-RL PPO checkpoint (4.7MB)
├── h1_navila_agent.yaml        # PPO agent 配置
└── h1_navila_env.yaml          # 训练环境配置（参考）
```

此 checkpoint 供 early access 用户使用，正式合入时可能移除。

### 自行训练

低层策略使用 IsaacLab 的标准 RL 训练流程。以下以 H1 为例，也可以训练其他机器人：

```bash
# H1 速度跟踪策略训练
cd submodules/IsaacLab
python -m isaaclab_rl.rsl_rl.train \
    --task Isaac-Velocity-Flat-H1-v0 \
    --headless --num_envs 4096

# 其他机器人（如 Go2）：
# python -m isaaclab_rl.rsl_rl.train --task Isaac-Velocity-Flat-Go2-v0 ...
```

关键要求：
- 观测必须包含 `velocity_commands`，且其在观测向量中的索引范围必须与 `VlnVlmLocomotionPolicy` 的 `vel_cmd_obs_indices` 参数匹配
- 网络结构和动作缩放必须与 agent config YAML 一致

当前 H1 checkpoint：网络 `[512, 256, 128]`，观测 69 维，速度命令在 indices 9:12，动作缩放 0.5。

## 相机配置

相机位置目前没有 CLI 参数。如需调整，修改源码：

```python
# isaaclab_arena/embodiments/h1/h1.py
_DEFAULT_H1_CAMERA_OFFSET = Pose(
    position_xyz=(0.1, 0.0, 0.5),  # (前, 左, 上) 相对于骨盆
)
```

## 集成新的 VLM

在 `isaaclab_arena_<你的模型>/` 下创建 `ServerSidePolicy` 子类，实现 `get_action()` 返回 `{"action": [vx, vy, yaw_rate], "duration": seconds}`。客户端代码不需要改动。详见英文版的 "Integrating a New VLM" 章节。

## 使用新数据集

默认使用 VLN-CE R2R 数据集（`--r2r_dataset_path`）。如果你的数据集格式相同，直接替换路径即可：

```bash
--r2r_dataset_path /path/to/your_custom_dataset.json.gz
```

如果数据集格式不同（如 RxR、REVERIE），需要创建新的 Task 类（参考 `VlnR2rMatterportTask`）和新的 Environment 注册。VLM 服务端和客户端策略不需要改动。

## 评估指标

| 指标 | 说明 |
|------|------|
| **Success** | 最终距离 < `success_radius` 的 episode 比例 |
| **SPL** | 成功率加权路径效率：`S × (最短路径 / 实际路径)` |
| **Distance-to-Goal** | 最终位置到目标的测地距离（仅 XY 水平面） |
| **Path Length** | 累计行走距离（仅 XY 水平面） |

指标使用 XY（水平）距离，因为机器人骨盆高度（~0.9m）与数据集路标高度（~0.17m 地面）不同。未来多层楼支持需要带高度偏移校正的 3D 测地距离。

## 已知限制

- **`num_envs` 必须为 1**：VLM 服务器跟踪单一指令和图像历史。多环境支持需要按环境跟踪，计划未来实现。
- **场景切换需重启**：Matterport USD 在初始化时加载。评估多个场景需要依次运行。
- **碰撞**：使用 z=0 的不可见地面平面（GPU 物理不支持 referenced USD mesh collision）。部分位置机器人可能倒下。
- **VLM 停止精度**：依赖 VLM 从视觉上下文判断任务完成。精度取决于 VLM 模型质量和图像历史采样。
