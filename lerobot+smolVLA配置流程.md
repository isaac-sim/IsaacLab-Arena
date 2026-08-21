# IsaacLab-Arena 环境构建与 LeRobot/SmolVLA 接入指南

本文以当前仓库代码为准，说明两件事：

1. IsaacLab-Arena 如何把场景、机器人和任务组合成可运行的 Gym 环境；
2. LeRobot 如何通过 EnvHub 调用该环境，并把 Arena 观测转换成 SmolVLA 的输入。

示例使用 `gr1_open_microwave`、`gr1_pink` 和 `mustard_bottle`。所有命令均在仓库根目录执行，并使用项目的 `.venv`，不需要 Docker。

## 1. 总体架构

Arena 把环境拆成三个可独立替换的部分：

- `Scene`：背景、灯光、微波炉、桌面物体等资产；
- `Embodiment`：机器人、控制方式、相机和本体观测；
- `Task`：任务描述、成功/失败条件、episode 时长和指标。

构建流程如下：

```text
EnvironmentCfg
      │
      ▼
ArenaEnvironmentFactory.build(cfg)
      │
      ▼
IsaacLabArenaEnvironment
  ├── Scene
  ├── Embodiment
  └── Task
      │
      ▼
ArenaEnvBuilder.compose_manager_cfg()
      │ 合并 scene / observations / actions / events / terminations / metrics
      ▼
IsaacLabArenaManagerBasedRLEnvCfg
      │
      ▼
gym.register() → gym.make() → Gym 包装后的 Arena env
```

`ArenaEnvBuilder.make_registered()` 返回的是 Gym 包装后的环境。访问 Isaac Lab 属性时必须使用 `env.unwrapped`：

```python
env.step(actions)
device = env.unwrapped.device
step_dt = env.unwrapped.step_dt
cfg = env.unwrapped.cfg
```

## 2. 关键源码位置

| 功能 | 源码 |
| --- | --- |
| 环境描述对象 | `isaaclab_arena/environments/isaaclab_arena_environment.py` |
| 环境 typed config/factory 接口 | `isaaclab_arena/environments/arena_environment_factory.py` |
| 环境注册表 | `isaaclab_arena/assets/registries.py` |
| 注册装饰器 | `isaaclab_arena/assets/register.py` |
| Manager 配置合成及 Gym 注册 | `isaaclab_arena/environments/arena_env_builder.py` |
| Builder typed config | `isaaclab_arena/environments/arena_env_builder_cfg.py` |
| 原生策略运行入口 | `isaaclab_arena/evaluation/policy_runner.py` |
| 微波炉环境定义 | `isaaclab_arena_environments/gr1_open_microwave_environment.py` |
| LeRobot Arena 配置 | `lerobot/src/lerobot/envs/configs.py` |
| LeRobot EnvHub 调度 | `lerobot/src/lerobot/envs/factory.py` |
| 新旧 Arena API 适配 | `lerobot/src/lerobot/envs/isaaclab_arena.py` |
| Arena 观测转 LeRobot 特征 | `lerobot/src/lerobot/processor/env_processor.py` |
| LeRobot evaluation | `lerobot/src/lerobot/scripts/lerobot_eval.py` |

## 3. 先用 Arena 原生脚本验证环境

在接入 LeRobot 前，应先确认 Arena 环境、物理、相机和录像都正常。这一步不加载 SmolVLA，可以把环境问题和模型问题分开。

### 3.1 随机动作运行并录制视频

```bash
.venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
    --headless \
    --enable_cameras \
    --device cuda:0 \
    --num_envs 1 \
    --policy_type random_action \
    --num_episodes 1 \
    --record_viewport_video \
    --record_camera_video \
    --output_base_dir outputs/arena_open_microwave_check \
    gr1_open_microwave \
    --embodiment gr1_pink \
    --object mustard_bottle
```

也可以把 `random_action` 换成 `zero_action`。两者用途不同：

- `zero_action`：最适合检查环境能否稳定启动、相机是否持续出图；
- `random_action`：生成小幅且有界的随机动作，并尽量保持 GR1 Pink 的末端四元数有效，适合检查动作链路。

Arena 会分别输出两类视频：

- viewport video：第三人称视口，通常能看到机器人整体；
- camera video：来自 `obs["camera_obs"]`，即策略实际看到的机器人相机。

`gr1_open_microwave` 的任务定义把 `episode_length_s` 固定为 5 秒，并通过 `set_control_rate_50hz` 使用 50 Hz 控制频率，因此一个完整 episode 通常约 250 个 step。

### 3.2 显示 Kit 窗口

当前 Isaac Lab 默认不启用任何 visualizer。需要交互式窗口时，必须显式传入
`--viz kit`；仅仅删除 `--headless` 不会打开窗口。`--headless` 与 `--viz kit`
不能同时使用，因为 `--headless` 会覆盖并禁用所有 visualizer。

```bash
.venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
    --viz kit \
    --enable_cameras \
    --device cuda:0 \
    --num_envs 1 \
    --policy_type zero_action \
    --num_episodes 1 \
    gr1_open_microwave \
    --embodiment gr1_pink \
    --object mustard_bottle
```

## 4. Arena 如何构建 `gr1_open_microwave`

### 4.1 typed environment config

该环境的配置类型是 `Gr1OpenMicrowaveEnvironmentCfg`：

```python
@dataclass
class Gr1OpenMicrowaveEnvironmentCfg(ArenaEnvironmentCfg):
    object: str | None = None
    teleop_device: str | None = None
    embodiment: str = "gr1_pink"
```

它继承的 `ArenaEnvironmentCfg` 还提供 `enable_cameras`。配置中的名称不是 Python 类名，而是注册表中的 key，例如：

- 环境：`gr1_open_microwave`；
- 机器人：`gr1_pink`；
- 可选物体：`mustard_bottle`。

### 4.2 factory 构造环境描述

`Gr1OpenMicrowaveEnvironment.build()` 的主要工作是：

1. 从 `AssetRegistry` 获取厨房背景和微波炉；
2. 根据 `cfg.embodiment` 创建 GR1，并把 `cfg.enable_cameras` 传给机器人；
3. 设置机器人、微波炉和可选物体的初始位姿；
4. 用这些资产创建 `Scene`；
5. 创建 `OpenDoorTask`，设定成功阈值和 5 秒 episode；
6. 返回 `IsaacLabArenaEnvironment`。

这里返回的仍然是环境描述，并未创建物理仿真。

### 4.3 Builder 编译并创建 Gym 环境

`ArenaEnvBuilder` 接收环境描述和 `ArenaEnvBuilderCfg`。其 `compose_manager_cfg()` 会把 Scene、Embodiment 和 Task 提供的配置合并为完整的 Isaac Lab Manager 配置，包括：

- scene；
- observations；
- actions；
- reset events；
- terminations；
- rewards/curriculum/commands；
- metrics 和 episode recorders；
- task language instruction。

随后 `make_registered()` 完成：

1. `gym.register()`；
2. `parse_env_cfg()`，应用 `device`、`num_envs` 和 Fabric 配置；
3. `gym.make()`；
4. 返回 Gym 包装环境。

## 5. 使用 Python API 构建已注册环境

Isaac Sim/Isaac Lab 相关模块必须在 `AppLauncher` 启动后导入。下面是当前 typed API 的最小示例：

```python
import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--enable_cameras", "--device", "cuda:0"])
app_launcher = AppLauncher(args)

env = None
try:
    # 必须放在 AppLauncher 之后。
    import torch
    import isaaclab_arena_environments  # 导入环境模块，触发 @register_environment

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name("gr1_open_microwave")
    env_cfg_type = registry.get_environment_cfg_type(factory_type)

    environment_cfg = env_cfg_type(
        enable_cameras=True,
        embodiment="gr1_pink",
        object="mustard_bottle",
    )
    arena_environment = factory_type().build(environment_cfg)

    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=1,
        device="cuda:0",
        seed=42,
    )
    env = ArenaEnvBuilder(arena_environment, builder_cfg).make_registered(
        render_mode="rgb_array"
    )

    observation, info = env.reset()
    print(observation.keys())
    print(observation["policy"].keys())
    print(observation["camera_obs"].keys())

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    observation, reward, terminated, truncated, info = env.step(actions)
finally:
    if env is not None:
        env.close()
    app_launcher.app.close()
```

`EnvironmentRegistry` 保存的是 factory 类型及其 config 类型，不应再导入旧路径 `isaaclab_arena.examples.example_environments`。当前注册名是 `gr1_open_microwave`，旧教程中的 `gr1_microwave` 仅作为兼容别名使用。

## 6. 自定义一个 Arena 环境

推荐沿用 typed factory 模式：

```python
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)


@dataclass
class MyEnvironmentCfg(ArenaEnvironmentCfg):
    embodiment: str = "gr1_pink"
    object: str | None = None


@register_environment
class MyEnvironment(ArenaEnvironmentFactory[MyEnvironmentCfg]):
    name = "my_environment"

    def build(self, cfg: MyEnvironmentCfg):
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene

        background = self.asset_registry.get_asset_by_name("kitchen")()
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras
        )
        assets = [background]
        if cfg.object is not None:
            assets.append(self.asset_registry.get_asset_by_name(cfg.object)())

        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=Scene(assets=assets),
            embodiment=embodiment,
            task=None,
        )
```

实际任务通常还需要设置资产位姿、空间关系和 `TaskBase`。可以直接参考 `gr1_open_microwave_environment.py`，因为它展示了完整的 Scene、Embodiment、Task 和控制频率回调。

## 7. LeRobot 如何调用 Arena

### 7.1 安装

如果 `lerobot/` 已存在，只需安装为 editable package：

```bash
.venv/bin/pip install -e "./lerobot[evaluation,datasets,smol_vla]"
```

### 7.2 当前调用链

```text
lerobot-eval CLI
  │
  ├── IsaaclabArenaEnv：描述环境参数及 observation/action feature
  │
  ├── make_env()
  │     ├── 下载 nvidia/isaaclab-arena-envs/env.py
  │     ├── patch_isaaclab_arena_hub_module()
  │     └── 用当前 Arena typed API 替换旧 _create_isaaclab_env()
  │            ├── AppLauncher
  │            ├── EnvironmentRegistry
  │            ├── factory.build(typed cfg)
  │            └── ArenaEnvBuilder.make_registered()
  │
  ├── IsaacLabEnvWrapper：适配 LeRobot 需要的向量 Gym 接口
  │
  ├── IsaaclabArenaProcessorStep
  │     ├── policy/* → observation.state
  │     └── camera_obs/* → observation.images.*
  │
  ├── rename_map：把环境图像 key 映射到模型训练使用的 key
  │
  └── SmolVLA → action → env.step(action)
```

这个适配层很重要：Hub 中的旧环境代码仍尝试导入 `isaaclab_arena.examples.example_environments`，而当前 Arena 已改为 typed factory + registry。LeRobot 在加载 Hub 模块后，只替换其 `_create_isaaclab_env()`，仍复用 Hub 提供的参数校验和 `IsaacLabEnvWrapper`。

### 7.3 Arena 观测到 SmolVLA 输入的映射

Arena 的原始观测是嵌套字典：

```text
obs["policy"]["robot_joint_pos"]
obs["camera_obs"]["robot_pov_cam_rgb"]
```

`IsaaclabArenaProcessorStep` 转换后为：

| Arena 原始观测 | LeRobot 特征 |
| --- | --- |
| `policy.robot_joint_pos` | `observation.state` |
| `camera_obs.robot_pov_cam_rgb` | `observation.images.robot_pov_cam_rgb` |

SmolVLA checkpoint 训练时使用的图像 key 是 `observation.images.robot_pov_cam`，所以还需要：

```bash
--rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}'
```

状态和动作维度由 Hub 的 `validate_config()` 在创建环境时检查。当前 GR1 配置为：

- `state_keys=robot_joint_pos`；
- `state_dim=54`；
- `action_dim=36`；
- `camera_keys=robot_pov_cam_rgb`。

## 8. SmolVLA evaluation 启动命令

在当前单 GPU 配置上，推荐让 Isaac Sim 使用 GPU、SmolVLA 使用 CPU：

```bash
.venv/bin/lerobot-eval \
    --policy.path=nvidia/smolvla-arena-gr1-microwave \
    --policy.device=cpu \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --env.environment=gr1_open_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.device=cuda:0 \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --trust_remote_code=true \
    --eval.batch_size=1 \
    --eval.n_episodes=1
```

注意不要同时写两次 `--policy.device`。CLI 中后一个值会覆盖前一个值，例如先写 CPU、后写 CUDA，最终仍会在 CUDA 上推理。

### 为什么策略使用 CPU

在本机单卡测试中：

- Arena 原生录像正常；
- LeRobot 使用零动作或直接动作时相机正常；
- SmolVLA 使用 CPU 推理时所有相机帧正常；
- SmolVLA 与 Isaac RTX 共用 `cuda:0` 时，首帧后相机可能变成全零。

因此这里把设备分工为：

- `--env.device=cuda:0`：物理仿真和 RTX/TiledCamera 渲染；
- `--policy.device=cpu`：SmolVLA 推理。

如果机器有多张显卡，可测试把策略放到另一张 GPU，例如 `--policy.device=cuda:1`，不要与仿真共享 `cuda:0`。

## 9. LeRobot 视频输出和视角

默认 evaluation 会录制前 10 个 episode，视频位于：

```text
outputs/eval/<日期>/<时间>_isaaclab_arena_smolvla/videos/gr1_open_microwave_0/
```

当前适配层优先录制 `robot_pov_cam_rgb`，也就是策略实际使用的机器人第一人称相机。它与 Arena 的 viewport video 不同：

- 第一人称相机不保证能看到机器人全身；
- 机械手只有进入头部相机视野后才会出现；
- 如果需要观察完整机器人，应使用 Arena 原生 `--record_viewport_video`，或为 LeRobot 单独增加第三人称调试录像；
- 不应把第三人称图像替换成策略输入，否则会改变模型输入分布。

`--env.video`、`--env.video_length` 和 `--env.video_interval` 是旧 Arena 录像器参数，不控制当前 LeRobot evaluation 视频长度。LeRobot 根据环境终止信号截取视频。

## 10. 直接从 LeRobot Python API 创建 Arena 环境

只创建环境而不运行 `lerobot-eval` 时，可以使用：

```python
import numpy as np

from lerobot.envs.configs import IsaaclabArenaEnv
from lerobot.envs.factory import make_env


cfg = IsaaclabArenaEnv(
    hub_path="nvidia/isaaclab-arena-envs",
    environment="gr1_open_microwave",
    embodiment="gr1_pink",
    object="mustard_bottle",
    device="cuda:0",
    headless=True,
    enable_cameras=True,
    state_keys="robot_joint_pos",
    camera_keys="robot_pov_cam_rgb",
)

envs = make_env(
    cfg,
    n_envs=1,
    use_async_envs=False,
    trust_remote_code=True,
)
env = envs["gr1_open_microwave"][0]

try:
    observation, info = env.reset()
    print(observation["policy"]["robot_joint_pos"].shape)
    print(observation["camera_obs"]["robot_pov_cam_rgb"].shape)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

`make_env()` 返回结构是：

```text
{suite_name: {task_id: vector_env}}
```

因此本例必须通过 `envs["gr1_open_microwave"][0]` 取得环境。

## 11. 常见问题

### `ModuleNotFoundError: isaaclab_arena.examples.example_environments`

原因：Hub 环境代码针对旧 Arena API，当前版本已使用 `EnvironmentRegistry` 和 typed factory。

检查：

- `lerobot/src/lerobot/envs/factory.py` 是否在 `env.type == "isaaclab_arena"` 时调用 `patch_isaaclab_arena_hub_module()`；
- `lerobot/src/lerobot/envs/isaaclab_arena.py` 是否存在；
- 使用当前名称 `gr1_open_microwave`。

### PyAV 报错：`float` 没有 `numerator`

原因：环境的 `render_fps` 可能是浮点数，而 PyAV 的 `add_stream(rate=...)` 需要有理数。

当前 `write_video()` 应把帧率转换为：

```python
Fraction(str(fps))
```

### 视频首帧正常，后续全黑

优先检查：

1. 命令中是否重复设置了 `--policy.device`；
2. SmolVLA 是否与 Isaac Sim 共用 `cuda:0`；
3. 是否启用了 `--env.enable_cameras=true`；
4. 是否使用 `robot_pov_cam_rgb`；
5. Arena 原生 camera video 是否正常。

单卡环境建议保留 `--policy.device=cpu --env.device=cuda:0`。

### 微波炉打开了，但视频看不到机械手

这通常不是动作失败，而是录像视角问题。LeRobot 视频来自机器人第一人称相机，不是第三人称 viewport。先用 Arena 原生命令同时录制 viewport 和 camera video，对比确认机器人和策略相机各自看到的内容。

### episode 只有约 5 秒

这是任务定义，不是 `video_length` 截断。`OpenDoorTask(..., episode_length_s=5.0)` 决定 episode 时长。

## 12. 推荐排查顺序

1. Arena + `zero_action`，验证启动和连续相机帧；
2. Arena + `random_action`，验证动作接口；
3. 同时录制 viewport 和 camera video，区分场景问题与相机视角问题；
4. LeRobot + CPU policy，验证 EnvHub、观测映射和模型；
5. 检查输出视频的帧数、帧率和是否存在全零帧；
6. 最后才尝试把策略移动到独立 GPU。

这样可以快速判断问题属于 Arena 环境、相机渲染、LeRobot 适配、模型输入映射，还是 CUDA/RTX 资源冲突。
