# Policy Runner 使用指南

## 概述

`policy_runner.py` 是 Isaac Lab-Arena 的运行时入口，用于加载一个已注册的环境（environment），并在其上运行一个策略（policy）进行 rollout。

---

## 一、基本命令格式

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  [全局参数] \
  <环境名称> [环境参数]
```

**核心概念**：`<环境名称>` 是一个**子命令**（subcommand），不是 `--env` 参数。每个已注册的环境会作为 argparse 子命令自动出现在解析器中。环境名称后面跟随的 `--xxx` 参数是该环境专属的配置项。

---

## 二、最简单的示例

```bash
# 用 zero_action policy 运行 kitchen_pick_and_place 环境，跑 100 步
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 100 \
  kitchen_pick_and_place \
  --embodiment franka_ik \
  --object cracker_box
```

```bash
# 用 zero_action policy 运行 gr1 桌子多物体环境
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 500 \
  --num_envs 16 \
  --env_spacing 4.0 \
  --enable_cameras \
  gr1_table_multi_object_no_collision \
  --embodiment gr1_joint \
  --episode_length_s 4.0
```

---

## 三、全局参数详解

全局参数位于环境子命令**之前**，分为以下几类：

### 3.1 Policy Runner 专属参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy_type` | str | 必填 | 策略类型名或 Python 类路径，见[第五节](#五可用-policy-类型) |
| `--num_steps` | int | — | 按步数运行（与 `--num_episodes` 二选一） |
| `--num_episodes` | int | — | 按 Episode 数量运行（与 `--num_steps` 二选一） |
| `--language_instruction` | str | — | 覆盖环境的默认语言指令 |
| `--record_viewport_video` | flag | False | 录制视口视频（mp4） |
| `--record_camera_video` | flag | False | 录制机器人相机视频 |
| `--output_base_dir` | str | `outputs` | 输出目录 |
| `--serve_evaluation_report` | flag | False | 运行后在 HTTP 上提供评估报告 |
| `--evaluation_report_port` | int | 8000 | 评估报告 HTTP 端口 |

### 3.2 通用 Builder 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_envs` | int | 1 | 并行环境数量 |
| `--env_spacing` | float | 30.0 | 环境间距（米），碰撞较多的场景建议调小 |
| `--seed` | int | 42 | 随机种子 |
| `--device` | str | `cuda:0` | 运行设备 |
| `--disable_fabric` | flag | False | 禁用 Fabric，使用 USB I/O |
| `--mimic` | flag | False | 启用 mimic 环境 |
| `--distributed` | flag | False | 多 GPU 分布式模式（配合 `torchrun` 使用） |

### 3.3 Arena 布局与物理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--no_solve_relations` | flag | — | 禁用空间关系求解（物体不做避碰摆放） |
| `--placement_seed` | int | — | 物体摆放随机种子（固定后每次摆放位置相同） |
| `--presets` | str | — | 物理后端预设：`physx` 或 `newton` |
| `--resolve_on_reset` / `--no-resolve_on_reset` | bool | True | 每次 reset 时是否重新摆放物体 |

### 3.4 AppLauncher 参数（来自 Isaac Lab）

| 参数 | 说明 |
|------|------|
| `--headless` | 无头模式（不显示 GUI） |
| `--viz` | 可视化后端：`kit`（Kit 渲染器） |
| `--enable_cameras` | 在 sim 启动时启用相机 |
| `--fps` | 仿真 FPS（如 `--fps 60`） |

### 3.5 高级参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--external_environment_class_path` | str | 外部环境类路径，格式：`module.path:ClassName` |
| `--env_graph_spec_yaml` | str | 使用 YAML 图规格文件定义环境 |
| `--list_variations` | flag | 列出当前环境的 Hydra 可配置变体并退出 |

---

## 四、查看帮助

```bash
# 查看所有全局参数和已注册的环境列表
python isaaclab_arena/evaluation/policy_runner.py --help

# 查看某个环境的专属参数
python isaaclab_arena/evaluation/policy_runner.py <环境名称> --help

# 示例
python isaaclab_arena/evaluation/policy_runner.py kitchen_pick_and_place --help
```

---

## 五、可用 Policy 类型

| 名称 | 类 | 说明 |
|------|-----|------|
| `zero_action` | `ZeroActionPolicy` | 始终输出全零动作，用于验证环境是否正常加载 |
| `rsl_rl` | `RSLRLPolicy` | 加载 RSL RL 训练的 checkpoint 进行推理 |
| `replay` | `ReplayActionPolicy` | 回放之前录制的 episode 动作序列 |

### 5.1 使用外部 Policy

如果不是已注册的 policy，可以直接给路径：

```bash
--policy_type my_package.my_module.MyPolicyClass
```

格式为 `模块路径.类名`（如 `isaaclab_arena.policy.zero_action_policy.ZeroActionPolicy`）。

---

## 六、已注册环境列表及专属参数

### 6.1 kitchen_pick_and_place — 厨房桌面抓取放置

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  kitchen_pick_and_place \
  --embodiment franka_ik \
  --object cracker_box
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | `cracker_box` | 要操作的物体 |
| `--object_set` | list[str] | — | 物体集合（每个 env 随机选一个） |
| `--embodiment` | str | `franka_ik` | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.2 pick_and_place_maple_table — 枫木桌抓取放置

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  pick_and_place_maple_table \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--embodiment` | str | — | 机器人 |
| `--hdr` | str | — | HDR 环境贴图 |
| `--light_intensity` | — | — | 光照强度 |
| `--pick_up_object` | — | — | 待抓取物体 |
| `--destination_location` | — | — | 目标位置 |
| `--additional_table_objects` | — | — | 桌面额外物体 |

### 6.3 gr1_table_multi_object_no_collision — GR1 多物体桌面

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 500 \
  --num_envs 16 --env_spacing 4.0 --enable_cameras \
  gr1_table_multi_object_no_collision \
  --embodiment gr1_joint \
  --episode_length_s 4.0
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--objects` | list[str] | — | 物体列表（默认 6 个 YCB 物体） |
| `--embodiment` | str | `gr1_joint` | 机器人：`gr1_joint` 或 `gr1_pink` |
| `--teleop_device` | str | — | 遥操作设备 |
| `--episode_length_s` | float | — | Episode 时长（触发周期性 reset） |
| `--mode` | str | `homogeneous` | 摆放模式：`homogeneous` 或 `heterogeneous` |
| `--num_envs` | int | 1 | 环境数量 |

### 6.4 galileo_pick_and_place — Galileo 抓取放置

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  galileo_pick_and_place \
  --embodiment g1 \
  --object cracker_box
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.5 lift_object — 举起物体

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  lift_object \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--object` | str | — | 目标物体 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |
| `--rl_training_mode` | — | — | RL 训练模式 |

### 6.6 press_button — 按按钮

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  press_button \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--object` | str | — | 目标按钮 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.7 cube_goal_pose — 方块目标位姿

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  cube_goal_pose \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--background` | str | — | 背景资产 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.8 franka_put_and_close_door — Franka 放物关门

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 300 \
  franka_put_and_close_door \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.9 put_item_in_fridge_and_close_door — GR1 放物入冰箱

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 500 \
  put_item_in_fridge_and_close_door \
  --embodiment gr1_joint
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--object_set` | list[str] | — | 物体集合 |
| `--kitchen_style` | str | — | 厨房风格 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.10 gr1_open_microwave — GR1 开微波炉

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 300 \
  gr1_open_microwave \
  --embodiment gr1_joint
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.11 gr1_turn_stand_mixer_knob — GR1 拧旋钮

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 300 \
  gr1_turn_stand_mixer_knob \
  --embodiment gr1_joint
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |
| `--target_level` | — | — | 目标档位 |
| `--reset_level` | — | — | 重置档位 |

### 6.12 tabletop_sort_cubes — 桌面方块分类

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  tabletop_sort_cubes \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--objects` | — | — | 物体列表 |
| `--destinations` | — | — | 目标位置列表 |
| `--background` | str | — | 背景 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.13 gear_mesh — 齿轮啮合装配

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  gear_mesh \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--background` | str | — | 背景资产 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.14 peg_insert — 插销装配

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  peg_insert \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--destination_object` | str | — | 目标孔位物体 |
| `--background` | str | — | 背景资产 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.15 tabletop_place_upright — 桌面竖直摆放

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  tabletop_place_upright \
  --embodiment franka_ik
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--background` | str | — | 背景资产 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |

### 6.16 dexsuite_lift — DexSuite 举起

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 \
  dexsuite_lift \
  --embodiment franka_ik
```

### 6.17 galileo_g1_locomanip_pick_and_place — G1 移动操作

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 500 \
  galileo_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_joint
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--destination` | str | — | 目标位置 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |
| `--task_description` | str | — | 任务描述 |
| `--mimic` | bool | — | 启用 mimic |
| `--auto` | bool | — | 自动模式 |

### 6.18 galileo_g1_static_pick_and_place — G1 固定底座抓取

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 500 \
  galileo_g1_static_pick_and_place \
  --embodiment g1_wbc_joint
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_cameras` | bool | False | 启用相机 |
| `--object` | str | — | 目标物体 |
| `--destination` | str | — | 目标位置 |
| `--embodiment` | str | — | 机器人 |
| `--teleop_device` | str | — | 遥操作设备 |
| `--task_description` | str | — | 任务描述 |
| `--lock_waist` | bool | — | 锁定腰部关节 |

---

## 七、可用 Embodiment（机器人）列表

| 名称 | 说明 |
|------|------|
| `franka_ik` | Franka Panda — IK 控制 |
| `franka_joint_pos` | Franka Panda — 关节位置控制 |
| `gr1` | GR1 人形机器人（GR1-T2） |
| `gr1_joint` | GR1 — 关节位置控制 |
| `gr1_pink` | GR1 — 欠驱动灵巧手控制 |
| `g1` | G1 人形机器人 |
| `g1_wbc_joint` | G1 — WBC 关节控制 |
| `g1_wbc_pink` | G1 — WBC 灵巧手控制 |
| `g1_wbc_agile_pink` | G1 — WBC 敏捷灵巧手 |
| `g1_wbc_agile_joint` | G1 — WBC 敏捷关节 |
| `agibot` | 智元机器人 |
| `galbot` | Galbot 移动操作机器人 |
| `kuka_allegro` | Kuka + Allegro 灵巧手 |
| `droid` | DROID 机器人 |
| `droid_differential_ik` | DROID — 差分 IK |
| `droid_rel_joint_pos` | DROID — 相对关节位置 |
| `droid_abs_joint_pos` | DROID — 绝对关节位置 |
| `no_embodiment` | 无机器的纯场景 |

---

## 八、运行原理

执行 `policy_runner.py` 的内部流程：

```
1. 解析全局参数 → 构建 SimulationAppContext（启动 Isaac Sim）
2. 解析 --policy_type → 从 PolicyRegistry 获取 Policy 类
3. 解析 <环境名称> 子命令 → 从 EnvironmentRegistry 获取环境 Factory
4. 读取环境专属参数 → 构建环境配置 dataclass → ArenaEnvBuilder.make_registered()
5. 构建 policy 实例 → policy.get_action() + env.step() 循环 rollout
6. 达到 --num_steps 或 --num_episodes 后停止
```

## 九、使用外部环境

### 方式一：通过类路径

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 100 \
  --external_environment_class_path my_package.my_module.MyEnvironment \
  <环境名称> ...
```

### 方式二：通过 YAML 图规格

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 100 \
  --env_graph_spec_yaml path/to/spec.yaml
```

YAML 中可以声明 `cli_override_specs` 来动态添加命令行覆盖参数。

---

## 十、常用组合速查

```bash
# 快速验证：zero_action + 单环境 + GUI
python isaaclab_arena/evaluation/policy_runner.py \
  --viz kit --policy_type zero_action --num_steps 100 \
  kitchen_pick_and_place --embodiment franka_ik

# 无头模式 + 多环境 + 录视频
python isaaclab_arena/evaluation/policy_runner.py \
  --headless --policy_type zero_action --num_steps 1000 \
  --num_envs 16 --enable_cameras --record_camera_video \
  kitchen_pick_and_place --embodiment franka_ik

# RL checkpoint 推理
python isaaclab_arena/evaluation/policy_runner.py \
  --viz kit --policy_type rsl_rl --num_steps 500 \
  kitchen_pick_and_place --embodiment franka_ik --object power_drill

# 列出某个环境的所有 Hydra 可配置变体
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  gr1_table_multi_object_no_collision --list_variations
```

---

## 十一、常见问题

### Q: 命令不生效 / 报 "unrecognized arguments"

环境参数必须放在环境子命令名称**之后**。比如：

```bash
# ✅ 正确：--object 在 kitchen_pick_and_place 后面
python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_steps 100 \
  kitchen_pick_and_place --object power_drill

# ❌ 错误：--object 在环境名称前面
python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_steps 100 \
  --object power_drill kitchen_pick_and_place
```

### Q: `--policy_type` 不生效

`--policy_type` 是全局参数，必须在环境子命令**之前**。

### Q: 如何查看某个环境支持的参数

```bash
python isaaclab_arena/evaluation/policy_runner.py <环境名称> --help
```

### Q: 资产下载失败 / 卡在 `omni.client.copy`

环境所需的 USD 资产托管在 NVIDIA Omniverse Nucleus 服务器上。首次运行时会自动下载并缓存。
如果网络受限，确保：
- Isaac Sim 已正确安装并登录了 Nucleus
- 检查 `~/.cache/ov/` 和 `~/.cache/isaaclab_arena/assets/` 缓存目录
