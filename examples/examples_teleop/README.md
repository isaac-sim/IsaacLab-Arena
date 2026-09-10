# 遥操作示例（examples_teleop）

> 从最简单的「桌子 + 机械臂 + 方块」遥操作开始（键盘 → Quest XR），逐步验证遥操作链路，后续逐级添加更复杂场景/任务/录制。

---

## 一、`example01_teleop_simple.py`（键盘遥操作，最简）

**场景**：一张桌子 + 一个 Franka 机械臂 + 一个方块（`dex_cube`）。**无任务、无成功判定**，只验证「键盘 → 末端增量 → Differential IK → Franka 关节」这条遥操作链路。

**为什么这么简单**：场景只用 `table`/`ground_plane`/`dex_cube` 这类本地已缓存资产，**启动快、无需等 AWS 下载**，适合先跑通链路。

### 1.1 键位表（Se3Keyboard）

| 按键 | 作用 |
|------|------|
| `W` / `S` | 末端前后平移（x） |
| `A` / `D` | 末端左右平移（y） |
| `Q` / `E` | 末端上下平移（z） |
| `Z` / `X` | 绕 x 轴旋转（roll） |
| `T` / `G` | 绕 y 轴旋转（pitch） |
| `C` / `V` | 绕 z 轴旋转（yaw） |
| `K` | 夹爪开/合切换 |
| `R` | 重置环境 |

### 1.2 运行（交互式，需要显示器 + 键盘焦点）

```bash
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu

# 容器内：
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_simple.py
```

**验证步骤**：

1. Kit 窗口打开，看到桌子 + Franka + 方块。
2. 控制台打印遥操作设备信息和键位提示。
3. **点击 Kit 窗口**（获得键盘焦点），按 `W/S/A/D/Q/E` 看机械臂末端是否跟着移动。
4. 移到方块上方，按 `K` 闭合夹爪。
5. 按 `R` 重置环境，方块/机械臂回到初始位姿。
6. 关闭 Kit 窗口或 `Ctrl+C` 退出。

### 1.3 headless 冒烟（无显示器，仅验证环境能起）

```bash
docker run --rm \
  --privileged --ipc=host --net=host --runtime=nvidia --gpus=all \
  -v "$(pwd)":/workspaces/isaaclab_arena \
  -v "$HOME/.cache:/home/$(id -un)/.cache" \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
  --env DOCKER_RUN_USER_ID="$(id -u)" --env DOCKER_RUN_USER_NAME="$(id -un)" \
  --env DOCKER_RUN_GROUP_ID="$(id -g)" --env DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  --env ISAACLAB_PATH=/workspaces/isaaclab_arena/submodules/IsaacLab \
  isaaclab_arena_magengyu:latest \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh examples/examples_teleop/example01_teleop_simple.py --headless --num_steps 50 --no-keep_open"
```

预期：环境创建并 reset、打印键位提示，跑 50 步后退出（键盘无输入，机械臂静止）。这验证了「场景 + 遥操作设备 + 环境」能正常搭建。

### 1.4 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num_steps` | 500 | `--no-keep_open` 时的最大步数 |
| `--keep_open` / `--no-keep_open` | 开 | 演示后是否保持 Kit 窗口打开 |
| `--headless` | 关 | 无显示器冒烟时开启 |

### 1.5 SpaceMouse 版（`example01_teleop_spacemouse.py`）

把输入设备从键盘换成 3Dconnexion SpaceMouse，场景完全一致。**原生设备、单终端、无需 CloudXR**：

#### 1.5.1 Linux 前置配置：开放 hidraw 权限

Isaac Lab 的 `Se3SpaceMouse` 使用 Python `hidapi` 直接读取 `/dev/hidraw*`，**不依赖 3DxWare/3dxsrv**。Ubuntu 24.04 不需要安装旧版 3DxWare 1.8.0。

如果 SpaceMouse 对应的 hidraw 节点是 `root:root 0600`，普通用户无法打开设备；如果设备在容器启动后才插入，容器中还可能没有对应的 `/dev/hidrawN` 节点。这两种情况都会导致 SpaceMouse 初始化失败，脚本可能直接报错退出，在 Kit 窗口中看起来像 Isaac Sim 崩溃。

> 这只说明“启动 SpaceMouse 脚本时崩溃”的一个已验证原因，并不代表所有 Isaac Sim 崩溃都由 hidraw 权限引起。应先执行下面的枚举验证再判断。

项目已验证的配置流程如下。

**第 1 步：在宿主机添加 udev 规则**

```bash
sudo tee /etc/udev/rules.d/99-3dxconnexion.rules <<'EOF'
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

规则只匹配 3Dconnexion 的 USB Vendor ID `256f`。`MODE="0666"` 允许本机所有用户读写匹配设备；这是当前项目实测配置。如果机器是多人共用环境，可改用设备组或 `TAG+="uaccess"` 做更严格的权限控制。

**第 2 步：重新启动容器**

SpaceMouse 最好在启动容器前插好。若设备是在容器启动后插入，需要重启容器，让容器重新获得 `/dev/hidrawN` 设备节点：

```bash
docker stop isaaclab_arena_magengyu-latest-magengyu
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu
```

**第 3 步：验证容器内能枚举 SpaceMouse**

在宿主机执行：

```bash
docker exec isaaclab_arena_magengyu-latest-magengyu /isaac-sim/python.sh -c \
  "import hid; print([d['product_string'] for d in hid.enumerate() if d['vendor_id'] == 0x256f])"
```

预期输出类似：

```text
['SpaceMouse Wireless']
```

只有这一步成功后，再运行遥操作脚本。如果输出为空，优先检查：SpaceMouse 是否插好、宿主机是否出现 `/dev/hidraw*`、udev 规则是否生效，以及容器是否在插入设备后重新启动。

#### 1.5.2 启动遥操作

```bash
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse.py
```

操作：6DoF 帽控制末端（推/拉/平移 + 扭/旋转），左键夹爪开合，右键重置。

**灵敏度**：默认 `--pos_sensitivity 0.4 --rot_sensitivity 0.8`（对齐 IsaacLab 原生；Arena 默认 0.05 太小会移动缓慢），可按需调：

```bash
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse.py \
  --pos_sensitivity 0.8 --rot_sensitivity 1.5
```

#### 1.5.3 常见问题

**启动后报错退出或 Kit 窗口关闭**

先运行 1.5.1 的 `hid.enumerate()` 验证。若无法打印 `SpaceMouse Wireless`，处理 hidraw 权限并重启容器；不要先调灵敏度，灵敏度不会解决设备打不开的问题。

**机械臂移动很慢**

确认脚本使用默认的 `0.4 / 0.8`，再逐步调高。`--pos_sensitivity 0.8 --rot_sensitivity 1.5` 只是较灵敏的示例，不是解决崩溃的参数。

**只能平移，不能旋转**

SpaceMouse Wireless 有线模式（`256f:c62e`）的 HID 报告是 **13 字节**，6 个自由度位于同一个 Report 中。本项目已修改 `submodules/IsaacLab/source/isaaclab/isaaclab/devices/spacemouse/se3_spacemouse.py`，让 `SpaceMouse Wireless` 按 13 字节读取；缺少该修复时会丢失后 6 字节的旋转轴数据。

完整排查记录见 `.talisman/dev_docs/2026-08/task05_3dxware遥操录制.md`。

### 1.6 SpaceMouse 录制、回放和训练

录制脚本把 `NoTask` 换成 `GoalPoseTask`，并接入 Isaac Lab recorder。把方块放进绿色目标区后自动保存成功 episode：

```bash
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_record.py \
  --dataset_file /tmp/franka_spacemouse_demos.hdf5 \
  --num_demos 5 --num_success_steps 2
```

SpaceMouse 和 Quest 只是产生动作的设备，录出的 HDF5 契约相同，所以回放与训练直接复用通用脚本：

```bash
# 独立的 SpaceMouse 数据回放脚本；回放时不需要连接 SpaceMouse
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_replay.py \
  --dataset_file /tmp/franka_spacemouse_demos.hdf5

# 独立的 SpaceMouse BC 训练脚本；不需要 Isaac Sim
python3 examples/examples_teleop/example01_teleop_spacemouse_train_bc.py \
  --dataset_file /tmp/franka_spacemouse_demos.hdf5 \
  --epochs 100 --output /tmp/franka_spacemouse_bc.pt
```

三个 SpaceMouse 文件现在可以分别运行。原理上，回放读取的是 `initial_state + actions`，BC 读取的是 `obs + actions`，两者仍不依赖录制时的输入设备；录制、回放时必须保持机器人、场景、控制器和动作维度一致。

如果想学习 Isaac Lab reset/event randomization，可以运行随机化录制版。它仍然是同一个桌面方块任务，但每次 reset 会重新采样方块初始位置和 yaw：

```bash
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_record_randomized.py \
  --dataset_file /tmp/franka_spacemouse_randomized_demos.hdf5 \
  --num_demos 5 \
  --randomize_cube_pose
```

随机化学习说明见 `examples/examples_teleop/README_randomization.md`。先从小范围随机开始，再逐步扩大范围，录制和训练时的随机化范围要保持一致或至少覆盖部署场景。

---

## 二、`example02_teleop_quest.py`（Quest XR 遥操作，最简场景）

**场景**与 example01 完全一致（桌子 + Franka + 方块，`NoTask`），只是把输入设备从**键盘**换成 **Quest 右手柄**：

- 右手柄 **6DoF 位姿**控制末端（retargeting → Differential IK → 关节）
- 右手柄 **Trigger** 控制夹爪开合

参考文档：
https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/locomanipulation/step_2_teleoperation.html

### 2.1 前置条件

| 条件 | 说明 |
|------|------|
| Meta Quest 头显 | 与主机**同一局域网** |
| 防火墙 | 开放 `49100/tcp`、`47998/udp`、`48322/tcp` |

### 2.2 运行步骤（两个终端）

**终端 1 — 启动 CloudXR runtime**：

```bash
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu
# 容器内：
/isaac-sim/python.sh -m isaacteleop.cloudxr --host-client
```

（首次会要求接受 NVIDIA CloudXR EULA；启动后生成 `~/.cloudxr/run/cloudxr.env`）

**终端 2 — 加载环境变量后运行脚本**（**顺序很重要**）：

```bash
source ~/.cloudxr/run/cloudxr.env
/isaac-sim/python.sh examples/examples_teleop/example02_teleop_quest.py --xr
```

**连接 Quest**：

1. 查主机局域网 IP（`ip -brief address`，别选 `docker0`/`lo`）
2. Quest 浏览器打开 `https://<主机IP>:48322/client/` → 接受证书 → **Connect**
3. Isaac Sim 的 **XR** 标签页启动 Session → Quest 点 **Play**

### 2.3 与键盘版（example01）的代码差异

| | example01 键盘 | example02 Quest |
|---|---|---|
| 遥操作设备 | `get_device_by_name("keyboard")` | `get_device_by_name("openxr")` |
| 设备接口 | `create_teleop_device`（IsaacLab） | `create_isaac_teleop_device`（IsaacTeleop） |
| 循环 | `advance()` 直接给动作 | `advance()` 可能返回 `None`（等 WebXR 数据） |
| 上下文 | 无 | `with teleop_interface:` 管理 Session |
| 启动参数 | 无 | 必须 `--xr` |

### 2.4 常见问题

- **有画面但机械臂不动**：检查终端 2 是否 `source cloudxr.env`、命令是否带 `--xr`、XR Session 是否已启动、Quest 是否点了 Play。
- **`Media connection could not be established`**：UDP 媒体通道没建立，检查防火墙端口 + 同一局域网 + 关 VPN/代理。
- **方向不对**：Quest 里长按 Meta/Oculus 键重置视角。
- **`Port 49100 is already in use`**（启动 `--host-client` 时报错）：上次的 CloudXR runtime 没退干净、仍占着 49100。清理后重试：

  ```bash
  pkill -KILL -f 'isaacteleop.cloudxr.runtime'
  rm -f ~/.cloudxr/run/ipc_cloudxr ~/.cloudxr/run/runtime_started ~/.cloudxr/run/cloudxr.pid
  ```

  退出时优先用 XR 标签页 **Stop Session** 正常停，而不是 `Ctrl-C`，可避免残留。

> 完整原理（retargeting、Trigger 夹爪、DLS IK、坐标系、抖动/灵敏度调节）见 [XR_TELEOP_TUTORIAL.md](../XR_TELEOP_TUTORIAL.md)。

---

## 三、`example03_teleop_quest_anchor.py`（锚点修正版）

`example02` 跑通后，你会发现「桌子贴地 + 机械臂朝向不匹配」——因为 Franka embodiment 没配 XrCfg，用了空锚点。`example03` 在 example02 基础上补上锚点（**方式 B**，外挂式、不改 fork 本体）：

```python
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

# 在 create_isaac_teleop_device 之前插入：
env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_pos = (0.0, 0.0, -0.9)              # 高度：眼睛在桌面上方
env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_rot = (0.0, 0.0, -0.70711, 0.70711)  # 绕 Z -90° 对齐朝向
env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_prim_path = "/World/envs/env_0/Robot/panda_link0"
env.unwrapped.cfg.isaac_teleop.xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
env.unwrapped.cfg.isaac_teleop.xr_cfg.fixed_anchor_height = True
```

- `anchor_pos` 的 Z（-0.9）和 `anchor_rot` 的 yaw（-0.70711）需按你的桌高/坐姿实测微调；yaw 反了改成 `+0.70711`（差 180°）。
- 坐标轴原理见 [坐标系与变换关系](../../../../docs/IsaacLab-Arena-遥操作坐标系与变换关系.md)。

### 3.1 运行步骤（两个终端）

**终端 1 — 启动 CloudXR runtime**：

```bash
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu
# 容器内：
/isaac-sim/python.sh -m isaacteleop.cloudxr --host-client
```

**终端 2 — 加载环境变量后运行脚本**（**顺序很重要**）：

```bash
source ~/.cloudxr/run/cloudxr.env
/isaac-sim/python.sh examples/examples_teleop/example03_teleop_quest_anchor.py --xr
```

**连接 Quest**：

1. 查主机局域网 IP（`ip -brief address`，别选 `docker0`/`lo`）
2. Quest 浏览器打开 `https://<主机IP>:48322/client/` → 接受证书 → **Connect**
3. Isaac Sim 的 **XR** 标签页启动 Session → Quest 点 **Play**

---

## 四、`example04_teleop_quest_visualize.py`（手柄位姿可视化）

在 example03 基础上，把「右手柄在世界系的位姿」用 debug_draw 画出来（红色点=位置，RGB 轴=朝向），用于直观检查锚点（anchor_pos / anchor_rot）对不对。运行步骤同 example03。

---

## 五、`example05_teleop_quest_record.py`（Quest 遥操作 + 录制示范）

把 example03 的「无任务」换成「目标位姿任务」并接入 recorder，形成完整 IL 链路的**录制**环节：

- **任务**：`NoTask` → `GoalPoseTask`，把方块放进目标区（脚本顶部 `TARGET_X/Y/Z_RANGE`）算成功。
- **录制**：通过 `env_cfg_callback` 注入 `ActionStateRecorderManagerCfg`，逐帧把 (obs, action) 写进 HDF5；连续 `--num_success_steps` 步成功后自动导出该 demo 并 reset。

### 5.1 运行（两个终端，同 example03）

```bash
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu

# 终端 1：
/isaac-sim/python.sh -m isaacteleop.cloudxr --host-client
# 终端 2：
source ~/.cloudxr/run/cloudxr.env
/isaac-sim/python.sh examples/examples_teleop/example05_teleop_quest_record.py \
  --xr --dataset_file /tmp/franka_demos.hdf5 --num_demos 5 --num_success_steps 2
```

连接 Quest 后，把方块放进目标区并**稳定停留**（连续 `--num_success_steps` 步都满足目标判定，默认 2）即保存一条 demo。**「连续 N 步」是防抖**：避免方块路过目标区或边缘抖动被误判，不是要放 N 次。录满 `--num_demos` 条自动退出。

### 5.2 录制结果（HDF5 结构）

```
data/demo_0/
├── actions            (T, act_dim)   遥操作动作（末端增量 + 夹爪）
├── obs/               (T, *)         策略观测（dict，按分量分组）
│   ├── actions / joint_pos / joint_vel / eef_pos / eef_quat / gripper_pos
├── states / initial_state / processed_actions
└── attrs["success"] = True
```

用 `h5py` 检查：`python -c "import h5py; f=h5py.File('/tmp/franka_demos.hdf5','r'); print(list(f['data'].keys()))"`。

### 5.3 带 START/STOP 的版本（`example05_teleop_quest_record_startstop.py`）

官方完整版录制脚本（`record_demos.py`）支持 Quest 端 **START 开始录制 / STOP 暂停 / RESET 重置这一条**。想在 Franka 简单场景里也体验这套交互，用 `example05_teleop_quest_record_startstop.py`——它保留控制通道（example05 关闭了它），其余逻辑一致：

```bash
/isaac-sim/python.sh examples/examples_teleop/example05_teleop_quest_record_startstop.py \
  --xr --dataset_file /tmp/franka_demos.hdf5
```

启动后默认「暂停」，Quest 端 **START** 才开始录，成功后打印 `✅ 已录制第 N 条成功 demo`，**RESET** 重置这一条（不计成功）。

---

## 六、`example06_teleop_quest_replay.py`（回放录制的示范）

把 example05 录制的 HDF5 在 Isaac Sim 里逐帧回放，视觉确认轨迹对不对。**不需要 Quest / 遥操作设备**——重建同样场景，把录制的 `action` 逐帧喂回 `env.step`：

```bash
# 容器内，单终端
/isaac-sim/python.sh examples/examples_teleop/example06_teleop_quest_replay.py \
  --dataset_file /tmp/franka_demos.hdf5

# 只回放第 0、2 条 demo：
/isaac-sim/python.sh examples/examples_teleop/example06_teleop_quest_replay.py \
  --dataset_file /tmp/franka_demos.hdf5 --select_episodes 0 2
```

参数：`--select_episodes`（指定 demo 编号，默认全部）、`--step_hz`（回放步率，默认 30，越大越快）。

> 回放原理：重建与录制时一致的场景 → `env.reset_to(initial_state)` 回到录制的起点 → 逐帧 `env.step(录制的 action)`。录制的 `actions` 是「末端增量 + 夹爪」动作，回放时仍走同样的 Differential IK，所以能还原轨迹。

---

## 七、`example07_train_bc.py`（行为克隆训练）

读 example05 录制的 HDF5，训练一个 MLP 把 obs 映射到 action。**纯 PyTorch + h5py，不需要 Isaac Sim**，任意装了 torch/h5py 的 Python 都能跑：

```bash
cd references/IsaacLab-Arena_Research/references/IsaacLab-Arena_magengyu
python3 examples/examples_teleop/example07_train_bc.py \
  --dataset_file /tmp/franka_demos.hdf5 --epochs 100 --output /tmp/franka_bc.pt
```

原理：把 (obs, action) 对当监督学习数据，最小化 `||π(obs) - action||²`（就是 MSE，跟 MNIST 一样）。训练完保存 checkpoint（含 `state_dict` + obs/action 维度 + obs 拼接顺序）。

### 完整链路回顾（遥操作 → 录制 → 回放 → 训练）

```
example03 遥操作（人戴 Quest 抓方块）
        ↓
example05 录制（把成功示范写成 HDF5）
        ↓
example06 回放（视觉验证轨迹对不对）← 训练前先确认录得对
        ↓
example07 训练（BC：obs → action）
        ↓
（下一步 example08：用训练好的 policy rollout 评估）
```

---

## 八、与复杂版 `example03_teleop.py` 的区别

| | 本示例 `example01_teleop_simple.py` | fork 自带 `example03_teleop.py` |
|---|---|---|
| 场景 | 桌子 + 方块 | 厨房 + 番茄汤罐 |
| 任务 | 无（`NoTask`） | `PickAndPlaceTask`（放罐子到橱柜） |
| 相机 | 关 | 开 |
| 启动 | 秒开 | 需等厨房 AWS 下载 |
| 用途 | 先验证遥操作链路 | 完整抓放任务演示 |

---

## 九、后续计划（逐级添加）

- [x] `example01_teleop_simple.py`：键盘遥操作（最简）
- [x] `example02_teleop_quest.py`：Quest XR 遥操作（最简场景）
- [x] `example03_teleop_quest_anchor.py`：Quest XR 锚点修正版（方式 B）
- [x] `example04_teleop_quest_visualize.py`：手柄位姿可视化（检查锚点）
- [x] `example05_teleop_quest_record.py`：Quest 录制示范（GoalPoseTask + recorder）
- [x] `example05_teleop_quest_record_startstop.py`：带 START/STOP 的录制版（对齐官方 record_demos.py）
- [x] `example06_teleop_quest_replay.py`：回放录制的 HDF5（视觉验证轨迹）
- [x] `example07_train_bc.py`：行为克隆训练（读 HDF5 训 MLP）
- [ ] 逐步对齐 `docs/IsaacLab-Arena_magengyu-遥操作指南.md` 里的完整流程

---

## 十、相关文档

| 文档 | 内容 |
|------|------|
| [IsaacLab-Arena_magengyu-遥操作指南.md](../../../../docs/IsaacLab-Arena_magengyu-遥操作指南.md) | 键盘 / XR 遥操作 + 官方全流程 |
| [gr1_open_microwave_workflow.md](gr1_open_microwave_workflow.md) | GR1 开门完整模仿学习流程（5 步） |
| [example03_teleop.py](../example03_teleop.py) | fork 自带复杂版（厨房抓放） |
| [XR_TELEOP_TUTORIAL.md](../XR_TELEOP_TUTORIAL.md) | XR 遥操作原理详解 |

---

## 十一、ABB IRB1200 键盘 / SpaceMouse 遥操作 + Robotiq 2F-140 夹爪

这组示例从 `references/abb/abb_irb1200_support/urdf/irb1200_7_70.xacro` 出发，新增 ABB IRB1200-7/0.70 机械臂，并挂载可开合 Robotiq 2F-140 或 Newton 专用简化夹爪。当前提供这些入口：

- `example21_teleop_spacemouse_abb_irb1200_robotiq_2f140_control.py`：SpaceMouse 控制末端，左键切换夹爪开合。
- `example22_teleop_keyboard_abb_irb1200_robotiq_2f140_control.py`：键盘控制末端，`K` 键切换夹爪开合。
- [example23](example23_teleop_keyboard_abb_irb1200_robotiq_2f140_newton.py)：复用 example40 的 Newton standalone + MuJoCo 物理、原生 GL 窗口键盘遥操作；真实 2F-140 单主动轴，无抓取辅助。
- `example24_teleop_keyboard_abb_irb1200_simple_gripper_newton.py`：Newton 物理后端 + 简化二指夹爪路线验证，不加载真实 Robotiq 闭链夹爪。
- `example25_teleop_keyboard_abb_irb1200_physical_simple_gripper_newton.py`：Newton 物理后端 + prismatic joint 简化物理夹爪测试。

### 11.1 转换 xacro 到 Isaac USD

先启动容器：

```bash
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu
```

容器内执行：

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh tools/convert_abb_irb1200_xacro_to_isaac_usd.py --headless
```

转换后会生成普通机械臂 USD：

```text
isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.usda
```

以及带可开合 Robotiq 2F-140 的 PhysX 组合 USD：

```text
isaaclab_arena/assets/robots/abb/irb1200_7_70_robotiq_2f140_physx/irb1200_7_70.usda
```

源模型在：

```text
../abb/abb_irb1200_support/
```

转换脚本会读取源 xacro，并把可直接加载的 USD 写入机器人公用目录。`example21/22` 加载 PhysX 组合；新版 `example23` 使用 [example40 的独立组合资产](assets/abb_irb1200_robotiq_2f140_urdf/abb_irb1200_robotiq_2f140.usda)。新资产的生成方式见第十三节，不影响 PhysX 版。

### 11.2 SpaceMouse 遥操作

```bash
/isaac-sim/python.sh examples/examples_teleop/example21_teleop_spacemouse_abb_irb1200_robotiq_2f140_control.py \
  --pos_sensitivity 0.25 \
  --rot_sensitivity 0.55
```

SpaceMouse 操作：

| 操作 | 作用 |
|------|------|
| 6DoF 帽 | 控制末端 6D 增量位姿 |
| 左键 | Robotiq 2F-140 夹爪开/合切换 |
| 右键 / `R` | 重置环境 |

### 11.3 键盘遥操作

```bash
/isaac-sim/python.sh examples/examples_teleop/example22_teleop_keyboard_abb_irb1200_robotiq_2f140_control.py
```

如果觉得键盘移动慢，可以调大灵敏度：

```bash
/isaac-sim/python.sh examples/examples_teleop/example22_teleop_keyboard_abb_irb1200_robotiq_2f140_control.py \
  --pos_sensitivity 0.12 \
  --rot_sensitivity 0.18
```

键盘控制方式沿用 Isaac Lab `Se3Keyboard`：

| 按键 | 作用 |
|------|------|
| `W` / `S` | 末端前后平移（x） |
| `A` / `D` | 末端左右平移（y） |
| `Q` / `E` | 末端上下平移（z） |
| `Z` / `X` | 绕 x 轴旋转（roll） |
| `T` / `G` | 绕 y 轴旋转（pitch） |
| `C` / `V` | 绕 z 轴旋转（yaw） |
| `K` | Robotiq 2F-140 夹爪开/合切换 |
| `R` | 重置环境 |

### 11.4 example21/22 的 Arena 控制结构（不适用于新版 example23）

动作维度是 7：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

前 6 维交给 Differential IK 控制 IRB1200 的 `joint_1` 到 `joint_6`，最后 1 维控制 Robotiq 的主动关节 `finger_joint`。其它 Robotiq 手指关节通过 USD mimic 关系跟随，不需要单独给 action。

脚本启动后会检查：

```text
[CHECK] robotiq_2f140_mount exists: True
[CHECK] Robotiq base_link exists: True
[CHECK] Robotiq finger_joint prim exists: True
```

看到这三项都是 `True`，说明夹爪已经挂到 `link_6` 下，并且 Isaac Lab 能找到开合控制关节。运行时如果出现类似 `7 != 14` 的 actuator/joint 数量提示，一般可以接受：IRB1200 六轴 + Robotiq 主动 `finger_joint` 一共 7 个主动控制项，剩下的 Robotiq 手指关节由夹爪资产里的约束/驱动关系跟随。

### 11.5 Newton 键盘遥操作

**2026-09-09 已修复：example23 改为复用 example40 的真实 ABB + 2F-140 资产和物理配置，不再使用旧 Kit 强写关节实现。** 在已有 Docker 容器内运行：

```bash
/isaac-sim/python.sh examples/examples_teleop/example23_teleop_keyboard_abb_irb1200_robotiq_2f140_newton.py \
  --viewer gl --pos-speed 0.12 --rot-speed 0.5 --keep_open
```

- VS Code 一键入口：**Terminal → Run Task → Robotiq 2F140：Docker 键盘遥操作（example23 已修复）**。
- `W/S`：世界 `+X/-X`；`A/D`：`+Y/-Y`；`Q/E`：`+Z/-Z`，**不再映射到单个关节**。
- `Z/X`、`T/G`、`C/V`：绕世界 X/Y/Z 正反旋转；按住 Shift 使用 20% 精细速度。
- `K`：按下一次切换开合，按住不重复切换；松开 K 不会自动张开。
- `R`：恢复初始机器人和物体，重建 solver、IK、双状态缓冲；按住 R 不重复重置。
- Space 暂停/恢复，Esc 关闭。手动模式失焦立即停止增量运动，**不自动松爪**；回到窗口后松键再按。
- 鼠标中键环绕，Shift+中键平移，滚轮缩放。WASDQE 不再同时移动相机。
- 末端从方块正上方开始。用 E 下降，靠近物体时按 Shift 精调；K 闭合后等待约 2 秒，再 Q 抬升。靠接触摩擦抓取，错误对位仍可能抓空或滑落。

控制链路是：**键盘事件 → 60 Hz 速度积分 → 可变指腹 TCP + 目标旋转补偿 → IK → 有限关节驱动 → 600 Hz 接触物理**。保留六个夹爪转动关节和五条原生 mimic，只驱动 `finger_joint`，上限 5 N·m。没有运行时 proxy、从动轴电机、物体绑定、机械臂状态强写或清速度。方块默认 5 cm、1 kg、摩擦系数 0.8，保留重力。

#### 自动验证的是同一键盘控制路径

```bash
/isaac-sim/python.sh examples/examples_teleop/example23_teleop_keyboard_abb_irb1200_robotiq_2f140_newton.py \
  --viewer null --replay --test --quiet \
  --report examples/examples_teleop/assets/abb_irb1200_robotiq_2f140_urdf/validation/keyboard_gpu_1kg.json
```

- 可视回放改成 `--viewer gl`。默认回放 1479 帧 / 24.65 s 仿真时间，完成后自动退出；真实运行时间取决于机器性能。
- 回放使用同一套 press/release 和驱动逻辑，不调用 example40 的自动轨迹；已知场景定时输入，不是视觉策略或对任意初始位置的保证。
- 回放时忽略人工运动键和窗口失焦，避免污染输入；Space 暂停、R 重置、Esc 退出仍有效。**提前关窗会正确判定未完成，不算通过。**
- 低力矩负对照加 `--grip-effort 0.05 --expect-grasp-failure`，应抓不起，而不是偷偷绑定物体。
- CPU 后端加 `--mujoco-cpu`。平移/姿态有跟踪误差预算、IK 可达性检查和驱动目标限速，但**没有自动避障/自碰撞规划**；桌面附近旋转仍需谨慎。

```bash
/isaac-sim/python.sh tools/test_example23_keyboard.py --gui
```

以上 [专项回归](../../tools/test_example23_keyboard.py) 检查 K 去重复、松键保持、失焦、暂停、真实 GL 事件路由、相机隔离、旋转开合 TCP、IK 拒绝和完整重置。CPU/GPU 1 kg 抓放与 0.05 N·m 负对照均已通过，指标及边界见 [修复记录](../../../../docs/ABB-IRB1200-Robotiq2F140-原生联动纯物理抓取.md#8-example23-键盘遥操作修复2026-09-09)。

#### 旧入口迁移

- `--num_steps` 是 `--num-frames` 的兼容别名；`--pos_sensitivity/--rot_sensitivity` 保留，但单位明确为 m/s 和 rad/s，不是每帧跳变量。
- `--lock_orientation`、`--no-lock_orientation`、`--hold_gripper_open` 仍可用。保持张开也只发送有限驱动目标，不夹紧状态。
- `--control_mode ik` 保留；`joint_direct`、`joint_relative`、`--direct_gripper`、proxy 和 `*_step_scale` 补丁参数已移除，传入会报错。
- `--visualizer kit` 只作显式迁移别名，并打印改用 GL 的提示；**不会启动 Kit**。`--presets`、`--visualizer none/newton` 不再适用。
- 旧版“直接写六轴、驱动两个夹爪主关节、高摩擦 runtime proxy 仍夹不住”的记录属于旧资产/旧控制链，不能再作为 Newton 不支持原生 mimic 的结论。

如果要验证更稳定的 Newton 简化夹爪路线，运行：

```bash
/isaac-sim/python.sh examples/examples_teleop/example24_teleop_keyboard_abb_irb1200_simple_gripper_newton.py \
  --debug_joints
```

这个脚本不加载真实 Robotiq 2F-140 多连杆闭链资产，而是在 ABB `link_6` 下挂一个带碰撞几何的简化二指夹爪。`K` 切换两个手指的目标开合距离，手指逐帧移动，用来验证 Newton 中简化夹爪的尺寸、位置和键盘流程。它不是最终真实夹爪动力学模型，也不能作为可靠物理抓取结果，但可以避开 mimic joints、闭链约束和被动关节在 Newton 下引起的乱动。

如果要测试更接近物理夹爪的版本，运行：

```bash
/isaac-sim/python.sh examples/examples_teleop/example25_teleop_keyboard_abb_irb1200_physical_simple_gripper_newton.py \
  --joint_step_scale 0.015 \
  --debug_joints
```

`example25` 加载一个专用 USD：

```text
isaaclab_arena/assets/robots/abb/irb1200_7_70_simple_gripper_newton/irb1200_7_70.usda
```

这个 USD 在 ABB `link_6` 下预先声明两个带 collision 的 finger rigid bodies，并用 `left_finger_joint` / `right_finger_joint` 两个 prismatic joints 控制开合。`K` 修改的是 prismatic joint position target，不再直接改视觉 transform。

当前 `example25` 的定位是 Newton 抓取链路调试，而不是 PhysX 版真实 Robotiq 的等价替代。它做了这些 Newton 专用处理：

- 场景仍使用世界重力，方块会正常落下。
- 机器人本体使用 `disable_gravity=True` 和 direct joint 写入，避免 ABB 在 Newton 下因缺少稳定重力补偿而下坠、抖动或关节缠绕。
- 抓取物不再使用 Nucleus `dex_cube_instanceable.usd`，而是脚本内的 `newton_grip_cube`，尺寸 `0.04 x 0.04 x 0.04 m`，质量 `0.015 kg`，高摩擦材料。
- 键盘不是末端 IK，而是把 `W/S A/D Q/E/Z/X/T/G/C/V` 映射到 `joint_1` 到 `joint_6` 的小步进直控；默认 `--joint_step_scale 0.025`，嫌大可用 `0.015` 或 `0.01`。
- 夹爪 finger 是显式长方体 mesh，不使用 `Cube + xformOp:scale`，避免 reference/articulation 场景里显示成过大 cube。
- 默认开启 `--grasp_assist`。纯 Newton 接触夹取时，cube 容易在合爪过程中被两指慢慢挤出；assist 只在合爪且 cube 靠近夹爪中心时把 cube 保持在夹爪中心，松开夹爪后释放。

如果要观察纯 Newton 物理抓取，不使用辅助：

```bash
/isaac-sim/python.sh examples/examples_teleop/example25_teleop_keyboard_abb_irb1200_physical_simple_gripper_newton.py \
  --no-grasp_assist \
  --joint_step_scale 0.015 \
  --debug_joints
```

当前实测现象是：纯 Newton 物理下，简化 prismatic 夹爪能接触 cube，但抬起时可能夹不住，或者合爪过程中把 cube 往外挤出。这通常不是键盘问题，而是位置驱动夹爪、接触约束、摩擦约束、动态物体质量和 solver iteration 共同作用的结果。继续做纯物理路线时，应优先考虑更 Newton 友好的 primitive/compound box collider、速度/力控夹爪、接触反馈闭环，而不是只继续堆摩擦系数。

新版 example23 的无窗口控制器冒烟（无需 Kit）：

```bash
/isaac-sim/python.sh tools/test_example23_keyboard.py
```

### 11.6 为什么 IRB1200 会下坠，而 franka_ik 不会

`example22` 使用 Differential IK 计算机械臂关节目标，但 IK 本身只负责把末端位姿误差转换成关节目标，不等于重力补偿。如果机械臂资产开启了重力，并且关节驱动刚度、阻尼、力矩上限不够，挂上 Robotiq 2F-140 后，`joint_2`、`joint_3` 这类承重关节就可能出现下坠、晃动，或者按 `E` 抬末端时夹爪跟着抖。

`example01` 里的 `franka_ik` 看起来不会下坠，是因为它用的是 Isaac Lab 自带的 `FRANKA_PANDA_HIGH_PD_CFG`。这个配置在 `submodules/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py` 里已经做了两件事：

```python
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
```

也就是说，`franka_ik` 的重力确实是关掉的，只是这个设置不在 `example01_teleop_simple.py` 里，而是在 Franka 资产配置里。Arena 里的 `FrankaIKEmbodiment` 会拷贝这个 high-PD 配置来生成场景。

IRB1200 当前配置默认更接近真实动力学：机器人和 2F-140 夹爪会受重力影响。因此：

- 如果目的是先调遥操作链路、调 IK、调夹爪开合，可以临时给 ABB 资产也设置 `disable_gravity=True`，让它像 `franka_ik` 一样稳定。
- 如果目的是做真实抓取仿真，不建议直接关重力；应该保留重力，然后调大关节 `stiffness`、`damping`、`effort_limit`，并检查 2F-140 的质量、惯量、碰撞体和安装偏移。
- 夹爪越重、安装越靠外，末端力矩越大，IRB1200 小臂关节越容易下坠或振荡。

### 11.7 无窗口冒烟

键盘版：

```bash
/isaac-sim/python.sh examples/examples_teleop/example22_teleop_keyboard_abb_irb1200_robotiq_2f140_control.py \
  --viz none \
  --no-keep_open \
  --num_steps 2 \
  --debug_joints
```

SpaceMouse 版：

```bash
/isaac-sim/python.sh examples/examples_teleop/example21_teleop_spacemouse_abb_irb1200_robotiq_2f140_control.py \
  --viz none \
  --no-keep_open \
  --num_steps 2 \
  --debug_joints
```

这两个冒烟主要看环境能否创建、动作维度是否是 7、`finger_joint` 是否在关节列表里。夹爪挂载位置如果视觉上还需要更贴合 IRB1200 法兰，可以微调带夹爪 USD 里的 `robotiq_2f140_mount` 位姿，并同步更新转换脚本里的默认偏移。

---

## 十二、example39：ABB IRB1200 + PGC140，1 kg cube 纯物理抓取

[example39](example39_newton_standalone_abb_irb1200_pgc140_cube_stacking.py) 使用官方 PGC-140-50 URDF 转换的组合 USD，配明确建模的 40–90 mm 外扩指尖。**不 attach、不写 cube 位姿、不锁腕；用有限夹力的接触摩擦搬运。** 已通过单块 1 kg cube、位置扰动、三块各 1 kg 堆叠测试；每侧 1 N 的负对照按预期抓不起。完整说明见 [选型与验证记录](../../../../docs/ABB-IRB1200-PGC140-纯物理抓取.md)。

在已有容器内执行（不用创建 venv 或重建 Docker）：

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh examples/examples_teleop/example39_newton_standalone_abb_irb1200_pgc140_cube_stacking.py \
  --viewer gl --cube-count 1 --cube-mass 1 --num-frames 1200 --test
```

- 无窗口：改 `--viewer null`。
- 三块堆叠：改 `--cube-count 3 --num-frames 3600`，每块仍为 1 kg。
- 查看真实碰撞体：添加 `--show-colliders`。
- [组合 USD](assets/abb_irb1200_pgc140/combined/abb_irb1200_pgc140.usda) 与 [独立夹爪 USD](assets/abb_irb1200_pgc140/gripper/pgc140_offset_fingers.usda) 已生成。
- 生成脚本：[tools/convert_pgc140_to_abb_irb1200_usd.py](../../tools/convert_pgc140_to_abb_irb1200_usd.py)。
- 资产检查：[tools/validate_pgc140_assets.py](../../tools/validate_pgc140_assets.py)。

注意：PGC140 是厂商推荐负载 3 kg 的候选，不意味着原装指尖能直接夹任意 1 kg 瓶子。本次指尖是仿真派生设计，瓶身尺寸、摩擦和瓶壁抗压还需验证。旧例的全臂外部 PD、世界坐标补偿、mimic 和 hydroelastic 参数不适用于新例。

---

## 十三、example40：ABB IRB1200 + Robotiq 2F-140，原生单主动轴联动抓取

[example40](example40_newton_abb_irb1200_robotiq_2f140_physical_grasp.py) 使用新下载的 ROS-Industrial 2F-140 URDF，保留**六个转动关节、五条原生 mimic、原装指腹**。先转 [夹爪 USD](assets/abb_irb1200_robotiq_2f140_urdf/gripper/robotiq_2f140.usda)，再与本地 ABB-only USD [引用组合](assets/abb_irb1200_robotiq_2f140_urdf/assembly.usda)，得到 [完整机器人 USD](assets/abb_irb1200_robotiq_2f140_urdf/abb_irb1200_robotiq_2f140.usda)。不替换为 prismatic，不覆盖 example39。

在已有 Docker 容器内：

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh examples/examples_teleop/example40_newton_abb_irb1200_robotiq_2f140_physical_grasp.py \
  --viewer gl --cube-mass 1 --num-frames 1200 --test
```

- 默认1 kg cube，只驱动夹爪 `finger_joint`，上限5 N·m；从动轴无电机，无 attach/物体位姿写入。
- 注册 Newton USD schema；分别设置原生 mimic 和接触软硬参数；用实测指腹中点补偿可变 TCP。
- 无窗口用 `--viewer null`；负对照加 `--grip-effort 0.05 --expect-grasp-failure --test`。
- [生成器](../../tools/convert_robotiq_2f140_urdf_to_abb_usd.py) 和 [五项资产/空载动态测试](../../tools/validate_robotiq_2f140_assets.py)。
- [完整中文说明](../../../../docs/ABB-IRB1200-Robotiq2F140-原生联动纯物理抓取.md) 包含来源、模型质量限制、失败对照、约束参数、运行方法及报告。

注意：上游是旧可视化模型，其质量、标称开口和理想 mimic 不能当成真实产品完整辨识模型；本次 cube 接触抓取验证不等于已经验证洗衣液瓶。

---

## 十四、example41：ABB IRB1200 + SFG N4049 三指连续体研究夹爪

[example41](example41_newton_abb_irb1200_sfg_n4049_soft_grasp.py) 将用户下载的 SRT IGS 转为 [组合 USD](assets/abb_irb1200_sfg_n4049/abb_irb1200_sfg_n4049.usda)，采用 **14,094 个四面体**模拟三根真正可变形的手指。动态 ABB、有限根部弹簧与反力回传、VBD 自由刚瓶接触；无绑定瓶体、无指尖位姿动画、无刚性指腹代理。

**物理网格是 CAD 凸包外包络近似，材料与主动参考曲率是假设，不是厂家气腔/压力模型。** 已通过 1 kg 圆柱瓶形物体抬升、保持、搬运、释放；不能据此断言真实 SRT 产品适合任意洗衣液瓶。

在已有 Docker 容器内运行：

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh examples/examples_teleop/example41_newton_abb_irb1200_sfg_n4049_soft_grasp.py \
  --viewer gl --test --quiet
```

- 无窗口：`--viewer null`；零驱动负对照：`--activation 0 --expect-grasp-failure --test`。
- 默认 40 子步 × 30 VBD 迭代；更细验证使用 `--substeps 60 --iterations 40`。
- [转换器](../../tools/convert_sfg_n4049_igs_to_abb_usd.py)；[资产/空载验证器](../../tools/validate_sfg_n4049_assets.py)；[CAD 依赖](../../tools/requirements_sfg_cad.txt)。运行已生成的例子不需要重新安装 CAD 依赖。
- VS Code 任务：**SFG N4049：Docker 内三指软体 1kg 纯接触抓取**。
- [完整记录](../../../../docs/ABB-IRB1200-SFG-N4049-三指连续体抓取.md) 包含宿主机一条命令、GUI/负对照报告、几何近似偏差、空载残余力及尚未标定的产品边界。
