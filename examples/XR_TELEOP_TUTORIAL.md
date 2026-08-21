# 使用 Quest 手柄遥操作 Franka 单臂

本教程说明如何在 Isaac Lab-Arena 中使用 Meta Quest Touch 手柄控制 Franka：

- Quest 右手柄的 6DoF 位姿控制机械臂末端；
- 右手柄 Trigger 控制夹爪闭合和打开；
- Isaac Lab 的 Differential IK 将末端位姿增量转换为 Franka 关节目标；
- 可将遥操作轨迹记录为模仿学习数据。

对应的完整示例是 [`example03_teleop_xr.py`](example03_teleop_xr.py)。

## 1. 系统结构

整个控制链如下：

```text
Quest Touch 右手柄
  │
  ├── grip pose: position + quaternion
  └── trigger: 0.0 ～ 1.0
          │
          ▼
OpenXR / CloudXR / IsaacTeleop
          │
          ├── Se3RelRetargeter
          │     └── [dx, dy, dz, rx, ry, rz]
          │
          └── GripperRetargeter
                └── +1.0（打开）或 -1.0（闭合）
          │
          ▼
Arena 7D action
[dx, dy, dz, rx, ry, rz, gripper]
          │
          ▼
DifferentialInverseKinematicsAction
          │
          ▼
Damped Least Squares IK
          │
          ▼
Franka 7 个关节的位置目标
```

这里分成两个不同阶段：

1. **Retargeting**：将人的手柄运动转换成机器人末端增量；
2. **IK**：将末端增量转换成机器人关节角目标。

OpenXR pipeline 不直接求关节角，真正的 IK 由 Isaac Lab ActionTerm 完成。

## 2. 环境准备

项目使用仓库内的 `.venv`：

```bash
cd /home/magengyu/IsaacLab-Arena
.venv/bin/python --version
```

需要保证 Quest 与运行 Isaac Sim 的主机网络互通。CloudXR 常用端口包括：

```bash
sudo ufw allow 49100/tcp
sudo ufw allow 47998/udp
sudo ufw allow 48322/tcp
```

如果系统不使用 `ufw`，请在对应防火墙中开放这些端口。

## 3. 启动 CloudXR

在第一个终端运行：

```bash
cd /home/magengyu/IsaacLab-Arena
.venv/bin/python -m isaacteleop.cloudxr --host-client
```

第一次启动会要求接受 NVIDIA CloudXR EULA。CloudXR 启动后会生成环境配置：

```text
~/.cloudxr/run/cloudxr.env
```

不要在 CloudXR 启动前加载这个文件。

## 4. 启动 Arena 示例

打开第二个终端：

```bash
cd /home/magengyu/IsaacLab-Arena
source ~/.cloudxr/run/cloudxr.env
.venv/bin/python examples/example03_teleop_xr.py --xr
```

`--xr` 由 Isaac Lab `AppLauncher` 提供，用于启动 XR 扩展和 XR Session。示例会使用：

```python
app_launcher = AppLauncher(args_cli)
```

该调用必须发生在导入大部分 Isaac Sim、USD 和机器人模块之前。

## 5. 连接 Quest

1. 查询运行 Arena 的主机局域网 IP，例如 `ip -brief address` 输出的
   `192.168.0.2`。不要选择 `docker0`、`lo` 或 Quest Link 创建的虚拟网卡地址。
2. 在 Quest 浏览器打开 `https://<主机IP>:48322/client/`。
3. 接受该页面的自签名证书警告。
4. 在客户端页面确认服务器地址为同一个主机 IP，然后点击 **Connect**。
5. 在 Isaac Sim 的 **XR** 标签页启动 XR Session。
6. 在 Quest 控制面板点击 **Play**。

使用 `--host-client` 可以让 CloudXR 启动器提供与已安装 `isaacteleop`
版本匹配的网页客户端，避免公共客户端更新后与本地 Runtime 出现协议或参数不兼容。

如果机器人运动方向与操作者不一致，在 Quest 中面向正前方并长按 Meta/Oculus 键重置视角。

## 6. 创建 OpenXR 遥操作设备

示例从 Arena 设备注册表取得 `openxr`：

```python
device_registry = DeviceRegistry()
teleop_device = device_registry.get_device_by_name("openxr")(
    sim_device=builder_cfg.device
)
```

然后将设备与 `franka_ik` embodiment 一起传给环境：

```python
environment = IsaacLabArenaEnvironment(
    name="franka_kitchen_pickup",
    embodiment=embodiment,
    scene=scene,
    task=task,
    teleop_device=teleop_device,
)
```

`ArenaEnvBuilder` 根据下面的注册键寻找 retargeter：

```text
openxr__franka_ik
```

注册代码位于：

```text
isaaclab_arena/assets/retargeter_library.py
```

它最终加载：

```text
isaaclab_arena/teleop/single_arm_openxr_pipeline.py
```

## 7. 手柄 6DoF Retargeting

管线通过 `ControllersSource` 读取左右 Quest 控制器：

```python
controllers = ControllersSource(name="controllers")
```

本示例只使用右手柄：

```python
ee_delta = Se3RelRetargeter(
    Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        delta_pos_scale_factor=10.0,
        delta_rot_scale_factor=10.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
    ),
    name="right_controller_ee_delta",
)
```

`Se3RelRetargeter` 保存上一帧手柄位姿，并计算：

```text
当前位置 - 上一帧位置
当前旋转 × 上一帧旋转的逆
```

输出是：

```text
[dx, dy, dz, rx, ry, rz]
```

其中旋转使用 axis-angle rotation vector，不是欧拉角。

参数含义：

- `delta_pos_scale_factor`：手柄平移增量放大倍数；
- `delta_rot_scale_factor`：手柄旋转增量放大倍数；
- `alpha_pos`：平移低通滤波系数；
- `alpha_rot`：旋转低通滤波系数；
- `zero_out_xy_rotation=False`：保留 roll、pitch、yaw 三轴旋转。

如果控制过于灵敏，可以把 scale factor 降到 `3.0～5.0`；如果运动抖动，可以把 alpha 降到 `0.2～0.4`。

## 8. Trigger 控制夹爪

夹爪由 IsaacTeleop 的 `GripperRetargeter` 控制：

```python
gripper = GripperRetargeter(
    GripperRetargeterConfig(
        hand_side="right",
        controller_threshold=0.5,
    ),
    name="right_controller_gripper",
)
```

映射关系是：

```text
Trigger > 0.5  → -1.0 → 闭合
Trigger ≤ 0.5  → +1.0 → 打开
```

Franka 的二值夹爪 ActionTerm 将其转换为手指关节目标：

```python
gripper_action = BinaryJointPositionActionCfg(
    joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},
    close_command_expr={"panda_finger_.*": 0.0},
)
```

因此 Trigger 并不直接表示夹爪宽度，而是二值开关。如果需要模拟模拟量夹爪，需要新增一个比例式 gripper retargeter 和连续关节 ActionTerm。

## 9. 7D Action 的组装

`TensorReorderer` 将末端增量和夹爪命令组合成：

```text
[dx, dy, dz, rx, ry, rz, gripper]
```

前 6 维被 `arm_action` 消费，最后 1 维被 `gripper_action` 消费。

遥操作循环中调用：

```python
action = teleop_interface.advance()
if action is not None:
    env.step(action.repeat(env.unwrapped.num_envs, 1))
```

XR Session 尚未开始或控制器暂时没有数据时，`advance()` 可能返回 `None`，因此必须先检查。

IsaacTeleop 设备还必须作为 context manager 使用：

```python
with teleop_interface:
    while simulation_app.is_running():
        ...
```

这样才能正确启动和关闭 DeviceIO、OpenXR tracker 与消息通道。

## 10. Differential IK

`franka_ik` 使用：

```python
DifferentialInverseKinematicsActionCfg(
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    ),
    scale=0.5,
)
```

配置位于：

```text
isaaclab_arena/embodiments/franka/franka.py
```

动作进入 ActionTerm 后首先乘以 `scale=0.5`，然后叠加到当前末端位姿。IK 控制器计算末端误差：

```text
Δx = [position_error, axis_angle_error]
```

再通过 Damped Least Squares 求解：

\[
\Delta q =
J^T \left(JJ^T + \lambda^2 I\right)^{-1}\Delta x
\]

目标关节位置为：

\[
q_{\text{desired}} = q_{\text{current}} + \Delta q
\]

最后写入 Franka articulation 的关节位置目标。DLS 相比直接 Jacobian 伪逆，在接近奇异位形时更稳定。

## 11. 坐标系

Quest 控制器位姿最初处于 XR 世界坐标系。Arena 将其转换到 Franka base frame：

```python
def get_teleop_target_frame_prim_path(self) -> str:
    return "/World/envs/env_0/Robot/panda_link0"
```

这个 prim path 位于：

```text
isaaclab_arena/embodiments/franka/franka.py
```

OpenXR pipeline 中的 `world_T_anchor` 用于执行坐标变换：

```python
transform_input = ValueInput("world_T_anchor", TransformMatrix())
transformed_controllers = controllers.transformed(
    transform_input.output(ValueInput.VALUE)
)
```

若机械臂运动方向错误，应优先检查：

1. XR view 是否已重新居中；
2. `get_teleop_target_frame_prim_path()` 是否指向正确的机器人基座；
3. 控制器是否使用 transformed output；
4. USD 中机器人 base prim 的实际路径。

## 12. 录制训练数据

如果要将 Quest 遥操作用于模仿学习，可以运行：

```bash
source ~/.cloudxr/run/cloudxr.env
mkdir -p datasets/franka_xr

.venv/bin/python \
  isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --xr \
  --viz kit \
  --device cpu \
  --enable_cameras \
  --dataset_file datasets/franka_xr/recorded.hdf5 \
  --num_demos 20 \
  --num_success_steps 10 \
  kitchen_pick_and_place \
  --embodiment franka_ik \
  --object cracker_box
```

Arena 会在看到 `--xr` 且没有显式选择其他设备时自动选择 `openxr`。

录制前应确认：

- 相机没有黑帧；
- Trigger 动作已记录到最后一个 action 维度；
- 每条成功轨迹完成了完整任务；
- 训练和回放使用同一个 embodiment；
- 控制频率和数据集 FPS 一致。

## 13. 常见问题

### Quest 能看到画面，但机械臂不动

检查：

- 第二个终端是否执行了 `source ~/.cloudxr/run/cloudxr.env`；
- 命令是否包含 `--xr`；
- XR 标签页中 Session 是否已经启动；
- Quest 客户端是否点击了 **Play**；
- 控制器是否被 Quest 正常追踪。

### `Media connection could not be established`

这表示 WebSocket 信令可能已经成功，但 WebRTC 的 UDP 媒体通道没有建立：

1. 确认 CloudXR 使用 `--host-client` 启动，并刷新
   `https://<主机IP>:48322/client/`，避免复用旧页面；
2. 确认 Quest 和主机在同一个局域网，并使用主机的物理网卡 IP；
3. 确认主机防火墙允许 `47998/udp`、`49100/tcp` 和 `48322/tcp`；
4. 暂时关闭 VPN、代理和 Quest Link/Air Link，排除虚拟网卡干扰；
5. 若主机与 Quest 之间存在 NAT，需要配置 ICE/STUN 或端口转发，只有 TCP
   代理可访问并不足以传输媒体。

可以检查最新 CloudXR 日志：

```bash
rg -i "error|fail|ice|media|47998" ~/.cloudxr/logs/*.log
```

### Trigger 没有控制夹爪

检查 `GripperRetargeterConfig` 是否连接到：

```python
ControllersSource.RIGHT
```

并确认输出顺序中 `gripper` 是第 7 维。

### 机械臂移动太快

降低：

```python
delta_pos_scale_factor
delta_rot_scale_factor
```

也可以降低 `FrankaIKActionCfg` 中的 action `scale`。

### 机械臂抖动

- 降低 retargeting scale；
- 降低 `alpha_pos` 和 `alpha_rot`；
- 提高仿真帧率；
- 降低 XR 渲染分辨率；
- 避免网络延迟和控制器追踪丢失。

### 接近奇异位形时动作异常

当前 IK 已使用 DLS，但仍应避免：

- 手臂完全伸直；
- 腕部姿态突然翻转；
- 单帧大角度旋转；
- 超出 Franka 工作空间的目标。

## 14. 关键文件

| 功能 | 文件 |
|---|---|
| 完整 XR 示例 | `examples/example03_teleop_xr.py` |
| 单臂 OpenXR pipeline | `isaaclab_arena/teleop/single_arm_openxr_pipeline.py` |
| `--xr` 自动设备选择 | `isaaclab_arena/teleop/cli.py` |
| OpenXR/Franka retargeter 注册 | `isaaclab_arena/assets/retargeter_library.py` |
| OpenXR 设备配置 | `isaaclab_arena/assets/device_library.py` |
| Franka IK 与夹爪 ActionTerm | `isaaclab_arena/embodiments/franka/franka.py` |
| 通用遥操作入口 | `isaaclab_arena/scripts/imitation_learning/teleop.py` |
| 示范录制入口 | `isaaclab_arena/scripts/imitation_learning/record_demos.py` |
| Isaac Lab IK ActionTerm | `submodules/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py` |
| DLS IK 实现 | `submodules/IsaacLab/source/isaaclab/isaaclab/controllers/differential_ik.py` |
