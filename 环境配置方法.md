# IsaacLab-Arena 环境配置方法

## 1. 基础环境：uv sync

首先使用 uv 建立虚拟环境并安装依赖：

```bash
uv sync
```

默认会安装 `isaaclab-from-source` 和 `openpi` 两个依赖组。如需使用 wheel 版本：

```bash
uv sync --no-default-groups --group isaaclab-from-wheel
```

## 2. GR00T 客户端支持

`uv sync` 不会安装 `gr00t` 包（因为 GR00T 要求 `python==3.10.*` 与 Arena 的 `>=3.12,<3.13` 冲突，且 GR00T 未写入 Arena 的 `pyproject.toml` 依赖）。

在 `uv sync` 完成后，执行以下两步添加 GR00T 客户端支持：

```bash
# 1. 以 no-deps 方式安装 gr00t 包（仅安装包本体，不拉取训练依赖）
uv pip install --no-deps -e submodules/Isaac-GR00T/

# 2. 安装 gr00t 客户端通信所需的网络库
uv pip install pyzmq msgpack msgpack-numpy
```

`--no-deps` 是关键：避免因 GR00T 的 python==3.10 要求导致依赖解析失败，也避免拉入 torch、deepspeed、flash-attn 等训练依赖（policy_runner 作为客户端不需要这些）。

验证安装：

```bash
.venv/bin/python -c "from gr00t.policy.server_client import PolicyClient; print('OK')"
.venv/bin/python -c "from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicy; print('OK')"
```

## 3. 运行 GR00T Policy

确保 GR00T policy server 已在运行，然后执行 `policy_runner.py`：

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --viz kit \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --language_instruction "Pick up the Rubik's cube and place it in the bowl." \
  --enable_cameras \
  --num_episodes 3 \
  pick_and_place_maple_table \
  --embodiment droid_abs_joint_pos \
  --pick_up_object rubiks_cube_hot3d_robolab \
  --destination_location bowl_ycb_robolab \
  --hdr home_office_robolab
```

## 注意事项

- 每次 `uv sync` 后**不需要**重新执行 `uv pip install` 步骤 — 安装的包在 `.venv` 中持久存在
- 如果执行了 `uv sync --reinstall` 或删除了 `.venv` 重建，需重新执行 GR00T 客户端安装步骤
