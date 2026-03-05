# Remote Policy v2 设计文档

**版本**: v4 | **日期**: 2026-03-05
**作者**: Hui Kang

---

## 1. v1 → v2 总览

### 1.1 v1 架构与问题

```
v1 架构：
┌──────────────────────────┐
│  Policy Layer            │  ServerSidePolicy（全局单例状态）
├──────────────────────────┤
│  RPC Layer               │  endpoint dispatch（无 client 识别）
├──────────────────────────┤
│  Serialization Layer     │  msgpack + numpy（无压缩）
├──────────────────────────┤
│  Transport Layer         │  ZMQ REQ/REP（串行，1 client）
└──────────────────────────┘
```

**v1 的 5 个问题**：


| #   | 问题               | 具体表现                                                                        |
| --- | ---------------- | --------------------------------------------------------------------------- |
| P1  | **无法识别请求来源**     | server 不知道是哪个 client 发的，无法做 per-client 状态隔离                                 |
| P2  | **状态全局共享**       | `_instruction` / `_image_history` / `_task_description` 是全局单例，多 client 互相覆盖 |
| P3  | **无压缩**          | 大 payload（Droid 双相机 N=10 ~53MB）传输慢                                          |
| P4  | **GPU→CPU 强制转换** | 即使 client/server 同机，observation 必须 `tensor.cpu().numpy()` → 序列化 → 反序列化      |
| P5  | **无传输层抽象**       | ZMQ 硬编码，无法切换到 UCX 零拷贝                                                       |


**P2 的具体 Bug**（影响 NaVILA 和 GR00T）：

```python
# 两个 client 同时运行时：
# t=0  Client A → set_task_description("Go to the kitchen")
# t=1  Client B → set_task_description("Go to the bedroom")  ← 覆盖 A！
# t=2  Client A → get_action(obs)   ← 用的是 "Go to the bedroom"
```

### 1.2 v2 解决的问题


| v1 问题           | v2 方案                                     | 文档章节       |
| --------------- | ----------------------------------------- | ---------- |
| P1 无法识别来源       | ZMQ ROUTER/DEALER，identity 自动识别           | §3         |
| P2 状态全局共享       | `ClientState` per-client 隔离，`env_ids` 索引  | §3.3, §4   |
| P3 无压缩          | lz4 CPU 压缩 / nvcomp GPU 压缩                | §5.8, §6   |
| P4 GPU→CPU 强制转换 | UCX 零拷贝（GPU→GPU，不经 CPU）                   | §5.3, §5.7 |
| P5 无传输层抽象       | `ServerTransport` / `ClientTransport` ABC | §5.1       |


### 1.3 v1 vs v2 架构对比

```
v1 流程：

Client (REQ)                              Server (REP)
    │                                         │
    │ obs.cpu().numpy()                       │
    │ → msgpack serialize                     │
    │ → zmq.send(bytes) ───────────────────>  │ zmq.recv()
    │                                         │ msgpack deserialize
    │                                         │ policy.get_action(numpy_obs)
    │                                         │   self._instruction  ← 全局！
    │ zmq.recv() <────────────────────────    │ zmq.send(action)
    │                                         │
    │  ⚠️ 同一时刻只能 1 个 client            │
    │  ⚠️ server 不知道是谁发的               │
    │  ⚠️ 没有压缩                            │
    │  ⚠️ GPU→CPU→GPU 强制转换                │
```

```
v2 流程（ZMQ + UCX 零拷贝模式）：

Client A (DEALER)    Client B (DEALER)    Server (ROUTER)
    │                     │                    │
    │ connect()           │ connect()          │ bind()
    │                     │                    │
    │  ─── get_init_info (ZMQ) ──────────────> │ 协商 transport/compression
    │  <── resp: ucx_port=13337 ────────────── │ 创建 ClientState(A)
    │  ─── UCX connect ──────────────────────> │ UCX accept
    │                     │                    │
    │                     │ ── get_init_info ─> │ 创建 ClientState(B)
    │                     │ <── resp: zmq ──── │ (B 无 UCX，用 ZMQ)
    │                     │                    │
    │  [get_action]       │                    │
    │  ZMQ: {env_ids,     │                    │ ZMQ recv → 知道是 A
    │        has_tensor,   │                    │
    │        tensor_layout}│                    │
    │  UCX: obs_gpu ──────┼──────────────────> │ UCX recv → GPU buffer
    │    (零拷贝,可nvcomp) │                    │ nvcomp decompress (GPU)
    │                     │                    │ policy.get_action(gpu_obs,
    │                     │                    │   env_ids, client_state_A)
    │  <── ZMQ: action ───┼──────────────────  │
    │                     │                    │
    │                     │ [get_action]       │
    │                     │ ZMQ: {env_ids,     │ ZMQ recv → 知道是 B
    │                     │   observation}     │ lz4 decompress (CPU)
    │                     │ ─────────────────> │ policy.get_action(np_obs,
    │                     │                    │   env_ids, client_state_B)
    │                     │ <── ZMQ: action ── │
```

### 1.4 v2 协议栈

```
┌──────────────────────────┐
│  Policy Layer            │  ServerSidePolicy + ClientState per-client
├──────────────────────────┤
│  RPC Layer               │  endpoint dispatch + client_id + env_ids
├──────────────────────────┤
│  Serialization Layer     │  msgpack + lz4(CPU) / nvcomp(GPU) 压缩
├──────────────────────────┤
│  Transport Layer         │  ZMQ ROUTER/DEALER (控制) + UCX (数据, 可选)
└──────────────────────────┘
```

---

## 2. 问题详述

### P6 详述：Per-Client 状态单例 Bug（影响所有现有 policy）

**受影响的 policy**：


| Policy                          | 全局单例状态                                    | Bug 表现        |
| ------------------------------- | ----------------------------------------- | ------------- |
| **NaVilaServerPolicy**          | `self._instruction`、`self._image_history` | 指令互相覆盖，图像历史混合 |
| **Gr00tRemoteServerSidePolicy** | `self._task_description`                  | 指令互相覆盖        |
| **未来 policy**                   | 任何 `self._xxx` 状态                         | 相同问题          |


```python
# v1 navila_server_policy.py __init__：
self._instruction: str = "Navigate to the target."  # ← 全局单例
self._image_history: list = []                       # ← 全局单例

# v1 gr00t_remote_policy.py __init__（同样的问题）：
self._task_description: str | None = None            # ← 全局单例

# 时序示意（2 个 eval env 同时运行时）：
# t=0  Client A → set_task_description("Go to the kitchen")
# t=1  Client B → set_task_description("Go to the bedroom")  ← 覆盖 A 的指令！
# t=2  Client A → get_action(obs)   ← 实际执行的是 "Go to the bedroom"
# t=3  Client B → get_action(obs)   ← NaVILA: image_history 混有 A 的帧
#                                      GR00T: task_description 被 A 覆盖
```

**修复方案**：见第 4 节（per-client 状态隔离）。

### P6 补充：连接生命周期与 GC

```
Client 连接 (DEALER.connect)
    → ZMQ 自动分配 identity (bytes) 作为 client_id
    → 无需显式注册

Client 发 get_init_info
    → Server 在 per-client 状态 dict 中创建条目

Client 运行多个 episode（reset → get_action 循环）

Client 断开 (DEALER.close / 进程退出)
    → ZMQ identity 不再有效
    → Server 定期扫描：超过 idle_timeout_s（默认 300s）无请求的条目自动删除
    → 旧 identity 占用内存直到 GC 扫描到，不影响其他 client
```

---

## 3. v1 vs 改造后流程对比

### 3.1 v1 REQ/REP 流程

```
Client (zmq.REQ)                          Server (zmq.REP)
─────────────────────────────────────────────────────────────────
                                          sock = ctx.socket(zmq.REP)
                                          sock.bind("tcp://*:5555")

sock = ctx.socket(zmq.REQ)
sock.connect("tcp://server:5555")

# 每次 get_action:
raw = MessageSerializer.to_bytes(obs)
sock.send(raw)                    ──────>  raw = sock.recv()
                                           msg = from_bytes(raw)
                                           result = policy.get_action(msg)
                                           sock.send(to_bytes(result))
resp = sock.recv()                <──────
# 必须等上面完成才能发下一条

# 第二个 Client 尝试同时连接：
# → sock.send() 挂起，等 REP server 空闲
# → REP 协议：同一时刻只服务一个 client
```

**代码位置**：

- `isaaclab_arena/remote_policy/policy_server.py` — server 主循环
- `isaaclab_arena/remote_policy/policy_client.py` — client 发包

---

### 3.2 改造后 ROUTER/DEALER 流程

ROUTER 的核心机制：每条收到的消息帧里自带 **sender identity**（ZMQ 自动附加），
server 回复时把这个 identity 填回帧头，ZMQ 自动路由到对应的 DEALER。

> **⚠️ DEALER 帧格式注意**：ZMQ REQ 发送时自动附加空分隔帧 `b""`，但 DEALER
> **不会**自动附加。因此 DEALER 客户端必须**显式发送** `send_multipart([b"", payload])`
> 来保持与 ROUTER 期望的 3-part 帧结构一致。否则 ROUTER 只收到 2 parts，导致解析错误。

```
时间轴 ──────────────────────────────────────────────────────────────────>

Client A (DEALER)   Client B (DEALER)   Server (ROUTER)
────────────────    ────────────────    ──────────────────────────────────
connect()                               # ZMQ 分配 A 的 identity = b"\x01"
                    connect()           # ZMQ 分配 B 的 identity = b"\x02"

send_multipart      ←── 注意：必须发 [b"", payload]，不是 send(payload)
([b"", req_A])──────────────────────> recv_multipart()
                                        # → [b"\x01", b"", req_A]    ✅ 3 parts
                                        #   知道是 A 发的，取出 A 的状态
                    send_multipart
                    ([b"", req_B])────> recv_multipart()
                                        # → [b"\x02", b"", req_B]
                                        #   B 的请求进入 server 内部队列
                                        #   server 先处理 A（串行）：
                                        send_multipart([b"\x01", b"", res_A])
recv_multipart()                         # DEALER 收到 [b"", res_A]
# → [b"", res_A]  <────────────────────
# 取 parts[1] 作为 payload
                                        # 再处理 B：
                                        send_multipart([b"\x02", b"", res_B])
                    recv_multipart()
                    # → [b"", res_B] <──
```

**DEALER 客户端发送/接收代码**：

```python
# policy_client.py — 改造后
sock = ctx.socket(zmq.DEALER)
sock.connect("tcp://server:5555")

# 发送（必须显式加空分隔帧）
sock.send_multipart([b"", payload])

# 接收（DEALER 收到的帧带有空分隔帧前缀）
parts = sock.recv_multipart()
response = parts[1]  # parts[0] 是 b""
```

**v1 vs 改造后的关键差异**：


|                 | v1 REQ/REP         | 改造后 ROUTER/DEALER                  |
| --------------- | ------------------ | ---------------------------------- |
| 多 client 连接     | ✅ 可以（ZMQ 队列）       | ✅ 可以                               |
| 请求处理方式          | 串行（与 v1 相同）        | 串行（单线程） / 可扩展为线程池                  |
| server 知道是谁发的   | ❌ 不知道              | ✅ identity 自动附带                    |
| per-client 状态隔离 | ❌ 做不到              | ✅ `client_id` 为 key + `env_ids` 索引 |
| client 发送方式     | `send(payload)`    | `send_multipart([b"", payload])`   |
| client 接收方式     | `recv()` → payload | `recv_multipart()` → `parts[1]`    |


---

### 3.3 Per-Client 状态结构

`**client_id` 做 dict key，每个 client 的状态预分配 `num_envs` 大小的数组。**

`num_envs` 在 `get_init_info` 握手时由 client 告知 server：

```python
# get_init_info 请求（新增 num_envs）
{
    "endpoint": "get_init_info",
    "data": {
        "requested_action_mode": "vln_velocity",
        "num_envs": 4,           # ← client 告诉 server 有几个 env
    }
}
```

**Server 侧状态结构**：

```python
@dataclass
class ClientState:
    """Per-client 状态，数组按 num_envs 预分配。"""
    num_envs: int
    instructions: list[str | None]       # [num_envs]，每个 env 的指令
    image_histories: list[list]          # [num_envs]，每个 env 的帧历史（仅 VLN）

    @classmethod
    def create(cls, num_envs: int) -> ClientState:
        return cls(
            num_envs=num_envs,
            instructions=[None] * num_envs,
            image_histories=[[] for _ in range(num_envs)],
        )

# PolicyServer 状态：client_id → ClientState
self._client_states: dict[bytes, ClientState] = {}

# get_init_info 时创建
def _handle_get_init_info(self, client_id, data):
    num_envs = data.get("num_envs", 1)
    self._client_states[client_id] = ClientState.create(num_envs)
    ...
```

**为什么不用 `(client_id, env_id)` 二元组做 key？**

- 二元组做 key 意味着 N 次 dict 查找 → 慢
- 正确做法：`client_id` 做 key 取到 `ClientState`，再用 `env_ids` 直接数组索引
- tensor 形状的观测和动作本身就是 `[N, ...]`，天然对齐

**Client 发送 batched tensor（一次 RPC 包含所有活跃 env）**：

```python
# client 端发 ONE 条消息，observation 里的 tensor 已经是 batched 的
request = {
    "endpoint": "get_action",
    "data": {
        "env_ids": [0, 1, 2, 3],           # list[int]，哪些 env 需要 action
        "observation": {
            "camera_obs.robot_head_cam_rgb": rgb_batch,   # [N, H, W, C] uint8
            "policy.robot_joint_pos": joints_batch,       # [N, J] float64
        },
    }
}

# 部分 env 查询（pipeline 推理场景）：
request["data"]["env_ids"] = [0, 3, 5]
request["data"]["observation"]["camera_obs.robot_head_cam_rgb"] = rgb_batch[[0,3,5]]
```

**Server 收到后的处理**：

```python
client_id, raw = self._transport.recv()
msg = from_bytes(raw)
env_ids = msg["data"]["env_ids"]

# 1 次 dict 查找拿到该 client 的全部状态
client_state = self._client_states[client_id]

# 传给 policy（batched tensor + 整个 client_state）
action, info = self._policy.get_action(
    observation=msg["data"]["observation"],  # batched [N, ...]
    env_ids=env_ids,                          # 哪些 env
    client_state=client_state,                # 整个 ClientState 对象
)

# 回复（action 也是 batched）
response = {"env_ids": env_ids, "action": action_batch}
```

**为什么一次发 batched tensor 而不是循环发 N 条？**

- **性能**：1 次 ZMQ roundtrip（~~1ms）vs N 次（~~Nms）
- **批推理**：GR00T 天然支持 batched input，一次 forward pass 处理 N 个 env
- **与 Arena 一致**：`pack_observation_for_server()` 已经返回 `[N, H, W, C]`

---

## 4. Per-Client 状态隔离

### 4.1 PolicyServer 改造（在现有 policy_server.py 基础上升级）

**核心改动**：socket 从 `zmq.REP` 改为 `zmq.ROUTER`，主循环读取 identity：

```python
# policy_server.py — 改造前（v1）
class PolicyServer:
    def __init__(self, policy: ServerSidePolicy, port: int = 5555):
        self._policy = policy          # 单例
        self._sock = ctx.socket(zmq.REP)  # ← 改为 ROUTER
        self._sock.bind(f"tcp://*:{port}")

    def run(self):
        while True:
            raw = self._sock.recv()                         # ← 无 identity
            msg = MessageSerializer.from_bytes(raw)
            result = self._dispatch(msg)
            self._sock.send(MessageSerializer.to_bytes(result))

# policy_server.py — 改造后（多客户端）
class PolicyServer:
    def __init__(self, policy: ServerSidePolicy, port: int = 5555,
                 idle_timeout_s: float = 300.0):
        self._policy = policy              # 共享（模型权重）
        self._client_states: dict[tuple[bytes, int], dict] = {}  # per-(client,env) 状态
        self._last_seen: dict[bytes, float] = {}  # GC 用，粒度到 client
        self._idle_timeout_s = idle_timeout_s
        self._sock = ctx.socket(zmq.ROUTER)  # ← 关键改动
        self._sock.bind(f"tcp://*:{port}")

    def run(self):
        while True:
            self._gc_stale_clients()
            parts = self._sock.recv_multipart()
            client_id, _, raw = parts[0], parts[1], parts[2]
            self._last_seen[client_id] = time.time()

            msg = MessageSerializer.from_bytes(raw)
            result = self._dispatch(msg, client_id)

            self._sock.send_multipart([client_id, b"", MessageSerializer.to_bytes(result)])

    def _gc_stale_clients(self):
        now = time.time()
        stale = [cid for cid, t in self._last_seen.items()
                 if now - t > self._idle_timeout_s]
        for cid in stale:
            # _client_states key 是 (client_id, env_id) 二元组，
            # 需要遍历删除该 client 的所有 env 状态
            for key in list(self._client_states):
                if key[0] == cid:
                    del self._client_states[key]
            self._last_seen.pop(cid, None)
            print(f"[PolicyServer] GC stale client {cid.hex()}")

    def get_client_state(self, client_id: bytes) -> ClientState:
        """获取 per-client 状态。在 get_init_info 时已创建。

        PolicyServer 是状态的唯一持有者。Policy 方法通过传入的
        ClientState 引用读写状态，用 env_ids 索引具体 env。
        """
        return self._client_states[client_id]
```

### 4.2 状态所有权：PolicyServer 持有，Policy 通过参数访问

**设计决策**：`_client_states` 由 `PolicyServer` 统一持有和 GC。Policy 方法
通过 `client_state: dict` 参数接收当前客户端的状态引用，直接读写该 dict。

这样做的好处：

- GC 逻辑集中在 PolicyServer，policy 不需要关心客户端生命周期
- Policy 代码只需将 `self._instruction` 换成 `client_state["instruction"]`
- 新增 policy 实现时不需要复制 GC 逻辑

**ServerSidePolicy ABC 签名变更**（**breaking change**，影响所有现有 policy）：

```python
# server_side_policy.py — 改造后
class ServerSidePolicy(ABC):

    @abstractmethod
    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        env_ids: list[int] | None = None,         # ← 新增：哪些 env
        client_state: ClientState | None = None,   # ← 新增：该 client 的完整状态
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    def set_task_description(
        self,
        task_description: str | None,
        *,
        env_ids: list[int] | None = None,
        client_state: ClientState | None = None,
    ) -> dict[str, Any]:
        # 默认：为 env_ids 中每个 env 设置相同的指令
        if client_state and env_ids:
            for eid in env_ids:
                client_state.instructions[eid] = task_description
        ...

    def reset(
        self,
        env_ids=None,
        reset_options=None,
        *,
        client_state: ClientState | None = None,
    ) -> dict[str, Any]:
        # 默认：清空指定 env 的状态
        if client_state and env_ids:
            for eid in env_ids:
                client_state.instructions[eid] = None
                client_state.image_histories[eid] = []
        ...
```

**⚠️ Breaking Change**：

v2 改了 `ServerSidePolicy` 的方法签名，**所有现有 policy 都必须更新**。
不兼容 v1，没有自动降级。

**迁移指南（给同事和第三方用户）**：

```python
# v1 签名（旧）：
def get_action(self, observation, options=None):
    ...

# v2 签名（新）——加 env_ids 和 client_state：
def get_action(self, observation, options=None, *,
               env_ids=None, client_state=None):
    # 不需要 per-env 状态的 policy（如 GR00T tabletop 单指令）
    # 可以忽略 env_ids 和 client_state，直接处理 batched observation。

    # 需要 per-env 状态的 policy（如 VLN 多指令）：
    if client_state and env_ids:
        for i, eid in enumerate(env_ids):
            instruction = client_state.instructions[eid]
            obs_i = observation["camera_obs.robot_head_cam_rgb"][i]
            ...
```

**PolicyServer dispatch 注入 client_states**：

```python
# policy_server.py — _dispatch() 改造
def _dispatch(self, msg, client_id):
    endpoint = msg["endpoint"]
    data = msg.get("data", {})
    env_ids = data.pop("env_ids", [0])

    # 1 次 dict 查找拿到该 client 的完整状态
    client_state = self.get_client_state(client_id)

    if endpoint == "get_action":
        action, info = self._policy.get_action(
            observation=data["observation"],   # batched [N, ...]
            options=data.get("options"),
            env_ids=env_ids,
            client_state=client_state,          # ClientState 对象
        )
        merged = {**action, **info, "env_ids": env_ids}
        return merged
    elif endpoint == "set_task_description":
        return self._policy.set_task_description(
            task_description=data.get("task_description"),
            env_ids=env_ids,
            client_state=client_state,
        )
    elif endpoint == "reset":
        return self._policy.reset(
            env_ids=env_ids,
            reset_options=data.get("options"),
            client_state=client_state,
        )
    # ... ping / kill 等无状态端点不传 client_state
```

### 4.3 NaVilaServerPolicy 改造示例

```python
# navila_server_policy.py — 改造后
class NaVilaServerPolicy(ServerSidePolicy):

    def get_action(self, observation, options=None, *,
                   env_ids=None, client_state=None):
        # env_ids 索引每个 env 的独立状态
        for i, eid in enumerate(env_ids):
            history = client_state.image_histories[eid]
            instruction = client_state.instructions[eid] or "Navigate to the target."
            rgb_i = observation["camera_obs.robot_head_cam_rgb"][i]

            img = Image.fromarray(rgb_i[:, :, :3].astype(np.uint8))
            history.append(img)
            if len(history) > self.config.max_history_frames:
                client_state.image_histories[eid] = history[-self.config.max_history_frames:]

            # VLM 推理（使用该 env 的 history 和 instruction）
            vlm_text = self._run_vlm_inference(history, instruction)
            vel_cmd, duration = parse_vlm_output_to_velocity(vlm_text)
            # ... 拼入 batched action ...

    def set_task_description(self, task_description, *,
                             env_ids=None, client_state=None):
        # 为每个指定 env 设置指令
        for eid in env_ids:
            client_state.instructions[eid] = task_description or "Navigate to the target."
        return {"status": "ok"}

    def reset(self, env_ids=None, reset_options=None, *, client_state=None):
        for eid in env_ids:
            client_state.image_histories[eid] = []
            client_state.instructions[eid] = None
        return {"status": "reset_success"}
```

### 4.4 Gr00tRemoteServerSidePolicy 改造示例

GR00T 也有同样的单例 bug（`self._task_description`），需要同步改造。

**v1 代码问题**（`gr00t_remote_policy.py`）：

```python
# v1 — self._task_description 是全局单例
class Gr00tRemoteServerSidePolicy(ServerSidePolicy):
    def __init__(self, config):
        ...
        self._task_description: str | None = None  # ← 全局单例

    def _build_policy_observations(self, observation, camera_names):
        assert self._task_description is not None  # ← 用的是全局 _task_description
        ...

    def set_task_description(self, task_description):
        self._task_description = task_description   # ← 被任意 client 覆盖
```

**改造后**：

```python
# gr00t_remote_policy.py — 改造后
class Gr00tRemoteServerSidePolicy(ServerSidePolicy):

    def get_action(self, observation, options=None, *,
                   env_ids=None, client_state=None):
        # GR00T 通常所有 env 共享同一个 instruction，直接取 env_ids[0] 的
        task_desc = (client_state.instructions[env_ids[0]]
                     if client_state else self.policy_config.language_instruction)
        policy_observations = self._build_policy_observations(
            observation, self.camera_names, task_description=task_desc
        )
        # batched forward pass（GR00T 天然支持）
        robot_action_policy, _ = self.policy.get_action(policy_observations)
        ...

    def set_task_description(self, task_description, *,
                             env_ids=None, client_state=None):
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if client_state and env_ids:
            for eid in env_ids:
                client_state.instructions[eid] = task_description
        return {"status": "ok"}

    def reset(self, env_ids=None, reset_options=None, *, client_state=None):
        # per-client 隔离：清 action buffer 和指令
        self.policy.reset()
        if client_state and env_ids:
            for eid in env_ids:
                client_state.instructions[eid] = None
        return {"status": "reset_success"}
```

**GR00T 与 NaVILA 的状态差异**：


| 状态                                 | NaVILA          | GR00T             | 需要 per-client 隔离？ |
| ---------------------------------- | --------------- | ----------------- | ----------------- |
| `instruction` / `task_description` | ✅ 每集不同          | ✅ 每 task 不同       | **是**             |
| `image_history`                    | ✅ 完整 episode 历史 | ❌ 无               | 是（仅 NaVILA）       |
| `policy.reset()`                   | N/A             | ✅ 清 action buffer | **是**（保守隔离）       |


> **注意**：上述是改造方向的伪代码，尚未实现。

---

## 5. 传输层设计

提供两种模式：**模式 1（ZMQ-only）** 默认，所有数据走 ZMQ 序列化传输；
**模式 2（ZMQ + UCX）** 零拷贝，GPU tensor 走 UCX，控制消息走 ZMQ。
两种模式在 `get_init_info` 握手时自动协商（§6.3），用户无需手动选择。

### 5.1 Transport ABC

Server 和 Client 各有一个 Transport。区别：Server 管多个 client 连接
（需要 `client_id` 路由），Client 只连一个 server（不需要 `client_id`）。

```python
from abc import ABC, abstractmethod
import torch


class ServerTransport(ABC):
    """Server 端传输。管理多个 client 连接，用 client_id 区分。"""

    @abstractmethod
    def send(self, client_id: bytes, payload: bytes) -> None: ...

    @abstractmethod
    def recv(self, timeout_ms: int = 5000) -> tuple[bytes, bytes]:
        """Returns (client_id, payload_bytes)."""
        ...

    def send_tensor(self, client_id: bytes, tensor: torch.Tensor) -> None:
        """发送 GPU tensor 给指定 client（默认序列化，UCX 子类覆盖为零拷贝）。"""
        self.send(client_id, tensor.cpu().numpy().tobytes())

    def recv_tensor(self, client_id: bytes, buffer: torch.Tensor) -> None:
        """从指定 client 接收 GPU tensor 到预分配 buffer。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None: ...


class ClientTransport(ABC):
    """Client 端传输。只连一个 server，不需要 client_id。"""

    @abstractmethod
    def send(self, payload: bytes) -> None: ...

    @abstractmethod
    def recv(self, timeout_ms: int = 5000) -> bytes: ...

    def send_tensor(self, tensor: torch.Tensor) -> None:
        """发送 GPU tensor 给 server（默认序列化，UCX 子类覆盖为零拷贝）。"""
        self.send(tensor.cpu().numpy().tobytes())

    @abstractmethod
    def close(self) -> None: ...
```

### 5.2 模式 1：ZMQ-only（默认）

所有 endpoint 的所有数据都走 ZMQ，observation 序列化为 bytes（可选 lz4 压缩）。

**Server 端**：

```python
class ZmqServerTransport(ServerTransport):
    def __init__(self, host="*", port=5555):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(f"tcp://{host}:{port}")

    def send(self, client_id, payload):
        self._sock.send_multipart([client_id, b"", payload])

    def recv(self, timeout_ms=5000):
        if not self._sock.poll(timeout_ms):
            raise TimeoutError
        parts = self._sock.recv_multipart()
        return parts[0], parts[2]    # (client_id, payload_bytes)

    def close(self):
        self._sock.close(0); self._ctx.term()
```

**Client 端**：

```python
class ZmqClientTransport(ClientTransport):
    def __init__(self, host, port):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.connect(f"tcp://{host}:{port}")

    def send(self, payload):
        self._sock.send_multipart([b"", payload])   # DEALER 必须显式加 b""

    def recv(self, timeout_ms=5000):
        parts = self._sock.recv_multipart()
        return parts[1]    # parts[0] 是 b""（空分隔帧），跳过

    def close(self):
        self._sock.close(0); self._ctx.term()
```

**完整调用链（从上层到底层）**：

```python
# ===== Server 端 =====
transport = ZmqServerTransport(port=5555)
server = PolicyServer(policy=my_gr00t_policy, transport=transport)

class PolicyServer:
    def run(self):
        while True:
            client_id, raw = self._transport.recv()      # ← ServerTransport.recv
            msg = MessageSerializer.from_bytes(raw)       # 反序列化（含自动解压）
            result = self._dispatch(msg, client_id)
            resp = MessageSerializer.to_bytes(result)     # 序列化（含可选压缩）
            self._transport.send(client_id, resp)         # ← ServerTransport.send

# ===== Client 端 =====
transport = ZmqClientTransport(host="server_ip", port=5555)
client = PolicyClient(transport=transport)

class PolicyClient:
    def call_endpoint(self, endpoint, data=None):
        request = {"endpoint": endpoint, "data": data or {}}
        payload = MessageSerializer.to_bytes(request, compression=self._compression)
        self._transport.send(payload)           # ← ClientTransport.send
        resp = self._transport.recv()            # ← ClientTransport.recv
        return MessageSerializer.from_bytes(resp)  # 反序列化（自动检测解压）

# 压缩在 MessageSerializer 层处理，Transport 只传 bytes，互不干扰。
# 序列化流程：dict → msgpack → (可选 lz4 compress) → bytes → Transport.send
# 反序列化流程：Transport.recv → bytes → (自动检测 decompress) → msgpack → dict
```

### 5.3 模式 2：ZMQ + UCX（GPU 零拷贝）

控制消息走 ZMQ，大 observation tensor 走 UCX 零拷贝。
UCX 是 CUDA IPC 的**超集**（同机自动用 shared memory / CUDA IPC，
跨机用 InfiniBand，无特殊硬件回退 TCP），不需要单独实现 CUDA IPC。
**UCX 通道不压缩**——GPU tensor 直接传，压缩反而增加开销。

**Server 端**：

```python
class ZmqUcxServerTransport(ServerTransport):
    def __init__(self, host="*", zmq_port=5555, ucx_port=13337):
        self._zmq = ZmqServerTransport(host, zmq_port)
        self._ucx_listener = ucp.create_listener(self._on_ucx_connect, ucx_port)
        self._ucx_eps: dict[bytes, ucp.Endpoint] = {}

    # 控制消息走 ZMQ
    def send(self, client_id, payload):  self._zmq.send(client_id, payload)
    def recv(self, timeout_ms=5000):     return self._zmq.recv(timeout_ms)

    # GPU tensor 走 UCX（用 client_id 找到对应的 UCX endpoint）
    def recv_tensor(self, client_id, buffer):
        asyncio.run(self._ucx_eps[client_id].recv(buffer))

    def send_tensor(self, client_id, tensor):
        asyncio.run(self._ucx_eps[client_id].send(tensor))
```

**Client 端**：

```python
class ZmqUcxClientTransport(ClientTransport):
    def __init__(self, host, zmq_port=5555, ucx_port=13337):
        self._zmq = ZmqClientTransport(host, zmq_port)
        self._ucx_ep = asyncio.run(ucp.create_endpoint(host, ucx_port))

    # 控制消息走 ZMQ
    def send(self, payload):          self._zmq.send(payload)
    def recv(self, timeout_ms=5000):  return self._zmq.recv(timeout_ms)

    # GPU tensor 走 UCX（Client 只有一个连接，不需要 client_id）
    def send_tensor(self, tensor):
        asyncio.run(self._ucx_ep.send(tensor))   # GPU → GPU 零拷贝
```

**完整调用链（get_action 零拷贝版）**：

```
Client (PolicyClient)                         Server (PolicyServer)
─────────────────                             ──────────────────────

[握手 — get_init_info]
client_transport.send(get_init_info_msg)      server_transport.recv → (client_id, msg)
                                              → 创建 ClientState(num_envs=4)
                                              → UCX accept → ucx_eps[client_id]
client_transport.recv ← resp                  server_transport.send(client_id, resp)

[get_action — 每步]
① client_transport.send(                      ① server_transport.recv → (client_id, msg)
     {"endpoint":"get_action",                    msg 里有 "has_tensor":true
      "env_ids":[0,1,2,3],                       知道接下来要从 UCX 收 tensor
      "has_tensor":true})

② client_transport.send_tensor(obs_gpu)       ② server_transport.recv_tensor(
     # GPU tensor → UCX → GPU buffer               client_id, gpu_buffer)
                                                    # 零拷贝写入预分配 buffer

                                              ③ policy.get_action(gpu_buffer, ...)
                                                 → action

③ client_transport.recv() ← action            ④ server_transport.send(client_id, action)
```

**PolicyServer 中 get_action 的处理逻辑（两种模式统一）**：

```python
def _dispatch(self, msg, client_id):
    endpoint = msg["endpoint"]
    data = msg.get("data", {})

    if endpoint == "get_action":
        env_ids = data["env_ids"]
        client_state = self.get_client_state(client_id)

        if data.get("has_tensor"):
            # 模式 2：UCX 零拷贝接收（不压缩）
            obs_gpu = self._get_gpu_buffer(client_id)
            self._transport.recv_tensor(client_id, obs_gpu)
            observation = {"camera_obs.robot_head_cam_rgb": obs_gpu}
        else:
            # 模式 1：ZMQ payload 里取（MessageSerializer 已自动解压）
            observation = data["observation"]

        action, info = self._policy.get_action(
            observation=observation,
            env_ids=env_ids,
            client_state=client_state,
        )
        return {**action, **info, "env_ids": env_ids}
    # ... 其他 endpoint 始终走 ZMQ
```

### 5.4 两种模式对比


|               | 模式 1：ZMQ-only       | 模式 2：ZMQ + UCX        |
| ------------- | ------------------- | --------------------- |
| 依赖            | pyzmq（已有）           | pyzmq + ucx-py        |
| obs 传输        | CPU 序列化 → ZMQ bytes | GPU → UCX 零拷贝         |
| 控制消息          | ZMQ                 | ZMQ（不变）               |
| action 回传     | ZMQ                 | ZMQ（action 小，不需要零拷贝）  |
| 跨机            | ✅ TCP               | ✅ TCP / IB / RoCE     |
| 同机优化          | 无                   | UCX 自动用 shared memory |
| 延迟（~1MB obs）  | ~1 ms               | ~0.1 ms               |
| 延迟（~50MB obs） | ~50 ms              | ~1 ms                 |
| 何时用           | 大部分场景（推理延迟 >> 传输）   | Droid 多相机 N>5、高频 VLA  |


### 5.5 NCCL 为什么不适合

NCCL 要求固定 `world_size` + 同步 `init_process_group`，不支持动态连接。
server 必须提前知道有几个 client，一个 crash 全组失效。
**结论**：不适合 RPC 场景。UCX 是正确的 GPU Direct 方案。

### 5.8 Client/Server 环境差异与版本兼容

IsaacLab-Arena 的 client（IsaacSim 进程）和 server（远程 policy 进程，如 GR00T / NaVILA）
通常运行在**不同的 Python 环境**中：


|         | Client (IsaacSim)    | Server (GR00T / NaVILA 等) |
| ------- | -------------------- | ------------------------- |
| Python  | IsaacSim 内置（3.10）    | conda / venv（3.10+）       |
| PyTorch | IsaacSim 打包（如 2.1.2） | 按 VLM 需求安装（如 2.3.1）       |
| CUDA    | IsaacSim 内置（12.x）    | 按 GPU 驱动匹配                |
| ZMQ     | pip install pyzmq    | pip install pyzmq         |
| msgpack | pip install msgpack  | pip install msgpack       |


**版本兼容性**：


| 方案            | 版本要求                               | 风险                    |
| ------------- | ---------------------------------- | --------------------- |
| ZMQ + msgpack | 线协议跨版本兼容                           | ✅ 安全                  |
| lz4 CPU 压缩    | frame 格式跨版本兼容                      | ✅ 安全                  |
| UCX (ucx-py)  | 两侧 UCX 版本兼容即可，不依赖 PyTorch 版本       | ✅ 安全                  |
| nvcomp GPU 压缩 | 两侧 CUDA runtime 版本匹配即可，不依赖 PyTorch | ✅ 安全（需在 Docker 中验证安装） |


### 5.7 传输协商（通过 get_init_info 握手）

**每个 client 可能有不同的传输能力**（有的装了 ucx-py，有的没有）。
Server 为每个 client 维护独立的传输模式。

**协商流程**：

```python
# Client → Server（get_init_info 请求）
{
    "endpoint": "get_init_info",
    "data": {
        "requested_action_mode": "vln_velocity",
        "num_envs": 4,
        "transport_capabilities": ["zmq", "ucx"],       # client 检测本地支持
        "compression_capabilities": ["none", "lz4", "nvcomp_lz4"],
    }
}

# Server → Client（get_init_info 响应）
{
    "status": "success",
    "config": { ... },
    "selected_transport": "ucx",                  # 双方取交集，选最优
    "selected_compression": "lz4",
    "ucx_port": 13337,                            # 如果选了 ucx，给端口
}
```

**Server 侧协商逻辑**：

```python
def _negotiate_transport(self, client_caps: list[str]) -> str:
    """双方取交集，优先级：ucx > zmq。"""
    server_caps = self._detect_transport_backends()  # 自动检测
    for preferred in ["ucx", "zmq"]:
        if preferred in server_caps and preferred in client_caps:
            return preferred
    return "zmq"
```

**Client 侧延迟加载**：

```python
class PolicyClient:
    def __init__(self, config):
        # 初始只建 ZMQ 连接（用于握手）
        self._zmq_transport = ZmqClientTransport(config.host, config.port)
        self._data_transport = None  # 握手后根据协商结果创建

    def connect(self):
        # 1. 检测本地能力
        caps = ["zmq"]
        try:
            import ucp; caps.append("ucx")
        except ImportError:
            pass

        # 2. 握手
        resp = self.call_endpoint("get_init_info", {
            "num_envs": self._num_envs,
            "transport_capabilities": caps,
        })

        # 3. 根据协商结果延迟创建 data transport
        selected = resp.get("selected_transport", "zmq")
        if selected == "ucx":
            ucx_port = resp["ucx_port"]
            self._data_transport = ZmqUcxClientTransport(
                self._config.host, self._config.port, ucx_port)
        else:
            self._data_transport = self._zmq_transport
```

**多 client 不同 transport 的 server 侧处理**：

```python
class PolicyServer:
    def __init__(self, ...):
        self._transport = ZmqServerTransport(port=5555)  # ZMQ 始终在
        self._ucx_transport = None  # 有 client 需要 UCX 时创建
        self._client_transport_mode: dict[bytes, str] = {}  # client_id → "zmq" | "ucx"

    def _handle_get_init_info(self, client_id, data):
        mode = self._negotiate_transport(data.get("transport_capabilities", ["zmq"]))
        self._client_transport_mode[client_id] = mode
        resp = {...}
        if mode == "ucx":
            if self._ucx_transport is None:
                self._ucx_transport = UcxServerEndpoint(port=13337)
            resp["ucx_port"] = 13337
        return resp

    def _dispatch(self, msg, client_id):
        if msg["endpoint"] == "get_action":
            mode = self._client_transport_mode.get(client_id, "zmq")
            if mode == "ucx" and msg["data"].get("has_tensor"):
                # 这个 client 用 UCX 零拷贝
                buf = self._get_gpu_buffer(client_id)
                self._ucx_transport.recv_tensor(client_id, buf)
                observation = {"camera_obs.robot_head_cam_rgb": buf}
            else:
                # 这个 client 用 ZMQ 序列化
                observation = msg["data"]["observation"]
            ...
```

### 5.8 GPU 压缩 + 零拷贝（nvcomp + UCX）

当 observation tensor 很大（如 Droid 双相机 N=10 ~53MB）时，
可以在 GPU 上压缩后再通过 UCX 传输。**全程不经 CPU**。

```
Client GPU                          Network                        Server GPU
──────────                          ───────                        ──────────
obs_gpu [N,720,1280,3]              
  ↓ nvcomp GPU compress (~3×)
compressed_gpu (~17MB)
  ↓ UCX send (GPU Direct)           ──────>  UCX recv (GPU Direct)
                                                ↓
                                             compressed_gpu
                                                ↓ nvcomp GPU decompress
                                             obs_gpu [N,720,1280,3]
                                                ↓ model.forward()
```

**代码流程**：

```python
# Client 端
from nvidia.nvcomp import Codec, from_dlpack

codec = Codec(algorithm="LZ4")
obs_gpu = camera_obs.cuda()                              # [N, 720, 1280, 3] on GPU
flat = obs_gpu.contiguous().view(-1)
compressed = codec.encode(from_dlpack(flat))              # GPU 压缩，返回 nvcomp Array
comp_torch = torch.from_dlpack(compressed.to_dlpack())
await ucx_ep.send(comp_torch)                            # UCX 发压缩后 GPU buffer

# Server 端
comp_buf = torch.empty(comp_size, dtype=torch.uint8, device="cuda")
await ucx_ep.recv(comp_buf)                              # UCX 收到 GPU buffer
decompressed = codec.decode(from_dlpack(comp_buf))
obs_gpu = torch.from_dlpack(decompressed.to_dlpack()).view(torch.uint8)  # dtype 修正
obs_gpu = obs_gpu[:orig_nbytes].view(N, 720, 1280, 3)   # 还原 shape
action = model.forward(obs_gpu)
```

**压缩方式选择**：


| 数据通道            | 压缩方式            | 说明                        |
| --------------- | --------------- | ------------------------- |
| ZMQ（CPU bytes）  | lz4（CPU）        | `MessageSerializer` 层自动处理 |
| UCX（GPU tensor） | nvcomp lz4（GPU） | Transport 层处理，全程不经 CPU    |
| UCX（GPU tensor） | 不压缩             | payload <10MB 时压缩开销不值得    |


压缩方式在 `get_init_info` 握手时协商，与传输方式一起确定。

---

## 6. 序列化与压缩

### 6.1 每次查询数据量估算

Arena 支持不同 policy 类型（VLN、VLA）、多相机、多环境。以下数据来源于
**代码中各 embodiment 和 task 的实际配置**（`observation_keys` + 相机分辨率）。

**各 Arena 任务的单次 `get_action` payload**（按 `num_envs` 缩放）：


| 任务                   | Embodiment | 相机              | 分辨率      | observation_keys                                                                            | 单 env payload | N=10              |
| -------------------- | ---------- | --------------- | -------- | ------------------------------------------------------------------------------------------- | ------------- | ----------------- |
| VLN (NaVILA)         | H1         | 1 (head)        | 512×512  | `camera_obs.robot_head_cam_rgb`                                                             | 0.79 MB       | N/A（仅 num_envs=1） |
| GR1 tabletop (GR00T) | GR1T2      | 1 (pov)         | 512×512  | `camera_obs.robot_pov_cam_rgb` + `policy.robot_joint_pos`                                   | 0.79 MB       | ~7.9 MB           |
| G1 locomanip (GR00T) | G1         | 1 (head)        | 480×640  | `camera_obs.robot_head_cam_rgb` + `policy.robot_joint_pos`                                  | 0.92 MB       | ~9.2 MB           |
| Droid manip (GR00T)  | Droid      | 2 (ext + wrist) | 720×1280 | `camera_obs.external_camera_rgb` + `camera_obs.wrist_camera_rgb` + `policy.robot_joint_pos` | **5.3 MB**    | **~53 MB**        |


> **注**：joint_pos 数据量（< 1 KB/env）相对图像可忽略。Droid 配置中
> `target_image_size=(180, 320, 3)` 是 server 端 resize，client 仍发送原始分辨率。
> 如果 client 端 resize 后再发送，Droid 单 env 降至 ~0.35 MB。

**瓶颈分析**：


| 场景                  | payload | 推理延迟   | 网络延迟（localhost） | 瓶颈                                |
| ------------------- | ------- | ------ | --------------- | --------------------------------- |
| 单 env VLN/VLA       | < 1 MB  | >50 ms | ~1 ms           | **推理**                            |
| 多 env GR1/G1 (N=10) | ~8-9 MB | ~50 ms | ~9 ms           | **推理**（但网络已占 ~15%）                |
| 多 env Droid (N=10)  | ~53 MB  | ~50 ms | ~53 ms          | **网络成为瓶颈**，需启用压缩或 client 端 resize |


**结论**：单相机场景（GR1/G1/H1）即使 N=10 也不是问题。
Droid 双相机高分辨率在 N>5 时网络会成为瓶颈，应考虑：
(1) client 端 resize 后再发送，(2) 启用 lz4 压缩，(3) 减少查询频率。

### 6.2 压缩实现（已完成）

压缩功能已在 `message_serializer.py` 中实现（`compress`/`decompress` 函数），
`RemotePolicyConfig.compression: str = "none"` 控制开关。

支持三种方法：


| 方法             | 说明                          | 依赖                | 状态                             |
| -------------- | --------------------------- | ----------------- | ------------------------------ |
| `"none"`       | 不压缩                         | 无                 | ✅ 默认                           |
| `"lz4"`        | CPU LZ4，~2-3× 压缩比 @ ~5 GB/s | `pip install lz4` | ✅ 代码完成，需 docker rebuild 安装 lz4 |
| `"nvcomp_lz4"` | GPU LZ4 via nvcomp          | `pip install nvidia-nvcomp-cu12` | ✅ 已验证（v5.1.0.21）             |


**压缩与传输层的关系**：压缩在序列化层（`MessageSerializer`）进行，与传输层正交。
无论使用 ZMQ 还是 SharedMemory，压缩逻辑不变。

压缩信息（`__compressed_`_ tag）嵌入在每个 numpy array 的序列化形式中。
接收方通过 tag **自动检测**解压方法，无需预先知道对方用了什么压缩。

### 6.3 能力协商（压缩 + 传输，统一在 get_init_info 握手中完成）

**设计决策**：压缩和传输方式在同一个 `get_init_info` 握手中协商。
server 广播自身能力，client 从交集中选择。不匹配时自动降级。

**Server 端**（PolicyServer 检测已安装的后端并加入 get_init_info 响应）：

```python
# policy_server.py — get_init_info handler 改造
def _handle_get_init_info(self, requested_action_mode, **_):
    resp = self._policy.get_init_info(requested_action_mode=requested_action_mode)
    # PolicyServer 附加自身能力信息
    resp["compression_supported"] = self._detect_compression_backends()
    resp["transport_supported"] = self._detect_transport_backends()
    return resp

def _detect_compression_backends(self) -> list[str]:
    """检测当前环境可用的压缩方法。"""
    supported = ["none"]
    try:
        import lz4.frame  # noqa: F401
        supported.append("lz4")
    except ImportError:
        pass
    try:
        from nvidia.nvcomp import Codec  # noqa: F401
        supported.append("nvcomp_lz4")
    except ImportError:
        pass
    return supported

def _detect_transport_backends(self) -> list[str]:
    """检测当前环境可用的数据传输方式。"""
    supported = ["zmq"]
    try:
        import ucp  # noqa: F401  (ucx-py)
        supported.append("ucx")
    except ImportError:
        pass
    return supported
```

**get_init_info 完整响应示例**：

```python
{
    "status": "success",
    "config": {
        "action_mode": "vln_velocity",
        "action_dim": 3,
        "observation_keys": ["camera_obs.robot_head_cam_rgb"],
        "default_duration": 0.5,
    },
    "compression_supported": ["none", "lz4"],    # ← server 广播压缩能力
    "transport_supported": ["zmq", "ucx"],        # ← server 广播传输能力
}
```

**Client 端**（从 server 支持列表中协商）：

```python
# policy_client.py — 握手时协商压缩和传输
def _negotiate_capabilities(self, init_info: dict) -> tuple[str, str]:
    """协商压缩和传输方式，不匹配时降级。"""
    # 压缩协商
    server_compression = init_info.get("compression_supported", ["none"])
    requested_comp = self._config.compression
    if requested_comp not in server_compression:
        print(f"[PolicyClient] Compression '{requested_comp}' not supported by server "
              f"(supports: {server_compression}), falling back to 'none'")
        requested_comp = "none"

    # 传输协商（优先级：ucx > zmq）
    server_transport = init_info.get("transport_supported", ["zmq"])
    client_transport = self._detect_local_transport()
    common = [t for t in ["ucx", "zmq"] if t in server_transport and t in client_transport]
    selected_transport = common[0] if common else "zmq"

    return requested_comp, selected_transport
```

**为什么用协商而非强制配置一致**：

- 避免 client/server 软件版本不同时 crash（如 server 没装 lz4 但 client 配了 lz4）
- 允许渐进式升级：先升级 server（支持 lz4/ucx），client 慢慢迁移
- `__compressed__` tag 在序列化数据中，**接收方始终能自动检测**解压方法，
即使双方配置不完全一致也不会解码失败
- 传输降级完全透明：无 ucx 时全走 ZMQ，用户无需改配置

---

## 7. 批处理：统一到 PolicyServer

> `BatchedPolicyServer` **不应该是单独的类**。batch_size=1 就是串行，batch_size>1
> 就是批处理，两者只是同一个 PolicyServer 的参数不同。

### 7.1 统一设计

```python
class PolicyServer:
    def __init__(self, policy: ServerSidePolicy, port: int = 5555,
                 max_batch_size: int = 1,      # 1 = 串行，立即处理，零额外延迟（向后兼容）
                 batch_wait_ms: float = 20.0,  # 仅 max_batch_size > 1 时生效
                 idle_timeout_s: float = 300.0):
        ...
```

**参数说明**：

- `max_batch_size=1`（默认）：每收到一条请求立即处理并回复，行为与 v1 完全一致，
**batch_wait_ms 不生效，无任何额外延迟**
- `max_batch_size=N`（N > 1）：启用批处理模式。收到**第一条** `get_action` 后开始计时，
满足以下**任一**条件即处理：①积累到 N 条 ②`batch_wait_ms` 到期
③收集数 ≥ 活跃 client 数（`_last_seen` 中未超时的 client）。
**少 client 场景不会白等**——1 个 client 连接时第 1 条即处理，零等待
- `idle_timeout_s`：某个 `(client_id, env_id)` 超过此时间没有任何请求，
服务端自动清理其 `image_history` 和 `instruction` 状态，释放内存。
与批处理无关，纯粹用于防止内存无限增长

### 7.2 `_collect_batch` 逻辑

**两个关键设计**：

1. `reset` / `ping` 等非推理请求**立即处理，不参与 batch 等待**
2. 收集到的请求数 ≥ **活跃 client 数**时**立即停止等待**，不浪费 `batch_wait_ms`

```python
def _collect_batch(self) -> list:
    batch = []
    # 阻塞等待第一条请求（无超时，避免空转 CPU）
    first = self._recv_one(timeout_ms=None)
    if first is None:
        return []

    # 非 get_action 请求（reset / ping）立即返回，不等待
    if first[1].get("endpoint") != "get_action":
        return [first]

    batch.append(first)
    if self._max_batch_size == 1:
        return batch  # 立即返回，不等待

    # 活跃 client 数 = _last_seen 中未超时的 client 数（server 已维护）
    n_active = len(self._last_seen)

    # 已收集到所有活跃 client 的请求 → 无需再等
    if len(batch) >= n_active:
        return batch

    # max_batch_size > 1：等 batch_wait_ms 内的后续 get_action 请求
    deadline = time.time() + self._batch_wait_ms / 1000
    while len(batch) < self._max_batch_size and time.time() < deadline:
        item = self._recv_one(timeout_ms=5)  # 短轮询
        if item is not None:
            batch.append(item)
            if len(batch) >= n_active:
                break  # 所有活跃 client 都到了
    return batch
```

**各配置下的行为（`batch_wait_ms=20`）**：


| 配置                              | 行为                              | 额外延迟                       |
| ------------------------------- | ------------------------------- | -------------------------- |
| `max_batch_size=1`              | 每条请求立即处理                        | **零**（`batch_wait_ms` 不生效） |
| `max_batch_size=4`，1 client     | `n_active=1`，收到第 1 条即返回         | **零**                      |
| `max_batch_size=4`，2 client 同步发 | 2 条几乎同时到，`2 >= n_active` → 立即返回 | **接近零**                    |
| `max_batch_size=4`，2 client 不同步 | 见下方时序图                          | **最多 20ms**                |
| `max_batch_size=4`，4 client 同步发 | 凑满 4 条 → 立即返回                   | **接近零**                    |
| `reset/ping`                    | 始终立即处理，不参与 batch                | **零**                      |


**2 client 不同步的时序（`max_batch_size=4, batch_wait_ms=20`）**：

`n_active` 短路的含义是"所有已知 client 都发了就别傻等了"，**不是**"强制
等所有人"。具体行为取决于第 2 条请求的到达时间：

```
场景 A：两条请求间隔 < batch_wait_ms（合并处理）
─────────────────────────────────────────────────────
t=0ms    req_A 到达 → batch=[A]，启动 20ms 倒计时
t=8ms    req_B 到达 → batch=[A,B]，len(2) >= n_active(2) → 立即处理
         → A 等了 8ms，B 等了 0ms

场景 B：两条请求间隔 > batch_wait_ms（各自处理）
─────────────────────────────────────────────────────
t=0ms    req_A 到达 → batch=[A]，启动 20ms 倒计时
t=20ms   倒计时到期 → batch=[A]，只处理 A
t=700ms  req_B 到达 → batch=[B]，n_active=2 但倒计时后只有 B → 单独处理 B
         → A 等了 20ms（batch_wait_ms），B 等了 0ms
```

**什么时候两条请求会同步到达？**

- **GR00T（VLA）**：各 client 以相同 sim 频率 step，请求**基本同步**。
batch_wait_ms=10-20ms 足以凑齐大部分请求，合并推理提升吞吐。
- **NaVILA（VLN）**：各 client 有独立的 `duration` 计时器（0.5-1.5s），
请求**不同步**。batch 的实际收益有限，但 `batch_wait_ms` 的额外延迟（≤20ms）
相对 VLM 推理（>500ms）可忽略。

`**batch_wait_ms` 推荐值**：10-20ms。对于 VLM 推理 >500ms 的场景，
20ms 只是 4% 的 overhead。对于 VLA 推理 ~50ms 的场景，10ms 是 20%
的 overhead，但换来批处理的吞吐收益。如果 client 只有 1 个，
`n_active=1` 短路使得 `batch_wait_ms` 完全不生效，零开销。

### 7.3 批处理主循环

```python
def _inference_loop(self):
    while True:
        batch = self._collect_batch()
        if not batch:
            continue

        action_reqs = [(cid, r) for cid, r in batch if r.get("endpoint") == "get_action"]
        other_reqs  = [(cid, r) for cid, r in batch if r.get("endpoint") != "get_action"]

        for cid, req in other_reqs:          # reset/ping 串行处理（很快）
            result = self._dispatch(req, cid)
            self._transport.send(cid, to_bytes(result))

        if action_reqs:
            self._batch_inference(action_reqs)   # 批量 VLM 推理
```

**批处理注意事项（按 policy 类型）**：


| Policy             | 批处理实现                                    | 难度  | 备注                                               |
| ------------------ | ---------------------------------------- | --- | ------------------------------------------------ |
| **NaVILA (LLaVA)** | `model.generate()` 需 padding 到相同 seq_len | 较高  | 各 client 图像帧数、prompt 长度可能不同                      |
| **GR00T**          | `policy.get_action()` 接受 batched tensor  | 较低  | 输入维度固定，天然支持 batch                                |
| **未来 policy**      | 取决于底层模型                                  | —   | 应在 `ServerSidePolicy` 中提供 `supports_batching` 属性 |


GR00T 的批处理比 NaVILA 简单得多：输入是固定维度的 RGB + joint_pos，
无需 padding。如果优先实现批处理，建议从 GR00T 开始验证。

### 7.4 Policy 实现者的批处理注意事项

批处理的实现细节（如 padding、early stopping）是每个 `ServerSidePolicy`
**实现者自己负责**的，不是 PolicyServer 框架的责任。PolicyServer 只负责：

- 收集多个 `get_action` 请求
- 将 batched observation 传给 policy
- 将 batched action 拆分回各 client

**policy 实现者需要注意**：

- 如果模型支持 batch（如 GR00T），直接处理 batched tensor 即可
- 如果模型不支持 batch（或 batch 有特殊要求），policy 可以内部 for 循环处理
- `ServerSidePolicy` 可声明 `supports_batching = True/False`，
让 PolicyServer 决定是否合并请求

> **NaVILA 特定**：当前 `EarlyCommandStopCriteria` 只检查 `input_ids[0]`，
> 在 batch 模式下需改为 per-sequence tracking。这是 NaVILA 实现者的责任，
> 不影响 remote policy 框架设计。详见 `isaaclab_arena_navila/README.md`。

---

## 8. 多 Server 拓扑

> **结论**：**不需要 mesh 网络或 Ray**。Arena 代码中没有任何 Ray 依赖。
> 每个 client 连接到一个 server 就够了。

### 方案 A：静态分配（当前推荐）

```
Isaac Sim 进程 0  ────────────────>  Policy Server A (GPU 0)
Isaac Sim 进程 1  ────────────────>  Policy Server A (GPU 0)
Isaac Sim 进程 2  ────────────────>  Policy Server B (GPU 1)
Isaac Sim 进程 3  ────────────────>  Policy Server B (GPU 1)
```

- 每个进程在启动时配置连接哪个 server（`--remote_host`）
- **无需任何额外基础设施**，GPU 数量无上限
- 缺点：负载可能不均（某些 episode 更长）
- 大规模场景（>4 GPU）可用 Slurm / Kubernetes 等集群管理工具自动分配

### 方案 B：不同模型 A/B 测试

与方案 A 架构相同，不同 server 加载不同模型 checkpoint，可同时对比评测。

---

## 9. 端到端集成：所有模块如何串联

本节展示 ROUTER 多客户端 + Transport 层 + 压缩 + UCX 零拷贝 + per-client 状态
如何在**同一个 server 进程中共存**，尤其是多个 client 同时连接且配置不同的场景。

### 9.1 场景：2 个 client 同时连接，配置不同

```
Client A (GR00T, num_envs=4, 有 UCX)    Client B (NaVILA, num_envs=1, 无 UCX)
      │                                        │
      │   DEALER connect                        │  DEALER connect
      └──────────────────┐    ┌─────────────────┘
                         ▼    ▼
                   PolicyServer (ROUTER)
                   ┌──────────────────────────────────────┐
                   │  ZmqServerTransport (port 5555)      │  ← 所有 client 共用
                   │  UcxServerEndpoint  (port 13337)     │  ← 仅 client A 用
                   │                                      │
                   │  _client_states:                      │
                   │    client_A → ClientState(num_envs=4) │
                   │    client_B → ClientState(num_envs=1) │
                   │                                      │
                   │  _client_transport_mode:              │
                   │    client_A → "ucx"                   │
                   │    client_B → "zmq"                   │
                   │                                      │
                   │  _compression:                        │
                   │    client_A → "nvcomp_lz4"            │  GPU 压缩
                   │    client_B → "lz4"                   │  CPU 压缩
                   └──────────────────────────────────────┘
```

### 9.2 用户 API（用户只碰这些，不碰 Transport）

```python
# ===== 用户启动 Server（和 v1 一样简单）=====
server = PolicyServer(policy=my_policy, host="0.0.0.0", port=5555)
server.run()
# Transport、UCX、压缩协商全部内部自动处理

# ===== 用户启动 Client（和 v1 一样简单）=====
client = PolicyClient(RemotePolicyConfig(host="server", port=5555))
client.connect(num_envs=4)  # 内部自动：检测能力 → get_init_info → 协商 → 建连
action = client.get_action(observation=obs_dict, env_ids=[0,1,2,3])
```

### 9.3 完整生命周期（内部实现）

以下代码是 PolicyServer / PolicyClient 的**内部实现**，用户不直接调用。
展示所有模块如何串联，包括 UCX 建连、多 tensor 传输、批量等待。

```python
# ==================== Phase 1: SERVER 启动 ====================

class PolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=5555, max_batch_size=1,
                 batch_wait_ms=20.0):
        self._policy = policy
        self._transport = ZmqServerTransport(host, port)   # ZMQ ROUTER（始终在）
        self._ucx_endpoint = None                          # UCX（第一个需要的 client 触发创建）
        self._client_states = {}                           # client_id → ClientState
        self._client_transport_mode = {}                   # client_id → "zmq" | "ucx"
        self._client_compression = {}                      # client_id → "none"|"lz4"|"nvcomp_lz4"
        self._last_seen = {}
        self._max_batch_size = max_batch_size
        self._batch_wait_ms = batch_wait_ms
```

```python
# ==================== Phase 2: CLIENT 连接 + 握手 ====================

class PolicyClient:
    def __init__(self, config: RemotePolicyConfig):
        self._config = config
        self._zmq = ZmqClientTransport(config.host, config.port)   # ZMQ 始终在
        self._ucx = None                                           # UCX 延迟创建
        self._transport_mode = "zmq"
        self._compression = "none"

    def connect(self, num_envs: int = 1):
        # 1. 自动检测本地能力（用户不需要配置）
        transport_caps = ["zmq"]
        compression_caps = ["none"]
        try:
            import ucp; transport_caps.append("ucx")
        except ImportError: pass
        try:
            import lz4.frame; compression_caps.append("lz4")
        except ImportError: pass
        try:
            from nvidia.nvcomp import Codec; compression_caps.append("nvcomp_lz4")
        except ImportError: pass

        # 2. 发 get_init_info（走 ZMQ）
        resp = self._call_zmq("get_init_info", {
            "num_envs": num_envs,
            "transport_capabilities": transport_caps,
            "compression_capabilities": compression_caps,
        })

        # 3. 根据协商结果建立数据通道
        self._transport_mode = resp.get("selected_transport", "zmq")
        self._compression = resp.get("selected_compression", "none")

        if self._transport_mode == "ucx":
            import ucp
            ucx_port = resp["ucx_port"]
            # UCX 建连：client 主动连 server 的 UCX listener
            self._ucx = asyncio.run(ucp.create_endpoint(self._config.host, ucx_port))
            # Server 侧：listener accept → self._ucx_endpoint.eps[client_id] = ep
```

```python
# ==================== Phase 2b: SERVER 处理 get_init_info ====================

# (在 PolicyServer.run 主循环内)
def _handle_init(self, client_id, data):
    num_envs = data.get("num_envs", 1)

    # 协商 transport
    mode = self._negotiate_transport(data.get("transport_capabilities", ["zmq"]))
    self._client_transport_mode[client_id] = mode

    # 协商 compression
    comp = self._negotiate_compression(data.get("compression_capabilities", ["none"]))
    self._client_compression[client_id] = comp

    # 创建 per-client 状态
    self._client_states[client_id] = ClientState.create(num_envs)

    resp = {"status": "success", "config": self._policy.protocol.to_dict(),
            "selected_transport": mode, "selected_compression": comp}

    # 按需创建 UCX listener（只创建一次）
    if mode == "ucx":
        if self._ucx_endpoint is None:
            self._ucx_endpoint = UcxServerEndpoint(port=13337)
        resp["ucx_port"] = 13337
        # UCX 建连在 client connect 后由 listener 自动 accept

    return resp
```

```python
# ==================== Phase 3: get_action（多 tensor concat 方案）====================

# --- Client A 侧（UCX + nvcomp）---
class PolicyClient:
    def get_action(self, observation: dict, env_ids: list[int]):
        if self._transport_mode == "ucx":
            return self._get_action_ucx(observation, env_ids)
        else:
            return self._get_action_zmq(observation, env_ids)

    def _get_action_ucx(self, observation, env_ids):
        # 多个 obs tensor 打包成一个连续 buffer + shape layout
        tensors = []
        layout = []
        for key, tensor in observation.items():
            gpu_tensor = tensor.cuda() if not tensor.is_cuda else tensor
            tensors.append(gpu_tensor.reshape(-1))
            layout.append({"key": key, "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype), "numel": tensor.numel()})
        concat_gpu = torch.cat(tensors)  # 一个连续 GPU buffer

        # 可选 nvcomp GPU 压缩
        if self._compression == "nvcomp_lz4":
            from nvidia.nvcomp import Codec, from_dlpack
            codec = Codec(algorithm="LZ4")
            concat_gpu = codec.encode(from_dlpack(concat_gpu))

        # Step 1: ZMQ 发控制消息（含 tensor layout）
        self._zmq.send(to_bytes({
            "endpoint": "get_action",
            "data": {"env_ids": env_ids, "has_tensor": True,
                     "tensor_layout": layout,
                     "tensor_bytes": concat_gpu.numel() * concat_gpu.element_size()}
        }))

        # Step 2: UCX 发 GPU tensor（一次传输，所有 obs concat 在一起）
        asyncio.run(self._ucx.send(concat_gpu))

        # Step 3: ZMQ 收 action（小数据）
        return MessageSerializer.from_bytes(self._zmq.recv())

    def _get_action_zmq(self, observation, env_ids):
        # 全走 ZMQ：observation 序列化 + 可选 lz4 CPU 压缩
        return self._call_zmq("get_action", {
            "env_ids": env_ids,
            "observation": {k: v.cpu().numpy() for k, v in observation.items()},
        })
```

```python
# --- Server 侧处理（UCX + nvcomp）---
# (在 PolicyServer.run 主循环内)

def _handle_get_action(self, client_id, data):
    env_ids = data["env_ids"]
    state = self._client_states[client_id]
    mode = self._client_transport_mode.get(client_id, "zmq")

    if mode == "ucx" and data.get("has_tensor"):
        # UCX 收一个 concat GPU buffer
        n_bytes = data["tensor_bytes"]
        buf = self._get_gpu_buffer(client_id, n_bytes)
        self._ucx_endpoint.recv_tensor(client_id, buf)

        # 可选 nvcomp GPU 解压
        comp = self._client_compression.get(client_id)
        if comp == "nvcomp_lz4":
            from nvidia.nvcomp import Codec, from_dlpack
            codec = Codec(algorithm="LZ4")
            decompressed = codec.decode(from_dlpack(buf))
            buf = torch.from_dlpack(decompressed.to_dlpack()).view(torch.uint8)

        # 按 tensor_layout 拆分成多个 obs tensor
        observation = {}
        offset = 0
        for item in data["tensor_layout"]:
            numel = item["numel"]
            t = buf[offset:offset+numel].reshape(item["shape"])
            observation[item["key"]] = t
            offset += numel
    else:
        # ZMQ 模式：observation 已在 payload 里（MessageSerializer 已解压）
        observation = data["observation"]

    action, info = self._policy.get_action(
        observation=observation,
        env_ids=env_ids,
        client_state=state,
    )
    return {**action, **info, "env_ids": env_ids}
```

```python
# ==================== Phase 4: 主循环（含批量等待）====================

class PolicyServer:
    def run(self):
        while self._running:
            self._gc_stale_clients()

            if self._max_batch_size > 1:
                batch = self._collect_batch()    # §7.2 的逻辑
                self._process_batch(batch)
            else:
                # max_batch_size=1：收一条处理一条
                try:
                    client_id, raw = self._transport.recv(timeout_ms=5000)
                except TimeoutError:
                    continue
                self._last_seen[client_id] = time.time()
                msg = MessageSerializer.from_bytes(raw)
                result = self._dispatch(msg, client_id)
                self._transport.send(client_id, MessageSerializer.to_bytes(result))

    def _dispatch(self, msg, client_id):
        endpoint = msg["endpoint"]
        if endpoint == "get_init_info":
            return self._handle_init(client_id, msg["data"])
        elif endpoint == "get_action":
            return self._handle_get_action(client_id, msg["data"])
        elif endpoint == "set_task_description":
            state = self._client_states[client_id]
            return self._policy.set_task_description(
                msg["data"].get("task_description"),
                env_ids=msg["data"].get("env_ids", list(range(state.num_envs))),
                client_state=state)
        elif endpoint == "reset":
            state = self._client_states[client_id]
            return self._policy.reset(
                env_ids=msg["data"].get("env_ids"),
                client_state=state)
        elif endpoint == "ping":  return {"status": "ok"}
        elif endpoint == "kill":  self._running = False; return {"status": "stopping"}
```

### 9.4 模块交互总结

```
┌─────────────────────────────────────────────────────────────────┐
│                     PolicyServer 主循环                          │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ ZMQ ROUTER       │ ← 所有 client 的所有 endpoint 都走这里     │
│  │ (ServerTransport)│    recv → (client_id, bytes)               │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ MessageSerializer│ ← 反序列化 + 自动检测解压（lz4 / none）    │
│  │ .from_bytes()    │    ZMQ 模式的 observation 在这里解压        │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │ _dispatch()      │───>│ client_transport_mode │               │
│  │                  │    │ [client_id] → "zmq"   │               │
│  │ 按 client_id     │    │             → "ucx"   │               │
│  │ 选择数据通道     │    └──────────────────────┘                │
│  └───┬─────────┬───┘                                            │
│      │         │                                                │
│  ZMQ 模式   UCX 模式                                            │
│      │         │                                                │
│      │    ┌────▼────────────┐                                   │
│      │    │ UCX recv_tensor  │ ← GPU tensor 零拷贝               │
│      │    │ + nvcomp decomp  │   (仅 get_action + has_tensor)    │
│      │    └────┬────────────┘                                   │
│      │         │                                                │
│      ▼         ▼                                                │
│  ┌─────────────────────────┐    ┌──────────────────┐            │
│  │ policy.get_action()      │───>│ ClientState       │           │
│  │   observation (batched)  │    │ [client_id]       │           │
│  │   env_ids                │    │  .instructions[]  │           │
│  │   client_state           │    │  .image_histories[]│           │
│  └────────┬────────────────┘    └──────────────────┘            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ ZMQ ROUTER       │ ← action 回复始终走 ZMQ（数据量小）        │
│  │ .send(client_id) │                                            │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 9.5 潜在冲突点和处理方式


| 冲突点                        | 场景                                   | 处理方式                                                                                               |
| -------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **ZMQ 串行**                 | Client A 和 B 同时发 get_action          | ROUTER 内部队列排队，server 串行处理。批处理模式下可合并（§7）                                                            |
| **UCX 和 ZMQ 消息顺序**         | Client A 先发 ZMQ 控制消息，再发 UCX tensor   | **同一 client 内有序**：ZMQ 先到，server 知道要收 UCX，再调 recv_tensor。不存在乱序                                      |
| **不同 client 不同 transport** | A 用 UCX，B 用 ZMQ                      | `_client_transport_mode[client_id]` 区分。两个通道独立，不冲突                                                  |
| **不同 client 不同压缩**         | A 用 nvcomp，B 用 lz4                   | `_client_compression[client_id]` 区分。ZMQ 模式下 `MessageSerializer` 自动检测；UCX 模式下按记录的 compression 选解压方式 |
| **UCX endpoint 生命周期**      | Client A 断开后 UCX endpoint 残留         | `_gc_stale_clients()` 清理 `_ucx_endpoints[client_id]`                                               |
| **GPU buffer 预分配**         | 不同 client 的 num_envs 和分辨率不同          | per-client GPU buffer，在 get_init_info 时根据 `num_envs` 和协议中的 `observation_keys` 预分配                  |
| **批处理 + 混合 transport**     | A 用 UCX，B 用 ZMQ，两个 get_action 要合并批处理 | 都先收完 observation（A 从 UCX，B 从 ZMQ），然后拼成 batched tensor 传给 policy                                    |


---

> **§9 VLN 推理性能优化**（`max_new_tokens=80`、`EarlyCommandStopCriteria`）
> 已移至 `isaaclab_arena_navila/README.md`。这些是 NaVILA 特定的优化，
> 与 remote policy 框架设计无关。

---

## 10. 迁移路径

### 9.1 Roadmap


| 阶段  | 内容                                 | 状态                      |
| --- | ---------------------------------- | ----------------------- |
| 当前  | REQ/REP 单客户端，无压缩                   | ✅ 已部署                   |
| v2  | ROUTER/DEALER 多客户端 + per-client 状态 | 🔲 设计完成，待实现             |
| v2  | 能力协商（压缩 + 传输，get_init_info 握手）     | 🔲 设计完成，待实现             |
| v2  | ZMQ + UCX 零拷贝传输（模式 2）              | 🔲 设计完成，待实现             |
| v2  | nvcomp GPU 压缩 + UCX                | ✅ pre-test 通过（`nvidia-nvcomp-cu12` v5.1.0.21）   |
| v2  | lz4 CPU 压缩（ZMQ 模式下）                | ✅ 代码完成，需 docker rebuild |
| v2  | 批量推理 (max_batch_size > 1)，GR00T 优先 | 🔲 设计完成，待实现             |


### 10.2 v1 → 多客户端迁移清单

**Server 端**（policy_server.py）：

```python
# 改动 1：socket 类型
sock = ctx.socket(zmq.ROUTER)   # 原 zmq.REP

# 改动 2：主循环读/写 multipart
parts = sock.recv_multipart()
client_id, _, raw = parts[0], parts[1], parts[2]
# ... 处理（注入 client_state）...
sock.send_multipart([client_id, b"", response])

# 改动 3：增加 _client_states / _last_seen / GC 逻辑
# 改动 4：_dispatch() 注入 client_state 到 policy 方法
```

**Client 端**（policy_client.py）：

```python
# 改动 1：socket 类型
sock = ctx.socket(zmq.DEALER)   # 原 zmq.REQ

# 改动 2（⚠️ 关键）：发送/接收帧格式变化
# DEALER 不自动附加空分隔帧，必须显式发送：
sock.send_multipart([b"", payload])   # 不是 sock.send(payload)

# 接收时跳过空分隔帧：
parts = sock.recv_multipart()
response = parts[1]                   # 不是 sock.recv()

# 改动 3：call_endpoint() 中添加 env_id 字段
request["data"]["env_id"] = env_id
```

**ServerSidePolicy ABC**（server_side_policy.py）：

```python
# 改动：get_action / set_task_description / reset 增加 client_state 参数
# 这是 breaking change，所有现有 policy 都需要更新签名
```

**NaVilaServerPolicy**（navila_server_policy.py）：

```python
# 改动：self._instruction → client_state["instruction"]
# 改动：self._image_history → client_state["image_history"]
# 改动：get_action / set_task_description / reset 接受 client_state 参数
```

**Gr00tRemoteServerSidePolicy**（gr00t_remote_policy.py）：

```python
# 改动：self._task_description → client_state["instruction"]
# 改动：get_action / set_task_description 接受 client_state 参数
# 注意：GR00T 的 self.policy.reset() 是模型级 reset，与 per-client 状态无关
```

---

## 11. 实现注意事项

### 11.1 实现时必须处理的技术点


| 项目                    | 说明                                                                                                   | 处理方式                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| DEALER 帧格式            | DEALER 不自动附加空分隔帧，必须显式发 `send_multipart([b"", payload])`，否则 ROUTER 解析 IndexError                      | 在 `PolicyClient` 中封装，调用者无感知                                  |
| ServerSidePolicy 签名变更 | `get_action / set_task_description / reset` 新增 `client_states` keyword-only 参数，是 **breaking change** | v1 policy 需在签名中加 `*, client_states=None`。详见 §4.2 迁移指南        |
| DEALER 超时恢复           | server crash 时 DEALER 的未回复请求会留在 ZMQ 缓冲区                                                              | client 检测超时后**重建 DEALER socket**（清空缓冲区），在 `PolicyClient` 中实现 |
| `idle_timeout_s` GC   | stale client 的 `(client_id, env_id)` 状态需要定期清理                                                        | PolicyServer 主循环中调用 `_gc_stale_clients()`，默认 300s            |


### 11.2 待验证项


| 项目            | 说明                                                      | 验证方法                                         |
| ------------- | ------------------------------------------------------- | -------------------------------------------- |
| nvcomp        | ~~`pip install nvidia-nvcomp`~~ → **`pip install nvidia-nvcomp-cu12`** | ✅ 已验证，见 §12.6。import 路径: `from nvidia.nvcomp import Codec, from_dlpack` |
| lz4 Docker 集成 | `Dockerfile.isaaclab_arena` 需 rebuild 安装 lz4            | `docker build` 后验证 `import lz4.frame`        |
| GR00T 批量推理    | GR00T 天然支持 batched tensor，验证 `max_batch_size>1` 的端到端正确性 | 用多 env GR00T 评测验证                            |
| CuPy 依赖       | ~~cupy 用于 nvcomp/UCX 桥接~~ → **不需要 CuPy**                | ✅ UCX 直接支持 PyTorch tensor；nvcomp 通过 DLPack 直接互操作 |


### 11.3 不实现的方案


| 项目            | 不实现原因                               |
| ------------- | ----------------------------------- |
| CUDA IPC      | UCX 是其超集，不需要单独实现                    |
| NCCL          | 要求固定 world_size + 同步 init，不适合动态 RPC |
| Ray           | Arena 代码中无 Ray 依赖，ZMQ 已满足需求         |
| 多 Server 负载均衡 | 当前方案 A 静态分配够用，需求不明确                 |


---

## 12. Pre-test 清单（设计定稿前在集群上验证）

以下测试项在 **2026-03-05** 于集群 GPU 节点（ad102 / L40 / L40S）上完成。
测试脚本和详细日志见 `claude_tmp/pretest/`。

> **关键修正**（相对设计初稿）：
> - nvcomp pip 包名: ~~`nvidia-nvcomp`~~ → **`nvidia-nvcomp-cu12`**（v5.1.0.21）
> - nvcomp import: ~~`import nvcomp`~~ → **`from nvidia.nvcomp import Codec, from_dlpack`**
> - Codec 构造: `Codec(algorithm="LZ4")`（必须用 keyword 参数）
> - CuPy: **不需要**。UCX 直接支持 PyTorch tensor（`__cuda_array_interface__`），nvcomp 通过 DLPack 直接互操作。
> - nvcomp decode 返回 `int8`，需 `.view(torch.uint8)` 修正 dtype。

### 12.1 依赖安装验证

在 3 个基础镜像中测试 `nvidia-nvcomp-cu12`、`ucx-py-cu12`、`lz4`：

```bash
# 验证命令（以 pytorch:24.07 为例）：
docker run --rm --gpus all --entrypoint bash nvcr.io/nvidia/pytorch:24.07-py3 -c '
pip install nvidia-nvcomp-cu12 ucx-py-cu12 lz4
python -c "from nvidia.nvcomp import Codec; print(\"nvcomp OK\")"
python -c "import ucp; print(\"ucx-py OK\")"
python -c "import lz4.frame; print(\"lz4 OK\")"
'

# Isaac Sim 需要修复 nvidia 命名空间冲突（见 §12.1C 备注）：
docker run --rm --gpus all --entrypoint bash -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  nvcr.io/nvidia/isaac-sim:5.1.0 -c '
/isaac-sim/python.sh -m pip install nvidia-nvcomp-cu12 ucx-py-cu12 lz4
/isaac-sim/python.sh -c "
import nvidia
nvidia.__path__.append(\"/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia\")
from nvidia.nvcomp import Codec; print(\"nvcomp OK\")
"
'
```

> **Isaac Sim 命名空间问题**：Isaac Sim 的 `nvidia` 包路径
> (`/isaac-sim/exts/.../pip_prebundle/nvidia/`) 优先于 pip 安装路径
> (`/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/`)。
> 需要在代码中 `nvidia.__path__.append(...)` 或在 Dockerfile 中
> 调整 `PYTHONPATH` 使两个路径都可见。

### 12.2 UCX 零拷贝验证

UCX (`ucx-py`) 可以**直接**收发 PyTorch GPU tensor，不需要 CuPy：

```python
# PyTorch tensor 支持 __cuda_array_interface__，UCX 直接使用
import torch, ucp, asyncio

async def test():
    async def handler(ep):
        buf = torch.empty(4, 480, 640, 3, dtype=torch.uint8, device="cuda")
        await ep.recv(buf)  # 直接接收到 GPU tensor
        assert buf.sum() > 0
        print("UCX PyTorch GPU transfer: PASS")
    listener = ucp.create_listener(handler, port=13337)
    await asyncio.sleep(0.5)
    ep = await ucp.create_endpoint("127.0.0.1", 13337)
    tensor = torch.randint(0, 255, (4, 480, 640, 3), dtype=torch.uint8, device="cuda")
    await ep.send(tensor)  # 直接发送 GPU tensor
    await asyncio.sleep(1)

asyncio.run(test())
```

### 12.3 nvcomp GPU 压缩 + PyTorch 互操作验证

通过 DLPack 实现 PyTorch ↔ nvcomp 零拷贝，**不需要 CuPy**：

```python
import torch
from nvidia.nvcomp import Codec, from_dlpack

codec = Codec(algorithm="LZ4")
tensor = torch.randint(0, 255, (4, 480, 640, 3), dtype=torch.uint8, device="cuda")

# 压缩：PyTorch → DLPack → nvcomp
flat = tensor.contiguous().view(-1)
compressed = codec.encode(from_dlpack(flat))

# 解压：nvcomp → DLPack → PyTorch（注意 dtype 修正）
decompressed = codec.decode(compressed)
result = torch.from_dlpack(decompressed.to_dlpack()).view(torch.uint8)
result = result[:flat.shape[0]].view(tensor.shape)
assert torch.equal(tensor, result)
```

> **注意**：`codec.decode()` 返回的 Array dtype 为 `int8`，需要
> `.view(torch.uint8)` 转回无符号类型。二进制数据完全一致，只是类型解释不同。

### 12.4 ZMQ ROUTER/DEALER 多 client 验证

（测试代码不变，已验证通过。）

```python
# test_router_dealer.py — 2 个 DEALER 同时连 1 个 ROUTER
import zmq, threading, time

def server():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind("tcp://*:5555")
    received = {}
    for _ in range(4):
        parts = sock.recv_multipart()
        client_id, _, payload = parts[0], parts[1], parts[2]
        received.setdefault(client_id, []).append(payload)
        sock.send_multipart([client_id, b"", b"ACK:" + payload])
    assert len(received) == 2
    print("ROUTER/DEALER multi-client: PASS")

def client(name):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.connect("tcp://localhost:5555")
    for i in range(2):
        sock.send_multipart([b"", f"{name}-msg-{i}".encode()])
        parts = sock.recv_multipart()
        assert parts[1].startswith(b"ACK:")

threading.Thread(target=server, daemon=True).start()
time.sleep(0.1)
threading.Thread(target=client, args=("A",)).start()
threading.Thread(target=client, args=("B",)).start()
time.sleep(1)
```

### 12.5 端到端 lz4 压缩验证

`MessageSerializer.to_bytes()` 目前不支持 `compression_method` 参数
（v2 新功能，待实现）。无压缩 roundtrip 已验证通过：

```python
import numpy as np
from isaaclab_arena.remote_policy.message_serializer import MessageSerializer

obs = {"camera_obs.robot_head_cam_rgb": np.random.randint(0, 255, (4, 480, 640, 3), dtype=np.uint8)}
request = {"endpoint": "get_action", "data": {"env_ids": [0,1,2,3], "observation": obs}}

raw = MessageSerializer.to_bytes(request)
decoded = MessageSerializer.from_bytes(raw)
assert np.array_equal(decoded["data"]["observation"]["camera_obs.robot_head_cam_rgb"],
                       obs["camera_obs.robot_head_cam_rgb"])
print(f"MessageSerializer roundtrip: PASS ({len(raw)} bytes)")
# → PASS (3686643 bytes)
```

### 12.6 测试结果记录表

> 测试日期: 2026-03-05 | 集群节点: ad102 (L40/L40S 48GB) | 测试脚本: `claude_tmp/pretest/`

| #                           | 测试项                                   | Docker 镜像                    | 结果 | 备注                                                  |
| --------------------------- | ------------------------------------- | ---------------------------- | -- | --------------------------------------------------- |
| **依赖安装 — VLN server**       |                                       | pytorch:24.02-py3            |    |                                                     |
| 1a                          | `nvidia-nvcomp-cu12` pip install      | pytorch:24.02                | ✅  | v5.1.0.21                                           |
| 1b                          | `ucx-py-cu12` pip install             | pytorch:24.02                | ✅  | v0.45.0                                             |
| 1c                          | `lz4` pip install                     | pytorch:24.02                | ✅  | v4.4.5                                              |
| **依赖安装 — GR00T server**     |                                       | pytorch:24.07-py3            |    |                                                     |
| 2a                          | `nvidia-nvcomp-cu12` pip install      | pytorch:24.07                | ✅  | v5.1.0.21                                           |
| 2b                          | `ucx-py-cu12` pip install             | pytorch:24.07                | ✅  | v0.45.0                                             |
| 2c                          | `lz4` pip install                     | pytorch:24.07                | ✅  | v4.4.5                                              |
| **依赖安装 — Isaac Sim client** |                                       | isaac-sim:5.1.0              |    |                                                     |
| 3a                          | `nvidia-nvcomp-cu12` pip install      | isaac-sim:5.1.0              | ✅  | 需 `nvidia.__path__` 修复，见上文                           |
| 3b                          | `ucx-py-cu12` pip install             | isaac-sim:5.1.0              | ✅  | Python 3.11, torch 2.7.0+cu128                      |
| 3c                          | `lz4` pip install                     | isaac-sim:5.1.0              | ✅  |                                                     |
| **功能验证**                    |                                       |                              |    |                                                     |
| 4                           | UCX PyTorch GPU tensor 零拷贝            | pytorch:24.07                | ✅  | **不需要 CuPy**，PyTorch 有 `__cuda_array_interface__`   |
| 5                           | nvcomp LZ4 GPU 压缩 roundtrip           | pytorch:24.07 / 24.02 / ISS | ✅  | 3 个镜像全部通过。decode 后需 `.view(torch.uint8)`             |
| 6                           | CuPy ↔ PyTorch 零拷贝桥接                  | —                            | N/A | **不需要 CuPy**，DLPack 直接桥接                            |
| 7                           | ROUTER/DEALER 多 client                | pytorch:24.07                | ✅  |                                                     |
| 8                           | MessageSerializer roundtrip（无压缩）      | pytorch:24.07                | ✅  | 3686643 bytes。lz4 压缩为 v2 待实现功能                      |
| **跨容器验证**                   |                                       |                              |    |                                                     |
| 9                           | UCX: 同节点跨容器（isaaclab_arena ↔ gr00t_server）  | 同节点两容器          | ✅  | GPU tensor 跨容器传输，checksum 匹配                        |
| 10                          | UCX/ZMQ/TCP 跨节点（L20↔L20, L20↔L40S） | ipp1-* + a1u1g-* | ✅  | TCP/ZMQ/UCX 全部通过。需用 ipp1-/ipp2- 系列节点（见备注） |

> **跨节点网络备注**：ipp1-/ipp2- 系列节点（L20, L40）间网络互通（TCP/ZMQ/UCX 均通过）。
> a1u1g-mil-* / alon-ts1-* 系列节点间网络不通（防火墙阻断）。
> 跨节点 UCX 测试应使用 ipp 系列节点。


> **结论：全部通过，设计定稿。**
> CuPy 依赖已移除（不需要）。nvcomp 和 UCX 均可通过 DLPack 直接与 PyTorch 互操作。

---

*文档结束 | Document End*