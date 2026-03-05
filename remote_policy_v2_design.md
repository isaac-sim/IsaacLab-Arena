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
        "requested_action_mode": "chunk",
        "num_envs": 4,           # ← client 告诉 server 有几个 env
    }
}
```

**Server 侧状态结构**：

```python
@dataclass
class ClientState:
    """Per-client 状态，数组按 num_envs 预分配。

    内置字段：
      - instructions: 每个 env 的指令字符串
      - metadata: per-client 级别的 key-value 存储

    可扩展字段（通过 register_per_env_field 注册）：
      Policy 子类可以在 get_init_info 或首次 get_action 时注册自定义
      per-env 字段（如 image_histories、lidar_buffers 等），无需修改
      ClientState 类本身。
    """
    num_envs: int
    instructions: list[str | None]       # [num_envs]，每个 env 的指令
    metadata: dict[str, Any]             # per-client key-value 存储
    _per_env_fields: dict[str, list]     # 动态注册的 per-env 字段

    @classmethod
    def create(cls, num_envs: int) -> ClientState:
        return cls(
            num_envs=num_envs,
            instructions=[None] * num_envs,
            metadata={},
        )

    def register_per_env_field(self, name, *, default_factory=None, default=None) -> list:
        """注册自定义 per-env 字段，返回长度为 num_envs 的列表。幂等。"""
        ...

    def get_per_env_field(self, name) -> list:
        """获取已注册的 per-env 字段。"""
        ...

    def has_per_env_field(self, name) -> bool:
        """检查字段是否已注册。"""
        ...

# 使用示例（NaVILA policy）：
# def _resolve_image_history(self, env_ids, client_state):
#     if not client_state.has_per_env_field("image_histories"):
#         client_state.register_per_env_field("image_histories", default_factory=list)
#     return client_state.get_per_env_field("image_histories")[env_ids[0]]

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
- 正确做法（也是当前实现）：`client_id` 做 key 取到 `ClientState`，再用 `env_ids` 直接数组索引
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
        self._client_states: dict[bytes, ClientState] = {}  # 当前实现：client_id -> ClientState
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
            # 当前实现直接按 client_id 删除整个 ClientState
            del self._client_states[cid]
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
通过 `client_state: ClientState` 参数接收当前客户端的状态引用，并通过：

- `client_state.instructions[eid]`
- `client_state.metadata`
- `client_state.get_per_env_field(...)`

访问状态，而不是把 `client_state` 当成普通 dict。

这样做的好处：

- GC 逻辑集中在 PolicyServer，policy 不需要关心客户端生命周期
- Policy 代码通过 `ClientState` 的显式字段 / helper API 访问状态
- 新增 policy 实现时不需要复制 GC 逻辑

**ServerSidePolicy ABC 签名变更**（**breaking change**，影响所有现有 policy）：

```python
# server_side_policy.py — 当前 v2 contract
class ServerSidePolicy(ABC):

    @abstractmethod
    def get_action(
        self,
        observation: dict[str, Any],
        *,
        env_ids: list[int] | None = None,         # ← 新增：哪些 env
        client_state: ClientState | None = None,  # ← 新增：该 client 的完整状态
        **kwargs: Any,                            # ← 如 options 等扩展字段
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
        # 注意：reset() 不清除 instructions（保持 v1 语义）。
        # 如果 policy 需要清除 per-env 状态（如 image_histories），
        # 在子类 override 中通过 get_per_env_field 清除。
        ...

    # 历史说明：
    # 早期草案曾讨论过 prepare_batch() hook。
    # 在 2026-03-06 的 no-batch 决策下，它不再属于当前 v2 生产接口。
    # 若未来进入 v3，再把 batching 作为新的候选扩展点讨论。
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
def get_action(self, observation, *, env_ids=None, client_state=None, **kwargs):
    # 不需要 per-env 状态的 policy，可以忽略大部分状态字段，
    # 只处理本次 request 的 observation。

    # 需要 per-env 状态的 policy（如 VLN 多指令）：
    if client_state and env_ids:
        for i, eid in enumerate(env_ids):
            instruction = client_state.instructions[eid]
            obs_i = observation["camera_obs.robot_head_cam_rgb"][i]
            ...
```

**PolicyServer dispatch 注入 `client_state`**：

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
# navila_server_policy.py — 改造后（使用 register_per_env_field）
class NaVilaServerPolicy(ServerSidePolicy):

    def get_action(self, observation, options=None, *,
                   env_ids=None, client_state=None):
        # 按需注册 per-env image history（幂等）
        if not client_state.has_per_env_field("image_histories"):
            client_state.register_per_env_field("image_histories", default_factory=list)

        for i, eid in enumerate(env_ids):
            history = client_state.get_per_env_field("image_histories")[eid]
            instruction = client_state.instructions[eid] or "Navigate to the target."
            rgb_i = observation["camera_obs.robot_head_cam_rgb"][i]

            img = Image.fromarray(rgb_i[:, :, :3].astype(np.uint8))
            history.append(img)
            if len(history) > self.config.max_history_frames:
                client_state.get_per_env_field("image_histories")[eid] = \
                    history[-self.config.max_history_frames:]

            vlm_text = self._run_vlm_inference(history, instruction)
            vel_cmd, duration = parse_vlm_output_to_velocity(vlm_text)
            # ... 拼入 batched action ...

    def set_task_description(self, task_description, *,
                             env_ids=None, client_state=None):
        for eid in env_ids:
            client_state.instructions[eid] = task_description or "Navigate to the target."
        return {"status": "ok"}

    def reset(self, env_ids=None, reset_options=None, *, client_state=None):
        # 清除 per-env 状态，但 **不清除 instructions**（保持 v1 语义）
        if client_state.has_per_env_field("image_histories"):
            for eid in env_ids:
                client_state.get_per_env_field("image_histories")[eid].clear()
        return {"status": "reset_success"}

    # 当前 no-batch v2 路线下，NaVILA 按单 request 处理；true batching 留到 v3
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
        "requested_action_mode": "chunk",
        "num_envs": 4,
        "transport_capabilities": ["zmq", "zmq_ucx"],   # client 检测本地支持
        "compression_capabilities": ["none", "lz4", "nvcomp_lz4"],
    }
}

# Server → Client（get_init_info 响应，当前 contract）
{
    "status": "success",
    "config": { ... },
    "negotiated_transport": "zmq_ucx",
    "negotiated_zmq_compression": "none",
    "negotiated_tensor_compression": "none",
    "negotiated_compression": "none",
    "ucx_port": 13337,                            # 如果选了 ucx，给端口
    "zmq_identity": b"...",
}
```

**Server 侧协商逻辑（当前实现）**：

```python
def _negotiate_transport(self, client_caps: list[str]) -> str:
    """双方取交集，优先级：zmq_ucx > zmq。"""
    server_caps = self._detect_transport_backends()  # 自动检测
    for preferred in ["zmq_ucx", "zmq"]:
        if preferred in server_caps and preferred in client_caps:
            return preferred
    return "zmq"
```

**Client 侧握手（当前实现）**：

```python
class PolicyClient:
    def __init__(self, config):
        # 初始只建 ZMQ 连接（用于握手）
        self._zmq_transport = ZmqClientTransport(config.host, config.port)
        self._data_transport = None  # 握手后根据协商结果创建

    def connect(self, num_envs: int, requested_action_mode: str):
        # 1. 检测本地能力
        caps = ["zmq"]
        try:
            import ucp; caps.append("zmq_ucx")
        except ImportError:
            pass

        # 2. 握手
        resp = self.call_endpoint("get_init_info", {
            "requested_action_mode": requested_action_mode,
            "num_envs": num_envs,
            "transport_capabilities": caps,
            "compression_capabilities": self._detect_compression_capabilities(),
        })

        # 3. 记录协商结果并按需建立 UCX 数据通道
        negotiated = resp.get("negotiated_transport", "zmq")
        self._zmq_compression = resp.get("negotiated_zmq_compression", "none")
        self._tensor_compression = resp.get("negotiated_tensor_compression", "none")

        if negotiated == "zmq_ucx":
            ucx_port = resp["ucx_port"]
            zmq_identity = resp["zmq_identity"]
            self._transport.connect_ucx(self._config.host, ucx_port, zmq_identity)
```

**多 client 不同 transport 的 server 侧处理（历史解释，当前实现更简化）**：

```python
class PolicyServer:
    def __init__(self, ...):
        self._transport = ZmqServerTransport(port=5555)  # ZMQ 始终在
        self._ucx_transport = None  # 有 client 需要 UCX 时创建
        self._client_transport_mode: dict[bytes, str] = {}  # 历史草案：client_id → transport mode

    def _handle_get_init_info(self, client_id, data):
        mode = self._negotiate_transport(data.get("transport_capabilities", ["zmq"]))
        self._client_transport_mode[client_id] = mode
        resp = {...}
        if mode == "zmq_ucx":
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
| `"nvcomp_lz4"` | GPU LZ4 via nvcomp          | 安装与 CUDA major 匹配的 wheel（如 `nvidia-nvcomp-cu12` / `nvidia-nvcomp-cu13`） | ✅ 已在 CUDA 12.x 镜像验证（v5.1.0.21）             |


**压缩与传输层的关系**：压缩在序列化层（`MessageSerializer`）进行，与传输层正交。
无论使用 ZMQ 还是 SharedMemory，压缩逻辑不变。

压缩信息（`__compressed_`_ tag）嵌入在每个 numpy array 的序列化形式中。
接收方通过 tag **自动检测**解压方法，无需预先知道对方用了什么压缩。

### 6.3 能力协商（压缩 + 传输，统一在 get_init_info 握手中完成）

**设计决策**：压缩和传输方式在同一个 `get_init_info` 握手中协商。
client 上报自身能力，server 取交集并返回当前协商结果。不匹配时自动降级。

**Server 端**（当前实现：直接返回 `negotiated_*` 结果）：

```python
# policy_server.py — 当前实现口径
def _handle_get_init_info(self, client_id, data):
    client_comp = set(data.get("compression_capabilities", ["none"]))
    client_trans = set(data.get("transport_capabilities", ["zmq"]))

    server_comp = {"none"}
    if self._has_lz4:
        server_comp.add("lz4")
    if self._has_nvcomp:
        server_comp.add("nvcomp_lz4")

    server_trans = {"zmq"}
    if self._has_ucx:
        server_trans.add("zmq_ucx")

    resp = self._policy.get_init_info(requested_action_mode=data["requested_action_mode"])
    resp["negotiated_transport"] = ...
    resp["negotiated_zmq_compression"] = ...
    resp["negotiated_tensor_compression"] = ...
    resp["negotiated_compression"] = ...
    if resp["negotiated_transport"] == "zmq_ucx":
        resp["ucx_port"] = ...
    resp["zmq_identity"] = client_id
    return resp
```

**get_init_info 完整响应示例（当前 contract）**：

```python
{
    "status": "success",
    "config": {
        "action_mode": "chunk",
        "action_dim": 16,
        "observation_keys": ["camera_obs.robot_head_cam_rgb"],
        "action_chunk_length": 8,
        "action_horizon": 8,
    },
    "negotiated_transport": "zmq_ucx",
    "negotiated_zmq_compression": "none",
    "negotiated_tensor_compression": "none",
    "negotiated_compression": "none",
    "ucx_port": 13337,
    "zmq_identity": b"...",
}
```

**Client 端握手（当前实现）**：

```python
def connect(self, num_envs: int, requested_action_mode: str):
    payload = {
        "requested_action_mode": requested_action_mode,
        "num_envs": num_envs,
        "transport_capabilities": ["zmq", "zmq_ucx"],
        "compression_capabilities": ["none", "lz4", "nvcomp_lz4"],
    }
    resp = self.call_endpoint("get_init_info", data=payload, requires_input=True)

    self._cache_zmq_identity_from_handshake(resp, required=True)
    self._negotiated_transport = resp.get("negotiated_transport", "zmq")
    self._zmq_compression = resp.get("negotiated_zmq_compression", "none")
    self._tensor_compression = resp.get("negotiated_tensor_compression", "none")

    if self._negotiated_transport == "zmq_ucx":
        self._transport.connect_ucx(
            self._config.host,
            resp["ucx_port"],
            resp["zmq_identity"],
        )
```

当前实现中，`zmq_identity` 是 transport 层的唯一会话标识：
- 首次连接时由 ZMQ 生成，server 在 `get_init_info` 响应里回传
- client 缓存这份 `zmq_identity`，供 `rebuild()` 复用
- `reconnect()` 会清掉缓存 identity，让 ZMQ 为新会话生成新的 `zmq_identity`

**当前 client/server 行为约定**：

1. **首次 `connect()` / `get_init_info`**
   - client 首次建立 DEALER socket 时不手工设置 identity，由 ZMQ 自动分配 routing identity。
   - server 的 ROUTER 在收到 `get_init_info` 时获得这份 `zmq_identity`，并把它原样放回响应。
   - client 收到响应后缓存这份 `zmq_identity`；后续若只做传输恢复（`rebuild()`），继续复用它。

2. **`rebuild()`**
   - 只用于“当前会话仍有效，但 ZMQ socket 需要清 stale buffer”的场景。
   - client 不重新握手，也不请求新会话。
   - transport 会把缓存的 `zmq_identity` 再设置回新的 DEALER socket，因此 server 继续把它视为同一个 live client。

3. **`reconnect()`**
   - 用于“当前会话作废，需要完整重建”的场景。
   - client 会先清掉缓存的 `zmq_identity`，再重建 socket，再重新执行 `get_init_info`。
   - 因为这时不再复用旧 identity，ZMQ 会为新 socket 分配一份新的 `zmq_identity`；server 将其视为全新会话。

4. **server 对重复 `get_init_info` 的处理**
   - 如果某个 `zmq_identity` 已经对应一条 live session，server 会拒绝该 identity 再次调用 `get_init_info`。
   - 也就是说：**同一个 live `zmq_identity` 不允许原地重开新会话**。
   - 如果调用方确实需要新会话，应走 `reconnect()`，让新 socket 获取新的 `zmq_identity` 后再握手。

5. **server GC stale client**
   - server 会按 `idle_timeout_s` 清理长时间未活跃的 `zmq_identity`，删除其：
     - `ClientState`
     - 压缩协商状态
     - UCX endpoint 关联
     - `last_seen`
   - 一旦某个 `zmq_identity` 被 GC，它就不再被视为 live session。
   - 旧 socket 若继续发送 stateful request，会得到“缺少 `ClientState` / 必须 reconnect”的错误。
   - 推荐恢复路径是：client 调 `reconnect()`，清掉本地缓存 identity，让 ZMQ 为新会话生成新的 `zmq_identity`，再重新握手。

**为什么用协商而非强制配置一致**：

- 避免 client/server 软件版本不同时 crash（如 server 没装 lz4 但 client 配了 lz4）
- 允许渐进式升级：先升级 server（支持 lz4/ucx），client 慢慢迁移
- 响应中的 `negotiated_*` 字段使双方在握手时就达成一致；
  接收方再通过 magic header / 数据格式做最终解码
- 传输降级完全透明：无 ucx 时全走 ZMQ，用户无需改配置

---

## 7. 批处理：统一到 PolicyServer（历史设计，已由 no-batch 决策取代）

> **2026-03-06 决策更新**：
> v2 的生产运行路径不再依赖通用 server-side batching。
> 本章 7.1-7.4 保留为历史设计讨论 / v3 候选方案。
> 当前 v2 实现应按：
> `recv -> validate -> dispatch_single -> send`
> 的单 request 语义理解。

### 7.1 统一设计

这一小节以下内容均为**历史草案**。当前 v2 生产实现已经不再暴露：

- `max_batch_size`
- `batch_wait_ms`
- 通用 server-side batch 主循环

当前 contract 更简单：

```python
class PolicyServer:
    def run(self):
        while self._running:
            client_id, raw = self._transport.recv()
            request = MessageSerializer.from_bytes(raw)
            self._dispatch_single(client_id, request)
```

`idle_timeout_s` 仍然保留，用于按 `client_id` 清理 stale `ClientState`。

### 7.2 `_collect_batch` 逻辑

已被 no-batch 决策取代。保留到 v3 再讨论。

### 7.3 批处理主循环

已被 no-batch 决策取代。当前生产主循环为单 request dispatch。

**批处理注意事项（按 policy 类型，供 v3 参考）**：


| Policy             | 批处理实现                                    | 难度  | 备注                                               |
| ------------------ | ---------------------------------------- | --- | ------------------------------------------------ |
| **NaVILA (LLaVA)** | `model.generate()` 需 padding 到相同 seq_len | 较高  | 各 client 图像帧数、prompt 长度可能不同                      |
| **GR00T**          | `policy.get_action()` 接受 batched tensor  | 较低  | 输入维度固定，天然支持 batch                                |
| **未来 policy**      | 取决于底层模型                                  | —   | 应在 `ServerSidePolicy` 中提供 `supports_batching` 属性 |


GR00T 的批处理比 NaVILA 简单得多：输入是固定维度的 RGB + joint_pos，
无需 padding。如果未来在 v3 优先实现 batching，建议从 GR00T 开始验证。

### 7.4 Policy 实现者的批处理注意事项

仍然保留为 v3 候选讨论：

- batching 需要 policy-specific 设计
- VLM policy 的 prompt/history/stopping criteria 不能被当前 v2 shared infra 简化掉
- 当前生产实现不应从这里推断出任何 active batching contract

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
client.connect(num_envs=4, requested_action_mode="chunk")  # 内部自动：检测能力 → get_init_info → 协商 → 建连
action = client.get_action(observation=obs_dict, env_ids=[0,1,2,3])
```

### 9.3 完整生命周期（内部实现）

以下代码是 PolicyServer / PolicyClient 的**内部实现草图**，用户不直接调用。
本节用于说明模块如何串联；其中个别对象/字段仍带历史背景，但已尽量对齐当前实现口径。

```python
# ==================== Phase 1: SERVER 启动 ====================

class PolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=5555):
        self._policy = policy
        self._transport = ZmqServerTransport(host, port)   # ZMQ ROUTER（始终在）
        self._ucx_endpoint = None                          # UCX（第一个需要的 client 触发创建）
        self._client_states = {}                           # client_id → ClientState
        self._client_transport_mode = {}                   # 历史草案：当前生产实现不再显式暴露这张表
        self._client_compression = {}                      # client_id → "none"|"lz4"|"nvcomp_lz4"
        self._last_seen = {}
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

    def connect(self, num_envs: int, requested_action_mode: str):
        # 1. 自动检测本地能力（用户不需要配置）
        transport_caps = ["zmq"]
        compression_caps = ["none"]
        try:
            import ucp; transport_caps.append("zmq_ucx")
        except ImportError: pass
        try:
            import lz4.frame; compression_caps.append("lz4")
        except ImportError: pass
        try:
            from nvidia.nvcomp import Codec; compression_caps.append("nvcomp_lz4")
        except ImportError: pass

        # 2. 发 get_init_info（走 ZMQ）
        resp = self._call_zmq("get_init_info", {
            "requested_action_mode": requested_action_mode,
            "num_envs": num_envs,
            "transport_capabilities": transport_caps,
            "compression_capabilities": compression_caps,
        })

        # 3. 根据协商结果建立数据通道
        self._transport_mode = resp.get("negotiated_transport", "zmq")
        self._compression = resp.get("negotiated_compression", "none")

        if self._transport_mode == "zmq_ucx":
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
            "negotiated_transport": mode, "negotiated_compression": comp}

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
    # ==================== Phase 4: 主循环（当前生产实现为 single-request）====================

class PolicyServer:
    def run(self):
        while self._running:
            self._gc_stale_clients()
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
| v2  | ROUTER/DEALER 多客户端 + per-client 状态 | ✅ 代码完成，待 focused tests 执行 |
| v2  | 能力协商（压缩 + 传输，get_init_info 握手）     | ✅ 代码完成，待 focused tests 执行 |
| v2  | ZMQ + UCX 零拷贝传输（模式 2）              | ✅ 代码完成，待 focused tests 执行 |
| v2  | nvcomp GPU 压缩 + UCX                | ✅ pre-test 通过（当前验证环境为 CUDA 12.x，对应 `nvidia-nvcomp-cu12` v5.1.0.21）   |
| v2  | lz4 CPU 压缩（ZMQ 模式下）                | ✅ 代码完成，需 docker rebuild |
| v2  | `env_ids` 定制化请求 targeting | ✅ 设计确定（shared infra 目标） |
| v3  | 通用 / policy-specific batching | 🔲 延后到 v3 讨论              |


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

# 改动 3：call_endpoint() 中添加 env_ids 字段
request["data"]["env_ids"] = env_ids
```

**ServerSidePolicy ABC**（server_side_policy.py）：

```python
# 改动：get_action / set_task_description / reset 增加 client_state 参数
# 这是 breaking change，所有现有 policy 都需要更新签名
```

**NaVilaServerPolicy**（navila_server_policy.py）：

```python
# 改动：self._instruction → client_state.instructions[eid]
# 改动：self._image_history → 通过 register_per_env_field("image_histories") 管理
# 改动：get_action / set_task_description / reset 接受 client_state 参数
```

**Gr00tRemoteServerSidePolicy**（gr00t_remote_policy.py）：

```python
# 改动：self._task_description → client_state.instructions[eid]
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
| `idle_timeout_s` GC   | stale client 的 `ClientState` 需要定期清理                                                        | PolicyServer 主循环中调用 `_gc_stale_clients()`，默认 300s            |


### 11.2 待验证项


| 项目            | 说明                                                      | 验证方法                                         |
| ------------- | ------------------------------------------------------- | -------------------------------------------- |
| nvcomp        | ~~`pip install nvidia-nvcomp`~~ → **安装与 CUDA major 匹配的 `nvidia-nvcomp-cuXX`** | ✅ 已在 CUDA 12.x 环境验证，见 §12.6。import 路径: `from nvidia.nvcomp import Codec, from_dlpack` |
| lz4 Docker 集成 | `Dockerfile.isaaclab_arena` 需 rebuild 安装 lz4            | `docker build` 后验证 `import lz4.frame`        |
| GR00T 批量推理    | **历史 / v3 候选**：若未来重新讨论 batching，再单独验证 | 不属于当前 v2 no-batch 收口范围                            |
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
> - nvcomp pip 包名: ~~`nvidia-nvcomp`~~ → **安装与 CUDA major 匹配的 `nvidia-nvcomp-cuXX`**
>   （当前 pre-test 验证环境使用 `nvidia-nvcomp-cu12` v5.1.0.21）
> - nvcomp import: ~~`import nvcomp`~~ → **`from nvidia.nvcomp import Codec, from_dlpack`**
> - Codec 构造: `Codec(algorithm="LZ4")`（必须用 keyword 参数）
> - CuPy: **不需要**。UCX 直接支持 PyTorch tensor（`__cuda_array_interface__`），nvcomp 通过 DLPack 直接互操作。
> - nvcomp decode 返回 `int8`，需 `.view(torch.uint8)` 修正 dtype。

### 12.1 依赖安装验证

在 3 个基础镜像中测试 CUDA 12.x 对应的 `nvidia-nvcomp-cu12`、`ucx-py-cu12`、`lz4`：

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

`MessageSerializer.to_bytes()` 当前已经支持 `compression_method`
（如 `"none"` / `"lz4"`）。下面这段仍主要展示 roundtrip 验证思路：

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
| 8                           | MessageSerializer roundtrip（无压缩）      | pytorch:24.07                | ✅  | 3686643 bytes。lz4 压缩代码已支持，需在实际镜像中执行 focused tests 验证 |
| **跨容器验证**                   |                                       |                              |    |                                                     |
| 9                           | UCX: 同节点跨容器（isaaclab_arena ↔ gr00t_server）  | 同节点两容器          | ✅  | GPU tensor 跨容器传输，checksum 匹配                        |
| 10                          | UCX/ZMQ/TCP 跨节点（L20↔L20, L20↔L40S） | ipp1-* + a1u1g-* | ✅  | TCP/ZMQ/UCX 全部通过。需用 ipp1-/ipp2- 系列节点（见备注） |

> **跨节点网络备注**：ipp1-/ipp2- 系列节点（L20, L40）间网络互通（TCP/ZMQ/UCX 均通过）。
> a1u1g-mil-* / alon-ts1-* 系列节点间网络不通（防火墙阻断）。
> 跨节点 UCX 测试应使用 ipp 系列节点。


> **结论：全部通过，设计定稿。**
> CuPy 依赖已移除（不需要）。nvcomp 和 UCX 均可通过 DLPack 直接与 PyTorch 互操作。

---

## Addendum: Post-review 设计补充（2026-03-06）

以下章节记录了在 6 轮 agent review + 1 轮 human review 后的设计变更。

### A.1 关于 Batch 的最新决定（2026-03-06）

**结论**：v2 不把通用 server-side batching 作为闭环目标。

原因：

1. batch 的收益真实存在，但只在部分 workload 上明显
2. 不同任务的 batch 语义差异过大，难以在 v2 抽象成一个干净的通用设计
3. `GR00T` 与 `NaVILA` 的模型语义差异非常大：
   - `GR00T` 更接近固定形状 tensor forward
   - `NaVILA` 更接近 instruction + history + prompt + generate 的 VLM 流程
4. v2 的主要收益来自：
   - 多 client 正确性
   - per-client state
   - `env_ids` targeting
   - transport / compression / reconnect / error semantics
   而不是通用 batch

因此：

- **v2 保留 `env_ids` 定制化能力**
- **v2 不要求 PolicyServer 作为通用层去合并 heterogeneous requests**
- **通用 / policy-specific batching 延后到 v3**

v2 推荐伪代码：

```python
def client_query(remote_client, observation, env_ids):
    sliced_obs = slice_observation(observation, env_ids)
    return remote_client.get_action(
        observation=sliced_obs,
        env_ids=env_ids,
    )
```

```python
def get_action(observation, *, env_ids, client_state):
    # policy-specific handling
    ...
```

如果未来进入 v3，再重新评估：

- 哪些 policy 值得做 batching
- batching 是 shared server 级能力还是 policy-specific 能力
- VLM 类 policy 的 batched prompt / history / stopping criteria 设计

### A.2 UCX CUDA Stream Sync

**规则**：所有 UCX `send_tensor` 调用前必须 `torch.cuda.current_stream().synchronize()`。

**原因**：PyTorch CUDA ops 是异步的。UCX 的 `ep.send(tensor)` 通过 `__cuda_array_interface__` 读 GPU 显存（DMA），不在 CUDA stream 上。如果前面的 CUDA kernel（如 `torch.cat`、`gpu_compress`）还没完成，DMA 会读到脏数据。

**实现位置**：`zmq_ucx_transport.py` 的 `ZmqUcxServerTransport.send_tensor()` 和 `ZmqUcxClientTransport.send_tensor()` 内部。

### A.3 nvcomp Sync 安全

nvCOMP Python API 的 `Codec` 支持显式 `cuda_stream` 参数；**如果不传，官方文档说明会为该 device 创建 internal CUDA stream**。因此：

- 仅在 `codec.encode()` / `codec.decode()` 之后调用 `torch.cuda.current_stream().synchronize()`，**并不能证明或保证** nvcomp 本身就在 PyTorch 当前 stream 上执行；
- `claude_tmp/test_nvcomp_sync.py` 这类实验最多只能说明“在某个场景里，encode 看起来会等前序工作”，**不能推出默认 stream 就是 `torch.cuda.current_stream()`**。

当前实现采用更严格、可解释的做法：

1. `from_dlpack(..., cuda_stream=stream_ptr)`：把输入 buffer 的可见性绑定到当前 PyTorch stream
2. `Codec(..., cuda_stream=stream_ptr, device_id=...)`：让 nvcomp encode/decode 也在同一条 stream 上执行
3. `to_dlpack(cuda_stream=stream_ptr)`：把 nvcomp 输出再按同一条 stream 交还给 PyTorch

这样一来，`tensor.contiguous()` / flatten → nvcomp encode/decode → `torch.from_dlpack(...)` 处在同一条有序的 CUDA stream 上，不再依赖“post-sync 猜测”。

### A.4 `disconnect` endpoint + `allow_remote_kill`

- `disconnect`：只清理该 client 的 state（从 `_client_states`、compression、UCX endpoint 中移除）。不影响其他 client，不关闭 server。
- `kill`：关闭整个 server。默认禁用（`allow_remote_kill=False`）。生产环境应通过 SIGTERM 关闭 server。
- `kill` → `disconnect` 是 v1→v2 的行为变更。v2 client 应调用 `disconnect()` 而非 `kill()`。

### A.5 压缩协商拆分

v2 的 `get_init_info` 响应中拆分为两个独立的压缩协商字段：
- `negotiated_zmq_compression`：ZMQ 控制消息的压缩（`"none"` 或 `"lz4"`）
- `negotiated_tensor_compression`：UCX tensor 的压缩（`"none"` 或 `"nvcomp_lz4"`）
- `negotiated_compression`：保留用于向后兼容，取两者中更优的

当前实现还增加了一条收口策略：
- 当 `negotiated_transport == "zmq_ucx"` 时，`negotiated_zmq_compression` 默认固定为 `"none"`。
- 理由是大 payload 已经离开 ZMQ 数据面，控制消息通常只有 metadata；继续做 CPU lz4 收益很小，复杂度和 CPU 开销反而更高。
- `negotiated_tensor_compression` 仍然独立协商，可继续选择 `"nvcomp_lz4"`。

CPU lz4 和 GPU nvcomp LZ4 **不可互换**（不同的帧格式）。协商机制确保双方一致。

### A.6 Client-Side Observation Preprocessing（图像 resize 等）

**当前现状**：图像 resize 在 server 端执行。以 GR00T 为例，`Gr00tRemoteServerSidePolicy.get_action()` 收到 client 发来的原始分辨率图像后，调用 `build_gr00t_policy_observations()` → `resize_rgb_for_policy()` → `resize_frames_with_padding()` 将图像 resize 到 `target_image_size`（如 Droid 的 `(180, 320, 3)`）。

**问题**：client 发送原始高分辨率图像导致 payload 很大。以 Droid 为例，双相机 720×1280 原始图像单 env payload 约 5.3 MB，N=10 时达到 ~53 MB，网络传输延迟与推理延迟持平，成为瓶颈（见 §6.1）。而 server 收到后立刻 resize 到 180×320，大量传输带宽被浪费。

**优化方向**：client 在发送前 resize 到 `target_image_size`，可显著降低 payload。以 Droid 为例，resize 到 180×320 后单 env payload 约 0.35 MB，降幅约 93%。

**为什么不做进 shared infra**：

1. **`target_image_size` 是 policy-specific config**。GR00T 使用 `(180, 320, 3)`（来自 `Gr00tClosedloopPolicyConfig`），NaVILA 或其他 policy 可能使用不同的目标尺寸，甚至不需要 resize。通用层（`ClientSidePolicy`、`PolicyClient`）不应该依赖特定 policy 的 config 结构。
2. **Padding 策略也是 policy-specific**。GR00T 的 `resize_frames_with_padding()` 在 resize 前做 letterbox padding（`pad_img=True`：先计算 `(W - H) // 2` 做对称上下黑边填充，使宽图变为正方形，再 resize 到目标尺寸）。这是为了匹配 GR00T 模型的训练数据格式，其他 policy 不一定需要此逻辑。
3. **不同 policy 可能需要不同的 resize 算法**。GR00T 使用 `cv2.resize`（默认双线性插值），其他 policy 可能使用 `torchvision.transforms.Resize`、`PIL.Image.resize` 或自定义的 GPU resize kernel，算法选择和插值方式各异。

**推荐模式**：

v2 的 `get_init_info` 响应中可以携带可选的 `target_image_size` 字段（如 `[180, 320, 3]`），client 侧的 policy-specific adapter（如 `ActionChunkingClientSidePolicy`）读取后，在 `pack_observation_for_server()` 中对图像做 resize。具体的 resize 逻辑（是否 padding、用什么插值算法）由各 policy 的 client adapter 自行实现，不进入 `ClientSidePolicy` 基类或 `PolicyClient`。

```python
# future API candidate：policy-specific client adapter 中的预处理。
# 注意：这不是当前仓库里的现成接口；这里只表达推荐的分层方式。
class DroidClientSidePolicy(ActionChunkingClientSidePolicy):
    def __init__(self, ...):
        super().__init__(...)
        self._target_image_size = None

    def update_preprocess_hints(self, handshake_resp):
        # future: get_init_info / config 可选返回 target_image_size 等提示
        self._target_image_size = handshake_resp.get("target_image_size")

    def pack_observation_for_server(self, observation):
        packed = super().pack_observation_for_server(observation)
        if self._target_image_size is None:
            return packed

        for key, value in packed.items():
            if "camera" not in key:
                continue
            if not isinstance(value, np.ndarray) or value.ndim != 4:
                continue
            packed[key] = self._resize_for_server(value, self._target_image_size)
        return packed

    def _resize_for_server(self, frames, target_size):
        # 这里应复用该 policy 自己的训练期预处理约定。
        # 对 GR00T 来说，通常不是单纯 cv2.resize，而是等价于
        # resize_frames_with_padding(..., pad_img=True) 的逻辑。
        return policy_specific_resize(frames, target_size)
```

**v2 范围说明**：这是 **v3 候选功能**，当前 v2 不要求实现。v2 的 server 端 resize 路径保持不变，功能正确且已验证。鼓励 client 开发者在带宽敏感场景下（如 Droid 多 env、跨节点部署）自行在 client adapter 中实现 resize 优化。

---

*文档结束 | Document End*