# Mooncake Handoff

## Stable commit on current branch

- Current branch: `remote_policy_v2`
- Stable nvcomp-related commit: `d5eb9004`
- Summary:
  - `gpu_compression.py` now clones the compressed CUDA tensor before returning it
  - helper-based nvcomp round-trip works again
  - capability detection uses the shared nvcomp import helper to apply Isaac Sim namespace fixes

## What is proven on the UCX path

### P0 nvcomp helper correctness

- Repo helper correctness is fixed.
- `claude_tmp/compat/nvcomp_roundtrip_probe.py --use-repo-helper` succeeds after the `clone()` fix.
- Remaining `exit 139` behavior is a separate `torch + ucp` teardown problem, not a data corruption problem.

### P1 torch + ucp + CUDA tensors

- `claude_tmp/compat/ucp_cuda_same_thread_server.py`
- `claude_tmp/compat/ucp_cuda_same_thread_client.py`

These same-thread scripts succeed across `l20-8 -> h20-8` with CUDA tensors.

- Conclusion: repo-level `invalid device context` is strongly tied to the current background event-loop thread model in `zmq_ucx_transport.py`.
- Restricting both sides to `CUDA_VISIBLE_DEVICES=0` did **not** fix the repo path, so the issue is not multi-GPU visibility by itself.

### P2 host-level RDMA and UCX

- Raw verbs RC is proven on the hosts:
  - `ibv_rc_pingpong -d mlx5_bond_0 -i 1 -g 3 -p 18555`
- Raw host-level UCX RC is also proven on the hosts:
  - `UCX_TLS=rc,ud,tcp`
  - `UCX_SOCKADDR_TLS_PRIORITY=rdmacm`
  - `UCX_NET_DEVICES=mlx5_bond_0:1`
  - `UCX_IB_GID_INDEX=3`
  - `ucx_perftest 192.168.253.49 -t tag_lat -p 13379 -n 1000`

### P2 Python ucp RC success conditions

Python `ucp` over RC succeeds only when the client and server runtime conditions are aligned.

Server-side success conditions:

- Image: `gr00t_policy_server:latest`
- Container flags:
  - `--privileged`
- Env:
  - `RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true`
  - `LD_LIBRARY_PATH=/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/ompi/lib`
  - `UCX_TLS=rc,ud,tcp`
  - `UCX_SOCKADDR_TLS_PRIORITY=rdmacm`
  - `UCX_NET_DEVICES=mlx5_bond_0:1`
  - `UCX_IB_GID_INDEX=3`
- Effective Python UCX version:
  - `1.17.0`

Client-side success conditions:

- Image: `isaaclab_arena:ucx_rc_runtime`
- Custom UCX install:
  - `claude_tmp/client_ucx_install_117`
- Env:
  - `RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true`
  - `LD_LIBRARY_PATH=/workspace/claude_tmp/client_ucx_install_117/lib/ucx:/workspace/claude_tmp/client_ucx_install_117/lib`
  - `UCX_TLS=rc,ud,tcp`
  - `UCX_SOCKADDR_TLS_PRIORITY=rdmacm`
  - `UCX_NET_DEVICES=mlx5_bond_0:1`
  - `UCX_IB_GID_INDEX=3`
- Effective Python UCX version:
  - `1.17.0`

Working Python RC ping:

- Server: `claude_tmp/transport/ucp_ping_server.py --port 5576`
- Client: `claude_tmp/transport/ucp_ping_client.py --host 192.168.253.49 --port 5576 --message rc117-ok`

## Remaining UCX issues

### Repo-level transport

- `isaaclab_arena/remote_policy/transport/zmq_ucx_transport.py` still uses a background loop/thread model.
- That model is the leading suspect for repo-level CUDA context failures on the client path.
- The repo transport has **not yet** been updated to the same-thread model that was proven to work in the standalone scripts.

### torch + ucp teardown

- `torch -> ucp` still crashes at process exit in the server image, even without `nvcomp`.
- `ucp -> torch` exits cleanly.
- This is likely a Python teardown / UCX cleanup ordering problem.

## Why Mooncake is worth testing

Based on the Mooncake docs:

- Primary Python transport modes are `tcp` and `rdma`.
- Control plane examples already use ZMQ-style metadata handoff.
- The Python API directly exposes:
  - `initialize(...)`
  - `register_memory(...)`
  - `transfer_sync_write/read(...)`
  - `transfer_write_on_cuda/read_on_cuda(...)`
- It looks more like a transport engine than a low-level communication toolkit.

Important nuance:

- Python Quick Start clearly treats `tcp` and `rdma` as the primary modes.
- Advanced local protocols like `nvlink_intra`, `nvlink`, and `cxl` are documented, but not presented as the default Python-path protocol selection.
- Expect to choose a top-level protocol (`rdma` or `tcp`) yourself. Do not assume a global auto-selector that always picks the best local-vs-remote protocol without extra policy code.

## Recommended Mooncake spike

1. Start with a minimal dual-host Python `TransferEngine` test over `tcp`.
2. Repeat the same test over `rdma` with:
   - explicit `device_name="mlx5_bond_0"` or auto-discovery
3. Verify whether Mooncake Python RDMA works in:
   - host Python directly
   - the same two container images
4. Check whether CUDA transfer APIs work with PyTorch stream pointers.
5. Compare operational friction versus the current UCX path:
   - required env vars
   - runtime package alignment
   - version sensitivity
   - container permissions

## Files worth reading first

- `claude_tmp/transport/PHASE_C_STATUS.md`
- `claude_tmp/compat/ucp_cuda_same_thread_server.py`
- `claude_tmp/compat/ucp_cuda_same_thread_client.py`
- `claude_tmp/transport/ucp_ping_server.py`
- `claude_tmp/transport/ucp_ping_client.py`
- `claude_tmp/compat/build_client_custom_ucx.sh`
- `claude_tmp/compat/cuda_set_device_bench.py`

## Intentional omissions from this backup commit

The following are intentionally **not** meant to be committed as artifacts:

- `claude_tmp/client_ucx_build/`
- `claude_tmp/client_ucx_install/`
- `claude_tmp/client_ucx_install_117/`

They are reproducible build outputs and too large for a useful handoff commit.
