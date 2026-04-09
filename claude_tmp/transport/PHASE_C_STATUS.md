# Phase C Status

## Current conclusions

- `gpu_compression.py` helper correctness is fixed by cloning the compressed tensor before returning it.
- `torch + ucp` exit-time crashes are independent of `nvcomp`; `torch -> ucp` crashes in the server image even without touching CUDA.
- Same-thread `ucp + torch + CUDA tensor` send/recv succeeds across `l20-8 -> h20-8` with `UCX_TLS=tcp,cuda_copy`, `CUDA_VISIBLE_DEVICES=0`, and a single network interface pinned on both sides.
- Repo-level `zmq_ucx_transport.py` still fails on the Isaac Sim client path with `invalid device context`, which strongly points to the background thread owning no valid CUDA context.
- Raw host-level RoCE/RC connectivity has been confirmed independently of UCX via `ibv_rc_pingpong` on `mlx5_bond_0`, GID index `3`, over `192.168.253.33 <-> 192.168.253.49`.
- Raw host-level UCX over RC has also been confirmed independently of Python `ucp` via `ucx_perftest` with:
  - `UCX_TLS=rc,ud,tcp`
  - `UCX_SOCKADDR_TLS_PRIORITY=rdmacm`
  - `UCX_NET_DEVICES=mlx5_bond_0:1`
  - `UCX_IB_GID_INDEX=3`
  - inter-node config reported as `tag(rc_mlx5/mlx5_bond_0:1 ...)`

## Evidence

### P0 helper lifetime

- Direct script `claude_tmp/compat/nvcomp_roundtrip_probe.py` succeeds in both images.
- Repo helper `gpu_compress/gpu_decompress` failed before the `clone()` fix and succeeds after it.
- Returning a tensor backed by nvcomp intermediate storage was the issue; cloning makes the returned CUDA tensor own its storage.

### P1 same-thread versus background thread

- `claude_tmp/compat/ucp_cuda_same_thread_server.py` + `ucp_cuda_same_thread_client.py` succeeds with GPU tensors and returns equal payloads.
- Repo path still fails with:
  - `cuda_copy_md.c`
  - `cuMemGetAddressRange invalid device context`
- Restricting both sides to `CUDA_VISIBLE_DEVICES=0` does not change the failure.
- `torch.cuda.set_device()` cost has been measured in-container:
  - Server image (`h20-8`, `gr00t_policy_server`):
    - first touch: device 0 ~1023 ms, device 1 ~226 ms
    - repeated same-device `set_device`: ~0.62 us
    - alternating `set_device` after contexts exist: ~0.85 us
    - alternating with tiny CUDA touch after contexts exist: ~0.0084 ms
  - Client image (`l20-8`, `isaaclab_arena`):
    - first touch: device 0 ~465 ms, device 1 ~148 ms
    - repeated same-device `set_device`: ~0.34 us
    - alternating `set_device` after contexts exist: ~0.50 us
    - alternating with tiny CUDA touch after contexts exist: ~0.0060 ms
- Therefore the expensive part is initial CUDA context creation, not steady-state `set_device()` calls.

### P2 client/server UCX userspace mismatch

- Client image:
  - Python: `/isaac-sim/kit/python/bin/python3`
  - `ucp.__file__`: `/isaac-sim/kit/python/lib/python3.11/site-packages/ucp/__init__.py`
  - `ucx-py-cu12==0.45.0`
  - `libucx-cu12==1.18.1`
  - no legacy `ucx-py` package
  - `ldd` on the extension reports `libucp.so.0 => not found`, consistent with bundled `libucx-cu12` wheels and Isaac Sim loader behavior.
- Server image:
  - Python: `/usr/bin/python`
  - `ucp.__file__`: `/usr/local/lib/python3.10/dist-packages/ucp/__init__.py`
  - `ucx-py-cu12==0.45.0`
  - `libucx-cu12==1.18.1`
  - also has legacy `ucx-py==0.37.0`
  - `ldd` on the extension resolves to `/opt/hpcx/ucx/lib/libucp.so.0`
- Therefore, same CUDA version does **not** imply identical UCX runtime behavior; the base image, Python runtime, wheel layout, and linked `libucx` differ.

### P2 transport visibility

- Client/container UCX reports only:
  - `tcp`
  - `cuda_copy`
  - `cuda_ipc`
  - `cma`
  - `self/posix/sysv`
- Server/container HPC-X UCX `ucx_info -d` also currently reports only:
  - `tcp`
  - `cma`
  - `self/posix/sysv`
- Yet server-side HPC-X build config reports `HAVE_IB=1`, `HAVE_MLX5_DV=1`, `HAVE_DC_DV=1`, and `/opt/hpcx/ucx/lib/ucx/` contains `libuct_ib.so` and `libuct_rdmacm.so`.
- `ldd /opt/hpcx/ucx/lib/ucx/libuct_rdmacm.so` originally showed `libuct_ib.so.0 => not found`, meaning the UCX transport module directory itself was missing from `LD_LIBRARY_PATH`.
- Non-privileged server containers can list HCAs via `ibv_devices`, but `ibv_devinfo -d mlx5_0` fails to open the device.
- With `--privileged` **and** `LD_LIBRARY_PATH=/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/ompi/lib:$LD_LIBRARY_PATH`, server-side `ucx_info -d` finally exposes:
  - `dc_mlx5`
  - `rc_verbs`
  - `rc_mlx5`
  - `ud_verbs`
  - `ud_mlx5`
  - `rdmacm`
- Therefore server-side `rc` is available, but only after fixing both container permissions and UCX module search path.
- Client-side custom UCX (`claude_tmp/client_ucx_install`) has now been built successfully with `--with-verbs --with-rdmacm` and its install-time `ucx_info -d` summary shows:
  - `rc_verbs`
  - `rc_mlx5`
  - `dc_mlx5`
  - `rdmacm`
- However, those results were obtained in the build container after installing `rdma-core`/`ibverbs` runtime packages. Fresh `isaaclab_arena:latest` containers do not retain those runtime packages, so client-side `rc` still disappears unless a derived image or equivalent runtime package install is used.
- A fresh derived client image variant `isaaclab_arena:ucx_rc_runtime` now includes the RDMA runtime packages. In that image, `ucp` can be forced to load the custom UCX install (`/workspace/claude_tmp/client_ucx_install/lib/...`) at runtime.
- With both sides switched to their RDMA-capable UCX user spaces, client-side logs show real `rc_mlx5` inter-node configuration rather than immediate TCP-only fallback, but endpoint establishment still fails later during wireup (`no remote ep address for lane[...]` / `exchange_peer_info` reset).
- Raw host-level `ucx_perftest` succeeds over `rc_mlx5` with:
  - `UCX_TLS=rc,ud,tcp`
  - `UCX_SOCKADDR_TLS_PRIORITY=rdmacm`
  - `UCX_NET_DEVICES=mlx5_bond_0:1`
  - `UCX_IB_GID_INDEX=3`
- After aligning both Python sides to UCX `1.17.0` (server using HPC-X 1.17.0; client using freshly built custom UCX 1.17.0) and using:
  - `UCX_TLS=rc,ud,tcp`
  - `UCX_SOCKADDR_TLS_PRIORITY=rdmacm`
  - `UCX_NET_DEVICES=mlx5_bond_0:1`
  - `UCX_IB_GID_INDEX=3`
  Python `ucp` client-server ping (`claude_tmp/transport/ucp_ping_server.py` / `ucp_ping_client.py`) now succeeds over `rc_mlx5` as well.
- Therefore the earlier Python `ucp` RC wireup failure was strongly correlated with runtime/version mismatch (server 1.17.0 vs client 1.18.1) and client userspace packaging, rather than general RC transport availability.

### P3 import order

- `torch -> ucp` crashes at exit in the server image.
- `ucp -> torch` exits cleanly in the server image.
- `nvcomp` is not required to trigger this crash.

## Recommended next steps

1. Keep the `clone()` fix in `gpu_compression.py` and treat P0 correctness as fixed.
2. Rework repo UCX data-path execution so GPU tensor creation/compression/send and recv/decompression all happen on the thread that owns the CUDA context.
3. Keep the client-side rdma-core / ibverbs runtime dependencies in a reusable image variant so custom UCX transport availability survives across fresh containers.
4. Move repo-level UCP transport to the same working runtime conditions:
   - same-thread CUDA path for GPU tensors
   - client/server UCX version alignment
   - `RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true`
   - correct `LD_LIBRARY_PATH`
   - `UCX_SOCKADDR_TLS_PRIORITY=rdmacm` and `UCX_IB_GID_INDEX=3`
5. Re-test repo-level `zmq_ucx_transport.py` once the runtime alignment is applied.
