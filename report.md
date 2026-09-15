# GB300 Arena Enablement Report

Date: 2026-09-12 (UTC)

## Executive summary

Isaac Lab-Arena `release/0.3.0` is runnable on this ARM64 NVIDIA GB300 machine through the
clone-specific Docker environment. The Arena test suite's camera and subprocess phases pass, the
Experiment Runner-specific tests pass, and headless camera and viewport recording work. OpenPI
Pi05 and GR00T N1.6-DROID both required temporary runtime compatibility changes for the GB300.

The Pi05 workaround has completed a successful DROID pick-and-place episode. The GR00T workaround
has completed a full model-backed DROID rollout, although that episode did not solve the task.

The policy compatibility environments currently live only in running containers. The temporary
setup and launch helpers are now preserved under `tools/gb300/`, but they have not yet been
integrated into the supported repository launchers or Docker images.

## Repository state

- Checkout: `/localhome/local-xyao/arena-0.3`
- Branch: `xyao/fix/v0.3_gb300`
- Upstream: `origin/xyao/fix/v0.3_gb300`
- Base commit: `ed0fd12be862078be316c73eb7cf423ba9b1c5cd`
- IsaacLab commit: `af1bab4dc173ba69b08fab779c14ead61d13fd33`
- Isaac-GR00T commit: `e29d8fc50b0e4745120ae3fb72447986fe638aa6`

Before this report and the `tools/gb300/` helpers were added, the Git worktree was clean. Generated
models, caches, and evaluation outputs were not added to Git.

## Docker versus native uv

| Runtime | State | What was exercised |
| --- | --- | --- |
| Host `uv` | `.venv` exists with Python 3.12.14 and development dependencies | Basic environment and package checks; not the basis of the final simulation results |
| Arena Docker | Clone-specific container `isaaclab_arena-latest-arena-0.3` | Isaac Sim, pytest, Experiment Runner, RL tests, policy clients, cameras, and viewport rendering |
| OpenPI Docker | Image `isaaclab_arena:openpi_server`, temporary container `arena-openpi-cuda13-probe` | CUDA 13 JAX validation and Pi05 inference on port 8000 |
| GR00T runtime | Isaac Sim Python in the Arena container plus ARM64 GR00T dependencies | N1.6-DROID inference using an eager-attention compatibility launcher on port 5555 |

Repository instructions require commands that touch Isaac Sim or Arena package runtime code to run
inside the clone-specific Arena container. For that reason, Docker is the authoritative validated
route in this report; the host `uv` environment is useful for host-side tooling and lightweight
checks.

## Docker setup

The clone-specific Arena runtime is:

```text
Container: isaaclab_arena-latest-arena-0.3
Image family: isaaclab_arena:latest
Repository mount: /workspaces/isaaclab_arena
GPU: NVIDIA GB300
Driver: 580.173.02
Architecture: aarch64
```

The container health status currently reads `unhealthy`, but Docker exec, Isaac Sim, GPU rendering,
pytest, and policy evaluations all continue to work. The health status was not used as proof of
runtime failure.

## Test results

The three documented Arena pytest phases were run inside the clone-specific Docker container.

| Phase | Selection | Result | Duration |
| --- | --- | --- | --- |
| 1 | No cameras, no subprocess | Native Isaac Sim exit 139 during the suite | Failed before JUnit finalization |
| 2 | Cameras, no subprocess | 20 passed, 1 skipped | 560.21 s |
| 3 | Subprocess tests | 53 passed | 1759.78 s |

The Phase 1 process crashed while starting
`test_achieve_cube_goal_pose_initial_state`. Its native stack included `pthread_join`,
`blas_thread_shutdown_`, `__libc_fork`, and Kit telemetry/launcher code. That test passed when run
alone. A previous Phase 1 attempt later stopped progressing at
`test_mouse_interaction_uses_d6_grab_for_current_stage`; that test also passed alone. The evidence
therefore points to an Isaac Sim process/order lifecycle problem rather than deterministic pytest
assertion failures.

The Experiment Runner tests were rerun separately:

| Phase | Result |
| --- | --- |
| No cameras | 3 passed |
| Cameras | 1 passed |
| Subprocess | 9 passed |

These tests covered typed YAML, native Hydra overrides, timestamped and exact output directories,
multiple environments, different embodiments, variations, camera-enabled runs, and graph specs.
The headless RL train/evaluate test `test_rl_train_and_eval_lift_object` also passed in Phase 3.

Full test report and logs:

```text
/tmp/arena-0.3-test-autorun-2026-09-11/REPORT.md
/tmp/arena-0.3-test-autorun-2026-09-11/arena_phase1.log
/tmp/arena-0.3-test-autorun-2026-09-11/arena_phase2.log
/tmp/arena-0.3-test-autorun-2026-09-11/arena_phase3.log
```

## OpenPI Pi05 compatibility work

### Original failure

The maintained OpenPI server image used JAX/JAXlib 0.5.3 with the CUDA 12 plugin. It could discover
the GB300 but failed on the first real inference compilation with an unsupported Blackwell/tcgen05
code-generation path.

### Temporary runtime changes

The existing OpenPI image was started as a disposable probe container and its `/.venv` was updated
without rebuilding or modifying the original image:

```text
jax==0.7.2
jaxlib==0.7.2
jax-cuda13-plugin==0.7.2
jax-cuda13-pjrt==0.7.2
numpy==2.4.6
ml-dtypes==0.6.0
flax==0.10.2
orbax-checkpoint==0.11.13
tensorstore==0.1.74
```

The CUDA 12 JAX plugin/PJRT packages were removed. Installation had to be performed outside the
OpenPI project directory with `uv --no-config` so its project-level dependency overrides did not
force `ml-dtypes==0.4.1` again.

A standalone bfloat16 JAX matrix multiplication compiled and executed on
`CudaDevice(id=0)`, compute capability 10.3. JAX prints repeated
`No SoL config found for device: NVIDIA GB300` warnings and falls back to its default configuration;
these warnings did not prevent inference.

OpenPI's pinned Orbax version expects old JAX layout class names. The following temporary shim maps
those names to the JAX 0.7.2 API before importing the policy server:

```python
import jax.experimental.layout as layout

device_local_layout = layout.Layout
layout.DeviceLocalLayout = device_local_layout
layout.Layout = layout.Format
```

### Temporary Pi05 scripts and launch command

The dependency and compatibility helpers are now preserved at:

```text
tools/gb300/prepare_openpi_cuda13.sh
tools/gb300/serve_openpi_cuda13.py
```

The live probe container was created from `isaaclab_arena:openpi_server` with GPU access, host
networking, the host OpenPI checkpoint cache mounted at `/cache/openpi`, and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`. Its primary command is `sleep infinity` so the modified
environment can be inspected. The policy server itself is the following retained exec process:

```bash
cd /app
PYTHONPATH=/app/src:/app/packages/openpi-client/src \
  /.venv/bin/python -u -c '
import jax.experimental.layout as layout
import runpy
import sys

device_local_layout = layout.Layout
layout.DeviceLocalLayout = device_local_layout
layout.Layout = layout.Format

sys.argv = [
    "scripts/serve_policy.py",
    "--port=8000",
    "policy:checkpoint",
    "--policy.config=pi05_droid_jointpos_polaris",
    "--policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos",
]
runpy.run_path("scripts/serve_policy.py", run_name="__main__")
'
```

The server readiness line was:

```text
INFO:websockets.server:server listening on 0.0.0.0:8000
```

The temporary server remains in container `arena-openpi-cuda13-probe` and is reachable at
`127.0.0.1:8000`. Stopping that disposable container loses the installed dependency changes, while
the host checkpoint cache remains available.

### Pi05 validation

The typed Experiment Definition was:

```text
isaaclab_arena_environments/experiment_configs/droid_pnp_openpi_experiment.yaml
```

One headless episode completed successfully:

| Output | Execution | Task result |
| --- | --- | --- |
| `outputs/2026-09-12_00-20-16` | Completed | Success, score 1.0, 192 steps |

A second episode was run with both recording modes enabled:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/experiment_configs/droid_pnp_openpi_experiment.yaml \
  --viz none \
  --record_camera_video \
  --record_viewport_video \
  runs.droid_pnp_openpi.rollout_limit.num_episodes=1
```

That execution completed and moved the object but did not finish placement: score 0.6667,
`object_moved_rate=1.0`, and `success_rate=0.0` over 342 steps. Policy nondeterminism means this does
not contradict the successful first episode.

The recorded output contains three policy-camera MP4 files and one viewport MP4. All are H.264,
1280x720, 15 FPS, and approximately 22.8 seconds long. FFmpeg found no black intervals, and midpoint
frames were visually inspected.

```text
outputs/2026-09-12_00-26-42/index.html
outputs/2026-09-12_00-26-42/droid_pnp_openpi/robot-cam-rebuild0-env0-external_camera_rgb-episode-0.mp4
outputs/2026-09-12_00-26-42/droid_pnp_openpi/robot-cam-rebuild0-env0-external_camera_2_rgb-episode-0.mp4
outputs/2026-09-12_00-26-42/droid_pnp_openpi/robot-cam-rebuild0-env0-wrist_camera_rgb-episode-0.mp4
outputs/2026-09-12_00-26-42/droid_pnp_openpi/viewport-rebuild0-env0-viewport-episode-0.mp4
outputs/2026-09-12_00-26-42/previews/viewport-pi05.gif
outputs/2026-09-12_00-26-42/previews/viewport-pi05-10s.png
outputs/2026-09-12_00-26-42/previews/contact-sheet.jpg
```

## GR00T compatibility work

### Original failure

The GR00T N1.6-DROID server could load through the ARM64 Isaac Sim environment, but the selected
FlashAttention path was not usable on the GB300 setup. The working prototype keeps the model and
Arena client unchanged and selects PyTorch eager attention after model construction.

### Temporary runtime changes

- PyTorch: `2.10.0+cu130`
- NumPy: `1.26.4`
- Model: `nvidia/GR00T-N1.6-DROID`
- Embodiment tag: `OXE_DROID`
- Endpoint: `127.0.0.1:5555`
- GR00T dependency path: `submodules/Isaac-GR00T/.venv-sbsa/lib/python3.12/site-packages`
- Server interpreter: `/isaac-sim/python.sh`

The compatibility launcher imports `torch` and `torchvision`, adds the ARM64 GR00T dependency
environment to `sys.path`, and wraps `Gr00tPolicy.__init__`. After the original constructor loads
the model, every module configuration exposing `_attn_implementation` is changed to `eager`.

### Temporary GR00T launcher

The exact live prototype remains inside the Arena container as `/tmp/gr00t_eager_server.py`. A
parameterized copy is now preserved as `tools/gb300/serve_gr00t_eager.py`:

```python
"""Launch GR00T using PyTorch eager attention on unsupported FlashAttention GPUs."""

import runpy
import sys

import torch
import torchvision


GR00T_ROOT = "/workspaces/isaaclab_arena/submodules/Isaac-GR00T"
GR00T_SITE_PACKAGES = f"{GR00T_ROOT}/.venv-sbsa/lib/python3.12/site-packages"

sys.path.insert(0, GR00T_SITE_PACKAGES)
sys.path.insert(0, GR00T_ROOT)

import gr00t.policy.gr00t_policy as gr00t_policy


original_init = gr00t_policy.Gr00tPolicy.__init__


def eager_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    changed = 0
    for module in self.model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
            changed += 1
    print(f"Switched {changed} loaded module configuration(s) to eager attention", flush=True)


gr00t_policy.Gr00tPolicy.__init__ = eager_init
sys.argv = [
    "gr00t/eval/run_gr00t_server.py",
    "--model-path",
    "nvidia/GR00T-N1.6-DROID",
    "--embodiment-tag",
    "OXE_DROID",
    "--device",
    "cuda",
    "--host",
    "127.0.0.1",
    "--port",
    "5555",
]
runpy.run_path(f"{GR00T_ROOT}/gr00t/eval/run_gr00t_server.py", run_name="__main__")
```

It is currently launched inside the Arena container with:

```bash
cd /workspaces/isaaclab_arena/submodules/Isaac-GR00T
NO_ALBUMENTATIONS_UPDATE=1 \
  /isaac-sim/python.sh -u /tmp/gr00t_eager_server.py
```

### GR00T validation

The model-backed DROID experiment completed its full 1,050-step episode and produced canonical
Arena result and report artifacts. It did not pick up or place the object: score 0.3333 and
`success=false`. This validates server startup, protocol readiness, observation transfer, repeated
inference, action transfer, simulation stepping, result finalization, and camera recording, but not
task success.

```text
outputs/2026-09-11_22-28-37/arena_experiment_result.json
outputs/2026-09-11_22-28-37/index.html
outputs/2026-09-11_22-28-37/droid_pnp_gr00t/robot-cam-rebuild0-env0-external_camera_rgb-episode-0.mp4
outputs/2026-09-11_22-28-37/droid_pnp_gr00t/robot-cam-rebuild0-env0-external_camera_2_rgb-episode-0.mp4
outputs/2026-09-11_22-28-37/droid_pnp_gr00t/robot-cam-rebuild0-env0-wrist_camera_rgb-episode-0.mp4
outputs/2026-09-11_22-28-37/droid_pnp_gr00t/robot-cam-rebuild0-env0-external_camera_rgb-episode-0.gif
```

## Changes made versus changes still needed

### Completed runtime changes

- Synchronized the checkout and IsaacLab submodule to the `release/0.3.0` pointers.
- Prepared the host `uv` development environment.
- Built and exercised the clone-specific Arena Docker environment.
- Created a disposable CUDA 13 OpenPI environment and verified Pi05 inference.
- Created an eager-attention GR00T launcher and verified N1.6-DROID inference.
- Verified headless policy-camera and viewport video recording.
- Converted the Pi05 viewport video to GIF and PNG previews with FFmpeg.
- Created and published the tracking branch `xyao/fix/v0.3_gb300`.

### Not yet durable

- The OpenPI CUDA 13 dependency selection is not encoded in the server Dockerfile or wrapper.
- The Orbax/JAX layout compatibility mapping is not in a maintained module or version-compatible
  dependency set.
- The GR00T eager-attention selection is preserved in `tools/gb300/serve_gr00t_eager.py`, but it is
  still a monkeypatch rather than a supported command-line option or server configuration.
- The GR00T ARM64 dependency environment is not captured by a reproducible repository workflow.
- The full Phase 1 pytest process-lifecycle crash remains unresolved.

## Recommended implementation work

1. Add a maintained GB300/CUDA 13 OpenPI image path or update the existing OpenPI pin after
   validating compatibility on other supported GPUs.
2. Replace the OpenPI layout monkeypatch with mutually compatible JAX, Flax, Orbax, TensorStore,
   NumPy, and `ml-dtypes` pins.
3. Add an explicit GR00T attention-backend option, defaulting to the upstream behavior and allowing
   `eager` on hardware where FlashAttention is unavailable.
4. Capture the ARM64 GR00T dependency installation in the supported setup or container workflow.
5. Add server startup smoke tests plus one model-backed protocol/inference check for each policy.
6. Repeat all three Arena pytest phases after the durable changes, keeping the known Phase 1 Isaac
   Sim lifecycle issue separate from deterministic test failures.

Changes under `docker/`, submodules, or shared workflow configuration require explicit approval
under this repository's contribution rules before implementation.
