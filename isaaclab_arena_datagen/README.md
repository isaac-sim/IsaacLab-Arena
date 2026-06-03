<!--
Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# `isaaclab_arena_datagen` — Synthetic data generation

Generates per-frame camera data (RGB, depth, normals, semantics, optical/scene
flow) plus dynamic-object poses and mesh samples from an Isaac Lab Arena
simulation and writes it to a single HDF5 file in the **SyntheticScene** schema.

It does this in **two modes**:

1. **Standalone scene generation** — step a curated scene with zero actions and
   record every frame ([`run_datagen.py`](run_datagen.py)). This is the port of
   the original `nvblox_next/datagen` pipeline.
2. **Policy-rollout collection** — record the *same* data while a policy drives
   the environment, via an opt-in hook in
   [`isaaclab_arena/evaluation/policy_runner.py`](../isaaclab_arena/evaluation/policy_runner.py)
   ([`collection/collector.py`](collection/collector.py)).

Both modes share the per-step recording logic in [`pipeline.py`](pipeline.py),
so they capture identical data.

## Self-contained by design

This package has **no dependency on `nvblox_next` and no dependency on
`pytorch3d`**. The SyntheticScene HDF5 writer is reimplemented directly on
`h5py` + `hdf5plugin` ([`io/hdf5_writer.py`](io/hdf5_writer.py)), and the SE(3)
geometry is a small torch-only reimplementation ([`geometry/`](geometry)). The
output schema is unchanged, so files remain loadable by `nvblox_next`'s
`SyntheticSceneFlowDataset` / `SyntheticSceneRGBDDataset`.

## Package layout

| Path | Purpose |
|------|---------|
| `run_datagen.py` | Standalone entry point: CLI parser, `AppLauncher`, `main()`. |
| `pipeline.py` | Shared per-step recording (`record_camera_step`, `run_simulation_loop`, `save_dynamic_objects`) + `SimDataCollectionSetup`. |
| `collection/collector.py` | `DatagenCollector` — the policy-runner hook. |
| `io/hdf5_writer.py` | `DatagenHDF5Writer` — self-contained SyntheticScene HDF5 writer (h5py + hdf5plugin). |
| `io/hdf5_keys.py` | HDF5 dataset / group / attribute name constants. |
| `geometry/` | Torch-only `Rotation` / `Translation` / `TransformSE3` (no pytorch3d). |
| `camera_handler.py` | Wraps an Isaac Lab `Camera`; RGB/depth/normals/flow/semantics + `create_static_camera`. |
| `camera_trajectory.py` | `CameraViewTrajectory` dataclass (static or dynamic view). |
| `object_registry.py` | Maps sim prims to stable semantic IDs / names / colors. |
| `scene_flow.py` | Exact scene-flow computation from cached point clouds. |
| `dynamic_object_tracker.py` | Per-step object poses + mesh-surface sampling. |
| `visualizer.py` | Optional PNG/MP4/HTML visualizations from an HDF5 file. |
| `environments/` | Datagen scene classes + `register_datagen_environments()`. |
| `scene_metadata.py` | Scene naming / on-disk-path conventions (single source of truth). |
| `generate_all_scenes.py` | Batch-generate every scene into the canonical layout. |
| `utils/` | Constants, camera utilities, mesh + transform helpers. |
| `tests/` | Pure-logic unit tests (geometry, writer schema) runnable without Isaac Sim. |

## Mode 1 — standalone scene generation

```bash
# Single scene (run inside the isaaclab_arena container; python == /isaac-sim/python.sh)
python -m isaaclab_arena_datagen.run_datagen \
    --headless --enable_cameras \
    --output-dir /datasets/dynamic_scenes/miscellaneous/ball_box_robot \
    --num-steps 30 \
    ball_box_robot
```

Cameras are **force-enabled** by `run_datagen.py` (the pipeline always renders
sensors). All other parameters have sensible defaults — run with `--help`.

Generate every scene (4 reference + 80 categorized) into the canonical layout:

```bash
python -m isaaclab_arena_datagen.generate_all_scenes            # -> /datasets/dynamic_scenes
python -m isaaclab_arena_datagen.generate_all_scenes \
    --output-root /data/my_runs --base-num-steps 60 --scene-num-steps 100 --visualizations
```

### CLI arguments (`run_datagen.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `scene_name` | — | Datagen scene name (positional, e.g. `ball_box_robot`, `lemon`). |
| `--output-dir` | — (required) | Folder where `dataset.h5` (and visualizations) are written. |
| `--num-steps` | `30` | Simulation steps (frames) to record. Must be > 1. |
| `--width` / `--height` | `640` / `480` | Image size in pixels, shared by all cameras. |
| `--dynamic-translation-eps` | `1e-4` | Per-step translation threshold (m) for "dynamic" classification. |
| `--dynamic-rotation-eps` | `1e-3` | Per-step rotation threshold (rad). |
| `--mesh-sample-spacing` | `0.01` | Mesh surface sample spacing (m). |
| `--visualizations` | off | Also write per-scene PNG grid, MP4, and HTML plots. |
| `--headless` | off | Run Isaac Sim without a GUI window. |

Camera configs are provided by each scene class via `get_default_cameras(num_steps)`
(returning `CameraViewTrajectory` objects), not the CLI. Resolution is global
because the HDF5 format requires all cameras to share spatial dimensions.

## Mode 2 — collect data during a policy rollout

Pass `--collect-datagen` to the policy runner. Data is recorded from
**dedicated** static datagen cameras (independent of the policy's observation
cameras) into `{--datagen-output-dir}/dataset.h5`:

```bash
python -m isaaclab_arena.evaluation.policy_runner \
    --enable_cameras \
    --policy_type <registered_policy_or_dotted.path> \
    --num_steps 100 \
    --collect-datagen \
    --datagen-output-dir /eval/datagen \
    <example_environment> [env args...]
```

> **Important:** `--collect-datagen` / `--datagen-output-dir` are *global* runner
> flags, so they must appear **before** the `example_environment` subcommand
> (alongside `--policy_type`, `--enable_cameras`, …), not after it. Placing them
> after the subcommand makes argparse route them to the environment subparser,
> which rejects them as unrecognized.

Requirements:

- **`--enable_cameras`** is required (sensor rendering is impossible otherwise);
  the runner asserts it.
- A fixed recording horizon (the writer pre-allocates `num_frames`):
  - `--num_steps N` → records exactly `N` frames (cleanest; one continuous sequence).
  - `--num_episodes K` → records up to `K * max_episode_length` frames contiguously
    **across episode resets**; episodes that finish early leave trailing frames
    unused. Prefer `--num_steps` when you want a clean fixed-length trajectory.
- Single environment (`num_envs == 1`).

Collection is **off by default**; without `--collect-datagen` the rollout is
byte-for-byte unchanged. The collector hook is non-invasive: `rollout_policy`
calls `collector.on_step(...)` after each step and `collector.finalize(...)` at
the end (both no-ops when no collector is passed).

Datagen-collection CLI flags:

| Argument | Default | Description |
|----------|---------|-------------|
| `--collect-datagen` | off | Enable collection during the rollout. |
| `--datagen-output-dir` | `/eval/datagen` | Output folder for `dataset.h5`. |
| `--datagen-width` / `--datagen-height` | `640` / `480` | Datagen camera image size. |
| `--datagen-mesh-sample-spacing` | `0.01` | Mesh surface sample spacing (m). |

Camera viewpoints: if the `example_environment` class defines
`get_default_cameras`, those views are used; otherwise a single sensible default
view is used. To customise programmatically, build a `DatagenCollectorConfig`
with explicit `cameras=[CameraViewTrajectory(...)]`.

## Output

Each run writes a single `dataset.h5` (plus an optional `visualizations/` dir).

```
dataset.h5
└── sequence_000000/                     (attrs: sequence_id, num_frames, camera_ids, object_ids)
    ├── cam0/                            (attrs: height, width)
    │   ├── color             (N, H, W, 3)  uint8
    │   ├── depth             (N, H, W)     float32   (metres, distance to image plane)
    │   ├── intrinsic         (N, 3, 3)     float32
    │   ├── extrinsic         (N, 3, 4)     float64   [R|t] camera-to-world
    │   ├── normal            (N, H, W, 3)  float32   (world-space)
    │   ├── semantic          (N, H, W)     int32     (1-based; 0 = background)
    │   ├── semantic_json     (N,)          str
    │   ├── flow2d            (N-1, H, W, 2) float32  (dx, dy pixels; index i: frame i->i+1)
    │   ├── flow3d            (N-1, H, W, 3) float32  (scene flow, metres)
    │   └── flow3d_track_type (N-1, H, W)   uint8     (0=Static,1=Rigid,2=Articulation,255=Unsupported)
    ├── cam1/ ...
    └── dynamic_objects/                 (attrs: metadata_json, object_ids)
        ├── poses/<object>   (N, 3, 4)    float32   per-step [R|t] world pose
        └── mesh_samples/<object> (P, 3, 4) float32 SE(3) from object centre to each surface sample
```

### Coordinate conventions

All SE(3) variables follow the **`T_destination_from_source`** naming
convention: `p_world = T_world_from_camera @ p_camera`. Composition reads
right-to-left with matching inner frame labels.

## Dependencies

Required (declared in the repo `setup.py`): `h5py`, `hdf5plugin` (Zstd
filter) — plus the repo's existing `torch` / `numpy`. Optional, only for
`--visualizations` and mesh sampling (lazily imported), installed via the
`datagen-viz` extra: `plotly`, `scipy`, `imageio`, `trimesh`.

## Testing

```bash
# Pure-logic tests (geometry, HDF5 schema) — no Isaac Sim required:
python -m pytest isaaclab_arena_datagen/tests -v
```

Isaac Sim imports in the recording modules are deferred to the methods/entry
points that need them, so the geometry and writer tests run in plain CI.
End-to-end runs (`run_datagen.py`, `--collect-datagen`) require the
`isaaclab_arena` container.
