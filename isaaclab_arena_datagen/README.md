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
cameras) into nested per-episode folders under `{--datagen-output-dir}`:

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

**One HDF5 file per episode.** The collector splits the rollout at episode
boundaries and writes `episode_0000/dataset.h5`,
`episode_0001/dataset.h5`, … under the output dir, each trimmed to that
episode's exact frame count. Isaac Lab resets a done env *within* `step()`, so
the collector treats a `done` step's frame as the next episode's first frame and
resets scene flow at each boundary (no spurious cross-reset flow). Works in both
`--num_steps` and `--num_episodes` modes.

Requirements:

- **`--enable_cameras`** is required (sensor rendering is impossible otherwise);
  the runner asserts it.
- Single environment (`num_envs == 1`).

Collection is **off by default**; without `--collect-datagen` the rollout is
byte-for-byte unchanged. The collector hook is non-invasive: `rollout_policy`
calls `collector.on_step(...)` after each step and `collector.finalize(...)` at
the end (both no-ops when no collector is passed).

### Mode 3 — collect during an eval-runner job sweep

`eval_runner` runs a JSON config of jobs; add a top-level `datagen` block to
collect per-episode data for every job (a per-job `datagen` block overrides it).
Files are written to `{output_dir}/{job_name}/episode_NNNN/dataset.h5`.

```bash
python isaaclab_arena/evaluation/eval_runner.py \
    --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/pick_and_place_datagen_jobs_config.json
```

```jsonc
{
  "datagen": {
    "output_dir": "/datasets/dynamic_scenes/openpi",
    "camera_position": [1.36, 0.0, 1.0],   // overhead look-at camera (target defaults to origin)
    "camera_target":   [0.0, 0.0, 0.0],
    "focal_length_mm": 14,
    "width": 640, "height": 480,
    "mesh_sample_spacing": 0.01
  },
  "jobs": [ { "name": "...", "num_episodes": 10, "arena_env_args": { "enable_cameras": true, ... }, ... } ]
}
```

Cameras are auto-enabled when a `datagen` block is present. Omit the `datagen`
block to disable collection (jobs run unchanged).

#### Multiple cameras

To capture several views of the same scene, use a `cameras` list instead of the
single `camera_position`. Each entry needs a `position`; `target` (defaults to
the origin) and `focal_length_mm` (defaults to the block-level `focal_length_mm`
or 24 mm) are optional. All cameras share the block-level `width`/`height` and
are written into the **same** `episode_NNNN/dataset.h5` as `cam0`, `cam1`, … :

```jsonc
"datagen": {
  "output_dir": "/datasets/dynamic_scenes/openpi_multicam",
  "width": 640, "height": 480,
  "cameras": [
    { "position": [1.36, 0.0, 1.0],  "target": [0.0, 0.0, 0.0], "focal_length_mm": 14 },
    { "position": [0.05, 0.57, 0.66], "target": [0.0, 0.0, 0.0], "focal_length_mm": 14 }
  ]
}
```

All cameras must share `width`/`height` (the SyntheticScene format requires it);
for views at different resolutions, run separate jobs/configs.

#### Random hemisphere of cameras

To surround the robot with `N` cameras at a fixed distance instead of listing
each one, use a `cameras_hemisphere` block. It places `num_cameras` cameras
**randomly** on the 180-degree hemisphere facing `front_dir` around `center`,
all at the same `radius` and all looking at `center`. The layout is
**re-randomised on every run** (so each invocation samples new positions) unless
you pin it with `seed`:

```jsonc
"datagen": {
  "output_dir": "/datasets/dynamic_scenes/openpi_hemisphere",
  "width": 640, "height": 480,
  "cameras_hemisphere": {
    "num_cameras": 10,            // how many cameras (cam0 .. cam9 in one dataset.h5)
    "radius": 1.5,                // equal distance from the robot (m)
    "center": [0.0, 0.0, 0.3],    // look-at point = robot / workspace centre
    "front_dir": [1.0, 0.0, 0.0], // hemisphere faces this way (cameras in front, looking back)
    "focal_length_mm": 14,
    "min_height": 0.1,            // reject samples below this world z (no under-floor cameras)
    "randomize_per_episode": true // re-sample a new layout for EVERY episode (see below)
    // "seed": 0                  // optional: pin for a reproducible layout (ignored per-episode)
  }
}
```

Notes:
- **`randomize_per_episode`** (default `false`): when `true`, a fresh random
  layout is drawn for *every episode* — the same N cameras are re-aimed in place
  via `set_world_pose` at each episode reset (no sensors are re-spawned). With
  `false`, the layout is sampled once per job and reused for all its episodes.
- Without `randomize_per_episode`, each job in a run still gets its **own** fresh
  layout; set `seed` to make every job share one fixed layout. (`seed` is ignored
  when `randomize_per_episode` is on, since each episode must differ.)
- `front_dir` is the direction the cameras sit on relative to `center` — set it
  to whatever counts as "in front of" your robot (e.g. `[0,1,0]` if the front is
  +y). With `min_height` the lower part of the hemisphere is trimmed so no camera
  ends up under the ground plane.

Datagen-collection CLI flags:

| Argument | Default | Description |
|----------|---------|-------------|
| `--collect-datagen` | off | Enable collection during the rollout. |
| `--datagen-output-dir` | `/eval/datagen` | Output folder for the per-episode directories. |
| `--datagen-width` / `--datagen-height` | `640` / `480` | Datagen camera image size. |
| `--datagen-mesh-sample-spacing` | `0.01` | Mesh surface sample spacing (m). |
| `--datagen-camera-position X Y Z` | — | Camera world position (look-at). Overrides the env/default view. |
| `--datagen-camera-target X Y Z` | world origin | Look-at point. Optional; defaults to `(0, 0, 0)`. |
| `--datagen-focal-length` | `24.0` | Camera focal length (mm). |

Camera viewpoints, in priority order:

1. **CLI look-at** — give `--datagen-camera-position X Y Z` (and optionally
   `--datagen-camera-target X Y Z`, default origin; `--datagen-focal-length`). E.g.:

   ```bash
   --datagen-camera-position 1.36 0.0 1.0 --datagen-camera-target 0.0 0.0 0.4 --datagen-focal-length 14
   ```

2. **Environment-provided** — if the `example_environment` class defines
   `get_default_cameras`, those views are used.
3. **Fallback** — a single elevated front-oblique default view.

To customise programmatically, build a `DatagenCollectorConfig` with explicit
`cameras=[CameraViewTrajectory(position=..., target=..., focal_length_mm=...)]`.

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
