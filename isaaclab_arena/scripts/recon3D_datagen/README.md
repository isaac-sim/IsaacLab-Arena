# Recon3D data generation

Script: `run_isaaclab_arena_datagen.py`. Generates per-frame camera data (RGB, depth, flow, semantics, etc.) from an Isaac Lab Arena simulation.

## How to run

Run the script with Python (or run cell-by-cell in a Jupyter-style environment). From the project root, with Isaac Lab and arena dependencies installed:

```bash
python isaaclab_arena/scripts/recon3D_datagen/run_isaaclab_arena_datagen.py
```

The script uses `# %%` cell markers; IDEs like VS Code can run it as a notebook.

## Main hyperparameters

All hyperparameters are defined at the **top** of `run_isaaclab_arena_datagen.py` (right after the license header). Edit that block to change.

### Camera configuration

Each camera is a dict with keys `position`, `target`, `width`, `height`, `focal_length`. Define one dict per viewpoint (`CAM0`, `CAM1`, …) and list the ones to use in `CAMERAS`:

```python
CAM0 = {"position": (0.0, -0.737, 1.0), "target": (0.466, -0.737, 0.4),
        "width": 640, "height": 480, "focal_length": 24.0}
CAMERAS = [CAM0]  # CAM1, CAM2... for multi-view  
```

#### Static vs dynamic position and target

`position` and `target` can be **static** or **dynamic** per camera, and they are independent (e.g. static position with dynamic target, or vice versa).

- **Static**: a single 3D tuple `(x, y, z)` in world frame. The camera keeps that coordinate for all `NUM_STEPS`.
- **Dynamic**: a **list** of `NUM_STEPS` 3D tuples, one per simulation step. The script asserts that the list length equals `NUM_STEPS` so mismatches fail early.

Example: camera that starts at the first position and slides horizontally to the right while always looking at the same target:

```python
NUM_STEPS = 30
_c1_x_start, _c1_x_end = 0.0, 0.4
CAM1 = {
    "position": [
        (_c1_x_start + (_c1_x_end - _c1_x_start) * t / max(NUM_STEPS - 1, 1), -0.337, 0.8)
        for t in range(NUM_STEPS)
    ],
    "target": (0.466, -0.737, 0.4),  # static look-at
    "width": 600, "height": 400, "focal_length": 12.0,
}
CAMERAS = [CAM0, CAM1]
```

### Other parameters

| Variable | Description |
|----------|-------------|
| `SCENE_NAME` | Environment name (e.g. `"dynamic_balls"`). |
| `OBJECT_NAME` | Object variant in the scene. |
| `OUTPUT_DIR` | Root folder where `cam0/`, `cam1/`, … are written. |
| `NUM_STEPS` | Simulation steps (frames) to record (e.g. 30). |
| `OCCLUSION_TOL` | Depth tolerance in metres for the visible-now mask (e.g. 0.1). |
| `ANCHOR_FRAMES` | List of frame indices for anchored 3D flow (default `[0]`, e.g. `[0, 4, 6]`). |
| `DYNAMIC_MOTION_EPS` | Motion threshold (metres/radians) for classifying an object as dynamic (default `1e-4`). |

---

## Output folder layout

Each run writes data under `OUTPUT_DIR`. Each camera has its own subfolder (e.g. `cam0/`, `cam1/`). Within a camera folder, data is organised by modality. Files use 10-digit zero-padded frame indices (e.g. `0000000000.png`, `0000000001.npy`).

```
(output root)/
├── dynamic_objects.json         Metadata for dynamic objects (types, names, array keys)
├── dynamic_objects_poses.npz   SE(3) pose arrays for dynamic objects
├── cam0/
│   ├── color/                  RGB images
│   ├── depth/                  Depth maps
│   ├── normal/                 Surface normals
│   ├── flow2d/                 Optical flow (2D). From current frame to next.
│   ├── flow3d/                 Adjacent-frame 3D scene flow. From current frame to next.
│   ├── flow3d_track_type/      Track type for adjacent flow
│   ├── flow3d_from_frame0/     Anchor-frame-0 3D flow (ANCHOR_FRAMES=[0]). From anchor frame to current frame.
│   ├── trackable_mask_frame0/  Trackable mask for anchor 0
│   ├── semantic/               Semantic segmentation + metadata
│   ├── in_frame_mask_frame0/   In-frame mask for anchor 0
│   ├── visible_now_mask_frame0/ Visible-now mask for anchor 0
│   ├── intrinsic/              Camera intrinsic matrices
│   ├── extrinsic/              Camera-to-world matrices
│   └── visualizations/         Pre-rendered visualisations (if generated)
├── cam1/
│   └── ...
```

All anchor frames use the consistent `*_frame{N}` naming pattern. Each anchor frame in `ANCHOR_FRAMES` gets its own set of subfolders.

## Data types

### Static modalities (one file per frame, indices 0 .. N-1)

| Subfolder   | Format | Description |
|------------|--------|-------------|
| **color**  | PNG    | RGB image (H×W×3), uint8. |
| **depth**  | .npy   | Depth map (H×W), float32, metres, distance to image plane. |
| **intrinsic** | .npy | 3×3 camera intrinsic matrix. |
| **extrinsic** | .npy | 4×4 camera-to-world homogeneous transform. |
| **normal** | .npy   | Surface normals (H×W×3), float32, world-space x,y,z. |
| **semantic** | .png + .json | 4-channel RGBA segmentation; JSON lists visible objects. |

### Adjacent-frame flow (indices 0 .. N-2; last frame has no forward flow)

| Subfolder   | Format | Description |
|------------|--------|-------------|
| **flow2d** | .npy   | Dense optical flow (H×W×2), (dx, dy) in pixels. Index i: frame i → frame i+1. |
| **flow3d** | .npy   | 3D scene flow (H×W×3), world-space displacement in metres. |
| **flow3d_track_type** | .npy + .png | Per-pixel track type: 0=Static, 1=Rigid, 2=Articulation, 255=Unsupported. |

### Anchor-frame flow (one file per frame, from anchor frame to N-1)

Flow from anchor frame A to current frame k: `flow_Ak = p_k - p_A`.

By default (`ANCHOR_FRAMES = [0]`), flow is computed from frame 0 to all subsequent frames. Multiple anchor frames can be specified (e.g. `[0, 4, 6]`), in which case flow from each anchor is only written for frames `≥ anchor`.

| Subfolder | Format | Description |
|-----------|--------|-------------|
| **flow3d_from_frame{N}** | .npy | 3D displacement anchor N → current (H×W×3), metres. |
| **trackable_mask_frame{N}** | PNG | Pixels with GT tracking (Static/Rigid/Articulation). |
| **in_frame_mask_frame{N}** | PNG | Reconstructed point projects inside current image. |
| **visible_now_mask_frame{N}** | PNG | In-frame and depth-consistent (not occluded). |

Relationship: `trackable_mask ⊇ in_frame_mask ⊇ visible_now_mask`.

### Dynamic object poses (hybrid: JSON + `.npz`)

Two files at the output root record world-frame SE(3) poses for every dynamic object (rigid or articulated) that was visible in at least one camera at some time step and exhibited actual motion above `DYNAMIC_MOTION_EPS`. Poses are recorded at **all** time steps regardless of visibility.

- **`dynamic_objects.json`** — metadata only (object names, types, body indices, and the key to look up each pose array in the `.npz`).
- **`dynamic_objects_poses.npz`** — NumPy archive. Each entry is a `(num_steps, 3, 4)` float32 array storing the 3x4 `[R|t]` portion of the SE(3) matrix at every step (the omitted last row is always `[0, 0, 0, 1]`).

Example JSON structure:

```json
{
  "metadata": { "num_steps": 30, "motion_threshold": 0.0001, ... },
  "objects": {
    "rigid_object_1_cracker_box": {
      "type": "rigid",
      "asset_name": "cracker_box",
      "pose_array_key": "rigid_object_1_cracker_box"
    },
    "articulated_object_1_robot": {
      "type": "articulation",
      "asset_name": "robot",
      "parts": {
        "base_link": { "body_index": 0, "pose_array_key": "articulated_object_1_robot/base_link" },
        "shoulder_link": { "body_index": 1, "pose_array_key": "articulated_object_1_robot/shoulder_link" }
      }
    }
  }
}
```

Loading poses in Python:

```python
import json, numpy as np
meta = json.load(open("dynamic_objects.json"))
poses = np.load("dynamic_objects_poses.npz")
# Rigid object poses at all steps: (N, 3, 4) array
box_poses = poses[meta["objects"]["rigid_object_1_cracker_box"]["pose_array_key"]]
```

### Frame index convention

- Static modalities: frame index = step index (0, 1, …, N-1).
- Anchor-frame flow: files exist from the anchor frame to N-1. For anchor A, files at indices A, A+1, …, N-1 describe the displacement from frame A to that frame. Frames before A have no data for that anchor.
- Adjacent flow: stored at the *source* frame index; file at i describes motion from frame i to frame i+1.

### Visualisations

If the script runs the visualisation step, each camera gets a `visualizations/` subfolder with:

- **data_vis.png** — Grid of sampled frames. Column order (left to right): color, depth, normals, track type, flow2d, flow3d, semantics, flow-from-first, in-frame mask, visible-now mask.
- **camera_trajectory_3d.png** — 3D camera path with coordinate frames.
- **scene_flow_3d_frame*.html**, **first_frame_flow_3d_frame*.html** — Interactive 3D scene-flow views.
