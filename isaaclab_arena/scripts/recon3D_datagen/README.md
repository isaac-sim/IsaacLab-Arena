# Recon3D data generation

Script: `run_isaaclab_arena_datagen.py`. Generates per-frame camera data (RGB, depth, flow, semantics, etc.) from an Isaac Lab Arena simulation.

## How to run

Run the script with Python (or run cell-by-cell in a Jupyter-style environment). From the project root, with Isaac Lab and arena dependencies installed:

```bash
python isaaclab_arena/scripts/recon3D_datagen/run_isaaclab_arena_datagen.py
```

The script uses `# %%` cell markers; IDEs like VS Code can run it as a notebook.

## Main hyperparameters

All hyperparameters are defined at the **top** of `run_isaaclab_arena_datagen.py` (right after the license header). Edit that block to change:

| What | Variable / CLI | Description |
|------|-----------------|-------------|
| **Scene** | `"dynamic_balls"` (and optional `--object "cracker_box"`) | Environment name. `dynamic_balls` uses a static camera; pass `--enable_cameras`. Other scenes may use different CLI args. |
| **Object** | `--object "cracker_box"` | Object variant in the scene (depends on the environment). |
| **Camera position** | `CAMERA_POSITION` | World-frame (x, y, z) of the camera. |
| **Camera target** | `CAMERA_TARGET` | World-frame look-at point. |
| **Image size** | `IMAGE_WIDTH`, `IMAGE_HEIGHT` | Output resolution (e.g. 640×480). |
| **Focal length** | `FOCAL_LENGTH` | In mm (e.g. 24.0). |
| **Output directory** | `OUTPUT_DIR` | Root folder where `cam0/`, `cam1/`, … are written. |
| **Number of steps** | `NUM_STEPS` | Simulation steps (frames) to record (e.g. 30). |
| **Occlusion tolerance** | `OCCLUSION_TOL` | Depth tolerance in metres for the visible-now mask (e.g. 0.1). |

---

## Output folder layout

Each run writes data under `OUTPUT_DIR`. Each camera has its own subfolder (e.g. `cam0/`, `cam1/`). Within a camera folder, data is organised by modality. Files use 10-digit zero-padded frame indices (e.g. `0000000000.png`, `0000000001.npy`).

```
(output root)/
├── cam0/
│   ├── color/         RGB images
│   ├── depth/         Depth maps
│   ├── flow2d/        Optical flow (2D)
│   ├── flow3d/        Adjacent-frame 3D scene flow
│   ├── flow3d_track_type/   Track type for adjacent flow
│   ├── flow3d_from_first/   First-frame-anchored 3D flow
│   ├── trackable_mask/      Trackable mask (frame-0 anchors)
│   ├── in_frame_mask/       In-frame projection mask
│   ├── visible_now_mask/    Visible-now (occlusion) mask
│   ├── normal/         Surface normals
│   ├── intrinsic/      Camera intrinsic matrices
│   ├── extrinsic/      Camera-to-world matrices
│   ├── semantic/       Semantic segmentation + metadata
│   └── visualizations/ Pre-rendered visualisations (if generated)
├── cam1/
│   └── ...
```

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

### First-frame-anchored flow (one file per frame, indices 0 .. N-1)

Flow from frame-0 to current frame: `flow_0k = p_k - p_0`.

| Subfolder   | Format | Description |
|------------|--------|-------------|
| **flow3d_from_first** | .npy | 3D displacement frame-0 → current (H×W×3), metres. |
| **trackable_mask** | PNG | Pixels with GT tracking (Static/Rigid/Articulation). |
| **in_frame_mask** | PNG | Reconstructed point projects inside current image. |
| **visible_now_mask** | PNG | In-frame and depth-consistent (not occluded). |

Relationship: `trackable_mask ⊇ in_frame_mask ⊇ visible_now_mask`.

### Frame index convention

- Static modalities and first-frame flow: frame index = step index (0, 1, …, N-1).
- Adjacent flow: stored at the *source* frame index; file at i describes motion from frame i to frame i+1.

### Visualisations

If the script runs the visualisation step, each camera gets a `visualizations/` subfolder with:

- **data_vis.png** — Grid of sampled frames. Column order (left to right): color, depth, normals, track type, flow2d, flow3d, semantics, flow-from-first, in-frame mask, visible-now mask.
- **camera_trajectory_3d.png** — 3D camera path with coordinate frames.
- **scene_flow_3d_frame*.html**, **first_frame_flow_3d_frame*.html** — Interactive 3D scene-flow views.
