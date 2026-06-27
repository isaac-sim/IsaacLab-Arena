# GaP ↔ Isaac Lab-Arena bridge (ILA side)

This directory holds the **ILA side** of the GaP↔ILA bridge: a client that drives the
`libero_object_packing` scene over GaP's real-robot msgpack protocol, so GaP's existing grocery-packing
graph (perception → cuRobo grasp planning → control) runs against the Arena sim with no changes to GaP.

GaP is the **server**; this client **connects to it** and plays the role the real Franka would. The gap
repos (`/home/rafael/Projects/gap/*`) are read-only; only this side is built here. The companion graphs
and wire contract live in `/home/rafael/Projects/Isaac-cap`.

## Wire contract (what the client implements)

- **Transport:** TCP, GaP hosts `127.0.0.1:9000`; the client connects. Frame = 4-byte big-endian length
  + `msgpack.packb(use_bin_type=True)` with `msgpack_numpy.patch()` (numpy arrays serialize natively).
  Requires `msgpack-numpy` in the ILA venv.
- **Exchange:** request/reply, client-driven — send one observation, receive the latest action. ~50 Hz.
- **Observation (client → GaP)** under camera key `robot0_robotview` (no `eye_in_hand` substring → GaP
  treats it as the exterior view):
  ```
  { "timestamp": <float>,
    "left": { "joint_pos": [q1..q7, gripper_frac] },     # gripper 0=closed..1=open
    "robot0_robotview": {
       "images": { "rgb": uint8[H,W,3] },
       "depth_data": float32[H,W],                         # TOP-LEVEL, meters (see note)
       "intrinsics": { "left": { "intrinsics_matrix": float32[3,3] } },
       "pose_mat": float32[4,4] } }                        # camera→base, OpenCV (see note)
  ```
- **Action (GaP → client):** `{ "left": { "joint_pos": [q1..q7], "gripper": <0..1> } }` — absolute joint
  targets. Applied via the env's absolute joint-position action term; gripper maps to the binary finger
  term (≥0.5 → open, else close).
- **Reset:** GaP `reset()` blocks until the first RGB frame. Isaac Sim boots in minutes but GaP's reset
  RGB timeout is ~60 s, so the client **boots the env first, then retry-connects** — start GaP only after
  the client logs `env ready; connecting`.

### Two contract subtleties (hard-won)
- **Depth key is top-level `depth_data`**, NOT `images.depth`. GaP's `_convert_observation` reads
  `cam["depth_data"]` (franka_real_env.py:387); the contract doc showed `images.depth`. Code wins.
- **Camera pose convention (review R5 — the silent-grasp-killer):**
  `pose_mat = T_world_base⁻¹ @ T_world_cam`, mapping an **OpenCV-optical** camera-frame point
  (+X right, +Y down, +Z forward) into the **robot-base** frame. Depth must be **`distance_to_image_plane`**
  (perpendicular z-depth), not `distance_to_camera`. The base is at world `(-0.20,0,0)`; the same
  `T_world_base` builds both `pose_mat` and any GT comparison, so the offset cancels. Verified at 0.000 cm
  (analytic frame-check) and 0.91 cm (live DINO/SAM3/VLM OBB). The exterior camera is aimed agentview-style
  at runtime; because `set_world_poses_from_view` leaves `data.pos_w/quat_w_ros` stale, `pose_mat` is
  computed deterministically from `create_rotation_matrix_from_view(eye,target)` (OpenGL) with the
  OpenGL→OpenCV flip `diag(1,-1,-1)` — consistent with the render by construction.

## Scene additions (in `isaaclab_arena_environments/`)

- `libero_object_packing_environment.py`: ground-mounted Franka, LIBERO home, basket as a solver anchor.
  `--control joint_pos` swaps to the `franka_joint_pos` embodiment with an **absolute** `JointPositionActionCfg`
  (`scale=1.0, use_default_offset=False`) and **HIGH-PD** gains (the stock joint-pos cfg sags ~0.25 rad at
  the gravity-loaded joints under position control). Default stays relative IK (zero action holds pose).
- `libero_cameras.py`: `LiberoPerceptionCameraCfg` adds a fixed exterior rgb+depth camera and depth on the
  wrist cam, injected only when `--enable_cameras` (core embodiment untouched).

## Scripts (milestone harnesses)

| Script | Purpose |
|---|---|
| `m1_env_smoke.py` | env runs headless, read joints + wrist cam, report step rate |
| `m2_joint_command.py` | absolute joint command lands at `q` (not `0.5q+offset`) |
| `m3_bridge_client.py` | motion bridge: recv action → env.step → send obs (blank cam) |
| `m4_framecheck.py` | R5 frame-check: back-project objects to GT base frame (<1 cm) |
| `m4_bridge_client.py` | **the bridge client**: real exterior rgb+depth+pose_mat, joint+gripper control, GT packing log, optional mp4 recording |

## Launch

Terminal A (GaP venv) — start the server *after* the client logs `env ready`:
```
cd /home/rafael/Projects/gap/graph-as-policy
set -a; source ~/.config/gap/vlm.env; set +a
MUJOCO_GL=egl uv run gap run /home/rafael/Projects/Isaac-cap/examples/grocery_packing \
    --real franka --no-rr-autostart --no-video
```

Terminal B (ILA venv) — the client (via the untracked `dev_run.sh` wrapper = main-clone `.venv` + PYTHONPATH):
```
./dev_run.sh isaaclab_arena/scripts/cap_bridge/m4_bridge_client.py \
    --headless --num_envs 1 --enable_cameras --placement_seed 0 \
    --record_video /tmp/packing.mp4 libero_object_packing --control joint_pos
```
Env-specific flags (`--control`) go **after** the env name; bridge/top-level flags before it.

If a boot hangs on a futex (stale Kit shared memory from a SIGKILL'd run):
`rm -f /dev/shm/carb-* /dev/shm/sem.carb* /dev/shm/sem.carbonite*` then relaunch.

## Status (all GT-verified, no cheats)

| Milestone | Result |
|---|---|
| M1 env runs | ✅ ~29 Hz, joints + wrist cam |
| M2 absolute joint control | ✅ lands at `q` to 1e-4 rad |
| M3 motion bridge | ✅ observe → move_to_joints → done(success) |
| M4 R5 frame-check | ✅ 0.000 cm; perceive OBB 0.91 cm |
| M5a single pick-place | ✅ can in basket, 4.3 cm from center |
| M5b full packing loop | ✅ 5/6 packed (cream_cheese skipped: too flat), clean termination, video |

Remaining/optional: Arena-side `pack_all_into` scoring out-of-band; a second wrist `eye_in_hand` camera
for fuller clouds; converging the Docker image onto the public wheel.
