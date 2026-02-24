# NaVILA VLN Benchmark for IsaacLab Arena

**Vision-Language Navigation (VLN)** benchmark using the [NaVILA](https://github.com/AnjieCheng/NaVILA) VLM on the H1 humanoid robot in Matterport 3D indoor scenes.

## Architecture

```
 ISAAC SIM CONTAINER                       NAVILA SERVER CONTAINER
 ══════════════════                        ═══════════════════════

 ┌──────────────┐  ┌───────────────────┐
 │ Matterport   │  │ VLN-CE R2R Data   │
 │ Scene (.usd) │  │ (.json.gz)        │
 └──────┬───────┘  └───────┬───────────┘
        │                  │
        ▼                  ▼
 ┌──────────────┐  ┌───────────────────┐
 │ H1 Robot     │  │VlnR2rMatterport   │
 │ + Head Cam   │  │Task               │
 │ + Follow Cam │  │ scene_filter      │
 └──────┬───────┘  │ termination       │
        │          │ metrics (SPL...)  │
        │          └───────┬───────────┘
        │                  │
        ▼                  ▼
 ┌─────────────────────────────────┐
 │  VlnVlmLocomotionPolicy        │
 │                                 │
 │  ┌─ High-level ──────────────┐  │
 │  │  Send 1 RGB frame ────────┼──┼──► ZeroMQ ──►┐
 │  │  Receive vel_cmd ◄────────┼──┼──◄ ZeroMQ ◄──┤
 │  └────────────┬──────────────┘  │              │
 │               │                 │              │
 │  ┌────────────▼──────────────┐  │              │
 │  │  Low-level: RSL-RL        │  │              │
 │  │  Inject vel_cmd → obs     │  │              │
 │  │  NN forward → joint act   │  │              │
 │  └────────────┬──────────────┘  │              │
 └───────────────┼─────────────────┘              │
                 │                                │
                 ▼                                │
 ┌─────────────────────────────┐    ┌─────────────▼────────────┐
 │  PhysX GPU Simulation       │    │  NaVilaServerPolicy      │
 │  joint act → robot moves    │    │                          │
 │  → new camera RGB           │    │  1. Receive 1 RGB frame  │
 └─────────────────────────────┘    │  2. Append to history    │
                                    │  3. Uniform sample 8     │
                                    │     from full history    │
                                    │  4. VLM inference        │
                                    │  5. Parse → vel_cmd      │
                                    │  6. Return vel_cmd + dur │
                                    └──────────────────────────┘
```

## Quick Start

### 1. Start the NaVILA VLM Server

```bash
# Option A: Use the launch script (rebuilds Docker if needed)
bash docker/run_vln_server.sh -m ~/models/navila/navila-llama3-8b-8f --port 5555

# Option B: Manual start with code mounted (no rebuild)
docker run --rm -d --gpus all --net host \
    --name vln_policy_server_container \
    -v ~/models:/models \
    -v ~/IsaacLab-Arena/isaaclab_arena_navila:/workspace/isaaclab_arena_navila \
    -v ~/IsaacLab-Arena/isaaclab_arena/remote_policy:/workspace/isaaclab_arena/remote_policy \
    vln_policy_server:latest \
    --host 0.0.0.0 --port 5555 --timeout_ms 15000 \
    --policy_type isaaclab_arena_navila.navila_server_policy.NaVilaServerPolicy \
    --model_path /models/navila/navila-llama3-8b-8f
```

Wait for: `listening on tcp://0.0.0.0:5555`

### 2. Run VLN Evaluation (Client)

```bash
# Enter Isaac Sim container
bash docker/run_docker.sh

# Inside the container:
/isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
    --enable_cameras --num_envs 1 \
    --policy_type isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy \
    --remote_host localhost --remote_port 5555 \
    --ll_checkpoint_path isaaclab_arena/policy/vln/pretrained/h1_navila_locomotion.pt \
    --ll_agent_cfg isaaclab_arena/policy/vln/pretrained/h1_navila_agent.yaml \
    --num_episodes 10 \
    h1_vln_matterport \
    --usd_path /path/to/matterport_usd/zsNo4HB9uLZ/zsNo4HB9uLZ.usd \
    --r2r_dataset_path /path/to/vln_ce_isaac_v1.json.gz
```

Add `--headless` for batch evaluation without GUI.

## Matterport Scene Collision

The default VLN-CE-Isaac Matterport USD scenes are primarily **visual meshes**.
They render correctly, but they do not provide reliable PhysX collision by
default. The current benchmark therefore uses an **overlay collision** approach:

- keep the original Matterport visual USD for rendering
- generate one or more collision-only USDA assets
- attach those collision USDA assets under `/World/matterport`
- let PhysX collide with the overlay mesh instead of trusting the visual mesh

By default, if no collision overlay is provided, an invisible ground plane at
`z=0` provides basic support so the robot can stand and walk.

### Why the default Matterport USD is not enough

The common failure mode is that the visual USD is a poor proxy for stable
collision:

1. raw scan / mesh -> visual USD
2. visual USD -> OBJ
3. OBJ -> collision USDA

If geometric detail is already lost or distorted in step 1, the later collision
asset inherits that loss. In practice this leads to:

- thin door-frame shells
- floating or near-ground fragments
- furniture-adjacent blue shells that do not match the visible structure
- floor noise that destabilizes humanoid locomotion

For better collision fidelity, prefer starting from a **cleaner raw Matterport /
MP3D mesh** instead of relying only on the already-converted visual USD.

### Collision overlay modes

The runtime currently supports three collision-overlay styles:

- **legacy combined overlay**
  - one collision-only USDA under the visual scene
- **split overlay**
  - `floor` collision for locomotion support
  - `obstacle` collision for walls, cabinets, sofas, tables, and door frames
- **explicit child-mesh colliders**
  - apply collision schemas directly to the visual USD descendant meshes

The split `floor + obstacle` route is usually the most practical for debugging
and tuning.

### Recommended asset layout

For each scene, keep:

- visual scene:
  - `matterport_usd/<scene>/<scene>.usd`
- floor collision:
  - `collision/<scene>_floor_collision.usda`
- obstacle collision:
  - `collision/<scene>_obstacle_collision.usda`
- optional combined collision:
  - `collision/<scene>_combined_collision.usda`

Put intermediate OBJ files and temporary logs under `claude_tmp/`.

### Build workflow

#### 1. Export raw mesh to OBJ

If you have the original raw MP3D / Matterport mesh, export or convert it to
OBJ first:

```bash
/isaac-sim/python.sh isaaclab_arena/scripts/assets/export_matterport_collision_proxy.py \
    --input_path /path/to/raw_mp3d_mesh.glb \
    --output_path /tmp/scene_raw.obj
```

If you do not have the raw mesh yet, you can still use the current visual USD as
a fallback, but collision quality will usually be worse.

#### 2. Clean / simplify the OBJ externally

Before building collision assets, the most useful cleanup is:

- remove tiny floating fragments
- remove ceiling-only details that do not matter to locomotion
- smooth or rebuild the floor if it is noisy or broken
- simplify small decorative clutter
- keep large walls, cabinets, sofas, tables, and door frames

Try to make the floor cleaner than the obstacle mesh. For humanoid locomotion,
floor quality usually matters more than visual fidelity.

#### 3. Build split floor / obstacle USDA files

Use the builder:

```bash
python3 isaaclab_arena/scripts/assets/build_matterport_collision_layers.py \
    --input_obj /tmp/scene_clean.obj \
    --output_floor_usd /path/to/collision/scene_floor_collision.usda \
    --output_obstacle_usd /path/to/collision/scene_obstacle_collision.usda \
    --output_combined_usd /path/to/collision/scene_combined_collision.usda \
    --floor_min_z -0.25 \
    --floor_max_z 0.35 \
    --floor_normal_min_z 0.9
```

This split is heuristic and scene-dependent. The key idea is:

- floor-like triangles: low enough and upward-facing
- everything else: obstacle

#### 4. Filter obstacle fragments

After the initial split, a collision mesh is often still too noisy. The new
`filter_collision_usda.py` script removes connected components that look like:

- tiny near-ground shards
- small floating fragments
- optionally, medium-sized suspended shells after a second pass

Example first pass:

```bash
python3 isaaclab_arena/scripts/assets/filter_collision_usda.py \
    --input_usd /path/to/collision/scene_obstacle_collision.usda \
    --output_usd /path/to/collision/scene_obstacle_collision_clean.usda \
    --ground_max_faces 20 \
    --ground_max_top_z 0.30 \
    --ground_max_diag 0.35 \
    --floating_max_faces 10 \
    --floating_min_bottom_z 0.35 \
    --floating_max_diag 0.25
```

This stage is what removes the obvious “bad collision geometry” that makes the
robot snag, trip, or appear to hit empty space.

### Runtime configurations

#### A. Stable baseline

Keep the invisible ground plane, add only obstacle collision:

```bash
/isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
    ... \
    h1_vln_matterport \
    --usd_path /path/to/scene.usd \
    --r2r_dataset_path /path/to/vln_ce_isaac_v1.json.gz \
    --use_global_matterport_prim \
    --matterport_obstacle_collision_usd_path /path/to/scene_obstacle_collision.usda
```

Use this first when debugging torso clipping, arm clipping, or corridor contact
without destabilizing the feet.

#### B. Split full collision

Disable the fallback ground plane and use both floor + obstacle collision:

```bash
/isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
    ... \
    h1_vln_matterport \
    --usd_path /path/to/scene.usd \
    --r2r_dataset_path /path/to/vln_ce_isaac_v1.json.gz \
    --use_global_matterport_prim \
    --disable_matterport_ground_plane \
    --matterport_floor_collision_usd_path /path/to/scene_floor_collision.usda \
    --matterport_obstacle_collision_usd_path /path/to/scene_obstacle_collision.usda
```

Use this after the floor mesh looks smooth enough to replace the fallback ground
plane.

#### C. Legacy combined overlay

This mode is still supported:

```bash
/isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
    ... \
    h1_vln_matterport \
    --usd_path /path/to/scene.usd \
    --r2r_dataset_path /path/to/vln_ce_isaac_v1.json.gz \
    --use_global_matterport_prim \
    --disable_matterport_ground_plane \
    --matterport_collision_usd_path /path/to/scene_combined_collision.usda
```

Do not mix the legacy combined overlay with the split `floor / obstacle`
overlays in the same run.

### Visual inspection in Isaac Sim

The recommended local inspection tool is:

- `isaaclab_arena/scripts/assets/debug_matterport_collision_stage.py`

Use it to export a debug stage and reopen it in Isaac Sim. When checking the
result, verify:

1. Stage paths exist:
   - `/World/matterport`
   - `/World/matterport/collision`
   - `/World/matterport/collisionFloor`
   - `/World/matterport/collisionObstacle`
   - `/World/GroundPlane`
2. Collision visualization is enabled:
   - eye icon -> `Show by type` -> `Physics Mesh` -> `All`
   - `Tools` -> `Physics API Editor`
   - `Window` -> `Simulation` -> `Debug`
3. Geometry alignment looks reasonable:
   - obstacle shells are not drifting away from walls or furniture
   - floor meshes are not floating above or below the visible floor
   - door frames are not dominated by noisy thin slivers
   - the fallback ground plane does not fight a custom floor mesh

### Validation ladder

Use this order when debugging a new scene:

1. Visual USD + ground plane only
2. Ground plane + obstacle collision
3. Floor + obstacle collision, no ground plane

This isolates whether failures come from:

- obstacle mesh quality
- floor mesh quality
- low-level locomotion mismatch

### Common failure modes

#### Robot clips through furniture

Likely causes:

- obstacle collision is too coarse
- only the ground plane is active
- upper-body posture is too wide for corridor / furniture clearance

What to try:

- enable obstacle-only collision first
- keep large furniture and wall boundaries in the obstacle mesh
- reduce decorative clutter that adds noisy contacts but does not help body clearance

#### Robot gets stuck and stops moving

Likely causes:

- floor mesh is noisy or non-manifold
- floor overlay and ground plane both support the robot in conflicting ways
- `robot_root_height_offset` is wrong
- the low-level locomotion policy was not trained for contact-rich indoor mesh geometry

What to try:

- keep the ground plane and add only obstacle collision first
- rebuild a cleaner floor mesh
- tune `--robot_root_height_offset`
- compare against a ground-plane-only baseline

#### Robot snags on door frames or thin geometry

Likely causes:

- obstacle collision is too detailed for the current locomotion controller
- the robot body is wider than the assumed corridor

What to try:

- simplify sharp mesh details near doors
- test `triangle` vs `convex_decomposition`
- compare with and without obstacle collision

### Handoff checklist

If another machine or another person needs to continue the work, keep these
unchanged:

- the visual Matterport USD
- the VLN episode file such as `vln_ce_isaac_v1.json.gz`
- the existing Arena / NaVILA evaluation pipeline

Replace or improve only:

- the collision-only assets used by PhysX

For the other machine, the most useful assets are:

- `matterport_usd/<scene>/<scene>.usd`
- `vln_ce_isaac_v1.json.gz`
- a cleaner raw MP3D / Matterport mesh
- generated collision outputs:
  - `<scene>_floor_collision.usda`
  - `<scene>_obstacle_collision.usda`
  - optional `<scene>_combined_collision.usda`

In this branch, the collision-related code and docs to carry forward are:

- `isaaclab_arena/assets/matterport_background.py`
- `isaaclab_arena_environments/vln_environment.py`
- `isaaclab_arena/scripts/assets/export_matterport_collision_proxy.py`
- `isaaclab_arena/scripts/assets/build_matterport_collision_layers.py`
- `isaaclab_arena/scripts/assets/filter_collision_usda.py`
- `isaaclab_arena/scripts/assets/debug_matterport_collision_stage.py`
- `isaaclab_arena/scripts/assets/open_usd_stage.py`

Collision-related CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--use_global_matterport_prim` | off | Spawn Matterport at `/World/matterport` with `collision_group=-1`. Required for collision overlay and height scanner. |
| `--disable_matterport_ground_plane` | off | Remove the invisible z=0 ground plane. Use when a collision overlay provides floor support. |
| `--matterport_collision_usd_path` | None | Legacy combined collision-only USD layered under the visual scene. |
| `--matterport_floor_collision_usd_path` | None | Floor-only collision USD layered under the visual scene. |
| `--matterport_obstacle_collision_usd_path` | None | Obstacle-only collision USD layered under the visual scene. |
| `--enable_matterport_child_mesh_colliders` | off | Explicitly apply collision schemas to descendant Mesh prims of the visual USD. |
| `--matterport_mesh_collider_type` | `triangle` | Mesh collider approximation: `triangle`, `sdf`, or `convex_decomposition`. |
| `--robot_root_height_offset` | 1.0 | Added to dataset `start_position.z` when resetting the robot (H1 pelvis is ~1m above the floor). |

## Training Your Own Low-Level Locomotion Policy

The VLN pipeline uses a pre-trained RSL-RL locomotion policy that converts
velocity commands `[vx, vy, yaw_rate]` into joint-position actions. A default
checkpoint is included at `isaaclab_arena/policy/vln/pretrained/`.

To train a replacement checkpoint with the built-in rough-terrain environment:

```bash
docker run --rm --gpus all --net host --entrypoint /bin/bash \
  -v "$(pwd):/workspaces/isaaclab_arena" \
  -v "/home/$USER:/home/$USER" \
  isaaclab_arena:latest -lc "
    cd /workspaces/isaaclab_arena && \
    /isaac-sim/python.sh -u isaaclab_arena/scripts/reinforcement_learning/train.py \
      --headless \
      --num_envs 4096 \
      --max_iterations 1000 \
      --save_interval 50 \
      --experiment_name h1_base_rough \
      --agent_cfg_path isaaclab_arena/policy/rl_policy/h1_base_rough_policy.json \
      h1_base_rough
  "
```

Checkpoints are saved under `logs/rsl_rl/h1_base_rough/<timestamp>/`.
To use a newly trained checkpoint for VLN evaluation, point to it with:

```bash
--ll_checkpoint_path logs/rsl_rl/h1_base_rough/<timestamp>/model_<iter>.pt
--ll_agent_cfg logs/rsl_rl/h1_base_rough/<timestamp>/params/agent.yaml
```

The training environment preserves the same 69-dim proprioceptive layout as the
VLN evaluation, so the velocity-command injection at `obs[:, 9:12]` remains
compatible without code changes.

## Sensor Configuration

### Camera Positions (CLI-configurable)

| Camera | Default Position (pelvis frame) | FOV | CLI Override |
|--------|--------------------------------|-----|-------------|
| Head camera | `(0.1, 0.0, 0.5)` | 54 deg | `--head_camera_offset_xyz X Y Z` |
| Follow camera | `(-1.0, 0.0, 0.57)` | 100 deg | `--follow_camera_offset_xyz X Y Z` |

### Optional Sensors

| Flag | Description |
|------|-------------|
| `--enable_head_camera_depth` | Add depth output to the head camera (not sent to VLM by default). |
| `--enable_height_scanner` | Enable a raycast-based height scanner on the pelvis. Requires `--use_global_matterport_prim`. |
| `--height_scanner_debug_vis` | Visualize the height-scanner rays in the viewport. |
| `--camera_resolution N` | Camera resolution in pixels (default: 512). |
| `--no_follow_camera` | Disable the third-person follow camera. |
| `--use_tiled_camera` | Use TiledCamera for parallel multi-env evaluation. |

### STOP Diagnostic Flags

| Flag | Description |
|------|-------------|
| `--ignore_vlm_stop` | Never let the VLM STOP command terminate an episode. Useful for diagnosing early-stop behaviour. |
| `--min_vlm_stop_distance D` | Ignore VLM STOP when distance-to-goal exceeds `D` meters. Negative value (default) disables. |
| `--debug_vln` | Enable verbose VLM query logging and debug frame saving. |
| `--save_interval N` | Save camera frames every N steps to `--save_dir`. |
| `--save_dir DIR` | Directory for saved camera frames. |

## CLI Parameters

### Client Parameters (before `h1_vln_matterport`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_envs` | 1 | Must be 1 (multi-env not yet supported). |
| `--policy_type` | (required) | `isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy` |
| `--remote_host` | localhost | VLM server hostname. |
| `--remote_port` | 5555 | VLM server port. |
| `--ll_checkpoint_path` | (required) | RSL-RL locomotion checkpoint (.pt). |
| `--ll_agent_cfg` | (required) | RSL-RL agent config (.yaml). |
| `--warmup_steps` | 200 | Low-level policy warmup steps. |
| `--num_episodes` | (required) | Number of episodes to evaluate. |

### Environment Parameters (after `h1_vln_matterport`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--usd_path` | (required) | Path to Matterport USD scene file. |
| `--r2r_dataset_path` | (required) | Path to VLN-CE R2R dataset (.json.gz). |
| `--episode_start` | None | Start index within filtered episodes. |
| `--episode_end` | None | End index (exclusive) within filtered episodes. |
| `--episode_length_s` | 60 | Max episode duration in seconds. |
| `--success_radius` | 3.0 | Success distance threshold (meters). |

### Server Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | (required) | Path to NaVILA / LLaVA model checkpoint. |
| `--port` | 5555 | Server listening port. |
| `--num_video_frames` | 8 | Number of frames sent to VLM per query. |
| `--conv_mode` | llama_3 | LLaVA conversation template. |
| `--history_padding_mode` | repeat_first | Early-frame padding: `repeat_first` or `black` (NaVILA-Bench style). |
| `--max_history_frames` | 200 | Maximum frames kept in server-side history. |
| `--max_new_tokens` | 80 | Maximum generated tokens per VLM query. |

## Metrics

| Metric | Description |
|--------|-------------|
| **Success** | Fraction of episodes where the agent emits `STOP` and the final distance-to-goal is below `success_radius`. |
| **SPL** | Mean over episodes of `S_i * l_i / max(p_i, l_i)`, where `S_i` is STOP-gated success, `l_i` is shortest-path distance, and `p_i` is actual path length. |
| **Distance-to-Goal** | Approximate geodesic distance from the final position to the goal in the XY plane, estimated from the reference path. |
| **Path Length** | Total distance traversed (XY only). |

The benchmark follows a `DONE/STOP`-style evaluation protocol: without an explicit `STOP`, an episode is counted as unsuccessful even if the agent passes near the goal.

## Known Limitations

- **`num_envs` must be 1.** Multi-env requires per-env VLM instruction tracking.
- **Scene switching requires restart.** Matterport USD is loaded at initialization.
- **Some episode start positions are outside the building envelope.** The
  Matterport scan may have incomplete or visually degraded geometry at these
  locations, which can confuse the VLM and cause the robot to get stuck.
  Consider filtering edge-case episodes or adjusting `--episode_start` / `--episode_end`.
- **Collision proxy is experimental.** The exported proxy mesh may not perfectly
  match the original building geometry. Use the default ground plane for stable
  baseline evaluation.
