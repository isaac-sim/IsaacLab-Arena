# isaaclab_arena_dreamzero

DreamZero remote policy integration for Isaac Lab-Arena.

`DreamZeroRemotePolicy` connects to a running DreamZero inference server over WebSocket + MessagePack, sends observations in DreamZero's flat wire format, and replays the returned action chunks step-by-step.

## Prerequisites

The DreamZero inference server must be running and reachable before launching the policy runner. The client connects eagerly at construction time and will raise a `ConnectionRefusedError` if the server is not up.

## Running

All global and policy-specific flags must appear **before** the environment name (subcommand). Flags like `--embodiment` that are specific to the environment go after it.

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy \
    --enable_cameras \
    --num_episodes 5 \
    --headless \
    --language_instruction "Pick up the cube and place it in the bowl." \
    pick_and_place_maple_table \
    --embodiment droid_abs_joint_pos
```

With the Kit viewport open, omit `--headless` and add `--viz kit`:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy \
    --enable_cameras \
    --num_episodes 5 \
    --viz kit \
    --language_instruction "Pick up the cube and place it in the bowl." \
    pick_and_place_maple_table \
    --embodiment droid_abs_joint_pos
```

Run inside the container:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
    "cd /workspaces/isaaclab_arena && <command above>"
```

## Configuration

All options have defaults matching the DreamZero wire protocol. Only override what differs from your setup.

| Flag | Default | Description |
|------|---------|-------------|
| `--dreamzero_host` | `localhost` | Hostname of the DreamZero inference server |
| `--dreamzero_port` | `5000` | Port the server listens on |
| `--dreamzero_open_loop_horizon` | `24` | Action steps replayed per server inference call |
| `--dreamzero_num_arm_joints` | `7` | Arm DOF count; remainder of `robot_joint_pos` is treated as gripper |
| `--dreamzero_cam_exterior_left` | `external_camera_rgb` | Arena camera key → `observation/exterior_image_0_left` |
| `--dreamzero_cam2_source` | `black` | Source for `observation/exterior_image_1_left`: `black`, `duplicate`, `right`, or `head` |
| `--dreamzero_cam_exterior_right` | `external_camera_2_rgb` | Camera used when `cam2_source=right` |
| `--dreamzero_cam_head` | `head_camera` | Camera used when `cam2_source=head` |
| `--dreamzero_cam_wrist` | `wrist_camera_rgb` | Arena camera key → `observation/wrist_image_left` |
| `--policy_device` | `cuda` | Torch device for the returned action tensor |

## Batch evaluation (eval_runner)

Use the dotted import path as `policy_type` and pass config fields directly in `policy_config_dict`:

```json
{
  "name": "dreamzero_pick_and_place",
  "arena_env_args": {
    "enable_cameras": true,
    "environment": "pick_and_place_maple_table",
    "embodiment": "droid_abs_joint_pos"
  },
  "num_episodes": 5,
  "language_instruction": "Pick up the cube and place it in the bowl.",
  "policy_type": "isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy",
  "policy_config_dict": {
    "remote_host": "localhost",
    "remote_port": 5000
  }
}
```

Pass this file to:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/eval_runner.py \
    --eval_jobs_config <path/to/config.json>
```

## Observation requirements

The environment must expose these keys in its observation dict:

- `observation["camera_obs"][cam_exterior_left]` — uint8 RGB tensor `(num_envs, H, W, 3)`
- `observation["camera_obs"][cam_wrist]` — uint8 RGB tensor `(num_envs, H, W, 3)`
- `observation["policy"]["robot_joint_pos"]` — float tensor `(num_envs, num_arm_joints + 1)`

Images are resized to `180 × 320` with letterbox padding before being sent to the server.
