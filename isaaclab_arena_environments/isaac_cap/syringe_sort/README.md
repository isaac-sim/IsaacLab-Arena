# Syringe disposal

Run the Arena environment in this checkout's Docker container and the GaP policy
in a separate Isaac-cap checkout on the host. Both processes use port `19000`.

Four Newton environments match the syringe definitions in Isaac-cap main
(`35e74598a7630597fb48d44bd71b94228c239737`):

| Variant | Environment | Goal | Episode limit |
| --- | --- | --- | --- |
| Single | `syringe_single_newton` | Dispose of one red-cap syringe | 228 s |
| Both | `syringe_both_newton` | Dispose of two red-cap syringes | 456 s |
| Designated | `syringe_designated_newton` | Dispose of the red-cap syringe beside an unscored blank syringe | 228 s |
| Cluttered | `syringe_cluttered_newton` | Dispose of six red-cap syringes | 1368 s |

## Run with CAP

Use a running Arena container with host networking and Arena's optional `cap`
dependencies installed, plus a configured Isaac-cap checkout with its submodules,
GaP tool environments, and VLM credentials. All four variants use the same
`syringe_packing/gap_perception` graph; no separate task-source checkout is needed.
See [Running the CAP policy](docs/running_cap_policy.md#prerequisites) for setup details.

### Terminal 1: Arena client

From `/workspaces/isaaclab_arena` inside the Arena container, choose one variant
and start it first. Each command runs one episode with placement seed 42, opens
the simulator GUI, and records camera videos.

Single:

```bash
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/single_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

Both:

```bash
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/both_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

Designated:

```bash
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/designated_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

Cluttered:

```bash
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/cluttered_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

Wait for `[CapPolicy] Environment ready; waiting for GaP at 127.0.0.1:19000`
before starting Terminal 2. Arena waits up to 180 seconds. Append
`shared.policy.connect_timeout_s=600` if more startup time is needed.

### Terminal 2: GaP policy server

In another terminal on the host, run the following from your Isaac-cap checkout.
Replace `/path/to/Isaac-cap` with your checkout path. **Use this same command for
Single, Both, Designated, and Cluttered.**

```bash
cd /path/to/Isaac-cap

GAP_PORT=19000 \
GAP_GRAPH=syringe_packing/gap_perception \
CAP_GAP_ROBOT_PROFILE=fr3 \
CAP_GAP_ARM_BASE_POSITION=-0.5,-0.1,0.912 \
CAP_GAP_CONTROL_FREQUENCY_HZ=50 \
CAP_GAP_HONOURS_ROLL=0 \
CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0 \
CAP_GAP_CAMERA_NAME=overhead,eye_in_hand,agentview \
GAP_HAND_TO_FINGERTIP_Z=0.157 \
GAP_TCP_ROTATION_Z=0.7853981633974483 \
./arena_gap/scripts/run_gap_graph.sh
```

Keep the syringe-specific 50 Hz control rate and `CAP_GAP_HONOURS_ROLL=0` settings.
Restart both commands for each new trial. If Arena exits before the graph finishes,
stop the graph with Ctrl-C before starting the next trial.

Use `--viz none` for headless operation, or omit `--record_camera_video` to skip
recording. To change the layout, append
`shared.environment_builder.placement_seed=43` to the Arena command. For a different
port, change both `GAP_PORT` and Arena's `shared.policy.port` override.

Results and videos are saved under the timestamped `outputs/` directory printed
by Arena. Open `index.html` to review the run and check the episode success flag;
CAP graph completion alone does not establish task success. See the
[detailed guide](docs/running_cap_policy.md) for troubleshooting and the optional
single-syringe comparison against native CAP.

## Environment details

All variants randomize the tray, sharps container, and syringe XY positions with
fixed headings. Cluttered starts three syringes on the tray and three 3 cm above
it to settle under gravity. Invalid placement fallbacks are disabled.

Success requires every scored syringe's center of mass inside the receiver-local
bounds, linear speed at most 0.01 m/s, angular speed at most 0.05 rad/s, and the
measured Robotiq driver position within +/-0.1 rad of open. All conditions must
hold together for 50 consecutive steps at 50 Hz. The blank syringe in designated
is not scored. A completed policy graph alone does not establish success.

Scene definitions and shared Newton settings live in `environments/*.yaml`.
`SyringeCameraCfg` declares CAP's calibrated views on the shared FR3 camera rig.
The factory selects this configuration and applies the `enable_multiccd` field
that Isaac Lab does not yet expose in its configuration. The shared FR3 embodiment
supplies CAP's binary gripper commands, actuator gains, and robot reset events.
The tray spawner restores CAP's original mesh collider in the hosted asset,
which otherwise adds primitive cavity faces. Newton uses Arena's supported
`replicate_physics=True` setting.

Assets load from
`{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/syringe_disposal/assets`.
The images in `docs/images/` show the original three-variant port and are historical.

## Scripted behavior check

Inside Arena Docker:

```bash
python isaaclab_arena_environments/isaac_cap/syringe_sort/syringe_env_behaviour_demo.py \
  --variant cluttered --cycles 1 --no-real-time --visualizer none
```

Choose `single`, `both`, `designated`, or `cluttered`. This check releases the
scored syringes through the aperture in the smaller scenes and places successive
layers inside the receiver in cluttered. It lets them settle under physics and
verifies that a closed gripper prevents success, then opens the gripper and
requires a settled success reset. It validates task physics and scoring, not
policy grasping. Add `--video-dir outputs/syringe-demo` for camera recordings, or
`--visualizer kit` for interactive inspection.

The zero-action cluttered experiment remains available for scene inspection:

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/cluttered_zero_action_experiment.yaml \
  --viz kit
```
