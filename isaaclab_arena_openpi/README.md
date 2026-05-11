# isaaclab_arena_openpi

`Pi0DroidRemotePolicy` — a thin client for the upstream
[openpi](https://github.com/Physical-Intelligence/openpi) WebSocket policy
server, evaluated inside [Isaac Lab - Arena](../README.md) on the DROID
embodiment.

This package extends arena via the policy hook only. It defines no
environments — the existing `pick_and_place_maple_table` env (and any
other built-in arena env that uses the DROID embodiment) is reusable by
name. Same pattern `isaaclab_arena_gr00t` follows.

## Layout

```
isaaclab_arena_openpi/
├── README.md
├── __init__.py
├── policy/
│   ├── __init__.py
│   ├── pi0_droid_config.py            # constants, variant table, Args dataclass
│   └── pi0_droid_remote_policy.py     # PolicyBase subclass
├── eval_jobs_configs/
│   └── droid_pnp_pi05_jobs_config.json
└── tests/
    └── test_pi0_droid_remote_policy.py
```

## Quickstart

The eval splits across two processes: the **openpi server** (hosts the
trained pi0 model, GPU-heavy) and the **arena client** (runs the sim and
this package's policy). Run the server in one terminal, the arena client
in another.

### 1. Clone openpi (one-time)

```bash
git clone https://github.com/Physical-Intelligence/openpi
cd openpi
```

Follow the openpi README for first-time setup (it uses `uv` for env management; it'll handle the
JAX install for you on first `uv run`).

### 2. Start the openpi server

In the openpi repo:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

First launch downloads the ~11 GB checkpoint to `~/.cache/openpi/`; subsequent runs are
instant. When you see:

```
INFO:websockets.server:server listening on 0.0.0.0:8000
```

the server is ready. Leave the terminal running.

### 3. Run the arena client

In a second terminal, start the arena container:

```bash
./docker/run_docker.sh
```

Then, inside the container:

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --viz kit \
  --policy_type isaaclab_arena_openpi.policy.pi0_droid_remote_policy.Pi0DroidRemotePolicy \
  --policy_variant pi05 \
  --remote_host localhost --remote_port 8000 \
  --num_steps 2000 \
  --enable_cameras --num_envs 1 \
  --language_instruction "Pick up the Rubik's cube and place it in the bowl." \
  pick_and_place_maple_table \
    --embodiment droid_abs_joint_pos \
    --pick_up_object rubiks_cube_hot3d_robolab \
    --destination_location bowl_ycb_robolab \
    --hdr home_office_robolab
```

`--hdr home_office_robolab` matches robolab's default training-time background
(`HomeOfficeBackgroundCfg`). Without it, the scene falls back to a sterile
default backdrop that pi0 has never seen during training, which substantially
degrades success rate.

`localhost` works because arena's `run_docker.sh` runs with `--net=host`, so the
container shares the host network with your openpi server. If the server is on
a *different* machine, replace `localhost` with that machine's reachable address.

The server terminal will start logging connection + inference events. The arena
Kit window shows the droid arm reacting to pi0's commanded joint positions.

## Batch eval via `eval_runner`

For evaluating multiple object / HDR / destination combinations in one run,
the JSON jobs-config form is supported:

```bash
python isaaclab_arena/evaluation/eval_runner.py \
  --jobs_config isaaclab_arena_openpi/eval_jobs_configs/droid_pnp_pi05_jobs_config.json \
  --enable_cameras
```

Edit `remote_host` in the JSON if your server isn't on `localhost`.

## Install (image build details)

Nothing extra to install at runtime — the package ships with arena and the
`openpi-client` library is baked into the arena docker image. The relevant
pieces:

- `docker/Dockerfile.isaaclab_arena` has a gr00t-style `pip install
  --no-deps "openpi-client @ git+..."` line that pulls the openpi client at a
  pinned commit. Bump it via `--build-arg OPENPI_COMMIT=<sha>` if you need a
  different version (match the commit your openpi server is running).
- arena's top-level `setup.py` lists `isaaclab_arena_openpi*` in
  `find_packages`, so the final `pip install -e .` step inside the Dockerfile
  picks this package up alongside `isaaclab_arena_gr00t`, etc.

Inside any container built from that image, `import
isaaclab_arena_openpi.policy.pi0_droid_remote_policy` and `from openpi_client
import websocket_client_policy` both work directly.

## Why no environment in this package?

The arena env `pick_and_place_maple_table` already exposes everything pi0
needs:

| Pi0 expects (from `_extract_droid_observation`) | Env produces it via |
|---|---|
| `obs["camera_obs"]["external_camera_rgb"]` (uint8, NHWC) | `DroidCameraCfg.external_camera` |
| `obs["camera_obs"]["wrist_camera_rgb"]` (uint8, NHWC) | `DroidCameraCfg.wrist_camera` |
| `obs["policy"]["joint_pos"]` (7 panda joints) | `arm_joint_pos` term |
| `obs["policy"]["gripper_pos"]` (1-dim, scaled 0–1) | `gripper_pos` term (rescaled by π/4) |

Action shape lines up: pi0 returns `(H, 8)`, first 7 dims feed into
`JointPositionActionCfg(panda_joint.*)`, last dim into
`BinaryJointPositionZeroToOneActionCfg(finger_joint)` (which thresholds
internally at 0.5). So we reference the env by name and let arena's registry
resolve it at runtime — no env file in this package, no
`--external_environment_class_path` flag needed.

If we ever want to customise the scene (different initial joint pose,
restricted asset set, alternative camera presets) the path is to subclass
`PickAndPlaceMapleTableEnvironment` and load via
`--external_environment_class_path` per
`docs/pages/arena_in_your_repo/external_environments.rst`.

## Supported variants

| `--policy_variant` | Trained checkpoint        | Pair with                | open_loop_horizon |
|--------------------|---------------------------|--------------------------|------------------:|
| `pi05` (default)   | `pi05_droid_jointpos`     | `droid_abs_joint_pos`    | 15                |
| `pi0`              | `pi0_droid_jointpos`      | `droid_abs_joint_pos`    | 10                |
| `pi0_fast`         | `pi0_fast_droid`          | `droid_rel_joint_pos`    | 10                |

The horizon table lives in `pi0_droid_config.py`
(`OPEN_LOOP_HORIZON_BY_VARIANT`); add a new key there if you train a new
openpi droid checkpoint.

## Status

- `num_envs == 1` only. The upstream openpi server takes one observation per
  call, so the policy asserts on `env.unwrapped.num_envs` at runtime.
- Chunking is a single cached array: the last server response is replayed for
  `open_loop_horizon` steps, then refetched.
- A dropped websocket triggers up to `MAX_RECONNECT_ATTEMPTS` reconnects; on
  each reconnect the cached chunk is flushed.
- If openpi gains batched inference, lift the `num_envs == 1` assertion and
  switch to `isaaclab_arena.policy.action_scheduling.ActionChunkScheduler`
  (mirrors what `Gr00tRemoteClosedloopPolicy` does today).
