# isaaclab_arena_openpi

`Pi0DroidRemotePolicy` — a thin client for the upstream
[openpi](https://github.com/Physical-Intelligence/openpi) WebSocket policy
server, evaluated inside [Isaac Lab - Arena](../README.md) on the DROID
embodiment.

## Quickstart

The eval splits across two processes: the **openpi server** (hosts the
trained pi0 model, GPU-heavy) and the **arena client** (runs the sim and
this package's policy). Run the server in one terminal, the arena client
in another.

### 1. Build the openpi server image (one-time)

```bash
./isaaclab_arena_openpi/docker/build_openpi_server.sh
```

Clones upstream openpi at a pinned commit and builds
`isaaclab_arena_openpi-server:<short-sha>` (also tagged `:latest`). The pinned
commit matches `OPENPI_COMMIT` in `docker/Dockerfile.isaaclab_arena` so client
and server speak the same wire format. ~3 min, ~19 GB image.

### 2. Start the openpi server

Pick a variant and launch the container. The first launch downloads the
~11 GB checkpoint into the container layer; subsequent runs reuse it.

**pi05** (default):

```bash
docker run --rm -it --gpus all --network=host \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  isaaclab_arena_openpi-server:latest \
  uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos_polaris \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

**pi0**: swap the two paths to `pi0_droid_jointpos_polaris` and
`gs://openpi-assets-simeval/pi0_droid_jointpos` respectively.

When you see:

```
INFO:websockets.server:server listening on 0.0.0.0:8000
```

the server is ready. Leave the terminal running.

> **Config vs weights.** `--policy.config` declares the architecture +
> data transforms (we use upstream's `_polaris` configs). `--policy.dir`
> declares where to load params + normalization stats from. Weights are
> loaded from `--policy.dir`, and norm stats found there override the
> config-baked path. We point at `gs://openpi-assets-simeval/...` because
> empirically those checkpoints grasp more reliably than the
> `gs://openpi-assets/checkpoints/polaris/...` weights the polaris
> configs default to (same Pi0 architecture, but trained without polaris
> cotrain data).

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

If the server is on a *different* machine, replace `localhost` with that machine's reachable address.
The server terminal will start logging connection + inference events.
The arena IsaacSim window shows the droid arm reacting to pi0's commanded joint positions.

## Supported variants

| `--policy_variant` | `--policy.config`                | `--policy.dir`                                       | Pair with             | open_loop_horizon |
|--------------------|----------------------------------|------------------------------------------------------|-----------------------|------------------:|
| `pi05` (default)   | `pi05_droid_jointpos_polaris`    | `gs://openpi-assets-simeval/pi05_droid_jointpos`     | `droid_abs_joint_pos` | 15                |
| `pi0`              | `pi0_droid_jointpos_polaris`     | `gs://openpi-assets-simeval/pi0_droid_jointpos`      | `droid_abs_joint_pos` | 10                |
| `pi0_fast`         | `pi0_fast_droid_jointpos_polaris`| `s3://openpi-assets-simeval/pi0_fast_droid_jointpos` *(untested — needs s3fs in the image)* | `droid_rel_joint_pos` | 10 |

The horizon table lives in `pi0_droid_config.py`
(`OPEN_LOOP_HORIZON_BY_VARIANT`); add a new key there if you train a new
openpi droid checkpoint.
