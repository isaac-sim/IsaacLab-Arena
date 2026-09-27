# Run the CAP policy against Arena

Arena runs the simulation in its Docker container; the Isaac-cap checkout runs
the policy graph on the host. They exchange observations and commands over
localhost port `19000`. Policy, perception, and planning code stay in Isaac-cap.

## Prerequisites

- A running Arena development container with this checkout mounted at
  `/workspaces/isaaclab_arena` and host networking enabled.
- Arena's optional `cap` dependencies installed in that container. If missing,
  run `/isaac-sim/python.sh -m pip install -e '.[cap]'` from the mounted repo root,
  as your host user.
- A configured Isaac-cap checkout, including its
  GaP runtime, tool environments, and VLM credentials. The graph launcher reads
  `~/.config/gap/vlm.env`; keep credentials there, not in Arena.
- Port `19000` free. Run one environment and one episode per graph process.

The commands below use the local `syringe_packing_v2` graph from Isaac-cap's
`alex/syringe-tested-baseline` branch. Replace `/path/to/isaac_arena` and
`/path/to/Isaac-cap` below with your checkout locations.

## Terminal 1: start the Arena simulation

Run on the host from the Arena checkout:

```bash
cd /path/to/isaac_arena
ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
test -n "$ARENA_CONTAINER" || { echo "Start this checkout's Arena container first"; exit 1; }

docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  'cd /workspaces/isaaclab_arena && \
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   /isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/cluttered_cap_remote_experiment.yaml \
     --viz none \
     shared.environment_builder.placement_seed=43'
```

This runs **cluttered** (four syringes). For **both** (red-cap and white), append
`shared.environment.type=syringe_both_newton` inside the quoted command. The Run
name in the output remains `cluttered`; check its `environment.name` to identify
the actual variant. Use `--viz kit` instead of `--viz none` for the GUI.

Wait for `Completed setting up the environment...`, then start Terminal 2.
Arena waits up to 180 seconds for CAP. Starting CAP before the scene is ready
can exhaust its first-image timeout. Cameras are enabled by the experiment even
with `--viz none`.

## Terminal 2: start the policy graph

Run on the host, outside the Arena container:

```bash
cd /path/to/Isaac-cap
GAP_PORT=19000 GAP_GRAPH=local/syringe_packing_v2 \
CAP_GAP_ROBOT_PROFILE=fr3 \
CAP_GAP_ARM_BASE_POSITION=-0.5,-0.1,0.912 \
CAP_GAP_CONTROL_FREQUENCY_HZ=50 CAP_GAP_HONOURS_ROLL=0 \
CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0 \
CAP_GAP_CAMERA_NAME=overhead,eye_in_hand,agentview \
GAP_HAND_TO_FINGERTIP_Z=0.157 GAP_TCP_ROTATION_Z=0.7853981633974483 \
./arena_gap/scripts/run_gap_graph.sh
```

These values match Arena's FR3 base pose, control rate, camera aliases, and
Robotiq tool transform. Keep them together when reproducing this setup.
For another port, change both `GAP_PORT` and Arena's `shared.policy.port` override.

## Results and repeat runs

Arena prints a timestamped directory under `outputs/`. Inspect:

- `arena_experiment_result.json`: Run status and each episode's `success`.
- `cluttered/episode_results_rebuild0.jsonl`: per-episode results.
- `index.html`: evaluation report. These commands do not record video.

Arena's episode result is authoritative: a completed CAP graph does not by itself
mean task success. If CAP disconnects, Arena allows two seconds of settling before
ending the episode. Conversely, Arena can finish successfully before the graph
finishes retracting; stop that graph with Ctrl-C after Arena exits. Restart both
commands for each new episode, changing the placement seed if desired.

CAP prints its trace directory under `Isaac-cap/outputs/gap/`. Preserve that trace
and both terminal logs when investigating failures. If Arena cannot connect,
check `ss -ltnp 'sport = :19000'`, host networking, and the graph startup log.

The syringe YAML leaves `replicate_physics` at Arena's Newton setting (`True`).
The runtime prints `Replicate physics : True`; do not add a `false` override,
which Isaac Lab documents as unsupported for Newton. This policy client still
supports only one environment, independently of physics replication.
