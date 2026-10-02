# Cable routing v2

Run the Arena environment in this checkout's Docker container and the GaP policy in a separate
Isaac-cap checkout on the host. Both processes use port `19000`.

## Get the policy sources

The policy uses graph, robot skill, perception, and motion code from the private
[AUTOLab-GaP-NVIDIA-Collaboration repository](https://github.com/BerkeleyAutomation/AUTOLab-GaP-NVIDIA-Collaboration).
You need access to that repository and GitHub SSH authentication configured for the commands below.

Run this setup once on the host, in the CAP terminal. Replace the path with your Arena checkout.
These commands populate `outputs/cap_sources` with separate copies pinned to the Easy and Medium
revisions. This folder is Git-ignored and is not included when you clone Arena.

```bash
export ARENA_SOURCE_ROOT=/absolute/path/to/IsaacLab-Arena
mkdir -p "$ARENA_SOURCE_ROOT/outputs/cap_sources"

# Easy
git clone --no-checkout git@github.com:BerkeleyAutomation/AUTOLab-GaP-NVIDIA-Collaboration.git \
  "$ARENA_SOURCE_ROOT/outputs/cap_sources/cable_easy_ae6de67"
git -C "$ARENA_SOURCE_ROOT/outputs/cap_sources/cable_easy_ae6de67" checkout --detach ae6de67

# Medium
git clone --no-checkout git@github.com:BerkeleyAutomation/AUTOLab-GaP-NVIDIA-Collaboration.git \
  "$ARENA_SOURCE_ROOT/outputs/cap_sources/cable_medium_bf985d0"
git -C "$ARENA_SOURCE_ROOT/outputs/cap_sources/cable_medium_bf985d0" checkout --detach bf985d0
```

Keep `ARENA_SOURCE_ROOT` set in the CAP terminal when running the policy commands below.

## Arena client

From the Arena repository root inside the container, choose one environment and start it first:

```bash
# Easy
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/cable_routing_v2/experiment_configs/cable_easy_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit

# Medium
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/cable_routing_v2/experiment_configs/cable_medium_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

Wait until the environment is ready before starting GaP.

## GaP policy server

From the Isaac-cap repository root in another terminal, run the command matching the Arena variant.

Easy uses the `ae6de67` task source and all three policy cameras:

```bash
TASKS_ROOT="$ARENA_SOURCE_ROOT/outputs/cap_sources/cable_easy_ae6de67" \
GAP_PORT=19000 \
GAP_GRAPH=cable_route/weave \
CAP_GAP_ROBOT_PROFILE=yam_i2rt \
CAP_GAP_ARM_BASE_POSITIONS='[[-0.335,-0.31,0.767],[-0.335,0.31,0.767]]' \
CAP_GAP_CONTROL_FREQUENCY_HZ=60 \
CAP_GAP_HONOURS_ROLL=0 \
CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0 \
CAP_GAP_CAMERA_NAME=overhead,wrist,cable \
./arena_gap/scripts/run_gap_graph.sh
```

Medium uses the `bf985d0` task source, its clearance adaptation, and only the cable camera:

```bash
TASKS_ROOT="$ARENA_SOURCE_ROOT/outputs/cap_sources/cable_medium_bf985d0" \
GAP_PORT=19000 \
GAP_GRAPH=cable_route/weave \
GAP_ADAPTATION=vabar_cable_medium_clearance \
CAP_GAP_ROBOT_PROFILE=yam_i2rt \
CAP_GAP_ARM_BASE_POSITIONS='[[-0.335,-0.31,0.767],[-0.335,0.31,0.767]]' \
CAP_GAP_CONTROL_FREQUENCY_HZ=60 \
CAP_GAP_HONOURS_ROLL=0 \
CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0 \
CAP_GAP_CAMERA_NAME=cable \
./arena_gap/scripts/run_gap_graph.sh
```

Restart the graph for each new episode.
