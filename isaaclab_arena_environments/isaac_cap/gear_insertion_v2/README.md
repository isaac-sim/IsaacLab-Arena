# Gear insertion v2

Run the Arena environment in this checkout's Docker container and the GaP policy in a separate
Isaac-cap checkout on the host. Both processes use port `19000`.

## Arena client

From the Arena repository root inside the container, choose one environment and start it first:

```bash
# Easy
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/gear_insertion_v2/experiment_configs/gear_easy_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit

# Easy pair
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/gear_insertion_v2/experiment_configs/gear_easy_pair_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit

# Medium
/isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/gear_insertion_v2/experiment_configs/gear_medium_train_cap_remote_experiment.yaml \
  --record_camera_video \
  --viz kit
```

Wait until the environment is ready before starting GaP.

## GaP policy server

From the Isaac-cap repository root in another terminal:

```bash
GAP_PORT=19000 \
GAP_GRAPH=gear_mesh/gap_perception \
CAP_GAP_ROBOT_PROFILE=fr3 \
CAP_GAP_ARM_BASE_POSITION=-0.5,-0.1,0.912 \
CAP_GAP_CONTROL_FREQUENCY_HZ=60 \
CAP_GAP_HONOURS_ROLL=1 \
CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0 \
CAP_GAP_CAMERA_NAME=overhead \
GAP_HAND_TO_FINGERTIP_Z=0.157 \
GAP_TCP_ROTATION_Z=0.7853981633974483 \
./arena_gap/scripts/run_gap_graph.sh
```

The gear policy uses only the overhead camera. Restart the graph for each new episode.
