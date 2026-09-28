# Tool hanging

Hang wrenches or scissors on pegboard hooks, straddle pliers over their supports, or stand
screwdrivers in a pegboard box with a bimanual I2RT YAM. Success requires every tool to satisfy
its loop-on-rod or containment geometry while touching its fixture and resting, for one second.

Each scene is a graph YAML that includes `tool_hanging_env_config.yaml` for AUTOLab's Newton
profile. `assets.py`, `embodiment.py`, and `task.py` supply the registered components the
YAMLs reference. Policy code stays in Isaac-cap.

| Scene | Layout |
| --- | --- |
| `wrench_easy`, `scissors_easy`, `screwdriver_easy`, `plier_easy` | One tool in a 4 x 4 cm region at the table center; fixed heading. |
| `wrench_medium`, `scissors_medium`, `screwdriver_medium`, `plier_medium` | Two matching tools with +/-5 cm and +/-5 degree jitter, two distractors, fixed fixtures. |

Assets load from
`{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/tool_hanging/assets`.

Top-camera views of the medium scenes after one second of settling (placement seed 42):

| Wrench | Scissors | Screwdriver | Pliers |
| --- | --- | --- | --- |
| ![Wrench medium](docs/images/wrench_medium.png) | ![Scissors medium](docs/images/scissors_medium.png) | ![Screwdriver medium](docs/images/screwdriver_medium.png) | ![Pliers medium](docs/images/plier_medium.png) |

## Zero action

Run a scene in the GUI from the repository root inside its container; the config defaults to
`wrench_easy`, and the override selects any other scene YAML in this directory:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/tool_hanging/experiment_configs/tool_hanging_zero_action_experiment.yaml \
  --viz kit shared.environment.type=isaaclab_arena_environments/isaac_cap/tool_hanging/scissors_medium.yaml
```

Cameras are exposed as `top_camera`, `side_camera`, `left_wrist_camera`, and `right_wrist_camera`,
the names Isaac-cap's graph policy binds to.

## Arena client + CAP policy

Run Arena in this checkout's Docker container (host networking) and the GAP graph in a
separate Isaac-cap checkout on the host. These commands were tested against Isaac-cap
commit `587c7b047c635f64404f4b6a1a88333b362253df`.

1. **Arena client** — from the repo root inside the container:

   ```bash
   /isaac-sim/python.sh -u isaaclab_arena/evaluation/experiment_runner.py \
     --experiment_config isaaclab_arena_environments/isaac_cap/tool_hanging/experiment_configs/tool_hanging_cap_remote_experiment.yaml \
     --viz kit
   ```

   The experiment defaults to `wrench_easy`. Select another scene with a Hydra override, for
   example `shared.environment.type=isaaclab_arena_environments/isaac_cap/tool_hanging/scissors_medium.yaml`.

   Wait for `Completed setting up the environment` before starting CAP so the graph's
   first-image timeout does not fire while the scene is still loading.

2. **CAP server** — in another host terminal, from the Isaac-cap checkout:

   ```bash
   GAP_PORT=19000 \
   GAP_GRAPH=tool_hanging/gap_perception \
   GAP_ADAPTATION=vabar_tool_hanging_artist_wrench \
   CAP_GAP_ROBOT_PROFILE=yam_bimanual \
   CAP_GAP_CONTROL_FREQUENCY_HZ=60 \
   CAP_GAP_HONOURS_ROLL=1 \
   CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0.01 \
   CAP_GAP_ARM_BASE_POSES='[
     [[0.3025,0.17,0.81],[0,0,0,1]],
     [[0.3025,-0.17,0.81],[0,0,0,1]]
   ]' \
   ./arena_gap/scripts/run_gap_graph.sh \
     --inputs instruction='"hang the wrench on the hook"'
   ```

   Match `GAP_PORT` to `shared.policy.port` if you change the experiment port. Restart the
   graph for each new episode.
