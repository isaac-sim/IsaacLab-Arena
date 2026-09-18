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
