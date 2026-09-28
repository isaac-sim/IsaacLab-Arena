# Syringe disposal

Dispose of every syringe in the sharps container and let
all syringes settle for 50 consecutive simulation steps. Scene layouts and goals
live in YAML. All variants include `environments/syringe_env_config.yaml` for Newton
physics settings. `environment.py` retains camera, gripper, and placement
adaptations, plus a direct assignment of `enable_multiccd`, which Isaac Lab does
not yet expose.
Policy code stays in Isaac-cap.

Assets load directly from
`{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/syringe_disposal/assets`.
These assets are available through the public staging S3 mirror.
No local asset preparation or separate robot asset is required.

External-right camera views after one second of settling (placement seed 43 for both and cluttered):

| Single | Both | Cluttered |
| --- | --- | --- |
| One red-cap syringe; fixed layout | Red-cap and white syringes; randomized layout | Four red-cap syringes; cluttered tray |
| ![Single syringe](docs/images/single.png) | ![Both syringes](docs/images/both.png) | ![Cluttered syringes](docs/images/cluttered.png) |

## Zero action

Run the cluttered variant in the GUI from the Arena repository root inside its container:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/cluttered_zero_action_experiment.yaml \
  --viz kit
```

## Isaac-cap policy — agent instructions

See [Running the CAP policy](docs/running_cap_policy.md) for a two-terminal guide,
prerequisites, result interpretation, and troubleshooting.

Run the environment in this checkout's Docker container and the policy graph in
the existing Isaac-cap checkout on the host; do not copy graph, perception, or
planning code into Arena. These instructions cover `both` and `cluttered`.

1. Use the `dev-container` and `run-experiment` skills. Confirm Nucleus access,
   Arena's optional `cap` dependencies
   (`/isaac-sim/python.sh -m pip install -e '.[cap]'` inside the container if missing),
   and CAP's configured tool environments and VLM credentials. Keep credentials
   in CAP's existing configuration. The container must use host networking.

2. From the Arena checkout on the host, start one episode with one environment:

   ```bash
   ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
   docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
     'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
       --experiment_config isaaclab_arena_environments/isaac_cap/syringe_sort/experiment_configs/cluttered_cap_remote_experiment.yaml \
       --viz none shared.environment_builder.placement_seed=43'
   ```

   Add `shared.environment.type=syringe_both_newton` for both, or use `--viz kit` for the GUI.
   Keep this process running and wait for `Completed setting up the environment`.
   Arena waits up to 180 seconds for CAP; starting CAP earlier can exhaust its
   60-second first-image timeout while the scene loads.

3. In another host terminal, from the Isaac-cap checkout, launch the graph:

   ```bash
   GAP_PORT=19000 GAP_GRAPH=local/syringe_packing_v2 \
   CAP_GAP_ROBOT_PROFILE=fr3 CAP_GAP_ARM_BASE_POSITION=-0.5,-0.1,0.912 \
   CAP_GAP_CONTROL_FREQUENCY_HZ=50 CAP_GAP_HONOURS_ROLL=0 \
   CAP_GAP_CARTESIAN_CORRECTION_LIMIT_M=0 \
   CAP_GAP_CAMERA_NAME=overhead,eye_in_hand,agentview \
   GAP_HAND_TO_FINGERTIP_Z=0.157 GAP_TCP_ROTATION_Z=0.7853981633974483 \
   ./arena_gap/scripts/run_gap_graph.sh
   ```

   For another port, set both `GAP_PORT` and Arena's `shared.policy.port` override.
   Restart the graph for each episode; the client supports one environment.

4. Wait for evaluation completion and inspect `arena_experiment_result.json`,
   `<run>/episode_results_rebuild0.jsonl`, and `index.html` in the reported output
   directory. Arena's episode result is authoritative: CAP completing its graph
   does not guarantee task success. On disconnect, Arena allows two seconds of
   settling before ending the episode. Preserve failed trials and their logs.
