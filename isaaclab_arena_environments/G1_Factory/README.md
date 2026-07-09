# G1 Factory environments

This package contains the G1 Factory task port. The task and robot-specific code is intentionally colocated under
`isaaclab_arena_environments/G1_Factory/` so the port builds on top of Arena main without modifying shared Arena task or
policy code.

## What is included

- G1 Factory task definitions for box lift, drill lift, drill pick-and-place, drill-from-holder, push-button, and
  push-shelf-forward tasks.
- Per-task GR00T closed-loop policy translation configs in `policy_configs/benchmark/`.
- A run path for Arena main's current `Gr00tRemoteClosedloopPolicy`, which talks to a GR00T policy server.

The zero-action policy is useful only as an environment/action-space smoke test. It does not represent a valid G1
Factory controller, and the robot can twitch or settle poorly because the zero vector is not a meaningful PINK/WBC
command.

## GR00T checkpoint mapping

Arena main's GR00T remote policy config handles observation/action translation. The model checkpoint is loaded by the
GR00T server process, so checkpoint paths are documented here instead of added as unsupported policy YAML fields.

| Environment | Policy config | Legacy checkpoint |
| --- | --- | --- |
| `LMBoxLift` | `policy_configs/benchmark/LMBoxLift_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/D1-100/g1_benchmark_LMBoxLiftD1_100_gn1_5_output/checkpoint-20000` |
| `LMBoxLiftFloor` | `policy_configs/benchmark/LMBoxLiftFloor_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/D1-100/g1_benchmark_LMBoxLiftFloorD1_100_gn1_5_output/checkpoint-20000` |
| `LMBoxTableToShelfPnP` | `policy_configs/benchmark/LMBoxTableToShelfPnP_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/D1-100/g1_benchmark_LMBoxTableToShelfPnPD1_100_gn1_5_output/checkpoint-20000` |
| `LMDrillLift` | `policy_configs/benchmark/LMDrillLift_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/D1-100/g1_benchmark_LMDrillLiftD1_100_gn1_5_output/checkpoint-20000` |
| `LMDrillLiftObs` | `policy_configs/benchmark/LMDrillLiftObs_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/g1_benchmark_LMDrillLiftObs-0629_processed100_gn1_5_output_6_29/checkpoint-20000` |
| `LMDrillPnP` | `policy_configs/benchmark/LMDrillPnP_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/D1-100/g1_benchmark_LMDrillPnPD1_100_gn1_5_output/checkpoint-20000` |
| `LMPickDrillFromHolder` | `policy_configs/benchmark/LMPickDrillFromHolder_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/g1_benchmark_LMPickDrillFromHolder-0610_processed100_gn1_5_output_6_10/checkpoint-20000` |
| `LMPushButton` | `policy_configs/benchmark/LMPushButton_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/D1-100/g1_benchmark_LMPushButtonD1_100_gn1_5_output/checkpoint-20000` |
| `LMPushShelfForward` | `policy_configs/benchmark/LMPushShelfForward_gr00t_closedloop.yaml` | `/mnt/data/isaaclab_arena/ckpts/D1-100/g1_benchmark_LMPushShelfForwardD1_100_gn1_5_output/checkpoint-20000` |

## Run a real GR00T policy

Start the GR00T policy server from an environment that has GR00T installed and can access the checkpoint. Replace
`<checkpoint>` with the task checkpoint from the table above.

```bash
python gr00t/eval/run_gr00t_server.py \
  --model_path <checkpoint> \
  --embodiment_tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port 5555
```

Then run the Arena client from this repository. This example uses `LMDrillLift`.

```bash
PYTHONPATH=/workspaces/isaaclab_arena:${PYTHONPATH:-} \
submodules/IsaacLab/isaaclab.sh -p \
  isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_environments/G1_Factory/policy_configs/benchmark/LMDrillLift_gr00t_closedloop.yaml \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --num_steps 800 \
  --viz kit \
  --enable_cameras \
  --external_environment_class_path isaaclab_arena_environments.G1_Factory.LMDrillLift:LMDrillLift \
  LMDrillLift
```

For a headless environment smoke test without the learned policy, run:

```bash
PYTHONPATH=/workspaces/isaaclab_arena:${PYTHONPATH:-} \
submodules/IsaacLab/isaaclab.sh -p \
  isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 20 \
  --viz none \
  --external_environment_class_path isaaclab_arena_environments.G1_Factory.LMDrillLift:LMDrillLift \
  LMDrillLift
```

Real-policy validation requires an external GR00T server and checkpoint access, so the repository tests validate the
Arena-side config schema but do not start the server.
