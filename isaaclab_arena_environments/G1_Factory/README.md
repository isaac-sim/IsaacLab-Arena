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

| Environment | Policy config | Legacy checkpoint reference |
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

The policy config passed to Arena is not the model checkpoint. It only configures Arena-side observation/action
translation for `Gr00tRemoteClosedloopPolicy`. To use a G1 Factory checkpoint, start the GR00T server with that
checkpoint, then run the Arena client against that server.

Populate the pinned GR00T submodule first:

```bash
git submodule update --init submodules/Isaac-GR00T
```

If SSH access to GitHub submodules is not configured in your environment, use an HTTPS rewrite for the submodule
checkout:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init submodules/Isaac-GR00T
```

The checkpoint references above are the paths used by the legacy G1 Factory repo on its original training/evaluation
machine. Cloning `arena_data` provides task assets, not these model checkpoints. Before starting the server, make sure
the checkpoint has been downloaded or mounted on the machine/container running GR00T.

In terminal 1, start the GR00T policy server from an environment that can access the checkpoint. This `LMDrillLift`
example uses the legacy G1 Factory checkpoint from the table above after you map it to a local path.

```bash
cd submodules/Isaac-GR00T
export G1_FACTORY_CHECKPOINT=/path/to/g1_benchmark_LMDrillLiftD1_100_gn1_5_output/checkpoint-20000
test -d "${G1_FACTORY_CHECKPOINT}"

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "${G1_FACTORY_CHECKPOINT}" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda \
  --host 127.0.0.1 \
  --port 5555
```

In terminal 2, run the Arena client from the Arena repository root. Include `submodules/Isaac-GR00T` on `PYTHONPATH`;
the client imports GR00T's `PolicyClient` even though inference runs in the server process.

```bash
PYTHONPATH="$(pwd):$(pwd)/submodules/Isaac-GR00T:${PYTHONPATH:-}" \
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
