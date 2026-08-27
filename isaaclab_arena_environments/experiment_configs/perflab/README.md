# PerfLab benchmark handoff

These definitions benchmark one Arena Run in one fresh Experiment Runner process. They do not use
OSMO, `torchrun`, or the legacy JSON/chunk runner. Each trial uses one simulator GPU. Remote-policy
trials normally also use one separate policy-server GPU.

## Decisions required before handoff

Do not send the benchmark to PerfLab until both items below are resolved and committed:

1. Replace the provisional `shared.rollout_limit.num_steps: 10` in every Experiment with the agreed
   fixed performance-rollout length. Ten steps is only a local smoke-test value. Use `num_steps`, not
   `num_episodes`, so task success and episode duration do not change the amount of measured work.
2. Select, implement, and verify the exact reduced camera resolution. The reduced-camera control
   must state its final width and height while keeping the three DROID cameras, their poses, the
   production aspect ratio, and the rest of the production-camera setup unchanged. Until the YAML
   is finalized, both dimensions must be supplied together as
   `shared.environment_builder.camera_height=<height>` and
   `shared.environment_builder.camera_width=<width>` overrides. A run without both overrides is not
   a reduced-resolution result.

Also pin the Arena commit, Isaac Lab submodule revision, simulator and policy-server image digests,
model checkpoints, task assets, renderer, seeds, and policy endpoints. PerfLab should run a clean
checkout at the pinned commit rather than a moving branch.

## Workloads and sweeps

Run every point three times. A point is stable only when all three trials pass. Report the highest
stable count, the best median throughput among stable points, and the first unstable or failed
count. After a genuine capacity failure, do not run larger points unless boundary finding is
requested.

| Experiment / Run | What it tests | Controlled difference | `num_envs` sweep |
|---|---|---|---|
| Camera-free baseline<br>`camera_free_benchmark_experiment.yaml`<br>`camera_free_baseline` | Core scene construction and physics-step scaling for the Maple-table Rubik's-cube-to-bowl task with a stationary zero-action policy. | Cameras and remote inference are disabled. This is the reference point for simulator capacity. | 1, 64, 256, 1024, 2048; then 4096 if stable |
| Production-camera baseline<br>`production_camera_benchmark_experiment.yaml`<br>`production_camera_baseline` | The same stationary task with all three production DROID cameras enabled. | Adds the maintained camera rig; task, assets, placement, policy, and camera settings otherwise stay fixed. | 1, 16, 64, 128, 256; then continue doubling if stable |
| Reduced-camera control<br>`reduced_camera_benchmark_experiment.yaml`<br>`reduced_camera_control` | Whether reducing rendered pixels changes camera-enabled capacity. | Only the common resolution of the same three cameras changes; final dimensions are still TBD. | Same camera-enabled sweep |
| Pi0.5<br>`pi05_benchmark_experiment.yaml`<br>`pi05_evaluation` | Full camera-enabled Pi0.5 evaluation, including the remote-policy client path. | Replaces zero action with the fixed Pi0.5 policy and server contract; production camera settings stay fixed. | Same camera-enabled sweep |
| Cosmos<br>`cosmos_benchmark_experiment.yaml`<br>`cosmos_evaluation` | Full camera-enabled Cosmos evaluation, including the remote-policy client path. | Replaces zero action with the fixed Cosmos policy and server contract; production camera settings stay fixed. | Same camera-enabled sweep |
| GR00T<br>`gr00t_benchmark_experiment.yaml`<br>`gr00t_evaluation` | Full camera-enabled GR00T evaluation, including the remote-policy client path. | Replaces zero action with the fixed GR00T policy and server contract; production camera settings stay fixed. | Same camera-enabled sweep |
| Same-object control<br>`same_object_benchmark_experiment.yaml`<br>`same_object_control` | Construction and stepping when every parallel environment uses the same apple asset. | Uses a deterministic one-member object set; cameras and remote inference are disabled. | Same camera-free sweep |
| Mixed-object workload<br>`mixed_object_benchmark_experiment.yaml`<br>`mixed_object_workload` | Construction and stepping when parallel environments use ten different fruit assets. | Only the deterministic object-set member list differs from the same-object control; assets are assigned round-robin. | Same camera-free sweep |

Use one fresh Experiment Runner process for each `(Experiment, num_envs, repeat)` tuple. Do not put
several sweep points in one Experiment process.

## Experiment Runner command

Run from the repository root inside the Arena container. Substitute an Experiment filename, a
trial-specific output path visible inside the container, and one value from the table:

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/experiment_configs/perflab/<experiment.yaml> \
  --experiment_output_directory <arena-output-directory> \
  --viz none \
  --device cuda:0 \
  --rendering_mode balanced \
  shared.environment_builder.num_envs=<num-envs>
```

The command intentionally uses the rollout length pinned in the YAML. A shorter
`shared.rollout_limit.num_steps=<smoke-steps>` override is allowed only for local debugging and must
not appear in measured trials. Do not add video flags or `--continue_on_error`.

`<arena-output-directory>` must be missing or empty. Give every repeat a unique directory. Capture
the runner log beside that directory rather than creating a log inside it before startup.

For local Docker execution, discover the container that mounts the current checkout instead of
using a fixed container name:

```bash
ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
HOST_USER=$(id -un)

docker exec "$ARENA_CONTAINER" su "$HOST_USER" -c \
  "cd /workspaces/isaaclab_arena && \
   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
   --experiment_config isaaclab_arena_environments/experiment_configs/perflab/<experiment.yaml> \
   --experiment_output_directory <arena-output-directory> \
   --viz none \
   --device cuda:0 \
   --rendering_mode balanced \
   shared.environment_builder.num_envs=<num-envs>"
```

## Pass and artifact contract

A completed trial must have all of the following:

- Process exit code zero.
- `arena_experiment_result.json` and `index.html` in the Experiment output directory.
- `arena_experiment_metadata.json`, with the expected command, revision, resolved Run settings, and
  final `completed` status.
- Exactly the expected Run in `arena_experiment_result.json`, with `status` equal to `completed`.
- `<run-name>/episode_results_rebuild0.jsonl`. A fixed-step trial can validly leave this file empty
  when no episode finishes.
- `arena_experiment_timings.json` when the process reaches graceful finalization.
- A runner log without CUDA OOM, fatal renderer, policy-connection, or GPU-reset errors.

The timing record's `rollout/step_total.count` and `rollout/env_step.count` must equal the pinned
`num_steps`. Diagnostic transitions per second can be calculated as:

```text
num_envs * count / (total_ms / 1000)
```

Use `rollout/step_total` for the full policy-plus-environment loop and `rollout/env_step` for the
environment step alone. These are diagnostic timings: CUDA synchronization is off by default,
there is no warm-up split, and nested timing totals must not be added together. The outer PerfLab
harness remains authoritative for total wall time and resource peaks.

An OOM, timeout, SIGKILL, or startup failure may prevent the result, report, or timing file from
being written. Preserve whatever partial output exists.

## PerfLab logging responsibilities

For every trial, the external harness must retain:

- The resolved command, trial identifier, repeat number, exit code, start/end timestamps, and full
  stdout/stderr log.
- Arena and submodule commits, image digests, checkpoints, camera contract, renderer, seeds, and
  host/GPU inventory.
- One-second simulator-GPU samples including UUID, utilization, VRAM, power, clocks, temperature,
  and Xid/ECC events, plus simulator host RAM and process RSS.
- Separately tagged policy-server GPU and host samples, server logs, readiness time, and checkpoint
  identity for remote-policy trials.
- Timeout configuration and host/cgroup OOM evidence.

OOM and timeout are valid capacity results. A server outage, image-pull failure, preemption, or other
infrastructure failure is an invalid trial: fix the external condition and rerun the identical
point. Never silently reduce `num_envs` or change another pinned setting. Before the next trial,
confirm the prior simulator process exited and GPU memory returned to its idle baseline.

## Policy-server requirements

Start and verify the required server before a measured Experiment begins. Keep its checkpoint,
endpoint, GPU allocation, and warm/cold state fixed across repeats. Pi0.5, Cosmos, and GR00T results
are production-path results, not direct model-speed comparisons.

The simulator and policy server should use separately monitored GPUs. Sharing one GPU is acceptable
for a local functional smoke test but is not comparable to the PerfLab result. OpenPI and GR00T have
maintained local serving workflows; the handoff must separately provide Cosmos startup and readiness
instructions. An endpoint configured as `127.0.0.1` also requires compatible host networking or
co-location.
