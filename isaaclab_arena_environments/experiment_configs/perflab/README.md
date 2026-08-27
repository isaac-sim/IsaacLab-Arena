# PerfLab first-pass Arena benchmarks

This README tells a PerfLab operator how to set up and run six non-OSMO Arena benchmarks:

1. `zero_action` without cameras.
2. `zero_action` with the maintained production DROID cameras.
3. Pi0.5 with the maintained production DROID cameras and a separate OpenPI server.
4. The same apple in every environment.
5. Ten fruit assets distributed across environments.
6. Cosmos with the maintained production DROID cameras and a separate Cosmos server.

Here, **Pi Zero means the checked-in Pi0.5 (`pi05`) experiment**. Do not substitute the older
`pi0` checkpoint without a separate agreed configuration.

Each command starts one fresh Arena Experiment Runner process on one simulator GPU. Pi0.5 and
Cosmos also need separately started policy servers. A policy server can share the simulator GPU or
use `-g` for another GPU. These commands do not use OSMO, `torchrun`, distributed evaluation,
video recording, or the legacy JSON runner.

## What the experiments show

All experiments use a DROID robot at the Maple table. The camera and policy workloads place a
Rubik's cube in a bowl. The object workloads place fruit in a bowl.

| Experiment | What runs | What it tells us |
|---|---|---|
| Camera-free baseline | `zero_action`; cameras off | Scene construction, physics stepping, throughput, and the maximum environment count without camera or policy cost. |
| Production-camera baseline | `zero_action`; the maintained three-camera DROID rig is on | The additional cost of producing the normal camera observations. This README does not change or state a camera resolution. |
| Pi0.5 | Pi0.5 remote policy; the maintained three-camera DROID rig is on | End-to-end production evaluation cost, including camera rendering and calls to the OpenPI server. |
| Homogeneous objects | `zero_action`; cameras off; one apple asset | Performance when every parallel environment uses the same object. |
| Heterogeneous objects | `zero_action`; cameras off; ten fruit assets | Performance when parallel environments use different object assets. |
| Cosmos | Cosmos remote policy; the maintained three-camera DROID rig is on | End-to-end production evaluation cost, including camera rendering and calls to the Cosmos server. |

The first two experiments use the same `droid_rel_joint_pos` embodiment, so comparing them isolates
the cost of enabling the production camera rig. Pi0.5 requires the `droid_abs_joint_pos`
embodiment. Cosmos also uses `droid_abs_joint_pos`. Therefore, compare Pi0.5 or Cosmos with the
production-camera baseline only as a complete production-path comparison, not as a measurement of
policy inference alone.

All three DROID cameras are rendered in the Pi0.5 experiment. The policy request uses the main
external-camera image and the wrist-camera image; the second external-camera image is rendered but
is not sent to OpenPI.

The homogeneous and heterogeneous experiments use the same task, zero-action policy, and
camera-free setup. The homogeneous run uses one apple in every environment. The heterogeneous run
assigns its ten fruit assets repeatedly in the listed order. Compare these two runs to measure the
cost of heterogeneous parallel environments. Do not compare them with the Rubik's-cube baseline,
which uses different task assets and an explicit HDR. All ten fruit assets appear when `NUM_ENVS`
is at least 10.

The zero-action policy keeps the robot stationary. A completed zero-action run proves that the
performance pipeline worked; it is not expected to solve the task.

## Run order

1. Prepare the checkout and Arena container.
2. Set `NUM_STEPS=300` and finish the camera-free sweep.
3. Finish the homogeneous-object sweep, followed by the heterogeneous-object sweep.
4. Finish the production-camera sweep.
5. Start the Pi0.5 server and finish the Pi0.5 sweep.
6. Stop Pi0.5, start the Cosmos server, and finish the Cosmos sweep.

Run one command at a time. Before starting the next command, confirm that the preceding simulator
process exited and its GPU memory returned to the idle level.

## Configuration used by every measured run

Keep these settings fixed:

- `num_steps`: **300**, the proposed measurement length for this first PerfLab pass.
- Renderer: `balanced`.
- Visualization and video: off.
- Environment seed: `42`.
- Placement seed: `42`.
- Environment spacing: `2.5`.
- Environment rebuilds: `1`.
- One fresh Experiment Runner process for every environment count.

The YAML files keep `num_steps: 10` as a debugging default. This runbook overrides it with
`NUM_STEPS=300`, which lets Pi0.5 fetch multiple action chunks instead of measuring only its first
request. Confirm this proposed value before scheduling the final PerfLab job, then use it for all
six experiments.

Before the handoff, the Arena owner must provide PerfLab with the exact Arena commit, Isaac Lab
submodule commit, Arena image digest, policy-server image digests, and trial timeout. Do not
benchmark a moving branch.

## Machine and access requirements

PerfLab needs:

- A Linux host supported by Isaac Sim 6.0.
- Docker and the NVIDIA Container Toolkit.
- Git access to Isaac Lab-Arena and access to its task assets.
- One simulator GPU for the zero-action experiments.
- At least one GPU for Pi0.5 and Cosmos. The simulator and policy server may share it.
- A second GPU when the agreed PerfLab layout isolates policy inference from the simulator.
- Enough local storage for the Arena image, an approximately 19 GB OpenPI image, an approximately
  11 GB Pi0.5 checkpoint, logs, and result directories.
- Access to the Cosmos checkpoint and an `HF_TOKEN` if PerfLab must build the Cosmos server image.
- Persistent storage mounted into the Arena container at `/eval`.
- A harness that records total wall time, process exit code, stdout, stderr, host memory, and GPU
  utilization and memory.

The simulator and policy server must be able to reach each other over TCP. Pi0.5 and Cosmos both
default to `127.0.0.1:8000`, so run them one at a time. If a server runs on another host, use the
host and port overrides shown below.

## 1. Prepare a clean Arena checkout

Run these commands on the host. PerfLab must replace `<PINNED_ARENA_COMMIT>` with the commit given
by the Arena owner.

```bash
git clone https://github.com/isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena

ARENA_COMMIT="<PINNED_ARENA_COMMIT>"
git fetch origin
git checkout "${ARENA_COMMIT}"
git submodule update --init --recursive
```

Check that the checkout is clean and record both revision commands in the PerfLab result:

```bash
git status --short
git rev-parse HEAD
git submodule status --recursive
```

`git status --short` should print nothing. A line from `git submodule status` must not begin with
`-`, `+`, or `U`.

## 2. Start the Arena container

Choose an absolute host directory that PerfLab will retain after the job. Start the container from
the repository root and mount that directory at `/eval`:

```bash
PERFLAB_OUTPUT_HOST=/absolute/path/to/perflab-output
mkdir -p "${PERFLAB_OUTPUT_HOST}"

./docker/run_docker.sh -e "${PERFLAB_OUTPUT_HOST}"
```

The first invocation may build or download the Arena image. Do that before timing any trial. The
launcher enters the Arena container interactively; leave that terminal open.

The checked-in launcher performs X11 setup even though these commands do not open a simulator
window. On a headless worker, use PerfLab's normal X11/container setup. If the launcher stops at
`xhost`, treat that as a setup problem rather than a benchmark result.

Inside the Arena container, verify that the source checkout is mounted and the simulator GPU is
visible:

```bash
cd /workspaces/isaaclab_arena

/isaac-sim/python.sh -c "import isaaclab_arena; print(isaaclab_arena.__file__)"
nvidia-smi
test -d /eval && test -w /eval
```

The import path must be below `/workspaces/isaaclab_arena`, and the final command must exit with
status zero. The `-e` mount is applied only when the launcher creates the container; attaching to
an older container does not change its mounts.

Create only the common output root. Do not create a trial's exact output directory before its
command runs:

```bash
PERFLAB_OUTPUT_ROOT=/eval/perflab-six-workloads
mkdir -p "${PERFLAB_OUTPUT_ROOT}"
```

## 3. Understand the trial variables

Set these variables inside the Arena container before each command:

```bash
NUM_STEPS=300
NUM_ENVS=1
```

- `NUM_STEPS` stays at `300` for every measured trial.
- `NUM_ENVS` selects one point from the experiment's sweep.

The exact output directory passed to Arena must be missing or empty.

PerfLab's harness should capture stdout and stderr outside the Arena output directory. Start the
trial wall-clock immediately before `/isaac-sim/python.sh` and stop it when that process exits.
Container startup, image downloads, asset downloads, and policy-server startup are preparation and
must not be included in this trial time.

## 4. Zero action without cameras

This is the simplest reference workload. It needs only the Arena container and one simulator GPU.
No policy server is involved.

Run this command inside the Arena container:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/experiment_configs/perflab/camera_free_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/camera-free/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
  --rendering_mode balanced \
  "shared.environment_builder.num_envs=${NUM_ENVS}" \
  "shared.rollout_limit.num_steps=${NUM_STEPS}"
```

In plain English, this starts a fresh Isaac Sim process, creates `NUM_ENVS` copies of the task with
no cameras, takes `NUM_STEPS` stationary actions, writes one result directory, and exits.

For the measured sweep, use these environment counts in order:

```text
1, 64, 256, 1024, 2048
```

Run `4096` only if `2048` completes successfully. Run each count once before increasing
`NUM_ENVS`.

## 5. Homogeneous objects

Every environment uses the same apple asset. Cameras and policy inference are off.

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/experiment_configs/perflab/same_object_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/same-object/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
  --rendering_mode balanced \
  "shared.environment_builder.num_envs=${NUM_ENVS}" \
  "shared.rollout_limit.num_steps=${NUM_STEPS}"
```

Use `1, 64, 256, 1024, 2048`, followed by `4096` only if `2048` succeeds.

## 6. Heterogeneous objects

The environments use ten fruit assets in a repeating order. Cameras and policy inference are off.

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/experiment_configs/perflab/mixed_object_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/mixed-object/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
  --rendering_mode balanced \
  "shared.environment_builder.num_envs=${NUM_ENVS}" \
  "shared.rollout_limit.num_steps=${NUM_STEPS}"
```

Use the same sweep as the homogeneous run: `1, 64, 256, 1024, 2048`, then `4096` if `2048`
succeeds.

## 7. Zero action with production cameras

This uses the same task and zero-action policy, but enables the maintained three-camera DROID rig.
The Experiment Runner detects the camera requirement from the YAML, so do not add an
`--enable_cameras` flag. Do not add video flags or camera-resolution overrides.

Run this command inside the Arena container:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/experiment_configs/perflab/production_camera_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/production-camera/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
  --rendering_mode balanced \
  "shared.environment_builder.num_envs=${NUM_ENVS}" \
  "shared.rollout_limit.num_steps=${NUM_STEPS}"
```

This repeats the same task and stationary actions, but renders the maintained production camera
rig. The difference from Experiment 1 shows the camera cost.

For the measured sweep, use these environment counts in order:

```text
1, 16, 64, 128, 256
```

If `256` completes successfully, continue by doubling to `512`, then `1024`, and so on while the
preceding point succeeds.

## 8. Prepare the Pi0.5 policy server

Before running Pi0.5, start OpenPI in a second terminal. To share one GPU with Arena:

```bash
./isaaclab_arena_openpi/docker/run_openpi_server.sh -v pi05 -p 8000
```

To keep Arena on `cuda:0` and run OpenPI on GPU 1 of the same machine:

```bash
./isaaclab_arena_openpi/docker/run_openpi_server.sh -g 1 -v pi05 -p 8000
```

The `-g` option only selects the server GPU. Networking is unchanged: the local client still uses
`127.0.0.1:8000`. Record whether the policy GPU was shared or dedicated.

On the first invocation, the wrapper may build the approximately 19 GB OpenPI image and download
the approximately 11 GB Pi0.5 checkpoint. Complete that work before starting the measured clock.

Wait until the server prints:

```text
INFO:websockets.server:server listening on 0.0.0.0:8000
```

Leave this terminal and server running throughout the Pi0.5 sweep. Do not include server startup
in the Arena trial time. Keep the server's checkpoint, GPU, endpoint, and warm state unchanged
throughout the sweep.

## 9. Pi0.5 with production cameras

Return to the Arena container terminal. The default setup uses the OpenPI server at
`127.0.0.1:8000`:

```bash
POLICY_HOST=127.0.0.1
POLICY_PORT=8000
```

Run this command inside the Arena container:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/experiment_configs/perflab/pi05_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/pi05/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
  --rendering_mode balanced \
  "shared.environment_builder.num_envs=${NUM_ENVS}" \
  "shared.rollout_limit.num_steps=${NUM_STEPS}" \
  "shared.policy.remote_host=${POLICY_HOST}" \
  "shared.policy.remote_port=${POLICY_PORT}"
```

This renders the production camera observations, sends one request per environment to OpenPI when
a new action chunk is needed, applies the returned actions, writes the result, and exits. The
current client sends environment requests one at a time and reuses each Pi0.5 action chunk for 15
simulation steps. That behavior is part of this end-to-end measurement, and the 300-step run
exercises repeated requests.

For the measured sweep, use these environment counts in order:

```text
1, 16, 64, 128, 256
```

If `256` completes successfully, continue by doubling while the preceding point succeeds. Finish
the Pi0.5 sweep before stopping the OpenPI server. Stop it with Ctrl-C in the same terminal that
started the wrapper so the wrapper can clean up correctly.

## 10. Prepare the Cosmos policy server

Stop the Pi0.5 server first because both servers use port `8000`. Start Cosmos on GPU 1:

```bash
./isaaclab_arena_cosmos/docker/run_cosmos_server.sh -g 1 -p 8000
```

Omit `-g 1` to share the simulator GPU. GPU selection does not change the endpoint:
`127.0.0.1:8000`.

If the Cosmos image is not already available, PerfLab must provide `HF_TOKEN` while the helper
builds the image and downloads the checkpoint. Finish that work before starting the measured
clock. Wait until the server reports that it is listening on port `8000`, then leave it running
throughout the Cosmos sweep.

## 11. Cosmos with production cameras

Set the Cosmos endpoint inside the Arena container:

```bash
COSMOS_HOST=127.0.0.1
COSMOS_PORT=8000
```

Then run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/experiment_configs/perflab/cosmos_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/cosmos/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
  --rendering_mode balanced \
  "shared.environment_builder.num_envs=${NUM_ENVS}" \
  "shared.rollout_limit.num_steps=${NUM_STEPS}" \
  "shared.policy.remote_host=${COSMOS_HOST}" \
  "shared.policy.remote_port=${COSMOS_PORT}"
```

Use `1, 16, 64, 128, 256`, then continue doubling while the preceding point succeeds. Stop the
Cosmos server with Ctrl-C after the sweep finishes.

## 12. Decide whether a trial passed

A successful trial has all of the following:

- Process exit code `0`.
- `arena_experiment_metadata.json` with top-level status `completed`.
- `arena_experiment_result.json` with the expected Run status `completed`.
- `arena_experiment_timings.json`.
- `index.html`.
- `<run-name>/episode_results_rebuild0.jsonl`. It may be empty when no episode finishes during the
  fixed number of steps.
- No CUDA OOM, host OOM, GPU reset, fatal renderer error, or policy-connection error in the log.

Expected Run names are:

| Experiment | Run name |
|---|---|
| Camera-free baseline | `camera_free_baseline` |
| Homogeneous objects | `same_object_control` |
| Heterogeneous objects | `mixed_object_workload` |
| Production-camera baseline | `production_camera_baseline` |
| Pi0.5 | `pi05_evaluation` |
| Cosmos | `cosmos_evaluation` |

The `rollout/step_total.count`, `rollout/env_step.count`, and
`rollout/policy_get_action.count` entries in `arena_experiment_timings.json` must equal
`NUM_STEPS`.

## 13. Handle failures without losing information

A clear CUDA OOM, confirmed host OOM, or repeatable high-count simulator failure is a capacity
result. An elapsed trial timeout is also a capacity result only when the simulator, policy server,
and worker infrastructure stayed healthy:

1. Keep the full log and every partial output file.
2. Record the failed environment count and exact error.
3. Do not run a larger environment count for that experiment.
4. Wait for the process to exit and GPU memory to return to idle.
5. Continue with the next experiment.

A server outage, image or asset download failure, network outage, preemption, or machine failure
is an infrastructure failure, not a performance result. Fix the problem, choose a new
`PERFLAB_OUTPUT_ROOT`, and rerun the same `NUM_ENVS` and `NUM_STEPS`. Never silently reduce an
environment count or change another setting.

Do not add `--continue_on_error`. A nonzero process exit is important evidence.

## 14. Measurements PerfLab must return

PerfLab's outer harness is authoritative for total elapsed time. Start the clock immediately before
the Python command and stop it after the process exits.

For these single-simulator-GPU trials:

```text
total environment steps = NUM_ENVS * NUM_STEPS

aggregate env-steps/second =
    NUM_ENVS * NUM_STEPS / total elapsed seconds
```

A Pi0.5 or Cosmos policy-server GPU is not a second simulator GPU and is not included in the
numerator. The same formula applies when the server shares the simulator GPU.

Arena records component timings in `arena_experiment_timings.json`, including:

- `run/build_environment`
- `rollout/initial_reset`
- `rollout/policy_get_action`
- `rollout/env_step`
- `rollout/step_total`
- `rollout/compute_metrics`
- `run/close_resources`

These timers are nested, CUDA-unsynchronized diagnostics and do not separate warm-up work. Do not
add their totals together. Use external total elapsed time for the primary throughput number and
the Arena timers only to explain where time was spent.

For every trial, return:

- The fully resolved command, environment count, step count, start/end timestamps, elapsed seconds,
  and exit code.
- Full stdout and stderr.
- Arena and Isaac Lab commits, image digests, renderer, camera contract, seeds, host inventory, GPU
  model, GPU UUID, and driver.
- One-second samples of simulator GPU utilization, VRAM, power, clocks, temperature, Xid, and ECC
  events, plus simulator process RSS and host RAM.
- For Pi0.5 and Cosmos, policy-server process memory, server logs, readiness time, endpoint,
  checkpoint identity, and GPU samples tagged as shared or dedicated.
- The complete Arena output directory, including partial output from a failed trial.
- The configured trial timeout and any host or cgroup OOM evidence.

For every successful point, report total elapsed time and aggregate throughput. For each
experiment, report the highest successful environment count and the first failed count.
