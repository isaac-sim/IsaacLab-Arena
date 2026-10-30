# PerfLab first-pass Arena benchmarks

This plan tells a PerfLab operator how to set up and run six non-OSMO Arena benchmarks on Arena's
`main` branch:

1. `zero_action` without cameras.
2. `zero_action` with the maintained production DROID cameras.
3. Pi0.5 with the maintained production DROID cameras and a separate OpenPI server.
4. The same apple in every environment.
5. Ten fruit assets distributed across environments.
6. Cosmos with the maintained production DROID cameras and a separate Cosmos server.

Each command starts one fresh Arena Experiment Runner process on one simulator GPU. Pi0.5 and
Cosmos also need separately started policy servers, which run in their own containers on the same
machine.

## What the experiments show

All experiments use a DROID robot at the Maple table. The camera and policy workloads place a
Rubik's cube in a bowl. The object workloads place fruit in a bowl.

| Experiment | What runs | What it tells us |
|---|---|---|
| Camera-free baseline | `zero_action`; cameras off | Scene construction, physics stepping, throughput, and the maximum environment count without camera or policy cost. |
| Production-camera baseline | `zero_action`; the maintained three-camera DROID rig is on | The additional cost of producing the normal camera observations. This plan does not change or state a camera resolution. |
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
camera-free setup. Their scenes come from `same_object_environment.yaml` and
`mixed_object_environment.yaml` in this folder, which differ only in the fruit list. The
homogeneous run uses one apple in every environment. The heterogeneous run assigns its ten fruit
assets repeatedly in the listed order. Compare these two runs to measure the cost of heterogeneous
parallel environments. Do not compare them with the Rubik's-cube baseline, which uses a different
scene definition, different task assets, and an explicit HDR. All ten fruit assets appear when
`NUM_ENVS` is at least 10.

The zero-action policy keeps the robot stationary. A completed zero-action run proves that the
performance pipeline worked; it is not expected to solve the task.

## Installation

Follow the [Isaac Lab-Arena installation guide](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html)
and use the Docker setup. Pi0.5 and Cosmos also require the policy servers described below.

The Arena image installs PyTorch built for CUDA 13.0, so the host needs an NVIDIA driver from the
R580 branch or newer. Check the driver version with `nvidia-smi` on the host. With an older driver,
every run stops at startup with `The NVIDIA driver on your system is too old`.

## 1. Prepare a clean Arena checkout

Clone Arena's `main` branch on the host:

```bash
git clone --branch main --recurse-submodules \
  https://github.com/isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena
```

To benchmark a release instead, replace `main` with the release tag.

Check that the checkout is clean and record the Arena and submodule revisions:

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

The first invocation builds the Arena image, which also downloads the Isaac Sim base image. Do that
before timing any trial. The launcher enters the Arena container interactively; leave that terminal
open.

The launcher runs `xhost` only when `DISPLAY` is set. These commands do not open a simulator window,
so leave `DISPLAY` unset on a headless worker. If the launcher stops at `xhost`, treat that as a
setup problem rather than a benchmark result.

Inside the Arena container, verify that the source checkout is mounted and the simulator GPU is
visible:

```bash
cd /workspaces/isaaclab_arena

./submodules/IsaacLab/isaaclab.sh -p -c "import isaaclab_arena; print(isaaclab_arena.__file__)"
nvidia-smi
test -d /eval && test -w /eval
```

The import path must be below `/workspaces/isaaclab_arena`, and the final command must exit with
status zero. The `-e` mount is applied only when the launcher creates the container; attaching to
an older container does not change its mounts.

Every Python command in this plan runs `./submodules/IsaacLab/isaaclab.sh -p` from
`/workspaces/isaaclab_arena`. This works in the launcher's shell and in shells attached later with
`docker exec`, where `isaaclab.sh` is not on `PATH`. `isaaclab.sh` prints a deprecation warning on
every call; it does not affect the run.

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

Each command writes two timing files: `arena_experiment_timings.json` in the directory passed to
`--experiment_output_directory`, and `arena_run_timings.json` in the Run's subdirectory. For
example, the first run writes
`${PERFLAB_OUTPUT_ROOT}/camera-free/envs-${NUM_ENVS}/steps-${NUM_STEPS}/arena_experiment_timings.json`
and `camera_free_baseline/arena_run_timings.json` below that directory.

Each timing record names one timer and gives its `count`, `mean_ms`, `total_ms`, `min_ms`,
`max_ms`, `p10_ms`, `p50_ms`, and `p90_ms`. The `step` timer covers one full control step;
`step/policy_inference` and `step/env_step` split it into the policy call and the environment step.
`arena_run_timings.json` is a list of these records. `arena_experiment_timings.json` lists them
under `runs`, with the Run name added to each record, and sums them per timer under `totals`. The
timers cover the rollout only. Isaac Sim startup and scene construction are part of the trial
wall-clock but have no timer of their own.

PerfLab's harness should capture stdout and stderr outside the Arena output directory. Start the
trial wall-clock immediately before `isaaclab.sh` and stop it when that process exits. Container
startup, image downloads, asset downloads, and policy-server startup are preparation and must not
be included in this trial time.

## 4. Zero action without cameras

This is the simplest reference workload. It needs only the Arena container and one simulator GPU.
No policy server is involved.

Run this command inside the Arena container:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/performance/camera_free_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/camera-free/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
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
  ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/performance/same_object_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/same-object/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
  "shared.environment_builder.num_envs=${NUM_ENVS}" \
  "shared.rollout_limit.num_steps=${NUM_STEPS}"
```

Use `1, 64, 256, 1024, 2048`, followed by `4096` only if `2048` succeeds.

## 6. Heterogeneous objects

The environments use ten fruit assets in a repeating order. Cameras and policy inference are off.

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/performance/mixed_object_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/mixed-object/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
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
  ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/performance/production_camera_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/production-camera/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
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

Before running Pi0.5, start OpenPI in a second terminal on the host, from the repository root:

```bash
./isaaclab_arena_openpi/docker/run_openpi_server.sh -v pi05 -p 8000
```

The script gives the server container access to every GPU on the machine and has no option to
restrict it to one. Record the GPU memory the server uses, as shown by `nvidia-smi` once the server
is ready.

On the first invocation, the wrapper may build the OpenPI server image and download the
approximately 11 GB Pi0.5 checkpoint. Complete that work before starting the measured clock.

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
  ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/performance/pi05_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/pi05/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
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

Stop the Pi0.5 server first because both servers use port `8000`. Then start Cosmos in the second
terminal on the host:

```bash
./isaaclab_arena_cosmos/docker/run_cosmos_server.sh -p 8000
```

Like the OpenPI script, this script gives the server container access to every GPU on the machine.
The local client still uses `127.0.0.1:8000`.

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
  ./submodules/IsaacLab/isaaclab.sh -p isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/performance/cosmos_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/cosmos/envs-${NUM_ENVS}/steps-${NUM_STEPS}" \
  --viz none \
  --device cuda:0 \
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
- `arena_experiment_result.json` with `runs.<run-name>.status` equal to `completed`.
- `arena_experiment_timings.json`.
- `<run-name>/arena_run_timings.json`.
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

In `<run-name>/arena_run_timings.json`, the `count` of the `step`, `step/env_step`, and
`step/policy_inference` records must equal `NUM_STEPS`.
