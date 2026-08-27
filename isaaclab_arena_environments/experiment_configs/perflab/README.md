# PerfLab first-pass Arena benchmarks

This README tells a PerfLab operator how to set up and run the first three non-OSMO Arena
benchmarks:

1. `zero_action` without cameras.
2. `zero_action` with the maintained production DROID cameras.
3. Pi0.5 with the maintained production DROID cameras and a separate OpenPI server.

Here, **Pi Zero means the checked-in Pi0.5 (`pi05`) experiment**. Do not substitute the older
`pi0` checkpoint without a separate agreed configuration.

Each command starts one fresh Arena Experiment Runner process on one simulator GPU. Pi0.5 also
needs a separately started OpenPI server process. The local helper does not automatically put that
server on another GPU. These commands do not use OSMO, `torchrun`, distributed evaluation, video
recording, or the legacy JSON runner.

## What the three experiments show

All three use the same Maple-table task: pick up a Rubik's cube and place it in a bowl.

| Experiment | What runs | What it tells us |
|---|---|---|
| Camera-free baseline | `zero_action`; cameras off | Scene construction, physics stepping, throughput, and the maximum environment count without camera or policy cost. |
| Production-camera baseline | `zero_action`; the maintained three-camera DROID rig is on | The additional cost of producing the normal camera observations. This README does not change or state a camera resolution. |
| Pi0.5 | Pi0.5 remote policy; the maintained three-camera DROID rig is on | End-to-end production evaluation cost, including camera rendering and calls to the OpenPI server. |

The first two experiments use the same `droid_rel_joint_pos` embodiment, so comparing them isolates
the cost of enabling the production camera rig. Pi0.5 requires the `droid_abs_joint_pos`
embodiment. Therefore, compare Pi0.5 with the production-camera baseline only as a complete
production-path comparison, not as a measurement of policy inference alone.

All three DROID cameras are rendered in the Pi0.5 experiment. The policy request uses the main
external-camera image and the wrist-camera image; the second external-camera image is rendered but
is not sent to OpenPI.

The zero-action policy keeps the robot stationary. A completed zero-action run proves that the
performance pipeline worked; it is not expected to solve the task.

## Configuration used by every measured run

Keep these settings fixed:

- `num_steps`: **300**, the proposed measurement length for this first PerfLab pass.
- Renderer: `balanced`.
- Visualization and video: off.
- Environment seed: `42`.
- Placement seed: `42`.
- Environment spacing: `2.5`.
- Environment rebuilds: `1`.
- Three valid repeats for every successful environment count.
- One fresh Experiment Runner process for every environment count and repeat.

The YAML files keep `num_steps: 10` as a quick smoke default. Every measured command below
explicitly overrides it with `NUM_STEPS=300`. This is long enough for Pi0.5 to fetch multiple action
chunks instead of measuring only its first request. Confirm this proposed value before scheduling
the final PerfLab job, then use the same value for all three experiments.

Before the handoff, the Arena owner must provide PerfLab with the exact Arena commit, Isaac Lab
submodule commit, Arena image digest, OpenPI image digest, and trial timeout. Do not benchmark a
moving branch.

## Machine and access requirements

PerfLab needs:

- A Linux host supported by Isaac Sim 6.0.
- Docker and the NVIDIA Container Toolkit.
- Git access to Isaac Lab-Arena and access to its task assets.
- One simulator GPU for the two zero-action experiments.
- At least one GPU for a local Pi0.5 smoke check. The simulator and policy server may share it.
- A second GPU for Pi0.5 only when the agreed PerfLab layout isolates policy inference from the
  simulator.
- Enough local storage for the Arena image, an approximately 19 GB OpenPI image, an approximately
  11 GB Pi0.5 checkpoint, logs, and result directories.
- Persistent storage mounted into the Arena container at `/eval`.
- A harness that records total wall time, process exit code, stdout, stderr, host memory, and GPU
  utilization and memory.

The simulator and OpenPI server must be able to reach each other over TCP. The checked-in Pi0.5
configuration uses `127.0.0.1:8000`, so that default requires the server and simulator to share a
host-network namespace. If they run on different hosts, use the host and port overrides shown
below.

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
PERFLAB_OUTPUT_ROOT=/eval/perflab-three-workloads
mkdir -p "${PERFLAB_OUTPUT_ROOT}"
```

## 3. Understand the trial variables

Set these variables inside the Arena container before each command:

```bash
NUM_STEPS=300
NUM_ENVS=1
REPEAT=1
ATTEMPT=1
```

- `NUM_STEPS` stays at `300` for every measured trial.
- `NUM_ENVS` selects one point from the experiment's sweep.
- `REPEAT` is `1`, `2`, or `3`.
- `ATTEMPT` starts at `1`. Increase it only when an infrastructure problem requires the exact
  trial to be run again.

The path includes all four values, so a smoke test, measured repeat, or retry cannot overwrite an
earlier result. The exact output directory passed to Arena must be missing or empty.

PerfLab's harness should capture stdout and stderr outside the Arena output directory. Start the
trial wall-clock immediately before `/isaac-sim/python.sh` and stop it when that process exits.
Container startup, image downloads, asset downloads, and OpenPI server startup are preparation and
must not be included in this trial time.

## 4. Run a one-environment smoke check

Before collecting measurements, run each available experiment once with:

```bash
NUM_STEPS=10
NUM_ENVS=1
REPEAT=1
ATTEMPT=1
```

Use the commands in the next sections without changing anything else. These smoke results only
check setup and output creation. Do not include them in the benchmark tables.

The smoke checks also warm the asset, shader, and image caches. The Pi0.5 smoke check warms the
policy server. Treat every later measured trial as a warm-cache run, and do not clear these caches
between repeats.

After all smoke checks pass, restore:

```bash
NUM_STEPS=300
```

## 5. Experiment 1: zero action without cameras

This is the simplest reference workload. It needs only the Arena container and one simulator GPU.
No policy server is involved.

Run this command inside the Arena container:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config \
    isaaclab_arena_environments/experiment_configs/perflab/camera_free_benchmark_experiment.yaml \
  --experiment_output_directory \
    "${PERFLAB_OUTPUT_ROOT}/camera-free/envs-${NUM_ENVS}/steps-${NUM_STEPS}/repeat-${REPEAT}/attempt-${ATTEMPT}" \
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

Run `4096` only if all three `2048` repeats complete. For each count, run repeats `1`, `2`, and `3`
before increasing `NUM_ENVS`.

## 6. Experiment 2: zero action with production cameras

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
    "${PERFLAB_OUTPUT_ROOT}/production-camera/envs-${NUM_ENVS}/steps-${NUM_STEPS}/repeat-${REPEAT}/attempt-${ATTEMPT}" \
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

If all three `256` repeats complete, continue by doubling to `512`, then `1024`, and so on while
the preceding point remains stable.

## 7. Prepare the Pi0.5 policy server

Do this before the Pi0.5 smoke check or measured sweep. Use a second host terminal in the same
clean Arena checkout.

The checked-in OpenPI helper does not select a policy-server GPU. It starts Docker with `--gpus
all`; the Arena launcher also exposes all visible GPUs, and the commands below run Arena on
`cuda:0`. On a normal local machine, both processes may therefore use the first visible GPU. That
is acceptable for a functional smoke check, but the resulting timing includes GPU contention.

If the agreed PerfLab measurement uses separate simulator and policy GPUs, PerfLab must isolate
them in its job or container harness. The current helper cannot enforce that layout by itself; it
would need a GPU-selection option or an equivalent PerfLab launch command. Record whether the run
used a shared or dedicated policy GPU, and do not compare results from the two layouts as if they
were the same workload.

Start the server:

```bash
./isaaclab_arena_openpi/docker/run_openpi_server.sh -v pi05 -p 8000
```

On the first invocation, the wrapper may build the approximately 19 GB OpenPI image and download
the approximately 11 GB Pi0.5 checkpoint. Complete that work before starting the measured clock.

Wait until the server prints:

```text
INFO:websockets.server:server listening on 0.0.0.0:8000
```

Leave this terminal and server running throughout the complete Pi0.5 smoke check and measured
sweep. Do not include server startup in the Arena trial time. Keep the server's checkpoint, GPU,
endpoint, and warm state unchanged across repeats.

## 8. Experiment 3: Pi0.5 with production cameras

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
    "${PERFLAB_OUTPUT_ROOT}/pi05/envs-${NUM_ENVS}/steps-${NUM_STEPS}/repeat-${REPEAT}/attempt-${ATTEMPT}" \
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
simulation steps. That behavior is part of this end-to-end measurement. A ten-step smoke check
normally makes only the first request for each environment; the 300-step run exercises repeated
requests.

For the measured sweep, use these environment counts in order:

```text
1, 16, 64, 128, 256
```

If all three `256` repeats complete, continue by doubling while the preceding point remains
stable. Finish all Pi0.5 repeats before stopping the OpenPI server. Stop it with Ctrl-C in the same
terminal that started the wrapper so the wrapper can clean up correctly.

## 9. Run order

Use this order:

1. Run the three one-environment, ten-step smoke checks. Start OpenPI before the Pi0.5 smoke check.
2. Set `NUM_STEPS=300`.
3. Finish the camera-free sweep.
4. Finish the production-camera sweep.
5. Use the same verified Pi0.5 server from the smoke check and finish the Pi0.5 sweep.

Run one command at a time. Do not put several sweep points into one Experiment Runner process.
Before starting the next command, confirm that the preceding simulator process exited and its GPU
memory returned to the idle level.

## 10. Decide whether a trial passed

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
| Production-camera baseline | `production_camera_baseline` |
| Pi0.5 | `pi05_evaluation` |

The `rollout/step_total.count`, `rollout/env_step.count`, and
`rollout/policy_get_action.count` entries in `arena_experiment_timings.json` must equal
`NUM_STEPS`.

## 11. Handle failures without losing information

A clear CUDA OOM, confirmed host OOM, or repeatable high-count simulator failure is a capacity
result. An elapsed trial timeout is also a capacity result only when the simulator, policy server,
and worker infrastructure stayed healthy:

1. Keep the full log and every partial output file.
2. Record the failed environment count and exact error.
3. Do not run a larger environment count for that experiment.
4. Wait for the process to exit and GPU memory to return to idle.
5. Continue with the next experiment.

A server outage, image or asset download failure, network outage, preemption, or machine failure
is an infrastructure failure, not a performance result. Fix the problem, increase `ATTEMPT`, and
rerun the same `NUM_ENVS`, `NUM_STEPS`, and `REPEAT`. Never silently reduce an environment count or
change another setting.

Do not add `--continue_on_error`. A nonzero process exit is important evidence.

## 12. Measurements PerfLab must return

PerfLab's outer harness is authoritative for total elapsed time. Start the clock immediately before
the Python command and stop it after the process exits.

For these single-simulator-GPU trials:

```text
total environment steps = NUM_ENVS * NUM_STEPS

aggregate env-steps/second =
    NUM_ENVS * NUM_STEPS / total elapsed seconds
```

A Pi0.5 policy-server GPU, when one is allocated, is not a second simulator GPU and is not included
in the numerator. The same formula applies when the server shares the simulator GPU.

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

- The fully resolved command, environment count, step count, repeat, attempt, start/end timestamps,
  elapsed seconds, and exit code.
- Full stdout and stderr.
- Arena and Isaac Lab commits, image digests, renderer, camera contract, seeds, host inventory, GPU
  model, GPU UUID, and driver.
- One-second samples of simulator GPU utilization, VRAM, power, clocks, temperature, Xid, and ECC
  events, plus simulator process RSS and host RAM.
- For Pi0.5, policy-server process memory, server logs, readiness time, endpoint, checkpoint
  identity, and GPU samples tagged as shared or dedicated.
- The complete Arena output directory, including partial output from a failed trial.
- The configured trial timeout and any host or cgroup OOM evidence.

For every successful point, report the median total elapsed time and median aggregate throughput
across its three valid repeats. For each experiment, report the highest stable environment count
and the first failed count.
