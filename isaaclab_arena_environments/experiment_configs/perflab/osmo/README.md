# OSMO camera-free scaling validation

This first validation runs the same eight camera-free, zero-action Arena Runs at four concurrency
limits: 1, 2, 4, and 8. Only the number of Runs allowed to execute concurrently changes. Each Run
uses 256 parallel environments for 300 steps on one GPU.

`K` is the requested maximum number of concurrent one-GPU Runs. Synchronized mode places each wave
of up to `K` Runs in one native OSMO barrier group. The rendered group has `barrier: true`,
`ignoreNonleadStatus: false`, and exactly one lead task. Each Run in a later wave depends on the Run
in the same lane from the preceding wave.

Synchronized mode supports local policies such as `zero_action`. Submission fails when a Run would
derive a policy-server task, because the server lifecycle still requires one group per Run.

## Preview and submit

Run these commands in the Arena development environment from the checkout containing this branch.
Set `ARENA_IMAGE` to the pinned image built from the same commit; do not use a moving `latest` tag
for measured results. The validation runs used the image and digest shown below.

```bash
EXPERIMENT_CFG=isaaclab_arena_environments/experiment_configs/perflab/osmo/camera_free_scaling_validation_experiment.yaml
ARENA_IMAGE=nvcr.io/nvstaging/isaac-amr/isaaclab_arena:osmo-perflab-scaling-4ee056866
MAX_PARALLEL_RUNS=1
```

Tested image digest:
`sha256:480b3a146e35b5469708eade4cd8298b606cea1ad4b9c0e0e8c59098e3a0c1da`.

Preview the rendered workflow first:

```bash
python osmo/submit_arena_experiment.py \
  --experiment_cfg "${EXPERIMENT_CFG}" \
  --dry_run \
  osmo.workflow_name="arena-camera-free-k${MAX_PARALLEL_RUNS}" \
  osmo.pool=isaac-apps-l40-05 \
  osmo.platform=ovx-l40 \
  osmo.max_parallel_runs="${MAX_PARALLEL_RUNS}" \
  osmo.synchronize_parallel_runs=true \
  experiment_runner.image="${ARENA_IMAGE}" \
  experiment_runner.record_camera_video=false
```

Remove `--dry_run` to submit the checked workflow. Complete and inspect `K=1` before submitting
`K=2`, then repeat for `K=4` and `K=8`. Use a unique workflow name for every submission.

Accept a measurement point only when all eight `experiment_runner_result.json` files report
`execution_status: completed` with exit code 0. A failed simulator Run may still leave the OSMO
workflow green so its diagnostics can be collected.

OSMO may place several Runs on one eight-GPU node. The group barrier waits for cold image pulls and
input downloads before starting the wave. Use the process start and finish timestamps to confirm
that `K` Runs actually overlapped. If fewer than `K` Runs overlapped, report the achieved peak
concurrency and do not use that point to calculate efficiency for `K`.

## Download outputs

After the workflow completes, download its collected output using its unique workflow name:

```bash
WORKFLOW_NAME=arena-camera-free-k1-synchronized-1
OUTPUT_DIRECTORY=outputs/osmo_perflab_scaling/${WORKFLOW_NAME}
mkdir -p "${OUTPUT_DIRECTORY}"
osmo data download \
  "swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/${WORKFLOW_NAME}" \
  "${OUTPUT_DIRECTORY}"
```

## Results

Use Runner makespan as the primary scaling time, where `T_K` is the earliest
`process_started_at` through the latest `process_finished_at` across the eight
`experiment_runner_result.json` files. This includes transitions between waves but excludes the
final output collector. For each `K`, report:

```text
speedup(K) = T1 / TK
parallel efficiency(K) = speedup(K) / K * 100%
```

Report the full OSMO workflow duration and initial queue time separately. Cold image pulls may occur
before the first Runner or between waves, so record the cache state and do not present a
greater-than-ideal speedup as compute scaling without a warm-cache repetition.

The collected output stores each Run's `experiment_runner_result.json`,
`arena_experiment_timings.json`, and `arena_experiment_metadata.json` under its Run directory. The
result file also records the simulator process start, finish, and elapsed time. These files explain
individual Run behavior and confirm the achieved concurrency.
