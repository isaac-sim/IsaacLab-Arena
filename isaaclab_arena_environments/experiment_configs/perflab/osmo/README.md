# OSMO camera-free scaling validation

This first validation runs the same eight camera-free, zero-action Arena Runs at four concurrency
limits: 1, 2, 4, and 8. Only the number of Runs allowed to execute concurrently changes. Each Run
uses 256 parallel environments for 300 steps on one GPU.

`K` is the requested maximum number of concurrent one-GPU Runs. OSMO may execute fewer Runs at
once when the pool is busy, and may place several Runs on one eight-GPU node.

## Preview and submit

Run these commands in the Arena development environment from the checkout containing this branch.
Set `ARENA_IMAGE` to the pinned image built from the same commit; do not use a moving `latest` tag
for measured results.

```bash
EXPERIMENT_CFG=isaaclab_arena_environments/experiment_configs/perflab/osmo/camera_free_scaling_validation_experiment.yaml
ARENA_IMAGE=nvcr.io/nvstaging/isaac-amr/isaaclab_arena:<pinned-tag>
MAX_PARALLEL_RUNS=1
```

Preview the rendered workflow first:

```bash
python osmo/submit_arena_experiment.py \
  --experiment_cfg "${EXPERIMENT_CFG}" \
  --dry_run \
  osmo.workflow_name="arena-camera-free-k${MAX_PARALLEL_RUNS}" \
  osmo.pool=isaac-apps-l40-05 \
  osmo.platform=ovx-l40 \
  osmo.max_parallel_runs="${MAX_PARALLEL_RUNS}" \
  experiment_runner.image="${ARENA_IMAGE}" \
  experiment_runner.record_camera_video=false
```

Remove `--dry_run` to submit the checked workflow. Complete and inspect `K=1` before submitting
`K=2`, then repeat for `K=4` and `K=8`. Use a unique workflow name for every submission.

Accept a measurement point only when all eight `experiment_runner_result.json` files report
`execution_status: completed` with exit code 0. A failed simulator Run may still leave the OSMO
workflow green so its diagnostics can be collected.

Use the task start and finish timestamps to confirm that the requested number of Runs actually
overlapped. If fewer than `K` Runs overlapped, report the achieved peak concurrency and do not use
that point to calculate efficiency for `K`.

## Results

Use OSMO execution timestamps for the primary Experiment completion time and exclude queue time.
For each `K`, report:

```text
speedup(K) = T1 / TK
parallel efficiency(K) = speedup(K) / K * 100%
```

The collected output stores each Run's `experiment_runner_result.json`,
`arena_experiment_timings.json`, and `arena_experiment_metadata.json` under its Run directory. The
result file also records the simulator process start, finish, and elapsed time. These files explain
individual Run behavior; the OSMO workflow duration remains the primary scaling measurement.
