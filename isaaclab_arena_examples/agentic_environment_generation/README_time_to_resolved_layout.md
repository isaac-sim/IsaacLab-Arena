# Time-to-resolved-layout benchmark

This benchmark measures the time from loading a valid environment YAML to
filling the Arena placement pool with strictly valid layouts. Isaac Sim startup,
environment instantiation, and physics settling are excluded.

## Run

Start the Arena development container, then run the following command from
`/workspaces/isaaclab_arena`:

```bash
/isaac-sim/python.sh \
  isaaclab_arena_examples/agentic_environment_generation/run_time_to_resolved_layout_benchmarks.py \
  --num_runs 100 \
  --warmup_runs 1 \
  --placement_seed 42 \
  --num_envs 1 \
  --num_envs 16 \
  --num_envs 64 \
  --num_envs 256 \
  --output_dir output/time_to_resolved_layout
```

This runs all eight benchmark cases. Each environment requests five layouts,
and `allow_best_loss_fallbacks` is disabled so every stored layout must pass
strict validation.

The complete matrix takes approximately 50 hours on one RTX 6000 Ada
Generation GPU. Use repeated `--case` arguments to run selected cases only.
Run `--help` to list the available case names.

## Resume

Add `--resume` to the same command after an interrupted run. Completed samples
are stored after every measurement and are not rerun.

Resume requires the same environment specification, placement seed, warmup
count, and requested run count. Run only one benchmark process per output
directory.

## Output

The output directory contains:

```text
<case>_envs_<N>.json
all_results.json
```

Each result includes all timed samples, failure counts, and nearest-rank p50,
p95, and p99 latency in milliseconds. Record the Arena commit, GPU model,
driver version, Isaac Sim version, and command arguments with reported results.
