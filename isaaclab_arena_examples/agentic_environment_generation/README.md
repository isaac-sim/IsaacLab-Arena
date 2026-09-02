# Time-to-first-spec benchmark

The time-to-first-spec benchmark measures the wall-clock time from sending an
environment-generation prompt until `EnvironmentGenerationAgent.generate_spec`
returns a valid `ArenaEnvGraphSpec`. Catalogue construction and endpoint setup
are outside the measured interval. The batch runner executes every case in
[`time_to_first_spec_cases.yaml`](time_to_first_spec_cases.yaml) sequentially.

The benchmark does not start Isaac Sim or build the generated environments.
It calls a remote inference endpoint, validates each response, resolves
background prim paths when needed, and writes timing results as JSON.

## Prerequisites

Export the API key on the host before starting the Arena container. The Docker
launcher forwards these variables when it creates the container:

| Endpoint | Access | Default model | API key variable |
| --- | --- | --- | --- |
| `internal` | NVIDIA internal network | `openai/openai/gpt-5.6-terra` | `NV_API_KEY` |
| `public` | NVIDIA public API | `openai/gpt-oss-120b` | `NVIDIA_API_KEY` |
| `openai` | OpenAI API | `gpt-5.6-terra` | `OPENAI_API_KEY` |

For example, configure one endpoint and start the container as documented in
the repository installation guide:

```bash
export NVIDIA_API_KEY=<your-ngc-api-key>
./docker/run_docker.sh
```

Do not put API keys in this repository. If the container was already running
when the variable was exported, recreate it so the launcher can forward the
key.

From the repository root, discover the container for the current checkout:

```bash
ARENA_CONTAINER=$(docker ps \
  --filter "volume=$(git rev-parse --show-toplevel)" \
  --format '{{.Names}}' | head -1)
test -n "$ARENA_CONTAINER"
```

The examples below use that `ARENA_CONTAINER` value and run as the host user.

## Run all cases

Each command runs every configured case 100 times and writes one combined JSON
file. `--no_save_specs` prevents the first generated environment from each case
from being saved as YAML.

NVIDIA internal endpoint:

```bash
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh \
   isaaclab_arena_examples/agentic_environment_generation/run_time_to_first_spec_benchmarks.py \
   --inference_endpoint internal \
   --num_runs 100 \
   --no_save_specs \
   --output_dir output/time_to_first_spec/internal"
```

NVIDIA public endpoint:

```bash
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh \
   isaaclab_arena_examples/agentic_environment_generation/run_time_to_first_spec_benchmarks.py \
   --inference_endpoint public \
   --num_runs 100 \
   --no_save_specs \
   --output_dir output/time_to_first_spec/public"
```

OpenAI endpoint:

```bash
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh \
   isaaclab_arena_examples/agentic_environment_generation/run_time_to_first_spec_benchmarks.py \
   --inference_endpoint openai \
   --num_runs 100 \
   --no_save_specs \
   --output_dir output/time_to_first_spec/openai"
```

To use the endpoint selected by `ARENA_INFERENCE_ENDPOINT`, omit
`--inference_endpoint`. If neither the argument nor the environment variable is
set, the public endpoint is used.

## Override the model or sampling temperature

Pass `--model` to either runner. The model must exist on the selected endpoint
and support OpenAI-compatible strict structured output. For example:

```bash
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh \
   isaaclab_arena_examples/agentic_environment_generation/run_time_to_first_spec_benchmarks.py \
   --inference_endpoint public \
   --model nvidia/nemotron-3-super-120b-a12b \
   --temperature 0.1 \
   --num_runs 100 \
   --no_save_specs \
   --output_dir output/time_to_first_spec/public-nemotron"
```

The internal and OpenAI endpoint presets currently ignore `--temperature`
because their configured APIs do not support that request parameter. Model
output can still be nondeterministic.

## Run one case

Use the single-case runner while developing a workload. Omitting
`--spec_output_dir` means no YAML is saved:

```bash
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh \
   isaaclab_arena_examples/agentic_environment_generation/benchmark_time_to_first_spec.py \
   --inference_endpoint internal \
   --case kitchen_open_fridge_door \
   --num_runs 100 \
   --output_path output/time_to_first_spec/kitchen_open_fridge_door.json"
```

Case names and prompts are defined in
[`time_to_first_spec_cases.yaml`](time_to_first_spec_cases.yaml). Use
`--prompt "..."` instead of `--case` to measure a custom prompt.

## Open the results

The batch runner prints a summary table when it finishes and writes:

```text
<output_dir>/all_results.json
```

Open the complete file in an editor or page through it in the terminal:

```bash
less output/time_to_first_spec/internal/all_results.json
```

Print only the per-case summaries with `jq`:

```bash
jq '.results[] | {case, summary}' \
  output/time_to_first_spec/internal/all_results.json
```

Print a compact tab-separated table:

```bash
jq -r '
  ["case", "ok", "failed", "p50_ms", "p95_ms", "p99_ms"],
  (.results[] | [
    .case,
    .summary.successful_samples,
    .summary.failed_samples,
    .summary.p50_ms,
    .summary.p95_ms,
    .summary.p99_ms
  ]) | @tsv
' output/time_to_first_spec/internal/all_results.json | column -t -s $'\t'
```

## Understand the results

The top-level fields describe the batch:

- `num_runs_per_case` is the requested number of samples for every case.
- `failed_cases` lists child benchmark processes that did not produce results.
- `spec_output_dir` is `null` when `--no_save_specs` was used.
- `results` contains one result object for each completed case.

Each result contains endpoint/model metadata, the prompt, a `summary`, and all
individual `samples`. The summary fields mean:

- `requested_samples`: total attempted generations for the case.
- `successful_samples`: samples with a valid spec and a non-null latency.
- `failed_samples`: requests that errored, returned no valid spec, or violated
  the case's expected object-reference structure.
- `p50_ms`: median successful latency using the nearest-rank method.
- `p95_ms`: successful latency at the 95th percentile.
- `p99_ms`: successful latency at the 99th percentile. Use at least 100 samples
  when reporting this value.

Percentiles exclude failed samples, so always interpret latency together with
the success and failure counts. Inspect `.samples[]` to diagnose failures:

```bash
jq '.results[] | {
  case,
  failures: [.samples[] | select(.time_to_first_spec_ms == null)]
}' output/time_to_first_spec/internal/all_results.json
```

For a no-save run, every sample's `generated_spec_path` is `null`. A successful
sample normally has a numeric `time_to_first_spec_ms`,
`final_spec_accepted: true`, and `error: null`.
