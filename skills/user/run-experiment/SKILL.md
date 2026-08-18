---
name: run-experiment
description: Runs existing Isaac Lab-Arena Experiment Definitions locally with experiment_runner.py in a ready native or Docker runtime, applies local CLI overrides, and verifies generated result and report artifacts. Use for local named-Run or batch policy evaluations, variation listing, visualization or video, and result inspection. Do not use for installation or runtime preparation (setup-arena), pytest or regression checks (run-tests), interactive no-policy inspection (environment_runner.py), direct Policy Runner workflows, or OSMO preview, submission, or management.
allowed-tools: Read Grep Glob Skill Bash(git rev-parse --show-toplevel) Bash(id -un) Bash(test -d *) Bash(test -f *) Bash(test -x .venv/bin/python) Bash(env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python isaaclab_arena/evaluation/experiment_runner.py *) Bash(docker ps *) Bash(docker exec *)
---

# Run Experiment

Execute an existing Experiment Definition locally and finish with evidence from its canonical
artifacts. Keep runtime readiness, regression testing, direct Policy Runner workflows, and managed
submission outside this skill.

## Read the checked-out documentation

Before composing a command, read:

- `docs/pages/concepts/concept_arena_experiments.rst` for Experiment structure and override rules.
- `docs/pages/quickstart/arena_experiment.rst` for the maintained local example.
- `docs/pages/quickstart/environment_variations.rst` when listing or applying variations.

Treat the current checkout as the source of truth. If this skill differs from the documentation or
runner CLI, follow the checkout and report the mismatch.

Use typed YAML as the primary interface. Accept legacy JSON only as a pass-through compatibility
path; do not create new legacy configurations or apply Hydra overrides to them.

## Preflight the run

1. Confirm the repository root with `git rev-parse --show-toplevel`.
2. Resolve an explicit Experiment path. Never rely on the runner's legacy default configuration.
3. Inspect the Experiment's Runs, policies, rollout limits, rebuild counts, environment counts,
   cameras, variations, and referenced files. Summarize unexpectedly large work before starting it.
4. Preserve a native or Docker route already selected by the user. If both are ready and the
   request does not choose one, use the preferred route in the current installation documentation
   and state the choice. If neither route is ready, use `setup-arena`; do not install dependencies,
   build an image, create mounts, or recreate a container here.
5. Confirm that referenced configs, datasets, checkpoints, and output locations are available from
   the selected runtime. For a remote policy, require an already-running and reachable server; do
   not start or submit one as part of this workflow.

For a built-in smoke evaluation, use
`isaaclab_arena_environments/experiment_configs/getting_started_experiment.yaml`. It uses the local
zero-action policy and requires no model or policy server.

## Select execution options

- Default to `--viz none` for unattended execution. Use `--viz kit` only when requested.
- Record viewport or camera video only when requested. Camera recording enables camera support and
  can materially increase GPU memory and output size.
- Preserve the runner's default stop-on-first-error behavior. Add `--continue_on_error` only when
  the user wants the remaining Runs attempted after a failure.
- Use `--serve_evaluation_report` only when explicitly requested; it binds an HTTP server and keeps
  the process running until interrupted.
- Use `--list_variations` as an inspection operation. It does not run rollouts or create Experiment
  result artifacts.
- Let the runner create a timestamped directory below `outputs/` unless the user chooses a base or
  exact output location. An exact `--experiment_output_directory` must be missing or empty. Never
  clear or reuse a nonempty directory. If a requested exact directory is nonempty, stop and offer
  either a different exact directory or a fresh timestamped child through `--output_base_dir`; do
  not silently change the output semantics.

## Build the local command

Always pass `--experiment_config` explicitly. Keep runner flags separate from trailing Hydra
overrides.

For native uv, run from the repository root and accept the Isaac Sim EULA non-interactively:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  .venv/bin/python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config <experiment.yaml> \
  --viz none \
  <override>...
```

For Docker, first use `docker ps` to select the single running container that mounts the absolute
repository root. Resolve the host username separately with `id -un`, then substitute both literal
values below rather than hardcoding a container name:

```bash
docker exec <container> su <host-user> -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh isaaclab_arena/evaluation/experiment_runner.py \
   --experiment_config <experiment.yaml> \
   --viz none \
   <override>..."
```

Use the local override namespace:

```text
shared.rollout_limit.num_episodes=4
runs.parallel_envs.environment_builder.num_envs=8
```

Do not prefix local overrides with `experiment_cfg.`; that prefix belongs to OSMO submission.
Quote override tokens that contain shell-sensitive characters. Do not edit the source YAML when a
declared field can be changed with an override.

## Execute and monitor

Run the command in the selected ready runtime and wait for it to finish. Runs execute in YAML order
inside one SimulationApp; each Run builds a fresh environment. Report meaningful progress during a
long evaluation.

On failure, preserve the logs and partial output. Do not delete artifacts or silently rerun with
different settings. By default, an early failure can prevent creation of the canonical result and
report. With `--continue_on_error`, the process can exit zero even when one or more Runs failed.

## Verify the outcome

For an evaluation that reaches finalization:

1. Locate the exact Experiment output directory printed by the runner.
2. Require `arena_experiment_result.json` and `index.html`.
3. Read every Run's `status` from `arena_experiment_result.json` and report completed and failed Runs.
4. Confirm expected `episode_results_rebuild<N>.jsonl` files under each completed Run directory.
5. Report requested videos and the HTML report path when present.

Treat the canonical result statuses, not process exit alone, as the workflow outcome. Keep policy
task success separate from execution success: zero-action episodes are expected to be semantically
unsuccessful while still proving that the evaluation pipeline works. Metrics are printed to the
console; the runner does not currently promise a separate `metrics.json`.

For `--list_variations`, report the catalogue and explicitly state that no rollout or output
artifacts were expected.

Finish with the selected runtime, Experiment path, effective overrides, output path, per-Run
statuses, episode counts, and any preserved partial artifacts or failures.

## Hand off other workflows

- Use `setup-arena` for installation, container creation, mounts, or readiness repair.
- Use `run-tests` for pytest and regression checks.
- Use the Environment Runner for interactive inspection without a policy.
- Use the Policy Runner for `torchrun`, external environments, or an explicitly requested direct
  rollout outside an Experiment Definition.
- Use a separate OSMO submission skill for previews, cluster resources, submission, monitoring, or
  remote result download. An unqualified request to "run" an Experiment means local execution.

## References

- [Evaluations](evaluations.md)
- [Arena Experiments](../../../docs/pages/concepts/concept_arena_experiments.rst)
- [First Arena Experiment](../../../docs/pages/quickstart/arena_experiment.rst)
- [Environment variations](../../../docs/pages/quickstart/environment_variations.rst)
- [Experiment Runner CLI](../../../isaaclab_arena/evaluation/experiment_runner_cli.py)
