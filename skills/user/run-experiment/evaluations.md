# Run Experiment Evaluations

## Scenario 1: Built-In Local Smoke Evaluation

Query: "Run the getting-started Experiment locally and tell me where the report is."

Expected behavior:

- Uses the ready runtime and the maintained `getting_started_experiment.yaml` configuration.
- Runs `experiment_runner.py` locally with an explicit config path and without OSMO or pytest.
- Waits for finalization, reads every Run status from `arena_experiment_result.json`, and reports
  both that file and `index.html`.
- Treats unsuccessful zero-action episodes as valid rollout results rather than a broken workflow.

Known failure modes:

- Runs the legacy default configuration because no explicit path was supplied.
- Claims success from process exit alone or requires a nonzero policy success rate.
- Reports a guessed output directory without checking the artifacts.

## Scenario 2: Local Shared Override

Query: "Run `my_experiment.yaml` headless, set the shared episode count to 4, and don't edit the YAML."

Expected behavior:

- Uses `--viz none` and the local override `shared.rollout_limit.num_episodes=4`.
- Leaves the Experiment Definition unchanged and reports the effective override.
- Verifies canonical results and per-Run episode records after execution.

Known failure modes:

- Uses the OSMO-only `experiment_cfg.shared...` prefix.
- Edits the source YAML despite the explicit request.
- Uses a global `--num_episodes` flag instead of a typed Experiment override.

## Scenario 3: Per-Run Override And An Exact Output Directory

Query: "For the `parallel_envs` Run, use 8 environments and save the Experiment exactly in `/eval/smoke`."

Expected behavior:

- Checks whether `/eval/smoke` is missing or empty before execution.
- Uses `runs.parallel_envs.environment_builder.num_envs=8` and an appropriate configuration that
  contains that Run.
- Uses `--experiment_output_directory /eval/smoke` and verifies the artifacts at that exact path.
- Does not clear or overwrite an existing nonempty directory.

Known failure modes:

- Applies the environment count to every Run through `shared`.
- Deletes an existing output directory or mixes exact and timestamped output options.
- Silently treats a nonempty requested exact directory as an output base without agreement.
- Confuses one Run with one simulated environment.

## Scenario 4: Continue After Failure With Camera Video

Query: "Run every configured Run even if one fails, and record camera videos."

Expected behavior:

- Adds `--continue_on_error` and `--record_camera_video` and explains the additional resource use.
- Waits for all attempted Runs and reads each canonical Run status.
- Reports failed Runs even if the overall process exits zero, plus the requested video artifacts.

Known failure modes:

- Treats exit zero as proof that every Run completed.
- Stops after the first failed Run or silently retries it with different settings.
- Claims a video exists without checking the Run output directories.

## Scenario 5: List Variations Without Rollouts

Query: "Show me which variations this Experiment supports; don't run any rollouts."

Expected behavior:

- Uses the Experiment Runner's `--list_variations` inspection path.
- Reports the variation catalogue for the configured Runs.
- States that no rollout, result JSON, or HTML report was expected.

Known failure modes:

- Starts the full evaluation or edits the Experiment Definition.
- Reports missing output artifacts as a failure.

## Scenario 6: Respect Workflow Boundaries

Query: "Submit this Experiment to OSMO pool `example-pool` and download the results."

Expected behavior:

- Does not execute the Experiment locally or invoke an OSMO command through this skill.
- Routes the request to the future `submit-osmo-experiment` workflow rather than acting on it.

Known failure modes:

- Treats local and managed execution as interchangeable.
- Submits remote compute using the local skill's permissions.

## Scenario 7: Route Runtime Setup And Regression Testing

Query: "I just cloned Arena. Install it, then run the no-camera pytest phase."

Expected behavior:

- Uses `setup-arena` for installation and readiness and `run-tests` for pytest.
- Does not start an Experiment merely because the request mentions Arena execution readiness.

Known failure modes:

- Absorbs installation or pytest into the Experiment workflow.
- Uses a zero-action Experiment as a substitute for the requested regression phase.
