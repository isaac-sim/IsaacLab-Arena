# Newton upgrade validation

Use the repository revision under test as the source of truth for commands and reference metrics.
Record the baseline Lab revision and the candidate's exact compatibility patch or commit. If Lab
source or companion dependencies change, describe the comparison as an integration-stack change,
not an isolated Newton-only measurement. Keep task, policy, and solver settings fixed.

## Capture the comparison inputs

Store a manifest beside each run, outside source history:

- Arena commit and relevant local diff; recorded and actual Lab commit and full local Lab diff; Newton release/full SHA;
  MuJoCo, MuJoCo Warp, Warp, and USD schema versions; lock-file hash.
- Runtime route, Python path, Docker image ID when applicable, GPU, driver, and CUDA/Torch versions.
- Exact command, seeds, environment count, episode budget, assets/layout configuration, timeout,
  success thresholds, physics timestep, control decimation, solver, substeps, iterations, and
  contact/material settings. Save the resolved configuration, not only the CLI overrides.
- Policy checkpoint/config and server version/image, prompts, action horizon, camera pose,
  resolution, renderer, lighting, and viewport/recording settings. `--viz kit` alone does not specify
  which renderer produces robot-camera observations.

Keep the same seed schedule and policy endpoint for both runs. Set the **placement seed separately**:
Policy Runner's `--seed` does not set `--placement_seed`; Experiment Runner has distinct
`environment_builder.seed` and `environment_builder.placement_seed` fields. Compare the recorded
placement poses, not just the seed arguments. For a controlled layout comparison, replay the same
episode-result JSONL through `environment_builder.recorded_variation_samples_path` on both versions.
Keep an unpaired workflow sample separate from that comparison; different layouts can change the
success rate independently of physics. Reusing a policy server also does not reset its sampling RNG.
Record that limitation unless the policy offers a supported way to repeat its random draws.

Before full policy runs, use short baseline/candidate preflights to verify actual starting poses
and a saved recording.

Deformable tetrahedralization can
vary even with the same seed; record mesh/node counts when available. Do not regenerate or change
an asset between runs and call the comparison controlled. Preserve caches; separate cold-start/JIT
time from warmed rollout throughput.

## Run the focused tests

Use [run-tests](../../run-tests/SKILL.md) with the selected runtime. Replace `<arena-pytest>` with
the full native or Docker launcher. Run these as separate processes, matching the CI split:

```bash
<arena-pytest> -sv --durations=20 \
  isaaclab_arena/tests/test_dependency_pins.py \
  isaaclab_arena/tests/test_newton_buffer_monitor.py \
  isaaclab_arena/tests/test_physics_presets.py

<arena-pytest> -sv --durations=20 \
  -m 'with_newton and not with_cameras and not with_subprocess' isaaclab_arena/tests

<arena-pytest> -sv --durations=20 \
  -m 'with_newton and with_cameras and not with_subprocess' isaaclab_arena/tests

<arena-pytest> -sv --durations=20 \
  -m 'with_newton and with_subprocess' isaaclab_arena/tests
```

Record selected/passed/failed/skipped counts and durations, preferably with per-invocation JUnit
files under the validation directory. Inspect the report's failures and errors even when the process
exits zero: Kit shutdown can mask pytest's exit status. An empty selection or skips are not passing coverage.
Inspect additional Newton callers because not every backend/configuration test is marked:

```bash
rg -n 'with_newton|PhysicsBackend.NEWTON|isaaclab_newton|import newton' isaaclab_arena/tests
```

The marked suite currently exercises DROID actuation and episode resets, deformable settling,
gear placement/grasp/success, camera reset, cable reset, background reset, and gear recording.
Include unmarked tests affected by changed APIs, such as asset physics configuration and event
pose writes. If changing shared dependencies, run relevant PhysX/camera regressions too. For full
suite validation, retain the separate camera/subprocess phases from `run-tests`.

For Lab API adaptations, run focused Lab tests for the changed contracts as well. Start with the
original failure reproduction, then complete the Newton phases once it passes. Preserve failures
from intermediate candidates; passing import/configuration tests alone does not verify simulation.

On failure, preserve the first failing node ID, traceback, warnings, and runtime provenance.
Reproduce under the baseline with the same configuration before classifying a failure as preexisting.
Log all attempts for intermittent failures; do not rerun until green or relax assertions to certify
the dependency update. Scripted-placement tests do not measure policy success rates.

## Re-run the documented Newton workflows

Read these pages from the checkout rather than an unrelated branch or a remembered command:

- [Deformable pick and place](../../../../docs/pages/example_workflows/newton/deformable_pick_and_place.rst)
- [Gear insertion](../../../../docs/pages/example_workflows/newton/gear_insertion.rst)

Use `<arena-python>` from `arena-runtime`. Set `NEWTON_VALIDATION_DIR` to a new host-storage
directory visible to the chosen runtime. Use separate `baseline` and `candidate` children; retain
partial outputs when a run fails. Native and Docker runs need distinct output directories too.

### Deformable pick and place

First run the page's bounded zero-action preview for 1,000 steps. Check initial plate contact,
settling, robot pose, and finite state values. For the policy evaluation, use
[serve-openpi-policy](../../../user/serve-openpi-policy/SKILL.md) to reuse or start the matching
`pi05` server; record its actual checkpoint/config and readiness. Then run the documented
16-episode, four-environment evaluation with video added for diagnosis:

```bash
<arena-python> isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
  --num_episodes 16 \
  --num_envs 4 \
  --seed 42 \
  --placement_seed 42 \
  --viz kit \
  --enable_cameras \
  --record_camera_video \
  --record_viewport_video \
  --output_base_dir "$NEWTON_VALIDATION_DIR/candidate/deformable" \
  droid_deformable_pick_and_place \
  --pick_object deformable_cube_newton
```

Use the identical command with the baseline source/dependencies and baseline output path. Resolve seed and
policy flags from the current CLI/config, and keep them identical. A headless or alternate-renderer
run is useful when Kit is unavailable, but report that difference rather than claiming to reproduce
the documented visual setup.

At skill creation, the page reports **9/16 Newton successes (56.25%)** with `pi05`; **15/16 is the
PhysX reference**, not Newton. Read the page again when using the skill because these values may
change. It explicitly describes a small stochastic validation, not an exact pass threshold.

Require 16 completed episode records, the report, and usable recordings. Policy Runner currently
writes `episode_results_rank0.jsonl` and `index.html` below its timestamped output directory;
Experiment Runner writes per-Run `episode_results_rebuild<N>.jsonl`,
`arena_experiment_result.json`, and `index.html`. Check actual artifacts instead of assuming the
two entrypoints share filenames. Separate execution failures from unsuccessful task episodes.
Verify each requested recording type exists and decodes: a recording flag or a rendered viewport is
not evidence that an MP4 was saved. If recording fails on the baseline too, document that integration
limitation separately from the Newton upgrade and use the available camera recordings for inspection.

### Gear insertion and other affected samples

Run the gear page's bounded zero-action preview, plus the marked grasp/hold, completion, camera,
and recording tests. These cover rigid contact/actuation separately from deformable VBD coupling.
Re-run the page's policy experiment when its endpoint and credentials are available, using
[run-experiment](../../../user/run-experiment/SKILL.md). Keep the same model, endpoint, prompt
variant, and pose-observation setting for baseline and candidate. If unavailable, report policy
validation as blocked while still completing the local preview and tests.

The gear page currently provides image-only, object-pose, and task-prompt VLM configurations and
an example video, but no fixed numeric success-rate reference. Do not invent one; use a measured
baseline and state the missing published rate. Check the Newton workflow index for other samples
affected by the dependency change. Do not substitute a different policy just to finish validation.

## Compare behavior, not just process exit

Inspect failed episodes and representative successful ones in the recorded camera and viewport
videos. Check for penetration after settling, cube blow-up/white frames on gripper contact,
grasp slippage, reset drift, broken robot motion, missing visuals, NaNs, contact-buffer overflows,
and new solver convergence warnings. White imagery alone is a symptom; correlate it with physics
state and logs before calling it a physical explosion.

Report a table with one row per workflow and runtime:

| Workflow / runtime | Documented reference | Baseline successes / episodes | Candidate successes / episodes | Failures and warnings | Warm rollout time / throughput |
| --- | --- | --- | --- | --- | --- |

Include stage durations when available: startup/JIT, scene creation/placement, rollout, and report/
video processing. Recording overhead belongs in both sides of a timing comparison.

One 16-episode sample cannot establish statistical equivalence or justify exact equality to 56.25%.
Report counts, percentage-point change, and uncertainty; matching rounded percentages is not proof.
For a material drop or ambiguous behavior, use additional matched seeds on both versions within
the evaluation budget and report every run. Agree on a larger budget or acceptance margin when
needed; do not repeatedly sample until a favorable rate appears. A consistent crash, new physical
instability, or broken reset is a regression even if aggregate policy success happens to improve.

Keep acceptance explicit: tests passed, runtime provenance matched, workflows completed, and no
regression was observed at the stated sample size, or list what failed/remains unverified. Preserve
the original documented benchmark; updating it requires a separately justified documentation change.
