# GaP to Arena: Arena-Side Handoff

Current as of 2026-06-30. Start from the `Isaac-cap` repository for the complete setup and runbook;
this file records the Arena-owned part of the integration.

## Reproducible Revisions

| Component | Branch | Required commit |
|---|---|---|
| Arena CAP source (this worktree) | `rcathomen/feature/cap-gap-eval` | `444307d3cf7b77d2c93c8a1303979accdc238217` |
| Arena native `uv` environment | `rcathomen/feature/uv-native-install` | `ce1b93de6ad5ca188a75f6ff36a387fceb860191` |
| Isaac-cap | `rcathomen/gap-ila-eval` | `f0487f2cc216adc4c10e7cc2ef2840eb7fcf4ca0` |
| `graph-as-policy` private handoff mirror | `cap-eval-c24fafb` | `c24fafb126dcb27b2c9f13fbdd143d436851e5a9` |
| DROID TCP skill fork | `rcathomen/cap-droid-tcp` | `4ce0af76d9cb4dbf92ebacf92de7af28ef1ff1fa` |

Use the full revisions from `Isaac-cap/external/PINNED_VERSIONS.md`; short revisions above are only
for orientation. The local and canonical remote branch are both
`rcathomen/feature/cap-gap-eval`.
The original GaP URL no longer resolves; the exact runtime is mirrored privately at
`rafaelcathomen/graph-as-policy-cap` and requires access.

## Current Result

The Maple/DROID five-object job completed with Arena `success=true`:

- task: five HOPE groceries into `grey_bin_robolab`;
- episode: 14,745 simulation steps;
- policy: GaP `grocery_packing`, managed by `GapRemotePolicy`;
- scene: DROID (Panda arm + Robotiq 2F-85), Maple table, relation-solved placement;
- evidence on the development machine:
  `Isaac-cap/outputs/videos/gap_front_view/gap_droid_five_object_success_result.jsonl`.

This supersedes the older notes in which Maple GaP end-to-end was still pending.

## Arena-Owned Changes

- `pick_and_place_maple_table` gained an opt-in `gap_profile`; stock behavior stays unchanged when
  the flag is absent.
- The profile uses the DROID embodiment, absolute joint-position commands, and an exterior RGB-D
  camera whose live pose and intrinsics are sent to GaP.
- Camera variation is synchronized into Fabric on Isaac Lab 3.0.0b2 so rendered pixels and
  `Camera.data.pos_w` agree on the first GaP observation.
- `pick_targets` supports two through five unique grocery assets and selects Arena's stock
  `SortMultiObjectTask`; one target continues to use `PickAndPlaceTask`.
- Grocery objects and the bin are placed through Arena relations and reset-time variation. The GaP
  profile adds a reachable XY region because unrestricted table placement produced approach-IK
  failures.
- Arena owns episode termination, contact-based scoring, video, provenance, and initial/final
  object-pose recording. GaP graph completion is not used as the benchmark score.
- Two scene-only job files exercise short reset/layout variation without running GaP.

## Canonical Jobs

The current CAP jobs live in `Isaac-cap/configs/arena/`:

- `droid_single_alphabet_seed71.json`: quick known-good single-object smoke;
- `droid_five_object_stress_seed1.json`: verified five-object task;
- `droid_five_object_layout_variations.json`: ten short zero-action scene resets;
- `droid_single_mustard_seed72.json` and `droid_single_milk_seed73.json`: additional object/layout
  demonstrations.

`auto_spawn=true` means the Arena policy starts and reaps one GaP subprocess per episode. There is
no second terminal in the canonical flow. From `Isaac-cap`:

```bash
CAP_WORKSPACE="$HOME/Projects/cap-eval" \
ARENA_JOB_CONFIG="$PWD/configs/arena/droid_single_alphabet_seed71.json" \
./scripts/run_arena_eval.sh
```

The exact clone/install/preflight sequence is in `Isaac-cap/docs/TEAMMATE_SETUP.md`.

## Asset Caveat

The Maple table and DROID stand are not yet promoted on the production asset host. The profile uses
the explicit, fail-closed `use_staging_assets` option to rewrite only those two URLs to the staging
host. This is not a silent fallback, and the resolved URLs are stored in episode provenance.

## Known Limits

- GaP is synchronous and supports one Arena environment per process; sweeps run sequentially.
- The success metric is Arena's stock direct-contact relation. A visually contained object stacked
  entirely on another object can therefore remain unsuccessful, although the verified five-object
  run satisfied the metric.
- The graph has no persistent object identity or explicit post-close grasp verification. A missed
  object can be perceived and attempted again.
- Runtime is slow because VLM inference and conservative cuRobo trajectories dominate.
- Native `uv` supports this evaluation path, but an upstream `numba`/`coverage` incompatibility
  remains for unrelated Isaac Lab features that import `numba`; use Docker for those features.

## Verification

The patch at `444307d3c` passed nine focused tests covering camera variation/Fabric coherence,
staging isolation, and two-to-five-target parsing. Both scene-variation job files pass JSON and
`JobManager` validation. `dev_run.sh` is a local absolute-path helper and is intentionally not
committed.
