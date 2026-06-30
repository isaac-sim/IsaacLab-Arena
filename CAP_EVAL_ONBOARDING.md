# GaP↔Arena eval — Arena-side onboarding & handoff status (authoritative)

## Reproducible refs (pin these)
- **Arena worktree:** `/home/rafael/Projects/IsaacLab-Arena-cap` (local working branch `rcathomen/cap`).
- **Arena teammate/handoff branch:** `rcathomen/feature/cap-gap-eval`
  (origin `git@github.com:isaac-sim/IsaacLab-Arena.git`; pushed, fast-forward only, no force.)
- **Current pushed tip:** the handoff branch points at **HEAD** (this onboarding-doc commit). The **Step-3 code
  tip** is `896f5c31b` (see the Step-3 section below), with this doc committed on top.
- **Step-3 commits (in order):** `236017d48` (opt-in `--gap_profile` Maple scene + camera-variation fix),
  `20705b60d` (G/H provenance + initial/final pose recorder metadata), `896f5c31b` (opt-in 2-3 object
  `SortMultiObjectTask` profile — the Step-3 code tip). All scene-smoke-only — **no GaP E2E claimed.**
- **GaP side:** `/home/rafael/Projects/Isaac-cap` branch `origin/rcathomen/gap-ila-eval`, current pushed tip
  `2b7a9f0` (after the earlier safety checkpoint `4f3bdd4`), plus `gap/graph-as-policy`. **2c-ii** at `2b7a9f0`
  adds the fail-closed single-use per-episode task payload and the first task-config graph-loader node
  (plumbing), but **those outputs are not yet consumed**; **#20** (explicit cursor / perception / confirmation)
  is active. **GaP E2E remains pending** — repin after #20 lands.

## Python env — generic Arena prerequisite (owned elsewhere; NOT a CAP branch)
The Arena runtime is a generic native uv install on `rcathomen/feature/uv-native-install`
(**Isaac Sim 6.0.0.1 / Isaac Lab 3.0.0b2**, `isaaclab_arena 1.0.0`). CAP treats it purely as a prerequisite —
CAP does NOT fork it or combine with it.

⚠️ **UNVERIFIED PIN (blocks full reproducibility):** the working venv used **uncommitted** `pyproject.toml` +
`uv.lock` fixes on top of branch tip `4082afd7e`. So `git checkout rcathomen/feature/uv-native-install && uv sync`
does **NOT** reproduce the exact env until those lock fixes are pushed by the native-install owner. **Get the pushed
install SHA from that owner before relying on the native path** — until then this handoff is NOT end-to-end reproducible.

Teammate env — pick one:
1. **Native (once the install SHA is pushed):** checkout that pushed `uv-native-install` commit + `uv sync`.
2. **Docker (supported, reproducible now):** `dev-container` skill / `./docker/run_docker.sh` (Isaac Sim 6.0 + Isaac
   Lab 3.0); repo mounts at `/workspaces/isaaclab_arena`, `python` = `/isaac-sim/python.sh`.

Then run CAP against that env via `PYTHONPATH` (two-step; no combined branch) — see the eval command below.

## Canonical eval — unchanged generic grocery_packing graph as the policy under test
**Baseline:** the **unchanged, generic** GaP `grocery_packing` graph is the policy under test (no per-episode
structured target payload, no graph inputs from Arena). **Arena owns** the DROID embodiment, the Maple scene +
its variation, the episode lifecycle, and scoring (stock `SortMultiObjectTask` over the 2-3 `pick_targets`).
The canonical policy is **managed-spawn** (`auto_spawn=true`): the policy launches/owns one GaP process per
episode, so there is **no** separate manual `gap run` terminal.

**Canonical job (tracked):** `isaaclab_arena_environments/eval_jobs_configs/maple_gap_droid_eval_jobs_config.json`
— DROID `droid_abs_joint_pos`, `gap_profile`, explicit `use_staging_assets`, `pick_targets` (2-3 scene objects)
→ `grey_bin_robolab`, `policy_config_dict.auto_spawn=true` + `gap_graph=grocery_packing`, `num_episodes=1`,
realistic `episode_length_s=450`. Variation/seeds come from **Arena's normal mechanism** (the `variations` block —
`camera_extrinsics_exterior_cam` — plus `seed` / `placement_seed`), NOT graph inputs.

Run it (single command; the policy manages GaP itself):
```
env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  PYTHONPATH=/home/rafael/Projects/IsaacLab-Arena-cap:/home/rafael/Projects/Isaac-cap/src \
  /home/rafael/Projects/IsaacLab-Arena/.venv/bin/python -m isaaclab_arena.evaluation.eval_runner \
  --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/maple_gap_droid_eval_jobs_config.json \
  --record_camera_video --output_base_dir /tmp/maple_gap_eval_out
```
⚠️ **Do not run the long E2E until the CAP cleanup/pin is confirmed** (see "PENDING" in the Step-3 section).
The auto_spawn launcher / generic-graph wiring lives on the GaP side (`Isaac-cap`), repinned after CAP confirms.

**Quick scene-only validation (no GaP, zero-action, fast):** the tracked smokes
`maple_gap_scene_smoke_jobs_config.json` (single) and `maple_gap_sort_smoke_jobs_config.json` (2-3 sort) build
the scene + `exterior_cam` RGB-D + variation + provenance without a GaP server — use these to check
construction/cameras/provenance quickly.

## Still uncommitted (must commit for a clean teammate setup)
- **uv-native-install:** `pyproject.toml` + `uv.lock` (the install) — env owner's; commit to make the venv reproducible.
- Local-only, do NOT commit: `dev_run.sh` (absolute paths). (This doc, `CAP_EVAL_ONBOARDING.md`, IS tracked and committed.)

## PANDA / DROID libero runs — HISTORICAL (pre-Step-3; does NOT override Step-3 status)
> The section below records earlier libero-scene grocery-packing runs and is kept for context only. It predates
> the Step-3 Maple work and its `success=true`/`success=false` notes do **not** describe the Maple profile. For
> the Maple GaP path, the authoritative status is the Step-3 section (scene-smoke-only; GaP E2E pending CAP).
**PANDA — `control=joint_pos`, `gap_adapter=franka`, no `GAP_HAND_TO_FINGERTIP_Z` (0.1029):**
- Single-object: scored `success=true` (proven end-to-end).
- Multi-object (grocery_packing, 3 items): perception fix VERIFIED — VLM selects groceries (not the robot mask),
  oversized table-box dropped before NMS; all 3 GRASPED + transported; no early-fire (gripper_open guard works).
- `success=false` here is an **HONEST scorer result, NOT a threshold bug**: GaP's 4th cycle re-detected a grocery
  that had BOUNCED/FALLEN OUT of the bin (near floor z~0.09, x~0.58 vs basket center ~0.28) — genuinely not all-in-bin.
  **Do NOT widen `resting_in_bin`** (keep the scorer honest).
- OPEN (GaP-side): tighten the transport/drop target to the bin INTERIOR + improve release/settle so objects don't
  bounce out; extend the episode timeout (`150*N+150`=600s is short once a bounce-out forces a re-pack 4th cycle).
  Re-verify after those fixes.

**DROID — `control=droid_joint_pos`, `gap_adapter=droid`, `GAP_HAND_TO_FINGERTIP_Z=0.157`:**
- Interface proven end-to-end; grasp closes on air — the Robotiq π/4 mount rotation is applied only to the position
  TCP in the connector IK, NOT to the grasp-pose orientation in `plan_grasp.py` → fingers 45° off. Fix = pre-rotate
  grasp candidates by the mount rotation. Plus TCP-by-embodiment cleanup (GaP `--real franka` sets Robotiq 0.157 even for Panda).

## Step 3 — Maple-table GaP scene profile (pushed; scene-smoke-only)
Arena-owned work to bring the stock `pick_and_place_maple_table` DROID path onto the GaP eval flow. **Opt-in:**
stock single-object defaults and all existing VLA jobs are **unchanged** (verified). Commits on
`rcathomen/feature/cap-gap-eval`: `236017d48`, `20705b60d`, `896f5c31b` (tip).

**What it adds (only under the opt-in flags):**
- `--gap_profile` (off by default; fails early unless `--enable_cameras` + a DROID embodiment): attaches a fixed
  exterior RGB-D agentview camera named `exterior_cam` (`rgb` + `distance_to_image_plane`, 512×800) with a STATIC
  pose, registers its `camera_extrinsics_exterior_cam` variation, and constrains object placement to the arm reach box.
- Camera contract is **live-pose**: Arena owns the pose; the adapter must read live `cam.data.pos_w` /
  `quat_w_ros` / `intrinsic_matrices` (no re-aim, no cached pose_mat). This Isaac Lab is **xyzw** (quat_from_matrix /
  OffsetCfg.rot / root_quat_w all `(x,y,z,w)`).
- `--episode_length_s` configurable (stock default 20 s; raise it for GaP rollouts — see "pending" below).
- `--pick_targets` (nargs='+', default None) → opt-in stock `SortMultiObjectTask` over 2-3 ordered, unique targets
  into `--destination_location`. Fail-closed: a present flag must carry 2-3 unique names, non-overlapping the
  destination/distractors. Unset keeps the stock single-object `PickAndPlaceTask`.

**Tracked scene-smoke configs** (`isaaclab_arena_environments/eval_jobs_configs/`), both `zero_action`, no GaP server:
- `maple_gap_scene_smoke_jobs_config.json` — single pick-place, `gap_profile` + staging + `exterior_cam` variation.
- `maple_gap_sort_smoke_jobs_config.json` — 3 groceries (alphabet_soup, tomato_sauce, butter) → `grey_bin`, sort profile.

Run either (Arena venv + worktree + Isaac-cap on PYTHONPATH; no GaP needed):
```
env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  PYTHONPATH=/home/rafael/Projects/IsaacLab-Arena-cap:/home/rafael/Projects/Isaac-cap/src \
  /home/rafael/Projects/IsaacLab-Arena/.venv/bin/python -m isaaclab_arena.evaluation.eval_runner \
  --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/maple_gap_sort_smoke_jobs_config.json \
  --record_camera_video --output_base_dir /tmp/maple_sort_out
```

**⚠️ Staging-only asset caveat:** the Maple table (`maple_table.usda`) and DROID stand (`franka_stand_grey.usda`)
are PR #786 assets that 404 on the production Nucleus but resolve on the Isaac staging bucket (CI used staging) —
this fails on Docker/official beta too; it is a documented external promotion blocker, not a regression. The smoke
configs opt into `--use_staging_assets` (targeted, fail-closed production→staging host swap for ONLY those two
files; instance-local, no leak into other jobs). Production stays the default until PR #786 is promoted.

**Provenance recorded per episode** (opt-in, under `gap_provenance` in the per-episode JSONL; schema is independent
of CAP's scalar `target_specs`): `profile`, `asset_channel` (`production|staging`), resolved `table_usd` +
`droid_stand_usd`, `task` (`PickAndPlaceTask|SortMultiObjectTask`), ordered `pick_targets` (exact CLI order),
`destination`, `distractors`, `placement_seed`, `seed`. Plus `initial_object_poses` and `final_object_poses`
(separate; world-frame `pos_w` + `quat_w_xyzw`, keyed by stable asset name; initial = post-reset snapshot, final =
episode end).

**PENDING (CAP-owned; NOT done here):** GaP end-to-end on Maple and the canonical `GapRemotePolicy` eval job (with
the intended 450–600 s timeout) remain blocked on CAP's active task-input / graph work — the adapter must consume
the live `exterior_cam` pose and the provenance/target payload, and the graph must drive the Maple targets. **No
passing GaP Maple job is claimed or provided yet.**
