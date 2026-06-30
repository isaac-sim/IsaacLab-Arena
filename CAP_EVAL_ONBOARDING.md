# GaP↔Arena eval — Arena-side onboarding & handoff status (authoritative)

## Reproducible refs (pin these)
- **Arena teammate branch:** `rcathomen/feature/cap-gap-eval`
- **Full pushed SHA:** `370b9e157dc0914cababba9e4e74a3fee6769041`
  (origin `git@github.com:isaac-sim/IsaacLab-Arena.git`; pushed, fresh branch, no force.)
- All placement + scoring changes ARE committed & pushed in that SHA (resting_in_bin/gripper_open,
  multi-object task, pooled-placement clearance thread, reach box). The local working branch is `rcathomen/cap`;
  the handoff branch points at the same commit.
- **GaP side:** `Isaac-cap` `main` @ `1249fb6` (+ uncommitted perception fix, see below) + `gap/graph-as-policy`.

## Python env — CANNOT `uv sync` from one checkout yet
The working venv came from a **separate** branch, `rcathomen/feature/uv-native-install` (commit `4082afd7e` +
**uncommitted** `pyproject.toml`/`uv.lock` fixes), which is **NOT an ancestor** of the cap eval branch. So a single
checkout of `rcathomen/feature/cap-gap-eval` + `uv sync` will NOT reproduce the env.

Pick ONE to make it reproducible (Arena-team decision, env owner = uv-native-install):
1. **Combined branch (cleanest):** commit `pyproject.toml`+`uv.lock` on `uv-native-install`, then rebase/merge the
   cap eval commit onto it → one branch → `git checkout <combined>` + `uv sync` gives env + eval code.
2. **Two-step (current reality):** teammate gets the env from `uv-native-install` (`uv sync` once its
   `pyproject/uv.lock` are committed), and runs the cap eval code via `PYTHONPATH=<cap-worktree>` (the `dev_run.sh`
   pattern). The cap branch carries the eval code only, not the install.
3. **Docker (supported fallback):** the `dev-container` skill / `./docker/run_docker.sh` — Arena's standard
   container (Isaac Sim 6.0 + Isaac Lab 3.0). Repo mounts at `/workspaces/isaaclab_arena`, `python` = `/isaac-sim/python.sh`.

Versions in the working env: **Isaac Sim 6.0.0.1, Isaac Lab 3.0.0b2**, `isaaclab_arena 1.0.0`.

## Canonical eval command / config
GaP server (gap venv) — start AFTER eval logs `[GapRemotePolicy] connecting`:
```
cd /home/rafael/Projects/gap/graph-as-policy
set -a; . ~/.config/gap/vlm.env; set +a
MUJOCO_GL=egl GAP_PERCEPTION_CACHE=0 uv run gap run \
  /home/rafael/Projects/Isaac-cap/examples/<pick_place|grocery_packing> --real franka --no-rr-autostart --no-video
```
eval_runner (Arena venv + worktree + Isaac-cap on PYTHONPATH):
```
env OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  PYTHONPATH=/home/rafael/Projects/IsaacLab-Arena-cap:/home/rafael/Projects/Isaac-cap/src \
  /home/rafael/Projects/IsaacLab-Arena/.venv/bin/python -m isaaclab_arena.evaluation.eval_runner \
  --eval_jobs_config <job.json> --headless --enable_cameras --record_camera_video --output_base_dir <out>
```
Job JSON (multi-object Panda, single seed):
```json
{"jobs":[{"name":"moverify","arena_env_args":{"environment":"libero_object_packing","num_envs":1,
 "enable_cameras":true,"placement_seed":1,"control":"joint_pos","eval_task":"pick_place_in_basket"},
 "policy_type":"isaac_cap.gap_remote_policy.GapRemotePolicy",
 "policy_config_dict":{"gap_host":"127.0.0.1","gap_port":9000,"policy_device":"cuda","gap_adapter":"franka"},
 "num_episodes":1}]}
```
NOTE: the job JSON + the overnight bash driver were in `/tmp` and were lost in a crash — recreate from the above.

## Still uncommitted (must commit for a clean teammate setup)
- **uv-native-install:** `pyproject.toml` + `uv.lock` (the install) — env owner's; commit to make the venv reproducible.
- **Isaac-cap `main`:** `examples/grocery_packing/scripts/perceive_dino_vlm.py` + `workflow.json` (perception fix —
  oversized-box >35%-frame filter before containment-NMS + prompt exclusion; verified working).
- Local-only, do NOT commit: `dev_run.sh` (absolute paths), `CAP_EVAL_ONBOARDING.md` is this doc.

## PANDA (verified) vs DROID (pending v2)
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
