# llm_env_gen — change log

## Xinjie Yao — 2026-04-27

### 1. What it can do

- **Compute the Franka EEF reachability map** with batched cuRobo IK on GPU,
  standalone (no SimulationApp / Kit boot). 25³ = 15,625 voxels solve in ~24s.
- **Save reach-map artifacts** to disk: a `.npz` with success / pos_err /
  rot_err arrays + axis grids, and a 3-D matplotlib voxel render PNG with the
  robot base origin and an x/y/z axis triad overlaid.
- **Visualize the reach map inside Kit**, overlaid on a registered Arena env:
  green/cyan sphere markers at feasible voxels (split by IK pos-error median),
  red sphere at the robot base. Loads a precomputed `.npz` to skip cuRobo,
  so the only cost is Kit boot.

**Assumptions:**

- `isaaclab_arena-curobo` container running.
- For `viz_reach_map_kit.py`: top-level CLI flags must come **before** the env
  name positional (Arena's env subparser owns args after the env name).

### 2. What was added

**`tools/compute_reach_map.py`** *(new)*

- Pure cuRobo + PyTorch — imports `IKSolver` directly from `curobo.wrap.reacher`
  with `RobotConfig.from_dict(load_yaml("franka.yml")["robot_cfg"])`. No Isaac
  Sim env needed, so no SimApp boot.
- 3-D `torch.linspace` grid in robot base frame (`--x_min / --x_max / --y_min /
  --y_max / --z_min / --z_max`, default `[-0.4, 1.0] × [-0.9, 0.9] × [0.0, 1.4]`).
- Top-down EE quaternion (wxyz `[0, 1, 0, 0]`) by default; configurable via
  `--quat_wxyz`. `--num_seeds` controls cuRobo IK seed count (default 20).
- `--save_npz <path>`: stores `success`, `pos_err`, `rot_err`, axis grids,
  and the EE quaternion.
- `--save_png <path>`: matplotlib `Agg` voxel render colored by IK position
  error, with a red base marker at the origin, dashed plumb line up through
  the workspace, and an RGB axis triad. Works headless.

**`tools/viz_reach_map_kit.py`** *(new)*

- Brings up a registered Arena env (e.g. `avocadoPnPbowltable`) under
  `--viz kit` and scatters `VisualizationMarkers` sphere instances at every
  feasible voxel in world frame.
- Two reach-map sphere bins (`tight` green, `loose` cyan) split at the
  feasible-voxel pos-error median to show robust vs marginal reach.
- Separate red sphere marker (`/World/Visuals/reach_map_base`, radius 0.06)
  at the robot's world root pose.
- `--load_npz <path>`: loads the artifact saved by `compute_reach_map.py` and
  skips the cuRobo step entirely. Falls back to in-process cuRobo IK if no
  NPZ is provided.
- Robot world pose fetched via `wp.to_torch(robot.data.root_pos_w)[0, :3]`
  and same for `root_quat_w` (Isaac Lab exposes these as warp arrays, not
  torch tensors). Voxels in base frame are rotated to world via
  `isaaclab.utils.math.quat_apply` before being passed to `markers.visualize`.

### 3. Commands

```
# Compute the reach map and save NPZ + PNG (no Kit, ~25s on GPU).
docker exec isaaclab_arena-curobo bash -c \
  'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh tools/compute_reach_map.py \
     --grid 25 \
     --save_npz tools/franka_reach_top_down.npz \
     --save_png tools/franka_reach_top_down.png'

# Show the reach map in the Kit viewer overlaid on avocadoPnPbowltable.
# Note: --load_npz must come BEFORE the env name positional.
docker exec isaaclab_arena-curobo bash -c \
  'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh tools/viz_reach_map_kit.py \
     --viz kit --num_envs 1 \
     --load_npz tools/franka_reach_top_down.npz \
     avocadoPnPbowltable --embodiment franka_ik'
```

### 4. TODOs

- Loop over a small set of EE quaternions (top-down, side approach, ±yaw)
  inside `compute_reach_map.py` and store *fraction reachable* per voxel —
  smoother [0, 1] field, suitable as the differentiable proxy `W` for
  the placement-optimization design.
- Color the Kit reach markers by manipulability `√det(JJᵀ)` instead of (or in
  addition to) IK pos-error bins — cuRobo exposes the Jacobian at `q*`.
- Wire `viz_reach_map_kit.py` to honor the `--quat_wxyz` from the NPZ so the
  Kit overlay matches the orientation the map was computed for.

## Qian Lin — 2026-04-24

### 1. What it can do

- **Generate open/close door environments** from a natural-language prompt: `llm_env_gen` now produces `OpenDoorTask` or `CloseDoorTask` environments for articulated appliances (microwave, fridge, cabinet).
- **Relational placement for openable objects**: the microwave is placed ON the counter via the constraint solver (no fixed world position), with `RotateAroundSolution` locking its door to face the robot.
- **Correct tabletop anchor for kitchen background**: `ObjectReference` for the kitchen counter is created without `ObjectType.RIGID` (the sub-prim lacks `RigidBodyAPI`), preventing the previous `RuntimeError`.
- **IK reachability check for open-door tasks**: `run_reachability_check.py` auto-detects `OpenDoorTask`/`CloseDoorTask` and checks a horizontal front-approach pose at the door center.
- **Two generated microwave envs**: `microwaveOpenkitchen` (kitchen background) and `microwaveOpentable` (table background) are registered and runnable with `zero_action` policy.

**Assumptions:**
- Docker container `isaaclab_arena-curobo` (or `-latest`) must be running.
- `NV_API_KEY` env var required for LLM generation (`try_schema.py`).
- `--door_facing_axis -x` is correct for kitchen-background envs (microwave oriented by `RotateAroundSolution(yaw=-π/2)`).

### 2. What was added

**`schema.py`**
- `RelationKind` extended with `"open"` and `"closed"` literals — Pydantic validation gate for unary door-state relations.

**`placement_proposer.py`**
- `_OPEN_DOOR_KINDS`, `_CLOSE_DOOR_KINDS`, `_DEFAULT_OPEN_DOOR_EPISODE_LENGTH_S` constants.
- `_APPLIANCE_FACING_YAW` dict: maps `(background, asset)` → yaw for `RotateAroundSolution`.
- `_BACKGROUND_TABLETOP_ANCHOR_BASE_TYPE`: backgrounds whose tabletop sub-prim lacks `RigidBodyAPI` (currently `{"kitchen"}`).
- `TabletopAnchorPlan.anchor_object_type` field (`"RIGID"` or `"BASE"`); `_plan_tabletop_anchor` sets it per background.
- `RelationSpec.kind` extended with `"rotate_around_solution"`; new `rotate_yaw_rad` field.
- `PlacementItem.is_openable` flag; `_propose_items` detects openable subjects from goal diff and appends `RotateAroundSolution` after building the `On` relation.
- `TaskPlan.kind` extended with `"open_door"` and `"close_door"`.
- `_open_door_plan` and `_close_door_plan` task plan builders; `_plan_task` dispatches on goal kind.
- `block_initial_goal_satisfaction` skips `open`/`closed` goals (no `Not(...)` wrapping needed).
- `_derive_env_name` / `_derive_class_name` handle `target=None` for unary goals.

**`env_writer.py`**
- `_render_anchor_setup` respects `anchor_object_type`: omits `object_type=ObjectType.RIGID` for `BASE`-type anchors.
- `_render_one_relation` handles `"rotate_around_solution"` → `RotateAroundSolution(yaw_rad=...)`.
- Dynamic `relation_imports` now includes `RotateAroundSolution` when used.

**`llm_agent.py`**
- System prompt updated: explains `open`/`closed` unary relations, null target, initial/final graph semantics, and that distractors still need `on(distractor, background)`.

**`reachability_utils.py`**
- `find_open_close_door_task(arena_env)`: duck-typed task finder for `openable_object` attribute.
- `get_articulation_world_pos(env, name, env_id)`: root position from `scene.articulations`.
- `get_scene_object_world_pos(env, name, env_id)`: tries rigid objects, falls back to articulations.
- `get_object_pos_in_robot_frame` updated to use `get_scene_object_world_pos`.
- `door_approach_quaternion_wxyz(door_facing_axis)`: quaternion mapping `{-x, +x, -y, +y}` to a horizontal hand approach.
- `build_curobo_door_approach_pose(...)`: places hand `door_approach_offset` m in front of door center with horizontal orientation.
- New CLI args: `--door_approach_offset` (default 0.10 m), `--door_facing_axis` (default `-x`).

**`run_reachability_check.py`**
- `main()` auto-detects task type: tries `find_pick_and_place_task`, falls back to `find_open_close_door_task`.
- Refactored into `_check_pick_and_place(...)` and `_check_open_door(...)` helper functions.
- JSON payload and marker names adapted per task type.

**New environment files**
- `isaaclab_arena_environments/microwaveOpenkitchen.py` — kitchen counter background.
- `isaaclab_arena_environments/microwaveOpentable.py` — maple table background.

### 3. Commands

```bash
# Generate an open-door env from prompt (kitchen background)
docker exec isaaclab_arena-curobo bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema \
  --prompt 'franka open microwave door on top of a kitchen table. there are other veggies on the table outside of the microwave as distractors' \
  --background kitchen \
  --write-env isaaclab_arena_environments/"

# Run zero-action policy on generated microwave env (kitchen)
docker exec isaaclab_arena-curobo bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 10 \
  microwaveOpenkitchen --embodiment franka_ik"

# IK reachability check for open-door task
docker exec isaaclab_arena-curobo bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_reachability_check.py \
  --viz kit --num_envs 1 microwaveOpenkitchen --embodiment franka_ik \
  --door_approach_offset 0.12 --door_facing_axis -x"
```

### 4. TODOs

- `isaaclab_arena_environments/microwaveOpenkitchen.py:87` — `# TODO(relation kind 'next_to' has no generator support yet)` for broccoli, sweet_potato, red_bell_pepper next_to microwave
- `isaaclab_arena_environments/microwaveOpentable.py` — same `next_to` TODOs for distractor veggies

## Xinjie Yao — 2026-04-23

### 1. What it can do

- **Randomize the robot base around the tabletop** on every regenerated env — a new `RobotPlacement` field in `Placement` samples one of the four tabletop edges (`x_min` / `x_max` / `y_min` / `y_max`), picks a fraction in `[0.4, 0.6]` along that edge, plants the base 0.1 m outside, and yaws it to face the table center. The avocado env now emits an `embodiment.set_initial_pose(...)` block driven by the runtime tabletop bbox rather than a hardcoded pose.
- **Seeded by env name** so regenerating the same env yields the same pose (`random.Random(env_name)`). Different envs → different edges; same prompt → reproducible pose.
- **Only emitted for tabletop-anchored backgrounds** (gated on `TabletopAnchorPlan.emit_position_limits`) so non-tabletop flows stay untouched.
- **Faster default viewer dwell** on reachability runs — `--dwell_steps` default dropped from 1500 (~50 s) to 300 (~10 s); bump it back up when you need longer manual inspection.

**Assumptions:** generated env uses a registered embodiment whose `set_initial_pose(Pose)` is wired through `EmbodimentBase._update_scene_cfg_with_robot_initial_pose` (verified for `franka_ik`). `no_embodiment` currently rejects `set_initial_pose` because its `scene_config` has no `robot` — use `franka_ik` for smoke tests that exercise the pose block. The template now materializes `_tbl_min_xyz` / `_tbl_max_xyz` **before** `embodiment = …`, so any downstream renderer must preserve that order.

### 2. What was added

- **`isaaclab_arena/llm_env_gen/placement_proposer.py`**
  - `RobotPlacement` dataclass — `edge`, `fraction`, `offset_m`, `z_m`, `rotation_xyzw`.
  - `Placement.robot_placement: RobotPlacement | None` — `None` when the background has no usable tabletop bbox.
  - `_EDGE_ROTATION_XYZW` — yaw quaternion table keyed by edge name (4 cardinal orientations).
  - `_ROBOT_EDGE_OFFSET_M = 0.1`, `_ROBOT_EDGE_FRACTION_RANGE = (0.4, 0.6)` — tunables kept at module scope so they're easy to find.
  - `_propose_robot_placement(env_name)` — seeded sampler wired into `propose_placement`.
- **`isaaclab_arena/llm_env_gen/env_writer.py`**
  - `_render_robot_pose(rp)` — emits the `_robot_x` / `_robot_y` / `embodiment.set_initial_pose(...)` block from a `RobotPlacement`, sharing one template across the four edges via compact axis-expr dispatch.
  - Template reordered: `{bbox_setup}` now renders **before** embodiment creation so the robot-pose block can reference `_tbl_min_xyz` / `_tbl_max_xyz`.
- **`isaaclab_arena/llm_env_gen/reachability_utils.py`** — `--dwell_steps` default dropped from 1500 to 300 (~10 s); help text updated to say "bump for longer inspection".
- **`isaaclab_arena/llm_env_gen/run_reachability_check.py`** — shrank the `_make_big_frame` default scale from 0.3 → 0.15 so the reachability target frame is still identifiable against Arena's 0.1 ee_frame markers without dominating the viewport.
- **`isaaclab_arena_environments/avocadoPnPbowltable.py`** — regenerated under the new pipeline. Sampled edge `x_max` at fraction 0.448 → Franka sits east of the table, yaw 180° facing inward. Verified with the `verify-env-with-zero-action` smoke test (500 steps, Kit viz, `franka_ik`, exit 0).

### 3. Commands

```bash
# Regenerate an env with a randomized robot placement
/isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema \
    --background maple_table_robolab \
    --write-env isaaclab_arena_environments

# Smoke-test the regenerated env with franka_ik + Kit viz
docker exec isaaclab_arena-curobo bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh -u -m isaaclab_arena.evaluation.policy_runner \
    --policy_type zero_action --num_envs 1 --num_steps 500 --viz kit \
    avocadoPnPbowltable --embodiment franka_ik"
```

### 4. TODOs

- `no_embodiment` still crashes because its `scene_config` has no `robot` attribute — generated envs now unconditionally call `embodiment.set_initial_pose(...)`. Either guard the emit site on `args_cli.embodiment != "no_embodiment"` or make `NoEmbodiment.set_initial_pose` a no-op so the `verify-env-with-zero-action` skill keeps working.
- Robot placement currently ignores the pick/destination geometry — a future feasibility gate (IK reachability on both objects) should reject or resample placements where neither object is in the Franka work envelope from the sampled edge.

## Qian Lin — 2026-04-23

### 1. What it can do

- **Check IK reachability** for any Arena pick-and-place environment: given any `PickAndPlaceTask` + Franka embodiment, determine whether the pick object's pose is kinematically reachable (IK, collision-unaware) from the robot's initial configuration.
- **Check motion-plan feasibility**: if IK passes, run a full cuRobo collision-aware trajectory planner to confirm a collision-free approach path exists to the top-down grasp pose.
- **Visualize in Rerun**: stream robot collision spheres, EE trajectory, and world meshes to a Rerun viewer via X11, or save a `.rrd` file for offline replay when no display is available.
- **Tune IK thresholds** at the CLI (`--position_threshold`, `--rotation_threshold`) to probe borderline poses.
- **Bake cuRobo into the Arena Docker image** with a single rebuild flag (`-c`) — installs `cuda-toolkit-12.8` + cuRobo from pinned source; no manual container patching required.

**Assumptions:** `./docker/run_docker.sh -c -r` builds the `isaaclab_arena:curobo` image. The `install_cuda.sh` script is always present in the Arena source tree. cuRobo must be compiled from source (PyPI stub is non-functional). Rerun viewer version on the host must match the SDK in the container (currently `0.31.3`). Do NOT start `rerun` manually before running visualization — `rr.spawn()` inside the script owns the viewer lifecycle.

### 2. What was added

**`isaaclab_arena/llm_env_gen/check_ik_reachability.py`** (new)
- Arena-CLI-compatible entry point using the standard subparser pattern
- `_assert_franka_embodiment`, `_resolve_pick_object_name` — validate env constraints before cuRobo init
- `_build_curobo_target_pose` — top-down grasp target (TCP offset + configurable axis rotation)
- `_run_ik_reachability` — calls `motion_gen.ik_solver.solve_single`; returns position/rotation errors
- `_run_motion_plan` — calls `planner.update_world()` + `planner.plan_motion()`; collision-aware
- `_log_initial_scene` — unconditionally logs robot spheres + world meshes to Rerun regardless of plan success
- `--save_rrd PATH` with `rr.spawn` monkey-patch — saves recording to file, bypasses display requirement
- `--position_threshold` / `--rotation_threshold` CLI overrides for IK diagnostics
- `--visualize_plan` / `--visualize_spheres` flags (Rerun)

**`docker/Dockerfile.isaaclab_arena`**
- New `ARG INSTALL_CUROBO=false` block: re-COPYs `install_cuda.sh` (avoiding the GR00T block's unconditional `rm`), installs `cuda-toolkit-12.8`, builds cuRobo from pinned commit, persists `CUDA_HOME` via `/etc/profile.d/cuda.sh`

**`docker/run_docker.sh`**
- New `-c` flag: sets `INSTALL_CUROBO=true`, tags image as `isaaclab_arena:curobo`, passes `--build-arg INSTALL_CUROBO` to `docker build`

### 3. Commands

```bash
# Build the Arena Docker image with cuRobo baked in
./docker/run_docker.sh -c -r

# Run IK-only check (fast, no motion planning)
docker exec isaaclab_arena-curobo bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/llm_env_gen/check_ik_reachability.py \
    --headless --num_envs 1 --ik_only avocadoPnPbowltable"

# Full check with Rerun visualization (X11; do NOT start rerun manually)
docker exec -it isaaclab_arena-curobo bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/llm_env_gen/check_ik_reachability.py \
    --headless --num_envs 1 --visualize_plan --visualize_spheres \
    --top_down_offset 0.12 avocadoPnPbowltable"

# Save Rerun recording to file (no display / headless SSH)
# Then on host: rerun /tmp/plan.rrd
docker exec isaaclab_arena-curobo bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/llm_env_gen/check_ik_reachability.py \
    --headless --num_envs 1 --visualize_plan --save_rrd /tmp/plan.rrd avocadoPnPbowltable"

# Diagnose rotation convention or relax thresholds for borderline poses
#   --grasp_axis y          (try if x-axis gives rot_err ~0.7 rad)
#   --rotation_threshold 0.8 --position_threshold 0.015
```

## Xinjie Yao — 2026-04-22

### 1. What it can do

- Spawn items **inside containers** via a new `In` containment relation — XY clamped to the parent's footprint, Z nudged just above the parent's rim so gravity completes the deposit. Replaces the long-standing TODO at `isaaclab_arena/relations/relations.py`.
- Generate envs through a **two-stage pipeline**: `propose_placement` (pure data transform → `Placement` dataclass) feeds `write_env` (thin renderer). Feasibility gates can slot in between without touching the renderer.
- **Override the LLM's background** at the CLI (`--background maple_table_robolab` is the default) — the full avocado scene swaps cleanly to the maple tabletop with `ObjectReference`-based anchoring and runtime bbox-derived `PositionLimits`.
- **Stable env names** regardless of the chosen USD — `_BACKGROUND_NAME_ALIASES` folds `maple_table_robolab`, `office_table`, `packing_table`, etc. under `"table"` and kitchen variants under `"kitchen"` for slug generation. The avocado env stays named `avocadoPnPbowltable` no matter which table USD drives it.
- **Smoke-test any registered env** via the new project-level `verify-env-with-zero-action` skill — Kit-viz by default, `no_embodiment` + `zero_action`, failure-to-fix hint table for the common bring-up errors.
- **Forbid initial states that already satisfy a goal** via a new `Not(inner)` relation + `block_initial_goal_satisfaction` stage in the proposer — the avocado env now carries `Not(In(bowl))` so the solver starts the avocado on the table, never already inside the bowl. Closes the Layer 2 negative-constraint TODO on `RelationSolver`.

**Assumptions:** Docker container running; `NV_API_KEY` on host for any LLM flow; `openai` pip installed inside the container; generated env files land under `isaaclab_arena_environments/` for auto-discovery; tabletop-style backgrounds are the only ones with a tested anchor / bbox path (`_BACKGROUND_TABLETOP_ANCHOR`); `Not(inner)` only wraps binary `Relation` subclasses (unary inners would need a separate solver dispatch).

### 2. What was added

- **`isaaclab_arena/relations/`**
  - `In(Relation)` class — XY containment without Z constraint at the relation layer.
  - `InLossStrategy(slope, z_slope_ratio=0.1, z_margin_m=0.02)` — XY band loss identical to `OnLossStrategy`, plus a soft Z-point term at `parent_rim + z_margin` scaled by `slope * z_slope_ratio`. `z_slope_ratio=0.0` recovers pure XY-only.
  - `Not(Relation)` wrapper — forwards `inner.parent` so the solver's binary-relation dispatch handles it without special-casing. Binary inners only for now.
  - `NotRelationLossStrategy(margin, slope)` — inverts the inner strategy's loss via `max(0, margin - inner_loss) * slope`. `inner_strategy` is injected by the solver at call time (no cached back-reference to the strategies dict, which would create a cycle the configclass validator recurses into).
  - New Not dispatch branch in `RelationSolver._compute_total_loss`: when it sees `Not(inner)` it looks up `_get_strategy(inner)` and threads it through as `inner_strategy=...`.
  - Registered as `In: InLossStrategy(slope=100.0)` and `Not: NotRelationLossStrategy(margin=0.05, slope=100.0)` in `RelationSolverParams._default_strategies`.
- **`isaaclab_arena/llm_env_gen/placement_proposer.py`** (new) — typed bundle (`Placement`, `PlacementItem`, `RelationSpec`, `TabletopAnchorPlan`, `TaskPlan`, `GoalBindingSpec`) with `propose_placement()` as the pure transform. Owns naming, anchor lookup, per-item relation planning (including `In`), and task dispatch. `RelationSpec` gains a `"not"` kind whose rendered form is `Not(inner)`, where `inner` is itself a `RelationSpec`.
- **`block_initial_goal_satisfaction(placement, resolved)`** (new, same module) — Layer 2a stage that walks `resolved.goal_added`, looks up each subject + target, and appends `RelationSpec(kind="not", inner=<on|in>)` to the subject's relations. Explicit `if kind == "in" / elif kind == "on" / else -> unsupported` dispatch so unhandled kinds land as visible TODO comments in the generated env. Also attaches a `GoalBindingSpec` so downstream feasibility gates can read "where should this item end up?"
- **`isaaclab_arena/llm_env_gen/env_writer.py`** (rewritten) — thin `write_env()` shim now chained as `propose_placement → block_initial_goal_satisfaction → _render_module → write`. Renders `In(...)` and `Not(...)` in addition to `On(...)` / `PositionLimits(...)`, factoring `_relation_call_expr(rel)` so `Not`'s inner is rendered without string surgery.
- **`isaaclab_arena/llm_env_gen/try_schema.py`** — new `--background` CLI flag (default `"maple_table_robolab"`) that patches `SceneSpec.background` and all matching relation targets after the LLM returns but before resolution.
- **`_BACKGROUND_NAME_ALIASES`** — alias map that folds background variants into short family names for env slugs.
- **`isaaclab_arena_environments/avocadoPnPbowltable.py`** — regenerated under the new pipeline: `maple_table_robolab` background + tabletop `ObjectReference` + bbox-derived `PositionLimits` per item + `PickAndPlaceTask` wiring (`pick_up=avocado`, `destination=bowl`) + `avocado_obj.add_relation(Not(In(bowl_obj)))` so the solver never initialises with the goal already satisfied.
- **`isaaclab_arena_environments/avocadoInBowlTest.py`** (new) — hand-authored test env exercising the `In` relation: bowl on the maple tabletop, avocado spawned `In(bowl)`. Observed init positions place the avocado's XY on the bowl's centre.
- **`.claude/skills/verify-env-with-zero-action/SKILL.md`** (new, project-level) — smoke-test skill with Kit-viz default and a failure→fix hint table.

### 3. Commands

```
# Override the LLM's background when regenerating an env
/isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema \
    --background maple_table_robolab \
    --write-env isaaclab_arena_environments

# Verify any registered env with zero_action + Kit viz (200 steps)
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --viz kit --policy_type zero_action --num_steps 200 --num_envs 1 \
    <env_name> --embodiment no_embodiment

# Bring up the hand-authored In-relation test env
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_steps 5 --num_envs 1 \
    avocadoInBowlTest --embodiment no_embodiment
```

### 4. TODOs

- `isaaclab_arena/llm_env_gen/placement_proposer.py` (`_BACKGROUND_TABLETOP_ANCHOR` docstring) — `get_world_bounding_box()` on a plain standalone table background doesn't account for the 90° Z rotation applied by the template, so PL derived from it clamps the wrong axes; prefer `ObjectReference` sub-prim entries until the asset layer handles pose-aware AABBs.

## Qian Lin — 2026-04-22

### 1. What it can do

- Generate a **pick-and-place** task env from a natural-language prompt — when the goal diff is a single `on`/`in` relation between two resolved items, the writer emits a `PickAndPlaceTask` (subject → `pick_up_object`, target → `destination_location`); everything else falls back to `NoTask` with per-goal TODO comments and a docstring note explaining why.
- Scan a folder (or every asset in `AssetRegistry`) of USD files and emit `object_catalog.json` with dimensions, physics properties, and semantic labels via pure `pxr` — no Isaac Sim runtime required. Read the catalog back with `--list-classes [--by-dataset]`.
- Register a fresh USD batch end-to-end via the new `isaaclab-arena-assetgen` skill: discover → catalog → `<basename>_object_library.py` with globally-unique `<basename>_<name>` keys → wire into `asset_registry.py` → USD-exists pytest.
- Run the converted `avocadoPnPbowltable` env with a real success predicate (avocado-in-bowl contact) instead of the old `NoTask` placeholder.

**Assumptions:**
- Docker container running; `NV_API_KEY` exported on host for any LLM-driven flow.
- `openai` picked up from the new `RUNTIME_DEPS` entry — requires `pip install -e .` (or image rebuild) inside the container.
- `isaaclab_arena/llm_env_gen/generate_catalog.py` as committed imports from `isaaclab_arena.scene_gen.catalog_utils`, but the helper was added at `isaaclab_arena/llm_env_gen/catalog_utils.py` — the import (or the module location) must be fixed before the script runs.
- Uncommitted working-tree edits are needed for the VOMP flow: `isaaclab_arena/assets/registries.py` gets `import isaaclab_arena.assets.vomp_object_library`.

### 2. What was added

- **`isaaclab_arena/llm_env_gen/env_writer.py`**
  - `_plan_task` / `_pick_and_place_plan` / `_no_task_plan` dispatch driven by `resolved.goal_added` / `goal_removed`.
  - Constants: `_PICK_AND_PLACE_KINDS = {"on", "in"}`, `_DEFAULT_PICK_AND_PLACE_EPISODE_LENGTH_S = 20.0`.
  - Generated module docstrings now carry a one-liner `Task wiring: …` so `head -n 20` explains which task got emitted.
  - Generated envs get inline `# goal_added (...)` / `# goal_removed (...)` comments documenting the success contract.
- **`isaaclab_arena/llm_env_gen/catalog_utils.py`** (new): `ARENA_ROOT`, `OBJECT_CATALOG_PATH`, `find_usd_files`, `iter_object_files`, `get_usd_rigid_body_info`, `get_dataset_from_path`, `load_catalog`, `print_object_info`. Pure `pxr`, no Isaac Sim runtime.
- **`isaaclab_arena/llm_env_gen/generate_catalog.py`** (new): CLI with `--objects`, `--output`, `--verbose`, `--list-classes`, `--by-dataset`. Falls back to `AssetRegistry` enumeration when `--objects` is omitted; rewrites absolute paths to repo-relative under `ARENA_ROOT`.
- **`isaaclab_arena_environments/avocadoPnPbowltable.py`**: retargeted from `NoTask` → `PickAndPlaceTask(pick_up_object=avocado_obj, destination_location=bowl_obj, background_scene=background, episode_length_s=20.0)` with task description preserved and the goal diff captured as comments.
- **`.claude/skills/`**
  - Top-level `SKILL.md` index and `isaaclab-arena-assetgen/SKILL.md` workflow.
  - `references/object_library_template.md` — minimal `LibraryObject` + `@register_asset` template.
  - New convention: a single `<basename>` drives the library filename, the catalog JSON path, and a `<basename>_<catalog_name>` prefix on every registered object → registry collisions impossible by construction.
- **`setup.py`**: `openai` added to `RUNTIME_DEPS`, consumed lazily by `isaaclab_arena/llm_env_gen/*`.

### 3. Commands

```
# Generate a dedicated catalog for a USD batch (skill step 1)
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/generate_catalog.py \
    --objects /path/to/usd/root \
    --output isaaclab_arena/scene_gen/catalogs/<basename>_object_catalog.json \
    --verbose

# Rescan every USD currently registered in AssetRegistry
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/generate_catalog.py

# Read back semantic labels from an existing catalog
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/generate_catalog.py \
    --list-classes --by-dataset

# Run the converted avocado pick-and-place env
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_steps 200 --num_envs 1 \
    avocadoPnPbowltable
```

## Xinjie Yao — 2026-04-21

### 1. What it can do

- Parse a natural-language scene prompt into a validated `SceneSpec` (Pydantic).
- Resolve item names to concrete Arena assets with a traceable tag-preference + fallback pipeline.
- Extract both initial and final scene graphs from one prompt and derive the goal diff.
- Auto-generate a registered `@register_environment` Arena module with `NoTask`.
- Bring the generated env up end-to-end with `no_embodiment` + `zero_action`.

**Assumptions:** Docker container running; `NV_API_KEY` set on host; `openai` pip installed inside the container; writer output placed under `isaaclab_arena_environments/` for auto-discovery; the uncommitted `NoEmbodiment` registration fix applied for the no-robot path.

### 2. What was added

- **`isaaclab_arena/llm_env_gen/`**
  - `schema.py` — `SceneSpec` / `Item` / `Relation` Pydantic models; `identity()` for diffing; `goal_added()` / `goal_removed()` helpers; trivial-task `model_validator`.
  - `resolver.py` — `Resolver` with exact → substring → difflib fallback; tag preference (not hard filter); `IK_DEFAULTS` for bare robot family names; structured `TraceEvent` log including `diff.goal_added` / `diff.goal_removed`.
  - `llm_agent.py` — OpenAI-compatible Claude call against NVIDIA inference endpoint; schema injection; robust JSON extraction (fenced / brace-matching fallback).
  - `try_schema.py` — CLI runner with `--print-schema`, `--print-catalog`, `--write-env`; chains resolver and prints the trace.
  - `env_writer.py` — `write_env()` renders a complete env module; compact goal-derived naming (`avocadoPnPbowltable`); filename auto-derived when `out_path` is a directory.
- **`isaaclab_arena/relations/`** — TODO markers on the `In` relation (in `relations.py`) and on the `RelationSolver` negative-constraint requirement.
- **`docker/run_docker.sh`** — passes host `NV_API_KEY` through to the container when set, so `docker exec` no longer needs `-e`.
- **`isaaclab_arena_examples/agents/`** — example artifact directory with the committed avocado env.
- **Uncommitted**: `@register_asset` on `NoEmbodiment` + import in `embodiments/__init__.py`; writer template now emits `IsAnchor` + `set_initial_pose(Pose(...))` for background and ground plane; regenerated env file under `isaaclab_arena_environments/`.

### 3. Commands

```
# Parse a prompt and show the resolved scene + trace
/isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema

# Dump the Pydantic schema (no LLM call)
/isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema --print-schema

# Dump the vocabulary catalog the LLM sees (no LLM call)
/isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema --print-catalog

# Parse + resolve + write an @register_environment module (dir auto-names the file)
/isaac-sim/python.sh -m isaaclab_arena.llm_env_gen.try_schema --write-env isaaclab_arena_environments

# Bring up the generated env with no robot, no task
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_steps 200 --num_envs 1 \
    avocadoPnPbowltable --embodiment no_embodiment

# Same, with the Kit visualizer
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py --viz kit \
    --policy_type zero_action --num_steps 200 --num_envs 1 \
    avocadoPnPbowltable --embodiment no_embodiment
```

### 4. TODOs

- `isaaclab_arena/relations/relations.py:129` — Implement `In` / containment relation (loss over parent's opening footprint, `IsInside` predicate for success checks).
- `isaaclab_arena/relations/relation_solver.py:31` — Support negative / not-holds constraints on initial placement so `ResolvedScene.goal_added` can be enforced.
- `isaaclab_arena/llm_env_gen/env_writer.py` (just above `item_decls`) — Support multiple instances of the same library asset (e.g. two bananas in one scene); re-introduce `instance_name=...` with a unique suffix.
- `isaaclab_arena/llm_env_gen/env_writer.py` (emitted into generated envs when a relation kind isn't supported) — `# TODO({rel['kind']}): no generator support yet for this relation kind.`
