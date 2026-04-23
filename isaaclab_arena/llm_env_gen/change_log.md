# llm_env_gen — change log

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
