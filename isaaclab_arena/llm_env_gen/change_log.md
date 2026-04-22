# llm_env_gen — change log

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
