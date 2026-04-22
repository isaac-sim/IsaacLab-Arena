# llm_env_gen — change log

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
