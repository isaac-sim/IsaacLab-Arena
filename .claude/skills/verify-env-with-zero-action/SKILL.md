---
name: verify-env-with-zero-action
description: >
  Smoke-test a registered Arena environment by running the `zero_action` policy
  against it inside the Docker container. Confirms the env name resolves in the
  registry, its relation graph is solvable, and one full rollout completes
  without exceptions. Use when the user says "verify <env>", "bring up <env>",
  "does <env> come up", or after generating a new env via llm_env_gen.
license: Apache-2.0
compatibility: >
  Isaac Lab Arena repo. Runs inside the `isaaclab_arena-latest` Docker container
  per AGENTS.md (needs Isaac Sim). Works for any env registered via
  @register_environment, not just llm_env_gen-generated ones.
metadata:
  author: arena-contributors
  version: "1.0.0"
---

# Verify an Arena environment with the zero-action policy

When the user asks to "verify" or "bring up" a registered env, run `isaaclab_arena/evaluation/policy_runner.py` with `--policy_type zero_action` so the env initializes, the relation solver places objects, and the rollout loop completes without touching any robot actuation. Failures localize cleanly to the env, not the policy.

## Inputs

The user provides at minimum the registered **env name** (e.g. `avocadoPnPbowltable`, `kitchen_pick_and_place`, `lift_object`). Optional arguments, with defaults:

| Argument | Default | Notes |
|----------|---------|-------|
| `env_name` (positional) | — | Must match a `name` on a class decorated with `@register_environment`. Verify with `grep -rn "name:.*$env_name" isaaclab_arena_environments/` if unsure. |
| `--num_steps` | `200` | Use `5` for a fast sanity check, `200+` for something that exercises reset cycles. |
| `--num_envs` | `1` | Bump only if parallel-env wiring is part of the thing you want to verify. |
| `--embodiment` | `no_embodiment` | Falls back to the env's declared default if `no_embodiment` is unsupported by the env's `--embodiment` argparse. |
| `--viz` | `kit` | Opens the Kit viewer by default so the user can eyeball the scene. Requires a working DISPLAY into the container (`docker/run_docker.sh` sets this up). Pass `--viz none` (or the `no-viz` arg) to force headless — use when the user says "just verify", runs remotely, or batches many envs. |
| Extra env-specific CLI args | — | Pass through verbatim. Read the target env's `add_cli_args` if unsure what it accepts (e.g. `--object`, `--objects`). |

## Steps

### 1. Resolve the env name

If the user's string is ambiguous, confirm by listing registered envs:

```
docker exec isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -c \
  'from isaaclab_arena.assets.registries import EnvironmentRegistry; print(sorted(EnvironmentRegistry().get_all_keys()))'"
```

(Cheap — only a Python import, no simulation.)

### 2. Pick the embodiment

- Default to `no_embodiment` so the rollout does not depend on a working robot.
- If the env's `add_cli_args` rejects `no_embodiment` (e.g. it passes `enable_cameras` unconditionally), fall back to the env's declared default (grep `default="..."` inside `add_cli_args`) and note the choice to the user.

### 3. Run the rollout

Default form — Kit visualizer on:

```
docker exec isaaclab_arena-latest bash -c \
  'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --viz kit --policy_type zero_action --num_steps <N> --num_envs <E> \
    <env_name> --embodiment <embodiment> <extra-env-args>'
```

Headless variant — when the user asks for "just verify", runs remotely without a display, or batches many envs:

```
docker exec isaaclab_arena-latest bash -c \
  'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_steps <N> --num_envs <E> \
    <env_name> --embodiment <embodiment> <extra-env-args>'
```

Because `--viz kit` is the default, run in background (`run_in_background=true`) for anything past ~50 steps — the viewer keeps the shell occupied until the Kit window is closed or the step budget is exhausted. A good pattern for `--num_steps 200` at `--num_envs 1` is a 3–4 minute background run that the user watches directly in the Kit window.

### 4. Classify the outcome

Grep the tail of the output for the definitive signals:

- **Success:** a `Steps: 100%|…| N/N` line from the tqdm progress bar.
- **Env build failure:** `AssertionError` or `ValueError` raised inside `get_env`, `compose_manager_cfg`, or `RelationSolver` — report the last few frames verbatim.
- **Registry miss:** `component X not found` from `AssetRegistry.get_component_by_name` — flag missing imports or missing `@register_asset` / `@register_environment` decorator.
- **Simulation crash:** `Exception caught in SimulationAppContext` without a preceding tqdm completion — surface the original traceback, not the wrapper.

## Failure → fix hints (non-exhaustive)

| Symptom | Likely cause | First fix to suggest |
|---------|--------------|----------------------|
| `component <name> not found` | Class declared but not imported from its package `__init__.py` | Add `from .<module> import *` to the relevant `embodiments/__init__.py`, `environments/__init__.py`, or add `@register_asset` / `@register_environment`. |
| `Anchor object 'X' must have an initial_pose set` | An `IsAnchor()`-marked asset has no pose | Call `set_initial_pose(Pose(...))` on that asset, or anchor a child `ObjectReference` whose pose is derived from a real USD prim. |
| Objects falling from too high / off the table | `On(background)` alone is a soft constraint with an imprecise bbox | Switch to an `ObjectReference` to the tabletop prim + `IsAnchor` on the reference; add `AtPosition(z=...)` and `PositionLimits(x/y)` per item — see the hand-edited `avocadoPnPbowltable.py` pattern. |
| `embodiment takes no arguments` | Env passes kwargs (e.g. `enable_cameras=...`) that `NoEmbodiment.__init__` does not accept | Drop the kwargs in the env's embodiment call, or use a robot embodiment for that env. |

## Reporting back

Summarize in at most three sentences:

1. Which env ran, with which embodiment and how many steps.
2. Pass / fail, plus the key line you grepped to decide.
3. If failed, the first fix to try (cite the table above or the specific symptom).

Do not dump the full sim log unless the user asks for it — the interesting frames are the last ~30 lines.
