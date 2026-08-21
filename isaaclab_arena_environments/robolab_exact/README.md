# RoboLab exact-pose generated data

This directory contains generated Arena graph and task YAML copied from
`isaaclab_arena_environments/robolab`. Each scene is sourced from the benchmark
USDA at `RoboLab/assets/scenes/<scene-basename>.usda`.

`SOURCE_MANIFEST.yaml` records every Arena object ID to benchmark prim mapping,
the benchmark `/world/table` (or `/World/table`) transform, and the matching
local table-scene wrapper used for that scene. Names not listed as renames still
appear explicitly as same-name mappings.

Each scene uses the local `oak_table.usda` or `maple_table.usda` wrapper matching
its source fixture. The wrappers contain the table transform and invisible
ground collision plane, so scene YAML backgrounds have no `initial_pose` offset.
Object poses are reframed from the source table to the wrapper table:

`T_arena_obj = T_benchmark_obj * inverse(T_benchmark_table) * T_wrapper_table`

Scene `initial_pose.rotation_xyzw` values are serialized in xyzw order.
Exact-pose scene files contain no spatial relations, so relation solving cannot
alter or reject the source poses.

Regenerate and source-check the YAMLs:

```bash
/isaac-sim/python.sh \
  isaaclab_arena_environments/robolab_exact/scripts/check_robolab_exact_pose.py \
  --update --skip-runtime
```

Then verify every spawned runtime pose in a source-mounted container:

```bash
/isaac-sim/python.sh \
  isaaclab_arena_environments/robolab_exact/scripts/check_robolab_exact_pose.py \
  --viz none
```

## Capture videos

Run all 38 tasks in one SimulationApp, with nine environments and ten frames
per clip:

```bash
/isaac-sim/python.sh \
  isaaclab_arena_environments/robolab_exact/scripts/capture_task_videos.py \
  --headless --output-dir output/robolab_exact_capture
```

Use one or more `--task <task-stem>` options for a subset. Each task directory
contains one `viewport.mp4` and one `env<N>_<camera>.mp4` per environment and
robot camera.

Extract every middle frame:

```bash
/isaac-sim/python.sh \
  isaaclab_arena_environments/robolab_exact/scripts/extract_middle_frames.py \
  output/robolab_exact_capture output/robolab_exact_middle_frames
```

Camera PNGs use `<task>_env<N>_<camera>.png`; viewport PNGs use
`<task>_viewport.png`.
