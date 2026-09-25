# RoboLab exact environments

This catalog mirrors every scene and task in `isaaclab_arena_environments/robolab`.
Unlike the regular catalog, each background loads the complete settled RoboLab
scene USDA and exposes its task objects as rigid `ObjectReference` entries.
Placement relations are intentionally omitted so resets restore the authored
scene poses.

Regenerate or validate the checked-in YAML files from the repository root:

```bash
python isaaclab_arena_environments/robolab_exact/scripts/generate_exact_catalog.py --update
python isaaclab_arena_environments/robolab_exact/scripts/generate_exact_catalog.py
```

`SOURCE_MANIFEST.yaml` maps Arena object ids to prim paths in the matching
`RoboLab/assets/scenes/<scene>.usda`. Runtime backgrounds load those scenes from
the mirrored Arena Nucleus directory.
