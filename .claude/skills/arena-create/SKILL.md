---
name: arena-create
description: Write all Python source files for an external IsaacLab-Arena project (assets, embodiment, task, environment) from a spec file.
---

## How to invoke

```
/arena-create /path/to/spec.json
```

The argument is the path to a filled-in spec file. Start from the template at
`.claude/skills/arena-spec-template.json` in the IsaacLab-Arena repo.

Run `/arena-install` first if the project scaffold does not yet exist.

---

## Step 1 — Read the spec and locate source references

Read the JSON file at the path given in the argument. Extract all fields.

Derived variables:
- `PROJECT_DIR` = `<base_dir>/<project_name>`
- `PKG` = `<PROJECT_DIR>/<project_name>`
- `ARENA_SRC` = the IsaacLab-Arena repo root (find it: `git rev-parse --show-toplevel` from any known Arena path, or look for `isaaclab_arena/` alongside the spec file's `submodules/IsaacLab-Arena/`)

**Before writing any code, read the Arena source** to get the exact current API.
Do not guess or rely on memory — read the files:

| What you need | Where to read |
|---|---|
| How to register assets | `<ARENA_SRC>/isaaclab_arena/assets/register.py` |
| Embodiment base class API | `<ARENA_SRC>/isaaclab_arena/embodiments/embodiment_base.py` |
| Built-in embodiment for the chosen robot | `<ARENA_SRC>/isaaclab_arena/embodiments/<robot>/` |
| Task base class | `<ARENA_SRC>/isaaclab_arena/tasks/task_base.py` |
| Scene class | `<ARENA_SRC>/isaaclab_arena/scene/scene.py` |
| Environment base | `<ARENA_SRC>/isaaclab_arena_environments/example_environment_base.py` |
| A complete working example | `<ARENA_SRC>/isaaclab_arena_environments/kitchen_pick_and_place/` |
| Pose utility | `<ARENA_SRC>/isaaclab_arena/utils/pose.py` |

Use the kitchen_pick_and_place example as the primary reference for structure and import patterns.

---

## Step 2 — Write `assets/asset_paths.py`

Only needed if any asset in `scene_assets` has `"source": "custom"`.
If all assets are built-in, write a one-line comment module.

One constant per custom asset, using the `container_usd_path` from the spec.

---

## Step 3 — Write `assets/background_library.py`

For each `background` asset with `"source": "custom"`, register it with `@register_asset`.
Use `AssetBaseCfg` (static, non-dynamic).

If no custom backgrounds exist, write a one-line comment module.

Read `<ARENA_SRC>/isaaclab_arena/assets/register.py` and an existing library file
(e.g. in `kitchen_pick_and_place`) for the exact `@register_asset` decorator signature and
`get_scene_cfg()` return type.

Conventions:
- `prim_path` must start with `{ENV_REGEX_NS}/`
- All rotations: **xyzw** `(x, y, z, w)`

---

## Step 4 — Write `assets/object_library.py`

For each `object` asset with `"source": "custom"`, register it with `@register_asset`.

- **Static objects** (surfaces, boards — not grasped): use `AssetBaseCfg`
- **Dynamic objects** (grasped, moved): use `RigidObjectCfg` with `activate_contact_sensors=True`

If no custom objects exist, write a one-line comment module.

---

## Step 5 — Write `embodiments/<project_name>_embodiment.py`

Look up the built-in embodiment class for the robot in the spec:

```bash
ls <ARENA_SRC>/isaaclab_arena/embodiments/
```

Read the matching embodiment file to find the exact class name and `__init__` signature.

Subclass it with `@register_asset`:

```python
@register_asset
class <ProjectName>Embodiment(<BuiltInClass>):
    name = "<project_name>_embodiment"

    def __init__(self, enable_cameras=False, initial_pose=None, **kwargs):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose, **kwargs)
```

**No-gripper robots** (`gr1_pink`, `g1_wbc_pink`, `kuka_allegro`, custom robots without gripper):
also register a keyboard retargeter in the same file. Read
`<ARENA_SRC>/isaaclab_arena/assets/register.py` for the `@register_retargeter` signature.

---

## Step 6 — Write `tasks/<project_name>_task.py`

Read `<ARENA_SRC>/isaaclab_arena/tasks/task_base.py` for the `TaskBase` interface.

Derive the success function from `success_condition` in the spec:
- EEF-to-object distance → check `ee_frame` vs `target` position
- Height above surface → check `object.pos.z - surface.pos.z > threshold`
- Object in region → check axis-aligned bounds

Always call `wp.to_torch()` on any `.data` field before indexing or arithmetic.
Read `<ARENA_SRC>/isaaclab_arena/tasks/` for existing success function examples.

If `mimicgen == true`:
- Read `<ARENA_SRC>/isaaclab_arena/` for `MimicEnvCfg`, `SubTaskConfig` usage
- Do **not** add `generation_relative` or `generation_joint_pos` — these fields no longer exist

---

## Step 7 — Write `environments/<project_name>_environment.py`

Read `<ARENA_SRC>/isaaclab_arena_environments/example_environment_base.py` for the
`ExampleEnvironmentBase` interface (`get_env`, `add_cli_args`, `asset_registry`, `device_registry`).

Read a complete working environment (e.g. `kitchen_pick_and_place`) to understand the exact
import pattern and assembly order.

Rules that must be followed:
1. **No top-level sim imports** — file is parsed before Isaac Sim starts; all Isaac Lab / warp / torch
   imports must be inside `get_env()`.
2. **Side-effect imports first** inside `get_env()` — library files must be imported before
   `get_asset_by_name()` so `@register_asset` decorators fire.
3. **Embodiment NOT in Scene** — pass it to `IsaacLabArenaEnvironment` directly.
4. **No-gripper teleop** — for robots without a gripper, use an inline `_KeyboardNoGripper`
   class with `gripper_term=False` in `Se3KeyboardCfg`; do not use the device registry.
