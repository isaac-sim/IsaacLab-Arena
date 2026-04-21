---
name: arena-create
description: Write all Python source files for an external IsaacLab-Arena project (assets, task, environment) from a spec file.
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

The API patterns below are embedded in this skill — **do not read Arena source files** before writing code. The one exception: if you need to verify a built-in background's sub-prim path, read the matching environment file from `<ARENA_SRC>/isaaclab_arena_environments/`.

---

## API Quick Reference

Use these embedded patterns to write all files. Do not re-read Arena source.

### Asset registration

```python
# Background (static, no physics)
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

@register_asset
class MyBackground(LibraryBackground):
    name = "my_background"          # registry key
    tags = ["background"]
    usd_path = "/container/path/bg.usd"
    object_min_z = -0.2             # drop-reset height threshold
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    def __init__(self):
        super().__init__()


# Object (dynamic / rigid — grasped or moved)
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject

@register_asset
class MyObject(LibraryObject):
    name = "my_object"              # registry key
    tags = ["object"]
    usd_path = "/container/path/obj.usd"
    # object_type defaults to ObjectType.RIGID — correct for graspable objects.
    # If the USD has no rigid body physics, set object_type = ObjectType.BASE to avoid
    # "No contact sensors added / no rigid bodies present" errors at sim start.

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None, scale=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path,
                         initial_pose=initial_pose, scale=scale)
```

### Pose

```python
from isaaclab_arena.utils.pose import Pose, PoseRange

# Fixed pose — all rotations are xyzw (x, y, z, w)
Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

# Random pose range — rpy in radians
PoseRange(
    position_xyz_min=(-0.1, -0.1, 0.0),
    position_xyz_max=( 0.1,  0.1, 0.0),
    rpy_min=(0.0, 0.0, -3.14),
    rpy_max=(0.0, 0.0,  3.14),
)
```

### Termination functions

```python
# All from: isaaclab_arena.tasks.terminations
from isaaclab_arena.tasks.terminations import (
    object_on_destination,      # pick-and-place success
    objects_on_destinations,    # multi-object variant
    lift_object_il_success,     # lift to fixed goal position
    objects_in_proximity,       # proximity-based success
    goal_pose_task_termination, # goal pose with optional orientation check
)

# Signatures (key params):
# object_on_destination(env, object_cfg, contact_sensor_cfg, force_threshold=1.0, velocity_threshold=0.5)
# lift_object_il_success(env, object_cfg, goal_position, position_tolerance=0.05)
# objects_in_proximity(env, object_cfg, target_object_cfg, max_y_separation, max_x_separation, max_z_separation)

# Drop / failure (from isaaclab.envs.mdp):
import isaaclab.envs.mdp as mdp_isaac_lab
# mdp_isaac_lab.root_height_below_minimum(env, minimum_height, asset_cfg)
# mdp_isaac_lab.time_out(env)
```

### Metrics

```python
from isaaclab_arena.metrics.success_rate import SuccessRateMetric        # always include
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric    # include when an object is manipulated

# ObjectMovedRateMetric(object: Asset, object_velocity_threshold: float = 0.5)
```

---

## Step 2 — Write `assets/asset_paths.py`

Only needed if any asset in `scene_assets` has `"source": "custom"`.
If all assets are built-in, write a one-line comment module.

One constant per custom asset. Derive `container_usd_path` using the same rule as arena-install Step 1:
find the first `extra_mounts` entry where `host_usd_path == mount["host"]` or
`host_usd_path.startswith(mount["host"] + "/")`, then substitute:
`container_usd_path = mount["container"] + host_usd_path[len(mount["host"]):]`.
If no matching mount is found, warn the user and use `host_usd_path` as a fallback.

---

## Step 3 — Write `assets/background_library.py`

For each `background` asset with `"source": "custom"`, register it with `@register_asset`
using the `LibraryBackground` pattern from the API Quick Reference above.

If no custom backgrounds exist, write a one-line comment module.

Conventions:
- `prim_path` must start with `{ENV_REGEX_NS}/`
- All rotations: **xyzw** `(x, y, z, w)`

---

## Step 4 — Write `assets/object_library.py`

For each `object` asset with `"source": "custom"`, register it with `@register_asset`
using the `LibraryObject` pattern from the API Quick Reference above.

`LibraryObject` defaults to `ObjectType.RIGID` (correct for graspable objects). If an object
is purely static (a surface, board, fixture), set `object_type = ObjectType.BASE` on the class.

If no custom objects exist, write a one-line comment module.

---

## Step 5 — Embodiment (no custom file for standard cases)

No embodiment file is generated for standard cases. The embodiment is selected inline
inside `get_env()` (see Step 7).

Only create `<PKG>/embodiments/<project_name>_embodiment.py` if the spec requires a custom
robot USD not available in the built-in registry. In that case read the built-in embodiment
closest to the desired robot and subclass it.

---

## Step 6 — Write `tasks/<project_name>_task.py`

Always generate this file using the template below. Do not subclass a built-in task type —
subclass `TaskBase` directly so the developer has full visibility of every extension point.
Fill in `<ProjectName>` and the docstring from the spec.

```python
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase


@configclass
class SceneCfg:
    # TODO: add ContactSensorCfg fields for objects that need contact detection, e.g.:
    # object_contact_sensor: ContactSensorCfg = MISSING
    pass


@configclass
class RewardsCfg:
    # TODO: add reward terms, e.g.:
    # reaching_object: RewardTermCfg = RewardTermCfg(func=..., params={...}, weight=1.0)
    pass


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    # TODO: add success and failure termination terms, e.g.:
    # success: TerminationTermCfg = MISSING
    # object_dropped: TerminationTermCfg = MISSING


class <ProjectName>Task(TaskBase):
    """<task description>"""

    def __init__(self, scene_objects: list, background_scene, episode_length_s: float = 20.0, **kwargs):
        super().__init__(episode_length_s=episode_length_s, **kwargs)
        self.scene_objects = scene_objects
        self.background_scene = background_scene
        self.termination_cfg = self._make_termination_cfg()

    def get_scene_cfg(self):
        return SceneCfg()

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_rewards_cfg(self):
        # TODO: implement reward terms
        return RewardsCfg()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        # TODO: add ObjectMovedRateMetric(obj) for each manipulated object
        return [SuccessRateMetric()]

    def _make_termination_cfg(self):
        # TODO: implement success and failure conditions.
        # Available termination functions (see API Quick Reference above):
        #   object_on_destination, lift_object_il_success, objects_in_proximity, ...
        return TerminationsCfg()
```

The `scene_objects` parameter is a placeholder list. Once the developer knows which objects
need termination or reward logic, replace it with named parameters and wire up accordingly.

---

## Step 7 — Write `environments/<project_name>_environment.py`

Use the template below. The output must be runnable immediately with `policy_runner.py`.

### Rules
1. **No top-level sim imports** — the file is parsed before Isaac Sim starts; all Isaac Lab /
   warp / torch imports must be inside `get_env()`.
2. **Side-effect imports first** inside `get_env()` — import project library files before
   `get_asset_by_name()` so `@register_asset` decorators fire.
3. **Embodiment NOT in Scene** — pass it directly to `IsaacLabArenaEnvironment`.

### Template

```python
import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class <ProjectName>Environment(ExampleEnvironmentBase):

    name: str = "<project_name>"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab.sim as sim_utils
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        from <project_name>.tasks.<project_name>_task import <ProjectName>Task

        # Side-effect imports so @register_asset decorators fire (only if custom assets exist)
        # from <project_name>.assets import background_library, object_library

        # Step 1 — Retrieve assets
        background = self.asset_registry.get_asset_by_name("<background_name>")()
        object = self.asset_registry.get_asset_by_name(args_cli.object)()

        # TODO: add spatial relationships (ObjectReference, On, IsAnchor, etc.)

        # Step 2 — Lighting
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=args_cli.light_intensity),
        )
        if args_cli.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())

        # Step 3 — Embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )

        # Step 4 — Scene (do NOT add embodiment here)
        scene = Scene(assets=[background, light, object])

        # Step 5 — Task
        task = <ProjectName>Task(
            scene_objects=[object],
            background_scene=background,
            episode_length_s=20.0,
        )

        # Step 6 — Assemble
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="<spec.robot>")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--hdr", type=str, default=None)
        parser.add_argument("--light_intensity", type=float, default=500.0)
        parser.add_argument("--object", type=str, default="<first_object_asset_name>")
```

Substitute all `<...>` placeholders from the spec:
- `<background_name>`: name of the background asset (built-in or custom)
- `<spec.robot>`: robot name from spec (e.g. `"franka_ik"`)
- `<first_object_asset_name>`: name of the first object-type asset in `scene_assets`

If the spec has multiple objects, add one `self.asset_registry.get_asset_by_name(...)()` call per object and append each to `scene_objects` and `Scene(assets=[...])`. Spatial relationships are left for the developer.

---

## Step 8 — Print the smoke-test command

After all files are written, output the following command block so the user can immediately
test the environment with a zero-action policy. Substitute `<project_name>` and
`<ProjectName>` from the spec.

```
To smoke-test your environment, run:

docker exec <project_name>-latest bash -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
   --policy_type zero_action \
   --num_steps 50 \
   --external_environment_class_path \
     <project_name>.environments.<project_name>_environment:<ProjectName>Environment \
   <project_name>"
```

If the container is not yet running, use `/arena-verify <spec_path>` which starts the
container, installs the package, and runs this test automatically.
