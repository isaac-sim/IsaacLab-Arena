# `*_object_library.py` template (Isaac Lab Arena)

Follow `isaaclab_arena/assets/object_library.py`: subclass **`LibraryObject`**, set class attributes, decorate with **`@register_asset`**.

## Minimal class

```python
# Copyright (c) 2026, The Isaac Lab Arena Project Developers ...
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.object import Object  # if needed for TYPE_CHECKING only
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class MyCustomPart(LibraryObject):
    """One-line description from USD metadata / catalog."""

    name = "my_custom_part"  # registry key; snake_case; unique globally
    tags = ["object"]
    usd_path = "/absolute/or/nucleus/path/to/my_custom_part.usd"
    object_type = ObjectType.RIGID  # default; omit if RIGID
    scale = (1.0, 1.0, 1.0)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)
```

## Naming rules

| Catalog field | Registry / `name` | Python class |
|---------------|-------------------|--------------|
| `my_part` | `name = "my_part"` | `MyPart` |
| `003_cracker_box` | Prefer stable snake: `cracker_box` if renaming | `CrackerBox` |

- Class name must be a valid Python identifier; **no leading digit** — prefix e.g. `Obj003Foo` if needed.
- **`name` must be unique** across all modules imported by `ensure_assets_registered()`.

## `usd_path` conventions

- **Local / repo**: absolute path at codegen time, or path relative to repo root that `Path(ARENA_ROOT / rel)` resolves.
- **Isaac Nucleus**: use `f"{ISAAC_NUCLEUS_DIR}/Props/..."` like existing YCB entries in `object_library.py`.

## After editing

1. Add `import isaaclab_arena.assets.<module>  # noqa: F401` to `asset_registry.py`.
2. Run pytest for path existence (see main `SKILL.md`).
3. Run `pre-commit run --files` on touched files before commit.
