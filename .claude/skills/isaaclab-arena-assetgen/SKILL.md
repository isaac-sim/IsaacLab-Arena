---
name: isaaclab-arena-assetgen
description: >
  Register a folder of USD objects in Isaac Lab Arena: run generate_catalog.py,
  emit a new *_object_library.py with LibraryObject subclasses, wire it into
  asset_registry.py, and add a pytest that validates every asset path exists.
license: Apache-2.0
compatibility: >
  Isaac Lab Arena repo. Catalog generation needs pxr (USD); run inside the
  Arena Docker container per AGENTS.md. Optional full-spawn tests need Isaac Sim.
metadata:
  author: arena-contributors
  version: "1.0.0"
---

# Isaac Lab Arena — USD asset batch registration

Use this skill when a user wants to **register many USD objects** from a directory into Arena: discover files, build catalog metadata, generate a dedicated `*_object_library.py`, hook it into `AssetRegistry`, and add a **unit test** that checks each `usd_path` exists.

## Reference files (this skill)

- `references/object_library_template.md` — minimal `LibraryObject` + `@register_asset` pattern and naming rules

## Prerequisites

- Repo: **Isaac Lab Arena** (this workspace).
- **Docker** (recommended): `./docker/run_docker.sh` then commands under `/workspaces/isaaclab_arena` with `/isaac-sim/python.sh` (see `AGENTS.md`).
- USD discovery and catalog JSON use **`pxr`** via `isaaclab_arena/scene_gen/catalog_utils.py` — same interpreter as `generate_catalog.py`.

## Core scripts and modules

| Purpose | Path |
|--------|------|
| Scan USDs, write JSON catalog | `isaaclab_arena/scene_gen/generate_catalog.py` |
| `find_usd_files` / USD helpers | `isaaclab_arena/scene_gen/catalog_utils.py` |
| Object base + `LibraryObject` | `isaaclab_arena/assets/object_library.py`, `isaaclab_arena/assets/object_base.py` |
| `@register_asset` | `isaaclab_arena/assets/register.py` |
| Lazy import side-effect registration | `isaaclab_arena/assets/asset_registry.py` → `ensure_assets_registered()` |

## When invoked — collect from the user

Ask **verbatim** (adapt placeholders):

---

I'll register your USD folder in Isaac Lab Arena. I need:

1. **USD root directory** — absolute path inside the container (e.g. `/workspaces/isaaclab_arena/assets/objects/my_batch`).
2. **Library module basename** — snake_case, e.g. `my_batch` → file `my_batch_object_library.py` and import `isaaclab_arena.assets.my_batch_object_library`.
3. **Catalog JSON path** (optional) — default: `isaaclab_arena/scene_gen/object_catalog.json` **overwrites** the global catalog. Prefer a **dedicated** path for one-off batches, e.g. `isaaclab_arena/scene_gen/catalogs/my_batch_object_catalog.json`, unless the user explicitly wants to refresh the global catalog.
4. **Collision policy** — if a generated `name` already exists in `object_library.py` / registry, stop and ask for a **name prefix** (e.g. `mybatch_`) or renamed USD stems.

---

## Workflow (agent checklist)

### Step 1 — Generate catalog from the folder

Run **inside Docker** (or any env where `pxr` works):

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh isaaclab_arena/scene_gen/generate_catalog.py \
  --objects /path/to/usd/root \
  --output isaaclab_arena/scene_gen/catalogs/<basename>_object_catalog.json \
  --verbose
```

- `--objects` may be a **directory** (recursive `.usd`/`.usda`/`.usdc`/`.usdz`) or a **single file**; see `iter_object_files` / `find_usd_files` in `catalog_utils.py`.
- Omit `--objects` only when re-scanning **all** paths already registered in `AssetRegistry` (not typical for a new batch).

Parse the output JSON: each entry includes at least `name`, `usd_path` (often repo-relative under `isaaclab_arena/...` or `assets/...`), `dims`, `class`, etc. (`get_usd_rigid_body_info` in `catalog_utils.py`).

### Step 2 — Generate `<basename>_object_library.py`

- Place under `isaaclab_arena/assets/<basename>_object_library.py`.
- For **each** catalog entry with a stable `name` and `usd_path`:

  - Define `class <PascalCaseFromName>(LibraryObject):` with:
    - `name = "<snake_case_registry_key>"` — must match catalog `name` and stay unique across **all** `*_object_library.py` files.
    - `tags = ["object"]`
    - `usd_path = "<absolute or ISAAC_NUCLEUS_DIR/...>"` — match existing `object_library.py` style: prefer **absolute** resolved path at codegen time, or a **known env prefix** (`ISAAC_NUCLEUS_DIR`, `ISAACLAB_NUCLEUS_DIR`) if files live under Nucleus.
  - `@register_asset` on each class.
  - Thin `__init__(self, instance_name=..., prim_path=..., initial_pose=..., scale=...)` delegating to `super().__init__(...)`.

- **Python class names**: derive from `name` by snake → PascalCase; strip invalid characters; avoid leading digits (prefix `_` or `Obj` if needed).
- **Duplicate registry keys**: `register_asset` warns and skips if `AssetRegistry().is_registered(cls.name)` — **do not** ship conflicting `name` values.

Read `references/object_library_template.md` in this skill for a minimal copy-paste template.

### Step 3 — Register the module in `asset_registry.py`

In `ensure_assets_registered()`, add an import next to the other library imports:

```python
import isaaclab_arena.assets.<basename>_object_library  # noqa: F401
```

Keep alphabetical or grouped order consistent with the file.

### Step 4 — Unit test (validate each object “exists”)

Add `isaaclab_arena/tests/test_<basename>_object_library_assets.py` (or under `isaaclab_arena/tests/assets/` if preferred by repo convention).

**Lightweight (recommended default):** no SimulationApp — for every class in the new module (or every key from the generated catalog):

1. Resolve `cls.usd_path` (if relative, resolve against repo root / `ARENA_ROOT` from `catalog_utils.ARENA_ROOT`).
2. `assert Path(resolved).is_file()` (or `exists()` for Nucleus URIs — document if paths are not local files).

**Optional heavy test:** spawn one env object in Isaac Sim — only if the user asks; follow `AGENTS.md` inner/outer pytest pattern and Docker.

Example pattern for file existence:

```python
from pathlib import Path

import pytest

from isaaclab_arena.scene_gen.catalog_utils import ARENA_ROOT
from isaaclab_arena.assets import my_batch_object_library as lib


def _resolve_usd(p: str) -> Path:
    path = Path(p)
    if path.is_file():
        return path
    candidate = ARENA_ROOT / p
    if candidate.is_file():
        return candidate
    pytest.fail(f"USD not found: {p}")


@pytest.mark.parametrize(
    "cls_name",
    [n for n in dir(lib) if n[0].isupper() and hasattr(getattr(lib, n), "usd_path")],
)
def test_usd_file_exists(cls_name):
    cls = getattr(lib, cls_name)
    if getattr(cls, "usd_path", None):
        assert _resolve_usd(cls.usd_path).is_file(), cls.usd_path
```

Run tests **in Docker**:

```bash
/isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_<basename>_object_library_assets.py -v
```

### Step 5 — Verify registration

Quick check:

```bash
/isaac-sim/python.sh -c "
from isaaclab_arena.assets.asset_registry import AssetRegistry
r = AssetRegistry()
for k in ('<name1>', '<name2>'):
    print(k, r.is_registered(k))
"
```

## Validation checklist

- [ ] `generate_catalog.py` completed without mass `[SKIP]`; investigate skips.
- [ ] Every `name` in JSON has a matching `@register_asset` class with the same `name`.
- [ ] `usd_path` resolves on disk (or documented Nucleus path) for every entry.
- [ ] `asset_registry.py` imports the new library module.
- [ ] Pytest file existence test passes in Docker.
- [ ] No duplicate `name` keys vs existing `object_library.py` classes.

## Naming and paths

- **Registry `name`**: stable snake_case; used in environments (`get_asset_by_name("cracker_box")`).
- **Catalog `usd_path`**: after `generate_catalog`, may be relative to `ARENA_ROOT`; generated `LibraryObject.usd_path` should match what runtime expects (see existing entries in `object_library.py`).

## Related docs (repo)

- Scene-gen catalog location default: `OBJECT_CATALOG_PATH` in `catalog_utils.py`.
- Full object catalog in repo: `assets/object_catalog.json` (large) — do not paste into chat; regenerate with the script when needed.
