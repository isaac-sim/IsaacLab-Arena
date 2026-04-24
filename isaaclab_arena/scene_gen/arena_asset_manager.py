"""Asset manager that wraps Arena's AssetRegistry for LLM scene generation.

Provides the object catalog interface that the LLM agent expects,
pulling from Arena's 700+ registered objects with dims/tags.
Also manages table selection for scene generation.
"""

from __future__ import annotations

import random
from typing import Any


# Tables available for scene generation, all standardized to ~0.7m x 1.0m surface
# Table bounds are in the table's local frame (before initial_pose is applied)
SCENE_GEN_TABLES = {
    "oak_table_robolab": {
        "registry_name": "oak_table_robolab",
        "source": "background_library",
        "native_size": True,
        "scale": (1.0, 1.0, 1.0),
        "pose": {"position": (0.547, 0.0, -0.35), "rotation": (1.0, 0.0, 0.0, 0.0)},
    },
    "maple_table_robolab": {
        "registry_name": "maple_table_robolab",
        "source": "background_library",
        "native_size": True,
        "scale": (1.0, 1.0, 1.0),
        # Background scene USD — table already positioned internally, no extra transform
        "pose": {"position": (0.0, 0.0, 0.0), "rotation": (1.0, 0.0, 0.0, 0.0)},
    },
    "bamboo_table_robolab": {
        "registry_name": "bamboo_table_robolab",
        "source": "background_library",
        "native_size": True,
        "scale": (1.0, 1.0, 1.0),
        "pose": {"position": (0.547, 0.0, -0.35), "rotation": (1.0, 0.0, 0.0, 0.0)},
    },
    "black_table_robolab": {
        "registry_name": "black_table_robolab",
        "source": "background_library",
        "native_size": True,
        "scale": (1.0, 1.0, 1.0),
        "pose": {"position": (0.547, 0.0, -0.35), "rotation": (1.0, 0.0, 0.0, 0.0)},
    },
    "office_table_background": {
        "registry_name": "office_table_background",
        "source": "background_library",
        "native_size": False,
        # X/Y scale 0.7 kept for width (arena droid reference); Z scale raised from 0.7 → 0.9195
        # so scaled leg height (0.758 × 0.9195 = 0.697) reaches the ground collision at
        # z=-0.697. Translate lowered from -0.531 → -0.697 so the base touches ground and
        # top still sits at z=0 (matching all other tables for consistent object placement).
        # Previously the 0.7 Z-scale left a 0.166 m gap under the legs → visible drop at t=0.
        "scale": (0.7, 1.0, 0.9195),
        "pose": {"position": (0.55, 0.0, -0.697), "rotation": (0.707, 0.0, 0.0, 0.707)},
    },
}

# Standard table surface bounds for the spatial solver (meters)
# These match the 4 RoboLab tables: 0.7m (X) x 1.0m (Y)
# In the base_empty.usda, the table is placed at x=0.547, so objects
# on the table range from roughly x=0.20 to x=0.90
# The solver uses table-relative coordinates.
DEFAULT_TABLE_BOUNDS = (0.20, 0.90, -0.45, 0.45)  # (min_x, max_x, min_y, max_y)
DEFAULT_TABLE_TOP_Z = 0.0  # Objects placed relative to table surface

# Max object footprint for on-table placement (meters).
# Objects exceeding these are auto-scaled down during catalog building.
MAX_OBJECT_WIDTH = 0.32
MAX_OBJECT_DEPTH = 0.40

# Large fixture objects (racks, shelving) — placed as fixed obstacles, not graspable
RACK_OBJECTS = {
    "rack_l04_vomp_robolab",
    "sm_rack_m01_vomp_robolab",
    "bulkstoragerack_a01_vomp_robolab",
    "heavydutysteelshelving_a01_vomp_robolab",
    "wireshelving_a01_vomp_robolab",
}

# Objects excluded from scene generation (too large, non-graspable, or fixtures)
EXCLUDED_OBJECTS = RACK_OBJECTS | {
    "large_storage_rack_vomp_robolab",
    "red_cube",       # Broken texture paths (robotwin relative paths don't resolve)
    "green_cube",     # Same broken texture paths
}



# Max object footprint for on-table placement (meters).
# Objects larger than this are auto-scaled down during catalog building
# so that LLM, spatial solver, and USD export all see the same dims.
MAX_OBJECT_WIDTH = 0.25
MAX_OBJECT_DEPTH = 0.40


class ArenaAssetManager:
    """Manages object selection from Arena's AssetRegistry for scene generation.

    Provides:
    - Object catalog in the format the LLM expects: [{name, dims, size}, ...]
    - Coverage tracking across batch generation (prioritize unused objects)
    - Tag-based filtering for themed scenes
    - Table selection and randomization

    All dims in the catalog are *effective* dims — i.e. they already include
    the registered scale AND any auto-scaling for oversized objects.  The
    corresponding ``effective_scale`` is also stored so that USD export can
    apply it without recomputation.
    """

    def __init__(self, tags: list[str] | None = None, require_dims: bool = True, compute_dims: bool = True):
        """Initialize with objects from AssetRegistry.

        Args:
            tags: Filter objects by these tags. If None, uses all objects
                  with "object" tag.
            require_dims: If True, only include objects that don't have dims
                         and can't be computed.
            compute_dims: If True, compute dims from USD bounding box for objects
                         that don't have stored dims (e.g., Lightwheel objects).
                         Requires pxr/Isaac Sim to be available.
        """
        self._require_dims = require_dims
        self._compute_dims = compute_dims
        # Cache of local mesh min_z per usd_path (scaled), populated during
        # _compute_dims_from_usd. Used by the placer to lift assets whose
        # prim origin sits above the mesh bottom (e.g. bowls).
        self._min_z_cache: dict[str, float] = {}
        self._catalog = self._build_catalog(tags)
        self._regular_objects = [
            obj for obj in self._catalog if obj["name"] not in EXCLUDED_OBJECTS
        ]
        self._rack_objects = [
            obj for obj in self._catalog if obj["name"] in RACK_OBJECTS
        ]

        # Coverage tracking for batch generation
        self._used: set[str] = set()
        self._unused: list[str] = [obj["name"] for obj in self._regular_objects]
        random.shuffle(self._unused)

        print(f"[ArenaAssetManager] Total catalog: {len(self._catalog)}")
        print(f"[ArenaAssetManager] Regular objects: {len(self._regular_objects)}")
        print(f"[ArenaAssetManager] Rack objects: {len(self._rack_objects)}")

    def _compute_dims_from_usd(self, usd_path: str, scale: tuple = (1.0, 1.0, 1.0)) -> list[float] | None:
        """Compute dims from USD bounding box for objects without stored dims.

        Args:
            usd_path: Path to the USD file.
            scale: Scale tuple applied to the asset.

        Returns:
            [width, depth, height] or None if computation fails.
        """
        try:
            from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd
            bbox = compute_local_bounding_box_from_usd(usd_path, scale)
            dims = [
                bbox.max_point[0] - bbox.min_point[0],
                bbox.max_point[1] - bbox.min_point[1],
                bbox.max_point[2] - bbox.min_point[2],
            ]
            # Cache scaled min_z so the placer can compensate for assets whose
            # origin sits above the mesh bottom (bowls with curved bases).
            self._min_z_cache[usd_path] = float(bbox.min_point[2])
            return dims
        except Exception as e:
            print(f"[ArenaAssetManager] Failed to compute dims for {usd_path}: {e}")
            return None

    def _build_catalog(self, tags: list[str] | None) -> list[dict]:
        """Build object catalog from AssetRegistry.

        For objects with stored dims (robolab), uses those directly.
        For objects without dims (lightwheel, arena), computes from USD bounding box.
        """
        from isaaclab_arena.assets.asset_registry import AssetRegistry

        registry = AssetRegistry()
        all_keys = registry.get_all_keys()

        catalog = []
        computed_count = 0
        failed_count = 0

        for name in all_keys:
            cls = registry.get_asset_by_name(name)
            obj_tags = getattr(cls, "tags", [])
            dims = getattr(cls, "dims", None)
            usd_path = getattr(cls, "usd_path", None)
            scale = getattr(cls, "scale", (1.0, 1.0, 1.0))

            # If object has non-default scale, recompute dims from USD
            # (hardcoded dims may be stale / unscaled)
            if scale != (1.0, 1.0, 1.0) and dims is not None:
                dims = None  # Force recomputation with correct scale
            obj_type = getattr(cls, "object_type", None)

            if "object" not in obj_tags:
                continue
            # Skip spawner-type objects (ground_plane, light)
            from isaaclab_arena.assets.object_base import ObjectType
            if obj_type == ObjectType.SPAWNER:
                continue

            # Apply optional tag filter
            if tags is not None:
                if not any(t in obj_tags for t in tags):
                    continue

            # Compute dims from USD if not stored
            if dims is None and usd_path is not None and self._compute_dims:
                dims_computed = self._compute_dims_from_usd(usd_path, scale)
                if dims_computed is not None:
                    dims = tuple(dims_computed)
                    computed_count += 1
                else:
                    failed_count += 1

            # Skip objects without dims if required
            if self._require_dims and dims is None:
                continue

            # Auto-scale oversized objects: correct dims AND compute
            # effective_scale (registered_scale × auto_factor) for USD export.
            # Microwaves use larger limits to stay realistically sized.
            effective_scale = scale
            # min_z already scaled by `scale` (from compute_local_bounding_box_from_usd).
            # If auto_factor kicks in below, it's scaled again to match effective_scale.
            local_min_z = self._min_z_cache.get(usd_path, 0.0)
            if dims is not None:
                w, d = dims[0], dims[1]
                is_microwave = "microwave" in name
                max_w = MAX_OBJECT_WIDTH * 1.2 if is_microwave else MAX_OBJECT_WIDTH
                max_d = MAX_OBJECT_DEPTH * 1.2 if is_microwave else MAX_OBJECT_DEPTH
                auto_sx = max_w / w if w > max_w else 1.0
                auto_sy = max_d / d if d > max_d else 1.0
                auto_factor = min(auto_sx, auto_sy, 1.0)
                if auto_factor < 1.0:
                    dims = tuple(dim * auto_factor for dim in dims)
                    effective_scale = tuple(s * auto_factor for s in scale)
                    local_min_z *= auto_factor

            description = ""
            if cls.__doc__:
                description = cls.__doc__.strip().split("\n")[0]

            # Detect affordances for articulated objects
            affordances = []
            if "openable" in obj_tags:
                affordances.append("openable")
            if "pressable" in obj_tags:
                affordances.append("pressable")
            if "turnable" in obj_tags:
                affordances.append("turnable")

            is_articulated = obj_type == ObjectType.ARTICULATION

            catalog.append({
                "name": name,
                "dims": list(dims) if dims else None,
                "min_z": local_min_z,
                "effective_scale": effective_scale,
                "tags": obj_tags,
                "description": description,
                "object_type": obj_type.value if obj_type else "rigid",
                "affordances": affordances,
                "is_articulated": is_articulated,
                "usd_path": usd_path,
            })

        if computed_count > 0 or failed_count > 0:
            print(f"[ArenaAssetManager] Dims computed from USD: {computed_count}, failed: {failed_count}")

        return catalog

    @property
    def catalog(self) -> list[dict]:
        return self._catalog

    @property
    def regular_objects(self) -> list[dict]:
        return self._regular_objects

    def get_catalog_for_llm(self, object_names: list[str] | None = None) -> list[dict]:
        """Get catalog in the simplified format the LLM prompt expects.

        Args:
            object_names: If provided, only include these objects.

        Returns:
            List of {name, size} dicts for the LLM prompt.
        """
        if object_names is not None:
            name_set = set(object_names)
            source = [obj for obj in self._catalog if obj["name"] in name_set]
        else:
            source = self._regular_objects

        result = []
        for obj in source:
            if obj["dims"]:
                result.append({
                    "name": obj["name"],
                    "size": f"{obj['dims'][0]:.2f}m × {obj['dims'][1]:.2f}m",
                })
            else:
                result.append({"name": obj["name"]})
        return result

    def is_articulated(self, name: str) -> bool:
        """Check if an object is articulated (has joints/affordances). Handles _N suffixes."""
        resolved = self._resolve_name(name)
        for obj in self._catalog:
            if obj["name"] == resolved:
                return obj.get("is_articulated", False)
        return False

    def get_affordances(self, name: str) -> list[str]:
        """Get affordance list for an object (openable, pressable, turnable). Handles _N suffixes."""
        resolved = self._resolve_name(name)
        for obj in self._catalog:
            if obj["name"] == resolved:
                return obj.get("affordances", [])
        return []

    def needs_fixed_orientation(self, name: str) -> bool:
        """Check if object needs to face the robot (articulated with affordances).

        Openable/pressable/turnable objects should face the robot arm so
        the interaction surface (door, button, knob) is reachable.
        """
        return self.is_articulated(name) and len(self.get_affordances(name)) > 0

    def get_objects_for_scene(
        self, max_objects: int, category: str | None = None
    ) -> list[str]:
        """Select objects for a scene, prioritizing unused assets for coverage.

        Args:
            max_objects: Maximum number of objects to select.
            category: Optional category hint (unused for now, reserved for themed filtering).

        Returns:
            List of object names.
        """
        if max_objects <= 5:
            num_objects = random.randint(3, max_objects)
        else:
            num_objects = random.randint(max_objects - 2, max_objects)

        selected = []

        # Prioritize unused assets
        available_unused = [n for n in self._unused if n not in selected]
        num_from_unused = min(num_objects, len(available_unused))
        if num_from_unused > 0:
            selected.extend(available_unused[:num_from_unused])
            for name in selected:
                if name in self._unused:
                    self._unused.remove(name)

        # Fill remaining from all regular objects
        if len(selected) < num_objects:
            remaining = num_objects - len(selected)
            available_all = [
                obj["name"] for obj in self._regular_objects
                if obj["name"] not in set(selected)
            ]
            if available_all:
                selected.extend(
                    random.sample(available_all, min(remaining, len(available_all)))
                )

        self._used.update(selected)
        return selected

    def _resolve_name(self, name: str) -> str:
        """Strip multi-instance suffix (_2, _3, …) to find the base catalog entry.

        The LLM can request e.g. ``bowl_005_2`` which is a second instance of
        ``bowl_005``.  The catalog only stores the base name.
        """
        # Fast path: name exists as-is
        if any(obj["name"] == name for obj in self._catalog):
            return name
        # Try stripping trailing _N
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return name

    def get_object_dims(self, name: str) -> tuple[float, float, float] | None:
        """Get (width, depth, height) for a named object (handles _N suffixes)."""
        resolved = self._resolve_name(name)
        for obj in self._catalog:
            if obj["name"] == resolved and obj["dims"]:
                return tuple(obj["dims"])
        return None

    def get_object_scale(self, name: str) -> tuple[float, float, float]:
        """Get effective scale (registered_scale × auto_factor) for USD export (handles _N suffixes)."""
        resolved = self._resolve_name(name)
        for obj in self._catalog:
            if obj["name"] == resolved:
                return tuple(obj.get("effective_scale", (1.0, 1.0, 1.0)))
        return (1.0, 1.0, 1.0)

    def get_object_min_z(self, name: str) -> float:
        """Local-frame min Z of the asset's mesh (already scaled).

        0.0 for origin-at-base assets (most of the catalog). Negative for
        assets whose prim origin sits above the mesh bottom (bowls with
        curved bases, cups). The placer subtracts this so the object's
        actual bottom (not its origin) lands at `clearance` above the table.
        """
        resolved = self._resolve_name(name)
        for obj in self._catalog:
            if obj["name"] == resolved:
                return float(obj.get("min_z", 0.0))
        return 0.0

    def get_all_object_dims(self, names: list[str]) -> dict[str, tuple[float, float, float]]:
        """Get dims for multiple objects. Skips objects without dims."""
        dims_map = {}
        for name in names:
            d = self.get_object_dims(name)
            if d is not None:
                dims_map[name] = d
        return dims_map

    # --- Table management ---

    def get_random_table(self) -> str:
        """Pick a random table from the 5 available tables."""
        return random.choice(list(SCENE_GEN_TABLES.keys()))

    def get_table_info(self, table_name: str) -> dict:
        """Get table configuration info."""
        if table_name not in SCENE_GEN_TABLES:
            raise ValueError(f"Unknown table: {table_name}. Available: {list(SCENE_GEN_TABLES.keys())}")
        return SCENE_GEN_TABLES[table_name]

    def get_table_bounds(self, table_name: str | None = None) -> tuple[float, float, float, float]:
        """Get solver bounds for a table. All tables standardized to same bounds."""
        return DEFAULT_TABLE_BOUNDS

    def get_table_top_z(self, table_name: str | None = None) -> float:
        """Get the Z height of the table surface (in table-relative coords)."""
        return DEFAULT_TABLE_TOP_Z

    # --- Rack management ---

    def get_rack_object(self) -> str | None:
        """Select a random rack object for fixture placement."""
        if not self._rack_objects:
            return None
        return random.choice(self._rack_objects)["name"]

    # --- Coverage stats ---

    def get_coverage_stats(self) -> dict:
        total = len(self._regular_objects)
        used = len(self._used)
        return {
            "total_regular": total,
            "used": used,
            "unused": len(self._unused),
            "coverage_percent": (used / total * 100) if total > 0 else 0,
        }

    def print_coverage_report(self):
        stats = self.get_coverage_stats()
        print(f"Total regular assets: {stats['total_regular']}")
        print(f"Assets used: {stats['used']}")
        print(f"Assets unused: {stats['unused']}")
        print(f"Coverage: {stats['coverage_percent']:.1f}%")
