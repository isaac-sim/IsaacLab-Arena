#!/usr/bin/env python3
"""Print the tabletop reference world bounding box to determine AtPosition coordinates.

Run:
  /isaac-sim/python.sh isaaclab_arena/tests/inspect_table_center.py
"""

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.relations.relations import IsAnchor
from isaaclab_arena.utils.pose import Pose

registry = AssetRegistry()

table_background = registry.get_asset_by_name("office_table")()
tabletop_reference = ObjectReference(
    name="table",
    prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
    parent_asset=table_background,
)
tabletop_reference.add_relation(IsAnchor())
tabletop_reference.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))

bbox = tabletop_reference.get_world_bounding_box()
print(f"Table world bbox min: {bbox.min_point[0].tolist()}")
print(f"Table world bbox max: {bbox.max_point[0].tolist()}")
print(f"Table center:         {bbox.center[0].tolist()}")
print(f"Table size:           {bbox.size[0].tolist()}")
print(f"Table top Z:          {bbox.top_surface_z[0].item()}")

# Also print container and beer bottle bboxes for reference
container_cls = registry.get_asset_by_name("red_container")
container = container_cls(scale=(0.5, 0.5, 1.0))
cb = container.get_bounding_box()
print(f"\nContainer (0.5,0.5,1.0) bbox size: {cb.size[0].tolist()}")
print(f"Container bbox center: {cb.center[0].tolist()}")

beer_cls = registry.get_asset_by_name("beer_bottle")
beer = beer_cls()
bb = beer.get_bounding_box()
print(f"\nBeer bottle bbox size: {bb.size[0].tolist()}")
print(f"Beer bottle bbox center: {bb.center[0].tolist()}")

app.close()
