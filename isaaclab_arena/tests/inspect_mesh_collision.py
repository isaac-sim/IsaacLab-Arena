#!/usr/bin/env python3
"""Diagnostic: use the ACTUAL demo positions from the log to test mesh collision."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import trimesh
import trimesh.collision
import trimesh.transformations

from isaaclab_arena.assets.asset_registry import AssetRegistry

registry = AssetRegistry()

container = registry.get_asset_by_name("red_container")(scale=(0.5, 0.5, 1.0))
can = registry.get_asset_by_name("tomato_soup_can")()

container_mesh = container.get_collision_mesh()
can_mesh = can.get_collision_mesh()

# Use the EXACT positions from the demo log output
container_pos = (0.0, 0.0, 0.5406449437141418)
can_pos_ok = (0.05, 0.0, 0.5745025277137756)
can_pos_fail = (-0.05, 0.0, 0.5745025277137756)

print("=== Using actual demo positions ===")
print(f"  container pos: {container_pos}")
print(f"  can1 (0.05):   {can_pos_ok}")
print(f"  can2 (-0.05):  {can_pos_fail}")

# Container mesh Z range in world
c_world_min_z = container_mesh.bounds[0][2] + container_pos[2]
c_world_max_z = container_mesh.bounds[1][2] + container_pos[2]
print(f"\n  Container mesh world Z: [{c_world_min_z:.6f}, {c_world_max_z:.6f}]")

can_world_min_z = can_mesh.bounds[0][2] + can_pos_ok[2]
can_world_max_z = can_mesh.bounds[1][2] + can_pos_ok[2]
print(f"  Can mesh world Z:       [{can_world_min_z:.6f}, {can_world_max_z:.6f}]")
print(f"  Can bottom vs container bottom: {can_world_min_z:.6f} vs {c_world_min_z:.6f}")
print(f"  Gap: {can_world_min_z - c_world_min_z:.6f} m")

# Test collision at both positions
for label, can_p in [("can at ( 0.05, 0)", can_pos_ok), ("can at (-0.05, 0)", can_pos_fail)]:
    mgr = trimesh.collision.CollisionManager()
    mgr.add_object("container", container_mesh, trimesh.transformations.translation_matrix(container_pos))
    mgr.add_object("can", can_mesh, trimesh.transformations.translation_matrix(can_p))
    collides, pairs = mgr.in_collision_internal(return_names=True)
    print(f"  {label}: collision={collides}")

# Now sweep clearance values to find what works
print("\n=== Sweep clearance values ===")
for extra_mm in [0, 1, 2, 3, 5, 10, 15, 20]:
    extra = extra_mm / 1000.0
    can_z_test = can_pos_ok[2] + extra
    for label, cx in [("x=0.05", 0.05), ("x=-0.05", -0.05), ("x=0.0", 0.0)]:
        mgr = trimesh.collision.CollisionManager()
        mgr.add_object("container", container_mesh, trimesh.transformations.translation_matrix(container_pos))
        mgr.add_object("can", can_mesh, trimesh.transformations.translation_matrix((cx, 0, can_z_test)))
        collides, _ = mgr.in_collision_internal(return_names=True)
        status = "COLLISION" if collides else "ok"
        print(f"  +{extra_mm:2d}mm {label}: {status}")

app.close()
