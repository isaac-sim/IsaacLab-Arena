"""Parse scene USD files to extract object info for task generation.

Reads a .usda scene file and returns the list of objects with their
names, positions, dimensions, and types. This is the input to the
LLM task generation prompt.

Adapted from RoboLab's get_usd_objects_info().
"""

from __future__ import annotations

from typing import Any


# Prims to skip (base scene infrastructure, not manipulable objects)
SKIP_PRIMS = {
    "table", "franka_table", "GroundPlane", "ground_plane",
    "PhysicsScene", "RenderCam", "render_distant_light",
    "render_dome_light", "light", "Looks",
}


def parse_scene(usd_path: str, asset_manager=None) -> list[dict[str, Any]]:
    """Extract object info from a scene USD file.

    Args:
        usd_path: Path to the .usda scene file.
        asset_manager: Optional ArenaAssetManager for dims lookup.
            If None, dims are computed from USD bounding boxes.

    Returns:
        List of object dicts with keys:
            name, position, rotation, dims, is_articulated, prim_path
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Failed to open stage: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        for root_name in ("/World", "/world"):
            prim = stage.GetPrimAtPath(root_name)
            if prim.IsValid():
                default_prim = prim
                break
    if not default_prim:
        raise ValueError(f"No default/root prim found in: {usd_path}")

    scene_objects = []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # Collect children from both /world and /World (objects may be under either)
    children = list(default_prim.GetChildren())
    for root_name in ("/World", "/world"):
        prim = stage.GetPrimAtPath(root_name)
        if prim.IsValid() and prim != default_prim:
            children.extend(prim.GetChildren())

    for child in children:
        name = child.GetName()

        if name in SKIP_PRIMS:
            continue
        if not child.IsA(UsdGeom.Xformable):
            continue

        # Extract position
        world_xform = xform_cache.GetLocalToWorldTransform(child)
        pos = world_xform.ExtractTranslation()

        # Extract rotation (strip scale from matrix)
        rot_mat = world_xform.ExtractRotationMatrix()
        col0 = Gf.Vec3d(rot_mat[0][0], rot_mat[1][0], rot_mat[2][0])
        col1 = Gf.Vec3d(rot_mat[0][1], rot_mat[1][1], rot_mat[2][1])
        col2 = Gf.Vec3d(rot_mat[0][2], rot_mat[1][2], rot_mat[2][2])
        sx, sy, sz = col0.GetLength(), col1.GetLength(), col2.GetLength()
        if sx > 1e-9 and sy > 1e-9 and sz > 1e-9:
            norm_mat = Gf.Matrix3d(
                col0[0]/sx, col1[0]/sy, col2[0]/sz,
                col0[1]/sx, col1[1]/sy, col2[1]/sz,
                col0[2]/sx, col1[2]/sy, col2[2]/sz,
            )
            rot = norm_mat.ExtractRotation().GetQuat()
        else:
            rot = Gf.Quatd(1, 0, 0, 0)
        rot_wxyz = (rot.GetReal(), rot.GetImaginary()[0],
                    rot.GetImaginary()[1], rot.GetImaginary()[2])

        # Check if rigid body
        rigid_api = UsdPhysics.RigidBodyAPI(child)
        is_rigid = bool(rigid_api and rigid_api.GetRigidBodyEnabledAttr().Get())

        # Get dims from asset manager or compute from bbox
        dims = None
        if asset_manager is not None:
            dims = asset_manager.get_object_dims(name)
        if dims is None:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_],
            )
            bbox = bbox_cache.ComputeWorldBound(child).ComputeAlignedBox()
            min_pt, max_pt = bbox.GetMin(), bbox.GetMax()
            dims = (max_pt[0] - min_pt[0], max_pt[1] - min_pt[1], max_pt[2] - min_pt[2])

        # Check articulated
        is_articulated = False
        if asset_manager is not None:
            is_articulated = asset_manager.is_articulated(name)

        scene_objects.append({
            "name": name,
            "position": (pos[0], pos[1], pos[2]),
            "rotation": rot_wxyz,
            "dims": dims,
            "is_rigid": is_rigid,
            "is_articulated": is_articulated,
            "prim_path": str(child.GetPath()),
        })

    return scene_objects


def get_scene_summary(scene_objects: list[dict]) -> str:
    """Format scene objects as readable summary."""
    lines = [f"Scene: {len(scene_objects)} objects"]
    for obj in scene_objects:
        d = obj["dims"]
        dims_str = f"{d[0]:.3f} x {d[1]:.3f} x {d[2]:.3f}" if d else "unknown"
        tag = " [A]" if obj["is_articulated"] else ""
        lines.append(
            f"  {obj['name']:40s} pos=({obj['position'][0]:.2f}, "
            f"{obj['position'][1]:.2f}, {obj['position'][2]:.2f})  "
            f"dims={dims_str}{tag}"
        )
    return "\n".join(lines)
