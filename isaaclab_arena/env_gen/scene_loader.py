"""Load a scene USD into Isaac Lab configs using RoboLab's import_scene pattern.

KEY INSIGHT: when the scene USD already has everything (table + objects + poses),
spawn the ENTIRE USD as a single sublayer, then create RigidObjectCfg/AssetBaseCfg
configs with spawn=None for each object. The objects are "already there" — the
configs just tell Isaac Lab how to track them (physics, contact sensors, etc.).

This is RoboLab's approach. Positions are read from the USD directly. No
AssetRegistry lookup, no position recalculation, no geometry origin mismatch.
"""

from __future__ import annotations

import os
from typing import Any

from isaaclab_arena.task_gen.goal_spec import GoalSpec


SKIP_PRIMS = {
    "PhysicsScene", "PhysicsMaterial", "RenderCam",
    "render_distant_light", "render_dome_light", "light", "Looks",
}


def _find_scene_usd(goal_spec: GoalSpec, scene_usd_dir: str) -> str | None:
    candidates = [
        os.path.join(scene_usd_dir, goal_spec.scene),
        os.path.join(os.path.dirname(__file__), "..", "scene_gen", "tmp", goal_spec.scene),
        os.path.join(os.path.dirname(__file__), "..", "scene_gen", "generated", goal_spec.scene),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def get_usd_objects_info(usd_path: str) -> list[dict[str, Any]]:
    """Read raw transform info from each prim in the USD.

    Returns list of {name, position, rotation, prim_path, is_rigid, is_kinematic}.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD: {usd_path}")

    # Collect children from /world and /World
    all_children = []
    for root_name in ("/World", "/world"):
        prim = stage.GetPrimAtPath(root_name)
        if prim.IsValid():
            all_children.extend(prim.GetChildren())

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    results = []

    for child in all_children:
        name = child.GetName()
        if name in SKIP_PRIMS:
            continue
        if not child.IsA(UsdGeom.Xformable):
            continue

        # Get world transform (composed)
        world_xform = xform_cache.GetLocalToWorldTransform(child)
        pos = world_xform.ExtractTranslation()

        # Extract rotation (strip scale)
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

        # Physics properties. Some assets (e.g. spray_016, pizza_cutter) put
        # PhysicsRigidBodyAPI on a nested sub-Xform rather than the top-level
        # prim. Walk the subtree so we don't misclassify those as "static".
        is_rigid = False
        is_kinematic = False
        for descendant in Usd.PrimRange(child):
            rb = UsdPhysics.RigidBodyAPI(descendant)
            if not (rb and rb.GetRigidBodyEnabledAttr().Get()):
                continue
            is_rigid = True
            kin_attr = descendant.GetAttribute("physics:kinematicEnabled")
            if kin_attr and kin_attr.Get():
                is_kinematic = True
            break  # first rigid body found is enough

        results.append({
            "name": name,
            "prim_path": str(child.GetPath()),
            "position": (float(pos[0]), float(pos[1]), float(pos[2])),
            "rotation": (float(rot.GetReal()),
                         float(rot.GetImaginary()[0]),
                         float(rot.GetImaginary()[1]),
                         float(rot.GetImaginary()[2])),
            "is_rigid": is_rigid,
            "is_kinematic": is_kinematic,
        })

    return results


def scrape_scene_to_cfgs(scene_usd_path: str) -> dict:
    """Parse scene USD and build Isaac Lab asset configs using RoboLab's pattern.

    Each asset has spawn=None — the object is already in the scene (loaded via
    the whole-scene sublayer). The config just registers it for physics/tracking.
    """
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    import isaaclab.sim as sim_utils

    scene_objects = get_usd_objects_info(scene_usd_path)
    scene_dict = {}

    # Full scene USD first: loads table + all object geometry at their exact poses.
    # Field name must be a non-dunder identifier — `configclass` skips `__foo__`
    # class attrs but still counts the annotation, causing a mismatch.
    scene_dict["scene"] = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=sim_utils.UsdFileCfg(usd_path=scene_usd_path),
    )

    for obj in scene_objects:
        name = obj["name"]
        if obj["is_rigid"] and not obj["is_kinematic"]:
            cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                spawn=None,  # Already in the scene USD
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=obj["position"],
                    rot=obj["rotation"],
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                ),
            )
        else:
            cfg = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                spawn=None,
            )
        scene_dict[name] = cfg
        print(f"[SceneLoader]   {name}: pos={obj['position']} "
              f"{'rigid' if obj['is_rigid'] else 'static'}")

    scene_dict["light"] = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )

    print(f"[SceneLoader] Loaded {len(scene_objects)} objects from USD + light")
    return scene_dict


def load_scene_from_goalspec(goal_spec: GoalSpec, scene_usd_dir: str, **kwargs) -> dict:
    scene_usd_path = _find_scene_usd(goal_spec, scene_usd_dir)
    if scene_usd_path is None:
        raise FileNotFoundError(f"Scene USD '{goal_spec.scene}' not found in {scene_usd_dir}")
    print(f"[SceneLoader] Loading scene: {scene_usd_path}")
    return scrape_scene_to_cfgs(scene_usd_path)
