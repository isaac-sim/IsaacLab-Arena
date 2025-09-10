# %%


import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%

from isaac_arena.assets.asset_registry import AssetRegistry

asset_registry = AssetRegistry()
microwave = asset_registry.get_asset_by_name("microwave")()
power_drill = asset_registry.get_asset_by_name("power_drill")()


# %%


def get_all_prims(stage, prim=None, prims_list=None):
    """Get all prims"""
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetAllChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def is_articulation_root(prim):
    """Check if prim is articulation root"""
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_rigid_body(prim):
    """Check if prim is rigidbody"""
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


# %%


from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdSkel

usd_path = microwave.usd_path
print(usd_path)


stage = Usd.Stage.Open(usd_path)
print(stage)

all_prims = get_all_prims(stage)
print(all_prims)


def get_prim_depth(prim):
    return len(str(prim.GetPath()).split("/")) - 2


for prim in all_prims:
    if is_articulation_root(prim):
        print("Articulation root: ", prim.GetPath())
        # print(f"HERE {get_all_prims(stage, prim)}")
        # articulation_sub_prims.append(get_all_prims(stage, prim))
    if is_rigid_body(prim):
        print("Rigidbody: ", prim.GetPath())
        # articulation_sub_prims.append(get_all_prims(stage, prim))

# print(articulation_sub_prims)


# %%


from isaac_arena.assets.object import ObjectType


def is_rigid_body_or_articulation_root(prim):
    return is_articulation_root(prim) or is_rigid_body(prim)


# for prim in all_prims:
#     print(f"{prim.GetPath()} {get_prim_depth(prim)}")


def detect_object_type(usd_path):
    """Detect the object type of the asset"""
    stage = Usd.Stage.Open(usd_path)
    open_prims = [stage.GetPseudoRoot()]
    found = False
    found_depth = -1
    interesting_prim = None
    while len(open_prims) > 0:
        # Update the DFS list
        prim = open_prims.pop(0)
        open_prims.extend(prim.GetChildren())
        # Check if we found an interesting prim on this level
        if is_rigid_body_or_articulation_root(prim):
            if found:
                raise ValueError(f"Found multiple rigid body or articulation roots at depth {get_prim_depth(prim)}")
            else:
                found_depth = get_prim_depth(prim)
                found = True
                interesting_prim = prim
        if found and get_prim_depth(prim) > found_depth:
            break
        # if found and is_rigid_body_or_articulation_root(prim):
        #     raise ValueError(f"Found multiple rigid body or articulation roots at depth {get_prim_depth(prim)}")
    if not found:
        return ObjectType.BASE
    if found and is_rigid_body(interesting_prim):
        return ObjectType.RIGID
    if found and is_articulation_root(interesting_prim):
        return ObjectType.ARTICULATION
    else:
        raise ValueError(f"This should not happen")


print(detect_object_type(usd_path))


# %%

# len("".split("/"))


stage = Usd.Stage.CreateInMemory()

prim2 = stage.DefinePrim("/test2")
assert get_prim_depth(prim2) == 0

prim3 = stage.DefinePrim("/test2/test3")
assert get_prim_depth(prim3) == 1


# %%
