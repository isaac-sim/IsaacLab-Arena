# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()


#%%

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.dummy_task import DummyTask
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

# kitchen = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
embodiment = asset_registry.get_asset_by_name("franka")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
microwave = asset_registry.get_asset_by_name("microwave")()


from isaaclab_arena.assets.background import Background

class ObjectReferenceTestKitchenBackground(Background):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            name="kitchen",
            tags=["background", "pick_and_place"],
            # usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/assets_for_tests/reference_object_test_kitchen.usd",
            usd_path="/workspaces/isaaclab_arena/test_scene.usd",
            # initial_pose=initial_pose,
            object_min_z=-0.2,
        )

kitchen = ObjectReferenceTestKitchenBackground()

#%%

from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.object_base import ObjectType

cracker_box = ObjectReference(
    name="cracker_box",
    # prim_path="{ENV_REGEX_NS}/kitchen/_03_cracker_box",
    prim_path="{ENV_REGEX_NS}/kitchen/cracker_box",
    parent_asset=kitchen,
    object_type=ObjectType.RIGID,
)


destination_location = ObjectReference(
    name="drawer",
    # prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    prim_path="{ENV_REGEX_NS}/kitchen/kitchen_with_open_drawer/Cabinet_B_02",
    parent_asset=kitchen,
    object_type=ObjectType.RIGID,
)

#%%

from pxr import Usd, PhysxSchema

from isaaclab_arena.utils.usd_helpers import open_stage

usd_path = "/workspaces/isaaclab_arena/test_scene.usd"
with open_stage(usd_path) as stage:

    box_path = "/World/cracker_box"

    boxPrim = stage.GetPrimAtPath(box_path)
    cr_api = contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(boxPrim)
    cr_api.CreateThresholdAttr().Set(0)

    stage.Save()

    # Save



#%%



# kitchen.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
# cracker_box.set_initial_pose(
#     Pose(position_xyz=(3.69020713150969, -0.804121657812894, 1.2531903565606817), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
# )
# microwave.set_initial_pose(
#     Pose(position_xyz=(2.862758610786719, -0.39786255771393336, 1.087924015237011), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
# )

from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

# scene = Scene(assets=[kitchen, cracker_box, microwave])
scene = Scene(assets=[kitchen, cracker_box])
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=DummyTask(),
    # task=PickAndPlaceTask(cracker_box, microwave, kitchen),
    teleop_device=None,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 300
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%

test = (1.01230912348123409812340982, 0.0, 0.0, 0.0)

print(type(test[0]))

# %%

from pxr import Usd

# Create a new stage for composition
stage = Usd.Stage.CreateInMemory()
# Add the root/default prim
root_prim_path = "/World"
world = stage.DefinePrim(root_prim_path, "Xform")
stage.SetDefaultPrim(world)

# %%

from pxr import Gf, UsdGeom

# asset = cracker_box
asset = microwave

# Get the default prim path
default_prim_path = stage.GetDefaultPrim().GetPath()
assert default_prim_path is not None
# Construct the path for the asset prim
asset_path = str(default_prim_path) + "/" + asset.name
# Create the prim and reference the asset USD file.
prim = stage.DefinePrim(asset_path, "Xform")
prim.GetReferences().AddReference(asset.usd_path)

prim_xform = UsdGeom.Xform(prim)

# #
# for op in prim_xform.GetOrderedXformOps():
#     print(f"op.GetPrecsion(): {op.GetPrecision()}")

print(f"HERE")

# print(f"UsdGeom.XformOp.PrecisionDouble: {UsdGeom.XformOp.PrecisionDouble}")


def _is_double_precision(op: UsdGeom.XformOp) -> bool | None:
    # Detect if the op is None or doesn't contain precision.
    if not op:
        print("op is None")
        return None
    print(f"op.GetPrecision() {op.GetPrecision()}")
    return op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble


# trans_double = prim_xform.GetTranslateOp().GetPrecision() == UsdGeom.XformOp.PrecisionDouble
# orient_double = prim_xform.GetOrientOp().GetPrecision() == UsdGeom.XformOp.PrecisionDouble
# scale_double = prim_xform.GetScaleOp().GetPrecision() == UsdGeom.XformOp.PrecisionDouble
# print(f"prim_xform.GetTranslateOp().GetPrecision(): {prim_xform.GetTranslateOp().GetPrecision()}")
# print(f"prim_xform.GetOrientOp().GetPrecision(): {prim_xform.GetOrientOp().GetPrecision()}")
# print(f"prim_xform.GetScaleOp().GetPrecision(): {prim_xform.GetScaleOp().GetPrecision()}")

print(f"detecting trans {prim_xform.GetTranslateOp()}")
trans_double = _is_double_precision(prim_xform.GetTranslateOp())
print(f"detecting orient {prim_xform.GetOrientOp()}")
orient_double = _is_double_precision(prim_xform.GetOrientOp())
print("HERE2")
print(f"detecting scale {prim_xform.GetScaleOp()}")
scale_double = _is_double_precision(prim_xform.GetScaleOp())

print(f"asset.name {asset.name}")
print(f"trans_double {trans_double}")
print(f"orient_double {orient_double}")
print(f"scale_double {scale_double}")

# Add the transform
print("overwriting xforms")
prim_xform.ClearXformOpOrder()
if asset.initial_pose is not None:
    t = Gf.Vec3d(asset.initial_pose.position_xyz) if trans_double else Gf.Vec3f(asset.initial_pose.position_xyz)
    r = Gf.Quatd(*asset.initial_pose.rotation_wxyz) if orient_double else Gf.Quatf(*asset.initial_pose.rotation_wxyz)
    t_precision = UsdGeom.XformOp.PrecisionDouble if trans_double else UsdGeom.XformOp.PrecisionFloat
    r_precision = UsdGeom.XformOp.PrecisionDouble if orient_double else UsdGeom.XformOp.PrecisionFloat
    print(f"t {t} or type {type(t)}")
    print(f"r {r} or type {type(r)}")
    prim_xform.AddTranslateOp(precision=t_precision).Set(t)
    prim_xform.AddOrientOp(precision=r_precision).Set(r)
s = Gf.Vec3d(asset.scale) if scale_double else Gf.Vec3f(asset.scale)
s_precision = UsdGeom.XformOp.PrecisionDouble if scale_double else UsdGeom.XformOp.PrecisionFloat
prim_xform.AddScaleOp(precision=s_precision).Set(s)

# %%
