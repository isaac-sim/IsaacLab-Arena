# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

rigid_object_variant_cfg = RigidObjectVariantCfg(
    assets={
        "box": RigidObjectCfg(
            name="box",
            spawn=UsdFileCfg(
                usd_path="path/to/box.usd",
            ),
            scale=(1.0, 1.0, 1.0),
            initial_pose=Pose(position=(1.0, 2.0, 3.0), orientation=(0.0, 0.0, 0.0, 1.0)),
        ),
        "sphere": RigidObjectCfg(
            name="sphere",
            spawn=UsdFileCfg(
                usd_path="path/to/sphere.usd",
            ),
            scale=(2.0, 2.0, 2.0),
            initial_pose=Pose(position=(4.0, 5.0, 6.0), orientation=(0.0, 0.0, 0.0, 1.0)),
        ),
    }
}



cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
object_set = RigidObjectSet(name="object_set", objects=[cracker_box, tomato_soup_can])
object_set.set_initial_pose(Pose(position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0)))




cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
object_set = RigidObjectSet(name="object_set", objects=[cracker_box, tomato_soup_can])
object_set.set_initial_pose(
    PoseVariant(poses={
        cracker_box: Pose(position=(1.0, 2.0, 3.0), orientation=(0.0, 0.0, 0.0, 1.0)),
        tomato_soup_can: Pose(position=(4.0, 5.0, 6.0), orientation=(0.0, 0.0, 0.0, 1.0)),
    })
)
