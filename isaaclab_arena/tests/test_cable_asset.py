# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def test_cable_asset_config():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import CableObjectCfg

    from isaaclab_arena.assets.cable import Cable
    from isaaclab_arena.utils.pose import Pose

    initial_pose = Pose(position_xyz=(0.1, 0.2, 0.3))
    cable = Cable(
        name="test_cable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
            physics_material=sim_utils.CableMaterialCfg(),
        ),
        initial_pose=initial_pose,
    )

    scene_key, cable_cfg = cable.get_object_cfg()
    assert scene_key == "test_cable"
    assert isinstance(cable_cfg, CableObjectCfg)
    assert cable_cfg.prim_path == "{ENV_REGEX_NS}/Cable"
    assert cable_cfg.init_state.pos == initial_pose.position_xyz
    assert cable_cfg.init_state.rot == initial_pose.rotation_xyzw
    assert cable.get_event_cfg() == ("test_cable", None)
