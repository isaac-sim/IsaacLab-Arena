from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaac_arena.scene.scene import SceneBase
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass


@configclass
class PickAndPlaceSceneCfg(InteractiveSceneCfg):
    background_scene: AssetBaseCfg = MISSING
    pick_up_object: RigidObjectCfg = MISSING
    destination_object: RigidObjectCfg = MISSING

    # TODO(cvolk): It seems like the scene needs to hold a robot
    robot: ArticulationCfg = MISSING


class KitchenPickAndPlaceScene(SceneBase):
    def __init__(
        self, pick_up_object: RigidObjectCfg, destination_object: RigidObjectCfg, robot: ArticulationCfg = None
    ):
        super().__init__()
        # The background scene
        self.background_scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            # These positions are hardcoded for the kitchen scene. Its important to keep them.
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]),
            spawn=UsdFileCfg(
                usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/kitchen_scene_teleop_v3.usd"
            ),
        )
        # An object, which has to be placed on/into the target object
        self.pick_up_object: RigidObjectCfg = pick_up_object

        # An object, which has to be placed on/into the target object
        self.destination_object: RigidObjectCfg = destination_object

        # Robot configuration
        self.robot: ArticulationCfg = robot

    def get_scene_cfg(self) -> PickAndPlaceSceneCfg:
        return PickAndPlaceSceneCfg(
            num_envs=4096,
            env_spacing=30.0,
            replicate_physics=False,
            background_scene=self.background_scene,
            pick_up_object=self.pick_up_object,
            destination_object=self.destination_object,
            robot=self.robot,
        )


class MugInDrawerKitchenPickAndPlaceScene(KitchenPickAndPlaceScene):
    def __init__(self, robot: ArticulationCfg = None):
        super().__init__(
            pick_up_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_mug",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            ),
            destination_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_mugs",
                spawn=sim_utils.CuboidCfg(
                    size=[0.4, 0.65, 0.01],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    activate_contact_sensors=True,
                ),
            ),
            robot=robot,
        )
