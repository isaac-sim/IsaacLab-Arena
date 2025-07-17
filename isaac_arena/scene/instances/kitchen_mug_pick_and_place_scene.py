from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaac_arena.scene.pick_and_place_scene import PickAndPlaceSceneBase, KitchenSceneCfg


class KitchenPickAndPlaceScene(PickAndPlaceSceneBase):

    def __init__(self):
        self.background_scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            # These positions are hardcoded for the kitchen scene. Its important to keep them.
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]
            ),
            spawn=UsdFileCfg(
                usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/kitchen_scene_teleop_v3.usd"
            ),
        )

        self.pick_up_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target_mug",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug_physics.usd",
                scale=(0.0125, 0.0125, 0.0125),
                activate_contact_sensors=True,
            ),
        )

        self.destination_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_mugs",
            spawn=sim_utils.CuboidCfg(
                size=[0.4, 0.65, 0.01],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,
            ),
        )

        super().__init__(
            background_scene=self.background_scene,
            pick_up_object=self.pick_up_object,
            destination_object=self.destination_object,
        )

    def get_scene_cfg(self) -> KitchenSceneCfg:
        return KitchenSceneCfg(background_scene=self.background_scene, object=self.destination_object, pick_up_object=self.pick_up_object)

    # TODO decide on how we do randomization.
    # def get_events_cfg(self) -> Dict[str, Any]:
    #     class EventsCfg:
    #         randomize_pick_up_object = RandomizeObjectCfg(
    #             object=self.pick_up_object,
    #             assets_list=[
    #                 "Object/mug.usd",
    #                 "Object/cup.usd",
    #                 "Object/bowl.usd",
    #             ],
    #         )
    #         randomize_pick_up_object_position = RandomizeObjectPositionCfg(
    #             object=self.pick_up_object,
    #             position_std=[0.1, 0.1, 0.1],
    #             orientation_std=[0.1, 0.1, 0.1],
    #         )

    #     return EventsCfg()
