from dataclasses import MISSING

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaac_arena.scene.scene import SceneBase

@configclass
class KitchenSceneCfg(InteractiveSceneCfg):
    background_scene: AssetBaseCfg = MISSING
    object: RigidObjectCfg = MISSING
    pick_up_object: RigidObjectCfg = MISSING


class PickAndPlaceSceneBase(SceneBase):
    def __init__(
        self, background_scene: AssetBaseCfg, pick_up_object: RigidObjectCfg, destination_object: RigidObjectCfg
    ):
        # The background scene
        self.background_scene = background_scene
        # An object, which has to be placed on/into the target object
        self.pick_up_object = pick_up_object
        # An object, which has to be placed on/into the target object
        self.destination_object = destination_object

    def get_scene_cfg(self) -> KitchenSceneCfg:
        return KitchenSceneCfg(background_scene=self.background_scene,
                        pick_up_object=self.pick_up_object,
                        destination_object=self.destination_object)

    # def get_observation_cfg(self) -> Any:
    #     class ObservationCfg:
    #         pick_up_object_position = ObjectPositionObservationCfg(
    #             object=self.pick_up_object,
    #         )

    #     return ObservationCfg()
