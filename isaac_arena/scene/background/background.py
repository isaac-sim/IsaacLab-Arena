import isaaclab.sim as sim_utils
from isaac_arena.scene.asset import Asset
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


class Background(Asset):
    """
    Encapsulates the background scene config for a environment.
    """

    def __init__(self, background_scene_cfg: AssetBaseCfg, name: str, tags: list[str]):
        super().__init__(name, tags)
        self.background_scene_cfg = background_scene_cfg


class PickAndPlaceBackground(Background):
    """
    Encapsulates the background scene config for a environment.
    """

    def __init__(
        self,
        background_scene_cfg: AssetBaseCfg,
        destination_object_cfg: RigidObjectCfg,
        pick_up_object_location_cfg: RigidObjectCfg.InitialStateCfg,
        name: str,
    ):
        super().__init__(background_scene_cfg, name, ["background", "pick_and_place"])
        self.destination_object_cfg = destination_object_cfg
        self.pick_up_object_location_cfg = pick_up_object_location_cfg

    def get_background_cfg(self) -> AssetBaseCfg:
        """Return the configured background scene asset."""
        return self.background_scene_cfg

    def get_destination_cfg(self) -> RigidObjectCfg:
        """Return the configured destination-object asset."""
        return self.destination_object_cfg

    def get_pick_up_object_location_cfg(self) -> RigidObjectCfg.InitialStateCfg:
        """Return the configured pick-up object location."""
        return self.pick_up_object_location_cfg


class KitchenPickAndPlaceBackground(PickAndPlaceBackground):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    def __init__(self):
        # Background scene (static kitchen environment)
        super().__init__(
            background_scene_cfg=AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Kitchen",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/kitchen_scene_teleop_v3.usd"
                    )
                ),
            ),
            destination_object_cfg=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_mugs",
                spawn=sim_utils.CuboidCfg(
                    size=[0.4, 0.65, 0.01],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    activate_contact_sensors=True,
                ),
            ),
            pick_up_object_location_cfg=RigidObjectCfg.InitialStateCfg(
                pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]
            ),
            name="kitchen_pick_and_place",
        )


class PackingTablePickAndPlaceBackground(PickAndPlaceBackground):
    """
    Encapsulates the background scene and destination-object config for a packing table pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            background_scene_cfg=AssetBaseCfg(
                prim_path="/World/envs/env_.*/PackingTable",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
                spawn=UsdFileCfg(
                    usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/mindmap/packing_table_arena.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            ),
            destination_object_cfg=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PackingTable/container_h20",
            ),
            pick_up_object_location_cfg=RigidObjectCfg.InitialStateCfg(
                pos=[-0.35, 0.40, 1.0413], rot=[1.0, 0.0, 0.0, 0.0]
            ),
            name="packing_table_pick_and_place",
        )
