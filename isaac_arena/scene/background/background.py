import isaaclab.sim as sim_utils
from isaac_arena.scene.asset import Asset
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


class Background(Asset):
    """
    Encapsulates the background scene config for a environment.
    """

    def __init__(
        self,
        background_scene: AssetBaseCfg,
        destination_object: RigidObjectCfg,
        pick_up_object_location: RigidObjectCfg.InitialStateCfg,
        name: str,
    ):
        self.background_scene = background_scene
        self.destination_object = destination_object
        self.pick_up_object_location = pick_up_object_location
        self.tags = ["background"]
        self.name = name

    def get_background(self) -> AssetBaseCfg:
        """Return the configured background scene asset."""
        return self.background_scene

    def get_destination(self) -> RigidObjectCfg:
        """Return the configured destination-object asset."""
        return self.destination_object

    def get_pick_up_object_location(self) -> RigidObjectCfg.InitialStateCfg:
        """Return the configured pick-up object location."""
        return self.pick_up_object_location


class Kitchen(Background):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    def __init__(self):
        # Background scene (static kitchen environment)
        super().__init__(
            background_scene=AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Kitchen",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]),
                spawn=UsdFileCfg(
                    usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/kitchen_scene_teleop_v3.usd"
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
            pick_up_object_location=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
            name="kitchen",
        )
