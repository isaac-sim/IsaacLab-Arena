from isaac_arena.scene.asset import Asset
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


class PickUpObject(Asset):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self, pick_up_object: RigidObjectCfg, name: str):
        self.pick_up_object = pick_up_object
        self.tags = ["pick_up_object"]
        self.name = name

    def get_pick_up_object(self) -> RigidObjectCfg:
        """Return the configured pick-up object asset."""
        return self.pick_up_object


class Mug(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            pick_up_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_mug",
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug_physics.usd"
                    ),
                    scale=(0.0125, 0.0125, 0.0125),
                    activate_contact_sensors=True,
                ),
            ),
            name="mug",
        )


class GelatinBox(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            pick_up_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_gelatin_box",
                spawn=UsdFileCfg(
                    usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/gelatin_box_physics.usd",
                    scale=(1.0, 1.0, 1.0),
                    activate_contact_sensors=True,
                ),
            ),
            name="gelatin_box",
        )


class MacandCheeseBox(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            pick_up_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_mac_and_cheese_box",
                spawn=UsdFileCfg(
                    usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mac_n_cheese_physics.usd",
                    scale=(1.0, 1.0, 1.0),
                    activate_contact_sensors=True,
                ),
            ),
            name="mac_and_cheese_box",
        )


class SugarBox(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            pick_up_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_sugar_box",
                spawn=UsdFileCfg(
                    usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/sugar_box_physics.usd",
                    scale=(1.0, 1.0, 1.0),
                    activate_contact_sensors=True,
                ),
            ),
            name="sugar_box",
        )


class TomatoSoupCan(PickUpObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            pick_up_object=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/target_tomato_soup_can",
                spawn=UsdFileCfg(
                    usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/tomato_soup_physics.usd",
                    scale=(1.0, 1.0, 1.0),
                    activate_contact_sensors=True,
                ),
            ),
            name="tomato_soup_can",
        )
