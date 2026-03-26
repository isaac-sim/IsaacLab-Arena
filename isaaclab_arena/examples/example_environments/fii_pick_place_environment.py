import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase
from isaaclab_arena.utils.pose import Pose

class FiiPickPlaceEnvironment(ExampleEnvironmentBase):
    name: str = "fii_pick_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.fii_pick_and_place_task import FiiPickAndPlaceTask
        from isaaclab.assets import AssetBaseCfg
        import isaaclab.sim as sim_utils
        object = self.asset_registry.get_asset_by_name("object")(initial_pose=Pose(position_xyz=(0.0, 0.75, 1.0)))
        packing_table = self.asset_registry.get_asset_by_name("packing_table")(initial_pose=Pose(position_xyz=(0.0, 0.85, 0.)))
        embodiment = self.asset_registry.get_asset_by_name("fii")()
        ground = self.asset_registry.get_asset_by_name("ground_plane")()
        dome_light = self.asset_registry.get_asset_by_name("dome_light")()

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None
        
        scene = Scene(assets=[object, packing_table, ground, dome_light])

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=FiiPickAndPlaceTask(object, packing_table, episode_length_s=20.0),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="fii")
        parser.add_argument("--teleop_device", type=str, default=None)