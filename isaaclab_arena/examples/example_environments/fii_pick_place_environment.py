import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

class FiiPickPlaceEnvironment(ExampleEnvironmentBase):
    name: str = "fii_pick_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.fii_pick_and_place_task import FiiPickAndPlaceTask

        pick_up_object = self.asset_registry.get_asset_by_name("io_board")()
        packing_table = self.asset_registry.get_asset_by_name("packing_table")()
        embodiment = self.asset_registry.get_asset_by_name("fii")()

        scene = Scene(assets=[packing_table, pick_up_object])
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None
        
        scene = Scene(assets=[pick_up_object, packing_table])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=FiiPickAndPlaceTask(pick_up_object, packing_table, episode_length_s=20.0),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="io_board")
        parser.add_argument("--embodiment", type=str, default="fii")
        parser.add_argument("--teleop_device", type=str, default=None)