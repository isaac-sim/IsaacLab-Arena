import argparse

from isaaclab.app import AppLauncher


def get_isaac_arena_cli_parser() -> argparse.ArgumentParser:
    """Get a complete argument parser with both Isaac Lab and Isaac Arena arguments."""
    parser = argparse.ArgumentParser(description="Isaac Arena CLI parser.")
    add_isaac_lab_cli_args(parser)
    add_isaac_arena_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def add_isaac_lab_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add Isaac Lab specific command line arguments to the given parser."""

    isaac_lab_group = parser.add_argument_group("Isaac Lab Arguments", "Arguments specific to Isaac Lab framework")

    isaac_lab_group.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    isaac_lab_group.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    isaac_lab_group.add_argument("--task", type=str, default=None, help="Name of the task.")
    isaac_lab_group.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Enable Pinocchio.",
    )


def add_isaac_arena_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add Isaac Arena specific command line arguments to the given parser."""

    isaac_arena_group = parser.add_argument_group(
        "Isaac Arena Arguments", "Arguments specific to Isaac Arena environment"
    )

    isaac_arena_group.add_argument("--embodiment", type=str, required=True, help="Name of the embodiment.")
    isaac_arena_group.add_argument("--scene", type=str, required=True, help="Name of the scene.")
    # TODO(covlk): "task" currently interferes with the isaac_lab "task" argument where we pass eg Isaac-Arena-Kitchen-v0.
    # Maybe we should generate this from all arguments dynamically and let the 'task' argument be an isaac arena task
    # such as "pick_and_place"
    isaac_arena_group.add_argument("--arena_task", type=str, required=True, help="Name of the task.")
