"""Launch Isaac Sim Simulator first."""

# from isaac_arena.app_launcher.app_launcher import app_launcher
from isaac_arena.isaaclab_utils.simulation_app import SimulationAppContext
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser


def main():
    """Script to run an Isaac Arena environment with a zero-action agent."""

    # Args
    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args()
    # simulation_app = app_launcher(args_cli)

    with SimulationAppContext(args_cli) as simulation_app: #headless=True, enable_cameras=True)

        # Post simulation app launch imports
        import gymnasium as gym
        import torch
        from isaac_arena.environments.compile_env import compile_arena_env_cfg
        from isaaclab_tasks.utils import parse_env_cfg

        # Compile an isaac arena environment configuration from existing isaac arena registry.
        arena_env_cfg = compile_arena_env_cfg(args_cli)
        gym.register(
            id=args_cli.task,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": arena_env_cfg,
            },
            disable_env_checker=True,
        )

        # Build the environment configuration in gym.
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        env = gym.make(args_cli.task, cfg=env_cfg)

        env.reset()
        # while simulation_app.is_running():
        import tqdm
        for _ in tqdm.tqdm(range(10)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        env.close()

    # simulation_app.close()


if __name__ == "__main__":
    main()
