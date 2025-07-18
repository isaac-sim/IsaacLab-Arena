import gymnasium as gym

from isaac_arena.environments.arena_env import ArenaEnvCfg

gym.register(
    id="Isaac-Arena-Kitchen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ArenaEnvCfg,
    },
    disable_env_checker=True,
)
