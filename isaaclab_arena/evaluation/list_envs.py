from isaaclab_arena_environments.cli import ensure_environments_registered
from isaaclab_arena.assets.registries import EnvironmentRegistry

ensure_environments_registered()

[print(n) for n in sorted(EnvironmentRegistry().get_all_keys())]