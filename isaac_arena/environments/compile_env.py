import argparse

from isaac_arena.embodiments.embodiment_base import EmbodimentBase
from isaac_arena.embodiments.franka.franka_embodiment import FrankaEmbodiment
from isaac_arena.environments.arena_env import IsaacArenaEnvCfg
from isaac_arena.scene.pick_and_place_scene import MugInDrawerKitchenPickAndPlaceScene
from isaac_arena.scene.scene import SceneBase
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTaskCfg
from isaac_arena.tasks.task import TaskBase


def compile_arena_env_cfg(args_cli: argparse.Namespace) -> "IsaacArenaEnvCfg":
    embodiment = get_isaac_arena_embodiment(args_cli.embodiment)
    observations_cfg = embodiment.get_observation_cfg()
    actions_cfg = embodiment.get_action_cfg()
    events_cfg = embodiment.get_event_cfg()

    scene = get_isaac_arena_scene_cfg(args_cli.scene)
    # NOTE(cvolk): The scene apparently needs to hold a robot.
    scene.robot = embodiment.get_robot_cfg()

    terminations_cfg = get_isaac_arena_task_cfg(args_cli.arena_task)

    arena_env_cfg = IsaacArenaEnvCfg(
        observations=observations_cfg,
        actions=actions_cfg,
        events=events_cfg,
        scene=scene,
        terminations=terminations_cfg,
    )
    return arena_env_cfg


def get_isaac_arena_embodiment(embodiment: str) -> EmbodimentBase:
    if embodiment == "franka":
        return FrankaEmbodiment()
    else:
        raise ValueError(f"Embodiment {embodiment} not supported.")


def get_isaac_arena_scene_cfg(scene: str) -> SceneBase:
    if scene == "kitchen":
        return MugInDrawerKitchenPickAndPlaceScene().get_scene_cfg()
    else:
        raise ValueError(f"Scene {scene} not supported.")


def get_isaac_arena_task_cfg(task: str) -> TaskBase:
    if task == "pick_and_place":
        return PickAndPlaceTaskCfg()
    else:
        raise ValueError(f"Task {task} not supported.")
