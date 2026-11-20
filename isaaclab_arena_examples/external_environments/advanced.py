# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Advanced example: externally-defined environment with a custom task and embodiment.

Builds on the basic example by defining a custom task, a custom embodiment
variant (with modified controller gains), and two custom episode recorder terms
in the same file::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_steps 100 \\
        --external_environment_class_path \\
        isaaclab_arena_examples.external_environments.advanced:ExternalFrankaTableWithTaskEnvironment \\
        franka_table_with_task
"""

import argparse
import torch

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTerm, EpisodeRecorderTermCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# ---------------------------------------------------------------------------
# Custom task
# ---------------------------------------------------------------------------


class SuccessAfterNStepsTask(TaskBase):
    """Minimal task: the episode succeeds after a fixed number of steps."""

    def __init__(self, num_steps_for_success: int = 50, episode_length_s: float = 10.0):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=f"Succeed after {num_steps_for_success} steps",
        )
        self.num_steps_for_success = num_steps_for_success

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="reach_step_count",
                    predicate_sequence=[self.has_reached_step_count],
                )
            ],
        )

    def has_reached_step_count(self, env):
        """Check whether each episode has reached the requested number of steps."""
        return env.episode_length_buf >= self.num_steps_for_success

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))


# ---------------------------------------------------------------------------
# Custom embodiment
# ---------------------------------------------------------------------------


@register_asset
class SoftFrankaIKEmbodiment(FrankaIKEmbodiment):
    """Franka IK embodiment with halved joint PD gains for more compliant behaviour.

    The standard ``franka_ik`` uses stiffness 400 / damping 80 on its shoulder
    and forearm actuator groups. This variant halves both, which is useful for
    contact-rich or force-sensitive tasks.
    """

    registry_name = "franka_ik_soft"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            actuator = self.scene_config.robot.actuators[actuator_name]
            actuator.stiffness = 200.0
            actuator.damping = 40.0


# ---------------------------------------------------------------------------
# Custom episode recorder terms
# ---------------------------------------------------------------------------


def record_object_final_height(env, env_id: int, object_name: str) -> dict[str, float]:
    """Records the height [m] of an object when the episode ends."""
    return {"final_height": float(env.arena_world.get_pose_w(object_name)[env_id, 2])}


class ObjectPeakSpeed(EpisodeRecorderTerm):
    """Records the highest linear speed [m/s] an object reached during the episode."""

    def __init__(self, cfg: EpisodeRecorderTermCfg, env):
        super().__init__(cfg, env)
        self._object_name = cfg.params["object_name"]
        self._peak_speed = torch.zeros(env.num_envs, device=env.device)

    def post_step(self):
        velocity_w = self._env.arena_world.get_mean_linear_velocity_w(self._object_name)
        speed = torch.linalg.norm(velocity_w, dim=-1)
        # Update in place: a tensor created inside torch.inference_mode(), where the env may be stepped,
        # cannot be written in place by reset() outside it.
        torch.maximum(self._peak_speed, speed, out=self._peak_speed)

    def reset(self, env_ids=None):
        self._peak_speed[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, env_id: int, object_name: str) -> dict[str, float]:
        # The manager passes every cfg.params entry as a keyword argument, so object_name must be
        # accepted here even though __init__ already read it.
        return {"peak_speed": float(self._peak_speed[env_id])}


# ---------------------------------------------------------------------------
# External environment
# ---------------------------------------------------------------------------


class ExternalFrankaTableWithTaskEnvironment(ExampleEnvironmentBase):

    name: str = "franka_table_with_task"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name("franka_ik_soft")()

        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
        )
        table_reference.add_relation(IsAnchor())
        pick_up_object.add_relation(On(table_reference))

        scene = Scene(assets=[background, table_reference, pick_up_object, light])

        task = SuccessAfterNStepsTask(
            num_steps_for_success=50,
            episode_length_s=10.0,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            episode_recorder_terms={
                "object_final_height": EpisodeRecorderTermCfg(
                    func=record_object_final_height, params={"object_name": pick_up_object.name}
                ),
                "object_peak_speed": EpisodeRecorderTermCfg(
                    func=ObjectPeakSpeed, params={"object_name": pick_up_object.name}
                ),
            },
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="cracker_box")
