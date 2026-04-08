# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Advanced example: externally-defined environment with a custom task and embodiment.

Builds on the basic example by defining a custom task and a custom embodiment
variant (with modified controller gains) in the same file::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_steps 100 \\
        --external_environment_class_path \\
        isaaclab_arena_examples.external_environments.advanced:ExternalFrankaTableWithTaskEnvironment \\
        franka_table_with_task
"""

import argparse
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
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

    def get_termination_cfg(self):
        n = self.num_steps_for_success
        success = TerminationTermCfg(func=lambda env, n=n: env.episode_length_buf >= n)
        return SuccessAfterNStepsTerminationsCfg(success=success)

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))


@configclass
class SuccessAfterNStepsTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


# ---------------------------------------------------------------------------
# Custom embodiment
# ---------------------------------------------------------------------------


@register_asset
class SoftFrankaIKEmbodiment(FrankaIKEmbodiment):
    """Franka IK embodiment with halved joint PD gains for more compliant behaviour.

    The standard ``franka_ik`` uses stiffness 400 / damping 80 on the shoulder
    and forearm actuator groups.  This variant halves both, which is useful for
    contact-rich or force-sensitive tasks.
    """

    name = "franka_ik_soft"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            actuator = self.scene_config.robot.actuators[actuator_name]
            actuator.stiffness = 200.0
            actuator.damping = 40.0


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
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="cracker_box")
