# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_tasks.utils import parse_env_cfg

from isaac_arena.assets.asset_registry import get_environment_configuration_from_registry
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.environments.isaac_arena_manager_based_env import IsaacArenaManagerBasedRLEnvCfg
from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaac_arena.utils.configclass import combine_configclass_instances


class ArenaEnvBuilder:
    """Compose Isaac Arena → IsaacLab configs"""

    DEFAULT_SCENE_CFG = InteractiveSceneCfg(num_envs=4096, env_spacing=30.0, replicate_physics=False)

    def __init__(self, arena_env: IsaacArenaEnvironment, args: argparse.Namespace):
        self.arena_env = arena_env
        self.args = args

    # ---------- factory ----------

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ArenaEnvBuilder:
        cfgs = get_environment_configuration_from_registry(args.background, args.object, args.embodiment)
        arena_env = IsaacArenaEnvironment(
            name=f"pick_and_place_{args.embodiment}_{args.background}_{args.object}",
            embodiment=cfgs["embodiment"],
            scene=PickAndPlaceScene(cfgs["background"], cfgs["object"]),
            task=PickAndPlaceTask(),
        )
        return cls(arena_env, args)

    def compose_manager_cfg(self) -> IsaacArenaManagerBasedRLEnvCfg:
        """Return base ManagerBased cfg (scene+events+terminations+xr), no registration."""
        robot_pose = self.arena_env.scene.get_robot_initial_pose()
        self.arena_env.embodiment.set_robot_initial_pose(robot_pose)

        scene_cfg = combine_configclass_instances(
            "SceneCfg",
            self.DEFAULT_SCENE_CFG,
            self.arena_env.scene.get_scene_cfg(),
            self.arena_env.embodiment.get_scene_cfg(),
        )
        events_cfg = combine_configclass_instances(
            "EventsCfg",
            self.arena_env.embodiment.get_event_cfg(),
            self.arena_env.scene.get_events_cfg(),
        )
        termination_cfg = combine_configclass_instances(
            "TerminationCfg",
            self.arena_env.task.get_termination_cfg(),
            self.arena_env.scene.get_termination_cfg(),
        )
        return IsaacArenaManagerBasedRLEnvCfg(
            observations=self.arena_env.embodiment.get_observation_cfg(),
            actions=self.arena_env.embodiment.get_action_cfg(),
            events=events_cfg,
            scene=scene_cfg,
            terminations=termination_cfg,
            xr=self.arena_env.embodiment.get_xr_cfg(),
        )

    def compose_mimic_cfg(
        self, base_cfg: IsaacArenaManagerBasedRLEnvCfg
    ) -> tuple[IsaacArenaManagerBasedRLEnvCfg, str | type[ManagerBasedRLMimicEnv]]:
        """Return (combined_mimic_cfg, entry_point) without registering."""
        task_mimic_env_cfg = self.arena_env.task.get_mimic_env_cfg(embodiment_name=self.arena_env.embodiment.name)
        combined = combine_configclass_instances("MimicEnvCfg", base_cfg, task_mimic_env_cfg)
        entry_point = self.arena_env.embodiment.get_mimic_env()
        return combined, entry_point

    def build_unregistered(self) -> tuple[str, ManagerBasedRLEnvCfg, str | type[ManagerBasedRLMimicEnv]]:
        """
        Compose final cfg and entry point WITHOUT registering or parsing.
        """
        base = self.compose_manager_cfg()
        if self.args.mimic:
            final, entry = self.compose_mimic_cfg(base)
        else:
            final, entry = base, "isaaclab.envs:ManagerBasedRLEnv"
        return self.arena_env.name, final, entry

    def build_registered(self) -> tuple[str, ManagerBasedRLEnvCfg]:
        """Register Gym env and parse runtime cfg."""
        name = self.arena_env.name
        _, cfg_entry, entry_point = self.build_unregistered()
        gym.register(
            id=name,
            entry_point=entry_point,
            kwargs={"env_cfg_entry_point": cfg_entry},
            disable_env_checker=True,
        )
        runtime_cfg = parse_env_cfg(
            name,
            device=self.args.device,
            num_envs=self.args.num_envs,
            use_fabric=not self.args.disable_fabric,
        )
        return name, runtime_cfg

    def make_registered(self) -> ManagerBasedEnv:
        name, runtime_cfg = self.build_registered()
        return gym.make(name, cfg=runtime_cfg).unwrapped


# TODO(Vik, 2025-08-29): Remove this function.
def get_arena_env_cfg(args_cli: argparse.Namespace) -> tuple[ManagerBasedRLEnvCfg, str]:
    builder = ArenaEnvBuilder.from_args(args_cli)
    name, cfg = builder.build_registered()
    return cfg, name
