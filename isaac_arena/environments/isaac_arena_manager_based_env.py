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


from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils import configclass

from isaac_arena.metrics.metric_base import MetricBase


@configclass
class IsaacArenaManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for an Isaac Arena environment."""

    # NOTE(alexmillane, 2025-07-29): The following definitions are taken from the base class.
    # scene: InteractiveSceneCfg
    # observations: object
    # actions: object
    # events: object
    # terminations: object
    # recorders: object

    # Kill the unused managers
    commands = None
    rewards = None
    curriculum = None

    # Metrics
    metrics: list[MetricBase] | None = None

    def __post_init__(self):
        """Post initialization."""
        # NOTE(xinjieyao, 2025-09-22): decimation & sim.dt are set to match the WBC policy trained frequency.
        # Any changes to these settings shall impact G1-WBC performance, therefore should be carefully considered.
        # Especially, any settings slower than 200Hz & 4 decimation shall impact G1-WBC performance.
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        self.wait_for_textures = False
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        # NOTE(peterd, 2025-09-23) Set the render interval lower than decimation to smooth out the rendering.
        self.sim.render_interval = 2


@configclass
class IsaacArenaManagerBasedMimicEnvCfg(IsaacArenaManagerBasedRLEnvCfg, MimicEnvCfg):
    """Configuration for an Isaac Arena environment."""

    def __post_init__(self):
        super().__post_init__()

    # NOTE(alexmillane, 2025-09-10): The following members are defined in the MimicEnvCfg class.
    # Restated here for clarity.
    # datagen_config: DataGenConfig = DataGenConfig()
    # subtask_configs: dict[str, list[SubTaskConfig]] = {}
    # task_constraint_configs: list[SubTaskConstraintConfig] = []
    pass
