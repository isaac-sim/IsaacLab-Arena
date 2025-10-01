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

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.dummy_task import DummyTask
from isaac_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka")()
# cracker_box = asset_registry.get_asset_by_name("cracker_box")()
microwave = asset_registry.get_asset_by_name("microwave")()

# cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
microwave.set_initial_pose(Pose(position_xyz=(0.45, 0.0, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

scene = Scene(assets=[background, microwave])
isaac_arena_environment = IsaacArenaEnvironment(
    name="reference_object_test",
    embodiment=embodiment,
    scene=scene,
    task=DummyTask(),
    teleop_device=None,
)

args_cli = get_isaac_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

#%
#%%
# Run some zero actions.
NUM_STEPS = 500
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%



class ObjectVelocityRecorder(RecorderTerm):
    """Records the linear velocity of an object for each sim step of an episode."""

    name = "object_linear_velocity"

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.object_name = cfg.object_name

    def record_post_step(self):
        # NOTE(alexmillane, 2025-09-30): This assumes the the object is a rigid object.
        object_linear_velocity = self._env.scene[self.object_name].data.root_link_vel_w[:, :3]
        assert object_linear_velocity.shape == (self._env.num_envs, 3)
        return self.name, object_linear_velocity


@configclass
class ObjectVelocityRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ObjectVelocityRecorder
    object_name: str = MISSING


class ObjectMovedRateMetric(MetricBase):
    """Computes the object-moved rate.

    The object-moved rate is the number of episodes in which the object moved, divided
    by the total number of episodes.
    """

    name = "object_moved_rate"
    recorder_term_name = ObjectVelocityRecorder.name

    def __init__(self, object: Asset, object_velocity_threshold: float = 0.5):
        """Initializes the object-moved rate metric.

        Args:
            object(Asset): The object to compute the object-moved rate for.
            object_velocity_threshold(float): The threshold for the object velocity to be considered moved.
        """
        super().__init__()
        self.object = object
        self.object_velocity_threshold = object_velocity_threshold

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        """Return the recorder term configuration for the object-moved rate metric."""
        return ObjectVelocityRecorderCfg(object_name=self.object.name)

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        """Computes the object-moved rate from the recorded metric data.

        Args:
            recorded_metric_data(list[np.ndarray]): The recorded object velocity per simulated episode.

        Returns:
            The object-moved rate(float). Value between 0 and 1. The proportion of episodes
                in which the object moved.
        """
        object_velocity_per_demo = recorded_metric_data
        object_moved_per_demo = []
        for object_velocity in object_velocity_per_demo:
            assert object_velocity.ndim == 2
            assert object_velocity.shape[1] == 3
            object_linear_velocity_magnitude = np.linalg.norm(object_velocity, axis=-1)
            object_moved = np.any(object_linear_velocity_magnitude > self.object_velocity_threshold)
            object_moved_per_demo.append(object_moved)
        object_moved_rate = np.mean(object_moved_per_demo)
        return object_moved_rate




#%%

microwave.open(env, env_ids=None)

microwave.get_openness(env)


#%%

from isaac_arena.utils.joint_utils import get_normalized_joint_position
from isaaclab.managers import SceneEntityCfg


get_normalized_joint_position(
    env,
    SceneEntityCfg(
        name=microwave.name,
        joint_names=[microwave.openable_joint_name],
    )
)

# %%
