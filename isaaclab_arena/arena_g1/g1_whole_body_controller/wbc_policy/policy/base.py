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

from abc import ABC, abstractmethod


class WBCPolicy(ABC):
    """Base class for implementing control policies in the Gear'WBC framework.

    A Policy defines how an agent should behave in an environment by mapping observations
    to actions. This abstract base class provides the interface that all concrete policy
    implementations must follow.
    """

    def set_goal(self, goal: dict[str, any]):
        """Set the command from the planner that the policy should follow.

        Args:
            goal: Dictionary containing high-level commands or goals from the planner
        """
        pass

    def set_observation(self, observation: dict[str, any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: Dictionary containing the current state/observation of the environment
        """
        self.observation = observation

    @abstractmethod
    def get_action(self, time: float | None = None) -> dict[str, any]:
        """Compute and return the next action at the specified time, based on current observation
        and planner command.

        Args:
            time: Optional "monotonic time" for time-dependent policies

        Returns:
            Dictionary containing the action to be executed
        """

    def close(self):
        """Clean up any resources used by the policy."""
        pass
