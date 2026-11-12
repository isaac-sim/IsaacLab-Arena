# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for tasks, which define scene additions, events, terminations and metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.relations.relations import RequiresReachability
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.task_transition import TaskTransition

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg


class TaskBase(ABC):
    """Base class for a task: the goal, success and failure conditions, and episode setup of an environment.

    Subclasses implement the abstract `get_*_cfg` methods to supply Isaac Lab manager configurations.
    """

    DEFAULT_EPISODE_LENGTH_S: float = 70.0

    def __init__(self, episode_length_s: float | None = None, task_description: str | None = None):
        """Initialize the task.

        Args:
            episode_length_s: Episode time limit in seconds. Falls back to `DEFAULT_EPISODE_LENGTH_S` when None.
            task_description: Natural-language description of the task, or None if there is none.
        """
        self.episode_length_s = episode_length_s if episode_length_s is not None else self.DEFAULT_EPISODE_LENGTH_S
        self.task_description = task_description

    @abstractmethod
    def get_scene_cfg(self) -> Any:
        """Return the scene configuration this task adds, such as contact sensors, or None if it adds none."""
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Declare the task's success criteria, failure conditions, and episode time budget."""
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_events_cfg(self) -> Any:
        """Return the Isaac Lab event configuration for this task."""
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        """Return the Isaac Lab Mimic environment configuration for this task.

        Args:
            arm_mode: Which arm or arms of the embodiment the Mimic subtasks are generated for.
        """
        raise NotImplementedError("Function not implemented yet.")

    @abstractmethod
    def get_metrics(self) -> list[MetricBase]:
        """Return the metrics computed for this task."""
        raise NotImplementedError("Function not implemented yet.")

    def get_observation_cfg(self) -> Any:
        """Return the observation configuration this task adds, or None if it adds none."""
        return None  # noqa: R501

    def get_tracked_events(self) -> list[TrackedEventCfg]:
        """Return recurring observations that do not affect task success or termination."""
        return []

    def get_rewards_cfg(self) -> Any:
        """Return the reward configuration of this task, or None if it defines no rewards."""
        return None  # noqa: R501

    def get_curriculum_cfg(self) -> Any:
        """Return the curriculum configuration of this task, or None if it has no curriculum."""
        return None  # noqa: R501

    def get_commands_cfg(self) -> Any:
        """Return the command configuration of this task, or None if it defines no commands."""
        return None  # noqa: R501

    def get_recorder_term_cfg(self) -> RecorderManagerBaseCfg:
        """Return the recorder configuration this task adds, or None if it adds none."""
        return None  # noqa: R501

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return the viewer camera configuration for this task."""
        return ViewerCfg()

    def get_episode_length_s(self) -> float:
        """Return the episode time limit in seconds."""
        return self.episode_length_s

    def get_task_description(self) -> str | None:
        """Return the natural-language task description, or None if there is none."""
        return self.task_description

    def configure_for_embodiment(self, embodiment: EmbodimentBase) -> None:
        """Configure this task with the interfaces owned by the selected embodiment.

        The base implementation does nothing. Tasks and embodiments are constructed independently, so a task
        that depends on its embodiment overrides this to receive it.

        Args:
            embodiment: The embodiment the task is paired with.
        """
        pass

    def apply_reachability_constraints(self) -> None:
        """Stamp RequiresReachability on the objects the robot must be able to reach for this task."""
        pass

    def _apply_reachability_constraints(self, targets: list[Asset]) -> None:
        """Stamp RequiresReachability on each placed object in targets.

        An object reference or a background location is a static, non-placed asset and is skipped;
        only placed objects are IK-checked during layout validation.
        """
        for target in targets:
            if isinstance(target, Object) and not target.has_relation(RequiresReachability):
                target.add_relation(RequiresReachability())

    @classmethod
    def success_state_transition(cls, **_) -> TaskTransition:
        """Inform constraint resolution what the task's success condition implies about the state change.

        Resolution forwards all of the task's args; subclasses override this to bind the ones they act
        on as named parameters and absorb the rest with ``**_``.
        """
        raise NotImplementedError("success_state_transition not implemented yet.")
