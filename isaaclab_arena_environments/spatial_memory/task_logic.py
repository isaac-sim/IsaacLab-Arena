# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interface every spatial-memory task implements, plus the shared memory-probe and plan helpers."""

from __future__ import annotations

import math
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena_environments.spatial_memory.controller import Waypoint
from isaaclab_arena_environments.spatial_memory.parts import PARKING_POSE_Z_M, PartKind, PartSpec
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityModel, VisibilityTarget

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.embodiment_adapter import FrankaAdapter
    from isaaclab_arena_environments.spatial_memory.runtime import SpatialMemoryRuntime


@dataclass
class LayoutSample:
    """One sampled episode layout."""

    part_poses_e: torch.Tensor
    """Environment-frame poses (K, 7) of every part, ordered like ``part_specs()``."""

    params: dict[str, torch.Tensor]
    """Task parameters of the episode, each a tensor with a fixed shape across episodes."""


class PlanMode:
    """Which scripted controller a plan is for."""

    ORACLE = "oracle"
    """Uses the privileged layout, including hidden geometry."""

    BLIND = "blind"
    """Ignores hidden geometry: straight-line motions or nominal guesses where the oracle uses the hidden layout."""


class SpatialMemoryTaskLogic(ABC):
    """Scene parts, layout sampling, phases, success and scripted plans of one spatial-memory task.

    Instances are copied by Isaac Lab's managers, so they hold configuration only; episode state lives in
    ``SpatialMemoryRuntime``.

    Args:
        difficulty: Value of the task's difficulty knob.
        embodiment: Embodiment adapter.
    """

    name: str = ""
    """Environment name."""

    difficulty_name: str = "difficulty"
    """Human-readable name of the difficulty knob."""

    default_difficulty: float = 0.0
    """Default value of the difficulty knob."""

    episode_length_s: float = 30.0
    """Episode time limit."""

    task_description: str = ""
    """Natural-language instruction."""

    forbids_searching_again: bool = False
    """Whether exceeding the re-observation budget fails the episode."""

    reobservation_budget_s: float = 1.0
    """Seconds the memory targets may be seen after the instruction before the first grasp."""

    withholds_camera_frames: bool = False
    """Whether ``camera_blocked`` can withhold egocentric frames (the camera observation is then masked)."""

    reveal_delay_s: float = 3.5
    """Seconds the memory targets must stay out of view before the instruction is revealed (longer than the
    default 3 s frame history)."""

    def __init__(self, difficulty: float | None, embodiment: FrankaAdapter):
        self.difficulty = self.default_difficulty if difficulty is None else float(difficulty)
        self.embodiment = embodiment
        self._part_specs: list[PartSpec] | None = None

    # ----------------------------------------------------------------------------------------------------------------
    # Scene
    # ----------------------------------------------------------------------------------------------------------------

    def part_specs(self) -> list[PartSpec]:
        """Return the scene parts, in a fixed order."""
        if self._part_specs is None:
            self._part_specs = self.build_part_specs()
            names = [spec.name for spec in self._part_specs]
            assert len(names) == len(set(names)), f"Duplicate part names in {self.name}: {names}"
        return self._part_specs

    @abstractmethod
    def build_part_specs(self) -> list[PartSpec]:
        """Create the scene parts; called once."""

    def spec(self, name: str) -> PartSpec:
        """Return the spec of a part by name."""
        for spec in self.part_specs():
            if spec.name == name:
                return spec
        raise KeyError(name)

    def part_names(self) -> list[str]:
        """Return the names of all parts."""
        return [spec.name for spec in self.part_specs()]

    def item_names(self) -> list[str]:
        """Return the names of the movable items."""
        return [spec.name for spec in self.part_specs() if spec.kind == PartKind.ITEM]

    @abstractmethod
    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        """Sample the layout of one episode."""

    def new_pose_table(self) -> dict[str, list[float]]:
        """Return a pose table with every part parked out of the way, to be filled by ``sample_layout``."""
        table = {}
        for index, spec in enumerate(self.part_specs()):
            table[spec.name] = [3.0 + 0.3 * index, -3.0, PARKING_POSE_Z_M - spec.size_xyz[2], 0.0, 0.0, 0.0, 1.0]
            if spec.kind == PartKind.ITEM:
                # Items rest on the floor far from the robot instead of falling forever.
                table[spec.name][2] = 0.5 * spec.size_xyz[2] + 0.001
        return table

    def pose_table_to_tensor(self, table: dict[str, list[float]]) -> torch.Tensor:
        """Convert a pose table filled by ``sample_layout`` into a (K, 7) tensor."""
        return torch.tensor([table[spec.name] for spec in self.part_specs()], dtype=torch.float32)

    # ----------------------------------------------------------------------------------------------------------------
    # Visibility and phases
    # ----------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def visibility_targets(self) -> list[VisibilityTarget]:
        """Return the task-relevant surfaces whose visibility is logged."""

    @abstractmethod
    def hidden_target_names(self) -> list[str]:
        """Return the targets that must be out of view during the action phase."""

    def memory_target_names(self) -> list[str]:
        """Return the targets whose disappearance from view gates the instruction (empty: no instruction gate)."""
        return []

    def make_visibility_model(self, env) -> VisibilityModel:
        """Build the visibility model for the embodiment's egocentric camera."""
        sizes = {spec.name: spec.size_xyz for spec in self.part_specs()}
        return VisibilityModel(
            camera=self.embodiment.camera_model(),
            targets=self.visibility_targets(),
            part_sizes=sizes,
            part_names=self.part_names(),
            robot_occluders=self.embodiment.robot_occluders(),
            device=env.device,
        )

    def adjust_camera_and_occluder_poses(self, env, runtime, T_W_mount, T_W_occluders):
        """Hook to modify the camera mount and occluder poses (default: unchanged)."""
        return T_W_mount, T_W_occluders

    @abstractmethod
    def action_phase(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return whether each environment is in the action phase, during which hidden targets must stay hidden."""

    def static_part_mask(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return (N, K) parts that must not move while hidden; by default every part except manipulated items."""
        manipulated = set(self.manipulated_item_names())
        mask = torch.tensor([name not in manipulated for name in runtime.part_names], device=env.device)
        return mask.unsqueeze(0).expand(env.num_envs, -1)

    def manipulated_item_names(self) -> list[str]:
        """Return the items the robot is meant to move."""
        return []

    # ----------------------------------------------------------------------------------------------------------------
    # Contacts, success and observations
    # ----------------------------------------------------------------------------------------------------------------

    def allowed_contacts(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return the (N, S, K) mask of contacts that are part of the task.

        Sensors are ordered arm links, gripper bodies, then items (see ``collision.filtered_sensor_labels``). The
        default allows robot contact with manipulated items, item contact with each other, and every item's contact
        with the boards named in ``item_support_names``. Any robot contact with a board is a collision.
        """
        from isaaclab_arena_environments.spatial_memory.collision import filtered_sensor_labels

        labels = filtered_sensor_labels(self)
        manipulated = set(self.manipulated_item_names())
        items = set(self.item_names())
        supports = set(self.item_support_names())
        allowed = torch.zeros(len(labels), len(runtime.part_names), dtype=torch.bool, device=env.device)
        for sensor_index, label in enumerate(labels):
            for part_index, part in enumerate(runtime.part_names):
                if label in self.embodiment.arm_links or label in self.embodiment.gripper_bodies:
                    allowed[sensor_index, part_index] = part in manipulated
                else:
                    allowed[sensor_index, part_index] = part in items or part in supports
        return allowed.unsqueeze(0).expand(env.num_envs, -1, -1)

    def item_support_names(self) -> list[str]:
        """Return boards items may rest on or slide against (start and goal supports)."""
        return []

    def collisions_counted(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return whether collisions count as failures at this step (N,)."""
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    @abstractmethod
    def success_terms(self, env, runtime: SpatialMemoryRuntime) -> dict[str, torch.Tensor]:
        """Return the named conditions (each (N,)) that together make up task success.

        Collision and searched-again are handled separately. The final value of each term is recorded per episode.
        """

    def success(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return the task-specific success condition (N,): all ``success_terms`` hold."""
        success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for term in self.success_terms(env, runtime).values():
            success &= term
        return success

    def extra_failures(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return additional task-specific failure flags (N,)."""
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def instruction_dim(self) -> int:
        """Return the size of the instruction observation (0: no instruction term)."""
        return 0

    def instruction(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return the instruction observation (N, instruction_dim); zeros until revealed."""
        return torch.zeros(env.num_envs, self.instruction_dim(), device=env.device)

    def camera_blocked(self, env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
        """Return whether the egocentric image is withheld at this step (N,)."""
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # ----------------------------------------------------------------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------------------------------------------------------------

    def on_reset(self, env, runtime: SpatialMemoryRuntime, env_ids: torch.Tensor) -> None:
        """Hook called after a layout is written."""

    def on_step(self, env, runtime: SpatialMemoryRuntime, visible: torch.Tensor, active: torch.Tensor) -> None:
        """Hook called once per control step after visibility, phases and collisions are updated.

        The default implementation runs the memory probe when ``memory_target_names`` is not empty.
        """
        if self.memory_target_names():
            self.update_memory_probe(env, runtime, visible, active)

    def update_memory_probe(self, env, runtime: SpatialMemoryRuntime, visible: torch.Tensor, active: torch.Tensor):
        """Reveal the instruction once the memory targets were seen and then all stayed hidden for ``reveal_delay_s``.

        After the reveal and before the first gripper close, count the steps in which a memory target is visible;
        exceeding ``reobservation_budget_s`` marks the episode as "searched again".
        """
        indices = [runtime.target_names.index(name) for name in self.memory_target_names()]
        memory_visible = visible[:, indices].any(dim=-1) & active
        runtime.memory_seen |= memory_visible
        runtime.memory_unseen_steps = torch.where(
            memory_visible, torch.zeros_like(runtime.memory_unseen_steps), runtime.memory_unseen_steps + 1
        )
        delay_steps = int(math.ceil(self.reveal_delay_s / runtime.step_dt))
        newly_revealed = ~runtime.revealed & runtime.memory_seen & (runtime.memory_unseen_steps >= delay_steps) & active
        runtime.reveal_step[newly_revealed] = env.episode_length_buf[newly_revealed]
        runtime.revealed |= newly_revealed
        before_commit = runtime.revealed & ~newly_revealed & (runtime.first_close_step == -1)
        runtime.reobservation_steps += (before_commit & memory_visible).long()
        budget_steps = int(math.floor(self.reobservation_budget_s / runtime.step_dt))
        runtime.searched_again |= runtime.reobservation_steps > budget_steps

    # ----------------------------------------------------------------------------------------------------------------
    # Embodiment helpers and scripted plans
    # ----------------------------------------------------------------------------------------------------------------

    def tool_pose_w(self, env) -> torch.Tensor:
        """Return the tool pose (N, 7) in the world frame."""
        return self.embodiment.tool_pose_w(env)

    def gripper_closed(self, env) -> torch.Tensor:
        """Return whether the gripper is commanded closed (N,)."""
        return self.embodiment.gripper_closed(env)

    @abstractmethod
    def plan(self, env, runtime: SpatialMemoryRuntime, env_id: int, mode: str) -> list[Waypoint]:
        """Return the scripted waypoint plan of one environment for ``PlanMode.ORACLE`` or ``PlanMode.BLIND``."""

    def plan_gates(self, env, runtime: SpatialMemoryRuntime) -> dict[str, torch.Tensor]:
        """Return the runtime conditions plans may wait on, keyed by gate name."""
        return {"revealed": runtime.revealed.clone()}
