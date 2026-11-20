# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interface every spatial-memory task implements, plus the shared memory probe."""

from __future__ import annotations

import math
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from isaaclab_arena_environments.spatial_memory.collision import filtered_sensor_labels
from isaaclab_arena_environments.spatial_memory.embodiment_adapter import FRANKA_READY_JOINT_POS, EmbodimentAdapter
from isaaclab_arena_environments.spatial_memory.episode_state import NOT_SET
from isaaclab_arena_environments.spatial_memory.parts import PARKING_POSE_Z_M, PartKind, PartSpec
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityModel, VisibilityTarget

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.episode_state import SpatialMemoryEpisodeState

PARKING_X_M = 3.0
PARKING_Y_M = -3.0
PARKING_SPACING_M = 0.3
"""Unused parts are parked in a row from (``PARKING_X_M``, ``PARKING_Y_M``), away from every robot."""


@dataclass
class LayoutSample:
    """One sampled episode layout."""

    part_poses_e: torch.Tensor
    """Environment-frame poses (K, 7) of every part, ordered like ``part_specs()``."""

    layout_params: dict[str, torch.Tensor]
    """Task parameters of the episode, each a tensor with a fixed shape across episodes."""


class PlanMode(str, Enum):
    """Which scripted controller a plan is for."""

    ORACLE = "oracle"
    """Uses the privileged layout, including hidden geometry."""

    BLIND = "blind"
    """Ignores hidden geometry: straight-line motions or nominal guesses where the oracle uses the hidden layout."""


class SpatialMemoryTaskLogic(ABC):
    """Scene parts, layout sampling, phases, success and scripted plans of one spatial-memory task.

    Instances are copied by Isaac Lab's managers, so they hold configuration only; episode state lives in
    ``SpatialMemoryEpisodeState``.

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
    """Seconds the memory targets may be seen after the instruction is revealed and before the robot commits (see
    ``commitment_made``)."""

    withholds_camera_frames: bool = False
    """Whether ``camera_blocked`` can withhold egocentric frames (the camera observation is then masked)."""

    instruction_delay_s: float = 3.5
    """Seconds the memory targets must stay out of view before the instruction is revealed (longer than the
    default 3 s frame history)."""

    robot: str = "franka"
    """Robot the task is built for: ``franka`` (which can also run on DROID) or ``g1``."""

    initial_joint_pose: tuple[float, ...] | None = None
    """Spawn positions of the Franka's 7 arm joints; None keeps Arena's ready posture."""

    rest_joint_pose: tuple[float, ...] = FRANKA_READY_JOINT_POS
    """Franka arm posture the scripted controller pulls towards in the inverse-kinematics null space."""

    def __init__(self, difficulty: float | None, embodiment: EmbodimentAdapter):
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

    def part_names(self) -> list[str]:
        """Return the names of all parts."""
        return [spec.name for spec in self.part_specs()]

    def item_names(self) -> list[str]:
        """Return the names of the movable items."""
        return [spec.name for spec in self.part_specs() if spec.kind == PartKind.ITEM]

    @abstractmethod
    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        """Sample the layout of one episode."""

    def parked_part_poses(self) -> dict[str, list[float]]:
        """Return the environment-frame pose of every part, parked out of the way, for ``sample_layout`` to fill."""
        poses = {}
        for index, spec in enumerate(self.part_specs()):
            # Boards are parked below the floor; items rest on the counter level far from the robot.
            z = 0.5 * spec.size_xyz[2] + 0.001 if spec.kind == PartKind.ITEM else PARKING_POSE_Z_M - spec.size_xyz[2]
            poses[spec.name] = [PARKING_X_M + PARKING_SPACING_M * index, PARKING_Y_M, z, 0.0, 0.0, 0.0, 1.0]
        return poses

    def part_poses_tensor(self, poses: dict[str, list[float]]) -> torch.Tensor:
        """Convert part poses keyed by name into a (K, 7) tensor ordered like ``part_specs()``."""
        return torch.tensor([poses[spec.name] for spec in self.part_specs()], dtype=torch.float32)

    # ----------------------------------------------------------------------------------------------------------------
    # Visibility and phases
    # ----------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def visibility_targets(self) -> list[VisibilityTarget]:
        """Return the task-relevant surfaces whose visibility is logged."""

    @abstractmethod
    def hidden_target_names(self) -> list[str]:
        """Return the targets that must be out of view during the hidden phase."""

    def remembered_target_names(self) -> list[str]:
        """Return the targets the robot must see before the hidden phase: what it is expected to remember."""
        return self.hidden_target_names()

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

    def adjust_camera_and_occluder_poses(self, env, episode, T_W_mount, T_W_occluders):
        """Hook to modify the camera mount and occluder poses (default: unchanged)."""
        return T_W_mount, T_W_occluders

    @abstractmethod
    def hidden_phase(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return whether each environment is in the hidden phase, during which hidden targets must stay hidden."""

    def hidden_target_mask(
        self, env, episode: SpatialMemoryEpisodeState, in_hidden_phase: torch.Tensor
    ) -> torch.Tensor:
        """Return which targets (N, T) must be out of view at this step: by default all of them, in the hidden phase.

        Tasks whose targets are hidden in different phases override it.
        """
        return in_hidden_phase.unsqueeze(-1).expand(-1, len(episode.target_names))

    def static_part_mask(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return (N, K) parts that must not move while hidden; by default every part except manipulated items."""
        manipulated = set(self.manipulated_item_names())
        mask = torch.tensor([name not in manipulated for name in episode.part_names], device=env.device)
        return mask.unsqueeze(0).expand(env.num_envs, -1)

    def manipulated_item_names(self) -> list[str]:
        """Return the items the robot is meant to move."""
        return []

    # ----------------------------------------------------------------------------------------------------------------
    # Contacts, success and observations
    # ----------------------------------------------------------------------------------------------------------------

    def allowed_contacts(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return the (N, S, K) mask of contacts between sensor bodies and parts that are part of the task.

        Sensors are ordered robot bodies, hand bodies, then items (see ``collision.filtered_sensor_labels``). Robot
        bodies may touch the parts in ``robot_touchable_part_mask`` (hand bodies also those in
        ``hand_touchable_part_names``); items may touch each other and the ``item_support_names`` boards, except the
        pairs in ``forbidden_item_contacts``. Any other contact is a collision.
        """
        labels = filtered_sensor_labels(self)
        hand_bodies = set(self.embodiment.hand_bodies)
        robot_bodies = set(self.embodiment.collision_bodies) | hand_bodies
        item_contact_parts = set(self.item_names()) | set(self.item_support_names())
        forbidden_pairs = {frozenset(pair) for pair in self.forbidden_item_contacts()}
        hand_touchable_ids = episode.part_ids(self.hand_touchable_part_names())
        robot_touchable = self.robot_touchable_part_mask(env, episode)
        allowed = torch.zeros(env.num_envs, len(labels), len(episode.part_names), dtype=torch.bool, device=env.device)
        for sensor_index, label in enumerate(labels):
            if label in robot_bodies:
                allowed[:, sensor_index] = robot_touchable
                if label in hand_bodies:
                    allowed[:, sensor_index, hand_touchable_ids] = True
                continue
            for part_index, part in enumerate(episode.part_names):
                if part in item_contact_parts and frozenset((label, part)) not in forbidden_pairs:
                    allowed[:, sensor_index, part_index] = True
        return allowed

    def robot_touchable_part_mask(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return which parts (N, K) the robot may touch: by default the manipulated items."""
        manipulated = set(self.manipulated_item_names())
        mask = torch.tensor([name in manipulated for name in episode.part_names], device=env.device)
        return mask.unsqueeze(0).expand(env.num_envs, -1).clone()

    def hand_touchable_part_names(self) -> list[str]:
        """Return parts the hand bodies may also touch, such as a table an item is placed on."""
        return []

    def forbidden_item_contacts(self) -> list[tuple[str, str]]:
        """Return pairs of items that must not touch each other."""
        return []

    def item_support_names(self) -> list[str]:
        """Return boards items may rest on or slide against (start and goal supports)."""
        return []

    def collisions_counted(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return whether collisions count as failures at this step (N,)."""
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    @abstractmethod
    def success_terms(self, env, episode: SpatialMemoryEpisodeState) -> dict[str, torch.Tensor]:
        """Return the named conditions (each (N,)) that together make up task success.

        Collision and searched-again are handled separately. The final value of each term is recorded per episode.
        """

    def success(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return the task-specific success condition (N,): all ``success_terms`` hold."""
        success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for term in self.success_terms(env, episode).values():
            success &= term
        return success

    def extra_failures(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return additional task-specific failure flags (N,)."""
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def instruction_dim(self) -> int:
        """Return the size of the instruction observation (0: no instruction term)."""
        return 0

    def instruction(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return the instruction observation (N, instruction_dim); zeros until revealed."""
        return torch.zeros(env.num_envs, self.instruction_dim(), device=env.device)

    def camera_blocked(self, env, episode: SpatialMemoryEpisodeState) -> torch.Tensor:
        """Return whether the egocentric image is withheld at this step (N,)."""
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def update_frame_withholding(
        self, env, episode: SpatialMemoryEpisodeState, visible: torch.Tensor, active: torch.Tensor
    ) -> None:
        """Hook called once per control step, before ``camera_blocked``, with the unmasked visibility (N, T).

        Only called when ``withholds_camera_frames`` is set.
        """

    # ----------------------------------------------------------------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------------------------------------------------------------

    def on_reset(self, env, episode: SpatialMemoryEpisodeState, env_ids: torch.Tensor) -> None:
        """Hook called after a layout is written."""

    def on_step(self, env, episode: SpatialMemoryEpisodeState, visible: torch.Tensor, active: torch.Tensor) -> None:
        """Hook called once per control step after visibility, phases and collisions are updated.

        The default implementation runs the memory probe when ``memory_target_names`` is not empty.
        """
        if self.memory_target_names():
            self.update_memory_probe(env, episode, visible, active)

    def update_memory_probe(
        self, env, episode: SpatialMemoryEpisodeState, visible: torch.Tensor, active: torch.Tensor
    ) -> None:
        """Reveal the instruction once the memory targets were seen and then stayed out of view for a while.

        Between the reveal and ``commitment_made``, the steps in which a memory target is visible are counted;
        exceeding ``reobservation_budget_s`` marks the episode as "searched again".
        """
        indices = [episode.target_names.index(name) for name in self.memory_target_names()]
        memory_visible = visible[:, indices].any(dim=-1) & active
        episode.memory_seen |= memory_visible
        episode.memory_unseen_steps = torch.where(
            memory_visible, torch.zeros_like(episode.memory_unseen_steps), episode.memory_unseen_steps + 1
        )
        delay_steps = int(math.ceil(self.instruction_delay_s / episode.step_dt))
        newly_revealed = episode.memory_seen & (episode.memory_unseen_steps >= delay_steps) & active
        newly_revealed &= ~episode.instruction_revealed
        episode.instruction_reveal_step[newly_revealed] = env.episode_length_buf[newly_revealed]
        episode.instruction_revealed |= newly_revealed
        before_commitment = episode.instruction_revealed & ~newly_revealed & ~self.commitment_made(env, episode, active)
        episode.reobservation_steps += (before_commitment & memory_visible).long()
        budget_steps = int(math.floor(self.reobservation_budget_s / episode.step_dt))
        episode.searched_again |= episode.reobservation_steps > budget_steps

    def commitment_made(self, env, episode: SpatialMemoryEpisodeState, active: torch.Tensor) -> torch.Tensor:
        """Return whether the robot has committed to its answer (N,): by default, it has closed the gripper.

        Called once per control step by ``update_memory_probe``.
        """
        return episode.first_close_step != NOT_SET

    def released_after_instruction(self, env, episode: SpatialMemoryEpisodeState, active: torch.Tensor) -> torch.Tensor:
        """Return whether the gripper has opened since the instruction was revealed (N,); call once per control step."""
        was_closed = episode.task_state_buffer("was_closed", False, torch.bool)
        released = episode.task_state_buffer("released_after_instruction", False, torch.bool)
        closed = self.gripper_closed(env) & active
        released |= was_closed & ~closed & episode.instruction_revealed
        was_closed[:] = closed
        return released

    # ----------------------------------------------------------------------------------------------------------------
    # Embodiment helpers and scripted plans
    # ----------------------------------------------------------------------------------------------------------------

    def tool_pose_w(self, env) -> torch.Tensor:
        """Return the tool centre point pose (N, 7) in the world frame (the G1's pelvis for the G1)."""
        return self.embodiment.tracked_pose_w(env)

    def gripper_closed(self, env) -> torch.Tensor:
        """Return whether the gripper is commanded closed (N,)."""
        return self.embodiment.gripper_closed(env)

    @abstractmethod
    def plan(self, env, episode: SpatialMemoryEpisodeState, env_id: int, mode: PlanMode) -> list:
        """Return the scripted plan of one environment: Franka ``Waypoint`` or ``G1Waypoint`` objects."""

    def plan_gates(self, env, episode: SpatialMemoryEpisodeState) -> dict[str, torch.Tensor]:
        """Return the episode conditions plans may wait on, keyed by gate name."""
        return {"instruction_revealed": episode.instruction_revealed.clone()}
