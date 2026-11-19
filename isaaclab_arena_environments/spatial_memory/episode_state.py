# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-environment episode state of a spatial-memory task: layout, phases, visibility, collisions and metrics.

Isaac Lab managers deep-copy their term parameters, so the state cannot live on a term or on the task logic. It is
kept in a registry keyed weakly by the environment and created on the first layout reset.
"""

from __future__ import annotations

import torch
import weakref
from typing import TYPE_CHECKING

from isaaclab_arena.terms.events import write_root_velocity_to_sim
from isaaclab_arena.utils.random import derive_seed
from isaaclab_arena_environments.spatial_memory.collision import (
    CONTACT_FORCE_THRESHOLD_N,
    filtered_sensor_labels,
    filtered_sensor_name,
)
from isaaclab_arena_environments.spatial_memory.parts import PartKind

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic

_EPISODE_STATES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

NOT_SET = -1
"""Step index stored before an event has happened."""


class SpatialMemoryEpisodeState:
    """Mutable per-environment state shared by the suite's event, termination, observation and recorder terms.

    Args:
        env: The unwrapped Arena environment.
        logic: Task logic describing parts, layouts and phases.
    """

    def __init__(self, env, logic: SpatialMemoryTaskLogic):
        self.logic = logic
        self.num_envs = env.num_envs
        self.device = env.device
        self.step_dt = env.step_dt
        specs = logic.part_specs()
        self.part_names = [spec.name for spec in specs]
        self.part_index = {name: index for index, name in enumerate(self.part_names)}
        self.item_mask = torch.tensor([spec.kind == PartKind.ITEM for spec in specs], device=self.device)
        self.visibility_model = logic.make_visibility_model(env)
        self.target_names = self.visibility_model.target_names
        self.hidden_target_mask = torch.tensor(
            [name in logic.hidden_target_names() for name in self.target_names], device=self.device, dtype=torch.bool
        )
        num_parts = len(self.part_names)
        num_targets = len(self.target_names)
        self._last_update_counter = NOT_SET
        # Layout.
        self.layout_poses_e = torch.zeros(self.num_envs, num_parts, 7, device=self.device)
        self.layout_poses_e[..., 6] = 1.0
        self.layout_params: dict[str, torch.Tensor] = {}
        # Visibility.
        self.visible_points = torch.zeros(self.num_envs, num_targets, dtype=torch.long, device=self.device)
        self.visible_steps = torch.zeros(self.num_envs, num_targets, dtype=torch.long, device=self.device)
        self.visible_steps_in_hidden_phase = torch.zeros_like(self.visible_steps)
        self.visible_before_hidden_phase = torch.zeros(self.num_envs, num_targets, dtype=torch.bool, device=self.device)
        # Phases.
        self.in_hidden_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hidden_phase_start_step = torch.full((self.num_envs,), NOT_SET, dtype=torch.long, device=self.device)
        self.hidden_phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.T_W_parts_at_hidden_phase_start = torch.zeros(self.num_envs, num_parts, 7, device=self.device)
        self.static_max_displacement_m = torch.zeros(self.num_envs, device=self.device)
        # Memory probe: instruction reveal and searched-again.
        self.memory_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.memory_unseen_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.instruction_revealed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.instruction_reveal_step = torch.full((self.num_envs,), NOT_SET, dtype=torch.long, device=self.device)
        self.reobservation_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.searched_again = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gripper_closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.first_close_step = torch.full((self.num_envs,), NOT_SET, dtype=torch.long, device=self.device)
        # Collisions.
        self.collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collision_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.first_collision_step = torch.full((self.num_envs,), NOT_SET, dtype=torch.long, device=self.device)
        self.first_collision_body = [""] * self.num_envs
        self.first_collision_partner = [""] * self.num_envs
        self.first_collision_body_position_e = torch.zeros(self.num_envs, 3, device=self.device)
        # Motion.
        self.tracked_path_length_m = torch.zeros(self.num_envs, device=self.device)
        self.previous_t_W_tracked = torch.zeros(self.num_envs, 3, device=self.device)
        # Task-specific state, owned by the task logic.
        self.task_state: dict[str, torch.Tensor] = {}
        self._task_state_fill_values: dict[str, object] = {}

    def part_ids(self, names: list[str]) -> list[int]:
        """Return the indices of the named parts."""
        return [self.part_index[name] for name in names]

    def task_state_buffer(self, name: str, fill_value, dtype: torch.dtype, shape: tuple[int, ...] = ()) -> torch.Tensor:
        """Return the task-owned buffer ``name`` of shape (N, *shape), created and reset to ``fill_value``."""
        if name not in self.task_state:
            self.task_state[name] = torch.full((self.num_envs, *shape), fill_value, dtype=dtype, device=self.device)
            self._task_state_fill_values[name] = fill_value
        return self.task_state[name]

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear the bookkeeping of the given environments; the layout is written separately."""
        for name in (
            "visible_points",
            "visible_steps",
            "visible_steps_in_hidden_phase",
            "visible_before_hidden_phase",
            "in_hidden_phase",
            "hidden_phase_steps",
            "static_max_displacement_m",
            "memory_seen",
            "memory_unseen_steps",
            "instruction_revealed",
            "reobservation_steps",
            "searched_again",
            "gripper_closed",
            "collided",
            "collision_steps",
            "tracked_path_length_m",
        ):
            getattr(self, name)[env_ids] = 0
        for name in ("hidden_phase_start_step", "instruction_reveal_step", "first_close_step", "first_collision_step"):
            getattr(self, name)[env_ids] = NOT_SET
        for env_id in env_ids.tolist():
            self.first_collision_body[env_id] = ""
            self.first_collision_partner[env_id] = ""
        self.first_collision_body_position_e[env_ids] = 0.0
        for name, fill_value in self._task_state_fill_values.items():
            self.task_state[name][env_ids] = fill_value

    def get_part_poses_w(self, env) -> torch.Tensor:
        """Return the live world poses (N, K, 7) of every scene part."""
        return torch.stack([env.arena_world.get_pose_w(name) for name in self.part_names], dim=1)

    def get_robot_body_poses_w(self, env, body_names: list[str]) -> torch.Tensor:
        """Return the live world poses (N, B, 7) of robot bodies."""
        robot = env.scene["robot"]
        indices = [robot.data.body_names.index(name) for name in body_names]
        return robot.data.body_link_pose_w.torch[:, indices]


def get_episode_state(env, logic: SpatialMemoryTaskLogic | None = None) -> SpatialMemoryEpisodeState:
    """Return the episode state of ``env``, creating it from ``logic`` on first use."""
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    episode = _EPISODE_STATES.get(env)
    if episode is None:
        assert logic is not None, "The spatial-memory episode state is created by the layout reset event."
        episode = SpatialMemoryEpisodeState(env, logic)
        _EPISODE_STATES[env] = episode
    return episode


def layout_generator(env, env_id: int) -> torch.Generator:
    """Return a CPU generator seeded by the environment seed, the env index and its episode index.

    Layouts therefore depend only on (seed, env_id, episode_index), not on how many other environments run or on
    other consumers of the global torch random state.
    """
    seed = env.cfg.seed if env.cfg.seed is not None else 0
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, env_id, env.get_episode_index(env_id)))
    return generator


def reset_layout(env, env_ids: torch.Tensor | None, logic: SpatialMemoryTaskLogic) -> None:
    """Reset event: sample a layout per environment and write every part's pose."""
    episode = get_episode_state(env, logic)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long)
    samples = [logic.sample_layout(layout_generator(env, env_id)) for env_id in env_ids.tolist()]
    T_E_parts = torch.stack([sample.part_poses_e for sample in samples]).to(env.device)
    episode.layout_poses_e[env_ids] = T_E_parts
    for key in samples[0].layout_params:
        values = torch.stack([sample.layout_params[key] for sample in samples]).to(env.device)
        if key not in episode.layout_params:
            shape = (env.num_envs, *values.shape[1:])
            episode.layout_params[key] = torch.zeros(shape, dtype=values.dtype, device=env.device)
        episode.layout_params[key][env_ids] = values
    write_part_poses(env, episode.part_names, env_ids, T_E_parts)
    episode.reset(env_ids)
    logic.on_reset(env, episode, env_ids)


def write_part_poses(env, part_names: list[str], env_ids: torch.Tensor, T_E_parts: torch.Tensor) -> None:
    """Write environment-frame poses (len(env_ids), K, 7) of the named parts and zero their velocities."""
    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)
    for part_index, name in enumerate(part_names):
        T_W_part = T_E_parts[:, part_index].clone()
        T_W_part[:, :3] += env.scene.env_origins[env_ids]
        asset = env.scene[name]
        asset.write_root_pose_to_sim(T_W_part, env_ids=env_ids)
        write_root_velocity_to_sim(env, asset, zero_velocity, env_ids)


def update_episode_state(env, logic: SpatialMemoryTaskLogic) -> SpatialMemoryEpisodeState:
    """Advance the episode state once per control step: visibility, phases, collisions and path length.

    Runs from the suite's termination terms, after physics and before resets; repeated calls within one control step
    return the state unchanged.
    """
    episode = get_episode_state(env, logic)
    if episode._last_update_counter == env.common_step_counter:
        return episode
    episode._last_update_counter = env.common_step_counter
    # Terminations run after the episode counter is incremented, so the first update of an episode sees step 1.
    active = env.episode_length_buf > 0
    T_W_parts = episode.get_part_poses_w(env)
    visible = _update_visibility(env, logic, episode, T_W_parts, active)
    _update_hidden_phase(env, logic, episode, T_W_parts, visible, active)
    _update_gripper(env, logic, episode, active)
    _update_collisions(env, episode, active)
    _update_tracked_path_length(env, logic, episode)
    logic.on_step(env, episode, visible, active)
    return episode


def _update_visibility(env, logic, episode, T_W_parts: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Count the visible points of every target and return which targets are visible (N, T)."""
    model = episode.visibility_model
    T_W_mount = episode.get_robot_body_poses_w(env, [model.camera.mount_body])[:, 0]
    occluder_names = [occluder.body_name for occluder in model.robot_occluders]
    if occluder_names:
        T_W_occluders = episode.get_robot_body_poses_w(env, occluder_names)
    else:
        T_W_occluders = torch.zeros(env.num_envs, 0, 7, device=env.device)
    T_W_mount, T_W_occluders = logic.adjust_camera_and_occluder_poses(env, episode, T_W_mount, T_W_occluders)
    counts = model.count_visible_points(T_W_mount, T_W_parts, T_W_occluders)
    episode.visible_points[:] = counts
    visible = counts > 0
    if logic.withholds_camera_frames:
        # A withheld frame shows nothing, so nothing counts as seen in it.
        logic.update_frame_withholding(env, episode, visible, active)
        visible = visible & ~logic.camera_blocked(env, episode).unsqueeze(-1)
    episode.visible_steps += (visible & active.unsqueeze(-1)).long()
    return visible


def _update_hidden_phase(env, logic, episode, T_W_parts, visible: torch.Tensor, active: torch.Tensor) -> None:
    """Track the hidden phase, what was seen before and during it, and how far static parts moved during it."""
    in_hidden_phase = logic.hidden_phase(env, episode) & active
    starting = in_hidden_phase & (episode.hidden_phase_start_step == NOT_SET)
    episode.hidden_phase_start_step[starting] = env.episode_length_buf[starting]
    episode.T_W_parts_at_hidden_phase_start[starting] = T_W_parts[starting]
    episode.in_hidden_phase[:] = in_hidden_phase
    episode.hidden_phase_steps += in_hidden_phase.long()
    episode.visible_steps_in_hidden_phase += (visible & logic.hidden_target_mask(env, episode, in_hidden_phase)).long()
    before_hidden_phase = active & (episode.hidden_phase_start_step == NOT_SET)
    episode.visible_before_hidden_phase |= visible & before_hidden_phase.unsqueeze(-1)
    displacement = (T_W_parts[..., :3] - episode.T_W_parts_at_hidden_phase_start[..., :3]).norm(dim=-1)
    displacement = torch.where(logic.static_part_mask(env, episode), displacement, torch.zeros_like(displacement))
    max_displacement = torch.where(in_hidden_phase, displacement.amax(dim=-1), torch.zeros_like(displacement[:, 0]))
    episode.static_max_displacement_m[:] = torch.maximum(episode.static_max_displacement_m, max_displacement)


def _update_gripper(env, logic, episode, active: torch.Tensor) -> None:
    """Record the commanded gripper state and the step of the first close."""
    closed = logic.gripper_closed(env) & active
    first_close = closed & (episode.first_close_step == NOT_SET)
    episode.first_close_step[first_close] = env.episode_length_buf[first_close]
    episode.gripper_closed[:] = closed


def _update_tracked_path_length(env, logic, episode) -> None:
    """Accumulate the path length of the embodiment's tracked body (tool centre point, or the G1 pelvis)."""
    t_W_tracked = logic.embodiment.tracked_pose_w(env)[:, :3]
    step_length = (t_W_tracked - episode.previous_t_W_tracked).norm(dim=-1)
    # The first update after a reset only records the starting point.
    accumulate = env.episode_length_buf > 1
    episode.tracked_path_length_m += torch.where(accumulate, step_length, torch.zeros_like(step_length))
    episode.previous_t_W_tracked[:] = t_W_tracked


def _update_collisions(env, episode: SpatialMemoryEpisodeState, active: torch.Tensor) -> None:
    """Record this step's collisions from the contact sensors: flags, step counts and the first collision."""
    logic = episode.logic
    colliding = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    body_labels: list[list[str]] = [[] for _ in range(env.num_envs)]
    partner_labels: list[list[str]] = [[] for _ in range(env.num_envs)]
    t_W_bodies = torch.zeros(env.num_envs, 3, device=env.device)

    labels = filtered_sensor_labels(logic)
    allowed = logic.allowed_contacts(env, episode)  # (N, S, K)
    for sensor_index, label in enumerate(labels):
        sensor = env.scene[filtered_sensor_name(label)]
        forces = sensor.data.force_matrix_w.torch[:, 0].norm(dim=-1)  # (N, K)
        touching = (forces > CONTACT_FORCE_THRESHOLD_N) & ~allowed[:, sensor_index]
        any_touching = touching.any(dim=-1)
        if any_touching.any():
            strongest = torch.where(touching, forces, torch.zeros_like(forces)).argmax(dim=-1)
            for env_id in torch.nonzero(any_touching).flatten().tolist():
                body_labels[env_id].append(label)
                partner_labels[env_id].append(episode.part_names[int(strongest[env_id])])
                if label in episode.part_index:
                    t_W_bodies[env_id] = env.arena_world.get_pose_w(label)[env_id, :3]
                else:
                    # A contact label may name a nested prim; its body is the prim path's last element.
                    body_name = logic.embodiment.body_prim_paths.get(label, label).split("/")[-1]
                    t_W_bodies[env_id] = episode.get_robot_body_poses_w(env, [body_name])[env_id, 0, :3]
        colliding |= any_touching

    colliding &= active & logic.collisions_counted(env, episode)
    first = colliding & (episode.first_collision_step == NOT_SET)
    for env_id in torch.nonzero(first).flatten().tolist():
        episode.first_collision_step[env_id] = int(env.episode_length_buf[env_id])
        episode.first_collision_body[env_id] = body_labels[env_id][0]
        episode.first_collision_partner[env_id] = partner_labels[env_id][0]
        episode.first_collision_body_position_e[env_id] = t_W_bodies[env_id] - env.scene.env_origins[env_id]
    episode.collision_steps += colliding.long()
    episode.collided |= colliding
