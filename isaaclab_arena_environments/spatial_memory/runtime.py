# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-environment runtime state of a spatial-memory task: layout, phases, visibility, collisions and metrics.

Isaac Lab managers deep-copy their term parameters, so the state cannot live on a term or on the task logic. It is
kept in a registry keyed weakly by the environment and created on the first layout reset.
"""

from __future__ import annotations

import torch
import weakref
from typing import TYPE_CHECKING

from isaaclab_arena_environments.spatial_memory.geometry import compose_poses
from isaaclab_arena_environments.spatial_memory.parts import PartKind

if TYPE_CHECKING:
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic

_RUNTIMES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

NOT_SET = -1
"""Step index stored before an event has happened."""


class SpatialMemoryRuntime:
    """Mutable per-environment state shared by the suite's event, termination, observation and recorder terms.

    Args:
        env: The (unwrapped) Arena environment.
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
        self.params: dict[str, torch.Tensor] = {}
        # Visibility.
        self.visible_points = torch.zeros(self.num_envs, num_targets, dtype=torch.long, device=self.device)
        self.visible_steps = torch.zeros(self.num_envs, num_targets, dtype=torch.long, device=self.device)
        self.visible_steps_in_action = torch.zeros_like(self.visible_steps)
        self.visible_before_action = torch.zeros(self.num_envs, num_targets, dtype=torch.bool, device=self.device)
        # Phases.
        self.in_action_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.action_phase_start_step = torch.full((self.num_envs,), NOT_SET, dtype=torch.long, device=self.device)
        self.action_phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.static_reference_poses_w = torch.zeros(self.num_envs, num_parts, 7, device=self.device)
        self.static_max_displacement_m = torch.zeros(self.num_envs, device=self.device)
        # Memory probe: instruction reveal and searched-again.
        self.memory_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.memory_unseen_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.revealed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reveal_step = torch.full((self.num_envs,), NOT_SET, dtype=torch.long, device=self.device)
        self.reveal_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
        self.first_collision_position_e = torch.zeros(self.num_envs, 3, device=self.device)
        # Motion.
        self.path_length_m = torch.zeros(self.num_envs, device=self.device)
        self.previous_tool_position_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.success_step = torch.full((self.num_envs,), NOT_SET, dtype=torch.long, device=self.device)
        # Task-specific state, owned by the task logic.
        self.task_state: dict[str, torch.Tensor] = {}

    def part_ids(self, names: list[str]) -> list[int]:
        """Return the indices of the named parts."""
        return [self.part_index[name] for name in names]

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear the bookkeeping of the given environments (the layout is written separately)."""
        for name in (
            "visible_points",
            "visible_steps",
            "visible_steps_in_action",
            "visible_before_action",
            "in_action_phase",
            "action_phase_steps",
            "static_max_displacement_m",
            "memory_seen",
            "memory_unseen_steps",
            "revealed",
            "reveal_count",
            "reobservation_steps",
            "searched_again",
            "gripper_closed",
            "collided",
            "collision_steps",
            "path_length_m",
        ):
            getattr(self, name)[env_ids] = 0
        for name in (
            "action_phase_start_step",
            "reveal_step",
            "first_close_step",
            "first_collision_step",
            "success_step",
        ):
            getattr(self, name)[env_ids] = NOT_SET
        for env_id in env_ids.tolist():
            self.first_collision_body[env_id] = ""
            self.first_collision_partner[env_id] = ""
        self.first_collision_position_e[env_ids] = 0.0

    def get_part_poses_w(self, env) -> torch.Tensor:
        """Return the live world poses (N, K, 7) of every scene part."""
        poses = [env.arena_world.get_pose_w(name) for name in self.part_names]
        return torch.stack(poses, dim=1)

    def get_robot_body_poses_w(self, env, body_names: list[str]) -> torch.Tensor:
        """Return the live world poses (N, B, 7) of robot bodies."""
        robot = env.scene["robot"]
        body_poses = robot.data.body_link_pose_w.torch
        indices = [robot.data.body_names.index(name) for name in body_names]
        return body_poses[:, indices]


def get_runtime(env, logic: SpatialMemoryTaskLogic | None = None) -> SpatialMemoryRuntime:
    """Return the runtime of ``env``, creating it from ``logic`` on first use."""
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    runtime = _RUNTIMES.get(env)
    if runtime is None:
        assert logic is not None, "The spatial-memory runtime is created by the layout reset event."
        runtime = SpatialMemoryRuntime(env, logic)
        _RUNTIMES[env] = runtime
    return runtime


def _env_seed(env) -> int:
    seed = getattr(env.cfg, "seed", None)
    return 0 if seed is None else int(seed)


def layout_generator(env, env_id: int) -> torch.Generator:
    """Return a CPU generator seeded by the environment seed, the env index and its episode index.

    Layouts therefore depend only on (seed, env_id, episode_index), not on how many other environments run or on
    other consumers of the global torch random state.
    """
    episode_index = env.get_episode_index(env_id) if hasattr(env, "get_episode_index") else 0
    generator = torch.Generator()
    generator.manual_seed((_env_seed(env) * 1_000_003 + env_id * 7_919 + episode_index * 104_729) % (2**63 - 1))
    return generator


def reset_layout(env, env_ids: torch.Tensor | None, logic: SpatialMemoryTaskLogic) -> None:
    """Reset event: sample a layout per environment and write every part's pose."""
    runtime = get_runtime(env, logic)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(dtype=torch.long)
    samples = [logic.sample_layout(layout_generator(env, int(env_id))) for env_id in env_ids.tolist()]
    poses_e = torch.stack([sample.part_poses_e for sample in samples]).to(env.device)
    runtime.layout_poses_e[env_ids] = poses_e
    for key in samples[0].params:
        values = torch.stack([sample.params[key] for sample in samples]).to(env.device)
        if key not in runtime.params:
            runtime.params[key] = torch.zeros((env.num_envs, *values.shape[1:]), dtype=values.dtype, device=env.device)
        runtime.params[key][env_ids] = values
    write_part_poses(env, runtime, env_ids, poses_e)
    runtime.reset(env_ids)
    logic.on_reset(env, runtime, env_ids)


def write_part_poses(env, runtime: SpatialMemoryRuntime, env_ids: torch.Tensor, poses_e: torch.Tensor) -> None:
    """Write environment-frame part poses (len(env_ids), K, 7) to the simulation and zero their velocities."""
    origins = env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)
    for part_index, name in enumerate(runtime.part_names):
        pose_w = poses_e[:, part_index].clone()
        pose_w[:, :3] += origins
        asset = env.scene[name]
        asset.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def tool_position_w(env, runtime: SpatialMemoryRuntime) -> torch.Tensor:
    """Return the world position (N, 3) of the embodiment's tool point."""
    T_W_tool = runtime.logic.tool_pose_w(env)
    return T_W_tool[:, :3]


def update_runtime(env, logic: SpatialMemoryTaskLogic) -> SpatialMemoryRuntime:
    """Advance the runtime once per control step: visibility, phases, collisions and path length.

    Called from the suite's failure termination term, which runs after physics and before resets. Repeated calls
    within one control step are ignored.
    """
    runtime = get_runtime(env, logic)
    if runtime._last_update_counter == env.common_step_counter:
        return runtime
    runtime._last_update_counter = env.common_step_counter
    # Terminations run after the episode counter is incremented, so the first update of an episode sees step 1.
    episode_step = env.episode_length_buf
    active = episode_step > 0

    T_W_parts = runtime.get_part_poses_w(env)
    # Visibility.
    model = runtime.visibility_model
    T_W_mount = runtime.get_robot_body_poses_w(env, [model.camera.mount_body])[:, 0]
    occluder_names = [occluder.body_name for occluder in model.robot_occluders]
    T_W_occluders = (
        runtime.get_robot_body_poses_w(env, occluder_names)
        if occluder_names
        else torch.zeros(env.num_envs, 0, 7, device=env.device)
    )
    T_W_mount, T_W_occluders = logic.adjust_camera_and_occluder_poses(env, runtime, T_W_mount, T_W_occluders)
    counts = model.compute(T_W_mount, T_W_parts, T_W_occluders)
    visible = counts > 0
    runtime.visible_points[:] = counts
    runtime.visible_steps += (visible & active.unsqueeze(-1)).long()

    # Action phase and static-while-hidden.
    in_action = logic.action_phase(env, runtime) & active
    starting = in_action & (runtime.action_phase_start_step == NOT_SET)
    if starting.any():
        runtime.action_phase_start_step[starting] = episode_step[starting]
        runtime.static_reference_poses_w[starting] = T_W_parts[starting]
    runtime.in_action_phase[:] = in_action
    runtime.action_phase_steps += in_action.long()
    runtime.visible_steps_in_action += (visible & in_action.unsqueeze(-1)).long()
    before_action = active & (runtime.action_phase_start_step == NOT_SET)
    runtime.visible_before_action |= visible & before_action.unsqueeze(-1)
    if in_action.any():
        static_mask = logic.static_part_mask(env, runtime)
        displacement = (T_W_parts[..., :3] - runtime.static_reference_poses_w[..., :3]).norm(dim=-1)
        displacement = torch.where(static_mask, displacement, torch.zeros_like(displacement))
        max_displacement = displacement.amax(dim=-1)
        runtime.static_max_displacement_m[:] = torch.where(
            in_action,
            torch.maximum(runtime.static_max_displacement_m, max_displacement),
            runtime.static_max_displacement_m,
        )

    # Gripper state.
    closed = logic.gripper_closed(env) & active
    first_close = closed & (runtime.first_close_step == NOT_SET)
    runtime.first_close_step[first_close] = episode_step[first_close]
    runtime.gripper_closed[:] = closed

    # Collisions.
    from isaaclab_arena_environments.spatial_memory.collision import detect_collisions

    detect_collisions(env, runtime, active)

    # Tool path length.
    tool_w = tool_position_w(env, runtime)
    step_length = (tool_w - runtime.previous_tool_position_w).norm(dim=-1)
    # The first update after a reset only records the starting point.
    accumulate = episode_step > 1
    runtime.path_length_m += torch.where(accumulate, step_length, torch.zeros_like(step_length))
    runtime.previous_tool_position_w[:] = tool_w

    logic.on_step(env, runtime, visible, active)
    return runtime


def compose_tool_pose(T_W_body: torch.Tensor, T_body_tool: tuple[float, ...]) -> torch.Tensor:
    """Return the tool pose (N, 7) from its body pose and a fixed offset."""
    offset = torch.tensor(T_body_tool, device=T_W_body.device, dtype=T_W_body.dtype).expand(T_W_body.shape[0], 7)
    return compose_poses(T_W_body, offset)
