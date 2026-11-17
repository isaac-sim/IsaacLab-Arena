# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation variants of the spatial-memory suite.

* ``current``: the egocentric image and proprioception of the current step (plus the task instruction).
* ``history_k``: the same terms with the last ``k`` steps stacked (Isaac Lab observation history).
* ``oracle_map``: ``current`` plus privileged scene geometry in the robot base frame, hidden parts included.
* ``mapped``: ``current`` plus the output of an online map provider (a placeholder until nvblox-next is plugged in).
"""

from __future__ import annotations

import math
import torch
import weakref
from enum import Enum
from typing import Protocol

import isaaclab.utils.math as math_utils

from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES, sample_box_face_points
from isaaclab_arena_environments.spatial_memory.runtime import get_runtime

DEFAULT_HISTORY_S = 3.0
"""Seconds of frames that ``history_k`` stacks by default."""

ORACLE_MAP_SPACING_M = 0.03
"""Surface sample spacing of the oracle point cloud."""

ONLINE_MAP_MAX_POINTS = 4096
"""Fixed number of points returned by the ``mapped`` observation."""


class ObservationVariant(str, Enum):
    """Observation variants selectable through the environment config."""

    CURRENT = "current"
    HISTORY_K = "history_k"
    ORACLE_MAP = "oracle_map"
    MAPPED = "mapped"

    def __str__(self) -> str:
        return self.value


def default_history_length(step_dt: float, history_s: float = DEFAULT_HISTORY_S) -> int:
    """Return the number of control steps that covers ``history_s`` seconds."""
    return int(math.ceil(history_s / step_dt - 1e-9))


def _robot_root_pose_w(env) -> torch.Tensor:
    return env.scene["robot"].data.root_pose_w.torch


def instruction(env, logic) -> torch.Tensor:
    """Task instruction (for example a one-hot colour); zeros until the task reveals it."""
    runtime = get_runtime(env, logic)
    if not runtime.params:
        # The observation manager evaluates terms once before the first layout is sampled.
        return torch.zeros(env.num_envs, logic.instruction_dim(), device=env.device)
    return logic.instruction(env, runtime)


def oracle_map_points(env, logic) -> torch.Tensor:
    """Privileged surface point cloud of every scene part in the robot base frame.

    Returns:
        (N, P, 4) tensor: x, y, z in the robot root frame and the part index (see the task's part list).
    """
    runtime = get_runtime(env, logic)
    cache = runtime.task_state
    if "oracle_points_local" not in cache:
        local_chunks = []
        part_chunks = []
        for part_index, spec in enumerate(logic.part_specs()):
            points = sample_box_face_points(
                spec.size_xyz, ALL_FACES, spacing_m=ORACLE_MAP_SPACING_M, outward_offset_m=0.0
            )
            local_chunks.append(points)
            part_chunks.append(torch.full((points.shape[0],), part_index))
        cache["oracle_points_local"] = torch.cat(local_chunks).to(env.device)
        cache["oracle_points_part"] = torch.cat(part_chunks).to(env.device)
    points_local = cache["oracle_points_local"]
    point_part = cache["oracle_points_part"]
    T_W_parts = runtime.get_part_poses_w(env)[:, point_part]
    points_W = math_utils.quat_apply(T_W_parts[..., 3:7], points_local.unsqueeze(0)) + T_W_parts[..., :3]
    T_W_root = _robot_root_pose_w(env)
    points_root = math_utils.quat_apply_inverse(T_W_root[:, 3:7].unsqueeze(1), points_W - T_W_root[:, None, :3])
    labels = point_part.to(points_root.dtype).unsqueeze(0).unsqueeze(-1).expand(env.num_envs, -1, 1)
    return torch.cat([points_root, labels], dim=-1)


def oracle_map_boxes(env, logic) -> torch.Tensor:
    """Privileged box geometry of every scene part in the robot base frame.

    Returns:
        (N, K, 10) tensor: centre (3), orientation x, y, z, w (4) and half extents (3) per part.
    """
    runtime = get_runtime(env, logic)
    T_W_parts = runtime.get_part_poses_w(env)
    T_W_root = _robot_root_pose_w(env).unsqueeze(1).expand_as(T_W_parts)
    t_root_parts, q_root_parts = math_utils.subtract_frame_transforms(
        T_W_root[..., :3].reshape(-1, 3),
        T_W_root[..., 3:7].reshape(-1, 4),
        T_W_parts[..., :3].reshape(-1, 3),
        T_W_parts[..., 3:7].reshape(-1, 4),
    )
    half = torch.tensor([spec.half_extents for spec in logic.part_specs()], device=env.device)
    half = half.unsqueeze(0).expand(env.num_envs, -1, -1)
    poses = torch.cat([t_root_parts, q_root_parts], dim=-1).reshape(env.num_envs, -1, 7)
    return torch.cat([poses, half], dim=-1)


class OnlineMapProvider(Protocol):
    """Interface for an online reconstruction that feeds the ``mapped`` observation variant."""

    def reset(self, env, env_ids: torch.Tensor) -> None:
        """Forget the map of the given environments."""

    def query(self, env) -> torch.Tensor:
        """Return (N, ONLINE_MAP_MAX_POINTS, 4) points in the robot base frame: x, y, z and a confidence in [0, 1]."""


class EmptyMapProvider:
    """Placeholder provider that returns an empty map (all confidences zero).

    TODO(spatial-memory): replace with the nvblox-next surfel map. The provider should integrate the egocentric depth
    and pose each step, and return the surfel centres in the robot base frame with their confidence. Register it with
    ``set_online_map_provider(env, provider)`` after the environment is created.
    """

    def reset(self, env, env_ids: torch.Tensor) -> None:
        pass

    def query(self, env) -> torch.Tensor:
        return torch.zeros(env.num_envs, ONLINE_MAP_MAX_POINTS, 4, device=env.device)


_MAP_PROVIDERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def set_online_map_provider(env, provider: OnlineMapProvider) -> None:
    """Install the online map provider used by the ``mapped`` observation of ``env``."""
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    _MAP_PROVIDERS[env] = provider


def get_online_map_provider(env) -> OnlineMapProvider:
    """Return the online map provider of ``env`` (an ``EmptyMapProvider`` unless one was installed)."""
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    if env not in _MAP_PROVIDERS:
        _MAP_PROVIDERS[env] = EmptyMapProvider()
    return _MAP_PROVIDERS[env]


def online_map(env) -> torch.Tensor:
    """Online reconstruction from the installed map provider (empty by default)."""
    return get_online_map_provider(env).query(env)


def _make_masked_camera_image_term():
    """Return the class-based observation term that withholds camera frames (Isaac Lab imports are deferred)."""
    from isaaclab.managers import ManagerTermBase
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    class MaskedCameraImage(ManagerTermBase):
        """Camera image with withheld steps replaced by zeros (used by the re-observation-window task).

        Wraps the embodiment's own camera term, which may be a function or a class-based term.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            image_func = cfg.params["image_func"]
            self._image_params = dict(cfg.params["image_params"])
            for value in self._image_params.values():
                if isinstance(value, SceneEntityCfg):
                    value.resolve(env.scene)
            if isinstance(image_func, type):
                self._image_term = image_func(ObsTerm(func=image_func, params=self._image_params), env)
            else:
                self._image_term = image_func

        def reset(self, env_ids=None) -> None:
            if hasattr(self._image_term, "reset"):
                self._image_term.reset(env_ids)

        def __call__(self, env, image_func, image_params: dict, logic) -> torch.Tensor:
            image = self._image_term(env, **self._image_params)
            runtime = get_runtime(env, logic)
            blocked = logic.camera_blocked(env, runtime)
            shape = (-1,) + (1,) * (image.dim() - 1)
            return torch.where(blocked.view(shape), torch.zeros_like(image), image)

    return MaskedCameraImage


def make_task_observation_cfg(logic, variant: ObservationVariant):
    """Return the observation groups the task adds for ``variant`` (instruction, oracle map or online map)."""
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    from isaaclab_arena.utils.configclass import make_configclass

    def group_post_init(self):
        self.enable_corruption = False
        self.concatenate_terms = False

    groups = []
    if logic.instruction_dim() > 0:
        task_group = make_configclass(
            "SpatialMemoryTaskObsCfg",
            [("instruction", ObsTerm, ObsTerm(func=instruction, params={"logic": logic}))],
            bases=(ObsGroup,),
            namespace={"__post_init__": group_post_init},
        )
        groups.append(("task_obs", task_group, task_group()))
    if variant == ObservationVariant.ORACLE_MAP:
        map_group = make_configclass(
            "SpatialMemoryOracleMapObsCfg",
            [
                ("points", ObsTerm, ObsTerm(func=oracle_map_points, params={"logic": logic})),
                ("boxes", ObsTerm, ObsTerm(func=oracle_map_boxes, params={"logic": logic})),
            ],
            bases=(ObsGroup,),
            namespace={"__post_init__": group_post_init},
        )
        groups.append(("oracle_map", map_group, map_group()))
    elif variant == ObservationVariant.MAPPED:
        mapped_group = make_configclass(
            "SpatialMemoryMappedObsCfg",
            [("online_map", ObsTerm, ObsTerm(func=online_map))],
            bases=(ObsGroup,),
            namespace={"__post_init__": group_post_init},
        )
        groups.append(("mapped", mapped_group, mapped_group()))
    if not groups:
        return None
    return make_configclass("SpatialMemoryObservationsCfg", groups)()


def apply_observation_variant(env_cfg, logic, variant: ObservationVariant, history_length: int | None):
    """Patch the compiled environment config for ``variant`` (history stacking and withheld camera frames)."""
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    if variant == ObservationVariant.HISTORY_K:
        length = history_length if history_length is not None else default_history_length(step_dt)
        assert length >= 1, "history_k needs at least one frame."
        for group_name in ("policy", "camera_obs", "task_obs"):
            group = getattr(env_cfg.observations, group_name, None)
            if group is None:
                continue
            group.history_length = length
            # Keep images as (N, k, H, W, C) instead of flattening them.
            group.flatten_history_dim = False
    camera_group = getattr(env_cfg.observations, "camera_obs", None)
    if logic.withholds_camera_frames and camera_group is not None:
        for term_name, term in list(vars(camera_group).items()):
            if not isinstance(term, ObsTerm):
                continue
            masked = ObsTerm(
                func=_make_masked_camera_image_term(),
                params={"image_func": term.func, "image_params": dict(term.params), "logic": logic},
            )
            setattr(camera_group, term_name, masked)
    return env_cfg
