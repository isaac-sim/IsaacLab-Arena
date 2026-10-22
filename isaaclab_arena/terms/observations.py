# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation terms owned by Arena."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

from isaaclab_arena.utils.timer import Timer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


class camera_image(ManagerTermBase):
    """Camera image observation that renders a new frame only when the environment requests one.

    Reading a camera's data triggers the renderer, which costs the same whether or not a policy looks at the
    frame. Environments expose ``camera_frames_requested``; while it is False this term returns the frame it
    rendered last instead of rendering again. The first call, and every call on an environment without that
    attribute, renders. The environment itself requests a frame whenever an episode starts.

    Accepts the same parameters as ``isaaclab.envs.mdp.image``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._last_image: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        normalize: bool = True,
        permute: bool = False,
    ) -> torch.Tensor:
        if self._last_image is None or getattr(env, "camera_frames_requested", True):
            with Timer("camera_render"):
                self._last_image = mdp.image(
                    env,
                    sensor_cfg=sensor_cfg,
                    data_type=data_type,
                    convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
                    normalize=normalize,
                    permute=permute,
                )
        return self._last_image
