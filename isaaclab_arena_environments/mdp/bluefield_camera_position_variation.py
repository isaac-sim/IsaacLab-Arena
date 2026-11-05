# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bluefield ego-camera translations in the calibrated camera's torso parent frame."""

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.sensors import Camera, TiledCamera

from isaaclab_arena.patches.camera_render_pose import CameraPoseWriter
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg


class BluefieldCameraPositionVariation(RunTimeVariationBase):
    """Sample ego-camera XYZ offsets in torso_link, preserving rotation and intrinsics."""

    def __init__(self):
        super().__init__(
            name="ego_camera_offset_torso",
            cfg=VariationBaseCfg(sampler_cfg=UniformSamplerCfg(low=[-0.03] * 3, high=[0.03] * 3)),
        )

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        return (
            "bluefield_ego_camera_offset_torso",
            EventTermCfg(func=_ApplyTorsoCameraOffset, mode="reset", params={"sampler": self.sampler}),
        )


class _ApplyTorsoCameraOffset(ManagerTermBase):
    """Write nominal position plus the latest sampled offset, without accumulating resets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert "robot_head_cam" in env.scene.sensors, "Bluefield camera offsets require enable_cameras=true."
        camera = env.scene["robot_head_cam"]
        assert isinstance(camera, (Camera, TiledCamera)), "Expected a camera sensor for robot_head_cam."
        assert tuple(cfg.params["sampler"].shape_per_sample) == (3,), "Expected XYZ camera offsets."
        self._nominal_position = torch.tensor(camera.cfg.offset.pos, device=env.device)
        self._writer = CameraPoseWriter(camera)

    def __call__(self, env, env_ids, sampler):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        if len(env_ids) == 0:
            return
        offsets = sampler.sample(num_samples=len(env_ids), env_ids=env_ids).to(self._nominal_position)
        self._writer.set_local_translations(self._nominal_position + offsets, env_ids)
