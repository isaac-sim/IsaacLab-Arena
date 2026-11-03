# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fake scene with a camera mounted on an articulation link."""

import torch
from types import SimpleNamespace


class FakeScene(dict):
    """Index sensors by name and expose one articulation like InteractiveScene."""

    def __init__(self, **sensors):
        super().__init__(sensors)
        self.articulations = {"robot": SimpleNamespace(cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot"))}


def make_mounted_camera(intrinsic_matrices: torch.Tensor) -> SimpleNamespace:
    """Return a camera on robot/mount_link whose optical frame matches the link frame."""
    return SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Robot/mount_link/camera",
            offset=SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="ros"),
        ),
        data=SimpleNamespace(intrinsic_matrices=SimpleNamespace(torch=intrinsic_matrices)),
    )
