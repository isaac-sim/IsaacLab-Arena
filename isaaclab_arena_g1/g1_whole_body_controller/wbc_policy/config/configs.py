# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING, asdict, dataclass
from typing import Literal


@dataclass
class ArgsConfig:
    """Args Config for running the data collection loop."""

    def update(self, config_dict: dict, strict: bool = False, skip_keys: list[str] = []):
        for k, v in config_dict.items():
            if k in skip_keys:
                continue
            if strict and not hasattr(self, k):
                raise ValueError(f"Config {k} not found in {self.__class__.__name__}")
            if not strict and not hasattr(self, k):
                continue
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, config_dict: dict, strict: bool = False, skip_keys: list[str] = []):
        instance = cls()
        instance.update(config_dict=config_dict, strict=strict, skip_keys=skip_keys)
        return instance

    def to_dict(self):
        return asdict(self)


@dataclass
class BaseConfig(ArgsConfig):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: str = MISSING
    """Version of the whole body controller."""

    wbc_model_path: str = MISSING
    """Path to WBC model file"""

    policy_config_path: str = MISSING
    """Policy related configuration to specify inputs/outputs dim"""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""


@dataclass
class HomieV2Config(BaseConfig):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: Literal["homie_v2"] = "homie_v2"
    """Version of the whole body controller."""

    # NOTE(peterd, 9/25/2025): Update OV paths once release location is finalized
    wbc_model_path: str = (
        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/wbc_policy/models/homie_v2/stand.onnx,omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/wbc_policy/models/homie_v2/walk.onnx"
    )
    """Path to WBC model file"""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""

    policy_config_path: str = "config/g1_homie_v2.yaml"
    """Policy related configuration to specify inputs/outputs dim"""
