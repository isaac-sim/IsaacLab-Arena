# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    wbc_model_path: str = "models/homie_v2/stand.onnx,models/homie_v2/walk.onnx"
    """Path to WBC model file"""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""

    policy_config_path: str = "config/g1_homie_v2.yaml"
    """Policy related configuration to specify inputs/outputs dim"""
