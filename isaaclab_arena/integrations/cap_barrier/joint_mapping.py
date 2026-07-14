# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit ROS-to-Arena joint-order mapping for CAP barrier frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

_Value = TypeVar("_Value")

FR3_ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))
PANDA_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))


@dataclass(frozen=True)
class JointOrderMapping:
    """Map an ordered barrier joint contract onto an articulation's joint array."""

    abi_joint_names: tuple[str, ...]
    simulation_joint_names: tuple[str, ...]
    mapped_simulation_joint_names: tuple[str, ...]
    simulation_indices_in_abi_order: tuple[int, ...]

    @classmethod
    def from_names(
        cls,
        abi_joint_names: Sequence[str],
        abi_to_simulation_name: dict[str, str],
        simulation_joint_names: Sequence[str],
    ) -> JointOrderMapping:
        abi_names = tuple(abi_joint_names)
        sim_names = tuple(simulation_joint_names)
        if len(set(abi_names)) != len(abi_names) or len(set(sim_names)) != len(sim_names):
            raise ValueError("joint name lists must not contain duplicates")
        if set(abi_to_simulation_name) != set(abi_names):
            raise ValueError("joint-name mapping must cover the ABI roster exactly")
        mapped_sim_names = tuple(abi_to_simulation_name[name] for name in abi_names)
        if len(set(mapped_sim_names)) != len(mapped_sim_names):
            raise ValueError("multiple ABI joints map to the same simulation joint")
        missing = [name for name in mapped_sim_names if name not in sim_names]
        if missing:
            raise ValueError(f"simulation is missing mapped joints: {missing}")
        indices = tuple(sim_names.index(name) for name in mapped_sim_names)
        return cls(abi_names, sim_names, mapped_sim_names, indices)

    def to_abi_order(self, simulation_values: Sequence[_Value]) -> tuple[_Value, ...]:
        if len(simulation_values) != len(self.simulation_joint_names):
            raise ValueError("simulation value count does not match its joint roster")
        return tuple(simulation_values[index] for index in self.simulation_indices_in_abi_order)

    def assert_action_order(self, action_joint_names: Sequence[str]) -> None:
        actual = tuple(action_joint_names)
        if actual != self.mapped_simulation_joint_names:
            raise RuntimeError(
                f"Franka arm action joint order mismatch: expected {self.mapped_simulation_joint_names}, got {actual}"
            )


def make_franka_joint_mapping(simulation_joint_names: Sequence[str]) -> JointOrderMapping:
    return JointOrderMapping.from_names(
        FR3_ARM_JOINTS,
        dict(zip(FR3_ARM_JOINTS, PANDA_ARM_JOINTS, strict=True)),
        simulation_joint_names,
    )
