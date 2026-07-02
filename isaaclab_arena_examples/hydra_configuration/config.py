# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration nodes and composition for the Hydra example."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


CONFIG_NAME = "hydra_example_suite"
CONFIG_PATH = Path(__file__).with_name(f"{CONFIG_NAME}.yaml")


@dataclass
class SimulationAppConfiguration:
    """Configure the process-global Isaac Sim application."""

    device: str = "cuda:0"
    enable_cameras: bool = False
    visualizer: str | None = None


@dataclass
class EnvironmentBuilderConfiguration:
    """Configure how Arena builds the selected environment."""

    num_envs: int = 1
    env_spacing: float = 30.0
    seed: int = 42
    solve_relations: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    random_yaw_init: bool = False
    disable_fabric: bool = False
    mimic: bool = False
    presets: str | None = None

    def __post_init__(self) -> None:
        assert self.num_envs > 0, "num_envs must be greater than zero"


@dataclass
class RolloutConfiguration:
    """Configure the short rollout used by this example."""

    num_steps: int = 2

    def __post_init__(self) -> None:
        assert self.num_steps > 0, "num_steps must be greater than zero"


@dataclass
class PolicyConfiguration:
    """Configure the eval-runner policy used by this example."""

    type: str = "zero_action"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.type, "policy type must not be empty"


@dataclass
class ArenaEnvironmentConfiguration(ABC):
    """Typed definition that builds, but is not, an Arena runtime environment."""

    @abstractmethod
    def build(self, *, enable_cameras: bool) -> IsaacLabArenaEnvironment:
        """Build the configured Arena environment after Isaac Sim has launched."""


@dataclass
class ArenaRunConfiguration:
    """Configure one Arena environment run."""

    name: str = "hydra_environment_configuration_example"
    simulation_app: SimulationAppConfiguration = field(default_factory=SimulationAppConfiguration)
    environment_builder: EnvironmentBuilderConfiguration = field(default_factory=EnvironmentBuilderConfiguration)
    policy: PolicyConfiguration = field(default_factory=PolicyConfiguration)
    rollout: RolloutConfiguration = field(default_factory=RolloutConfiguration)
    environment: ArenaEnvironmentConfiguration = MISSING


def register_environment_configuration(name: str):
    """Register an environment configuration in Hydra's ``environment`` group."""

    def decorator(configuration_type: type[ArenaEnvironmentConfiguration]) -> type[ArenaEnvironmentConfiguration]:
        ConfigStore.instance().store(group="environment", name=name, node=configuration_type)
        return configuration_type

    return decorator


def compose_hydra_example_suite(
    overrides: list[str] | None = None,
    config_path: str | Path = CONFIG_PATH,
) -> ArenaRunConfiguration:
    """Compose the example YAML and return its fully typed configuration.

    Args:
        overrides: Optional Hydra override tokens applied after the YAML.
        config_path: Path to the root suite YAML.

    Returns:
        The composed root configuration with a concrete environment configuration.
    """
    path = Path(config_path).resolve()
    assert path.is_file(), f"Hydra example config does not exist: {path}"

    ConfigStore.instance().store(name="arena_run_schema", node=ArenaRunConfiguration)

    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        dict_config = compose(config_name=path.stem, overrides=overrides or [])

    configuration = OmegaConf.to_object(dict_config)
    assert isinstance(configuration, ArenaRunConfiguration)
    return configuration
