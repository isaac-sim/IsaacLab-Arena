# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import get_original_bases
from typing import TYPE_CHECKING, get_args, get_origin

from isaaclab_arena.assets.registries import (
    AssetRegistry,
    DeviceRegistry,
    EnvironmentRegistry,
    HDRImageRegistry,
    ObjectRelationLibraryRegistry,
    PlacementValidatorRegistry,
    PolicyRegistry,
    RetargeterRegistry,
    TaskRegistry,
)
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


# Decorator to register an asset with the AssetRegistry.
def register_asset(cls):
    if AssetRegistry().is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: Asset {cls.name} is already registered. Doing nothing.")
    else:
        AssetRegistry().register(cls, cls.name)
    return cls


# Decorator to register an device with the DeviceRegistry.
def register_device(cls):
    if DeviceRegistry().is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: Device {cls.name} is already registered. Doing nothing.")
    else:
        DeviceRegistry().register(cls, cls.name)
    return cls


# Decorator to register an retargeter with the RetargeterRegistry.
def register_retargeter(cls):
    retargeter_key = (cls.device, cls.embodiment)
    retargeter_key_str = RetargeterRegistry().convert_tuple_to_str(retargeter_key)
    if RetargeterRegistry().is_registered(retargeter_key_str, ensure_loaded=False):
        print(f"WARNING: Retargeter {cls.device} for {cls.embodiment} is already registered. Doing nothing.")
    else:
        RetargeterRegistry().register(cls, retargeter_key_str)
    return cls


# Decorator to register a policy with the PolicyRegistry.
def register_policy(cls: type["PolicyBase"]):
    """Register a policy and its typed configuration."""
    if PolicyRegistry().is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: Policy {cls.name} is already registered. Doing nothing.")
    else:
        PolicyRegistry().register_policy(cls, _policy_cfg_type_from_policy(cls))
    return cls


# Decorator to register an HDRImage with the HDRImageRegistry.
def register_hdr(cls):
    if HDRImageRegistry().is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: HDRImage {cls.name} is already registered. Doing nothing.")
    else:
        HDRImageRegistry().register(cls, cls.name)
    return cls


# Decorator to register an environment with the EnvironmentRegistry.
def register_environment(cls):
    registry = EnvironmentRegistry()
    if registry.is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: Environment {cls.name} is already registered. Doing nothing.")
    else:
        registry.register_environment(cls, _environment_cfg_type_from_factory(cls))
    return cls


# Decorator to register a RelationBase subclass with the ObjectRelationLibraryRegistry.
def register_object_relation(cls):
    registry = ObjectRelationLibraryRegistry()
    if registry.is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: Object relation {cls.name} is already registered. Doing nothing.")
    else:
        registry.register(cls, cls.name)
    return cls


# Decorator to register a TaskBase subclass with the TaskRegistry.
# Keyed by `cls.__name__` so the YAML `type: PascalCase` lookups match without
# requiring a separate `name` attribute on every task class.
def register_task(cls):
    registry = TaskRegistry()
    if registry.is_registered(cls.__name__, ensure_loaded=False):
        print(f"WARNING: Task {cls.__name__} is already registered. Doing nothing.")
    else:
        registry.register(cls, cls.__name__)
    return cls


def agent_ready(cls):
    """Mark a task class as available to the environment-generation agent."""
    cls.agent_ready = True
    return cls


# Decorator to register a PlacementValidator subclass with the PlacementValidatorRegistry.
# Keyed by `cls.check` (the PlacementCheck it reports) so build_validators() can resolve it.
def register_validator(cls):
    registry = PlacementValidatorRegistry()
    if registry.is_registered(cls.check, ensure_loaded=False):
        print(f"WARNING: Placement validator for {cls.check} is already registered. Doing nothing.")
    else:
        registry.register(cls, cls.check)
    return cls


def _policy_cfg_type_from_policy(policy_type: type["PolicyBase"]) -> type["PolicyCfg"]:
    """Read ``PolicyBase[Cfg]`` so the registry can map a config back to its policy."""
    # Importing PolicyBase at module load time creates assets.register -> policy package ->
    # concrete policy -> assets.register. Delay it until a concrete policy is decorated.
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

    policy_bases = [base for base in get_original_bases(policy_type) if get_origin(base) is PolicyBase]
    assert len(policy_bases) == 1, f"{policy_type.__name__} must directly inherit PolicyBase[ConcretePolicyCfg]"

    cfg_types = get_args(policy_bases[0])
    assert len(cfg_types) == 1, f"{policy_type.__name__} must declare exactly one policy config"

    cfg_type = cfg_types[0]
    assert isinstance(cfg_type, type) and issubclass(
        cfg_type, PolicyCfg
    ), f"{policy_type.__name__} must use a concrete PolicyCfg subclass"
    return cfg_type


def _environment_cfg_type_from_factory(
    factory_type: type["ArenaEnvironmentFactory"],
) -> type["ArenaEnvironmentCfg"]:
    """Read the config type from a factory's ``ArenaEnvironmentFactory[Cfg]`` base.

    The generic base is the single declaration of which config a factory consumes.
    Registration records that relationship so typed execution can later resolve the
    factory from a concrete config instance.
    """
    generic_factory_bases = [
        declared_base
        for declared_base in get_original_bases(factory_type)
        if get_origin(declared_base) is ArenaEnvironmentFactory
    ]
    assert (
        len(generic_factory_bases) == 1
    ), f"{factory_type.__name__} must directly inherit ArenaEnvironmentFactory[ConcreteEnvironmentCfg]"

    declared_cfg_types = get_args(generic_factory_bases[0])
    assert len(declared_cfg_types) == 1, f"{factory_type.__name__} must declare exactly one environment config"

    declared_cfg_type = declared_cfg_types[0]
    assert isinstance(declared_cfg_type, type) and issubclass(
        declared_cfg_type, ArenaEnvironmentCfg
    ), f"{factory_type.__name__} must use a concrete ArenaEnvironmentCfg subclass"
    return declared_cfg_type
