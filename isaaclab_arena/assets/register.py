# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import (
    AssetRegistry,
    DeviceRegistry,
    EnvironmentRegistry,
    HDRImageRegistry,
    ObjectRelationLibraryRegistry,
    PolicyRegistry,
    RetargeterRegistry,
    TaskRegistry,
)

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


def _register_policy(cls: type["PolicyBase"], cfg_type: type["PolicyCfg"] | None):
    if PolicyRegistry().is_registered(cls.name, ensure_loaded=False):
        print(f"WARNING: Policy {cls.name} is already registered. Doing nothing.")
    else:
        PolicyRegistry().register_policy(cls, cfg_type)
    return cls


# Decorator to register a policy with the PolicyRegistry.
def register_policy(policy_type: type["PolicyBase"] | None = None, *, cfg_type: type["PolicyCfg"] | None = None):
    """Register a policy and its typed configuration."""
    if cfg_type is None:
        # TODO(cvolk, 2026-07-03): After the policy_runner and eval_runner compatibility
        # fallbacks for untyped downstream policies are removed, require cfg_type and delete
        # the optional policy_type argument and this warning branch.
        assert policy_type is not None, "Typed policy registration requires cfg_type"
        warnings.warn(
            "Bare @register_policy is deprecated; use @register_policy(cfg_type=PolicyCfgType)",
            DeprecationWarning,
            stacklevel=2,
        )
        return _register_policy(policy_type, None)

    assert policy_type is None, "Pass the policy config with the cfg_type keyword"

    def decorator(cls):
        return _register_policy(cls, cfg_type)

    return decorator


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
        registry.register(cls, cls.name)
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
