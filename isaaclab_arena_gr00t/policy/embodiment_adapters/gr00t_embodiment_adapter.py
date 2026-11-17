# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment adapters between Arena observations/actions and GR00T state/action dicts."""

from __future__ import annotations

import inspect
import numpy as np
import torch
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any


class Gr00tEmbodimentAdapter(ABC):
    """Translates one embodiment's Arena observations and actions to and from GR00T's state and action dicts.

    The adapter owns the embodiment's state units and action layout. Cameras, language, and the request
    format stay in the shared GR00T pipeline.
    """

    embodiment_tag: str
    """GR00T embodiment tag of the checkpoints this adapter serves; it selects the GR00T modality config."""

    action_dim: int
    """Width of one env action produced by build_action_np."""

    @abstractmethod
    def extract_state_np(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return GR00T state arrays keyed by state modality key, each with shape (num_envs, dim)."""

    @abstractmethod
    def build_action_np(self, policy_action: dict[str, np.ndarray]) -> np.ndarray:
        """Convert GR00T's action dict to env actions with shape (num_envs, horizon, action_dim)."""

    @abstractmethod
    def build_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Return the (num_envs, action_dim) action that holds each robot at its current state."""


GR00T_EMBODIMENT_ADAPTER_CLASS_PATHS = {
    "droid": "isaaclab_arena_gr00t.policy.embodiment_adapters.droid_adapter.Gr00tDroidAdapter",
    "gr1_arms": "isaaclab_arena_gr00t.policy.embodiment_adapters.gr1_arms_adapter.Gr00tGr1ArmsAdapter",
    "g1_wbc": "isaaclab_arena_gr00t.policy.embodiment_adapters.g1_wbc_adapter.Gr00tG1WbcAdapter",
}
"""Built-in adapter names and the class paths they select."""


def import_gr00t_embodiment_adapter_class(name_or_class_path: str) -> type[Gr00tEmbodimentAdapter]:
    """Import the concrete Gr00tEmbodimentAdapter subclass named by a built-in adapter name or a dotted class path."""
    class_path = GR00T_EMBODIMENT_ADAPTER_CLASS_PATHS.get(name_or_class_path, name_or_class_path)
    assert "." in class_path, (
        f"gr00t_embodiment_adapter must be one of {sorted(GR00T_EMBODIMENT_ADAPTER_CLASS_PATHS)} or a dotted class"
        f" path, got {name_or_class_path!r}"
    )
    module_path, class_name = class_path.rsplit(".", 1)
    adapter_type = getattr(import_module(module_path), class_name, None)
    assert adapter_type is not None, f"gr00t_embodiment_adapter {class_path!r}: {module_path} has no {class_name}"
    assert isinstance(adapter_type, type) and issubclass(
        adapter_type, Gr00tEmbodimentAdapter
    ), f"gr00t_embodiment_adapter {class_path!r} is not a Gr00tEmbodimentAdapter subclass"
    assert not inspect.isabstract(
        adapter_type
    ), f"gr00t_embodiment_adapter {class_path!r} does not implement {sorted(adapter_type.__abstractmethods__)}"
    return adapter_type
