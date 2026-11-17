# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Embodiment adapters between Arena observations/actions and GR00T state/action dicts."""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg


class Gr00tEmbodimentAdapter(ABC):
    """Translates one embodiment's Arena observations and actions to and from GR00T's state and action dicts.

    The adapter owns the embodiment's state units and action layout. Cameras, language, and the request
    format stay in the shared GR00T pipeline.
    """

    embodiment_tag: str
    """GR00T embodiment tag of the checkpoints this adapter serves. The policy config must use the same tag."""

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


def _load_droid_adapter() -> type[Gr00tEmbodimentAdapter]:
    from isaaclab_arena_gr00t.policy.embodiment_adapters.droid_adapter import Gr00tDroidAdapter

    return Gr00tDroidAdapter


def _load_gr1_arms_adapter() -> type[Gr00tEmbodimentAdapter]:
    from isaaclab_arena_gr00t.policy.embodiment_adapters.gr1_arms_adapter import Gr00tGr1ArmsAdapter

    return Gr00tGr1ArmsAdapter


def _load_g1_wbc_adapter() -> type[Gr00tEmbodimentAdapter]:
    from isaaclab_arena_gr00t.policy.embodiment_adapters.g1_wbc_adapter import Gr00tG1WbcAdapter

    return Gr00tG1WbcAdapter


_EMBODIMENT_ADAPTER_LOADERS: dict[str, Callable[[], type[Gr00tEmbodimentAdapter]]] = {
    "droid": _load_droid_adapter,
    "gr1_arms": _load_gr1_arms_adapter,
    "g1_wbc": _load_g1_wbc_adapter,
}
"""Registry mapping gr00t_embodiment_adapter keys to loaders for their adapter classes. Loaders defer the
import so adapter modules can import Gr00tEmbodimentAdapter without a cycle, and so only the selected
embodiment's dependencies load."""


def create_gr00t_embodiment_adapter(policy_config: Gr00tClosedloopPolicyCfg) -> Gr00tEmbodimentAdapter:
    """Create the embodiment adapter named by the policy config's ``gr00t_embodiment_adapter``."""
    key = policy_config.gr00t_embodiment_adapter
    loader = _EMBODIMENT_ADAPTER_LOADERS.get(key)
    assert (
        loader is not None
    ), f"Unknown gr00t_embodiment_adapter {key!r}; expected one of {sorted(_EMBODIMENT_ADAPTER_LOADERS)}"
    adapter_type = loader()
    assert policy_config.embodiment_tag == adapter_type.embodiment_tag, (
        f"gr00t_embodiment_adapter {key!r} requires embodiment_tag {adapter_type.embodiment_tag!r}, got"
        f" {policy_config.embodiment_tag!r}"
    )
    return adapter_type()
