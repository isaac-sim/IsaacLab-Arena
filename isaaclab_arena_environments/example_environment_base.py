# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

ArenaEnvironmentCfgT = TypeVar("ArenaEnvironmentCfgT", bound=ArenaEnvironmentCfg)


# TODO(cvolk, 2026-07-03): Remove after external factories migrate from argparse to build(cfg).
class ExampleEnvironmentBase(ArenaEnvironmentFactory[ArenaEnvironmentCfgT]):
    """Provide deprecated argparse compatibility for external environment factories."""

    @abstractmethod
    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        # TODO(cvolk, 2026-07-03): Deprecate this legacy argparse entry point; build(cfg) will
        # become the primary environment construction API.
        pass

    def build(self, cfg: ArenaEnvironmentCfgT) -> IsaacLabArenaEnvironment:
        """Reject typed construction until a legacy external factory migrates."""
        raise NotImplementedError(f"Legacy factory {type(self).__name__} only supports get_env(args_cli)")

    @abstractmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass
