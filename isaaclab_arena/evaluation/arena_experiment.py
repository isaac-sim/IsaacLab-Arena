# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Domain type for an Arena Experiment."""

from typing import TypeAlias

from isaaclab_arena.evaluation.arena_run import ArenaRunCfg

ArenaExperiment: TypeAlias = list[ArenaRunCfg]
"""An Experiment expressed as its ordered list of Runs."""
