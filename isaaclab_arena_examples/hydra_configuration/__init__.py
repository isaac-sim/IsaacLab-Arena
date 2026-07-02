# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal Hydra configuration example for an Isaac Lab Arena environment."""

# Importing the environment module applies its ConfigStore registration decorator.
from isaaclab_arena_examples.hydra_configuration import pick_and_place_maple_table

__all__ = ["pick_and_place_maple_table"]
