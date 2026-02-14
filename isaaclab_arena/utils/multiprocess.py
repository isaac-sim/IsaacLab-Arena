# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os


def get_local_rank():
    """Get the local rank of the current process, could be set by torchrun command."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size():
    """Get the world size of the current process, could be set by torchrun command."""
    return int(os.environ.get("WORLD_SIZE", "1"))
