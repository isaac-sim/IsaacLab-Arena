# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model-response errors shared by VLM policies and embodiment adapters."""


class InvalidAgentResponseError(ValueError):
    """A model response was rejected and may be corrected on another inference attempt."""
