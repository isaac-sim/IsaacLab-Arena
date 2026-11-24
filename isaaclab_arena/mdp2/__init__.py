# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build Arena environments with Isaac Lab's MDP2.

Terms receive the scene data they read as inputs and write their outputs in place, so that Isaac Lab's MDP2 builder
can merge them into a step that replays as one CUDA graph. They run as Torch hooks between MDP2's generated kernels.
Their modules follow the components that use them: :mod:`.embodiment_terms`, :mod:`.task_terms`, and
:mod:`.placement`.
"""
