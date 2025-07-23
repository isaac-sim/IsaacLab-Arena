# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment


def test_isaac_arena_environment():
    with SimulationAppContext():
        # Arena Environment
        isaac_arena_environment = IsaacArenaEnvironment(
            embodiment=FrankaEmbodiment(),
            scene=MugInDrawerKitchenPickAndPlaceScene(),
            task=PickAndPlaceTaskCfg(),
        )
