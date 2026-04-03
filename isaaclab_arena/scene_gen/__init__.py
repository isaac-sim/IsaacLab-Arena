# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-driven scene generation for IsaacLab Arena.

Ported from RoboLab's scene generation pipeline, adapted to use Arena's
AssetRegistry (700+ objects) and ObjectPlacer (differentiable relation solver).

Usage:
    from isaaclab_arena.scene_gen import SceneGenerator

    gen = SceneGenerator(background="maple_table_robolab")
    scene = gen.generate_scene("A kitchen counter with fruits and tools", max_objects=8)

    # Or fully automatic batch generation:
    scenes = gen.generate_batch(num_scenes=100)
"""
