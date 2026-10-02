# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.pose import Pose, PosePerEnv


class _StubLookatObject:
    def __init__(self, initial_pose):
        self.name = "stub_object"
        self.initial_pose = initial_pose

    def get_initial_pose(self):
        return self.initial_pose


def test_get_viewer_cfg_look_at_object_pose_per_env():
    """Relation-placed objects store PosePerEnv; viewer cfg uses env 0."""
    obj = _StubLookatObject(PosePerEnv(poses=[Pose(position_xyz=(1.0, 2.0, 3.0))]))
    viewer_cfg = get_viewer_cfg_look_at_object(obj, offset=np.array([-1.0, -1.0, 1.0]))
    assert viewer_cfg.lookat == (1.0, 2.0, 3.0)
    assert viewer_cfg.eye == (0.0, 1.0, 4.0)


def _test_arena_preserves_scene_authored_visualizer_background(simulation_app):
    """Arena environments show scene-authored HDR backgrounds by default."""
    from isaaclab.envs import ViewerCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab_arena.environments.arena_env_builder import _configure_arena_visualizer_defaults
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

    cfg = IsaacLabArenaManagerBasedRLEnvCfg()
    cfg.viewer = ViewerCfg(
        eye=(1.0, -2.0, 3.0),
        lookat=(0.1, 0.2, 0.3),
        origin_type="env",
        env_index=2,
        resolution=(640, 480),
    )
    _configure_arena_visualizer_defaults(cfg)
    assert isinstance(cfg.sim.default_visualizer_cfg, KitVisualizerCfg)
    assert cfg.sim.default_visualizer_cfg.background_color is None
    assert cfg.sim.default_visualizer_cfg.eye == (1.0, -2.0, 3.0)
    assert cfg.sim.default_visualizer_cfg.lookat == (0.1, 0.2, 0.3)
    assert cfg.sim.default_visualizer_cfg.origin_type == "env"
    assert cfg.sim.default_visualizer_cfg.origin_env_index == 2
    assert cfg.sim.default_visualizer_cfg.window_width == 640
    assert cfg.sim.default_visualizer_cfg.window_height == 480

    cfg = IsaacLabArenaManagerBasedRLEnvCfg()
    _configure_arena_visualizer_defaults(cfg)
    assert cfg.sim.default_visualizer_cfg is not None
    assert cfg.sim.default_visualizer_cfg.background_color is None

    custom_eye = (3.0, -3.0, 2.0)
    cfg.sim.default_visualizer_cfg = KitVisualizerCfg(eye=custom_eye)
    _configure_arena_visualizer_defaults(cfg)
    assert cfg.sim.default_visualizer_cfg.eye == custom_eye
    assert cfg.sim.default_visualizer_cfg.background_color is None

    custom_background = (0.1, 0.2, 0.3)
    cfg.sim.default_visualizer_cfg.background_color = custom_background
    _configure_arena_visualizer_defaults(cfg)
    assert cfg.sim.default_visualizer_cfg.background_color == custom_background
    return True


def test_arena_preserves_scene_authored_visualizer_background():
    assert run_function_with_persistent_simulation_app(_test_arena_preserves_scene_authored_visualizer_background)
