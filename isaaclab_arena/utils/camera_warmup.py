# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Warm scene cameras after reset so recordings do not start with a black frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def uses_newton_physics_backend(sim) -> bool:
    """Return whether ``sim`` is stepping with the Newton physics backend."""
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
    except ImportError:
        return False
    physics_manager = getattr(sim, "physics_manager", None)
    return isinstance(physics_manager, NewtonManager)


def _iter_scene_cameras(env: ManagerBasedRLEnv) -> list:
    from isaaclab.sensors.camera import Camera

    return [sensor for sensor in env.scene.sensors.values() if isinstance(sensor, Camera)]


def warm_scene_cameras_after_reset(env: ManagerBasedRLEnv, *, render_iters: int = 2) -> bool:
    """Render and refresh cameras so policy observations include a populated RGB buffer.

    Args:
        env: Live manager-based environment with scene cameras.
        render_iters: Number of ``sim.render()`` calls before refreshing sensor buffers.

    Returns:
        True when at least one scene camera was warmed up.
    """
    cameras = _iter_scene_cameras(env)
    if not cameras:
        return False
    if not env.sim.is_rendering:
        return False

    for _ in range(render_iters):
        env.sim.render()
    for camera in cameras:
        camera.update(dt=0.0, force_recompute=True)
    return True


def warm_scene_cameras_if_needed(env: ManagerBasedRLEnv) -> bool:
    """Warm cameras after reset when Newton rendering skips RTX post-reset rerenders."""
    if not uses_newton_physics_backend(env.sim):
        return False
    # ManagerBasedEnv.reset() only calls sim.render() when has_rtx_sensors is set; Newton Warp
    # cameras often render through a different path, leaving frame 0 black until primed here.
    if env.has_rtx_sensors and env.cfg.num_rerenders_on_reset > 0:
        return False
    return warm_scene_cameras_after_reset(env)
