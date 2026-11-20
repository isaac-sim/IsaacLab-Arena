# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for looking at a running Arena simulation through Isaac Sim's Python server.

Send this file once per simulation process; the functions stay defined in the server's default context::

    python3 -I run_in_sim.py --port PORT --file sim_helpers.py

Requests run inside Kit's update loop, so these helpers only schedule work and never call ``app.update()``.
"""

_arena_env = None


def find_arena_env():
    """Return the Arena environment running in this process (the unwrapped env)."""
    global _arena_env
    if _arena_env is None:
        import gc
        import weakref

        from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

        arena_envs = []
        for candidate in gc.get_objects():
            # A weakref proxy also passes isinstance(); keep only the env itself.
            if isinstance(candidate, IsaacLabArenaManagerBasedRLEnv) and not isinstance(candidate, weakref.ProxyTypes):
                arena_envs.append(candidate)
        assert len(arena_envs) == 1, f"Expected one Arena environment in this process, found {len(arena_envs)}."
        _arena_env = arena_envs[0]
    return _arena_env


def capture_viewer(output_path, width=640, height=360):
    """Save the viewer camera image as a PNG; the file appears a few frames after this returns.

    Call set_camera() in an earlier request: a capture in the same request still shows the previous view.
    """
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    assert viewport is not None, "No viewport in this process; launch with --viz kit."
    viewport.resolution = (width, height)
    capture_viewport_to_file(viewport, file_path=output_path)
    return output_path


def set_camera(eye, target, env_index=None):
    """Point the viewer camera from eye to target, in meters.

    Args:
        eye: Camera position, in the world frame or, with env_index, relative to that env's origin.
        target: Point the camera looks at, in the same frame as eye.
        env_index: Env whose origin eye and target are relative to.
    """
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

    if env_index is not None:
        t_W_E = find_arena_env().scene.env_origins[env_index].tolist()
        eye = [eye_coordinate + origin for eye_coordinate, origin in zip(eye, t_W_E)]
        target = [target_coordinate + origin for target_coordinate, origin in zip(target, t_W_E)]
    set_kit_renderer_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")


def print_scene_state(env_index=0):
    """Print each rigid object's and articulation's position relative to the env origin and its speed."""
    import torch

    env = find_arena_env()
    for scene_key in sorted([*env.scene.rigid_objects.keys(), *env.scene.articulations.keys()]):
        t_E_O = env.arena_world.get_pose_e(scene_key)[env_index, :3].tolist()
        speed = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(scene_key)[env_index]).item()
        position_text = ", ".join(f"{coordinate:.3f}" for coordinate in t_E_O)
        print(f"{scene_key}: position [{position_text}] m, speed {speed:.3f} m/s")


def pause_physics():
    """Pause the simulation; the server keeps answering, so the frozen state can be captured from several views."""
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    # Newton runs keep the Kit timeline stopped and step regardless, so pausing it would do nothing.
    assert timeline.is_playing(), "The Kit timeline is not playing; pausing works only in PhysX runs."
    timeline.pause()


def resume_physics():
    """Resume a simulation paused with pause_physics()."""
    import omni.timeline

    omni.timeline.get_timeline_interface().play()
