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

        # Match by class name instead of importing the class: Isaac Lab removes its modules from sys.modules while
        # the app starts, and an import from a request in that window loads a second copy that breaks the env build.
        arena_envs = []
        for candidate in gc.get_objects():
            # type() of a weakref proxy is the proxy type, so proxies to the env do not match.
            if any(cls.__name__ == "IsaacLabArenaManagerBasedRLEnv" for cls in type(candidate).__mro__):
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
    """Print each rigid object's and articulation's position relative to the env origin, speed, and tilt.

    The tilt is the angle between the object's own z axis and the world z axis, so an upright object shows about 0.
    """
    import math
    import torch

    env = find_arena_env()
    for scene_key in sorted([*env.scene.rigid_objects.keys(), *env.scene.articulations.keys()]):
        T_E_O = env.arena_world.get_pose_e(scene_key)[env_index]
        t_E_O = T_E_O[:3].tolist()
        qx, qy = T_E_O[3].item(), T_E_O[4].item()
        # World z component of the object's z axis for a unit quaternion (qx, qy, qz, qw).
        z_axis_W_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, z_axis_W_z))))
        speed = torch.linalg.vector_norm(env.arena_world.get_root_linear_velocity_w(scene_key)[env_index]).item()
        position_text = ", ".join(f"{coordinate:.3f}" for coordinate in t_E_O)
        print(f"{scene_key}: position [{position_text}] m, speed {speed:.3f} m/s, tilt {tilt_deg:.1f} deg")


def move_object(scene_key, position, env_index=0):
    """Teleport a rigid object to position (meters, relative to the env origin), keeping its orientation, at rest."""
    import torch

    env = find_arena_env()
    rigid_object = env.scene.rigid_objects[scene_key]
    T_W_O = env.arena_world.get_pose_w(scene_key)[env_index : env_index + 1].clone()
    T_W_O[0, :3] = env.scene.env_origins[env_index] + torch.tensor(position, device=T_W_O.device)
    rigid_object.write_root_pose_to_sim_index(root_pose=T_W_O, env_ids=[env_index])
    rigid_object.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=T_W_O.device), env_ids=[env_index]
    )


def pause_physics():
    """Pause the simulation; the server keeps answering, so the frozen state can be captured from several views.

    Pausing also stops the episode clock. The pause takes effect a step or two later. Needs a playing Kit timeline:
    PhysX runs, or Newton runs with the Kit viewer (--viz kit).
    """
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    # Newton runs without the Kit viewer keep the timeline stopped and step regardless, so pausing would do nothing.
    assert not timeline.is_stopped(), "The Kit timeline is stopped; pausing needs PhysX or --viz kit."
    timeline.pause()


_pause_observer = None


def pause_when(condition):
    """Pause the simulation on the first Kit update where condition(env) is true.

    For events faster than a request round trip, such as an object falling. The pause takes effect a step or two later.
    Can be armed while paused, before resume_physics(). A new call replaces the previous condition, so watch several
    things with one condition joined by ``or``. Needs PhysX or --viz kit, like pause_physics().

    Args:
        condition: Function of the Arena env that returns true when to pause, for example
            ``lambda env: env.arena_world.get_root_linear_velocity_w("dex_cube")[0, 2] < -1.0``.
    """
    global _pause_observer
    import carb.eventdispatcher
    import omni.kit.app
    import omni.timeline

    env = find_arena_env()
    timeline = omni.timeline.get_timeline_interface()
    assert not timeline.is_stopped(), "The Kit timeline is stopped; pausing needs PhysX or --viz kit."

    def pause_if_condition_holds(_event):
        global _pause_observer
        if _pause_observer is not None and condition(env):
            timeline.pause()
            # Dropping the observer handle unsubscribes it, so resume_physics() is not undone.
            _pause_observer = None

    _pause_observer = carb.eventdispatcher.get_eventdispatcher().observe_event(
        order=omni.kit.app.POST_UPDATE_ORDER_PYTHON_EXEC,
        event_name=omni.kit.app.GLOBAL_EVENT_POST_UPDATE,
        on_event=pause_if_condition_holds,
        observer_name="sim_helpers.pause_when",
    )


def resume_physics():
    """Resume a simulation paused with pause_physics()."""
    import omni.timeline

    omni.timeline.get_timeline_interface().play()


def is_paused():
    """Return whether the simulation is paused, for example because a pause_when() condition fired."""
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    return not timeline.is_playing() and not timeline.is_stopped()


def end_episode():
    """Start a new episode in every env at the next env step, without waiting for the timeout.

    Uses Isaac Lab's reset request (the Kit window's reset button), which env.step() handles through the normal reset
    path, so Arena's episode bookkeeping stays consistent.
    """
    find_arena_env().sim.request_reset()
