"""Focused keyboard/physics/optional GL regression in the existing Arena Docker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/examples_teleop"))
from example23_teleop_keyboard_abb_irb1200_robotiq_2f140_newton import (
    KeyboardInput,
    KeyboardRobotiqExample,
    KeyReplay,
    create_parser,
    keyboard_viewer,
    rotation_error_degrees,
)


def test_input():
    keys = KeyboardInput()
    keys.press("W")
    assert np.array_equal(keys.axes(), [1, 0, 0, 0, 0, 0])
    keys.press("S")
    assert not keys.axes().any()
    keys.release("S")
    keys.press("A")
    assert np.isclose(np.linalg.norm(keys.axes()[:3]), 1)
    keys.press("LSHIFT")
    assert np.isclose(np.linalg.norm(keys.axes()[:3]), 0.2)
    keys.press("K")
    for _ in range(50):
        keys.press("K")
    assert keys.closed and keys.toggle_count == 1
    keys.clear()
    assert not keys.axes().any() and keys.closed
    keys.press("K")
    assert keys.closed, "Focus/pause repeats must not toggle grip"
    keys.release("K")
    keys.press("K")
    assert not keys.closed and keys.toggle_count == 2
    keys.press("R")
    assert keys.reset_requested
    keys.reset()
    keys.press("R")
    assert not keys.reset_requested, "Held R must not repeatedly reset"
    keys.release("R")
    keys.press("E")
    keys.press("R")
    keys.reset(replay=True)
    keys.press("E")
    keys.press("R")
    assert "E" in keys.held and not keys.reset_requested, "Replay reset must not block synthetic movement"
    args = create_parser().parse_args(["--num_steps", "99", "--pos_sensitivity", ".1", "--control_mode", "ik"])
    assert args.num_frames == 99 and args.pos_speed == 0.1
    print("[PASS] Input cancellation, normalization, fine speed, repeat, focus clear, reset, CLI aliases", flush=True)


def test_physics(gui):
    args = create_parser().parse_args(["--viewer", "gl" if gui else "null", "--mujoco-cpu", "--quiet"])
    keys = KeyboardInput()
    viewer = keyboard_viewer(args, keys)
    try:
        example = KeyboardRobotiqExample(viewer, args)
        modes = example.model.joint_target_mode.numpy()
        assert sum(modes[6:12] != 0) == 1
        assert example.solver.mj_model.neq == 5 and not example.sequence
        initial = example.tcp_position(example.state_0).copy()
        initial_cube = example.state_0.body_q.numpy()[example.cube_indices].copy()

        def steps(count):
            for _ in range(count):
                example.step(keys)
                if gui:
                    example.render()

        keys.press("W")
        snapshots = {n: getattr(example.state_0, n).numpy().copy() for n in ("joint_q", "joint_qd", "body_q", "body_qd")}
        old_targets = example.control.joint_target_pos.numpy().copy()
        example.update_command(keys)
        for n, before in snapshots.items():
            assert np.array_equal(before, getattr(example.state_0, n).numpy()), f"Control wrote physical {n}"
        targets = example.control.joint_target_pos.numpy()
        assert np.max(np.abs(targets[:6] - old_targets[:6])) <= args.joint_speed * example.frame_dt + 1e-6
        assert np.array_equal(targets[7:12], old_targets[7:12])
        steps(14)
        keys.release("W")
        stopped_target = example.command_tcp.copy()
        steps(60)
        assert np.array_equal(example.command_tcp, stopped_target)
        assert abs(example.tcp_position(example.state_0)[0] - initial[0] - .03) < .003

        keys.press("C")
        steps(60)
        keys.release("C")
        steps(60)
        assert rotation_error_degrees(example.state_0.body_q.numpy()[example.ee_index, 3:], np.array(example.command_rot)) < 1
        assert np.linalg.norm(example.tcp_position(example.state_0) - stopped_target) < .003
        keys.press("K")
        steps(180)
        keys.release("K")
        assert example.state_0.joint_q.numpy()[example.master_dof] > .6
        assert np.linalg.norm(example.tcp_position(example.state_0) - stopped_target) < .003
        assert example.max_mimic_error < .01
        # Simulate a lagged TARGET (not physical state). Reverse input must
        # remain possible even outside the angular tracking-error budget.
        old_rotation = example.command_rot
        example.command_rot = wp.mul(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), .3), old_rotation)
        before_angle = rotation_error_degrees(np.array(old_rotation), np.array(example.command_rot))
        keys.press("V")
        example.update_command(keys)
        keys.release("V")
        assert rotation_error_degrees(np.array(old_rotation), np.array(example.command_rot)) < before_angle
        example.command_rot = old_rotation
        # A failed solve must reject/revert targets and remain recoverable.
        old_solve = example.solve_ik
        def failed_solve(position, iterations=None):
            old_solve(position, iterations)
            q = example.joint_q_ik.numpy()
            q[0, 0] = np.nan
            example.joint_q_ik.assign(q)
        example.solve_ik = failed_solve
        keys.press("Q")
        example.update_command(keys)
        keys.clear()
        assert example.rejected_commands == 1
        assert np.array_equal(example.command_tcp, stopped_target)
        assert np.isfinite(example.control.joint_target_pos.numpy()).all()
        del example.solve_ik

        old_solver = example.solver
        example.reset()
        keys.reset()
        assert example.solver is not old_solver and example.episode_steps == 0
        assert np.linalg.norm(example.tcp_position(example.state_0) - initial) < .002
        assert np.allclose(example.state_0.body_q.numpy()[example.cube_indices], initial_cube, atol=1e-6)
        assert abs(example.state_0.joint_q.numpy()[example.master_dof]) < 1e-5
        assert example.rejected_commands == 0 and example.command_grip == 0
        steps(10)
        print("[PASS] Finite drives only; release holds; rotated TCP + aperture compensation; bad IK rejection; full reset", flush=True)

        if gui:
            from pyglet.window import key
            window = viewer.renderer.window
            # Dispatch through the real window -> renderer -> subclass callback.
            def event(name, *arguments):
                window.dispatch_event(name, *arguments)
                window.dispatch_events()
            event("on_key_press", key.W, 0)
            assert "W" in keys.held
            camera = np.array(viewer.camera.pos)
            viewer._update_camera(1.0)
            assert np.array_equal(camera, np.array(viewer.camera.pos)), "W also moved camera"
            event("on_key_release", key.W, 0)
            assert not keys.axes().any()
            event("on_key_press", key.K, 0)
            event("on_key_press", key.K, 0)
            assert keys.closed and keys.toggle_count == 1
            event("on_key_release", key.K, 0)
            event("on_key_press", key.SPACE, 0)
            assert viewer.is_paused() and not keys.held
            event("on_key_press", key.W, 0)
            assert not keys.held
            event("on_key_release", key.SPACE, 0)
            event("on_key_press", key.SPACE, 0)
            assert not viewer.is_paused()
            event("on_key_press", key.W, 0)
            event("on_deactivate")
            assert not keys.held and keys.closed
            event("on_key_press", key.R, 0)
            assert keys.reset_requested
            print("[PASS] GL window event routing, camera isolation, K repeat, pause, focus loss, R", flush=True)
            args.replay = True
            keys.__init__()
            replay = KeyReplay(example)
            while not replay.finished:
                replay.advance(keys)
                held = keys.held.copy()
                event("on_deactivate")
                event("on_key_release", key.W, 0)
                event("on_key_press", key.K, 0)
                assert keys.held == held, "GUI focus/live input corrupted scripted replay"
            assert keys.toggle_count == 2
            event("on_key_press", key.SPACE, 0)
            assert viewer.is_paused()
            event("on_key_release", key.SPACE, 0)
            event("on_key_press", key.SPACE, 0)
            assert not viewer.is_paused()
            print("[PASS] Replay isolated from GUI focus/live robot keys; pause/resume retained", flush=True)
        return dict(passed=True, input=True, finite_drives=True, rotation_tcp=True,
                    reset=True, rejected_ik=True, gui_events=gui)
    finally:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    test_input()
    report = test_physics(args.gui)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")