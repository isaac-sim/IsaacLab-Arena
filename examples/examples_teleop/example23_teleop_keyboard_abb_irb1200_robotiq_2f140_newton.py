"""Keyboard TCP teleoperation using example40's stock 2F-140 physics.

Run in the existing Arena Docker. Replaces the old Kit/direct-state controller:
six arm servos, ONE gripper servo, five native mimic constraints. No runtime
proxy pads, follower motors, object attachment or physical state clamping.
W/S = world +/-X; A/D = +/-Y; Q/E = +/-Z. Z/X, T/G, C/V rotate
about world X/Y/Z. K toggles grip; Shift is fine motion; R resets the episode.
Space pauses, Escape exits. Mouse controls the camera; keyboard does not.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import warp as wp

import newton
from example40_newton_abb_irb1200_robotiq_2f140_physical_grasp import (
    Robotiq2F140PhysicalExample,
    create_parser as physical_parser,
    init_viewer,
    rotation_error_degrees,
)


class KeyboardInput:
    """Shared event-to-command path for the window and deterministic replay."""

    PAIRS = (("W", "S"), ("A", "D"), ("Q", "E"), ("Z", "X"), ("T", "G"), ("C", "V"))
    KEYS = frozenset(key for pair in PAIRS for key in pair) | {"K", "R", "LSHIFT", "RSHIFT"}

    def __init__(self):
        self.held = set()
        self.blocked = set()
        self.closed = False
        self.reset_requested = False
        self.toggle_count = 0

    def press(self, key):
        key = key.upper()
        if key not in self.KEYS or key in self.held or key in self.blocked:
            return  # Ignore OS repeat, including repeated K/R events.
        self.held.add(key)
        if key == "K":
            self.closed = not self.closed
            self.toggle_count += 1
        elif key == "R":
            self.reset_requested = True

    def release(self, key):
        self.held.discard(key.upper())
        self.blocked.discard(key.upper())

    def clear(self):
        """Stop motion on focus loss/pause, without opening a held object."""
        self.blocked.update(self.held)
        self.held.clear()

    def reset(self, replay=False):
        down = self.held | self.blocked
        self.__init__()
        # Replay motion keys are synthetic, not physically held after restart.
        self.blocked = down & {"R"} if replay else down

    def axes(self):
        axes = np.array([float(p in self.held) - float(n in self.held) for p, n in self.PAIRS])
        for part in (axes[:3], axes[3:]):
            part /= max(1.0, float(np.linalg.norm(part)))
        return axes * (0.2 if self.held & {"LSHIFT", "RSHIFT"} else 1.0)


class KeyboardRobotiqExample(Robotiq2F140PhysicalExample):
    """Reuse asset/solver/contact setup; replace only the trajectory controller."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.command_rot = wp.quat(*self.home_rot)
        self.ik_probe = self.robot_model.state()
        self.lower = self.robot_model.joint_limit_lower.numpy()[:6]
        self.upper = self.robot_model.joint_limit_upper.numpy()[:6]
        self.rejected_commands = 0
        self.hold_min_lift = math.inf
        self.max_rotation_tracking_error = 0.0

    def make_sequence(self):
        return []  # Disable the inherited automatic pick/place path.

    def reset(self):
        # SolverMuJoCo 1.2.1 has no reset API. Rebuild both state buffers,
        # control, IK AND solver, rather than retaining contact warm starts.
        viewer, args = self.viewer, self.args
        self.__dict__.clear()  # Parent startup must not see the previous live TCP/rotation.
        self.__init__(viewer, args)

    def solve_ik(self, position, iterations=None):
        # Parent initialization calls this before command_rot/ik_probe exist.
        rotation = getattr(self, "command_rot", self.home_rot)
        self.rot_obj.set_target_rotations(wp.array([rotation], dtype=wp.vec4))
        original = self.home_rot
        self.home_rot = rotation  # Same rotation for objective AND live TCP offset.
        try:
            super().solve_ik(position, iterations)
        finally:
            self.home_rot = original

    def update_command(self, keyboard):
        axes = keyboard.axes()
        old_tcp, old_rot = self.command_tcp.copy(), self.command_rot
        old_ik = self.joint_q_ik.numpy()
        candidate = old_tcp + axes[:3] * self.args.pos_speed * self.frame_dt
        # Conservative tabletop target envelope, NOT collision/path planning.
        candidate = np.clip(candidate, [-0.15, -0.32, 0.139], [0.35, 0.32, 0.60])
        actual = self.tcp_position(self.state_0)
        # Do not accumulate a huge target behind a collision/unreachable pose.
        distance = np.linalg.norm(candidate - actual)
        if distance <= max(0.04, np.linalg.norm(old_tcp - actual)):
            self.command_tcp = candidate
        angular = axes[3:] * self.args.rot_speed * self.frame_dt
        if not self.args.lock_orientation and np.linalg.norm(angular) > 0:
            delta = wp.quat_from_axis_angle(wp.vec3(*(angular / np.linalg.norm(angular))), float(np.linalg.norm(angular)))
            candidate_rot = wp.normalize(wp.mul(delta, old_rot))
            actual_rot = self.state_0.body_q.numpy()[self.ee_index, 3:]
            old_error = rotation_error_degrees(actual_rot, np.array(old_rot))
            if rotation_error_degrees(actual_rot, np.array(candidate_rot)) <= max(12, old_error):
                self.command_rot = candidate_rot

        self.solve_ik(self.command_tcp)
        q = self.joint_q_ik.numpy()[0]
        # Verify arm IK on a SCRATCH state, never overwrite physical state.
        newton.eval_fk(self.robot_model, wp.array(q, dtype=float), self.robot_model.joint_qd, self.ik_probe)
        flange = self.ik_probe.body_q.numpy()[self.ee_index]
        pos_error = np.linalg.norm(flange[:3] - self.pos_obj.target_positions.numpy()[0])
        rot_error = rotation_error_degrees(flange[3:], np.array(self.command_rot))
        valid = (np.isfinite(q).all() and pos_error < 0.005 and rot_error < 3
                 and np.all(q[:6] >= self.lower - 0.001) and np.all(q[:6] <= self.upper + 0.001))
        if not valid:
            self.command_tcp, self.command_rot = old_tcp, old_rot
            self.joint_q_ik.assign(old_ik)
            q = old_ik[0]
            self.rejected_commands += 1
            if self.rejected_commands % 60 == 1:
                print(f"[LIMIT] IK target rejected ({pos_error:.4f}m, {rot_error:.2f}deg); reverse away.", flush=True)

        desired_grip = 0.7 if keyboard.closed and not self.args.hold_gripper_open else 0.0
        self.command_grip += float(np.clip(desired_grip - self.command_grip,
                                          -self.args.gripper_speed * self.frame_dt,
                                          self.args.gripper_speed * self.frame_dt))
        targets = self.control.joint_target_pos.numpy()
        increment = np.clip(q[:6] - targets[:6], -self.args.joint_speed * self.frame_dt,
                            self.args.joint_speed * self.frame_dt)
        targets[:6] = np.clip(targets[:6] + increment, self.lower, self.upper)
        targets[self.master_dof] = self.command_grip
        self.control.joint_target_pos.assign(targets)  # Follower targets/drives untouched.

    def step(self, keyboard):
        self.update_command(keyboard)
        for _ in range(10):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt
        self.episode_steps += 1
        self.measure()
        angle = rotation_error_degrees(self.state_0.body_q.numpy()[self.ee_index, 3:], np.array(self.command_rot))
        self.max_rotation_tracking_error = max(self.max_rotation_tracking_error, angle)

    def test_final(self):
        assert self.rejected_commands == 0, "Replay encountered rejected IK targets"
        assert self.max_tracking_error < 0.015, "Excessive TCP tracking error"
        assert self.max_rotation_tracking_error < 3, "Excessive orientation tracking error"
        if not self.args.expect_grasp_failure:
            assert math.isfinite(self.hold_min_lift) and self.hold_min_lift > 0.10, "Object slipped during hold"
        super().test_final()
        if self.args.report:
            report = json.loads(self.args.report.read_text())
            report.update(control_path="keyboard events -> velocity integration -> IK -> finite servos",
                          replay=True, inherited_automatic_sequence=False,
                          hold_min_lift_m=self.hold_min_lift, rejected_ik_commands=self.rejected_commands,
                          max_rotation_tracking_error_degrees=self.max_rotation_tracking_error)
            self.args.report.write_text(json.dumps(report, indent=2) + "\n")


class KeyReplay:
    """Known-layout regression using the SAME press/release events as the GUI.

    No live object tracking and no direct TCP/robot/object pose commands. Rounded
    key durations can leave <=one input step of positioning error per move.
    """

    def __init__(self, example):
        assert example.args.cube_count == 1, "Keyboard regression currently tests one free cube"
        self.segments = []
        self.index = self.elapsed = 0
        self.active = set()
        self.finished = False
        self.label = ""

        def segment(label, keys, frames):
            self.segments.append((label, set(keys), max(1, int(round(frames)))))

        def hold(label, seconds):
            segment(label, [], seconds / example.frame_dt)

        def move(label, key, metres):
            segment(label, [key], metres / example.args.pos_speed / example.frame_dt)
            hold(label + "/settle", 0.5)

        hold("initial", 0.5)
        for plus, minus in (("Z", "X"), ("T", "G"), ("C", "V")):
            segment("rotation/" + plus, [plus], 12)
            hold("rotation/hold", 0.25)
            segment("rotation/" + minus, [minus], 12)
            hold("rotation/return", 0.5)
        move("descend", "E", 0.206)
        segment("close", ["K"], 12)  # Held K must toggle exactly once.
        hold("close/settle", 0.7 / example.args.gripper_speed + 0.5)
        move("lift", "Q", 0.20)
        hold("hold", 2.0)
        delta = example.drop_position - example.pick_positions[0]
        for axis, pair in enumerate((("W", "S"), ("A", "D"))):
            if abs(delta[axis]) > 1e-9:
                move("transport", pair[0] if delta[axis] > 0 else pair[1], abs(delta[axis]))
        move("place", "E", 0.20)
        segment("release", ["K"], 12)
        hold("release/settle", 0.7 / example.args.gripper_speed + 0.5)
        move("retract", "Q", 0.20)
        hold("settle", 1.0)
        self.total_frames = sum(frames for _, _, frames in self.segments)

    def advance(self, keyboard):
        if self.finished:
            return
        self.label, keys, frames = self.segments[self.index]
        for key in self.active - keys:
            keyboard.release(key)
        for key in keys - self.active:
            keyboard.press(key)
        self.active = keys
        self.elapsed += 1
        if self.elapsed == frames:
            self.index += 1
            self.elapsed = 0
            if self.index == len(self.segments):
                self.finished = True


def keyboard_viewer(args, keyboard):
    """Local ViewerGL subclass: reserve robot keys; leave Newton sources intact."""
    if args.viewer != "gl":
        return init_viewer(args)
    from newton.viewer import ViewerGL
    from pyglet.window import key

    class TeleopViewer(ViewerGL):
        def _update_camera(self, dt):
            pass  # Mouse orbit/pan/zoom still work. WASDQE belong to the robot.

        def should_step(self):
            if not args.replay and self._ui_is_capturing_keyboard():
                keyboard.clear()
            return super().should_step()

        def on_key_press(self, symbol, modifiers):
            name = key.symbol_string(symbol)
            if name in keyboard.KEYS:
                if args.replay and name != "R":
                    return  # Do not mix live robot input with deterministic replay.
                if not self._ui_is_capturing_keyboard() and (not self.is_paused() or name == "R"):
                    keyboard.press(name)
                return
            super().on_key_press(symbol, modifiers)
            if symbol == key.SPACE and not args.replay:
                keyboard.clear()

        def on_key_release(self, symbol, modifiers):
            name = key.symbol_string(symbol)
            if not args.replay or name == "R":
                keyboard.release(name)
            super().on_key_release(symbol, modifiers)

    # Reuse common device/Warp setup without opening an extra GL window.
    setup_args = argparse.Namespace(**vars(args))
    setup_args.viewer = "null"
    init_viewer(setup_args).close()
    viewer = TeleopViewer(headless=args.headless, vsync=True)
    viewer.renderer.window.push_handlers(on_deactivate=lambda: None if args.replay else keyboard.clear())
    viewer.set_reset_callback(lambda: setattr(keyboard, "reset_requested", True))
    return viewer


def create_parser():
    parser = physical_parser()
    parser.description = __doc__
    parser.epilog = "Old joint_direct/direct_gripper/proxy flags are removed: use IK + stock finite drives."
    parser.set_defaults(num_frames=50000, output_path="robotiq_keyboard_recording.usd")
    parser.add_argument("--pos-speed", "--pos_sensitivity", type=float, default=0.12, help="TCP translation speed, m/s.")
    parser.add_argument("--rot-speed", "--rot_sensitivity", type=float, default=0.5, help="World-axis rotation speed, rad/s.")
    parser.add_argument("--joint-speed", type=float, default=1.0, help="Arm target slew limit, rad/s.")
    parser.add_argument("--gripper-speed", type=float, default=0.35, help="Master target slew limit, rad/s.")
    parser.add_argument("--lock-orientation", "--lock_orientation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hold-gripper-open", "--hold_gripper_open", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Replay keyboard events for one-cube regression, then exit.")
    parser.add_argument("--num_steps", dest="num_frames", type=int, default=argparse.SUPPRESS, help="Alias for --num-frames.")
    parser.add_argument("--keep_open", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep interactive GL open beyond --num-frames; replay always terminates.")
    parser.add_argument("--control_mode", choices=["ik"], default="ik", help="Legacy alias; only finite-drive IK is supported.")
    parser.add_argument("--visualizer", choices=["kit", "gl"], help="Legacy kit value migrates to Newton GL, not Isaac Kit.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.visualizer:
        args.viewer = "gl"
        print("[MIGRATION] example23 now uses Newton standalone GL (same backend as example40), not Kit.")
    speeds = (args.pos_speed, args.rot_speed, args.joint_speed, args.gripper_speed)
    if not all(math.isfinite(speed) and speed > 0 for speed in speeds):
        parser.error("All control speeds must be finite and positive")
    if args.test and not args.replay:
        parser.error("--test requires --replay; manual exploration is not an automatic grasp test")
    if not args.replay and (args.viewer != "gl" or args.headless):
        parser.error("Interactive keyboard needs a visible --viewer gl window; use --replay for headless runs")
    if args.keep_open and args.viewer != "gl":
        parser.error("--keep_open is only for interactive GL")
    keyboard = KeyboardInput()
    viewer = keyboard_viewer(args, keyboard)
    try:
        example = KeyboardRobotiqExample(viewer, args)
        viewer.set_camera(wp.vec3(1.0, 1.1, 0.85), -24.0, -140.0)
        replay = KeyReplay(example) if args.replay else None
        if replay and args.num_frames < replay.total_frames:
            parser.error(f"Replay needs at least {replay.total_frames} frames")
        print("[KEYS] W/S:+/-X A/D:+/-Y Q/E:+/-Z; Z/X T/G C/V:world XYZ rotation; Shift:fine; K:grip R:reset Space:pause", flush=True)
        while viewer.is_running() and (args.keep_open or example.episode_steps < args.num_frames):
            started = time.perf_counter()
            if keyboard.reset_requested:
                keyboard.reset(replay=args.replay)
                example.reset()
                replay = KeyReplay(example) if args.replay else None
                print("[RESET] Fresh physical episode, solver, keyboard and IK state.", flush=True)
            if viewer.should_step():
                if replay:
                    replay.advance(keyboard)
                example.step(keyboard)
                if replay and replay.label == "hold":
                    lift = example.state_0.body_q.numpy()[example.cube_indices[0], 2] - example.pick_positions[0, 2]
                    example.hold_min_lift = min(example.hold_min_lift, float(lift))
                if replay and replay.finished:
                    keyboard.clear()
                    assert keyboard.toggle_count == 2, "K repeat/release handling failed"
                    example.finished = True
            example.render()
            if example.finished:
                break
            if args.viewer == "gl":
                # Pace against elapsed work; physics/control dt stays 1/60.
                time.sleep(max(0.0, example.frame_dt - (time.perf_counter() - started)))
        if args.test:
            example.test_final()
        print(f"[DONE] {example.episode_steps} frames; mimic max={example.max_mimic_error:.6f} rad", flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()