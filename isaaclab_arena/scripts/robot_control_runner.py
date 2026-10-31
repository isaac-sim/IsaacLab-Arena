# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute declarative absolute-pose robot-control recipes in an Arena environment."""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import sys
from pathlib import Path
from typing import Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

_VIDEO_FPS = 20


class _TaskSucceeded(Exception):
    """End recipe execution when a control step reports task success."""


_POSE_FIELDS = {
    "values",
    "position",
    "orientation",
    "gripper",
    "reference",
    "reference_body",
    "reference_point",
    "offset",
    "offset_frame",
    "orientation_frame",
    "interpolation_steps",
    "planner",
    "planner_max_attempts",
    "planner_waypoint_stride",
    "planner_steps_per_waypoint",
    "max_steps",
    "accept_blocked",
    "accept_timeout",
}
_COMMAND_FIELDS = {
    "pose": (set(), _POSE_FIELDS),
    "gripper": ({"value"}, {"max_steps", "accept_blocked"}),
    "trajectory": (set(), {"waypoints", "values", "max_steps", "accept_blocked"}),
    "wait": (set(), {"steps", "seconds"}),
    "mark": ({"index", "boundary", "title"}, set()),
    "check": ({"index"}, {"timeout_steps"}),
    "wait_final": (set(), {"timeout_steps"}),
    "inspect": (set(), set()),
    "finish": (set(), set()),
}


def _assert_number(value: Any, field: str, line_number: int) -> None:
    assert isinstance(value, int | float) and not isinstance(
        value, bool
    ), f"Recipe line {line_number}: '{field}' must be a number"
    assert math.isfinite(value), f"Recipe line {line_number}: '{field}' must be finite"


def _validate_vector(value: Any, length: int, field: str, line_number: int) -> None:
    assert (
        isinstance(value, list) and len(value) == length
    ), f"Recipe line {line_number}: '{field}' must contain {length} numbers"
    for component in value:
        _assert_number(component, field, line_number)


def _validate_action_values(values: Any, field: str, line_number: int) -> None:
    _validate_vector(values, 8, field, line_number)
    assert np.isclose(
        np.linalg.norm(values[3:7]), 1.0, atol=1e-3
    ), f"Recipe line {line_number}: '{field}' must contain a unit XYZW quaternion"
    assert values[7] in (0, 1), f"Recipe line {line_number}: '{field}' gripper value must be 0 or 1"


def _validate_pose_fields(command: dict[str, Any], line_number: int, *, waypoint: bool = False) -> None:
    prefix = "trajectory waypoint" if waypoint else "pose"
    unknown = set(command) - _POSE_FIELDS - (set() if waypoint else {"action"})
    assert not unknown, f"Recipe line {line_number}: {prefix} has unknown fields {sorted(unknown)}"
    if "values" in command:
        assert not (
            set(command) & (_POSE_FIELDS - {"values", "interpolation_steps", "max_steps", "accept_blocked"})
        ), f"Recipe line {line_number}: '{prefix}' values cannot be combined with structured target fields"
        _validate_action_values(command["values"], "values", line_number)
        return
    assert (
        "position" in command or "reference" in command
    ), f"Recipe line {line_number}: {prefix} requires 'values', 'position', or 'reference'"
    assert not (
        "position" in command and "reference" in command
    ), f"Recipe line {line_number}: {prefix} accepts exactly one of 'position' and 'reference'"
    if "position" in command:
        _validate_vector(command["position"], 3, "position", line_number)
    if "orientation" in command:
        _validate_vector(command["orientation"], 4, "orientation", line_number)
        assert np.isclose(
            np.linalg.norm(command["orientation"]), 1.0, atol=1e-3
        ), f"Recipe line {line_number}: 'orientation' must be a unit XYZW quaternion"
    if "offset" in command:
        _validate_vector(command["offset"], 3, "offset", line_number)
    if "gripper" in command:
        assert command["gripper"] in (0, 1), f"Recipe line {line_number}: 'gripper' must be 0 or 1"
    for field in ("reference", "reference_body"):
        if field in command:
            assert (
                isinstance(command[field], str) and command[field]
            ), f"Recipe line {line_number}: '{field}' must be a non-empty string"
    assert (
        "reference_body" not in command or "reference" in command
    ), f"Recipe line {line_number}: 'reference_body' requires 'reference'"
    assert not (
        "reference_body" in command and command.get("reference_point", "position") == "aabb_center"
    ), f"Recipe line {line_number}: 'reference_body' cannot use an AABB center"
    assert (
        "offset" not in command or "reference" in command
    ), f"Recipe line {line_number}: 'offset' requires 'reference'"
    if "reference_point" in command:
        assert command["reference_point"] in {
            "position",
            "aabb_center",
        }, f"Recipe line {line_number}: 'reference_point' must be 'position' or 'aabb_center'"
        assert "reference" in command, f"Recipe line {line_number}: 'reference_point' requires 'reference'"
    if "offset_frame" in command:
        assert "reference" in command, f"Recipe line {line_number}: 'offset_frame' requires 'reference'"
        assert command["offset_frame"] in {
            "reference",
            "world",
        }, f"Recipe line {line_number}: 'offset_frame' must be 'reference' or 'world'"
    if "orientation_frame" in command:
        assert command["orientation_frame"] in {
            "root",
            "reference",
            "world",
        }, f"Recipe line {line_number}: 'orientation_frame' must be 'root', 'reference', or 'world'"
        assert "orientation" in command, f"Recipe line {line_number}: 'orientation_frame' requires 'orientation'"
        if command["orientation_frame"] == "reference":
            assert "reference" in command, f"Recipe line {line_number}: reference orientation requires 'reference'"


def _validate_command(command: Any, line_number: int) -> dict[str, Any]:
    """Validate and return one JSONL robot command."""
    assert isinstance(command, dict), f"Recipe line {line_number}: each line must contain a JSON object"
    assert isinstance(command.get("action"), str), f"Recipe line {line_number}: missing string field 'action'"
    action = command["action"]
    assert action in _COMMAND_FIELDS, f"Recipe line {line_number}: unknown action '{action}'"
    required, optional = _COMMAND_FIELDS[action]
    fields = set(command) - {"action"}
    missing = required - fields
    unknown = fields - required - optional
    assert not missing, f"Recipe line {line_number}: action '{action}' is missing {sorted(missing)}"
    assert not unknown, f"Recipe line {line_number}: action '{action}' has unknown fields {sorted(unknown)}"

    if action == "pose":
        _validate_pose_fields(command, line_number)
    if action == "gripper":
        assert command["value"] in (0, 1), f"Recipe line {line_number}: 'value' must be 0 or 1"
    if action == "trajectory":
        assert ("waypoints" in command) != (
            "values" in command
        ), f"Recipe line {line_number}: trajectory requires exactly one of 'waypoints' and 'values'"
        if "values" in command:
            assert command["values"], f"Recipe line {line_number}: trajectory 'values' must be non-empty"
            for values in command["values"]:
                _validate_action_values(values, "values", line_number)
        else:
            assert (
                isinstance(command["waypoints"], list) and command["waypoints"]
            ), f"Recipe line {line_number}: trajectory 'waypoints' must be a non-empty list"
            for waypoint in command["waypoints"]:
                assert isinstance(waypoint, dict), f"Recipe line {line_number}: each waypoint must be an object"
                _validate_pose_fields(waypoint, line_number, waypoint=True)
    if action == "wait":
        assert ("steps" in command) != (
            "seconds" in command
        ), f"Recipe line {line_number}: wait requires exactly one of 'steps' and 'seconds'"
    for field in ("index", "steps", "timeout_steps", "max_steps", "interpolation_steps"):
        if field in command:
            assert (
                isinstance(command[field], int) and not isinstance(command[field], bool) and command[field] >= 0
            ), f"Recipe line {line_number}: '{field}' must be a non-negative integer"
    if action == "trajectory":
        for waypoint in command.get("waypoints", ()):
            if "max_steps" in waypoint:
                assert (
                    isinstance(waypoint["max_steps"], int)
                    and not isinstance(waypoint["max_steps"], bool)
                    and waypoint["max_steps"] >= 0
                ), f"Recipe line {line_number}: waypoint 'max_steps' must be a non-negative integer"
            if "accept_blocked" in waypoint:
                assert isinstance(
                    waypoint["accept_blocked"], bool
                ), f"Recipe line {line_number}: waypoint 'accept_blocked' must be a boolean"
    if "seconds" in command:
        _assert_number(command["seconds"], "seconds", line_number)
        assert command["seconds"] >= 0, f"Recipe line {line_number}: 'seconds' must be non-negative"
    for field in ("accept_blocked",):
        if field in command:
            assert isinstance(command[field], bool), f"Recipe line {line_number}: '{field}' must be a boolean"
    for field in ("boundary", "title"):
        if field in command:
            assert (
                isinstance(command[field], str) and command[field]
            ), f"Recipe line {line_number}: '{field}' must be a non-empty string"
    return command


def load_robot_control_recipe(path: Path) -> tuple[dict[str, Any], ...]:
    """Load and validate a robot-control JSONL recipe."""
    commands = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw_command = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"Recipe line {line_number}: invalid JSON: {error.msg}") from error
        commands.append(_validate_command(raw_command, line_number))
    assert commands, f"Recipe '{path}' has no commands"
    return tuple(commands)


def _quat_conjugate(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]], dtype=np.float64)


def _quat_multiply(q_A_B: np.ndarray, q_B_C: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q_A_B
    x2, y2, z2, w2 = q_B_C
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _rotate_vector(q_A_B: np.ndarray, vector_B: np.ndarray) -> np.ndarray:
    vector_quaternion = np.concatenate((vector_B, [0.0]))
    return _quat_multiply(_quat_multiply(q_A_B, vector_quaternion), _quat_conjugate(q_A_B))[:3]


def _world_pose_to_root(
    T_W_R: np.ndarray, t_W_G: np.ndarray, q_W_G: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a world-frame goal pose into robot-root coordinates."""
    q_R_W = _quat_conjugate(T_W_R[3:])
    t_R_G = _rotate_vector(q_R_W, t_W_G - T_W_R[:3])
    q_R_G = None if q_W_G is None else _quat_multiply(q_R_W, q_W_G)
    return t_R_G, q_R_G


class RobotCommandRunner:
    """Execute absolute-pose commands and evaluate the environment's task predicates."""

    def __init__(self, env, args: argparse.Namespace, embodiment=None):
        from isaaclab_arena_vlm_agent.droid_adapter import DroidVLMAdapter

        self.env = env
        self.base = env.unwrapped
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.adapter = DroidVLMAdapter().make_goal_action_adapter(
            position_tolerance_m=args.position_tolerance_m,
            rotation_tolerance_rad=args.rotation_tolerance_rad,
        )
        assert self.adapter.is_robot_action_space_supported(
            env
        ), "robot_control_runner requires the droid_abs_ik 8D absolute pose and gripper action layout"
        self.motion_generator = self.adapter.motion_generator
        self.indices = self.adapter._command_indices
        self.embodiment = embodiment
        self.curobo_planner = None
        self.steps = 0
        self.frames = 0
        self.events = []
        self.pose_results = []
        self._success_report = None
        self.current_subtask_index = None
        self.current_subtask_title = "Preparing robot-control recipe"
        self.last_image = None
        self.writer = None
        self.viewport = None
        self.last_target = self._current_target()
        if not args.no_record:
            import imageio.v2 as imageio
            from omni.kit.viewport.utility import get_active_viewport

            self.viewport = get_active_viewport()
            self.viewport.resolution = (960, 720)
            self.writer = imageio.get_writer(str(self.output / "demo.mp4"), fps=_VIDEO_FPS, codec="libx264", quality=8)
            for _ in range(30):
                self.base.sim.render()

    def _current_target(self) -> np.ndarray:
        target = np.zeros(self.indices.action_dim, dtype=np.float32)
        for pose in self.indices.poses:
            measured = self.base.arena_world.get_body_pose_in_root(pose.asset_name, pose.body_name)
            target[pose.indices] = measured[0].detach().cpu().numpy()
        for gripper in self.indices.grippers:
            target[gripper.index] = float(self.base.action_manager.action[0, gripper.index])
        return target

    def _decode_values(self, values: list[float]) -> np.ndarray:
        from isaaclab_arena_vlm_agent.commands import Command

        return self.adapter.decode_command(Command(values=values), self.base.arena_world, 0)

    def _reference_pose_w(self, command: dict[str, Any]) -> np.ndarray:
        name = command["reference"]
        if "reference_body" in command:
            T_W_reference = self.base.arena_world.get_body_pose_w(name, command["reference_body"])[0]
            return T_W_reference.detach().cpu().numpy()
        T_W_reference = self.base.arena_world.get_pose_w(name)[0].detach().cpu().numpy()
        if command.get("reference_point", "position") == "aabb_center":
            center_W = self.base.arena_world.get_aabb_w(name).center[0].detach().cpu().numpy()
            T_W_reference[:3] = center_W
        return T_W_reference

    def _target_from_pose_command(self, command: dict[str, Any]) -> np.ndarray:
        if "values" in command:
            return self._decode_values(command["values"])
        target = self.last_target.copy()
        pose = self.indices.poses[0]
        gripper = self.indices.grippers[0]
        current = target[pose.indices]
        position_R = np.asarray(command.get("position", current[:3]), dtype=np.float64)
        orientation_R = np.asarray(command.get("orientation", current[3:]), dtype=np.float64)
        T_W_R = self.base.arena_world.get_pose_w(pose.asset_name)[0].detach().cpu().numpy()
        if "reference" in command:
            T_W_reference = self._reference_pose_w(command)
            offset = np.asarray(command.get("offset", (0.0, 0.0, 0.0)), dtype=np.float64)
            if command.get("offset_frame", "reference") == "reference":
                offset = _rotate_vector(T_W_reference[3:], offset)
            t_W_goal = T_W_reference[:3] + offset
            position_R, _ = _world_pose_to_root(T_W_R, t_W_goal)
            if command.get("orientation_frame", "root") == "reference":
                q_W_goal = _quat_multiply(T_W_reference[3:], orientation_R)
                _, orientation_R = _world_pose_to_root(T_W_R, t_W_goal, q_W_goal)
        if command.get("orientation_frame") == "world":
            _, orientation_R = _world_pose_to_root(T_W_R, T_W_R[:3], orientation_R)
        target[pose.indices] = np.concatenate((position_R, orientation_R))
        target[gripper.index] = command.get("gripper", target[gripper.index])
        return self._decode_values(target.tolist())

    def _capture_frame(self) -> None:
        import ctypes

        from omni.kit.viewport.utility import capture_viewport_to_buffer

        captured = []

        def receive(buffer, buffer_size, width, height, byte_format):
            get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
            get_pointer.restype = ctypes.c_void_p
            get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            pointer = ctypes.cast(get_pointer(buffer, None), ctypes.POINTER(ctypes.c_uint8))
            pixels = np.ctypeslib.as_array(pointer, shape=(buffer_size,))
            captured.append(pixels.reshape(height, width, 4)[:, :, :3].copy())

        capture_viewport_to_buffer(self.viewport, receive)
        for _ in range(30):
            self.base.sim.render()
            if captured:
                self.last_image = self._overlay_progress_banner(captured[0])
                self.writer.append_data(self.last_image)
                self.frames += 1
                return
        raise RuntimeError("Viewport capture did not return an image.")

    def _overlay_progress_banner(self, frame):
        from PIL import Image, ImageDraw, ImageFont

        completion = self._task_completion()[0].tolist()
        labels = []
        for index, complete in enumerate(completion):
            status = "done" if complete else "active" if index == self.current_subtask_index else "pending"
            labels.append(f"{index + 1}:{status}")
        image = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, image.width - 1, 75), fill=(8, 12, 18, 205))
        try:
            title_font = ImageFont.truetype("DejaVuSans.ttf", 24)
            progress_font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            title_font = ImageFont.load_default()
            progress_font = title_font
        if self.current_subtask_index is None:
            title = self.current_subtask_title
        else:
            title = f"Subtask {self.current_subtask_index + 1}/{len(completion)}  |  {self.current_subtask_title}"
        draw.text((18, 10), title, fill=(255, 255, 255, 255), font=title_font)
        draw.text((18, 45), "Subtasks  " + "  ".join(labels), fill=(136, 220, 170, 255), font=progress_font)
        return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))

    def _step_target(self, target: np.ndarray) -> None:
        action = self.adapter.command_to_action(self.env, target).unsqueeze(0)
        self.last_target = target.copy()
        self.steps += 1
        _, _, terminated, _, _ = self.env.step(action)
        if terminated.all().item() and self.base.termination_manager.get_term("success").all().item():
            raise _TaskSucceeded
        if self.viewport is not None:
            frames_due = round(self.steps * self.base.step_dt * _VIDEO_FPS) - self.frames
            if frames_due > 0:
                self._capture_frame()
                for _ in range(frames_due - 1):
                    self.writer.append_data(self.last_image)
                    self.frames += 1

    def _task_completion(self):
        tracker = self.base.progress_tracker
        if all(criteria.parent_subtask_idx is None for criteria in tracker.completion_criteria):
            return tracker.is_complete()[:, None]
        return tracker.get_subtask_completion()

    def _criterion_status(self) -> dict[str, list]:
        status = {}
        for criterion in self.base.cfg.task_progress.success_criteria:
            values = []
            for predicate in criterion.predicate_sequence:
                evaluate = getattr(predicate, "predicate", predicate)
                values.append(evaluate(self.base).tolist())
            status[criterion.name] = values
        return status

    def pose(self, command: dict[str, Any]) -> str:
        target = self._target_from_pose_command(command)
        if command.get("planner") == "curobo":
            self._execute_curobo_path(target, command)
        interpolation_steps = command.get("interpolation_steps", 0)
        if interpolation_steps:
            start = self._current_target()
            pose_indices = self.indices.poses[0].indices
            q_start = start[pose_indices][3:]
            q_target = target[pose_indices][3:]
            if np.dot(q_start, q_target) < 0:
                q_target = -q_target
            for index in range(interpolation_steps):
                alpha = (index + 1) / interpolation_steps
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                waypoint = (1.0 - alpha) * start + alpha * target
                quaternion = (1.0 - alpha) * q_start + alpha * q_target
                quaternion /= np.linalg.norm(quaternion)
                waypoint[pose_indices.start + 3 : pose_indices.stop] = quaternion
                self._step_target(waypoint)
        max_steps = command.get("max_steps", self.args.max_pose_steps)
        outcome = "timeout"
        for _ in range(max_steps):
            generated = self.motion_generator.compute_command(target, self.env, 0)
            self._step_target(generated)
            if self.motion_generator.goal_reached(target, self.env, 0):
                outcome = "reached"
                break
            if self.motion_generator.goal_blocked(target, self.env, 0):
                outcome = "blocked"
                break
        result = {
            "outcome": outcome,
            "steps": self.steps,
            "target": target.tolist(),
            "measured": self._current_target().tolist(),
        }
        self.pose_results.append(result)
        print("ROBOT_POSE_RESULT", json.dumps(result), flush=True)
        if outcome != "reached":
            accepted = (outcome == "blocked" and command.get("accept_blocked", False)) or (
                outcome == "timeout" and command.get("accept_timeout", False)
            )
            assert accepted, f"Robot pose goal ended with outcome '{outcome}': {command}"
        return outcome

    def _execute_curobo_path(self, target: np.ndarray, command: dict[str, Any]) -> None:
        if self.curobo_planner is None:
            from isaaclab_arena_curobo.embodiment_curobo_registry import get_embodiment_curobo_cfg
            from isaaclab_arena_curobo.motion_planner import CuroboPosePlanner

            assert self.embodiment is not None, "cuRobo planning requires the Arena embodiment"
            self.curobo_planner = CuroboPosePlanner(
                get_embodiment_curobo_cfg(self.embodiment),
                self.base.device,
            )
        robot = self.base.scene["robot"]
        joint_ids = [robot.joint_names.index(name) for name in self.curobo_planner.joint_names]
        joint_positions = robot.data.joint_pos.torch[0, joint_ids]
        pose = self.indices.poses[0]
        target_pose = target[pose.indices]
        current_pose = self._current_target()[pose.indices]
        waypoints = self.curobo_planner.plan_pose_waypoints(
            joint_positions,
            current_pose[:3],
            current_pose[3:],
            target_pose[:3],
            target_pose[3:],
            max_attempts=command.get("planner_max_attempts", 5),
        )
        print(
            "CUROBO_POSE_PATH",
            json.dumps({
                "waypoint_count": len(waypoints),
                "first": waypoints[0].tolist(),
                "last": waypoints[-1].tolist(),
            }),
            flush=True,
        )
        stride = command.get("planner_waypoint_stride", 4)
        selected_indices = list(range(0, len(waypoints), stride))
        if selected_indices[-1] != len(waypoints) - 1:
            selected_indices.append(len(waypoints) - 1)
        for index in selected_indices:
            waypoint = target.copy()
            waypoint[pose.indices] = waypoints[index]
            for _ in range(command.get("planner_steps_per_waypoint", 2)):
                self._step_target(waypoint)

    def gripper(self, command: dict[str, Any]) -> str:
        pose_command = {
            "position": self.last_target[self.indices.poses[0].indices][:3].tolist(),
            "orientation": self.last_target[self.indices.poses[0].indices][3:].tolist(),
            "gripper": command["value"],
            "max_steps": command.get("max_steps", self.args.max_pose_steps),
            "accept_blocked": command.get("accept_blocked", False),
        }
        return self.pose(pose_command)

    def trajectory(self, command: dict[str, Any]) -> None:
        if "values" in command:
            waypoints = [{"values": values} for values in command["values"]]
        else:
            waypoints = command["waypoints"]
        defaults = {field: command[field] for field in ("max_steps", "accept_blocked") if field in command}
        for waypoint in waypoints:
            self.pose({**defaults, **waypoint})

    def wait(self, count: int) -> None:
        for _ in range(count):
            self._step_target(self.last_target)

    def mark(self, index: int, boundary: str, title: str) -> None:
        import imageio.v3 as iio

        self.current_subtask_index = index
        self.current_subtask_title = title
        if self.viewport is not None:
            self._capture_frame()
        record = {
            "subtask": index,
            "boundary": boundary,
            "title": title,
            "simulation_time_s": self.steps * self.base.step_dt,
            "completion": self._task_completion()[0].tolist(),
        }
        if self.last_image is not None:
            filename = f"{index:02d}_{boundary}.png"
            iio.imwrite(self.output / filename, self.last_image)
            record["viewport"] = filename
        self.events.append(record)
        print("SUBTASK", json.dumps(record), flush=True)
        (self.output / "progress.json").write_text(json.dumps(self.events, indent=2))

    def check(self, index: int, timeout_steps: int = 600) -> None:
        for _ in range(timeout_steps):
            if self._task_completion()[0, index].item():
                return
            self.wait(1)
        raise AssertionError(
            f"Subtask {index} did not complete: {self._task_completion()[0].tolist()}; "
            f"current predicates: {self._criterion_status()}"
        )

    def wait_final(self, timeout_steps: int = 1200) -> None:
        from isaaclab_arena.tasks.terminations import task_success_from_progress

        for _ in range(timeout_steps):
            if task_success_from_progress(self.base).all().item():
                return
            self.wait(1)

    def inspect(self) -> None:
        print(
            "ROBOT_CONTROL_STATE",
            json.dumps({
                "target": self.last_target.tolist(),
                "measured_pose": self._current_target().tolist(),
                "completion": self._task_completion().tolist(),
                "current_predicates": self._criterion_status(),
            }),
            flush=True,
        )

    def _execute(self, command: dict[str, Any]) -> bool:
        action = command["action"]
        if action == "pose":
            self.pose(command)
        elif action == "gripper":
            self.gripper(command)
        elif action == "trajectory":
            self.trajectory(command)
        elif action == "wait":
            count = command.get("steps")
            if count is None:
                count = int(float(command["seconds"]) / self.base.step_dt)
            self.wait(count)
        elif action == "mark":
            self.mark(command["index"], command["boundary"], command["title"])
        elif action == "check":
            self.check(command["index"], command.get("timeout_steps", 600))
        elif action == "wait_final":
            self.wait_final(command.get("timeout_steps", 1200))
        elif action == "inspect":
            self.inspect()
        elif action == "finish":
            return False
        else:
            raise AssertionError(f"Unhandled action '{action}'")
        return True

    def _report_state(self) -> dict[str, Any]:
        from isaaclab_arena.tasks.terminations import task_success_from_progress

        success = task_success_from_progress(self.base).tolist()
        return {
            "mode": "robot_control_recipe",
            "success": success,
            "subtasks": self._task_completion().tolist(),
            "current_predicates": self._criterion_status(),
            "video_frames": self.frames,
            "steps": self.steps,
            "simulation_time_s": self.steps * self.base.step_dt,
            "final_target": self.last_target.tolist(),
            "final_measured_pose": self._current_target().tolist(),
            "events": self.events,
            "pose_results": self.pose_results,
            "episode_index": self.base.get_episode_index(0),
            "articulation_joint_positions": {
                name: dict(zip(asset.joint_names, asset.data.joint_pos.torch[0].tolist()))
                for name, asset in self.base.scene.articulations.items()
            },
        }

    def record_success_before_reset(self) -> None:
        """Save the successful episode's state and final frame before automatic reset."""
        if self.viewport is not None:
            # Flush the viewport pipeline before requesting the terminal image.
            # Rendering updates transforms without advancing physics or reset.
            for _ in range(2):
                self.base.sim.render()
        self.pose_results.append({
            "outcome": "task_success",
            "steps": self.steps,
            "target": self.last_target.tolist(),
            "measured": self._current_target().tolist(),
        })
        if self.current_subtask_index is not None:
            self.mark(self.current_subtask_index, "end", self.current_subtask_title)
        elif self.viewport is not None:
            self._capture_frame()
        self._success_report = self._report_state()

    def _write_report(self, error: str | None = None) -> dict[str, Any]:
        report = self._success_report if self._success_report is not None else self._report_state()
        report["execution_error"] = error
        report["stopped_on_success"] = self._success_report is not None
        (self.output / "validation.json").write_text(json.dumps(report, indent=2))
        print("TASK_SUCCESS" if all(report["success"]) else "TASK_INCOMPLETE", json.dumps(report), flush=True)
        return report

    def run(self, commands: tuple[dict[str, Any], ...]) -> None:
        error = None
        try:
            self.wait(self.args.settle_steps)
            for command in commands:
                print("ROBOT_CONTROL_COMMAND", json.dumps(command), flush=True)
                if not self._execute(command):
                    break
        except _TaskSucceeded:
            pass
        except Exception as exception:
            error = str(exception)
            raise
        finally:
            try:
                if error is not None and self.current_subtask_index is not None:
                    self.mark(self.current_subtask_index, "failure", self.current_subtask_title)
                report = self._write_report(error)
            finally:
                if self.writer is not None:
                    self.writer.close()
        if not self.args.allow_incomplete:
            assert all(report["success"]), f"Final task conditions failed: {report}"


def _assert_runner_args(args: argparse.Namespace) -> None:
    assert args.num_envs == 1, "robot_control_runner supports exactly one environment"
    assert not args.distributed, "robot_control_runner does not support distributed execution"
    assert not args.list_variations, "robot_control_runner does not support --list_variations"
    if not args.no_record:
        assert args.visualizer is not None and "kit" in args.visualizer, "recording requires the Kit visualizer"


def _add_runner_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("DROID Absolute-IK Robot-Control Runner")
    group.add_argument("--commands", type=Path, required=True, help="JSONL robot-control recipe.")
    group.add_argument("--output", type=Path, default=Path("artifacts/robot_control"))
    group.add_argument("--no_record", action="store_true", help="Do not capture viewport video or marker images.")
    group.add_argument("--allow_incomplete", action="store_true", help="Write a report without requiring task success.")
    group.add_argument("--settle_steps", type=int, default=30)
    group.add_argument("--max_pose_steps", type=int, default=120)
    group.add_argument("--position_tolerance_m", type=float, default=0.01)
    group.add_argument("--rotation_tolerance_rad", type=float, default=0.1)


def _run_recipe(builder, args: argparse.Namespace, commands: tuple[dict[str, Any], ...]) -> None:
    """Build one wrapped environment, execute its recipe, and always close it."""
    from types import SimpleNamespace

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    runner = None

    def record_success(env, env_id):
        if runner is not None and env.termination_manager.get_term("success")[env_id].item():
            runner.record_success_before_reset()
        return {}

    env_cfg, env_kwargs = builder.build_env_cfg()
    env_cfg.sim.enable_scene_query_support = True
    env_cfg.recorders = {}
    env_cfg.episode_recorders = SimpleNamespace(robot_control=EpisodeRecorderTermCfg(func=record_success))
    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.reset()
        arena_env = getattr(builder, "arena_env", None)
        runner = RobotCommandRunner(env, args, getattr(arena_env, "embodiment", None))
        runner.run(commands)
    finally:
        env.close()


def main() -> None:
    """Launch an Arena environment and execute a robot-control recipe."""
    parser = get_isaaclab_arena_cli_parser()
    _add_runner_args(parser)
    parser.set_defaults(device="cpu", visualizer=["kit"])
    parser.allow_abbrev = False
    original_cli_args = sys.argv[1:]
    app_args, _ = parser.parse_known_args(original_cli_args)
    _assert_runner_args(app_args)

    with SimulationAppContext(app_args):
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args, hydra_overrides = parser.parse_known_args(original_cli_args)
        assert_hydra_overrides(hydra_overrides, parser)
        _assert_runner_args(args)
        commands = load_robot_control_recipe(args.commands)
        builder = get_arena_builder_from_cli(args, hydra_overrides=hydra_overrides)
        _run_recipe(builder, args, commands)


if __name__ == "__main__":
    main()
