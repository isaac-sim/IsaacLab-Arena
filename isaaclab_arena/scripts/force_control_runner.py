# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute reusable physical force-control recipes in an Arena environment."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

_VECTOR_FIELDS = {"offset", "orientation"}
_COMMAND_FIELDS = {
    "setup_joint": ({"name", "joint_name"}, {"effort_limit", "replace_existing"}),
    "move": (
        {"name"},
        {
            "goal",
            "reference",
            "reference_point",
            "reference_axes",
            "offset",
            "orientation",
            "seconds",
            "linear_gain",
            "linear_damping",
            "minimum_stiffness",
            "max_force",
            "rotation_mode",
            "rotation_gain",
            "angular_damping",
            "max_torque",
        },
    ),
    "lift": (
        {"name", "height"},
        {
            "seconds",
            "linear_gain",
            "linear_damping",
            "minimum_stiffness",
            "max_force",
            "rotation_mode",
            "rotation_gain",
            "angular_damping",
            "max_torque",
        },
    ),
    "joint": (
        {"name", "joint_name"},
        {"goal", "normalized_goal", "seconds", "gain", "damping", "limit"},
    ),
    "release": ({"name"}, set()),
    "wait": (set(), {"steps", "seconds"}),
    "mark": ({"index", "boundary", "title"}, set()),
    "check": ({"index"}, {"timeout_steps"}),
    "wait_final": (set(), {"timeout_steps"}),
    "inspect": (set(), set()),
    "finish": (set(), set()),
}


@dataclass(frozen=True)
class ForceControlRecipe:
    """Validated setup and runtime commands loaded from JSONL."""

    joint_setups: tuple[dict[str, Any], ...]
    commands: tuple[dict[str, Any], ...]


@dataclass
class PoseHold:
    """Pose target and controller gains for one rigid object."""

    t_W_goal: Any
    q_W_goal: Any
    linear_gain: float = 150.0
    linear_damping: float = 24.0
    minimum_stiffness: float = 0.0
    max_force: float = 100.0
    rotation_mode: str = "acceleration"
    rotation_gain: float = 100.0
    angular_damping: float = 20.0
    max_torque: float = 5.0


def _assert_number(value: Any, field: str, line_number: int) -> None:
    assert isinstance(value, int | float) and not isinstance(
        value, bool
    ), f"Recipe line {line_number}: '{field}' must be a number"


def _validate_command(command: Any, line_number: int) -> dict[str, Any]:
    """Validate and return one JSONL command."""
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

    for field in ("name", "joint_name", "reference", "reference_point", "rotation_mode", "title", "boundary"):
        if field in command:
            assert (
                isinstance(command[field], str) and command[field]
            ), f"Recipe line {line_number}: '{field}' must be a non-empty string"
    for field in (
        "height",
        "normalized_goal",
        "seconds",
        "linear_gain",
        "linear_damping",
        "minimum_stiffness",
        "max_force",
        "rotation_gain",
        "angular_damping",
        "max_torque",
        "gain",
        "damping",
        "limit",
        "effort_limit",
    ):
        if field in command:
            _assert_number(command[field], field, line_number)
    for field in ("index", "steps", "timeout_steps"):
        if field in command:
            assert (
                isinstance(command[field], int) and command[field] >= 0
            ), f"Recipe line {line_number}: '{field}' must be a non-negative integer"
    for field in _VECTOR_FIELDS & set(command):
        expected_length = 4 if field == "orientation" else 3
        value = command[field]
        assert (
            isinstance(value, list) and len(value) == expected_length
        ), f"Recipe line {line_number}: '{field}' must contain {expected_length} numbers"
        for component in value:
            _assert_number(component, field, line_number)
    if "reference_axes" in command:
        axes = command["reference_axes"]
        assert isinstance(axes, list) and axes, f"Recipe line {line_number}: 'reference_axes' must be a non-empty list"
        assert len(axes) == len(set(axes)) and set(axes) <= {
            "x",
            "y",
            "z",
        }, f"Recipe line {line_number}: 'reference_axes' must contain unique x, y, or z values"
    if action == "move":
        assert (
            "goal" in command or "reference" in command
        ), f"Recipe line {line_number}: move requires 'goal' or 'reference'"
        if "goal" in command:
            goal = command["goal"]
            assert (
                isinstance(goal, list) and len(goal) == 3
            ), f"Recipe line {line_number}: move 'goal' must contain 3 numbers"
            for component in goal:
                _assert_number(component, "goal", line_number)
        assert (
            "reference_axes" not in command or "reference" in command
        ), f"Recipe line {line_number}: 'reference_axes' requires 'reference'"
        if "reference_point" in command:
            assert command["reference_point"] in {
                "position",
                "aabb_center",
            }, f"Recipe line {line_number}: 'reference_point' must be 'position' or 'aabb_center'"
    if action == "joint":
        assert ("goal" in command) != (
            "normalized_goal" in command
        ), f"Recipe line {line_number}: joint requires exactly one of 'goal' and 'normalized_goal'"
        if "goal" in command:
            _assert_number(command["goal"], "goal", line_number)
        if "normalized_goal" in command:
            assert (
                0.0 <= command["normalized_goal"] <= 1.0
            ), f"Recipe line {line_number}: 'normalized_goal' must be between 0 and 1"
    if action == "wait":
        assert ("steps" in command) != (
            "seconds" in command
        ), f"Recipe line {line_number}: wait requires exactly one of 'steps' and 'seconds'"
    if "rotation_mode" in command:
        assert command["rotation_mode"] in {
            "acceleration",
            "torque",
        }, f"Recipe line {line_number}: 'rotation_mode' must be 'acceleration' or 'torque'"
    if "replace_existing" in command:
        assert isinstance(
            command["replace_existing"], bool
        ), f"Recipe line {line_number}: 'replace_existing' must be a boolean"
    return command


def load_force_control_recipe(path: Path, require_runtime_commands: bool = True) -> ForceControlRecipe:
    """Load and validate a force-control JSONL recipe."""
    joint_setups = []
    commands = []
    runtime_command_seen = False
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw_command = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"Recipe line {line_number}: invalid JSON: {error.msg}") from error
        command = _validate_command(raw_command, line_number)
        if command["action"] == "setup_joint":
            assert (
                not runtime_command_seen
            ), f"Recipe line {line_number}: setup_joint commands must precede runtime commands"
            joint_setups.append(command)
        else:
            runtime_command_seen = True
            commands.append(command)
    assert commands or not require_runtime_commands, f"Recipe '{path}' has no runtime commands"
    return ForceControlRecipe(tuple(joint_setups), tuple(commands))


def _apply_joint_setups(env_cfg, joint_setups: tuple[dict[str, Any], ...]) -> None:
    """Add effort-controlled actuators requested before environment creation."""
    if not joint_setups:
        return
    from isaaclab.actuators import ImplicitActuatorCfg

    for setup in joint_setups:
        name = setup["name"]
        assert hasattr(env_cfg.scene, name), f"Recipe setup refers to unknown scene entity '{name}'"
        asset_cfg = getattr(env_cfg.scene, name)
        existing_actuators = dict(asset_cfg.actuators or {})
        if setup.get("replace_existing", False):
            existing_actuators = {}
        actuator_name = "force_control_" + re.sub(r"\W+", "_", setup["joint_name"]).strip("_")
        assert (
            actuator_name not in existing_actuators
        ), f"Recipe defines duplicate actuator '{actuator_name}' for scene entity '{name}'"
        existing_actuators[actuator_name] = ImplicitActuatorCfg(
            joint_names_expr=[re.escape(setup["joint_name"])],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=float(setup.get("effort_limit", 100.0)),
        )
        asset_cfg.actuators = existing_actuators


def _combine_reference_goal(
    goal: list[float] | None,
    reference: list[float],
    offset: list[float],
    reference_axes: list[str] | None,
) -> list[float]:
    """Combine absolute and reference-relative components into a world-frame goal."""
    axes = {"x": 0, "y": 1, "z": 2}
    selected_axes = set(reference_axes or axes)
    assert goal is not None or len(selected_axes) == 3, "Unreferenced axes require an absolute goal"
    combined = list(goal or (0.0, 0.0, 0.0))
    for axis in selected_axes:
        index = axes[axis]
        combined[index] = reference[index] + offset[index]
    return combined


class PhysicalCommandRunner:
    """Execute physical commands and evaluate the environment's task predicates."""

    def __init__(self, env, args: argparse.Namespace):
        import torch

        self.env = env
        self.args = args
        self.torch = torch
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.steps = 0
        self.holds: dict[str, PoseHold] = {}
        self.joint_holds = {}
        self.frames = 0
        self.writer = None
        self.viewport = None
        self.last_image = None
        self.events = []
        self.last_wrenches = {}
        self.current_subtask_index = None
        self.current_subtask_title = "Preparing force-control recipe"
        if not args.no_record:
            import imageio.v2 as imageio
            from omni.kit.viewport.utility import get_active_viewport

            self.viewport = get_active_viewport()
            self.viewport.resolution = (960, 720)
            self.writer = imageio.get_writer(str(self.output / "demo.mp4"), fps=20, codec="libx264", quality=8)
            for _ in range(30):
                self.env.sim.render()

    def _capture_frame(self) -> None:
        import ctypes
        import numpy as np

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
            self.env.sim.render()
            if captured:
                self.last_image = self._overlay_progress_banner(captured[0])
                self.writer.append_data(self.last_image)
                self.frames += 1
                return
        raise RuntimeError("Viewport capture did not return an image.")

    def _overlay_progress_banner(self, frame):
        """Draw the current subtask and latched completion state on a viewport frame."""
        import numpy as np

        from PIL import Image, ImageDraw, ImageFont

        completion = self._task_completion()[0].tolist()
        status_labels = []
        for index, complete in enumerate(completion):
            if complete:
                status = "done"
            elif index == self.current_subtask_index:
                status = "active"
            else:
                status = "pending"
            status_labels.append(f"{index + 1}:{status}")
        progress = "Subtasks  " + "  ".join(status_labels)

        image = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        banner_height = 76
        draw.rectangle((0, 0, image.width - 1, banner_height - 1), fill=(8, 12, 18, 205))
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
        draw.text((18, 45), progress, fill=(136, 220, 170, 255), font=progress_font)
        return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))

    def _apply_forces(self) -> None:
        from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse

        torch = self.torch
        for name, hold in self.holds.items():
            obj = self.env.scene[name]
            T_W_O = self.env.arena_world.get_pose_w(name)
            v_W = self.env.arena_world.get_root_linear_velocity_w(name)
            omega_W = self.env.arena_world.get_root_angular_velocity_w(name)
            _, rotation_error_W = compute_pose_error(T_W_O[:, :3], T_W_O[:, 3:], hold.t_W_goal, hold.q_W_goal)
            masses = obj.data.body_mass.torch
            mass = masses.sum().item()
            stiffness = max(hold.linear_gain * mass, hold.minimum_stiffness)
            if hold.minimum_stiffness > 0:
                damping = 2 * (stiffness * mass) ** 0.5
            else:
                damping = hold.linear_damping * mass
            force_W = stiffness * (hold.t_W_goal - T_W_O[:, :3]) - damping * v_W
            force_W[:, 2] += mass * 9.81
            force_W = force_W.clamp(-hold.max_force, hold.max_force)

            if hold.rotation_mode == "torque":
                torque_W = (hold.rotation_gain * rotation_error_W - hold.angular_damping * omega_W).clamp(
                    -hold.max_torque, hold.max_torque
                )
            else:
                q_W_B = T_W_O[:, 3:]
                acceleration_B = quat_apply_inverse(
                    q_W_B, hold.rotation_gain * rotation_error_W - hold.angular_damping * omega_W
                )
                inertia_B = obj.data.body_inertia.torch.reshape(1, -1, 3, 3).sum(dim=1)
                torque_B = torch.bmm(inertia_B, acceleration_B[:, :, None]).squeeze(-1)
                torque_W = quat_apply(q_W_B, torque_B).clamp(-hold.max_torque, hold.max_torque)

            self.last_wrenches[name] = {
                "goal": hold.t_W_goal.tolist(),
                "velocity": v_W.tolist(),
                "force": force_W.tolist(),
                "mass": mass,
            }
            forces_W = masses[:, :, None] / mass * force_W[:, None, :]
            torques_W = torch.zeros_like(forces_W)
            torques_W[:, 0, :] = torque_W
            obj.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces_W,
                torques=torques_W,
                is_global=True,
            )

        for (name, joint_name), (goal, gain, damping, limit) in self.joint_holds.items():
            obj = self.env.scene[name]
            joint_id = obj.data.joint_names.index(joint_name)
            q = obj.data.joint_pos.torch[:, joint_id]
            qd = obj.data.joint_vel.torch[:, joint_id]
            effort = (gain * (goal - q) - damping * qd).clamp(-limit, limit)
            obj.set_joint_effort_target(effort[:, None], joint_ids=[joint_id])

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self._apply_forces()
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            self.env.scene.update(self.env.physics_dt)
            self.steps += 1
            if self.steps % self.env.cfg.decimation == 0:
                self.env.common_step_counter += 1
                self.env.episode_length_buf += 1
                self.env.update_task_progress()
            if self.steps % 6 == 0 and self.viewport is not None:
                self._capture_frame()

    def _task_completion(self):
        tracker = self.env.progress_tracker
        if all(criteria.parent_subtask_idx is None for criteria in tracker.completion_criteria):
            return tracker.is_complete()[:, None]
        return tracker.get_subtask_completion()

    def _criterion_status(self) -> dict[str, list]:
        status = {}
        for criterion in self.env.cfg.task_progress.success_criteria:
            values = []
            for predicate in criterion.predicate_sequence:
                evaluate = getattr(predicate, "predicate", predicate)
                values.append(evaluate(self.env).tolist())
            status[criterion.name] = values
        return status

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
            "simulation_time_s": self.steps * self.env.physics_dt,
            "completion": self._task_completion()[0].tolist(),
        }
        if self.last_image is not None:
            filename = f"{index:02d}_{boundary}.png"
            iio.imwrite(self.output / filename, self.last_image)
            record["viewport"] = filename
        self.events.append(record)
        print("SUBTASK", json.dumps(record), flush=True)
        (self.output / "progress.json").write_text(json.dumps(self.events, indent=2))

    def _pose_hold_from_command(self, command: dict[str, Any], t_W_goal, q_W_goal) -> PoseHold:
        return PoseHold(
            t_W_goal=t_W_goal,
            q_W_goal=q_W_goal,
            linear_gain=float(command.get("linear_gain", 150.0)),
            linear_damping=float(command.get("linear_damping", 24.0)),
            minimum_stiffness=float(command.get("minimum_stiffness", 0.0)),
            max_force=float(command.get("max_force", 100.0)),
            rotation_mode=command.get("rotation_mode", "acceleration"),
            rotation_gain=float(command.get("rotation_gain", 100.0)),
            angular_damping=float(command.get("angular_damping", 20.0)),
            max_torque=float(command.get("max_torque", 5.0)),
        )

    def _move_goal(self, command: dict[str, Any]) -> Any:
        goal = command.get("goal")
        if "reference" not in command:
            return goal
        reference_name = command["reference"]
        if command.get("reference_point", "position") == "aabb_center":
            reference = self.env.arena_world.get_aabb_w(reference_name).center[0].tolist()
        else:
            reference = self.env.arena_world.get_position_w(reference_name)[0].tolist()
        offset = command.get("offset", [0.0, 0.0, 0.0])
        return _combine_reference_goal(goal, reference, offset, command.get("reference_axes"))

    def move(self, command: dict[str, Any]) -> None:
        torch = self.torch
        name = command["name"]
        T_W_start = self.env.arena_world.get_pose_w(name).clone()
        t_W_goal = torch.tensor([self._move_goal(command)], dtype=torch.float32, device=self.env.device)
        if "orientation" in command:
            q_W_goal = torch.tensor([command["orientation"]], dtype=torch.float32, device=self.env.device)
        elif name in self.holds:
            q_W_goal = self.holds[name].q_W_goal
        else:
            q_W_goal = T_W_start[:, 3:]
        count = max(1, int(float(command.get("seconds", 2.0)) / self.env.physics_dt))
        for index in range(count):
            alpha = (index + 1) / count
            alpha = alpha * alpha * (3 - 2 * alpha)
            t_W_interpolated = (1 - alpha) * T_W_start[:, :3] + alpha * t_W_goal
            self.holds[name] = self._pose_hold_from_command(command, t_W_interpolated, q_W_goal)
            self.step()
        self.step(60)
        print("MOVE_RESULT", name, self.env.arena_world.get_pose_w(name).tolist(), flush=True)

    def lift(self, command: dict[str, Any]) -> None:
        t_W = self.env.arena_world.get_position_w(command["name"])[0].tolist()
        move_command = dict(command)
        move_command["action"] = "move"
        move_command["goal"] = [t_W[0], t_W[1], command["height"]]
        move_command.pop("height")
        self.move(move_command)

    def joint(self, command: dict[str, Any]) -> None:
        name = command["name"]
        joint_name = command["joint_name"]
        obj = self.env.scene[name]
        joint_id = obj.data.joint_names.index(joint_name)
        start = obj.data.joint_pos.torch[0, joint_id].item()
        if "normalized_goal" in command:
            joint_min, joint_max = obj.data.joint_pos_limits.torch[0, joint_id].tolist()
            normalized_goal = float(command["normalized_goal"])
            if joint_min < 0.0:
                normalized_goal = 1.0 - normalized_goal
            goal = joint_min + (joint_max - joint_min) * normalized_goal
        else:
            goal = float(command["goal"])
        gain = float(command.get("gain", 40.0))
        damping = float(command.get("damping", 6.0))
        limit = float(command.get("limit", 30.0))
        count = max(1, int(float(command.get("seconds", 2.0)) / self.env.physics_dt))
        for index in range(count):
            alpha = (index + 1) / count
            self.joint_holds[(name, joint_name)] = (start + alpha * (goal - start), gain, damping, limit)
            self.step()
        self.step(90)

    def release(self, name: str) -> None:
        self.holds.pop(name, None)
        self.env.scene[name].permanent_wrench_composer.reset()

    def check(self, index: int, timeout_steps: int = 600) -> None:
        for _ in range(timeout_steps):
            if self._task_completion()[0, index].item():
                return
            self.step()
        raise AssertionError(
            f"Subtask {index} did not complete: {self._task_completion()[0].tolist()}; "
            f"current predicates: {self._criterion_status()}"
        )

    def wait_final(self, timeout_steps: int = 1200) -> None:
        from isaaclab_arena.tasks.terminations import task_success_from_progress

        for _ in range(timeout_steps):
            if task_success_from_progress(self.env).all().item():
                return
            self.step()

    def inspect(self) -> None:
        print(
            "FORCE_CONTROL_STATE",
            json.dumps({
                "poses": {
                    name: self.env.arena_world.get_pose_w(name).tolist()
                    for name in (*self.env.scene.articulations, *self.env.scene.rigid_objects)
                },
                "joints": {
                    name: dict(zip(obj.data.joint_names, obj.data.joint_pos.torch[0].tolist()))
                    for name, obj in self.env.scene.articulations.items()
                },
                "completion": self._task_completion().tolist(),
                "current_predicates": self._criterion_status(),
                "wrenches": self.last_wrenches,
            }),
            flush=True,
        )

    def _execute(self, command: dict[str, Any]) -> bool:
        action = command["action"]
        if action == "move":
            self.move(command)
        elif action == "lift":
            self.lift(command)
        elif action == "joint":
            self.joint(command)
        elif action == "release":
            self.release(command["name"])
        elif action == "wait":
            count = command.get("steps")
            if count is None:
                count = int(float(command["seconds"]) / self.env.physics_dt)
            self.step(count)
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

    def _execute_finite(self, commands: tuple[dict[str, Any], ...]) -> None:
        for command in commands:
            print("FORCE_CONTROL_COMMAND", json.dumps(command), flush=True)
            if not self._execute(command):
                break

    def _execute_followed(self, path: Path) -> None:
        cursor = 0
        print("FORCE_CONTROL_READY", str(path), flush=True)
        while True:
            lines = path.read_text().splitlines() if path.exists() else []
            if cursor == len(lines):
                self.env.sim.render()
                time.sleep(0.05)
                continue
            line_number = cursor + 1
            command = _validate_command(json.loads(lines[cursor]), line_number)
            cursor += 1
            if command["action"] == "setup_joint":
                continue
            print("FORCE_CONTROL_COMMAND", json.dumps(command), flush=True)
            if not self._execute(command):
                return
            print("FORCE_CONTROL_DONE", cursor, flush=True)

    def _write_report(self) -> dict[str, Any]:
        from isaaclab_arena.tasks.terminations import task_success_from_progress

        success = task_success_from_progress(self.env).tolist()
        report = {
            "mode": "followed_force_commands" if self.args.follow_commands else "force_control_recipe",
            "success": success,
            "subtasks": self._task_completion().tolist(),
            "current_predicates": self._criterion_status(),
            "video_frames": self.frames,
            "final_poses": {
                name: self.env.arena_world.get_pose_w(name).tolist()
                for name in (*self.env.scene.articulations, *self.env.scene.rigid_objects)
            },
            "final_joints": {
                name: dict(zip(obj.data.joint_names, obj.data.joint_pos.torch[0].tolist()))
                for name, obj in self.env.scene.articulations.items()
            },
            "events": self.events,
        }
        (self.output / "validation.json").write_text(json.dumps(report, indent=2))
        print("TASK_SUCCESS" if all(success) else "TASK_INCOMPLETE", json.dumps(report), flush=True)
        return report

    def run(self, recipe: ForceControlRecipe) -> None:
        try:
            self.step(240)
            if self.args.follow_commands:
                self._execute_followed(self.args.commands)
            else:
                self._execute_finite(recipe.commands)
            report = self._write_report()
            if not self.args.allow_incomplete:
                assert all(report["success"]), f"Final task conditions failed: {report}"
        finally:
            if self.writer is not None:
                self.writer.close()


def _assert_runner_args(args: argparse.Namespace) -> None:
    assert args.num_envs == 1, "force_control_runner supports exactly one environment"
    assert not args.distributed, "force_control_runner does not support distributed execution"
    assert not args.list_variations, "force_control_runner does not support --list_variations"
    if not args.no_record:
        assert args.visualizer is not None and "kit" in args.visualizer, "recording requires the Kit visualizer"


def _add_runner_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Physical Force-Control Runner")
    group.add_argument("--commands", type=Path, required=True, help="JSONL force-control recipe.")
    group.add_argument("--output", type=Path, default=Path("artifacts/force_control"))
    group.add_argument("--follow_commands", action="store_true", help="Wait for commands appended to the JSONL file.")
    group.add_argument("--no_record", action="store_true", help="Do not capture viewport video or marker images.")
    group.add_argument("--allow_incomplete", action="store_true", help="Write a report without requiring task success.")


def _run_recipe(builder, args: argparse.Namespace, recipe: ForceControlRecipe) -> None:
    """Build one environment, execute its recipe, and always close it."""
    env_cfg, env_kwargs = builder.build_env_cfg()
    env_cfg.sim.enable_scene_query_support = True
    env_cfg.recorders = {}
    env_cfg.episode_recorders = {}
    _apply_joint_setups(env_cfg, recipe.joint_setups)
    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.reset()
        PhysicalCommandRunner(env.unwrapped, args).run(recipe)
    finally:
        env.close()


def main() -> None:
    """Launch an Arena environment and execute a physical force-control recipe."""
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
        recipe = load_force_control_recipe(args.commands, require_runtime_commands=not args.follow_commands)
        builder = get_arena_builder_from_cli(args, hydra_overrides=hydra_overrides)
        assert builder.resolved_physics_backend is PhysicsBackend.PHYSX, "force_control_runner requires PhysX"
        _run_recipe(builder, args, recipe)


if __name__ == "__main__":
    main()
