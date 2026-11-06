# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute reusable physical force-control recipes in an Arena environment."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    "setup_joint": (
        {"name", "joint_name"},
        {"effort_limit", "replace_existing", "settle_hold", "drive_stiffness", "drive_damping"},
    ),
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


class PhysicalCommandRunner:
    """Execute physical commands and evaluate the environment's task predicates."""

    def __init__(self, env, args: argparse.Namespace):
        import torch

        from isaaclab_arena.terms.force_actions import ForceJointActionTerm, ForcePoseActionTerm
        from isaaclab_arena_examples.agentic_environment_generation.force_control_video import ForceControlVideo

        self.wrapped_env = env
        self.env = env.unwrapped
        self.args = args
        self.torch = torch
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.start_sim_step = self.env._sim_step_counter
        self.actions = torch.zeros(
            (self.env.num_envs, self.env.action_manager.total_action_dim), device=self.env.device
        )
        self.pose_actions = {}
        self.joint_actions = {}
        offset = 0
        for name in self.env.action_manager.active_terms:
            term = self.env.action_manager.get_term(name)
            action = self.actions[:, offset : offset + term.action_dim]
            if isinstance(term, ForcePoseActionTerm):
                self.pose_actions[term.cfg.asset_name] = (term, action)
            elif isinstance(term, ForceJointActionTerm):
                self.joint_actions[term.cfg.asset_name] = action.reshape(self.env.num_envs, -1, 5)
            offset += term.action_dim
        self.events = []
        self.current_subtask_index = None
        self.current_subtask_title = "Preparing force-control recipe"
        self.video = None if args.no_record else ForceControlVideo(self.env, self.output)
        self._record_progress()

    @property
    def steps(self) -> int:
        """Elapsed physics steps, owned and counted by the environment."""
        return self.env._sim_step_counter - self.start_sim_step

    def _progress(self) -> dict:
        return {
            "index": self.current_subtask_index,
            "title": self.current_subtask_title,
            "completion": self._task_completion()[0].tolist(),
        }

    def _record_progress(self) -> None:
        if self.video is not None:
            self.video.record_progress(self.steps * self.env.physics_dt, self._progress())

    def _step_env(self) -> None:
        self.wrapped_env.step(self.actions)
        self._record_progress()

    def step(self, count: int = 1) -> None:
        """Advance a physics-step budget, rounding up to a whole control step."""
        for _ in range(math.ceil(count / self.env.cfg.decimation)):
            self._step_env()

    def _task_completion(self):
        tracker = self.env.progress_tracker
        if all(criteria.parent_subtask_idx is None for criteria in tracker.completion_criteria):
            return tracker.is_complete()[:, None]
        return tracker.get_subtask_completion()

    def _criterion_status(self) -> dict[str, list | dict]:
        from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_SEQUENCE_NAME

        status = {}
        for runner in self.env.progress_tracker.runners:
            sequences = {}
            for sequence_name, chain in runner.predicate_chains.items():
                values = []
                for predicate, _ in chain:
                    evaluate = getattr(predicate, "predicate", predicate)
                    values.append(evaluate(self.env).tolist())
                sequences[sequence_name] = values
            status[runner.completion_criteria.name] = (
                sequences[DEFAULT_SEQUENCE_NAME] if list(sequences) == [DEFAULT_SEQUENCE_NAME] else sequences
            )
        return status

    @staticmethod
    def _buffer_to_list(value):
        """Convert an Isaac Lab runtime buffer or tensor to JSON-compatible lists."""
        if value is None:
            return None
        value = getattr(value, "torch", value)
        if hasattr(value, "tolist"):
            return value.tolist()
        return None

    def _contact_sensor_state(self) -> dict[str, dict]:
        """Return configured contact-sensor forces without relying on task-specific names."""
        from isaaclab.sensors import ContactSensorCfg

        result = {}
        for name, sensor in self.env.scene.sensors.items():
            if not isinstance(sensor.cfg, ContactSensorCfg):
                continue
            data = sensor.data
            result[name] = {
                "prim_path": sensor.cfg.prim_path,
                "filter_prim_paths_expr": list(sensor.cfg.filter_prim_paths_expr or []),
                "net_forces_w": self._buffer_to_list(getattr(data, "net_forces_w", None)),
                "force_matrix_w": self._buffer_to_list(getattr(data, "force_matrix_w", None)),
            }
        return result

    def _collect_state(self) -> dict:
        """Collect live task, object, articulation, controller, and sensor state."""
        scene_object_names = (*self.env.scene.articulations, *self.env.scene.rigid_objects)
        return {
            "simulation_step": self.steps,
            "simulation_time_s": self.steps * self.env.physics_dt,
            "active_subtask": {
                "index": self.current_subtask_index,
                "title": self.current_subtask_title,
            },
            "poses": {name: self.env.arena_world.get_pose_w(name).tolist() for name in scene_object_names},
            "velocities": {
                name: {
                    "linear_w": self.env.arena_world.get_root_linear_velocity_w(name).tolist(),
                    "angular_w": self.env.arena_world.get_root_angular_velocity_w(name).tolist(),
                }
                for name in scene_object_names
            },
            "joints": {
                name: {
                    joint_name: {
                        "position": position,
                        "velocity": velocity,
                    }
                    for joint_name, position, velocity in zip(
                        obj.data.joint_names,
                        obj.data.joint_pos.torch[0].tolist(),
                        obj.data.joint_vel.torch[0].tolist(),
                        strict=True,
                    )
                }
                for name, obj in self.env.scene.articulations.items()
            },
            "contact_sensors": self._contact_sensor_state(),
            "completion": self._task_completion().tolist(),
            "current_predicates": self._criterion_status(),
            "wrenches": {
                name: {key: value.tolist() if hasattr(value, "tolist") else value for key, value in wrench.items()}
                for name, (term, _) in self.pose_actions.items()
                if (wrench := term.last_wrench) is not None
            },
        }

    def _write_failure_snapshot(self, trigger: str, subtask_index: int | None = None) -> Path:
        """Persist live state and an optional viewport image for a failed command."""
        suffix = f"_subtask_{subtask_index:02d}" if subtask_index is not None else ""
        stem = f"failure_{trigger}{suffix}_step_{self.steps:06d}"
        state = self._collect_state()
        state["trigger"] = trigger
        output_path = self.output / f"{stem}.json"
        # Preserve diagnostics even if the renderer cannot produce the optional image.
        output_path.write_text(json.dumps(state, indent=2))
        self._failure_snapshot_path = output_path
        if self.video is not None:
            image_name = f"{stem}.png"
            self.video.snapshot(image_name, self._progress())
            state["viewport"] = image_name
        output_path.write_text(json.dumps(state, indent=2))
        print("FORCE_CONTROL_FAILURE", json.dumps({"trigger": trigger, "output": str(output_path)}), flush=True)
        return output_path

    def mark(self, index: int, boundary: str, title: str) -> None:
        self.current_subtask_index = index
        self.current_subtask_title = title
        self._record_progress()
        record = {
            "subtask": index,
            "boundary": boundary,
            "title": title,
            "simulation_time_s": self.steps * self.env.physics_dt,
            "completion": self._task_completion()[0].tolist(),
        }
        if self.video is not None:
            filename = f"{index:02d}_{boundary}.png"
            self.video.snapshot(filename, self._progress())
            record["viewport"] = filename
        self.events.append(record)
        state_filename = f"{index:02d}_{boundary}_state.json"
        record["state"] = state_filename
        (self.output / state_filename).write_text(json.dumps(self._collect_state(), indent=2))
        print("SUBTASK", json.dumps(record), flush=True)
        (self.output / "progress.json").write_text(json.dumps(self.events, indent=2))

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
        term, action = self.pose_actions[name]
        T_W_start = self.env.arena_world.get_pose_w(name).clone()
        t_W_goal = torch.tensor([self._move_goal(command)], dtype=torch.float32, device=self.env.device)
        if "orientation" in command:
            q_W_goal = torch.tensor([command["orientation"]], dtype=torch.float32, device=self.env.device)
        elif action[0, 7] > 0:
            q_W_goal = action[:, 3:7].clone()
        else:
            q_W_goal = T_W_start[:, 3:]
        # Each command supplies a fresh set of gains, as in the JSONL controller contract.
        defaults = type(term.cfg)(asset_name=name)
        for field in (
            "linear_gain",
            "linear_damping",
            "minimum_stiffness",
            "max_force",
            "rotation_mode",
            "rotation_gain",
            "angular_damping",
            "max_torque",
        ):
            setattr(term.cfg, field, command.get(field, getattr(defaults, field)))
        action[:, 3:7] = q_W_goal
        action[:, 7] = 1
        count = max(1, math.ceil(float(command.get("seconds", 2.0)) / self.env.step_dt))
        for index in range(count):
            alpha = (index + 1) / count
            alpha = alpha * alpha * (3 - 2 * alpha)
            action[:, :3] = (1 - alpha) * T_W_start[:, :3] + alpha * t_W_goal
            self._step_env()
        self.step(60)
        print("MOVE_RESULT", name, self.env.arena_world.get_pose_w(name).tolist(), flush=True)

    def lift(self, command: dict[str, Any]) -> None:
        t_W = self.env.arena_world.get_position_w(command["name"])[0].tolist()
        move_command = dict(command)
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
        count = max(1, math.ceil(float(command.get("seconds", 2.0)) / self.env.step_dt))
        for index in range(count):
            alpha = (index + 1) / count
            self._set_joint_hold(name, joint_name, start + alpha * (goal - start), gain, damping, limit)
            self._step_env()
        self.step(90)

    def _set_joint_hold(self, name, joint_name, goal, gain, damping, limit) -> None:
        joint_id = self.env.scene[name].data.joint_names.index(joint_name)
        self.joint_actions[name][:, joint_id] = self.torch.tensor(
            [goal, gain, damping, limit, 1], device=self.env.device
        )

    def release(self, name: str) -> None:
        term, action = self.pose_actions[name]
        action[:, 7] = 0
        # Clear the persistent wrench immediately, including when release precedes a snapshot.
        term.process_actions(action)

    def check(self, index: int, timeout_steps: int = 600) -> None:
        for _ in range(math.ceil(timeout_steps / self.env.cfg.decimation)):
            if self._task_completion()[0, index].item():
                return
            self.step()
        if self._task_completion()[0, index].item():
            return
        snapshot_path = self._write_failure_snapshot("check_timeout", index)
        raise AssertionError(
            f"Subtask {index} did not complete: {self._task_completion()[0].tolist()}; "
            f"current predicates: {self._criterion_status()}; diagnostics: {snapshot_path}"
        )

    def wait_final(self, timeout_steps: int = 1200) -> None:
        from isaaclab_arena.tasks.terminations import task_success_from_progress

        for _ in range(math.ceil(timeout_steps / self.env.cfg.decimation)):
            if task_success_from_progress(self.env).all().item():
                return
            self.step()

    def inspect(self) -> None:
        print(
            "FORCE_CONTROL_STATE",
            json.dumps(self._collect_state()),
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
        runtime_command_seen = False
        print("FORCE_CONTROL_READY", str(path), flush=True)
        while True:
            lines = _recipe_lines(path, complete_only=True) if path.exists() else []
            if cursor == len(lines):
                self.env.sim.render()
                time.sleep(0.05)
                continue
            line_number = cursor + 1
            line = lines[cursor]
            cursor += 1
            if not line.strip():
                continue
            command = _validate_command(json.loads(line), line_number)
            if command["action"] == "setup_joint":
                assert not runtime_command_seen, "setup_joint commands must precede runtime commands"
                continue
            runtime_command_seen = True
            print("FORCE_CONTROL_COMMAND", json.dumps(command), flush=True)
            if not self._execute(command):
                return
            print("FORCE_CONTROL_DONE", cursor, flush=True)

    def _write_report(self) -> dict[str, Any]:
        from isaaclab_arena.tasks.terminations import task_success_from_progress

        success = task_success_from_progress(self.env).tolist()
        state = self._collect_state()
        report = {
            "mode": "followed_force_commands" if self.args.follow_commands else "force_control_recipe",
            "success": success,
            "video_frames": self.video.frames if self.video is not None else 0,
            "state": state,
            "subtasks": state["completion"],
            "current_predicates": state["current_predicates"],
            "final_poses": state["poses"],
            "final_joints": {
                name: {joint_name: values["position"] for joint_name, values in joints.items()}
                for name, joints in state["joints"].items()
            },
            "final_joint_state": state["joints"],
            "events": self.events,
        }
        (self.output / "validation.json").write_text(json.dumps(report, indent=2))
        if self.video is not None:
            self.video.close()
            report["video_frames"] = self.video.frames
            (self.output / "validation.json").write_text(json.dumps(report, indent=2))
        print("TASK_SUCCESS" if all(success) else "TASK_INCOMPLETE", json.dumps(report), flush=True)
        return report

    def _hold_reset_joint_positions(self, joint_setups: tuple[dict[str, Any], ...]) -> None:
        """Apply optional effort holds at the reset angles before initial settling."""
        for setup in joint_setups:
            hold = setup.get("settle_hold")
            if hold is None:
                continue
            name, joint_name = setup["name"], setup["joint_name"]
            obj = self.env.scene[name]
            joint_id = obj.data.joint_names.index(joint_name)
            goal = obj.data.joint_pos.torch[0, joint_id].item()
            self._set_joint_hold(name, joint_name, goal, hold["gain"], hold["damping"], hold["limit"])

    def run(self, recipe: ForceControlRecipe) -> None:
        try:
            self._hold_reset_joint_positions(recipe.joint_setups)
            self.step(240)
            if self.args.follow_commands:
                self._execute_followed(self.args.commands)
            else:
                self._execute_finite(recipe.commands)
            report = self._write_report()
            if not self.args.allow_incomplete:
                if not all(report["success"]):
                    snapshot = self._write_failure_snapshot("final_conditions")
                    raise AssertionError(f"Final task conditions failed; diagnostics: {snapshot}")
        except Exception:
            if getattr(self, "_failure_snapshot_path", None) is None:
                try:
                    self._write_failure_snapshot("execution_error")
                except Exception:
                    logging.exception("Could not finish the execution-error snapshot")
            raise
        finally:
            if self.video is not None:
                if sys.exception() is None:
                    self.video.close()
                else:
                    try:
                        self.video.close()
                    except Exception:
                        logging.exception("Could not finish the failure video")


def _recipe_lines(path: Path, complete_only: bool = False) -> list[str]:
    """Read JSONL records, withholding an unterminated append in follow mode."""
    content = path.read_bytes()
    if complete_only:
        content = content[: content.rfind(b"\n") + 1]
    return content.decode("utf-8").splitlines(keepends=True)


def load_force_control_recipe(path: Path, require_runtime_commands: bool = True) -> ForceControlRecipe:
    """Load and validate a force-control JSONL recipe."""
    joint_setups = []
    commands = []
    runtime_command_seen = False
    for line_number, line in enumerate(_recipe_lines(path, complete_only=not require_runtime_commands), 1):
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


def main() -> None:
    """Launch an Arena environment and execute a physical force-control recipe."""
    parser = get_isaaclab_arena_cli_parser()
    _add_runner_args(parser)
    parser.set_defaults(device="cpu", visualizer=["kit"])
    parser.allow_abbrev = False
    original_cli_args = sys.argv[1:]
    app_args, _ = parser.parse_known_args(original_cli_args)
    _assert_runner_args(app_args)
    # Load RTX/Replicator for Lab's Kit recorder without adding scene cameras.
    if not app_args.no_record:
        app_args.enable_cameras = True

    with SimulationAppContext(app_args):
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args, hydra_overrides = parser.parse_known_args(original_cli_args)
        assert_hydra_overrides(hydra_overrides, parser)
        _assert_runner_args(args)
        recipe = load_force_control_recipe(args.commands, require_runtime_commands=not args.follow_commands)
        builder = get_arena_builder_from_cli(args, hydra_overrides=hydra_overrides)
        assert builder.resolved_physics_backend is PhysicsBackend.PHYSX, "force_control_runner requires PhysX"
        _run_recipe(builder, args, recipe)


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
        "drive_stiffness",
        "drive_damping",
    ):
        if field in command:
            _assert_number(command[field], field, line_number)
    for field in ("drive_stiffness", "drive_damping"):
        if field in command:
            assert command[field] >= 0, f"Recipe line {line_number}: '{field}' must be non-negative"
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
    if "settle_hold" in command:
        hold = command["settle_hold"]
        assert isinstance(hold, dict) and set(hold) == {
            "gain",
            "damping",
            "limit",
        }, f"Recipe line {line_number}: 'settle_hold' requires exactly gain, damping, and limit"
        for field, value in hold.items():
            _assert_number(value, f"settle_hold.{field}", line_number)
            assert (
                math.isfinite(value) and value >= 0
            ), f"Recipe line {line_number}: 'settle_hold.{field}' must be finite and non-negative"
    return command


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
            stiffness=float(setup.get("drive_stiffness", 0.0)),
            damping=float(setup.get("drive_damping", 0.0)),
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


def _configure_force_actions(env_cfg) -> None:
    """Attach force actions to scene assets; existing robot drives remain at reset targets."""
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg

    from isaaclab_arena.terms.force_actions import ForceJointActionCfg, ForcePoseActionCfg
    from isaaclab_arena.utils.configclass import make_configclass

    terms = []
    for name, asset_cfg in vars(env_cfg.scene).items():
        if isinstance(asset_cfg, (ArticulationCfg, RigidObjectCfg)):
            terms.append((f"pose_{name}", ForcePoseActionCfg, ForcePoseActionCfg(asset_name=name)))
        if isinstance(asset_cfg, ArticulationCfg):
            terms.append((f"joint_{name}", ForceJointActionCfg, ForceJointActionCfg(asset_name=name)))
    env_cfg.actions = make_configclass("ForceActionsCfg", terms)()


def _run_recipe(builder, args: argparse.Namespace, recipe: ForceControlRecipe) -> None:
    """Build one environment, execute its recipe, and always close it."""
    env_cfg, env_kwargs = builder.build_env_cfg()
    env_cfg.sim.enable_scene_query_support = True
    env_cfg.recorders = {}
    env_cfg.episode_recorders = {}
    _apply_joint_setups(env_cfg, recipe.joint_setups)
    _configure_force_actions(env_cfg)
    env_cfg.video_recorders = []
    if not args.no_record:
        from isaaclab_arena_examples.agentic_environment_generation.force_control_video import configure_video

        configure_video(env_cfg, args.output)
    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.unwrapped.configure_episode_limit(1)
        env.reset()
        PhysicalCommandRunner(env, args).run(recipe)
    finally:
        env.close()


if __name__ == "__main__":
    main()
