# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute object-control commands in an initialized Arena environment."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from isaaclab_arena_examples.agentic_environment_generation.object_control.object_control_recipe import (
    Axis,
    CheckCommand,
    FinishCommand,
    InspectCommand,
    JointCommand,
    LiftCommand,
    MarkCommand,
    MoveCommand,
    PoseControlCommand,
    ReferencePoint,
    ReleaseCommand,
    RuntimeCommand,
    SetupJointCommand,
    WaitCommand,
    WaitFinalCommand,
)


class ObjectCommandExecutor:
    """Execute object-control commands and evaluate the environment's task predicates."""

    def __init__(self, env, args: argparse.Namespace):
        import torch

        from isaaclab_arena_examples.agentic_environment_generation.object_control.force_actions import (
            ForceJointActionTerm,
            ForcePoseActionTerm,
        )
        from isaaclab_arena_examples.agentic_environment_generation.object_control.object_control_video import (
            ObjectControlVideo,
        )

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
        self.joint_terms = {}
        offset = 0
        for name in self.env.action_manager.active_terms:
            term = self.env.action_manager.get_term(name)
            action = self.actions[:, offset : offset + term.action_dim]
            if isinstance(term, ForcePoseActionTerm):
                self.pose_actions[term.cfg.asset_name] = (term, action)
            elif isinstance(term, ForceJointActionTerm):
                self.joint_terms[term.cfg.asset_name] = term
                self.joint_actions[term.cfg.asset_name] = action.reshape(self.env.num_envs, -1, 5)
            offset += term.action_dim
        self.events = []
        self.current_subtask_index = None
        self.current_subtask_title = "Preparing object-control commands"
        self.video = None if args.no_record else ObjectControlVideo(self.env, self.output)
        self._record_progress()

    @property
    def steps(self) -> int:
        """Elapsed physics steps, owned and counted by the environment."""
        return self.env._sim_step_counter - self.start_sim_step

    def step(self, count: int = 1) -> None:
        """Advance a physics-step budget, rounding up to a whole control step."""
        for _ in range(math.ceil(count / self.env.cfg.decimation)):
            self._step_env()

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

    def move(self, command: MoveCommand) -> None:
        self._drive_pose(command.name, self._move_goal(command), command.orientation, command)

    def lift(self, command: LiftCommand) -> None:
        t_W = self.env.arena_world.get_position_w(command.name)[0].tolist()
        self._drive_pose(command.name, [t_W[0], t_W[1], command.height], None, command)

    def joint(self, command: JointCommand) -> None:
        from isaaclab_arena.utils.joint_utils import unnormalize_joint_position

        name = command.name
        joint_name = command.joint_name
        obj = self.env.scene[name]
        joint_id = obj.data.joint_names.index(joint_name)
        start = obj.data.joint_pos.torch[0, joint_id].item()
        if command.normalized_goal is not None:
            joint_min, joint_max = obj.data.joint_pos_limits.torch[0, joint_id].tolist()
            goal = unnormalize_joint_position(command.normalized_goal, joint_min, joint_max)
        else:
            assert command.goal is not None
            goal = command.goal
        hold_action = self.joint_actions[name][:, joint_id]
        hold_action[:, 1:] = self.torch.tensor(
            [command.gain, command.damping, command.limit, 1], device=self.env.device
        )
        count = max(1, math.ceil(command.seconds / self.env.step_dt))
        for index in range(count):
            alpha = (index + 1) / count
            hold_action[:, 0] = start + alpha * (goal - start)
            self._step_env()
        self.step(90)

    def release(self, command: ReleaseCommand) -> None:
        if command.joint_name is not None:
            self._release_joint(command.name, command.joint_name)
        else:
            term, action = self.pose_actions[command.name]
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
            "OBJECT_CONTROL_STATE",
            json.dumps(self._collect_state()),
            flush=True,
        )

    def run(
        self,
        commands: Iterable[RuntimeCommand],
        joint_setups: Sequence[SetupJointCommand] = (),
        report_mode: str = "object_control_commands",
    ) -> None:
        """Execute validated commands after holding configured joints through initial settling."""
        try:
            self._hold_reset_joint_positions(joint_setups)
            self.step(240)
            self._release_reset_joint_holds(joint_setups)
            self._execute_commands(commands)
            report = self._write_report(report_mode)
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
        print("OBJECT_CONTROL_FAILURE", json.dumps({"trigger": trigger, "output": str(output_path)}), flush=True)
        return output_path

    def _drive_pose(
        self,
        name: str,
        t_W_goal: list[float],
        q_W_goal: list[float] | None,
        command: PoseControlCommand,
    ) -> None:
        """Drive one object to a world-frame pose target."""
        torch = self.torch
        term, action = self.pose_actions[name]
        T_W_start = self.env.arena_world.get_pose_w(name).clone()
        t_W_goal_tensor = torch.tensor([t_W_goal], dtype=torch.float32, device=self.env.device)
        if q_W_goal is not None:
            q_W_goal_tensor = torch.tensor([q_W_goal], dtype=torch.float32, device=self.env.device)
        elif action[0, 7] > 0:
            q_W_goal_tensor = action[:, 3:7].clone()
        else:
            q_W_goal_tensor = T_W_start[:, 3:]
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
            value = getattr(command, field)
            if hasattr(value, "value"):
                value = value.value
            setattr(term.cfg, field, getattr(defaults, field) if value is None else value)
        action[:, 3:7] = q_W_goal_tensor
        action[:, 7] = 1
        count = max(1, math.ceil(command.seconds / self.env.step_dt))
        for index in range(count):
            alpha = (index + 1) / count
            alpha = alpha * alpha * (3 - 2 * alpha)
            action[:, :3] = (1 - alpha) * T_W_start[:, :3] + alpha * t_W_goal_tensor
            self._step_env()
        self.step(60)
        print("MOVE_RESULT", name, self.env.arena_world.get_pose_w(name).tolist(), flush=True)

    def _move_goal(self, command: MoveCommand) -> list[float]:
        goal = command.goal
        if command.reference is None:
            assert goal is not None
            return goal
        reference_name = command.reference
        if command.reference_point is ReferencePoint.AABB_CENTER:
            reference = self.env.arena_world.get_aabb_w(reference_name).center[0].tolist()
        else:
            reference = self.env.arena_world.get_position_w(reference_name)[0].tolist()
        return _combine_reference_goal(goal, reference, command.offset, command.reference_axes)

    def _set_joint_hold(self, name, joint_name, goal, gain, damping, limit) -> None:
        joint_id = self.env.scene[name].data.joint_names.index(joint_name)
        self.joint_actions[name][:, joint_id] = self.torch.tensor(
            [goal, gain, damping, limit, 1], device=self.env.device
        )

    def _release_joint(self, name: str, joint_name: str) -> None:
        joint_id = self.env.scene[name].data.joint_names.index(joint_name)
        actions = self.joint_actions[name]
        actions[:, joint_id, 4] = 0
        self.joint_terms[name].process_actions(actions.reshape(self.env.num_envs, -1))

    def _execute(self, command: RuntimeCommand) -> bool:
        if isinstance(command, MoveCommand):
            self.move(command)
        elif isinstance(command, LiftCommand):
            self.lift(command)
        elif isinstance(command, JointCommand):
            self.joint(command)
        elif isinstance(command, ReleaseCommand):
            self.release(command)
        elif isinstance(command, WaitCommand):
            count = command.steps
            if count is None:
                assert command.seconds is not None
                count = int(command.seconds / self.env.physics_dt)
            self.step(count)
        elif isinstance(command, MarkCommand):
            self.mark(command.index, command.boundary.value, command.title)
        elif isinstance(command, CheckCommand):
            self.check(command.index, command.timeout_steps)
        elif isinstance(command, WaitFinalCommand):
            self.wait_final(command.timeout_steps)
        elif isinstance(command, InspectCommand):
            self.inspect()
        elif isinstance(command, FinishCommand):
            return False
        else:
            raise AssertionError(f"Unhandled command type '{type(command).__name__}'")
        return True

    def _execute_commands(self, commands: Iterable[RuntimeCommand]) -> None:
        for command in commands:
            print("OBJECT_CONTROL_COMMAND", command.model_dump_json(exclude_unset=True), flush=True)
            if not self._execute(command):
                break

    def _write_report(self, mode: str) -> dict[str, Any]:
        from isaaclab_arena.tasks.terminations import task_success_from_progress

        success = task_success_from_progress(self.env).tolist()
        state = self._collect_state()
        report = {
            "mode": mode,
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

    def _hold_reset_joint_positions(self, joint_setups: Sequence[SetupJointCommand]) -> None:
        """Apply optional effort holds at the reset angles before initial settling."""
        for setup in joint_setups:
            hold = setup.settle_hold
            if hold is None:
                continue
            name, joint_name = setup.name, setup.joint_name
            obj = self.env.scene[name]
            joint_id = obj.data.joint_names.index(joint_name)
            goal = obj.data.joint_pos.torch[0, joint_id].item()
            self._set_joint_hold(name, joint_name, goal, hold.gain, hold.damping, hold.limit)

    def _release_reset_joint_holds(self, joint_setups: Sequence[SetupJointCommand]) -> None:
        """Release effort holds after initial settling."""
        for setup in joint_setups:
            if setup.settle_hold is not None:
                self._release_joint(setup.name, setup.joint_name)


def configure_object_command_executor(env_cfg, joint_setups: tuple[SetupJointCommand, ...]) -> None:
    """Configure scene assets and force actions required by ``ObjectCommandExecutor``."""
    _apply_joint_setups(env_cfg, joint_setups)
    _configure_force_actions(env_cfg)


def _apply_joint_setups(env_cfg, joint_setups: tuple[SetupJointCommand, ...]) -> None:
    """Add effort-controlled actuators requested before environment creation."""
    if not joint_setups:
        return
    from isaaclab.actuators import ImplicitActuatorCfg

    asset_cfgs = {}
    for setup in joint_setups:
        name = setup.name
        assert hasattr(env_cfg.scene, name), f"Recipe setup refers to unknown scene entity '{name}'"
        asset_cfgs[name] = getattr(env_cfg.scene, name)
    for name in {setup.name for setup in joint_setups if setup.replace_existing}:
        asset_cfgs[name].actuators = {}

    for setup in joint_setups:
        name = setup.name
        asset_cfg = asset_cfgs[name]
        existing_actuators = dict(asset_cfg.actuators or {})
        actuator_name = "object_control_" + re.sub(r"\W+", "_", setup.joint_name).strip("_")
        assert (
            actuator_name not in existing_actuators
        ), f"Recipe defines duplicate actuator '{actuator_name}' for scene entity '{name}'"
        existing_actuators[actuator_name] = ImplicitActuatorCfg(
            joint_names_expr=[re.escape(setup.joint_name)],
            stiffness=setup.drive_stiffness,
            damping=setup.drive_damping,
            effort_limit_sim=setup.effort_limit,
        )
        asset_cfg.actuators = existing_actuators


def _configure_force_actions(env_cfg) -> None:
    """Attach force actions to scene assets; existing robot drives remain at reset targets."""
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg

    from isaaclab_arena.utils.configclass import make_configclass
    from isaaclab_arena_examples.agentic_environment_generation.object_control.force_actions import (
        ForceJointActionCfg,
        ForcePoseActionCfg,
    )

    terms = []
    for name, asset_cfg in vars(env_cfg.scene).items():
        if isinstance(asset_cfg, (ArticulationCfg, RigidObjectCfg)):
            terms.append((f"pose_{name}", ForcePoseActionCfg, ForcePoseActionCfg(asset_name=name)))
        if isinstance(asset_cfg, ArticulationCfg):
            terms.append((f"joint_{name}", ForceJointActionCfg, ForceJointActionCfg(asset_name=name)))
    env_cfg.actions = make_configclass("ForceActionsCfg", terms)()


def _combine_reference_goal(
    goal: list[float] | None,
    reference: list[float],
    offset: list[float],
    reference_axes: list[Axis] | None,
) -> list[float]:
    """Combine absolute and reference-relative components into a world-frame goal."""
    axes = {"x": 0, "y": 1, "z": 2}
    selected_axes = set(reference_axes or axes)
    assert goal is not None or len(selected_axes) == 3, "Unreferenced axes require an absolute goal"
    combined = list(goal or (0.0, 0.0, 0.0))
    for axis in selected_axes:
        index = axes[axis.value]
        combined[index] = reference[index] + offset[index]
    return combined
