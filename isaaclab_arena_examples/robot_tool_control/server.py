# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep a DROID environment alive for bounded commands from a local client."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file atomically."""
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(path)


def save_observation(env, output_dir: Path) -> dict:
    """Save camera images and measured robot state without exposing object poses."""
    import numpy as np

    from isaaclab.utils.math import subtract_frame_transforms
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    observation = env.observation_manager.compute(update_history=False)
    robot = env.scene["robot"]
    position_world = observation["policy"]["eef_pos"]
    quaternion_world = observation["policy"]["eef_quat"]
    position_base, quaternion_base = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        position_world,
        quaternion_world,
    )
    images = {}
    for name, values in observation["camera_obs"].items():
        pixels = values[0].detach().cpu().numpy()
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels * (255 if pixels.max() <= 1 else 1), 0, 255).astype(np.uint8)
        camera_image = Image.fromarray(pixels[..., :3])
        image_path = output_dir / f"{name}.png"
        camera_image.save(image_path)
        images[name] = str(image_path)

    finger_positions = {}
    finger_quaternions = {}
    for name in ("left_inner_finger", "right_inner_finger"):
        body_index = robot.data.body_names.index(name)
        finger_positions[name] = robot.data.body_pos_w.torch[0, body_index].tolist()
        finger_quaternions[name] = robot.data.body_quat_w.torch[0, body_index].tolist()
    camera_calibration = {}
    for name, sensor in env.scene.sensors.items():
        if hasattr(sensor.data, "intrinsic_matrices"):
            camera_calibration[name] = {
                "intrinsic_matrix": sensor.data.intrinsic_matrices[0].tolist(),
                "position_world": sensor.data.pos_w[0].tolist(),
                "quaternion_world_ros_xyzw": sensor.data.quat_w_ros[0].tolist(),
                "image_shape": list(sensor.data.image_shape),
            }
    state = {
        "episode": env.get_episode_index(0),
        "simulation_steps": int(env.episode_length_buf[0].item()),
        "step_dt": env.step_dt,
        "instruction": env.get_language_instruction(),
        "frame": "robot_base",
        "tool_frame": "Robotiq base_link (gripper flange, not fingertip center)",
        "position": position_base[0].tolist(),
        "quaternion_xyzw": quaternion_base[0].tolist(),
        "position_world": position_world[0].tolist(),
        "quaternion_world_xyzw": quaternion_world[0].tolist(),
        "robot_position_world": robot.data.root_pos_w.torch[0].tolist(),
        "robot_quaternion_world_xyzw": robot.data.root_quat_w.torch[0].tolist(),
        "joint_positions": observation["policy"]["joint_pos"][0].tolist(),
        "gripper_position": observation["policy"]["gripper_pos"][0].tolist(),
        "finger_link_positions_world": finger_positions,
        "finger_link_quaternions_world_xyzw": finger_quaternions,
        "images": images,
        "camera_calibration": camera_calibration,
    }
    assert all(math.isfinite(value) for value in state["position"]), "Nonfinite robot pose"
    write_json(output_dir / "observation.json", state)
    return state


def record_terminal_observation(env, env_id: int, output_dir: str) -> dict:
    """Capture the finishing episode before Isaac Lab automatically resets it."""
    assert env_id == 0, "The command server supports one environment"
    snapshot_dir = Path(output_dir) / f"episode_{env.get_episode_index(0):03d}_terminal"
    state = save_observation(env, snapshot_dir)
    state["termination"] = {
        name: bool(env.termination_manager.get_term(name)[0].item()) for name in env.termination_manager.active_terms
    }
    write_json(snapshot_dir / "observation.json", state)
    write_json(Path(output_dir) / "latest_terminal.json", state)
    return {"terminal_observation_path": str(snapshot_dir / "observation.json")}


class RobotControlSession:
    """Execute bounded Cartesian goals and report their measured outcomes."""

    def __init__(self, env, session_dir: Path):
        self.env = env
        self.base_env = env.unwrapped
        self.session_dir = session_dir
        self.gripper_command = 0.0
        self.episode_finished = False
        self.workspace_min = [0.05, -0.65, 0.10]
        self.workspace_max = [0.85, 0.65, 0.80]
        self.position_tolerance = 0.005
        self.rotation_tolerance = 0.05
        self.translation_per_step = 0.004
        self.rotation_per_step = 0.04
        self.terminal_observation = None
        self.env.reset()

    def validate(self, request: dict) -> None:
        """Reject malformed or out-of-bounds requests before any simulation step."""
        command = request.get("command")
        if not isinstance(command, str) or command not in {
            "observe",
            "move_to",
            "set_gripper",
            "wait",
            "reset",
            "shutdown",
        }:
            raise ValueError(f"Unknown command: {command!r}")
        unsupported_fields = set(request) - {"id", "command", "position", "quaternion", "gripper", "steps", "note"}
        if unsupported_fields:
            raise ValueError(f"Unsupported request fields: {sorted(unsupported_fields)}")
        if command != "move_to" and ("position" in request or "quaternion" in request):
            raise ValueError("position and quaternion are only valid for move_to")
        if "gripper" in request and command not in {"move_to", "set_gripper"}:
            raise ValueError("gripper is only valid for move_to and set_gripper")
        if "steps" in request and command not in {"move_to", "set_gripper", "wait"}:
            raise ValueError("steps is only valid for move_to, set_gripper, and wait")
        if self.episode_finished and command in {"move_to", "set_gripper", "wait"}:
            raise ValueError("Episode finished; inspect the terminal observation and explicitly reset")
        steps = request.get("steps", 180 if command == "move_to" else 24)
        if type(steps) is not int or not 1 <= steps <= 240:
            raise ValueError("steps must be an integer between 1 and 240")
        if "gripper" in request:
            gripper = request["gripper"]
            if type(gripper) not in (float, int) or gripper not in (0, 1):
                raise ValueError("gripper must be 0 (open) or 1 (closed)")
        if command == "set_gripper" and "gripper" not in request:
            raise ValueError("set_gripper requires gripper")
        if command == "move_to":
            position = request.get("position", [])
            if (
                not isinstance(position, list)
                or len(position) != 3
                or not all(type(value) in (float, int) and math.isfinite(value) for value in position)
            ):
                raise ValueError("position must contain three finite coordinates in meters")
            if any(
                value < lower or value > upper
                for value, lower, upper in zip(position, self.workspace_min, self.workspace_max)
            ):
                raise ValueError(f"Target outside workspace: {self.workspace_min} to {self.workspace_max}")
        if "quaternion" in request:
            quaternion = request["quaternion"]
            if (
                not isinstance(quaternion, list)
                or len(quaternion) != 4
                or not all(type(value) in (float, int) and math.isfinite(value) for value in quaternion)
            ):
                raise ValueError("quaternion must contain four finite xyzw coordinates")
            if not math.isclose(math.hypot(*quaternion), 1.0, abs_tol=1e-3):
                raise ValueError("quaternion must have unit norm")

    def measured_pose(self):
        """Return the gripper flange pose in the robot-base frame."""
        from isaaclab.utils.math import subtract_frame_transforms

        robot = self.base_env.scene["robot"]
        body_index = robot.data.body_names.index("base_link")
        return subtract_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, body_index],
            robot.data.body_quat_w.torch[:, body_index],
        )

    def execute(self, request: dict) -> dict:
        """Execute one validated request on the simulation thread."""
        import torch

        from isaaclab.utils.math import apply_delta_pose, compute_pose_error

        command = request["command"]
        command_id = request["id"]
        started = time.monotonic()
        outcome = "observed"
        executed_steps = 0
        residual = {}
        if command == "reset":
            # A terminated step has already reset Isaac Lab and finished the camera videos.
            if not self.episode_finished:
                self.env.reset()
            self.episode_finished = False
            self.terminal_observation = None
            self.gripper_command = 0.0
            outcome = "reset"
        elif command in {"move_to", "set_gripper", "wait"}:
            current_position, current_quaternion = self.measured_pose()
            target_position = torch.tensor(
                [request.get("position", current_position[0].tolist())],
                device=self.base_env.device,
                dtype=current_position.dtype,
            )
            target_quaternion = torch.tensor(
                [request.get("quaternion", current_quaternion[0].tolist())],
                device=self.base_env.device,
                dtype=current_quaternion.dtype,
            )
            target_quaternion = target_quaternion / target_quaternion.norm(dim=-1, keepdim=True)
            waypoint_position = current_position.clone()
            waypoint_quaternion = current_quaternion.clone()
            self.gripper_command = float(request.get("gripper", self.gripper_command))
            max_steps = request.get("steps", 180 if command == "move_to" else 24)
            outcome = "incomplete"
            for step_index in range(max_steps):
                current_position, current_quaternion = self.measured_pose()
                position_error, rotation_error = compute_pose_error(
                    waypoint_position,
                    waypoint_quaternion,
                    target_position,
                    target_quaternion,
                )
                waypoint_at_goal = (
                    position_error.norm().item() <= self.translation_per_step
                    and rotation_error.norm().item() <= self.rotation_per_step
                )
                position_delta = position_error * min(
                    1.0,
                    self.translation_per_step / max(position_error.norm().item(), 1e-9),
                )
                rotation_delta = rotation_error * min(
                    1.0,
                    self.rotation_per_step / max(rotation_error.norm().item(), 1e-9),
                )
                waypoint_position, waypoint_quaternion = apply_delta_pose(
                    waypoint_position,
                    waypoint_quaternion,
                    torch.cat((position_delta, rotation_delta), dim=-1),
                )
                # Advance the reference independently so load-induced tracking error can accumulate.
                position_error, rotation_error = compute_pose_error(
                    current_position,
                    current_quaternion,
                    waypoint_position,
                    waypoint_quaternion,
                )
                action_scale = self.base_env.cfg.actions.arm_action.scale
                actions = torch.cat(
                    (
                        position_error / action_scale,
                        rotation_error / action_scale,
                        torch.tensor([[self.gripper_command]], device=self.base_env.device),
                    ),
                    dim=-1,
                )
                with torch.inference_mode():
                    _, _, terminated, truncated, _ = self.env.step(actions)
                executed_steps += 1
                if bool((terminated | truncated).any()):
                    self.episode_finished = True
                    self.terminal_observation = json.loads((self.session_dir / "latest_terminal.json").read_text())
                    outcome = "episode_finished"
                    break
                current_position, current_quaternion = self.measured_pose()
                position_error, rotation_error = compute_pose_error(
                    current_position,
                    current_quaternion,
                    target_position,
                    target_quaternion,
                )
                residual = {
                    "position_error_m": position_error.norm().item(),
                    "rotation_error_rad": rotation_error.norm().item(),
                }
                with (self.session_dir / "trajectory.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({
                            "command_id": command_id,
                            "episode": self.base_env.get_episode_index(0),
                            "step": int(self.base_env.episode_length_buf[0].item()),
                            "position": current_position[0].tolist(),
                            "quaternion_xyzw": current_quaternion[0].tolist(),
                            "gripper_command": self.gripper_command,
                            "target_position": target_position[0].tolist(),
                            "waypoint_position": waypoint_position[0].tolist(),
                            "waypoint_quaternion_xyzw": waypoint_quaternion[0].tolist(),
                            **residual,
                        })
                        + "\n"
                    )
                converged = (
                    residual["position_error_m"] < self.position_tolerance
                    and residual["rotation_error_rad"] < self.rotation_tolerance
                )
                if command == "move_to" and step_index >= 5 and waypoint_at_goal and converged:
                    outcome = "converged"
                    break
                if command != "move_to" and step_index == max_steps - 1:
                    outcome = "completed"
        elif command == "shutdown":
            outcome = "shutdown"
        observation = (
            self.terminal_observation
            if self.episode_finished
            else save_observation(self.base_env, self.session_dir / "observations" / command_id)
        )
        return {
            "ok": True,
            "id": command_id,
            "outcome": outcome,
            "executed_steps": executed_steps,
            "wall_time_s": time.monotonic() - started,
            "episode_finished": self.episode_finished,
            "observation": observation,
            **residual,
        }


def build_environment(args):
    """Build one fixed-layout DROID pick-and-place environment with recording."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
        PickAndPlaceMapleTableEnvironment,
        PickAndPlaceMapleTableEnvironmentCfg,
    )

    environment = PickAndPlaceMapleTableEnvironment().build(
        PickAndPlaceMapleTableEnvironmentCfg(
            embodiment="droid_differential_ik",
            enable_cameras=True,
            pick_up_object=args.object,
            destination_location="bowl_ycb_robolab",
            episode_length_s=90.0,
            hdr="home_office_robolab",
        )
    )
    environment.embodiment.camera_config.wrist_camera.update_latest_camera_pose = True
    environment.episode_recorder_terms["terminal_observation"] = EpisodeRecorderTermCfg(
        func=record_terminal_observation, params={"output_dir": str(args.session_dir)}
    )
    object_description = "banana" if "banana" in args.object else "Rubik's cube"
    builder = ArenaEnvBuilder(
        environment,
        ArenaEnvBuilderCfg(
            num_envs=1,
            seed=args.seed,
            placement_seed=args.seed,
            resolve_on_reset=False,
            language_instruction=f"Pick up the {object_description} and place it in the bowl.",
        ),
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.compute_final_obs = True
    env = builder.make_registered(env_cfg, env_kwargs, render_mode=None)
    env.unwrapped.episode_recorder.set_job_name("robot_tool_control")
    env.unwrapped.episode_recorder.set_output_path(args.session_dir / "episodes.jsonl")
    return wrap_env_for_video(
        env,
        VideoRecordingCfg(
            record_camera_video=True,
            video_base_dir=str(args.session_dir / "videos"),
        ),
        num_steps=None,
        num_episodes=1,
    )


def main() -> None:
    """Serve file-queue requests while physics advances only during commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object", default="rubiks_cube_hot3d_robolab")
    args = parser.parse_args()
    args.session_dir = args.session_dir.resolve()
    args.session_dir.mkdir(parents=True, exist_ok=True)
    ready_path = args.session_dir / "ready.json"
    assert not ready_path.exists(), "Session is already running; choose a new session directory"
    request_dir = args.session_dir / "requests"
    response_dir = args.session_dir / "responses"
    request_dir.mkdir(exist_ok=True)
    response_dir.mkdir(exist_ok=True)
    assert not list(request_dir.glob("*.json")), "Session has pending requests; use a new directory"

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(argparse.Namespace(headless=True, enable_cameras=True, visualizer=["kit"])):
        env = build_environment(args)
        try:
            session = RobotControlSession(env, args.session_dir)
            initial_result = session.execute({"id": "initial", "command": "observe"})
            write_json(
                ready_path,
                {
                    "pid": os.getpid(),
                    "workspace_min": session.workspace_min,
                    "workspace_max": session.workspace_max,
                    "frame": "robot_base",
                    "quaternion_order": "xyzw",
                    "initial_observation": initial_result["observation"],
                },
            )
            print(f"Robot command server ready: {ready_path}", flush=True)
            should_stop = False
            while not should_stop:
                for request_path in sorted(request_dir.glob("*.json")):
                    request = json.loads(request_path.read_text())
                    assert request["id"] == request_path.stem, "Request id must match its filename"
                    try:
                        session.validate(request)
                    except ValueError as error:
                        result = {
                            "ok": False,
                            "id": request["id"],
                            "outcome": "rejected",
                            "error": str(error),
                        }
                    else:
                        result = session.execute(request)
                    with (args.session_dir / "commands.jsonl").open("a") as stream:
                        stream.write(json.dumps({"request": request, "result": result}) + "\n")
                    write_json(response_dir / request_path.name, result)
                    request_path.unlink()
                    should_stop = result["outcome"] == "shutdown"
                    if should_stop:
                        break
                if not should_stop:
                    time.sleep(0.1)
        finally:
            ready_path.unlink(missing_ok=True)
            env.close()


if __name__ == "__main__":
    main()
