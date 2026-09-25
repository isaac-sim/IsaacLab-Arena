# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visually exercise the current Isaac CAP gear-mesh success behavior.

This is a scripted environment validation, not a robot policy. It first moves
the arm down to grasp, lift, and drop a loose gear, then places the gears on
their stations, presses the board button, and drives the rotation required by
the task termination term.
"""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.isaac_cap.tools import EnvBehaviourDemo

_NUM_ENVS = 1
_GEAR_SPEED_RAD_S = 8.0
_BUTTON_PRESSED_M = -0.007
_DEMO_START_JOINT_POS = (
    0.07864274298115631,
    0.3298062995394371,
    0.044335304471795095,
    -2.5311067752004224,
    -0.05172078111771767,
    2.860254870341865,
    0.17028217529806985,
)
_ROBOTIQ_BASE_TO_GRASP_M = 0.1545
_GRASP_HEIGHT_OFFSET_M = 0.008
_PREGRASP_DISTANCE_M = 0.080
_LIFT_DISTANCE_M = 0.120
_MAX_TRANSLATION_PER_STEP_M = 0.006
_POSITION_TOLERANCE_M = 0.008


def _build_gear_demo_environment(variant: str):
    """Build one current CAP gear-mesh family for the behavior demo."""
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.gear_mesh_environment import (
        GearInsertionEasyNewtonEnvironment,
        GearInsertionEasyNewtonEnvironmentCfg,
        GearMeshPairNewtonEnvironment,
        GearMeshPairNewtonEnvironmentCfg,
        GearMeshTrainNewtonEnvironment,
        GearMeshTrainNewtonEnvironmentCfg,
    )

    canonical_variant = "medium_train" if variant == "medium" else variant
    factories = {
        "easy": (GearInsertionEasyNewtonEnvironment, GearInsertionEasyNewtonEnvironmentCfg),
        "easy_pair": (GearMeshPairNewtonEnvironment, GearMeshPairNewtonEnvironmentCfg),
        "medium_train": (GearMeshTrainNewtonEnvironment, GearMeshTrainNewtonEnvironmentCfg),
    }
    assert canonical_variant in factories, f"Unsupported gear variant {variant!r}."
    factory_type, cfg_type = factories[canonical_variant]
    arena_environment = factory_type().build(cfg_type())

    # Keep the current scene/task/physics, but use the demo-only Cartesian
    # controller so the arm can make a short physical interaction first.
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.embodiment import (
        IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
    )

    initial_pose = arena_environment.embodiment.get_initial_pose()
    arena_environment.embodiment = IndustrialFr3Robotiq2f85DifferentialIKEmbodiment(
        initial_pose=initial_pose,
        initial_joint_pose=list(_DEMO_START_JOINT_POS),
    )
    return arena_environment


class GearMeshBehaviourDemo(EnvBehaviourDemo):
    """Grasp, place, and spin the gears to validate the complete sequence."""

    label = "gear-validation"

    def __init__(
        self,
        simulation_app,
        arena_environment,
        builder_cfg,
        *,
        pause_steps: int,
        real_time: bool = True,
        visualizer_cfg=None,
    ) -> None:
        """Configure the gear-mesh behavior validation."""
        assert pause_steps >= 1, "pause_steps must be positive."
        super().__init__(
            simulation_app,
            arena_environment,
            builder_cfg,
            real_time=real_time,
            visualizer_cfg=visualizer_cfg,
        )
        self.pause_steps = pause_steps

    def setup_demo(self) -> None:
        """Resolve runtime assets, task targets, and Cartesian arm control."""
        import torch

        self.torch = torch
        self.num_envs = self.base_env.num_envs
        assert self.num_envs == _NUM_ENVS, f"Expected {_NUM_ENVS} environment, got {self.num_envs}."
        self.env_ids = torch.arange(self.num_envs, device=self.base_env.device, dtype=torch.int32)

        action_manager = self.base_env.action_manager
        assert action_manager.active_terms == [
            "arm_action",
            "gripper_action",
        ], f"Unexpected action terms: {action_manager.active_terms}."
        assert action_manager.total_action_dim == 7, (
            "The demo requires six relative IK commands and one gripper target; "
            f"got {action_manager.total_action_dim} actions."
        )
        self.arm_action = action_manager.get_term("arm_action")
        self.robot = self.base_env.scene["robot"]
        body_ids, _ = self.robot.find_bodies("robotiq_base")
        assert len(body_ids) == 1, f"Expected one robotiq_base body, got {body_ids}."
        self.ee_body_id = int(body_ids[0])

        success_objective = self.arena_environment.task.get_termination_cfg().success[0]
        gear_mesh_cfg = success_objective.predicate_sequence[0]
        progress_tracker = self.base_env.progress_tracker
        assert progress_tracker is not None, "Gear mesh validation requires task success tracking."
        self.gear_mesh_predicate = progress_tracker.get_predicate(success_objective.name)
        self.board = self.gear_mesh_predicate.board
        self.gears = self.gear_mesh_predicate.gears
        self.gear_names = tuple(asset.cfg.prim_path.rsplit("/", 1)[-1] for asset in self.gears)

        offsets = torch.as_tensor(
            gear_mesh_cfg.params["target_offsets_xyz"],
            device=self.base_env.device,
            dtype=self.board.data.root_pos_w.torch.dtype,
        )
        if offsets.ndim == 2:
            offsets = offsets[None].expand(self.num_envs, -1, -1)
        assert offsets.shape == (
            self.num_envs,
            len(self.gears),
            3,
        ), f"Expected one target offset per environment and gear; got {tuple(offsets.shape)}."
        self.target_offsets_xyz = offsets
        self.gear_angles = torch.zeros((self.num_envs, len(self.gears)), device=self.base_env.device)

    def _ee_position(self):
        """Return the live Robotiq base position."""
        return self.robot.data.body_link_pos_w.torch[:, self.ee_body_id].clone()

    def _action(self, translation_delta_w=None, *, gripper_closed: bool = False):
        """Build a bounded world-frame translation command for relative IK."""
        import isaaclab.utils.math as math_utils

        action = self.torch.zeros(
            (self.num_envs, self.base_env.action_manager.total_action_dim),
            device=self.base_env.device,
        )
        if translation_delta_w is not None:
            delta_b = math_utils.quat_apply_inverse(self.robot.data.root_quat_w.torch, translation_delta_w)
            distance = self.torch.linalg.vector_norm(delta_b, dim=-1, keepdim=True)
            fraction = self.torch.clamp(_MAX_TRANSLATION_PER_STEP_M / distance.clamp_min(1.0e-9), max=1.0)
            action[:, :3] = delta_b * fraction / self.arm_action._scale[:, :3]
        action[:, -1] = float(gripper_closed)
        return action

    def _step(self, action=None, *, drive_gears: bool = False):
        """Apply scripted gear/button state, step once, and return the done mask."""
        if drive_gears:
            self._write_gear_state(spinning=True)
        if action is None:
            action = self._action()
        _, _, terminated, truncated, _ = self.step(action)
        return terminated | truncated

    def _hold(self, steps: int, *, gripper_closed: bool = False) -> bool:
        """Hold the Cartesian target and return whether the episode ended."""
        action = self._action(gripper_closed=gripper_closed)
        return any(bool(self._step(action).any().item()) for _ in range(steps))

    def _move_to(self, target_position_w, *, gripper_closed: bool, label: str) -> bool:
        """Move the Robotiq base to a world-space target using relative IK."""
        print(f"[gear-validation] {label}", flush=True)
        for _ in range(240):
            error_w = target_position_w - self._ee_position()
            error_m = self.torch.linalg.vector_norm(error_w, dim=-1)
            if bool((error_m <= _POSITION_TOLERANCE_M).all().item()):
                return self._hold(10, gripper_closed=gripper_closed)
            if bool(self._step(self._action(error_w, gripper_closed=gripper_closed)).any().item()):
                return True
        error_m = self.torch.linalg.vector_norm(target_position_w - self._ee_position(), dim=-1)
        raise RuntimeError(f"Timed out during {label}; end-effector error is {float(error_m.max()):.3f} m.")

    def _grasp_lift_drop_first_gear(self, cycle: int) -> None:
        """Descend onto the first loose gear, grasp it, lift it, and let go."""
        gear = self.gears[0]
        gear_position = gear.data.root_pos_w.torch.clone()
        grasp = gear_position.clone()
        grasp[:, 2] += _ROBOTIQ_BASE_TO_GRASP_M + _GRASP_HEIGHT_OFFSET_M
        pregrasp = grasp.clone()
        pregrasp[:, 2] += _PREGRASP_DISTANCE_M

        if self._move_to(
            pregrasp,
            gripper_closed=False,
            label=f"cycle {cycle}: approach {self.gear_names[0]}",
        ):
            raise RuntimeError("Environment ended unexpectedly during gear approach.")
        if self._move_to(
            grasp,
            gripper_closed=False,
            label=f"cycle {cycle}: lower onto {self.gear_names[0]}",
        ):
            raise RuntimeError("Environment ended unexpectedly during gear descent.")

        close_steps = max(30, round(1.5 / self.base_env.step_dt))
        print(f"[gear-validation] cycle {cycle}: close gripper", flush=True)
        if self._hold(close_steps, gripper_closed=True):
            raise RuntimeError("Environment ended unexpectedly while closing the gripper.")

        gear_z_before_lift = gear.data.root_pos_w.torch[:, 2].clone()
        lift = grasp.clone()
        lift[:, 2] += _LIFT_DISTANCE_M
        if self._move_to(
            lift,
            gripper_closed=True,
            label=f"cycle {cycle}: lift {self.gear_names[0]}",
        ):
            raise RuntimeError("Environment ended unexpectedly during gear lift.")
        if self._hold(self.pause_steps, gripper_closed=True):
            raise RuntimeError("Environment ended unexpectedly while displaying the gear lift.")

        lift_displacement = (gear.data.root_pos_w.torch[:, 2] - gear_z_before_lift).max()
        print(f"[gear-validation] physical gear lift: {float(lift_displacement):.3f} m", flush=True)

        print(f"[gear-validation] cycle {cycle}: open gripper and drop", flush=True)
        drop_steps = max(30, round(0.75 / self.base_env.step_dt))
        if self._hold(drop_steps, gripper_closed=False):
            raise RuntimeError("Environment ended unexpectedly while dropping the gear.")
        if self._move_to(
            pregrasp,
            gripper_closed=False,
            label=f"cycle {cycle}: retreat from gear",
        ):
            raise RuntimeError("Environment ended unexpectedly during gear retreat.")

    def _target_positions(self):
        """Return each station's current world-space target position."""
        import isaaclab.utils.math as math_utils

        board_pos = self.board.data.root_pos_w.torch
        board_quat = self.board.data.root_quat_w.torch
        offsets_w = math_utils.quat_apply(
            board_quat[:, None, :].expand(-1, len(self.gears), -1).reshape(-1, 4),
            self.target_offsets_xyz.reshape(-1, 3),
        ).reshape(self.num_envs, len(self.gears), 3)
        return board_pos[:, None, :] + offsets_w

    def _write_gear_state(self, *, spinning: bool) -> None:
        """Place gears exactly on their stations and optionally rotate them."""
        import isaaclab.utils.math as math_utils

        board_quat = self.board.data.root_quat_w.torch
        target_positions = self._target_positions()
        local_z = self.torch.tensor([0.0, 0.0, 1.0], device=self.base_env.device, dtype=board_quat.dtype).expand(
            self.num_envs, -1
        )
        board_up = math_utils.quat_apply(board_quat, local_z)
        if spinning:
            signs = self.torch.tensor(
                [(-1.0 if index % 2 == 0 else 1.0) for index in range(len(self.gears))],
                device=self.base_env.device,
            )
            self.gear_angles += signs[None, :] * _GEAR_SPEED_RAD_S * self.base_env.step_dt
            # The demo authors each visible gear pose every control frame. Seed
            # the task's matching finite-window velocity state as well because
            # Newton can dissipate an injected rigid-body velocity almost fully
            # while resolving the intentionally intermeshed teeth in one step.
            self.gear_mesh_predicate.spin_history[:] = signs[None, None, :] * _GEAR_SPEED_RAD_S
            self.gear_mesh_predicate.spin_samples_seen.fill_(self.gear_mesh_predicate.spin_window_steps)

        for gear_index, gear in enumerate(self.gears):
            half_angle = self.gear_angles[:, gear_index] * 0.5
            yaw_quat = self.torch.zeros((self.num_envs, 4), device=self.base_env.device, dtype=board_quat.dtype)
            yaw_quat[:, 2] = self.torch.sin(half_angle)
            yaw_quat[:, 3] = self.torch.cos(half_angle)
            gear_quat = math_utils.quat_mul(board_quat, yaw_quat)
            gear.write_root_pose_to_sim_index(
                root_pose=self.torch.cat((target_positions[:, gear_index], gear_quat), dim=-1),
                env_ids=self.env_ids,
            )
            root_velocity = self.torch.zeros((self.num_envs, 6), device=self.base_env.device, dtype=board_quat.dtype)
            if spinning:
                sign = -1.0 if gear_index % 2 == 0 else 1.0
                root_velocity[:, 3:] = board_up * (_GEAR_SPEED_RAD_S * sign)
            gear.write_root_velocity_to_sim_index(
                root_velocity=root_velocity,
                env_ids=self.env_ids,
            )

    def _write_button(self, position_m: float) -> None:
        """Write and hold the board button at one prismatic position."""
        position = self.torch.full(
            (self.num_envs, 1),
            position_m,
            device=self.base_env.device,
            dtype=self.board.data.joint_pos.torch.dtype,
        )
        joint_ids = self.torch.tensor(
            [self.gear_mesh_predicate.button_joint], device=self.base_env.device, dtype=self.torch.int32
        )
        self.board.set_joint_position_target(
            position,
            joint_ids=[self.gear_mesh_predicate.button_joint],
        )
        self.board.write_joint_position_to_sim_index(
            position=position,
            joint_ids=joint_ids,
            env_ids=self.env_ids,
        )

    def _show_placed_gears(self) -> None:
        """Place gears one at a time without advancing the physics."""
        for gear_index, gear_name in enumerate(self.gear_names):
            print(
                f"[gear-validation] place {gear_name} on station {gear_index + 1}/{len(self.gears)}",
                flush=True,
            )
            self._write_gear_state(spinning=False)
            for _ in range(self.pause_steps):
                self.render()

    def run_cycle(self, cycle: int) -> None:
        """Run seating, post-seating button latch, rotation, and success reset."""
        print(f"[gear-validation] cycle {cycle}: settling", flush=True)
        if self._hold(30):
            raise RuntimeError("Environment ended unexpectedly while settling.")

        self._grasp_lift_drop_first_gear(cycle)
        self.gear_angles.zero_()
        self._show_placed_gears()
        # Newton resolves contacts after each scripted state write, so wait for
        # the task itself to observe a fully seated frame before starting it.
        for _ in range(30):
            self._write_button(0.0)
            if bool(self._step(drive_gears=True).any().item()):
                raise RuntimeError("Environment ended before the board was started.")
            if bool(self.gear_mesh_predicate.seated_seen.all().item()):
                break
        if not bool(self.gear_mesh_predicate.seated_seen.all().item()):
            raise RuntimeError("Task did not observe every gear seated before button press.")
        if bool(self.gear_mesh_predicate.latched.any().item()):
            raise RuntimeError("Board motor latched before the scripted button press.")

        print(f"[gear-validation] cycle {cycle}: pressing the start button", flush=True)
        for _ in range(30):
            if bool(self.gear_mesh_predicate.latched.all().item()):
                break
            self._write_button(_BUTTON_PRESSED_M)
            if bool(self._step(drive_gears=True).any().item()):
                raise RuntimeError("Environment ended while pressing the board button.")
        if not bool(self.gear_mesh_predicate.latched.all().item()):
            button_position = self.board.data.joint_pos.torch[:, self.gear_mesh_predicate.button_joint]
            raise RuntimeError(f"Board button did not latch; joint position is {button_position.tolist()}.")

        print(f"[gear-validation] cycle {cycle}: motor latched; validating driven rotation", flush=True)
        maximum_steps = self.gear_mesh_predicate.spin_window_steps + self.gear_mesh_predicate.required_steps + 60
        for _ in range(maximum_steps):
            if bool(self._step(drive_gears=True).all().item()):
                print(
                    f"[gear-validation] cycle {cycle}: success reset observed in all {self.num_envs} environment",
                    flush=True,
                )
                return

        windowed_spin = self.gear_mesh_predicate.spin_history.mean(dim=0)
        raise RuntimeError(
            "Driven gear state did not trigger success: "
            f"latched={self.gear_mesh_predicate.latched.tolist()}, "
            f"seated_seen={self.gear_mesh_predicate.seated_seen.tolist()}, "
            f"started_after_seating={self.gear_mesh_predicate.started_after_seating.tolist()}, "
            f"success_steps={self.gear_mesh_predicate.success_steps.tolist()}, "
            f"windowed_spin={windowed_spin.tolist()}."
        )


def run_demo(
    simulation_app,
    *,
    variant: str = "medium",
    cycles: int = 0,
    pause_steps: int = 30,
    real_time: bool = True,
) -> None:
    """Compose a gear-mesh task and run its behavior validation lifecycle."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = GearMeshBehaviourDemo(
        simulation_app,
        _build_gear_demo_environment(variant),
        ArenaEnvBuilderCfg(num_envs=_NUM_ENVS, env_spacing=1.5, solve_relations=True),
        real_time=real_time,
        visualizer_cfg=KitVisualizerCfg(
            eye=(2.0, -2.5, 2.0),
            lookat=(0.0, 0.0, 0.82),
            origin_type="world",
        ),
        pause_steps=pause_steps,
    )
    demo.run_demo(cycles)


def main() -> None:
    """Launch the visual gear-mesh behavior validation."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        nargs="?",
        choices=("easy", "easy_pair", "medium", "medium_train"),
        default="medium",
        help="Gear family; 'medium' is a compatibility alias for 'medium_train'.",
    )
    parser.add_argument("--cycles", type=int, default=0, help="Cycles to run; zero repeats until Kit closes.")
    parser.add_argument("--pause-steps", type=int, default=30, help="Rendered frames shown after placing gears.")
    parser.add_argument("--no-real-time", action="store_true", help="Run without wall-clock rate limiting.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    args.limit_cpu_threads = 1

    with SimulationAppContext(args) as simulation_app:
        run_demo(
            simulation_app,
            variant=args.variant,
            cycles=args.cycles,
            real_time=not args.no_real_time,
            pause_steps=args.pause_steps,
        )


if __name__ == "__main__":
    main()
