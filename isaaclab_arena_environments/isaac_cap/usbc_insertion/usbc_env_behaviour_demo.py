# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Teleport through every USB-C success predicate, then verify automatic reset.

Run inside the Arena container from ``/workspaces/isaaclab_arena``::

    DEMO=isaaclab_arena_environments/isaac_cap/usbc_insertion/usbc_env_behaviour_demo.py
    python "$DEMO" easy --cycles 2 --viz kit
    python "$DEMO" medium --cycles 2 --viz kit

For a fast headless check, replace ``--viz kit`` with
``--viz none --pause-steps 1 --no-real-time``.

This is a predicate and reset diagnostic, not a robot policy. Its predefined
teleport trajectory makes depth, lateral alignment, low velocity, gripper
release, and TCP withdrawal become true one at a time. The final state is passed
through the ordinary task termination and automatic environment reset path.
"""

from __future__ import annotations

import argparse
import math

from isaaclab_arena_environments.isaac_cap.tools import EnvBehaviourDemo

_MATING_ROTATIONS_XYZW = {
    "easy": (-0.5, -0.5, -0.5, 0.5),
    "medium": (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
}
"""Plug-to-receiver rotations aligning both the insertion and wide cross-section axes."""

_LATERAL_FAILURE_MARGIN_M = 0.005
_TCP_WITHDRAWAL_MARGIN_M = 0.05


def _build_demo_environment(variant: str):
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )

    assert variant in ("easy", "medium"), f"Unsupported USB-C variant: {variant!r}."
    if variant == "easy":
        return UsbcInsertionEasyEnvironment().build(UsbcInsertionEasyEnvironmentCfg())
    return UsbcInsertionMediumEnvironment().build(UsbcInsertionMediumEnvironmentCfg())


def _plug_pose_at_depth(T_W_R, q_R_P, mating_params, depth: float):
    """Return the centered plug pose at a specified receiver-axis depth.

    Args:
        T_W_R: Batched receiver-to-world poses in XYZ/XYZW order.
        q_R_P: Batched plug-to-receiver mating orientations in XYZW order.
        mating_params: Task predicate parameters defining the tip, mouth, and axis.
        depth: Signed tip depth along the receiver's inward axis, in meters.

    Returns:
        Batched plug-to-world poses satisfying the requested depth and zero lateral error.
    """
    import torch

    from isaaclab.utils.math import quat_apply, quat_mul

    axis_R = T_W_R.new_tensor(mating_params["receiver_axis"])
    axis_R = axis_R / torch.linalg.vector_norm(axis_R)
    mouth_R = T_W_R.new_tensor(mating_params["target_offset_xyz"])
    tip_P = T_W_R.new_tensor(mating_params["subject_offset_xyz"]).expand(T_W_R.shape[0], -1)
    q_W_P = quat_mul(T_W_R[:, 3:], q_R_P)
    tip_W = T_W_R[:, :3] + quat_apply(T_W_R[:, 3:], (mouth_R + depth * axis_R).expand(T_W_R.shape[0], -1))
    return torch.cat((tip_W - quat_apply(q_W_P, tip_P), q_W_P), dim=-1)


class UsbcEnvBehaviourDemo(EnvBehaviourDemo):
    """Walk the unmodified task predicates through a deterministic teleport trajectory."""

    label = "usbc-validation"

    def __init__(self, *args, variant: str, pause_steps: int = 60, **kwargs) -> None:
        assert pause_steps > 0, "pause_steps must be positive."
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.pause_steps = pause_steps

    def setup_demo(self) -> None:
        """Resolve the five predicates and state writers used by the trajectory."""
        import torch

        from isaaclab_arena.tasks.predicates.gripper import gripper_released
        from isaaclab_arena.tasks.predicates.spatial import (
            depth_in_range,
            gripper_distance_from_object_exceeds_threshold,
            lateral_in_proximity,
            velocity_below_threshold,
        )
        from isaaclab_arena_environments.isaac_cap.embodiments.cable_routing.config import (
            GRIPPER_CLOSED_POSITION,
            GRIPPER_OPEN_POSITION,
        )

        self.torch = torch
        self.task = self.arena_environment.task
        self.predicates = self.task.get_termination_cfg().success[0].predicate_sequence[0].params["predicates"]
        expected_functions = [
            depth_in_range,
            lateral_in_proximity,
            velocity_below_threshold,
            gripper_released,
            gripper_distance_from_object_exceeds_threshold,
        ]
        actual_functions = [predicate.func for predicate in self.predicates]
        assert actual_functions == expected_functions, (
            "The USB-C validation trajectory must be updated for the task's predicate sequence: "
            f"{[function.__name__ for function in actual_functions]}."
        )

        self.mating = self.predicates[0].params
        self.lateral_tolerance_m = self.predicates[1].params["tolerance_lateral"]
        self.speed_threshold_m_s = self.predicates[2].params["linear_velocity_threshold"]
        depth_min = self.mating["depth_min"]
        depth_max = self.mating["depth_max"]
        self.target_depth = (depth_min + depth_max) / 2 if depth_max is not None else depth_min + 0.001

        self.num_envs = self.base_env.num_envs
        self.env_ids = torch.arange(self.num_envs, device=self.base_env.device, dtype=torch.int32)
        self.zero_velocity = torch.zeros((self.num_envs, 6), device=self.base_env.device)
        self.q_R_P = self.zero_velocity.new_tensor(_MATING_ROTATIONS_XYZW[self.variant]).expand(self.num_envs, -1)
        self.receiver_axis_R = self.zero_velocity.new_tensor(self.mating["receiver_axis"][:3])
        self.receiver_axis_R /= torch.linalg.vector_norm(self.receiver_axis_R)
        least_aligned_axis = int(torch.argmin(self.receiver_axis_R.abs()).item())
        basis_R = torch.zeros_like(self.receiver_axis_R)
        basis_R[least_aligned_axis] = 1.0
        self.lateral_axis_R = torch.linalg.cross(self.receiver_axis_R, basis_R)
        self.lateral_axis_R /= torch.linalg.vector_norm(self.lateral_axis_R)

        release_params = self.predicates[3].params
        self.gripper = release_params["gripper"]
        self.work_robot = self.base_env.scene[self.gripper.articulation_name]
        gripper_joint_ids, gripper_joint_names = self.work_robot.find_joints(self.gripper.driver_joint_name)
        assert len(gripper_joint_ids) == 1, f"Expected one work-hand gripper joint, got {gripper_joint_names}."
        self.gripper_joint_ids = gripper_joint_ids
        self.gripper_open_position = GRIPPER_OPEN_POSITION
        self.gripper_closed_position = GRIPPER_CLOSED_POSITION

        withdrawal_params = self.predicates[4].params
        assert withdrawal_params["gripper"] is self.gripper, "Release and withdrawal must use the same gripper."
        self.withdrawal_distance_m = withdrawal_params["distance_threshold_m"]
        self.ee_sensor = self.base_env.scene[self.gripper.frame_transformer_name]
        self.ee_sensor_data = self.ee_sensor.data
        self.ee_target_index = self.ee_sensor_data.target_frame_names.index(self.gripper.target_frame_name)
        target_frame_cfg = next(
            frame for frame in self.ee_sensor.cfg.target_frames if frame.name == self.gripper.target_frame_name
        )
        ee_body_name = target_frame_cfg.prim_path.rsplit("/", maxsplit=1)[-1]
        ee_body_ids, ee_body_names = self.work_robot.find_bodies(ee_body_name)
        assert len(ee_body_ids) == 1, f"Expected one work-hand end-effector body, got {ee_body_names}."
        self.ee_body_id = int(ee_body_ids[0])
        assert self.ee_body_id > 0, "The fixed articulation root has no Jacobian entry."
        self.ee_offset = self.zero_velocity.new_tensor(target_frame_cfg.offset.pos)
        arm_term_name = next(
            name
            for name in self.base_env.action_manager.active_terms
            if name.endswith("arm_action") and self.base_env.action_manager.get_term(name)._asset is self.work_robot
        )
        self.arm_joint_ids = self.base_env.action_manager.get_term(arm_term_name)._joint_ids
        self.arm_joint_seed = self.work_robot.data.joint_pos.torch[:, self.arm_joint_ids].clone()
        self.ee_home_quaternion = self.work_robot.data.body_link_quat_w.torch[:, self.ee_body_id].clone()

    def _write_pose(self, name: str, T_W_O, velocity) -> None:
        """Teleport a connector pose and velocity into every demo environment."""
        asset = self.base_env.scene[name]
        asset.write_root_pose_to_sim_index(root_pose=T_W_O, env_ids=self.env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=self.env_ids)

    def _set_gripper_released(self, released: bool) -> None:
        """Teleport the measured work-hand joint to its open or closed position."""
        position = self.gripper_open_position if released else self.gripper_closed_position
        joint_position = self.zero_velocity.new_full((self.num_envs, 1), position)
        joint_velocity = self.zero_velocity.new_zeros((self.num_envs, 1))
        self.work_robot.write_joint_state_to_sim_index(
            position=joint_position,
            velocity=joint_velocity,
            joint_ids=self.gripper_joint_ids,
            env_ids=self.env_ids,
        )

    def _place(self, T_W_R, *, depth: float, lateral_m: float, speed_m_s: float):
        """Teleport both connectors to an exact relative pose and plug speed."""
        from isaaclab.utils.math import quat_apply

        T_W_P = _plug_pose_at_depth(T_W_R, self.q_R_P, self.mating, depth)
        lateral_R = lateral_m * self.lateral_axis_R
        T_W_P[:, :3] += quat_apply(T_W_R[:, 3:], lateral_R.expand(self.num_envs, -1))

        plug_velocity = self.zero_velocity.clone()
        speed_R = speed_m_s * self.receiver_axis_R
        plug_velocity[:, :3] = quat_apply(T_W_R[:, 3:], speed_R.expand(self.num_envs, -1))
        self._write_pose(self.task.receiver.name, T_W_R, self.zero_velocity)
        self._write_pose(self.task.plug.name, T_W_P, plug_velocity)
        return T_W_P

    def _forward_attached_cables(self) -> None:
        """Refresh cable links after a paused connector teleport."""
        from isaaclab_newton.physics import NewtonManager

        # The connector and its attached cable are separate Newton articulations.
        # Recompute the full graph so the cable's parent-relative links follow a
        # connector teleport before any preview frame is rendered.
        NewtonManager.invalidate_fk()
        self.base_env.sim.forward()

    def _work_tcp_from_robot(self):
        """Return the TCP position from the work robot's live link pose."""
        from isaaclab.utils.math import quat_apply

        T_W_E = self.work_robot.data.body_link_pose_w.torch[:, self.ee_body_id]
        return T_W_E[:, :3] + quat_apply(T_W_E[:, 3:], self.ee_offset.expand(self.num_envs, -1))

    def _teleport_work_tcp(self, target_position_W) -> None:
        """Teleport the work arm joints so its TCP reaches a world-space keyframe."""
        from isaaclab.utils.math import compute_pose_error, quat_apply

        desired_quaternion = self.ee_home_quaternion
        desired_link_position = target_position_W - quat_apply(
            desired_quaternion, self.ee_offset.expand(self.num_envs, -1)
        )
        identity = self.torch.eye(6, device=self.base_env.device).unsqueeze(0)
        joint_velocity = self.zero_velocity.new_zeros((self.num_envs, len(self.arm_joint_ids)))

        for _ in range(80):
            T_W_E = self.work_robot.data.body_link_pose_w.torch[:, self.ee_body_id]
            position_error, orientation_error = compute_pose_error(
                T_W_E[:, :3],
                T_W_E[:, 3:],
                desired_link_position,
                desired_quaternion,
            )
            if float(self.torch.linalg.vector_norm(position_error, dim=-1).max()) < 5.0e-4:
                break
            jacobian = self.work_robot.data.body_link_jacobian_w.torch[:, self.ee_body_id - 1, :, self.arm_joint_ids]
            orientation_weight = 0.25
            weighted_jacobian = self.torch.cat(
                (jacobian[:, :3], orientation_weight * jacobian[:, 3:6]),
                dim=1,
            )
            task_error = self.torch.cat((position_error, orientation_weight * orientation_error), dim=-1)
            jacobian_t = weighted_jacobian.transpose(-1, -2)
            pseudo_inverse = jacobian_t @ self.torch.linalg.solve(
                weighted_jacobian @ jacobian_t + 0.025**2 * identity,
                identity,
            )
            joint_position = self.work_robot.data.joint_pos.torch[:, self.arm_joint_ids]
            task_delta = (pseudo_inverse @ task_error.unsqueeze(-1)).squeeze(-1)
            null_space = identity - pseudo_inverse @ weighted_jacobian
            posture_delta = (null_space @ (self.arm_joint_seed - joint_position).unsqueeze(-1)).squeeze(-1)
            joint_position = joint_position + (0.65 * task_delta + 0.05 * posture_delta).clamp(-0.10, 0.10)
            self.work_robot.write_joint_state_to_sim_index(
                position=joint_position,
                velocity=joint_velocity,
                joint_ids=self.arm_joint_ids,
                env_ids=self.env_ids,
            )

        tcp_position = self._work_tcp_from_robot()
        error_m = self.torch.linalg.vector_norm(tcp_position - target_position_W, dim=-1)
        assert bool((error_m < 0.015).all().item()), f"Work TCP failed to reach keyframe: {error_m.tolist()} m."
        # Frame-transformer outputs update during physics steps. Keep the diagnostic
        # sensor synchronized with this preview-only joint teleport without stepping.
        self.ee_sensor_data.target_pos_w.torch[:, self.ee_target_index].copy_(tcp_position)

    def _tcp_position(self):
        """Return the work-hand TCP position used by the withdrawal predicate."""
        return self.gripper.get_position_w(self.base_env.arena_world)

    def _predicate_results(self) -> list[bool]:
        """Evaluate every child predicate without advancing task termination."""
        return [bool(predicate.func(self.base_env, **predicate.params).all().item()) for predicate in self.predicates]

    def _diagnostics(self) -> dict:
        """Return compact geometric and predicate state for waypoint logging."""
        from isaaclab_arena.tasks.predicates.spatial import _relative_axial_distances

        depth, lateral = _relative_axial_distances(
            self.base_env,
            **{key: value for key, value in self.mating.items() if key not in ("depth_min", "depth_max")},
        )
        tcp_distance = self.torch.linalg.vector_norm(
            self._tcp_position() - self.base_env.arena_world.get_pose_w(self.task.plug.name)[:, :3], dim=-1
        )
        return {
            "depth_mm": (1000 * depth).tolist(),
            "lateral_mm": (1000 * lateral).tolist(),
            "speed_mm_s": (
                1000
                * self.torch.linalg.vector_norm(
                    self.base_env.arena_world.get_root_linear_velocity_w(self.task.plug.name), dim=-1
                )
            ).tolist(),
            "tcp_distance_mm": (1000 * tcp_distance).tolist(),
        }

    def _apply_state(
        self,
        T_W_R,
        *,
        lateral_m: float,
        speed_m_s: float,
        gripper_released: bool,
        depth: float | None = None,
        tcp_withdrawn: bool = False,
    ) -> None:
        """Apply one predefined trajectory state without advancing physics."""
        T_W_P = self._place(
            T_W_R,
            depth=self.target_depth if depth is None else depth,
            lateral_m=lateral_m,
            speed_m_s=speed_m_s,
        )
        target_position_W = T_W_P[:, :3]
        if tcp_withdrawn:
            from isaaclab.utils.math import quat_apply

            withdrawal_R = -(self.withdrawal_distance_m + _TCP_WITHDRAWAL_MARGIN_M) * self.receiver_axis_R
            target_position_W = target_position_W + quat_apply(T_W_R[:, 3:], withdrawal_R.expand(self.num_envs, -1))
        self._teleport_work_tcp(target_position_W)
        self._set_gripper_released(gripper_released)

    def _show_state(self, label: str, expected_results: list[bool], **state) -> None:
        """Apply, render, and verify one trajectory state."""
        self._apply_state(**state)
        self._forward_attached_cables()
        for _ in range(self.pause_steps):
            self.render()
        actual_results = self._predicate_results()
        assert (
            actual_results == expected_results
        ), f"Waypoint {label!r} produced {actual_results}, expected {expected_results}: {self._diagnostics()}"
        active = [predicate.func.__name__ for predicate, passed in zip(self.predicates, actual_results) if passed]
        print(f"[{self.label}] {label}: active={active}; {self._diagnostics()}", flush=True)

    def _hold_open_action(self):
        """Hold arm positions and command both normalized gripper actions open."""
        action = self.torch.zeros(self.env.action_space.shape, device=self.base_env.device)
        offset = 0
        for name in self.base_env.action_manager.active_terms:
            term = self.base_env.action_manager.get_term(name)
            if name.endswith("arm_action"):
                action[:, offset : offset + term.action_dim] = term._asset.data.joint_pos.torch[:, term._joint_ids]
            offset += term.action_dim
        return action

    def _verify_success_reset(self, final_state: dict) -> None:
        """Pass the final state through task termination and require automatic reset."""
        succeeded = self.torch.zeros(self.num_envs, dtype=self.torch.bool, device=self.base_env.device)
        episode_indices = [self.base_env.get_episode_index(env_id) for env_id in range(self.num_envs)]
        verification_state = {**final_state, "depth": self.target_depth + 0.003}
        velocity_compensation_W = self.zero_velocity.clone()
        for _ in range(max(self.pause_steps, 30)):
            self._apply_state(**verification_state)
            self.base_env.scene[self.task.plug.name].write_root_velocity_to_sim_index(
                root_velocity=velocity_compensation_W,
                env_ids=self.env_ids,
            )
            _, _, terminated, truncated, _ = self.step(self._hold_open_action())
            success = self.base_env.termination_manager.get_term("success")
            unexpected = (terminated | truncated) & ~success & ~succeeded
            assert not bool(unexpected.any().item()), "Episode ended without USB-C success."
            succeeded |= terminated & success
            if bool(succeeded.all().item()):
                assert all(
                    self.base_env.get_episode_index(env_id) > episode_indices[env_id] for env_id in range(self.num_envs)
                ), "Success did not trigger the normal environment reset."
                assert not all(self._predicate_results()), "The automatic reset left the task in its success state."
                print(f"[{self.label}] success reset observed in all {self.num_envs} environments", flush=True)
                return
            cable_induced_velocity_W = self.base_env.arena_world.get_root_linear_velocity_w(self.task.plug.name)
            velocity_compensation_W[:, :3] = (velocity_compensation_W[:, :3] - 0.25 * cable_induced_velocity_W).clamp(
                -0.25, 0.25
            )
        raise RuntimeError(f"Final waypoint did not produce success/reset: {self._diagnostics()}")

    def run_cycle(self, cycle: int) -> None:
        """Make one additional predicate true at each predefined waypoint."""
        home_receiver_pose = self.base_env.arena_world.get_pose_w(self.task.receiver.name).clone()

        lateral_failure_m = self.lateral_tolerance_m + _LATERAL_FAILURE_MARGIN_M
        fast_speed_m_s = 2.0 * self.speed_threshold_m_s
        start_state = {
            "T_W_R": home_receiver_pose,
            "lateral_m": lateral_failure_m,
            "speed_m_s": fast_speed_m_s,
            "gripper_released": False,
        }
        self._apply_state(**start_state, depth=-0.004)
        self._forward_attached_cables()
        for _ in range(self.pause_steps):
            self.render()
        assert self._predicate_results() == [False] * len(
            self.predicates
        ), f"Cycle {cycle} did not start with all predicates false: {self._diagnostics()}"
        print(f"[{self.label}] cycle {cycle}: start with all predicates false", flush=True)

        trajectory = (
            ("depth_in_range", [True, False, False, False, False], start_state),
            (
                "lateral_in_proximity",
                [True, True, False, False, False],
                {**start_state, "lateral_m": 0.0},
            ),
            (
                "velocity_below_threshold",
                [True, True, True, False, False],
                {**start_state, "lateral_m": 0.0, "speed_m_s": 0.0},
            ),
            (
                "gripper_released",
                [True, True, True, True, False],
                {**start_state, "lateral_m": 0.0, "speed_m_s": 0.0, "gripper_released": True},
            ),
            (
                "gripper_distance_from_object_exceeds_threshold",
                [True, True, True, True, True],
                {
                    "T_W_R": home_receiver_pose,
                    "lateral_m": 0.0,
                    "speed_m_s": 0.0,
                    "gripper_released": True,
                    "tcp_withdrawn": True,
                },
            ),
        )
        for index, (name, expected_results, state) in enumerate(trajectory):
            assert self.predicates[index].func.__name__ == name
            self._show_state(f"cycle {cycle} waypoint {index + 1}: {name}", expected_results, **state)

        self._verify_success_reset(trajectory[-1][2])


def run_demo(
    simulation_app,
    *,
    variant: str = "medium",
    cycles: int = 0,
    pause_steps: int = 60,
    real_time: bool = True,
    device: str = "cuda:0",
    visualizer_cfg=None,
) -> None:
    """Run the predefined predicate trajectory against the unchanged USB-C task."""
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = UsbcEnvBehaviourDemo(
        simulation_app,
        _build_demo_environment(variant),
        ArenaEnvBuilderCfg(num_envs=1, device=device),
        variant=variant,
        pause_steps=pause_steps,
        real_time=real_time,
        visualizer_cfg=visualizer_cfg,
    )
    demo.run_demo(cycles)


def main() -> None:
    """Launch the predefined USB-C predicate trajectory."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=("easy", "medium"), default="medium")
    parser.add_argument("--cycles", type=int, default=0, help="Zero repeats until the simulation closes.")
    parser.add_argument("--pause-steps", type=int, default=60)
    parser.add_argument("--no-real-time", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    if args.cycles < 0 or args.pause_steps < 1:
        parser.error("cycles must be non-negative and pause-steps must be positive.")
    args.limit_cpu_threads = 1
    with SimulationAppContext(args) as simulation_app:
        visualizer_cfg = None
        if "kit" in (args.visualizer or ()):
            from isaaclab_visualizers.kit import KitVisualizerCfg

            center = (0.44, 0.0, 0.82)
            visualizer_cfg = KitVisualizerCfg(
                eye=(center[0] + 0.45, center[1] - 0.6, center[2] + 0.4), lookat=center, origin_type="world"
            )
        run_demo(
            simulation_app,
            variant=args.variant,
            cycles=args.cycles,
            pause_steps=args.pause_steps,
            real_time=not args.no_real_time,
            device=args.device,
            visualizer_cfg=visualizer_cfg,
        )


if __name__ == "__main__":
    main()
