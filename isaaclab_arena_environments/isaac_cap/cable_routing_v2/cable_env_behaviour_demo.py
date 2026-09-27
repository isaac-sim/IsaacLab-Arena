# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visually validate the current CAP cable-routing task over Arena Cable.

This is a scripted environment check, not a robot policy. It first moves both
arms into the native cable for a physical push, authors a connected terminal
route through the goal regions, and demonstrates success followed by reset.
"""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.isaac_cap.tools import EnvBehaviourDemo

_NUM_ENVS = 1
_REGION_DWELL_HALF_LENGTH = 0.020
_MAX_ARM_JOINT_STEP = 0.035
_GRIPPER_OPEN = 0.0
_GRIPPER_CONTACT = 0.65
_TOOL_OFFSET = (0.0, 0.0, -0.1347)
_HIGH_HOVER_CLEARANCE_M = 0.100
_CONTACT_CLEARANCE_M = 0.006
_PUSH_DISTANCE_M = 0.035
_RETREAT_CLEARANCE_M = 0.100
_RIGHT_APPROACH_JOINTS = (0.409067, 2.066704, 0.474773, 1.433090, 0.400142, 0.105784)
_LEFT_APPROACH_JOINTS = (
    -_RIGHT_APPROACH_JOINTS[0],
    _RIGHT_APPROACH_JOINTS[1],
    _RIGHT_APPROACH_JOINTS[2],
    _RIGHT_APPROACH_JOINTS[3],
    -_RIGHT_APPROACH_JOINTS[4],
    -_RIGHT_APPROACH_JOINTS[5],
)


def _build_cable_demo_environment(variant: str):
    """Build one current CAP cable-routing variant."""
    from isaaclab_arena_environments.isaac_cap.cable_routing_v2.environment import (
        CableRoutingEasyEnvironment,
        CableRoutingEasyEnvironmentCfg,
        CableRoutingMediumEnvironment,
        CableRoutingMediumEnvironmentCfg,
    )

    factories = {
        "easy": (CableRoutingEasyEnvironment, CableRoutingEasyEnvironmentCfg),
        "medium": (CableRoutingMediumEnvironment, CableRoutingMediumEnvironmentCfg),
    }
    assert variant in factories, f"Unsupported cable-routing variant {variant!r}."
    factory_type, cfg_type = factories[variant]
    return factory_type().build(cfg_type())


def _polyline_length(points):
    """Return the length of a 3-D polyline."""
    import torch

    return torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum()


def _sample_polyline(points, distances):
    """Sample a polyline at monotonically increasing arc distances."""
    import torch

    edges = points[1:] - points[:-1]
    edge_lengths = torch.linalg.vector_norm(edges, dim=-1)
    cumulative = torch.cat((edge_lengths.new_zeros(1), edge_lengths.cumsum(dim=0)))
    segment = torch.searchsorted(cumulative.contiguous(), distances.contiguous(), right=True) - 1
    segment = segment.clamp(min=0, max=edges.shape[0] - 1)
    fraction = (distances - cumulative[segment]) / edge_lengths[segment].clamp_min(torch.finfo(points.dtype).eps)
    return points[segment] + fraction[:, None] * edges[segment]


def _vertices_to_segment_poses(vertices):
    """Convert connected centerline vertices into XYZW capsule poses."""
    import torch

    edge = vertices[1:] - vertices[:-1]
    direction = torch.nn.functional.normalize(edge, dim=-1)
    center = 0.5 * (vertices[:-1] + vertices[1:])

    # Arena Cable capsules use local +Z as their long axis. This is the
    # shortest-arc quaternion rotating +Z onto each segment direction.
    quaternion = torch.stack(
        (
            -direction[:, 1],
            direction[:, 0],
            torch.zeros_like(direction[:, 0]),
            1.0 + direction[:, 2],
        ),
        dim=-1,
    )
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    return torch.cat((center, quaternion), dim=-1)


class CurrentCableRoutingBehaviourDemo(EnvBehaviourDemo):
    """Validate current CAP terminal scoring with Arena's native Cable asset."""

    label = "cable-behaviour-demo"

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
        """Configure the current cable behavior validation."""
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
        """Resolve the native cable, current goal, and bimanual controls."""
        import torch

        self.torch = torch
        self.num_envs = self.base_env.num_envs
        assert self.num_envs == _NUM_ENVS, f"Expected {_NUM_ENVS} environment, got {self.num_envs}."
        assert (
            self.base_env.action_manager.total_action_dim == 14
        ), "The current I2RT embodiment requires two six-joint arms and two gripper commands."
        self.env_ids = torch.arange(self.num_envs, device=self.base_env.device, dtype=torch.int32)
        self.hold_action = torch.zeros((self.num_envs, 14), device=self.base_env.device)
        offset = 0
        self.arm_joint_ids = []
        for side in ("left", "right"):
            action_term = self.base_env.action_manager.get_term(f"{side}_arm_action")
            self.arm_joint_ids.append(action_term._joint_ids)
            joint_position = self.base_env.scene[f"{side}_robot"].data.joint_pos.torch[:, action_term._joint_ids]
            self.hold_action[:, offset : offset + joint_position.shape[1]] = joint_position
            offset += joint_position.shape[1] + 1

        self.gripper_body_ids = []
        self.gripper_quaternions = []
        for side in ("left", "right"):
            robot = self.base_env.scene[f"{side}_robot"]
            body_ids, _ = robot.find_bodies(f"{side}_gripper")
            assert len(body_ids) == 1, f"Expected one {side}_gripper body, got {body_ids}."
            body_id = int(body_ids[0])
            assert body_id > 0, "The fixed articulation root has no Jacobian entry."
            self.gripper_body_ids.append(body_id)
            self.gripper_quaternions.append(robot.data.body_link_quat_w.torch[:, body_id].clone())

        success_cfg = self.base_env.termination_manager.get_term_cfg("success")
        self.success_term = success_cfg.func
        self.success_params = success_cfg.params
        self.goal = self.success_params["goal"]
        # A state teleport produces a one-frame solver velocity spike even
        # when the authored cable is connected at exact rest length. Ignore
        # that demo artifact; the production task retains CAP's 0.05 m/s
        # terminal speed requirement.
        self.goal.max_mean_speed = float("inf")
        self.cable = self.base_env.scene[self.success_params["cable_asset_name"]]
        self.peg_names = tuple(self.success_params["peg_asset_names"])
        self.port = self.base_env.scene[self.success_params["port_asset_name"]]
        # Current CAP's Arena fork exposes this buffer through its graph-policy
        # finish API. Keep the compatibility shim local to the demo rather than
        # changing Arena core merely to drive this temporary port.
        self.base_env.external_policy_termination_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.base_env.device
        )
        self.segment_lengths = 2.0 * torch.as_tensor(
            self.success_params["cable_half_lengths"],
            device=self.base_env.device,
            dtype=self.cable.data.segment_pose_w.torch.dtype,
        )

    def _tool_targets(self, clearance_m: float):
        """Return targets over cable sections naturally assigned to each arm."""
        cable_positions = self.cable.data.segment_pose_w.torch[..., :3]
        last_index = self.cable.num_segments - 1
        indices = (round(0.75 * last_index), round(0.25 * last_index))
        targets = []
        for index in indices:
            target = cable_positions[:, index].clone()
            target[:, 2] += clearance_m
            targets.append(target)
        return tuple(targets)

    def _run_joint_approach(self, cycle: int) -> None:
        """Move out of the folded home posture before Cartesian cable motion."""
        print(f"[cable-behaviour-demo] cycle {cycle}: unfold both arms toward the cable", flush=True)
        targets = (_LEFT_APPROACH_JOINTS, _RIGHT_APPROACH_JOINTS)
        maximum_error = float("inf")
        for step in range(180):
            action = self.torch.zeros_like(self.hold_action)
            errors = []
            for side_index, (side, target_values) in enumerate(zip(("left", "right"), targets, strict=True)):
                robot = self.base_env.scene[f"{side}_robot"]
                joint_ids = self.arm_joint_ids[side_index]
                position = robot.data.joint_pos.torch[:, joint_ids]
                target = position.new_tensor(target_values).expand_as(position)
                command = position + (target - position).clamp(-0.012, 0.012)
                action_offset = 0 if side_index == 0 else 7
                action[:, action_offset : action_offset + 6] = command
                action[:, action_offset + 6] = _GRIPPER_OPEN
                errors.append(float(self.torch.abs(target - position).max()))
            _, _, terminated, truncated, _ = self.step(action)
            if bool((terminated | truncated).any().item()):
                raise RuntimeError("Environment ended unexpectedly while unfolding the cable arms.")
            maximum_error = max(errors)
            if step + 1 >= 70 and maximum_error <= 0.03:
                break
        self.hold_action.copy_(action)
        for side_index, side in enumerate(("left", "right")):
            robot = self.base_env.scene[f"{side}_robot"]
            body_id = self.gripper_body_ids[side_index]
            self.gripper_quaternions[side_index] = robot.data.body_link_quat_w.torch[:, body_id].clone()
        print(
            f"[cable-behaviour-demo] cycle {cycle}: unfolded with maximum joint error {maximum_error:.3f} rad",
            flush=True,
        )

    def _cartesian_arm_action(self, side_index: int, target_tool_position, joint_seed):
        """Return damped-IK joint targets for one YAM tool point."""
        import isaaclab.utils.math as math_utils

        side = ("left", "right")[side_index]
        robot = self.base_env.scene[f"{side}_robot"]
        body_id = self.gripper_body_ids[side_index]
        body_position = robot.data.body_link_pos_w.torch[:, body_id]
        body_quaternion = robot.data.body_link_quat_w.torch[:, body_id]
        desired_quaternion = self.gripper_quaternions[side_index].expand_as(body_quaternion)
        desired_quaternion = self.torch.where(
            ((body_quaternion * desired_quaternion).sum(dim=-1) < 0.0).unsqueeze(-1),
            -desired_quaternion,
            desired_quaternion,
        )
        tool_offset = body_position.new_tensor(_TOOL_OFFSET).expand(self.num_envs, -1)
        target_body_position = target_tool_position - math_utils.quat_apply(desired_quaternion, tool_offset)
        position_error, orientation_error = math_utils.compute_pose_error(
            body_position,
            body_quaternion,
            target_body_position,
            desired_quaternion,
        )

        jacobian = robot.data.body_link_jacobian_w.torch[:, body_id - 1, :, :6]
        orientation_weight = 0.30
        weighted_jacobian = self.torch.cat(
            (jacobian[:, :3], orientation_weight * jacobian[:, 3:6]),
            dim=1,
        )
        task_error = self.torch.cat((position_error, orientation_weight * orientation_error), dim=-1)
        jacobian_t = weighted_jacobian.transpose(-1, -2)
        identity = self.torch.eye(6, device=body_position.device, dtype=body_position.dtype).unsqueeze(0)
        pseudo_inverse = jacobian_t @ self.torch.linalg.solve(
            weighted_jacobian @ jacobian_t + 0.025**2 * identity,
            identity,
        )

        joint_ids = self.arm_joint_ids[side_index]
        joint_position = robot.data.joint_pos.torch[:, joint_ids]
        task_delta = (pseudo_inverse @ task_error.unsqueeze(-1)).squeeze(-1)
        null_space = identity - pseudo_inverse @ weighted_jacobian
        posture_delta = (null_space @ (joint_seed - joint_position).unsqueeze(-1)).squeeze(-1)
        target = joint_position + (0.65 * task_delta + 0.08 * posture_delta).clamp(
            -_MAX_ARM_JOINT_STEP,
            _MAX_ARM_JOINT_STEP,
        )
        return target, float(self.torch.linalg.vector_norm(position_error, dim=-1).max())

    def _run_cartesian_phase(
        self,
        label: str,
        targets,
        *,
        gripper_command: float,
        minimum_steps: int,
        maximum_steps: int,
        tolerance_m: float | None,
    ) -> None:
        """Move both YAM tool points toward a pair of world-space targets."""
        print(f"[cable-behaviour-demo] {label}", flush=True)
        robots = tuple(self.base_env.scene[f"{side}_robot"] for side in ("left", "right"))
        seeds = tuple(
            robot.data.joint_pos.torch[:, joint_ids].clone()
            for robot, joint_ids in zip(robots, self.arm_joint_ids, strict=True)
        )
        for step in range(maximum_steps):
            action = self.torch.zeros_like(self.hold_action)
            errors = []
            for side_index, (target, seed) in enumerate(zip(targets, seeds, strict=True)):
                arm_target, error = self._cartesian_arm_action(side_index, target, seed)
                action_offset = 0 if side_index == 0 else 7
                action[:, action_offset : action_offset + 6] = arm_target
                action[:, action_offset + 6] = gripper_command
                errors.append(error)
            _, _, terminated, truncated, _ = self.step(action)
            if bool((terminated | truncated).any().item()):
                raise RuntimeError(f"Environment ended unexpectedly during {label}.")
            minimum_reached = step + 1 >= minimum_steps
            target_reached = tolerance_m is None or max(errors) <= tolerance_m
            if minimum_reached and target_reached:
                self.hold_action.copy_(action)
                print(
                    f"[cable-behaviour-demo] {label}: maximum tool error {max(errors):.3f} m",
                    flush=True,
                )
                return
        self.hold_action.copy_(action)
        print(
            f"[cable-behaviour-demo] {label}: stopped with maximum tool error {max(errors):.3f} m",
            flush=True,
        )

    def _push_cable(self, cycle: int) -> None:
        """Move both arms onto the live cable and physically nudge it."""
        self._run_joint_approach(cycle)
        high_targets = self._tool_targets(_HIGH_HOVER_CLEARANCE_M)
        self._run_cartesian_phase(
            f"cycle {cycle}: approach two cable sections",
            high_targets,
            gripper_command=_GRIPPER_OPEN,
            minimum_steps=60,
            maximum_steps=180,
            tolerance_m=0.012,
        )
        contact_targets = self._tool_targets(_CONTACT_CLEARANCE_M)
        self._run_cartesian_phase(
            f"cycle {cycle}: lower the grippers onto the cable",
            contact_targets,
            gripper_command=_GRIPPER_CONTACT,
            minimum_steps=60,
            maximum_steps=180,
            tolerance_m=0.008,
        )
        before_push = self.cable.data.segment_pose_w.torch[..., :3].clone()
        push_targets = tuple(target + target.new_tensor((_PUSH_DISTANCE_M, 0.0, 0.0)) for target in contact_targets)
        self._run_cartesian_phase(
            f"cycle {cycle}: physically push the cable",
            push_targets,
            gripper_command=_GRIPPER_CONTACT,
            minimum_steps=45,
            maximum_steps=140,
            tolerance_m=0.010,
        )
        after_push = self.cable.data.segment_pose_w.torch[..., :3]
        displacement = self.torch.linalg.vector_norm(after_push - before_push, dim=-1).max()
        print(f"[cable-behaviour-demo] cable displacement during push: {float(displacement):.3f} m", flush=True)
        retreat_targets = tuple(target + target.new_tensor((0.0, 0.0, _RETREAT_CLEARANCE_M)) for target in push_targets)
        self._run_cartesian_phase(
            f"cycle {cycle}: open and retreat from the cable",
            retreat_targets,
            gripper_command=_GRIPPER_OPEN,
            minimum_steps=60,
            maximum_steps=160,
            tolerance_m=0.012,
        )

    def _goal_vertices(self):
        """Return an exact-rest-length route dwelling in every scored region."""
        from isaaclab_arena_environments.isaac_cap.cable_routing_v2.geometry import capsule_centerline

        default_pose = self.cable.data.default_segment_pose_w.torch[0]
        default_points = capsule_centerline(default_pose[None], 0.5 * self.segment_lengths[None])[0]
        z = default_points[0, 2]
        controls = [default_points[0]]

        # Traverse each world-axis-aligned seat box along its long axis. The
        # dwell span guarantees the native point-fraction thresholds see at
        # least three centerline samples in every guide region.
        for peg_name, region in zip(self.peg_names, self.goal.seat_regions, strict=True):
            peg_position = self.base_env.scene[peg_name].data.root_pos_w.torch[0]
            x_center = 0.5 * (region[0] + region[2])
            target = self.torch.stack((peg_position[0] + x_center, peg_position[1], z))
            controls.extend((
                target + target.new_tensor((0.0, -_REGION_DWELL_HALF_LENGTH, 0.0)),
                target + target.new_tensor((0.0, _REGION_DWELL_HALF_LENGTH, 0.0)),
            ))

        port_position = self.port.data.root_pos_w.torch[0]
        port_end = self.torch.stack((port_position[0], port_position[1], z))
        controls_tensor = self.torch.stack(controls)
        used_length = _polyline_length(controls_tensor)
        remaining_length = self.segment_lengths.sum() - used_length
        direct_length = self.torch.linalg.vector_norm(port_end - controls_tensor[-1])
        assert float(remaining_length) > float(direct_length), (
            "Current cable is too short for the scripted terminal route: "
            f"remaining={float(remaining_length):.4f}, direct={float(direct_length):.4f}."
        )

        # Put all extra length into one planar ellipse detour. The two legs
        # have total length `remaining_length`, so the sampled endpoint lands
        # exactly at the port while every cable segment remains at rest length.
        chord = port_end - controls_tensor[-1]
        chord_xy = chord[:2]
        chord_length = self.torch.linalg.vector_norm(chord_xy)
        perpendicular = self.torch.stack((-chord_xy[1], chord_xy[0])) / chord_length
        detour_height = 0.5 * self.torch.sqrt(remaining_length**2 - chord_length**2)
        detour = 0.5 * (controls_tensor[-1] + port_end)
        detour[:2] += detour_height * perpendicular
        route = self.torch.cat((controls_tensor, detour[None], port_end[None]), dim=0)

        distances = self.torch.cat((self.segment_lengths.new_zeros(1), self.segment_lengths.cumsum(dim=0)))
        vertices = _sample_polyline(route, distances)
        vertices[-1] = port_end
        return vertices

    def _write_success_route(self) -> None:
        """Write the connected successful route through the Arena Cable API."""
        vertices = self._goal_vertices()
        segment_pose = _vertices_to_segment_poses(vertices)[None]
        self.cable.write_segment_pose_to_sim_index(segment_pose=segment_pose, env_ids=self.env_ids)
        self.cable.write_segment_velocity_to_sim_index(
            segment_velocity=self.torch.zeros(
                (self.num_envs, self.cable.num_segments, 6),
                device=self.base_env.device,
                dtype=segment_pose.dtype,
            ),
            env_ids=self.env_ids,
        )

    def _step_hold(self):
        """Advance one frame and return separate success and timeout masks."""
        _, _, terminated, truncated, _ = self.step(self.hold_action)
        return terminated, truncated

    def run_cycle(self, cycle: int) -> None:
        """Show the live cable, author a completed route, and verify reset."""
        print(f"[cable-behaviour-demo] cycle {cycle}: running current Arena cable", flush=True)
        for _ in range(self.pause_steps):
            terminated, truncated = self._step_hold()
            if bool((terminated | truncated).any().item()):
                raise RuntimeError("Environment ended unexpectedly before the route demonstration.")

        self._push_cable(cycle)
        print(f"[cable-behaviour-demo] cycle {cycle}: route cable through all guides and the port", flush=True)
        self._write_success_route()
        # Current CAP evaluates the final state only after its graph policy has
        # finished. Exercise that signal through the demo-local compatibility
        # buffer installed in setup_demo().
        self.base_env.external_policy_termination_buf.fill_(True)
        success = self.success_term(self.base_env, **self.success_params)
        if not bool(success.all().item()):
            raise RuntimeError("Scripted cable route did not satisfy the current CAP terminal goal.")
        self.base_env.external_policy_termination_buf.fill_(False)

        for _ in range(self.pause_steps):
            self.render()

        self._write_success_route()
        self.base_env.external_policy_termination_buf.fill_(True)
        terminated, truncated = self._step_hold()
        self.base_env.external_policy_termination_buf.fill_(False)
        if not bool(terminated.all().item()):
            raise RuntimeError(
                "Successful cable route did not trigger task success; "
                f"terminated={terminated.tolist()}, truncated={truncated.tolist()}."
            )
        print(f"[cable-behaviour-demo] cycle {cycle}: success reset observed", flush=True)

        for _ in range(self.pause_steps):
            terminated, truncated = self._step_hold()
            if bool((terminated | truncated).any().item()):
                raise RuntimeError("Environment ended unexpectedly while displaying the reset state.")


def run_demo(
    simulation_app,
    *,
    variant: str = "medium",
    cycles: int = 0,
    pause_steps: int = 30,
    real_time: bool = True,
) -> None:
    """Compose the current cable task and run its behavior validation."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = CurrentCableRoutingBehaviourDemo(
        simulation_app,
        _build_cable_demo_environment(variant),
        ArenaEnvBuilderCfg(num_envs=_NUM_ENVS, env_spacing=1.5, solve_relations=False),
        pause_steps=pause_steps,
        real_time=real_time,
        visualizer_cfg=KitVisualizerCfg(
            eye=(1.25, -1.1, 1.55),
            lookat=(0.0125, 0.0, 0.77335),
            origin_type="world",
        ),
    )
    demo.run_demo(cycles)


def main() -> None:
    """Launch the visual v2 cable behavior demo."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("easy", "medium"),
        default="medium",
        help="Cable-routing variant to display and validate.",
    )
    parser.add_argument("--cycles", type=int, default=0, help="Cycles to run; zero repeats until Kit closes.")
    parser.add_argument("--pause-steps", type=int, default=30, help="Rendered frames shown around scripted states.")
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
            pause_steps=args.pause_steps,
            real_time=not args.no_real_time,
        )


if __name__ == "__main__":
    main()


__all__ = ["CurrentCableRoutingBehaviourDemo", "run_demo"]
