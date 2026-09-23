# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate easy tool sorting with an optional battery IK pick and scripted slot drops."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_arena_environments.isaac_cap.tools.env_behaviour_demo import DifferentialIKEnvBehaviourDemo

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
_PREGRASP_DISTANCE_M = 0.08
_LIFT_DISTANCE_M = 0.12
_MAX_TRANSLATION_PER_STEP_M = 0.008
_PREGRASP_TOLERANCE_M = 0.01
# Descent ends on first contact; random layouts can stop the Robotiq base about 16 mm above its
# nominal target while the fingers contact the battery or source bin.
_GRASP_DESCENT_TOLERANCE_M = 0.02
_GRASP_Z_OFFSET_M = _ROBOTIQ_BASE_TO_GRASP_M + _GRASP_HEIGHT_OFFSET_M
_PREGRASP_Z_OFFSET_M = _GRASP_Z_OFFSET_M + _PREGRASP_DISTANCE_M
_MOVE_TO_MAX_STEPS = 720
_DROP_HEIGHT_ABOVE_SLOT_M = 0.05
_NUM_ENVS = 2
_EASY_LEVELS = ("1", "2", "3")
_PICK_TARGET_BY_LEVEL = {
    "1": "battery_0",
    "2": "slotted_screwdriver_0",
    "3": "wire_spool_0",
}


def _build_tool_sort_demo_environment(level: str):
    """Compose one easy tool-sort graph with relative IK for the battery pick."""
    assert level in _EASY_LEVELS, f"Unsupported easy level {level!r}."

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena_environments.isaac_cap import register_components
    from isaaclab_arena_environments.isaac_cap.tool_sorting.embodiment import (
        ToolSortingFr3Robotiq2f85DifferentialIKEmbodiment,
    )

    register_components()
    spec_path = Path(__file__).with_name(f"tool_sorting_easy_{level}.yaml")
    arena_environment = ArenaEnvGraphSpec.from_yaml(str(spec_path)).to_arena_env(enable_cameras=False)

    source_embodiment = arena_environment.embodiment
    arena_environment.embodiment = ToolSortingFr3Robotiq2f85DifferentialIKEmbodiment(
        initial_pose=source_embodiment.get_initial_pose(),
        initial_joint_pose=list(_DEMO_START_JOINT_POS),
    )
    return arena_environment


class ToolSortingEnvBehaviourDemo(DifferentialIKEnvBehaviourDemo):
    """Optionally pick the battery with IK, then drop every tool into its compartment."""

    label = "tool-sort-validation"
    max_translation_per_step_m = _MAX_TRANSLATION_PER_STEP_M
    move_to_max_steps = _MOVE_TO_MAX_STEPS
    position_tolerance_m = _PREGRASP_TOLERANCE_M

    def __init__(
        self,
        simulation_app,
        arena_environment,
        builder_cfg,
        *,
        pick_target_object_name: str,
        teleport_only: bool,
        pause_steps: int,
        ik_log_interval: int,
        real_time: bool = True,
        visualizer_cfg=None,
    ) -> None:
        """Configure the tool-sort validation demo.

        Args:
            simulation_app: Active Arena simulation application context.
            arena_environment: Composed Arena environment to instantiate.
            builder_cfg: Configuration for building the stepable environment.
            pick_target_object_name: Object picked with IK before scripted slot drops.
            teleport_only: Whether to skip the physical pick and validate only scripted teleports.
            pause_steps: Number of steps to display each validation state.
            ik_log_interval: IK diagnostic logging interval in control steps; zero disables logging.
            real_time: Whether to pace environment steps in real time.
            visualizer_cfg: Optional default simulator visualizer configuration.
        """
        assert pause_steps >= 1, "pause_steps must be positive."
        assert ik_log_interval >= 0, "ik_log_interval must be non-negative."
        super().__init__(
            simulation_app,
            arena_environment,
            builder_cfg,
            real_time=real_time,
            visualizer_cfg=visualizer_cfg,
        )
        self.pick_target_object_name = pick_target_object_name
        self.teleport_only = teleport_only
        self.pause_steps = pause_steps
        self.ik_log_interval = ik_log_interval

    def setup_demo(self) -> None:
        """Resolve IK controls, task objects, and compartment bounds."""
        from isaaclab_arena_environments.isaac_cap.tool_sorting.task import objects_in_regions

        self.setup_differential_ik(_NUM_ENVS)
        self._objects_in_regions = objects_in_regions

        self.gripper_action = self.base_env.action_manager.get_term("gripper_action")
        gripper_joint_ids, _ = self.robot.find_joints("left_driver_joint")
        assert len(gripper_joint_ids) == 1, f"Expected one left_driver_joint, got {gripper_joint_ids}."
        self.gripper_joint_id = int(gripper_joint_ids[0])

        task = self.arena_environment.task
        self.object_names = tuple(object_.name for object_ in task.objects)
        self.region_name = task.regions[0].name
        assert all(
            region.name == self.region_name for region in task.regions
        ), "The easy tool-sort demo expects every object to target the same destination bin."
        self.bounds = task.bounds
        assert (
            self.pick_target_object_name in self.object_names
        ), f"Pick target {self.pick_target_object_name!r} is not one of the task objects: {self.object_names}."

    def _zero_action(self):
        return self.torch.zeros(
            (self.num_envs, self.base_env.action_manager.total_action_dim),
            device=self.base_env.device,
        )

    def _hold_with_action(self, steps: int, action) -> bool:
        return any(bool(self._step(action).any().item()) for _ in range(steps))

    def _hold_zero(self, steps: int) -> bool:
        return self._hold_with_action(steps, self._zero_action())

    def _target_above_object(self, object_name: str, z_offset_m: float):
        object_position = self.base_env.scene[object_name].data.root_link_pos_w.torch.clone()
        object_position[:, 2] += z_offset_m
        return object_position

    def _log_ik_state(self, label: str, step: int, target_position_w) -> None:
        """Log Cartesian IK state, Jacobian conditioning, and applied joint targets."""
        if self.ik_log_interval == 0 or step % self.ik_log_interval != 0:
            return

        ee_position_w = self._ee_position()
        error_w = target_position_w - ee_position_w
        jacobian = self.arm_action.jacobian_b
        singular_values = self.torch.linalg.svdvals(jacobian)
        joint_ids = self.arm_action._joint_ids
        joint_pos = self.robot.data.joint_pos.torch[:, joint_ids]
        joint_target = self.robot.data.joint_pos_target.torch[:, joint_ids]
        gripper_slice = slice(self.gripper_joint_id, self.gripper_joint_id + 1)
        gripper_pos = self.robot.data.joint_pos.torch[:, gripper_slice]
        gripper_target = self.robot.data.joint_pos_target.torch[:, gripper_slice]
        controller = self.arm_action._ik_controller
        for env_id in range(self.num_envs):
            print(
                f"[{self.label}][ik] phase={label} step={step} env={env_id} "
                f"target_w={target_position_w[env_id].tolist()} "
                f"ee_w={ee_position_w[env_id].tolist()} "
                f"error_w={error_w[env_id].tolist()} error_norm={float(error_w[env_id].norm()):.6f} "
                f"controller_target_b={controller.ee_pos_des[env_id].tolist()} "
                f"jacobian_s={singular_values[env_id].tolist()} "
                f"joint_pos={joint_pos[env_id].tolist()} "
                f"joint_target={joint_target[env_id].tolist()} "
                f"gripper_pos={gripper_pos[env_id].tolist()} "
                f"gripper_target={gripper_target[env_id].tolist()}",
                flush=True,
            )

    def _move_to_object(
        self,
        object_name: str,
        *,
        z_offset_m: float,
        gripper_closed: bool,
        label: str,
        position_tolerance_m: float,
    ) -> bool:
        """Track a tool root while driving the gripper toward a Z offset."""
        for step in range(_MOVE_TO_MAX_STEPS):
            target_position_w = self._target_above_object(object_name, z_offset_m)
            error_w = target_position_w - self._ee_position()
            errors_m = self.torch.linalg.vector_norm(error_w, dim=-1)
            self._log_ik_state(label, step, target_position_w)
            if bool((errors_m <= position_tolerance_m).all().item()):
                return self._hold_ik(10, gripper_closed=gripper_closed)
            if bool(self._step(self._ik_action(error_w, gripper_closed=gripper_closed)).any().item()):
                return True
        target_position_w = self._target_above_object(object_name, z_offset_m)
        errors_m = self.torch.linalg.vector_norm(target_position_w - self._ee_position(), dim=-1)
        if bool((errors_m <= position_tolerance_m).all().item()):
            return self._hold_ik(10, gripper_closed=gripper_closed)
        raise RuntimeError(f"Timed out during {label}; maximum end-effector position error is {errors_m.max():.3f} m.")

    def _teleport(self, asset_name: str, pose_w) -> None:
        asset = self.base_env.scene[asset_name]
        env_ids = self.torch.arange(self.num_envs, device=self.base_env.device, dtype=self.torch.int32)
        asset.write_root_pose_to_sim_index(root_pose=pose_w, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=self.torch.zeros((self.num_envs, 6), device=self.base_env.device),
            env_ids=env_ids,
        )

    def _compartment_floor_position_w(self, tool_index: int):
        """Return the world position of one compartment floor center."""
        import isaaclab.utils.math as math_utils

        bounds = self.bounds[tool_index]
        xmin, ymin, zmin, xmax, ymax, _zmax = bounds
        center_local = self.torch.tensor(
            [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, zmin],
            device=self.base_env.device,
            dtype=self.base_env.scene[self.region_name].data.root_pos_w.torch.dtype,
        ).expand(self.num_envs, -1)
        region = self.base_env.scene[self.region_name]
        region_pos_w = region.data.root_pos_w.torch
        region_quat_w = region.data.root_quat_w.torch
        return region_pos_w + math_utils.quat_apply(region_quat_w, center_local)

    def _hover_pose_w(self, tool_index: int):
        """Return a hover pose ten centimeters above one destination compartment."""
        import isaaclab.utils.math as math_utils

        floor_position_w = self._compartment_floor_position_w(tool_index)
        region = self.base_env.scene[self.region_name]
        region_quat_w = region.data.root_quat_w.torch
        local_up = self.torch.tensor([0.0, 0.0, 1.0], device=self.base_env.device, dtype=region_quat_w.dtype)
        local_up = local_up.expand(self.num_envs, -1)
        region_up_w = math_utils.quat_apply(region_quat_w, local_up)
        hover_position_w = floor_position_w + region_up_w * _DROP_HEIGHT_ABOVE_SLOT_M
        object_asset = self.base_env.scene[self.object_names[tool_index]]
        object_quat_w = object_asset.data.root_quat_w.torch
        return self.torch.cat((hover_position_w, object_quat_w), dim=-1)

    def _success_mask(self):
        return self._objects_in_regions(
            self.base_env,
            list(self.object_names),
            [self.region_name] * len(self.object_names),
            list(self.bounds),
        )

    def _report_success_reset(self, cycle: int) -> None:
        print(
            f"[{self.label}] cycle {cycle}: success reset observed in all {self.num_envs} environments",
            flush=True,
        )

    def _run_ik_pick(self, cycle: int) -> None:
        """Approach the configured target with IK, grasp it, lift, and release in the source bin."""
        grasp_name = self.pick_target_object_name
        grasp_object = self.base_env.scene[grasp_name]

        print(f"[{self.label}] cycle {cycle}: IK approach and grasp {grasp_name}", flush=True)
        if self._move_to_object(
            grasp_name,
            z_offset_m=_PREGRASP_Z_OFFSET_M,
            gripper_closed=False,
            label="pregrasp",
            position_tolerance_m=_PREGRASP_TOLERANCE_M,
        ):
            raise RuntimeError("Environment ended unexpectedly during pregrasp.")
        if self._move_to_object(
            grasp_name,
            z_offset_m=_GRASP_Z_OFFSET_M,
            gripper_closed=False,
            label="descent",
            position_tolerance_m=_GRASP_DESCENT_TOLERANCE_M,
        ):
            raise RuntimeError("Environment ended unexpectedly during descent.")

        close_steps = max(60, round(2.5 / self.base_env.step_dt))
        print(f"[{self.label}] cycle {cycle}: closing gripper on {grasp_name}", flush=True)
        if self._hold_ik(close_steps, gripper_closed=True):
            raise RuntimeError("Environment ended unexpectedly while closing the gripper.")

        object_z_before_lift = grasp_object.data.root_link_pos_w.torch[:, 2].clone()
        lift_position = self._ee_position().clone()
        lift_position[:, 2] += _LIFT_DISTANCE_M
        print(f"[{self.label}] cycle {cycle}: lift, open, and drop {grasp_name}", flush=True)
        if self._move_to(
            lift_position,
            gripper_closed=True,
            label="lift",
            position_tolerance_m=_PREGRASP_TOLERANCE_M,
        ):
            raise RuntimeError("Environment ended unexpectedly during lift.")
        if self._hold_ik(self.pause_steps, gripper_closed=True):
            raise RuntimeError("Environment ended unexpectedly while displaying the lift.")

        object_z_after_lift = grasp_object.data.root_link_pos_w.torch[:, 2]
        lift_displacements = object_z_after_lift - object_z_before_lift
        lift_summary = ", ".join(
            f"env_{env_id}={float(displacement):.3f} m" for env_id, displacement in enumerate(lift_displacements)
        )
        print(f"[{self.label}] physical lift displacement: {lift_summary}", flush=True)
        if self._hold_ik(max(30, round(0.75 / self.base_env.step_dt)), gripper_closed=False):
            raise RuntimeError("Environment ended unexpectedly while dropping the tool.")
        retreat_position = self._ee_position().clone()
        retreat_position[:, 2] += _PREGRASP_DISTANCE_M
        if self._move_to(
            retreat_position,
            gripper_closed=False,
            label="retreat",
            position_tolerance_m=_PREGRASP_TOLERANCE_M,
        ):
            raise RuntimeError("Environment ended unexpectedly during retreat.")

    def _teleport_tool_above_slot(self, cycle: int, tool_index: int, *, is_last: bool):
        """Place one tool above its compartment and return resets observed while settling."""
        tool_name = self.object_names[tool_index]
        print(
            f"[{self.label}] cycle {cycle}: drop {tool_name} from "
            f"{_DROP_HEIGHT_ABOVE_SLOT_M:.2f} m above slot ({tool_index + 1}/{len(self.object_names)})",
            flush=True,
        )
        self._teleport(tool_name, self._hover_pose_w(tool_index))
        settle_steps = max(1, round(0.25 / self.base_env.step_dt))
        reset_observed = self.torch.zeros(self.num_envs, device=self.base_env.device, dtype=self.torch.bool)
        zero_action = self._zero_action()
        for _ in range(settle_steps):
            reset_observed |= self._step(zero_action)
        if not is_last and bool(reset_observed.any().item()):
            raise RuntimeError(f"Environment ended unexpectedly while settling {tool_name}.")
        return reset_observed

    def run_cycle(self, cycle: int) -> None:
        """Optionally pick the target, teleport every tool into its slot, and require success reset."""
        print(f"[{self.label}] cycle {cycle}: settling", flush=True)
        if self._hold_zero(10):
            raise RuntimeError("Environment ended unexpectedly while settling.")

        if not self.teleport_only:
            self._run_ik_pick(cycle)
            if self._hold_zero(self.pause_steps):
                raise RuntimeError("Environment ended unexpectedly after the IK pick.")

        last_index = len(self.object_names) - 1
        for tool_index in range(len(self.object_names)):
            is_last = tool_index == last_index
            reset_observed = self._teleport_tool_above_slot(cycle, tool_index, is_last=is_last)

            if not is_last:
                continue

            if bool(reset_observed.all().item()):
                self._report_success_reset(cycle)
                return
            wait_steps = max(self.pause_steps, round(2.0 / self.base_env.step_dt))
            zero_action = self._zero_action()
            for _ in range(wait_steps):
                reset_observed |= self._step(zero_action)
                if bool(reset_observed.all().item()):
                    self._report_success_reset(cycle)
                    return

        success_mask = self._success_mask()
        raise RuntimeError(
            f"Final placement did not trigger the environment reset; objects_in_regions={success_mask.tolist()}"
        )


def run_demo(
    simulation_app,
    *,
    level: str = "1",
    cycles: int = 0,
    teleport_only: bool = False,
    pause_steps: int = 30,
    ik_log_interval: int = 0,
    real_time: bool = True,
) -> None:
    """Compose one easy tool-sort task and run its validation lifecycle."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = ToolSortingEnvBehaviourDemo(
        simulation_app,
        _build_tool_sort_demo_environment(level),
        ArenaEnvBuilderCfg(num_envs=_NUM_ENVS, env_spacing=1.5, solve_relations=True),
        pick_target_object_name=_PICK_TARGET_BY_LEVEL[level],
        teleport_only=teleport_only,
        real_time=real_time,
        visualizer_cfg=KitVisualizerCfg(
            eye=(2.0, -2.0, 2.0),
            lookat=(0.1, 0.0, 0.85),
            origin_type="world",
        ),
        pause_steps=pause_steps,
        ik_log_interval=ik_log_interval,
    )
    demo.run_demo(cycles)


def main() -> None:
    """Launch the visual validation demo."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", nargs="?", choices=_EASY_LEVELS, default="1")
    parser.add_argument("--cycles", type=int, default=0, help="Cycles to run; zero repeats until Kit closes.")
    parser.add_argument(
        "--teleport-only",
        action="store_true",
        help="Skip the physical pick/lift/drop phase and validate scripted object teleports only.",
    )
    parser.add_argument("--pause-steps", type=int, default=30, help="Frames shown between scripted phases.")
    parser.add_argument(
        "--ik-log-interval",
        type=int,
        default=0,
        help="Log IK state every N control steps; zero disables logging.",
    )
    parser.add_argument("--no-real-time", action="store_true", help="Run without wall-clock rate limiting.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    args.limit_cpu_threads = 1

    with SimulationAppContext(args) as simulation_app:
        run_demo(
            simulation_app,
            level=args.level,
            cycles=args.cycles,
            teleport_only=args.teleport_only,
            real_time=not args.no_real_time,
            pause_steps=args.pause_steps,
            ik_log_interval=args.ik_log_interval,
        )


if __name__ == "__main__":
    main()
