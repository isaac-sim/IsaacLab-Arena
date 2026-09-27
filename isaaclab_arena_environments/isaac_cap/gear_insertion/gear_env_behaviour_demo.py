# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab_arena_environments.isaac_cap.tools.env_behaviour_demo import DifferentialIKEnvBehaviourDemo

# This configuration starts the Robotiq fingers above the work surface with the
# gripper pointing down. It keeps the deliberately rough scripted motion short.
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
_MAX_TRANSLATION_PER_STEP_M = 0.006
_POSITION_TOLERANCE_M = 0.008
_NUM_ENVS = 2


def _build_gear_demo_environment(variant: str):
    """Compose a gear task with the relative-IK embodiment used by this demo."""
    assert variant in ("easy", "medium"), f"Unsupported gear variant {variant!r}."

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena_environments.isaac_cap import register_components
    from isaaclab_arena_environments.isaac_cap.embodiments.insertion_task import (
        IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
    )

    register_components()
    spec_path = Path(__file__).with_name(f"gear_{variant}.yaml")
    arena_environment = ArenaEnvGraphSpec.from_yaml(spec_path).to_arena_env(enable_cameras=False)

    # The normal smoke environment intentionally retains Cap's absolute joint
    # actions. This demo alone swaps to relative Cartesian commands so the
    # approach, grasp, and lift can be expressed as a few rough waypoints.
    initial_pose = arena_environment.embodiment.get_initial_pose()
    arena_environment.embodiment = IndustrialFr3Robotiq2f85DifferentialIKEmbodiment(
        initial_pose=initial_pose,
        initial_joint_pose=list(_DEMO_START_JOINT_POS),
    )
    return arena_environment


class GearEnvBehaviourDemo(DifferentialIKEnvBehaviourDemo):
    """Implement the gear-specific validation setup and motion sequence."""

    label = "gear-validation"
    max_translation_per_step_m = _MAX_TRANSLATION_PER_STEP_M
    move_to_max_steps = 240
    position_tolerance_m = _POSITION_TOLERANCE_M

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
        """Configure the gear-specific behavior.

        Args:
            simulation_app: Active Arena simulation application context.
            arena_environment: Composed Arena environment to instantiate.
            builder_cfg: Configuration for building the stepable environment.
            real_time: Whether to pace environment steps in real time.
            pause_steps: Number of steps to display each validation state.
            visualizer_cfg: Optional default simulator visualizer configuration.
        """
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
        """Resolve gear-specific action terms, bodies, and success targets."""
        self.setup_differential_ik(_NUM_ENVS)

        progress_tracker = self.base_env.progress_tracker
        assert progress_tracker is not None, "Gear insertion diagnostics require task success tracking."
        self.success_predicate = progress_tracker.get_predicate("gear_insertion")
        task = self.arena_environment.task
        self.plate_name = task.plate.name
        self.gear_names = tuple(gear.name for gear in task.gears)
        self.target_offsets_xyz = task.target_offsets_xyz

    def _teleport(self, asset_name: str, pose_w) -> None:
        asset = self.base_env.scene[asset_name]
        env_ids = self.torch.arange(self.num_envs, device=self.base_env.device, dtype=self.torch.int32)
        asset.write_root_pose_to_sim_index(root_pose=pose_w, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=self.torch.zeros((self.num_envs, 6), device=self.base_env.device),
            env_ids=env_ids,
        )

    def _put_first_gear_under_gripper(self) -> None:
        """Move the first gear beneath the pre-positioned downward gripper."""
        gear = self.base_env.scene[self.gear_names[0]]
        pose_w = gear.data.root_link_pose_w.torch.clone()
        pose_w[:, :2] = self._ee_position()[:, :2]
        self._teleport(self.gear_names[0], pose_w)

    def _successful_pose(self, gear_index: int):
        """Return the exact task target pose for one gear."""
        import isaaclab.utils.math as math_utils

        plate = self.base_env.scene[self.plate_name]
        plate_pos_w = plate.data.root_link_pos_w.torch
        plate_quat_w = plate.data.root_link_quat_w.torch
        offset = self.torch.tensor(
            [self.target_offsets_xyz[gear_index]],
            device=self.base_env.device,
            dtype=plate_pos_w.dtype,
        ).expand(self.num_envs, -1)
        target_pos_w = plate_pos_w + math_utils.quat_apply(plate_quat_w, offset)
        return self.torch.cat((target_pos_w, plate_quat_w), dim=-1)

    def run_cycle(self, cycle: int) -> None:
        """Grasp/drop one gear, then place all gears and require a success reset."""
        print(f"[gear-validation] cycle {cycle}: settling", flush=True)
        if self._hold_ik(30, gripper_closed=False):
            raise RuntimeError("Environment ended unexpectedly while settling.")

        self._put_first_gear_under_gripper()
        if self._hold_ik(20, gripper_closed=False):
            raise RuntimeError("Environment ended unexpectedly while positioning the grasp gear.")

        grasp_gear = self.base_env.scene[self.gear_names[0]]
        grasp_position = grasp_gear.data.root_link_pos_w.torch.clone()
        grasp_position[:, 2] += _ROBOTIQ_BASE_TO_GRASP_M + _GRASP_HEIGHT_OFFSET_M
        pregrasp_position = grasp_position.clone()
        pregrasp_position[:, 2] += _PREGRASP_DISTANCE_M

        print(f"[gear-validation] cycle {cycle}: approach and grasp {self.gear_names[0]}", flush=True)
        if self._move_to(pregrasp_position, gripper_closed=False, label="pregrasp"):
            raise RuntimeError("Environment ended unexpectedly during pregrasp.")
        if self._move_to(grasp_position, gripper_closed=False, label="descent"):
            raise RuntimeError("Environment ended unexpectedly during descent.")

        close_steps = max(30, round(1.5 / self.base_env.step_dt))
        if self._hold_ik(close_steps, gripper_closed=True):
            raise RuntimeError("Environment ended unexpectedly while closing the gripper.")

        gear_z_before_lift = grasp_gear.data.root_link_pos_w.torch[:, 2].clone()
        lift_position = grasp_position.clone()
        lift_position[:, 2] += _LIFT_DISTANCE_M
        print(f"[gear-validation] cycle {cycle}: lift, open, and drop", flush=True)
        if self._move_to(lift_position, gripper_closed=True, label="lift"):
            raise RuntimeError("Environment ended unexpectedly during lift.")
        if self._hold_ik(self.pause_steps, gripper_closed=True):
            raise RuntimeError("Environment ended unexpectedly while displaying the lift.")

        gear_z_after_lift = grasp_gear.data.root_link_pos_w.torch[:, 2]
        lift_displacements = gear_z_after_lift - gear_z_before_lift
        lift_summary = ", ".join(
            f"env_{env_id}={float(displacement):.3f} m" for env_id, displacement in enumerate(lift_displacements)
        )
        print(
            f"[gear-validation] physical lift displacement: {lift_summary}",
            flush=True,
        )
        if self._hold_ik(max(30, round(0.75 / self.base_env.step_dt)), gripper_closed=False):
            raise RuntimeError("Environment ended unexpectedly while dropping the gear.")
        if self._move_to(pregrasp_position, gripper_closed=False, label="retreat"):
            raise RuntimeError("Environment ended unexpectedly during retreat.")

        for gear_index, gear_name in enumerate(self.gear_names):
            print(
                f"[gear-validation] cycle {cycle}: teleport {gear_name} to target "
                f"({gear_index + 1}/{len(self.gear_names)})",
                flush=True,
            )
            self._teleport(gear_name, self._successful_pose(gear_index))
            if gear_index < len(self.gear_names) - 1:
                if self._hold_ik(self.pause_steps, gripper_closed=False):
                    raise RuntimeError("Environment reported success before every gear was placed.")
                continue

            # The task requires ten consecutive successful frames. The normal
            # environment reset must fire; a manual reset would hide a broken
            # success predicate or reset path.
            reset_observed = self.torch.zeros(self.num_envs, device=self.base_env.device, dtype=self.torch.bool)
            for _ in range(max(self.pause_steps, 60)):
                reset_observed |= self._step(self._ik_action(gripper_closed=False))
                if bool(reset_observed.all().item()):
                    print(
                        f"[gear-validation] cycle {cycle}: success reset observed in all {self.num_envs} environments",
                        flush=True,
                    )
                    return

        diagnostics = {
            gear_name: self.success_predicate.per_gear_results[gear_name].tolist() for gear_name in self.gear_names
        }
        raise RuntimeError(f"Final placement did not trigger the environment reset: {diagnostics}")


def run_demo(
    simulation_app,
    *,
    variant: str = "medium",
    cycles: int = 0,
    pause_steps: int = 30,
    real_time: bool = True,
) -> None:
    """Compose the gear task and run its behavior through the shared lifecycle."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = GearEnvBehaviourDemo(
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
    """Launch the visual validation demo."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=("easy", "medium"), default="medium")
    parser.add_argument("--cycles", type=int, default=0, help="Cycles to run; zero repeats until Kit closes.")
    parser.add_argument("--pause-steps", type=int, default=30, help="Frames shown between each teleported gear.")
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
