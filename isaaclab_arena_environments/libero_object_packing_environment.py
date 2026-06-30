# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Plain (no-stand) Franka panda so the robot is ground-mounted at the origin, matching the
# LIBERO floor scene (the default franka_ik USD is mounted on a table-height stand).
_PLAIN_PANDA = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
# LIBERO Franka home configuration: q1..q7 + the two finger joints (open).
_HOME_Q = [0.0, -0.16104, 0.0, -2.4446, 0.0, 2.22675, 0.7854, 0.04, 0.04]
# DROID (Franka+Robotiq) home: 7 arm joints (LIBERO posture) + finger_joint + 5 Robotiq linkage joints = 13 DOF.
_DROID_HOME_Q = [0.0, -0.16104, 0.0, -2.4446, 0.0, 2.22675, 0.7854, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Comfortable reach box in front of the base for the groceries (x,y; On() supplies z at the table top).
# Tightened so EVERY pooled placement is within the Panda's orientation-constrained top-down reach: the base
# is on the table at world x=-0.20, so x_max=0.42 / |y|<=0.22 keeps the approach pose ~<=0.76 m from the base
# (safely under the ~0.85 m practical reach). The far corner of the old box (0.52, 0.26) put approaches
# ~0.84-0.92 m out -> approach IK failed -> grocery_packing ended the episode with ~0 packed.
_REACH_BOX = dict(x_min=0.15, x_max=0.42, y_min=-0.22, y_max=0.22)
# Tabletop height. Robot base + basket + groceries all sit on the table at this z; raising everything by
# the same amount preserves the base->object geometry, so reachability/grasps are unchanged vs the floor.
_TABLE_TOP_Z = 0.40


@register_environment
class LiberoObjectPackingEnvironment(ExampleEnvironmentBase):
    """LIBERO grocery-packing scene on the floor, placed by the relation solver.

    A basket plus six HOPE groceries; positions are solved (On a thin invisible surface,
    bounded to the Franka reach box, jittered per reset) rather than hardcoded.
    """

    name = "libero_object_packing"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
        from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
        from isaaclab_arena.utils.pose import Pose

        def q(w, x, y, z):  # MuJoCo wxyz -> Arena xyzw
            return (x, y, z, w)

        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(1500.0)
        ground = self.asset_registry.get_asset_by_name("ground_plane")()

        # Ground-mounted robot at the origin with the LIBERO home pose. The control mode picks the
        # embodiment and arm action term: relative differential IK (default; a zero action holds pose,
        # which the viewer and smoke tests rely on), absolute joint position for the GaP bridge
        # (where the action IS the target q), or droid_joint_pos for the DROID Franka+Robotiq variant.
        # A zero action under joint-position control drives every joint to 0, so this is not a safe default.
        if args_cli.control == "droid_joint_pos":
            # DROID Franka+Robotiq: use the embodiment's own absolute-joint-pos action, USD, PD gains,
            # and stand. Do NOT apply the Panda-specific overrides (plain panda USD, HIGH_PD, panda_joint.*
            # action term, 9-dof _HOME_Q) — they would drop the Robotiq gripper and mismatch the 13-DOF dof.
            from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

            embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=args_cli.enable_cameras)
            # The DROID stand USD (Arena/srl_robolab_assets) is not hosted on the public S3 Nucleus and
            # is redundant with the libero table anyway; swap its spawn for a tiny invisible cuboid. The
            # stand AssetBaseCfg object is kept (the embodiment's _update_scene_cfg_with_robot_initial_pose
            # writes stand.init_state), but nothing is fetched and no prop overlaps the table.
            import isaaclab.sim as sim_utils

            embodiment.scene_config.stand.spawn = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01), visible=False)
            embodiment.set_initial_pose(
                Pose(position_xyz=(-0.20, 0.0, _TABLE_TOP_Z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )
            embodiment.set_initial_joint_pose(initial_joint_pose=_DROID_HOME_Q)
            if args_cli.enable_cameras:
                from isaaclab_arena_environments.libero_cameras import LiberoDroidPerceptionCameraCfg

                embodiment.camera_config = LiberoDroidPerceptionCameraCfg()
        else:
            embodiment_name = "franka_joint_pos" if args_cli.control == "joint_pos" else "franka_ik"
            embodiment = self.asset_registry.get_asset_by_name(embodiment_name)(enable_cameras=args_cli.enable_cameras)
            embodiment.set_initial_pose(
                Pose(position_xyz=(-0.20, 0.0, _TABLE_TOP_Z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )
            embodiment.set_initial_joint_pose(initial_joint_pose=_HOME_Q)
            embodiment.scene_config.robot.spawn.usd_path = _PLAIN_PANDA
            if args_cli.control == "joint_pos":
                from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
                from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

                # High-PD gains so the arm actually tracks the commanded absolute q. The stock joint-pos cfg
                # (FRANKA_PANDA_CFG, stiffness=80) is tuned for RL compliance and sags ~0.25 rad at the
                # gravity-loaded joints 2/4; HIGH_PD (the gains the IK embodiment uses) holds the target. The
                # plain ground-mounted panda USD is kept.
                high_pd_robot = FRANKA_PANDA_HIGH_PD_CFG.copy()
                high_pd_robot.spawn.usd_path = _PLAIN_PANDA
                embodiment.scene_config.robot = high_pd_robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

                # Absolute joint targets: action[i] is the commanded q_i directly. The stock franka_joint_pos
                # term ships scale=0.5/use_default_offset=True, which would land a target q at 0.5*q+default;
                # scale=1.0/use_default_offset=False makes the action the absolute target the bridge expects.
                embodiment.action_config.arm_action = JointPositionActionCfg(
                    asset_name="robot",
                    joint_names=["panda_joint.*"],
                    scale=1.0,
                    use_default_offset=False,
                )
            if args_cli.enable_cameras:
                from isaaclab_arena_environments.libero_cameras import LiberoPerceptionCameraCfg

                embodiment.camera_config = LiberoPerceptionCameraCfg()

        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)() if args_cli.teleop_device else None
        )

        # Visible static table: a thick kinematic cuboid (floor -> table top) that also serves as the
        # relation solver's On()/anchor surface. Center at z=H/2 so the top is at z=_TABLE_TOP_Z; robot,
        # basket, and groceries all rest on this top. The per-instance spawn override (the registry's
        # procedural_table is an invisible thin slab) makes it a real visible table without touching the
        # shared asset cfg.
        import isaaclab.sim as sim_utils

        surface = self.asset_registry.get_asset_by_name("procedural_table")()
        surface.set_initial_pose(Pose(position_xyz=(0.16, 0.0, _TABLE_TOP_Z / 2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        surface.bounding_box = AxisAlignedBoundingBox(
            min_point=(-0.6, -0.8, -_TABLE_TOP_Z / 2), max_point=(0.6, 0.8, _TABLE_TOP_Z / 2)
        )
        surface.object_cfg.spawn = sim_utils.CuboidCfg(
            size=(1.2, 1.6, _TABLE_TOP_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22)),
            visible=True,
        )
        surface.add_relation(IsAnchor())

        # Basket: a fixed anchor. IsAnchor keeps its pose un-optimized while still entering the
        # solver's no-overlap loss and validation as an obstacle, so groceries are placed clear of
        # it rather than spawning inter-penetrating and getting ejected by physics. The LIBERO quat
        # is a clean 90 deg yaw (opening up); the source data's ~0.0017 x/y noise is dropped so the
        # anchor bbox path (yaw-only, 1e-3 tolerance) accepts it.
        basket = self.asset_registry.get_asset_by_name(args_cli.basket)()
        basket.set_initial_pose(
            Pose(position_xyz=(0.28, 0.42, _TABLE_TOP_Z), rotation_xyzw=q(0.70710678, 0.0, 0.0, 0.70710678))
        )
        basket.add_relation(IsAnchor())

        # Groceries: relation-solved placement (On surface, within reach, jittered per reset).
        # The object set is data-driven via --objects (job_manager expands a JSON list into --objects a b ...),
        # so 2-3 object variations are configured from the job config, not hardcoded here.
        objects = []
        for obj_name in args_cli.objects:
            obj = self.asset_registry.get_asset_by_name(obj_name)()
            obj.add_relation(On(surface, edge_margin_m=0.03))
            obj.add_relation(PositionLimits(**_REACH_BOX))
            objects.append(obj)

        scene = Scene(assets=[ground, light, surface, basket, *objects])

        if args_cli.eval_task == "pick_place_in_basket":
            task = _make_libero_packing_task(
                object_asset_names=[obj.name for obj in objects],
                basket_asset_name=basket.name,
                episode_length_s=150.0 * len(objects) + 150.0,
            )
        elif args_cli.eval_task == "stock_pick_place":
            # Hybrid metric-pivot (STEP 2a): reuse the libero scene's proven exterior_cam + pose_mat, but score
            # with the STOCK PickAndPlaceTask / object_on_destination (contact-on-destination + low velocity)
            # instead of our custom resting_in_bin. Single object (objects[0]) -> basket. Fires success the moment
            # the object rests on the destination; no gripper_open coincidence, no multi-object perceive loop
            # (so no phantom-4th re-detection). Intended for GaP's single-cycle examples/pick_place graph.
            from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

            task = PickAndPlaceTask(
                pick_up_object=objects[0],
                destination_location=basket,
                background_scene=surface,
                episode_length_s=180.0,
            )
        elif args_cli.eval_task == "stock_sort":
            # Multi-object (N) pure-stock pick-place: the STOCK SortMultiObjectTask scores per-object
            # object_on_destination composed under SuccessMode.ALL (all objects into the grey_bin), building one
            # destination-filtered contact sensor per object. Authoritative metric kept identical to the single-object
            # stock_pick_place for VLA comparability. Data-driven N via --objects.
            from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask

            task = SortMultiObjectTask(
                pick_up_object_list=objects,
                destination_location_list=[basket] * len(objects),
                background_scene=surface,
                episode_length_s=150.0 * len(objects),
            )
            # SortMultiObjectTask takes no task_description, so get_language_instruction() would be None.
            # GapRemotePolicy reads env.get_language_instruction(), so set it explicitly from the object set + basket.
            # NOTE: the stock task scores object_on_destination at force_threshold=1.0 (intentional — the stock value;
            # the retired custom factory used 0.1). Velocity threshold 0.1 m/s is unchanged.
            task.task_description = (
                f"Pick up the {' and the '.join(obj.name for obj in objects)}, and place all into the {basket.name}"
            )
        else:
            task = NoTask()

        placer_params = ObjectPlacerParams(
            placement_seed=args_cli.placement_seed,
            solver_params=RelationSolverParams(
                clearance_m=0.06,
                verbose=False,
                save_position_history=False,
            ),
        )
        if args_cli.resolve_on_reset is not None:
            placer_params.resolve_on_reset = args_cli.resolve_on_reset

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            placer_params=placer_params,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--objects",
            nargs="*",
            default=[
                "alphabet_soup_can_hope_robolab",
                "tomato_sauce_can_hope_robolab",
                "milk_carton_hope_robolab",
            ],
            help="grocery assets to pack (relation-solved placement)",
        )
        parser.add_argument("--basket", type=str, default="grey_bin_robolab", help="container asset")
        parser.add_argument(
            "--control",
            type=str,
            default="ik",
            choices=["ik", "joint_pos", "droid_joint_pos"],
            help=(
                "Arm action term: 'ik' (relative differential IK, default), 'joint_pos' "
                "(absolute joint-position targets, for the GaP bridge), or 'droid_joint_pos' "
                "(DROID Franka+Robotiq absolute joint targets, Robotiq binary gripper)."
            ),
        )
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--eval_task",
            type=str,
            choices=["none", "pick_place_in_basket", "stock_pick_place", "stock_sort"],
            default="none",
            help=(
                "Scoring task to attach for eval_runner. "
                "'none' = NoTask (viewer/bridge runs unaffected); "
                "'pick_place_in_basket' = custom multi-object resting_in_bin success; "
                "'stock_pick_place' = STOCK PickAndPlaceTask/object_on_destination (single object -> basket); "
                "'stock_sort' = STOCK SortMultiObjectTask: N objects (from --objects) scored per-object "
                "object_on_destination under SuccessMode.ALL into the basket."
            ),
        )


def _make_libero_packing_task(
    object_asset_names: list[str],
    basket_asset_name: str,
    max_x_separation: float = 0.15,
    max_y_separation: float = 0.10,
    max_z_separation: float = 0.11,
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.5,
    gripper_open_threshold: float = 0.035,
    episode_length_s: float = 390.0,
):
    """Pack-all-N task: success = ALL N objects resting in the bin (footprint + below-rim + settled) AND the
    gripper is open (released). Thresholds calibrated to the grey_bin (footprint half (0.21,0.14), rim 0.105)."""
    from dataclasses import MISSING

    import isaaclab.envs.mdp as mdp_isaac_lab
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
    from isaaclab.utils import configclass

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.task_base import TaskBase
    from isaaclab_arena.tasks.terminations import SuccessMode, check_success, gripper_open, resting_in_bin

    @configclass
    class _TermsCfg:
        time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        success: TerminationTermCfg = MISSING

    class LiberoMultiPackingTask(TaskBase):
        def __init__(self):
            super().__init__(
                episode_length_s=episode_length_s,
                task_description=f"Pack all of {object_asset_names} into the basket.",
            )
            predicates = [
                TerminationTermCfg(
                    func=resting_in_bin,
                    params={
                        "object_cfg": SceneEntityCfg(name),
                        "target_object_cfg": SceneEntityCfg(basket_asset_name),
                        "max_x_separation": max_x_separation,
                        "max_y_separation": max_y_separation,
                        "max_z_separation": max_z_separation,
                        "lin_vel_threshold": lin_vel_threshold,
                        "ang_vel_threshold": ang_vel_threshold,
                    },
                )
                for name in object_asset_names
            ] + [
                TerminationTermCfg(
                    func=gripper_open,
                    params={"robot_cfg": SceneEntityCfg("robot"), "open_threshold": gripper_open_threshold},
                )
            ]
            self._termination_cfg = _TermsCfg(
                success=TerminationTermCfg(
                    func=check_success, params={"mode": SuccessMode.ALL, "predicates": predicates}
                )
            )

        def get_scene_cfg(self):
            return None

        def get_termination_cfg(self):
            return self._termination_cfg

        def get_events_cfg(self):
            return None

        def get_mimic_env_cfg(self, arm_mode: ArmMode):
            return None

        def get_metrics(self):
            return [SuccessRateMetric()]

        def get_viewer_cfg(self) -> ViewerCfg:
            return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))

    return LiberoMultiPackingTask()
