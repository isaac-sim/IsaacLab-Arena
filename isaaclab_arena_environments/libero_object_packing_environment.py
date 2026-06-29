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
_REACH_BOX = dict(x_min=0.12, x_max=0.52, y_min=-0.30, y_max=0.26)
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
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits, RandomAroundSolution
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
        objects = []
        for obj_name in args_cli.objects:
            obj = self.asset_registry.get_asset_by_name(obj_name)()
            obj.add_relation(On(surface, edge_margin_m=0.03))
            obj.add_relation(PositionLimits(**_REACH_BOX))
            obj.add_relation(RandomAroundSolution(x_half_m=0.04, y_half_m=0.04, yaw_half_rad=0.4))
            objects.append(obj)

        scene = Scene(assets=[ground, light, surface, basket, *objects])

        if args_cli.eval_task == "pick_place_in_basket":
            task = _make_libero_packing_task(
                can_asset_name=objects[0].name,  # alphabet_soup_can_hope_robolab
                basket_asset_name=basket.name,   # grey_bin_robolab
            )
        else:
            task = NoTask()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
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
                "salad_dressing_bottle_hot3d_robolab",
                "cream_cheese_hope_robolab",
                "butter_hope_robolab",
            ],
            help="grocery assets to pack (relation-solved placement)",
        )
        parser.add_argument("--basket", type=str, default="grey_bin_robolab", help="container asset")
        parser.add_argument(
            "--control",
            type=str,
            default="ik",
            choices=["ik", "joint_pos", "droid_joint_pos"],
            help="Arm action term: 'ik' (relative differential IK, default), 'joint_pos' "
            "(absolute joint-position targets, for the GaP bridge), or 'droid_joint_pos' "
            "(DROID Franka+Robotiq absolute joint targets, Robotiq binary gripper).",
        )
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--eval_task",
            type=str,
            choices=["none", "pick_place_in_basket"],
            default="none",
            help=(
                "Scoring task to attach for eval_runner. "
                "'none' = NoTask (viewer/bridge runs unaffected); "
                "'pick_place_in_basket' = proximity success for alphabet_soup_can into basket."
            ),
        )


def _make_libero_packing_task(
    can_asset_name: str,
    basket_asset_name: str,
    max_x_separation: float = 0.12,
    max_y_separation: float = 0.12,
    max_z_separation: float = 0.20,
    episode_length_s: float = 180.0,
):
    """Factory that builds a ``LiberoPackingTask`` with all Isaac Lab imports deferred.

    All imports of Isaac Lab configclasses happen inside this function so that
    the environment module can be imported and registered without booting Isaac Sim.
    """
    from dataclasses import MISSING

    import isaaclab.envs.mdp as mdp_isaac_lab
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
    from isaaclab.utils import configclass

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.task_base import TaskBase
    from isaaclab_arena.tasks.terminations import SuccessMode, check_success, objects_in_proximity

    @configclass
    class _LiberoPackingTerminationsCfg:
        time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        success: TerminationTermCfg = MISSING

    class LiberoPackingTask(TaskBase):
        """Proximity-only success task for the LIBERO object-packing eval.

        Fires 'success' when the alphabet_soup_can is within (max_x, max_y, max_z)
        of the basket centroid.  No contact sensor, no object_min_z, no mimic cfg.
        """

        def __init__(
            self,
            can_asset_name: str,
            basket_asset_name: str,
            max_x_separation: float,
            max_y_separation: float,
            max_z_separation: float,
            episode_length_s: float,
        ):
            super().__init__(
                episode_length_s=episode_length_s,
                task_description=f"Pick up the {can_asset_name} and place it in the basket.",
            )
            predicate = TerminationTermCfg(
                func=objects_in_proximity,
                params={
                    "object_cfg": SceneEntityCfg(can_asset_name),
                    "target_object_cfg": SceneEntityCfg(basket_asset_name),
                    "max_x_separation": max_x_separation,
                    "max_y_separation": max_y_separation,
                    "max_z_separation": max_z_separation,
                },
            )
            success_term = TerminationTermCfg(
                func=check_success,
                params={"mode": SuccessMode.ALL, "predicates": [predicate]},
            )
            self._termination_cfg = _LiberoPackingTerminationsCfg(success=success_term)

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

    return LiberoPackingTask(
        can_asset_name=can_asset_name,
        basket_asset_name=basket_asset_name,
        max_x_separation=max_x_separation,
        max_y_separation=max_y_separation,
        max_z_separation=max_z_separation,
        episode_length_s=episode_length_s,
    )
