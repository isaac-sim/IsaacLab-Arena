# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained bimanual YAM cable-routing environment using Newton."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


_ASSET_DIRECTORY = Path(__file__).resolve().parent / "cable_routing_yam_newton_assets"
_YAM_USD_PATH = _ASSET_DIRECTORY / "yam" / "i2rt_yam_cable_routing.usda"
_TABLE_USD_PATH = _ASSET_DIRECTORY / "table" / "industrial__yam_workcell_table" / "industrial__yam_workcell_table.usda"
_BOARD_USD_PATH = _ASSET_DIRECTORY / "manipulationnet" / "board.usdc"
_ROUND_PEG_USD_PATH = _ASSET_DIRECTORY / "manipulationnet" / "round_peg.usdc"

# These values come from Isaac-cap's cable-routing scene.yaml. Keeping them here
# preserves this environment's one-module layout without requiring Isaac-cap.
_EMBODIMENT_MIDPOINT = (-0.335, 0.0, 0.767)
_TABLE_FRAME_POSITION = (-0.335, 0.0, 0.767)
_TABLE_FRAME_ROTATION = (0.0, 0.0, 0.0, 1.0)
_BOARD_POSITION = (0.0125, 0.0, 0.770175)

_TABLE_TOP_Z = 0.767
_TABLE_CENTER_X = _EMBODIMENT_MIDPOINT[0] + 0.3475
_BOARD_SIZE = (0.30, 0.40)
_BOARD_THICKNESS = 0.00635
_BOARD_TOP_Z = _TABLE_TOP_Z + _BOARD_THICKNESS
_PEG_HEIGHT = 0.0235
_PEG_CENTER_Z = _BOARD_TOP_Z + 0.5 * _PEG_HEIGHT

_CABLE_LENGTH = 1.0
_CABLE_SEGMENT_LENGTH = 0.01
_CABLE_NUM_SEGMENTS = round(_CABLE_LENGTH / _CABLE_SEGMENT_LENGTH)
_CABLE_THICKNESS = 0.006
_CABLE_RADIUS = 0.5 * _CABLE_THICKNESS
_CABLE_CENTER_Z = _BOARD_TOP_Z + _CABLE_RADIUS + 0.002
_CABLE_DENSITY = 1200.0
_CABLE_TARGET_STRETCH_STIFFNESS = 2.0e5
_CABLE_TARGET_BEND_STIFFNESS = 0.02
# CableMaterialCfg accepts elastic moduli while ModelBuilder.add_rod accepted
# per-joint stiffness. Convert the original task tuning to preserve the same
# axial and bending response with the supported CableObject path.
_CABLE_CROSS_SECTION_AREA = math.pi * _CABLE_RADIUS**2
_CABLE_SECOND_MOMENT_OF_AREA = math.pi * _CABLE_RADIUS**4 / 4.0
_CABLE_STRETCH_MODULUS = _CABLE_TARGET_STRETCH_STIFFNESS * _CABLE_SEGMENT_LENGTH / _CABLE_CROSS_SECTION_AREA
_CABLE_BEND_MODULUS = _CABLE_TARGET_BEND_STIFFNESS * _CABLE_SEGMENT_LENGTH / _CABLE_SECOND_MOMENT_OF_AREA
# Fingertips bind their own friction-60 material in the task YAM layer. Keep the
# procedural cable's material moderate so the unlifted span can slide over the
# board instead of making the gripper fight the full fixture drag.
_CABLE_CONTACT_FRICTION = 0.1

_FIXTURE_CONTACT_FRICTION = 0.5
# Match the cable-routing task proposed in IsaacLab PR #7082.
_CONTACT_STIFFNESS = 4.0e4
_CONTACT_DAMPING = 1.0e-5
_CONTACT_GAP = 0.001
_COLLISION_SUBSTEP_INTERVAL = 2

_YAM_BASE_COLLISION_DEPTH = 0.017
_YAM_BASE_Z = _EMBODIMENT_MIDPOINT[2]
_YAM_VISUAL_BASE_DEPTH = 0.07
_YAM_VISUAL_BASE_WIDTH = 0.20
_YAM_BOARD_GAP = 0.15
_YAM_FRONT_X = _EMBODIMENT_MIDPOINT[0]
_YAM_LATERAL_OFFSET = 0.5 * (_BOARD_SIZE[1] + _YAM_VISUAL_BASE_WIDTH)
_YAM_GRIPPER_OPEN_POS = 0.0375
_YAM_GRIPPER_CLOSED_POS = 0.0
_YAM_INITIAL_JOINT_POSITIONS = {
    "joint1": 0.0,
    "joint2": 0.85,
    "joint3": 0.60,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "left_finger": _YAM_GRIPPER_OPEN_POS,
    "right_finger": -_YAM_GRIPPER_OPEN_POS,
}

_PEG_POSITIONS = (
    (0.0575, -0.055, 0.7851),
    (-0.0225, 0.085, 0.7851),
)

_SUCCESS_ROUTE_PEG_INDICES = (0, 1)
_SUCCESS_ROUTE_DIRECTIONS = (-1.0, 1.0)
_ROUTE_RADIAL_CUTOFF = 0.05
_ROUTE_AXIAL_CUTOFF = 0.5 * _PEG_HEIGHT + _CABLE_RADIUS
_ROUTE_COMPLETION_WINDING = 2.6
_ROUTE_MAXIMUM_COMPLETION_WINDING = 2.0 * math.pi + 0.25
_ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH = 0.25


def _make_neutral_rounded_cable_positions() -> list[tuple[float, float, float]]:
    """Return the upstream smooth, exact-length cable initialization curve."""
    corner_segments = 6
    corner_step = 0.5 * math.pi / corner_segments
    corner_radius = _CABLE_SEGMENT_LENGTH / (2.0 * math.sin(0.5 * corner_step))
    horizontal_segments = 18
    vertical_segments = 30
    half_horizontal = 0.5 * horizontal_segments * _CABLE_SEGMENT_LENGTH
    half_vertical = 0.5 * vertical_segments * _CABLE_SEGMENT_LENGTH
    positions = [(-half_horizontal, -half_vertical - corner_radius, 0.0)]

    def append_straight(heading: float, count: int) -> None:
        for _ in range(count):
            x, y, z = positions[-1]
            positions.append((
                x + _CABLE_SEGMENT_LENGTH * math.cos(heading),
                y + _CABLE_SEGMENT_LENGTH * math.sin(heading),
                z,
            ))

    def append_corner(center_x: float, center_y: float, start_angle: float) -> None:
        for step in range(1, corner_segments + 1):
            angle = start_angle + step * corner_step
            positions.append((
                center_x + corner_radius * math.cos(angle),
                center_y + corner_radius * math.sin(angle),
                0.0,
            ))

    append_straight(0.0, horizontal_segments)
    append_corner(half_horizontal, -half_vertical, -0.5 * math.pi)
    append_straight(0.5 * math.pi, vertical_segments)
    append_corner(half_horizontal, half_vertical, 0.0)
    append_straight(math.pi, horizontal_segments)
    append_corner(-half_horizontal, half_vertical, 0.5 * math.pi)
    append_straight(-0.5 * math.pi, vertical_segments)
    append_corner(-half_horizontal, -half_vertical, math.pi)

    assert len(positions) > _CABLE_NUM_SEGMENTS, "Rounded cable template is shorter than the requested cable."
    return positions[: _CABLE_NUM_SEGMENTS + 1]


_CABLE_LOCAL_POSITIONS = _make_neutral_rounded_cable_positions()


def _fixed_route_success_from_geometry(
    cable_points_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the upstream peg-winding success test for the fixed full route."""
    import torch

    assert (
        cable_points_w.ndim == 3 and cable_points_w.shape[-1] == 3
    ), f"Expected cable points with shape (N, S, 3), got {tuple(cable_points_w.shape)}."
    assert (
        peg_positions_w.ndim == 3 and peg_positions_w.shape[-1] == 3
    ), f"Expected peg positions with shape (N, P, 3), got {tuple(peg_positions_w.shape)}."
    assert (
        cable_points_w.shape[0] == peg_positions_w.shape[0]
    ), "Cable points and peg positions must contain the same number of environments."
    assert cable_points_w.shape[1] >= 2, "At least two ordered cable points are required."

    finite_geometry = torch.isfinite(cable_points_w).all(dim=(1, 2)) & torch.isfinite(peg_positions_w).all(dim=(1, 2))
    safe_cable_points = torch.where(finite_geometry[:, None, None], cable_points_w, torch.zeros_like(cable_points_w))
    safe_peg_positions = torch.where(
        finite_geometry[:, None, None],
        peg_positions_w,
        torch.zeros_like(peg_positions_w),
    )

    relative_xy = safe_cable_points[:, None, :, :2] - safe_peg_positions[:, :, None, :2]
    local_points = torch.linalg.vector_norm(relative_xy, dim=-1) <= _ROUTE_RADIAL_CUTOFF
    relative_z = safe_cable_points[:, None, :, 2] - safe_peg_positions[:, :, None, 2]
    local_points &= relative_z.abs() <= _ROUTE_AXIAL_CUTOFF

    angle = torch.atan2(relative_xy[..., 1], relative_xy[..., 0])
    angle_delta = angle[..., 1:] - angle[..., :-1]
    angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
    local_edges = local_points[..., :-1] & local_points[..., 1:]
    clockwise_winding = -torch.where(local_edges, angle_delta, 0.0).sum(dim=-1)

    previous_local_edges = torch.zeros_like(local_edges)
    previous_local_edges[..., 1:] = local_edges[..., :-1]
    local_span_count = (local_edges & ~previous_local_edges).sum(dim=-1)
    edge_lengths = torch.linalg.vector_norm(safe_cable_points[:, 1:] - safe_cable_points[:, :-1], dim=-1)
    local_cable_length = torch.where(local_edges, edge_lengths[:, None, :], 0.0).sum(dim=-1)
    geometrically_eligible = (local_span_count == 1) & (local_cable_length <= _ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH)

    route_peg_indices = torch.tensor(_SUCCESS_ROUTE_PEG_INDICES, device=cable_points_w.device)
    route_directions = torch.tensor(
        _SUCCESS_ROUTE_DIRECTIONS,
        device=cable_points_w.device,
        dtype=cable_points_w.dtype,
    )
    route_peg_indices = route_peg_indices.unsqueeze(0).expand(cable_points_w.shape[0], -1)
    route_directions = route_directions.unsqueeze(0).expand(cable_points_w.shape[0], -1)
    directed_winding = torch.gather(clockwise_winding, 1, route_peg_indices) * route_directions
    route_geometry_valid = torch.gather(geometrically_eligible, 1, route_peg_indices)
    route_steps_complete = (
        route_geometry_valid
        & (directed_winding >= _ROUTE_COMPLETION_WINDING)
        & (directed_winding <= _ROUTE_MAXIMUM_COMPLETION_WINDING)
    )
    return finite_geometry & route_steps_complete.all(dim=1)


def _cable_route_success(env) -> torch.Tensor:
    """Terminate when the cable completes peg-0 CCW followed by peg-1 CW routing."""
    import torch

    cable_points_w = env.scene["cable"].data.segment_pose_w.torch[..., :3]
    peg_positions_w = torch.stack(
        [
            env.scene["peg_0"].data.root_pos_w.torch,
            env.scene["peg_1"].data.root_pos_w.torch,
        ],
        dim=1,
    )
    return _fixed_route_success_from_geometry(cable_points_w, peg_positions_w)


def _configure_cable_routing_newton_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure coupled MJWarp rigid-body and VBD cable physics."""
    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg, VBDSolverCfg
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.sim.render_interval = 4
    env_cfg.sim.use_newton_actuators = True
    env_cfg.sim.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=_FIXTURE_CONTACT_FRICTION,
        dynamic_friction=_FIXTURE_CONTACT_FRICTION,
        restitution=0.0,
        contact_stiffness=_CONTACT_STIFFNESS,
        contact_damping=_CONTACT_DAMPING,
    )
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        njmax=300,
                        nconmax=200,
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                        ccd_iterations=100,
                    ),
                    bodies=[
                        r"/World/envs/env_.*/YamLeft",
                        r"/World/envs/env_.*/YamRight",
                        r"/World/envs/env_.*/Board",
                        r"/World/envs/env_.*/Peg0",
                        r"/World/envs/env_.*/Peg1",
                    ],
                ),
                CouplerEntryCfg(
                    name="cable",
                    solver_cfg=VBDSolverCfg(iterations=10),
                    bodies=[r"/World/envs/env_.*/Cable"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cable",
                    bodies=[
                        r"/World/envs/env_.*/Yam(Left|Right)/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6",
                        r"/World/envs/env_.*/Board",
                        r"/World/envs/env_.*/Peg(0|1)",
                    ],
                    mode="lagged",
                    mass_scale=1.0,
                    collide_interval=_COLLISION_SUBSTEP_INTERVAL,
                )
            ],
            iterations=1,
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=_CONTACT_STIFFNESS,
            kd=_CONTACT_DAMPING,
            mu=_CABLE_CONTACT_FRICTION,
            margin=0.0,
            gap=_CONTACT_GAP,
        ),
        num_substeps=10,
        use_cuda_graph=True,
        debug_mode=False,
    )
    env_cfg.decimation = 4
    env_cfg.scene.replicate_physics = True
    return env_cfg


def _make_yam_cfg(prim_path: str, position: tuple[float, float, float], yaw: float):
    """Create one fixed-base YAM articulation using the upstream asset."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.schemas.schemas_cfg import JointDriveBaseCfg
    from isaaclab_newton.sim.schemas import (
        MujocoRigidBodyPropertiesCfg,
        NewtonArticulationRootPropertiesCfg,
        NewtonCollisionPropertiesCfg,
    )

    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path="/Geometry/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_YAM_USD_PATH),
            copy_from_source=False,
            rigid_props=MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
            articulation_props=NewtonArticulationRootPropertiesCfg(self_collision_enabled=False),
            collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=_CONTACT_GAP),
            joint_drive_props=JointDriveBaseCfg(ensure_drives_exist=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
            joint_pos=_YAM_INITIAL_JOINT_POSITIONS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm_proximal": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"],
                effort_limit_sim=28.0,
                stiffness=80.0,
                damping=6.0,
            ),
            "arm_joint_4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=10.0,
                stiffness=30.0,
                damping=2.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-6]"],
                effort_limit_sim=10.0,
                stiffness=30.0,
                damping=2.0,
            ),
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["left_finger"],
                stiffness=1000.0,
                damping=100.0,
            ),
            # MJWarp imports the source YAM's -1 equality from left_finger to
            # right_finger. Keep the mirrored coordinate passive, matching PR
            # #7082, instead of making a second drive fight that constraint.
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=["right_finger"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


def _make_bimanual_yam_camera_cfg():
    """Create the YAM wrist and overhead camera configuration locally."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.utils.cameras import ArenaCameraCfg

    camera_width = 1280
    camera_height = 720
    mount_position = (-0.0107, 0.079729, 0.066021)
    mount_rotation_xyzw = (0.423, 0.0, 0.0, 0.906)
    top_offset_from_robot_midpoint = (0.335, 0.0, 0.93732053)
    sqrt_half = math.sqrt(0.5)
    top_rotation_xyzw = (sqrt_half, sqrt_half, 0.0, 0.0)
    vertical_aperture = 4.8
    vertical_fov_deg = 58.0
    focal_length = vertical_aperture / (2.0 * math.tan(math.radians(vertical_fov_deg / 2.0)))
    link_six_suffix = "/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"

    def d405_camera(
        prim_path: str,
        *,
        position: tuple[float, float, float] = mount_position,
        rotation_xyzw: tuple[float, float, float, float] = mount_rotation_xyzw,
    ) -> CameraCfg:
        return CameraCfg(
            prim_path=prim_path,
            height=camera_height,
            width=camera_width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=28.0,
                horizontal_aperture=6.4,
                vertical_aperture=vertical_aperture,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=position,
                rot=rotation_xyzw,
                convention="ros",
            ),
        )

    @configclass
    class BimanualYamCameraCfg(ArenaCameraCfg):
        """One overhead camera and one wrist camera on each YAM."""

        left_wrist_camera: CameraCfg = d405_camera(f"{{ENV_REGEX_NS}}/YamLeft{link_six_suffix}/left_wrist_camera")
        right_wrist_camera: CameraCfg = d405_camera(f"{{ENV_REGEX_NS}}/YamRight{link_six_suffix}/right_wrist_camera")
        top_camera: CameraCfg = d405_camera(
            "{ENV_REGEX_NS}/top_camera",
            position=top_offset_from_robot_midpoint,
            rotation_xyzw=top_rotation_xyzw,
        )

        def set_robot_prim_paths(self, left: str, right: str) -> None:
            """Attach wrist cameras to the articulation roots in this scene."""
            self.left_wrist_camera.prim_path = f"{left}{link_six_suffix}/left_wrist_camera"
            self.right_wrist_camera.prim_path = f"{right}{link_six_suffix}/right_wrist_camera"

        def set_robot_mount_positions(
            self,
            left: tuple[float, float, float],
            right: tuple[float, float, float],
        ) -> None:
            """Place the overhead camera relative to the robot pair."""
            midpoint = tuple((float(left_axis) + float(right_axis)) * 0.5 for left_axis, right_axis in zip(left, right))
            self.top_camera.offset.pos = tuple(
                midpoint_axis + offset_axis
                for midpoint_axis, offset_axis in zip(midpoint, top_offset_from_robot_midpoint)
            )

    return BimanualYamCameraCfg()


def _make_bimanual_yam_embodiment(enable_cameras: bool = False):
    """Create the two-YAM Arena embodiment and its 14-dimensional action interface."""
    import torch

    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import mdp
    from isaaclab.envs.mdp.actions import BinaryJointPositionAction, JointPositionAction
    from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    class FiniteJointPositionAction(JointPositionAction):
        """Keep absolute joint targets finite and inside the soft limits."""

        def process_actions(self, actions: torch.Tensor) -> None:
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            super().process_actions(actions)
            default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
            limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
            target = torch.where(
                torch.isfinite(self._processed_actions),
                self._processed_actions,
                default,
            )
            self._processed_actions = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])

    class FiniteContinuousJointPositionAction(BinaryJointPositionAction):
        """Map one finite 0=open, 1=closed command to the driven jaw."""

        def process_actions(self, actions: torch.Tensor) -> None:
            fraction = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            self._raw_actions[:] = fraction
            self._processed_actions = self._open_command + fraction * (self._close_command - self._open_command)
            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions,
                    min=self._clip[:, :, 0],
                    max=self._clip[:, :, 1],
                )

    @configclass
    class FiniteJointPositionActionCfg(JointPositionActionCfg):
        class_type: type = FiniteJointPositionAction

    @configclass
    class FiniteContinuousJointPositionActionCfg(BinaryJointPositionActionCfg):
        class_type: type = FiniteContinuousJointPositionAction

    def arm_action_cfg(asset_name: str) -> FiniteJointPositionActionCfg:
        return FiniteJointPositionActionCfg(
            asset_name=asset_name,
            joint_names=["joint[1-6]"],
            preserve_order=True,
            use_default_offset=False,
        )

    def gripper_action_cfg(asset_name: str) -> FiniteContinuousJointPositionActionCfg:
        return FiniteContinuousJointPositionActionCfg(
            asset_name=asset_name,
            # MJWarp owns the robot and mirrors this coordinate onto the passive
            # right jaw through the YAM's imported equality constraint.
            joint_names=["left_finger"],
            open_command_expr={
                "left_finger": _YAM_GRIPPER_OPEN_POS,
            },
            close_command_expr={
                "left_finger": _YAM_GRIPPER_CLOSED_POS,
            },
        )

    @configclass
    class BimanualYamSceneCfg:
        left_robot: ArticulationCfg | None = None
        right_robot: ArticulationCfg | None = None

    @configclass
    class BimanualYamActionsCfg:
        left_arm_action = arm_action_cfg("left_robot")
        left_gripper_action = gripper_action_cfg("left_robot")
        right_arm_action = arm_action_cfg("right_robot")
        right_gripper_action = gripper_action_cfg("right_robot")

    @configclass
    class BimanualYamObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            left_joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("left_robot")},
            )
            left_joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("left_robot")},
            )
            right_joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("right_robot")},
            )
            right_joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("right_robot")},
            )
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    class BimanualYamEmbodiment(EmbodimentBase):
        """Two fixed-base YAM manipulators controlled in joint space."""

        name = "bimanual_yam_cable_routing"
        default_arm_mode = ArmMode.DUAL_ARM

        def __init__(self) -> None:
            super().__init__(enable_cameras=enable_cameras)
            self.scene_config = BimanualYamSceneCfg(
                left_robot=_make_yam_cfg(
                    "{ENV_REGEX_NS}/YamLeft",
                    (_YAM_FRONT_X, _YAM_LATERAL_OFFSET, _YAM_BASE_Z),
                    0.0,
                ),
                right_robot=_make_yam_cfg(
                    "{ENV_REGEX_NS}/YamRight",
                    (_YAM_FRONT_X, -_YAM_LATERAL_OFFSET, _YAM_BASE_Z),
                    0.0,
                ),
            )
            self.action_config = BimanualYamActionsCfg()
            self.observation_config = BimanualYamObservationsCfg()
            self.camera_config = _make_bimanual_yam_camera_cfg() if enable_cameras else None
            if self.camera_config is not None:
                self.camera_config.use_tiled_camera = False
                self.camera_config.set_robot_prim_paths(
                    "{ENV_REGEX_NS}/YamLeft",
                    "{ENV_REGEX_NS}/YamRight",
                )
                self.camera_config.set_robot_mount_positions(
                    self.scene_config.left_robot.init_state.pos,
                    self.scene_config.right_robot.init_state.pos,
                )

        def get_command_body_name(self) -> str:
            return "link_6"

        def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
            return "link_6"

    return BimanualYamEmbodiment()


def _make_cable_routing_task():
    """Create the minimal scene, reset, route-success, and timeout task."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, CableObjectCfg, RigidObjectCfg
    from isaaclab.envs import mdp
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg, NewtonMaterialPropertiesCfg

    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.task_base import TaskBase

    def fixture_material() -> NewtonMaterialPropertiesCfg:
        return NewtonMaterialPropertiesCfg(
            static_friction=_FIXTURE_CONTACT_FRICTION,
            dynamic_friction=_FIXTURE_CONTACT_FRICTION,
            restitution=0.0,
            contact_stiffness=_CONTACT_STIFFNESS,
            contact_damping=_CONTACT_DAMPING,
        )

    def peg_cfg(name: str, position: tuple[float, float, float]) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_ROUND_PEG_USD_PATH),
                copy_from_source=False,
                physics_material=fixture_material(),
                rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
                collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=0.002),
            ),
        )

    @configclass
    class CableRoutingSceneCfg:
        # The canonical table is authored as static collision geometry, not a
        # rigid body. VBD imports static shapes directly, so it must remain an
        # AssetBase instead of being bound through RigidObject's root-body
        # lookup. Board and pegs remain kinematic rigid fixtures.
        table: AssetBaseCfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=_TABLE_FRAME_POSITION,
                rot=_TABLE_FRAME_ROTATION,
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_TABLE_USD_PATH),
                copy_from_source=False,
            ),
        )
        board: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Board",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_BOARD_POSITION),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_BOARD_USD_PATH),
                copy_from_source=False,
                physics_material=fixture_material(),
                rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
                collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=0.002),
            ),
        )
        peg_0: RigidObjectCfg = peg_cfg("Peg0", _PEG_POSITIONS[0])
        peg_1: RigidObjectCfg = peg_cfg("Peg1", _PEG_POSITIONS[1])
        cable: CableObjectCfg = CableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cable",
            spawn=sim_utils.CableCfg(
                positions=_CABLE_LOCAL_POSITIONS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.07, 0.07, 0.08)),
                physics_material=sim_utils.CableMaterialCfg(
                    thickness=_CABLE_THICKNESS,
                    density=_CABLE_DENSITY,
                    stretch_stiffness=_CABLE_STRETCH_MODULUS,
                    bend_stiffness=_CABLE_BEND_MODULUS,
                ),
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=(_TABLE_CENTER_X, 0.0, _CABLE_CENTER_Z)),
        )
        ground: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(
                color=(0.20, 0.20, 0.20),
                physics_material=fixture_material(),
            ),
            collision_group=-1,
        )
        sky_light: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=1500.0,
                color=(0.75, 0.75, 0.75),
            ),
        )

    @configclass
    class CableRoutingEventsCfg:
        reset_scene = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )

    @configclass
    class CableRoutingTerminationsCfg:
        success = DoneTerm(func=_cable_route_success)
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    class CableRoutingTask(TaskBase):
        """Minimal task that keeps the upstream cable-routing scene steppable."""

        def __init__(self) -> None:
            super().__init__(
                episode_length_s=3600.0,
                task_description=(
                    "Route the cable counterclockwise around the first peg, then clockwise around the second peg, "
                    "using both YAM manipulators."
                ),
            )
            self._scene_cfg = CableRoutingSceneCfg()
            self._events_cfg = CableRoutingEventsCfg()
            self._terminations_cfg = CableRoutingTerminationsCfg()

        def get_scene_cfg(self):
            return self._scene_cfg

        def get_termination_cfg(self):
            return self._terminations_cfg

        def get_events_cfg(self):
            return self._events_cfg

        def get_mimic_env_cfg(self, arm_mode):
            return None

        def get_metrics(self):
            return [SuccessRateMetric()]

        def get_viewer_cfg(self) -> ViewerCfg:
            return ViewerCfg(
                eye=(1.25, -1.10, 1.55),
                lookat=(_TABLE_CENTER_X, 0.0, _BOARD_TOP_Z),
            )

    return CableRoutingTask()


@dataclass
class CableRoutingYamNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the self-contained YAM cable-routing environment."""


@register_environment
class CableRoutingYamNewtonEnvironment(ArenaEnvironmentFactory[CableRoutingYamNewtonEnvironmentCfg]):
    """Build the self-contained two-YAM Newton cable-routing environment."""

    name = "cable_routing_yam_newton"
    _legacy_argparse_cfg_type = CableRoutingYamNewtonEnvironmentCfg

    def build(self, cfg: CableRoutingYamNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment using the locally bundled upstream assets."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        for asset_path in (
            _YAM_USD_PATH,
            _TABLE_USD_PATH,
            _BOARD_USD_PATH,
            _ROUND_PEG_USD_PATH,
        ):
            assert asset_path.is_file(), f"Cable-routing asset not found: {asset_path}"

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=_make_bimanual_yam_embodiment(enable_cameras=cfg.enable_cameras),
            scene=Scene(assets=[]),
            task=_make_cable_routing_task(),
            env_cfg_callback=_configure_cable_routing_newton_physics,
        )
