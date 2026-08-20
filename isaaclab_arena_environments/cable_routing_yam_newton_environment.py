# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-environment Newton/VBD bring-up of the bimanual YAM cable-routing task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


_ASSET_DIRECTORY = Path(__file__).resolve().parent / "cable_routing_yam_newton_assets"
_YAM_USD_PATH = _ASSET_DIRECTORY / "yam" / "i2rt_yam_cable_routing.usda"
_BOARD_USD_PATH = _ASSET_DIRECTORY / "manipulationnet" / "board.usdc"
_ROUND_PEG_USD_PATH = _ASSET_DIRECTORY / "manipulationnet" / "round_peg.usdc"

_TABLE_TOP_Z = 0.75
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
_CABLE_STRETCH_STIFFNESS = 2.0e5
_CABLE_BEND_STIFFNESS = 0.02
_CABLE_BEND_DAMPING = 1.0
# Use the material values from Newton 1.2's own VBD cable-pile reference.
# The upstream coupled Newton 1.5 task uses a proxy-specific friction of 60,
# which is not transferable to the Newton 1.2 compatibility solver.
_CABLE_CONTACT_FRICTION = 1.0

_FIXTURE_CONTACT_FRICTION = 0.5
# Newton 1.2's VBD cable reference uses a stiff, undamped rigid contact.
_CONTACT_STIFFNESS = 1.0e5
_CONTACT_DAMPING = 0.0
_CONTACT_GAP = 0.001
_COLLISION_SUBSTEP_INTERVAL = 2

_YAM_BASE_COLLISION_DEPTH = 0.017
_YAM_BASE_Z = _TABLE_TOP_Z + _YAM_BASE_COLLISION_DEPTH
_YAM_VISUAL_BASE_DEPTH = 0.07
_YAM_VISUAL_BASE_WIDTH = 0.20
_YAM_BOARD_GAP = 0.15
_YAM_FRONT_X = -0.5 * (_BOARD_SIZE[0] + _YAM_VISUAL_BASE_DEPTH) - _YAM_BOARD_GAP
_YAM_LATERAL_OFFSET = 0.5 * (_BOARD_SIZE[1] + _YAM_VISUAL_BASE_WIDTH)
_YAM_GRIPPER_OPEN_POS = 0.0375
_YAM_GRIPPER_CLOSED_POS = 0.0
_YAM_BODY_PATH_MARKERS = ("/YamLeft/", "/YamRight/")
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
    ((20.0 - 15.5) * 0.01, (15.0 - 20.5) * 0.01, _PEG_CENTER_Z),
    ((12.0 - 15.5) * 0.01, (29.0 - 20.5) * 0.01, _PEG_CENTER_Z),
)

_SUCCESS_ROUTE_PEG_INDICES = (0, 1)
_SUCCESS_ROUTE_DIRECTIONS = (-1.0, 1.0)
_ROUTE_RADIAL_CUTOFF = 0.05
_ROUTE_AXIAL_CUTOFF = 0.5 * _PEG_HEIGHT + _CABLE_RADIUS
_ROUTE_COMPLETION_WINDING = 2.6
_ROUTE_MAXIMUM_COMPLETION_WINDING = 2.0 * math.pi + 0.25
_ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH = 0.25

_CABLE_VBD_MANAGER_TYPE: type | None = None
_CABLE_INITIAL_BODY_POSES: dict[str, tuple[float, ...]] = {}


@wp.kernel
def _apply_yam_gravity_compensation(
    body_ids: wp.array(dtype=wp.int32),
    body_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Cancel gravity on YAM bodies while retaining their full dynamics."""
    body_index = body_ids[wp.tid()]
    force = -gravity * body_mass[body_index]
    wp.atomic_add(body_f, body_index, wp.spatial_vector(force, wp.vec3(0.0)))


def _is_yam_label(label: object) -> bool:
    """Return whether a Newton model label belongs to either YAM robot."""
    return isinstance(label, str) and any(marker in label for marker in _YAM_BODY_PATH_MARKERS)


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


def _rotate_vector_xyzw(
    quaternion: tuple[float, float, float, float] | list[float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate one vector by an XYZW quaternion."""
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _create_cable_segment_visuals(world_index: int) -> list[str]:
    """Create one render-only USD capsule under each Newton cable body path."""
    from isaaclab.sim import get_current_stage
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    cable_root_path = f"/World/envs/env_{world_index}/Cable"
    UsdGeom.Xform.Define(stage, cable_root_path)

    segment_paths = []
    for segment_index in range(_CABLE_NUM_SEGMENTS):
        segment_path = f"{cable_root_path}/Segment_{segment_index:03d}"
        segment_paths.append(segment_path)
        UsdGeom.Xform.Define(stage, segment_path)

        capsule = UsdGeom.Capsule.Define(stage, f"{segment_path}/Visual")
        capsule.CreateAxisAttr("Z")
        capsule.CreateHeightAttr(_CABLE_SEGMENT_LENGTH)
        capsule.CreateRadiusAttr(_CABLE_RADIUS)
        capsule.CreateDisplayColorAttr([Gf.Vec3f(0.07, 0.07, 0.08)])

        visual_xform = UsdGeom.Xformable(capsule.GetPrim())
        translate_attr = capsule.GetPrim().GetAttribute("xformOp:translate")
        if translate_attr.IsValid():
            translate_attr.Set(Gf.Vec3d(0.0, 0.0, 0.5 * _CABLE_SEGMENT_LENGTH))
        else:
            visual_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5 * _CABLE_SEGMENT_LENGTH))

    return segment_paths


def _add_cable_to_newton_builder(
    builder,
    world_index: int,
    environment_position: list[float],
    environment_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Add the segmented cable to one Newton world after USD assets are imported."""
    cable_root_path = f"/World/envs/env_{world_index}/Cable"
    if any(isinstance(label, str) and label.startswith(cable_root_path) for label in builder.body_label):
        return

    positions = []
    for x, y, z in _CABLE_LOCAL_POSITIONS:
        rotated = _rotate_vector_xyzw(environment_rotation, (x, y, z + _CABLE_CENTER_Z))
        positions.append((
            environment_position[0] + rotated[0],
            environment_position[1] + rotated[1],
            environment_position[2] + rotated[2],
        ))

    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = _CABLE_DENSITY
    shape_cfg.ke = _CONTACT_STIFFNESS
    shape_cfg.kd = _CONTACT_DAMPING
    shape_cfg.mu = _CABLE_CONTACT_FRICTION
    shape_cfg.margin = 0.0
    shape_cfg.gap = _CONTACT_GAP
    shape_cfg.collision_filter_parent = True

    shape_start = builder.shape_count
    joint_start = builder.joint_count
    articulation_start = builder.articulation_count
    body_indices, joint_indices = builder.add_rod(
        positions=positions,
        radius=_CABLE_RADIUS,
        cfg=shape_cfg,
        stretch_stiffness=_CABLE_STRETCH_STIFFNESS,
        stretch_damping=0.0,
        bend_stiffness=_CABLE_BEND_STIFFNESS,
        bend_damping=_CABLE_BEND_DAMPING,
        label=cable_root_path,
    )
    assert (
        len(body_indices) == _CABLE_NUM_SEGMENTS
    ), f"Expected {_CABLE_NUM_SEGMENTS} cable bodies, got {len(body_indices)}."
    assert (
        len(joint_indices) == _CABLE_NUM_SEGMENTS - 1
    ), f"Expected {_CABLE_NUM_SEGMENTS - 1} cable joints, got {len(joint_indices)}."

    segment_paths = _create_cable_segment_visuals(world_index)
    for segment_path, body_index in zip(segment_paths, body_indices):
        builder.body_label[body_index] = segment_path
        _CABLE_INITIAL_BODY_POSES[segment_path] = tuple(builder.body_q[body_index])
    for segment_index, shape_index in enumerate(range(shape_start, builder.shape_count)):
        builder.shape_label[shape_index] = f"{segment_paths[segment_index]}/Collision"
    for joint_index, cable_joint_index in enumerate(range(joint_start, builder.joint_count)):
        builder.joint_label[cable_joint_index] = f"{cable_root_path}/Joint_{joint_index:03d}"
    for articulation_index in range(articulation_start, builder.articulation_count):
        builder.articulation_label[articulation_index] = cable_root_path


def _finalize_cable_newton_builder(builder) -> None:
    """Color the complete dynamic model for the VBD solve."""
    if not any(isinstance(label, str) and "/Cable/Segment_" in label for label in builder.shape_label):
        return
    builder.color()


def _install_cable_builder_hooks() -> None:
    """Install scoped Newton builder hooks for the procedural cable."""
    from isaaclab_newton.physics import NewtonManager

    _CABLE_INITIAL_BODY_POSES.clear()
    if not hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = []
    if not hasattr(NewtonManager, "_post_replicate_hooks"):
        NewtonManager._post_replicate_hooks = []
    if _add_cable_to_newton_builder not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(_add_cable_to_newton_builder)
    if _finalize_cable_newton_builder not in NewtonManager._post_replicate_hooks:
        NewtonManager._post_replicate_hooks.append(_finalize_cable_newton_builder)


def _clear_cable_builder_hooks() -> None:
    """Remove only the builder hooks owned by this environment."""
    from isaaclab_newton.physics import NewtonManager

    if hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = [
            hook for hook in NewtonManager._per_world_builder_hooks if hook is not _add_cable_to_newton_builder
        ]
    if hasattr(NewtonManager, "_post_replicate_hooks"):
        NewtonManager._post_replicate_hooks = [
            hook for hook in NewtonManager._post_replicate_hooks if hook is not _finalize_cable_newton_builder
        ]
    _CABLE_INITIAL_BODY_POSES.clear()


def _get_cable_body_state_cache(manager_type: type) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached ordered cable body indices and their initial poses."""
    import torch

    cached_state = getattr(manager_type, "_cable_body_state_cache", None)
    if cached_state is not None:
        return cached_state

    model = manager_type.get_model()
    cable_entries = sorted(
        (
            (body_label, body_index)
            for body_index, body_label in enumerate(model.body_label)
            if isinstance(body_label, str) and "/Cable/Segment_" in body_label
        ),
        key=lambda entry: entry[0],
    )
    assert (
        len(cable_entries) == _CABLE_NUM_SEGMENTS
    ), f"Expected {_CABLE_NUM_SEGMENTS} labeled cable bodies, got {len(cable_entries)}."
    missing_initial_poses = [label for label, _ in cable_entries if label not in _CABLE_INITIAL_BODY_POSES]
    assert not missing_initial_poses, f"Missing initial poses for cable bodies: {missing_initial_poses}."

    device = wp.to_torch(manager_type.get_state_0().body_q).device
    body_ids = torch.tensor([body_index for _, body_index in cable_entries], dtype=torch.long, device=device)
    body_poses = torch.tensor(
        [_CABLE_INITIAL_BODY_POSES[label] for label, _ in cable_entries],
        dtype=torch.float32,
        device=device,
    )
    manager_type._cable_body_state_cache = (body_ids, body_poses)
    return body_ids, body_poses


def _restore_cable_initial_body_state(manager_type: type) -> None:
    """Restore the curved rod pose erased by Newton 1.2's unmasked startup FK."""
    body_ids, body_poses = _get_cable_body_state_cache(manager_type)

    for state in (manager_type.get_state_0(), manager_type.get_state_1()):
        if state is None:
            continue
        wp.to_torch(state.body_q)[body_ids] = body_poses
        wp.to_torch(state.body_qd)[body_ids] = 0.0
    manager_type._mark_transforms_dirty()


def _initialize_yam_vbd_rest_pose(manager_type: type) -> None:
    """Make VBD's structural robot pose match the configured reset joints."""
    import torch

    from newton import eval_fk

    model = manager_type.get_model()
    state_0 = manager_type.get_state_0()
    state_1 = manager_type.get_state_1()
    joint_q_starts = model.joint_q_start.numpy()
    model_joint_q = wp.to_torch(model.joint_q)
    state_0_joint_q = wp.to_torch(state_0.joint_q)
    state_1_joint_q = wp.to_torch(state_1.joint_q)

    for joint_index, joint_label in enumerate(model.joint_label):
        if not _is_yam_label(joint_label):
            continue
        joint_name = joint_label.rsplit("/", 1)[-1]
        if joint_name not in _YAM_INITIAL_JOINT_POSITIONS:
            continue
        dof_index = int(joint_q_starts[joint_index])
        value = _YAM_INITIAL_JOINT_POSITIONS[joint_name]
        model_joint_q[dof_index] = value
        state_0_joint_q[dof_index] = value
        state_1_joint_q[dof_index] = value

    eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, None)
    robot_body_ids = torch.tensor(
        [body_index for body_index, body_label in enumerate(model.body_label) if _is_yam_label(body_label)],
        dtype=torch.long,
        device=wp.to_torch(state_0.body_q).device,
    )
    assert robot_body_ids.numel() > 0, "No YAM bodies were imported into the Newton model."
    robot_body_q = wp.to_torch(state_0.body_q)[robot_body_ids]
    wp.to_torch(model.body_q)[robot_body_ids] = robot_body_q
    wp.to_torch(state_1.body_q)[robot_body_ids] = robot_body_q
    wp.to_torch(state_0.body_qd)[robot_body_ids] = 0.0
    wp.to_torch(state_1.body_qd)[robot_body_ids] = 0.0
    manager_type._mark_transforms_dirty()


def _reset_cable_to_initial_state(env, env_ids) -> None:
    """Restore the procedural cable, which is not an Isaac Lab scene asset."""
    del env_ids
    assert env.num_envs == 1, "The cable reset compatibility path supports one environment only."
    physics_manager = env.sim.physics_manager
    _restore_cable_initial_body_state(physics_manager)
    physics_manager.forward()


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
    safe_peg_positions = torch.where(finite_geometry[:, None, None], peg_positions_w, torch.zeros_like(peg_positions_w))

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

    physics_manager = env.sim.physics_manager
    cable_body_ids, _ = _get_cable_body_state_cache(physics_manager)
    state_body_q = wp.to_torch(physics_manager.get_state_0().body_q)
    cable_points_w = state_body_q[cable_body_ids, :3].unsqueeze(0)
    peg_positions_w = torch.stack(
        [env.scene["peg_0"].data.root_pos_w.torch, env.scene["peg_1"].data.root_pos_w.torch],
        dim=1,
    )
    return _fixed_route_success_from_geometry(cable_points_w, peg_positions_w)


def _reset_vbd_history_from_current_state(manager_type: type) -> None:
    """Synchronize VBD's persistent state after Arena teleports the scene."""
    state_0 = manager_type.get_state_0()
    state_1 = manager_type.get_state_1()
    solver = manager_type._solver

    # VBD uses two Newton state buffers and its own previous-body transforms.
    # Leaving either at the pre-reset pose turns a reset teleport into a very
    # large inferred velocity on the next solver step.
    state_1.assign(state_0)
    solver.body_q_prev.assign(state_0.body_q)
    state_0.clear_forces()
    state_1.clear_forces()

    # Hard articulation constraints retain history independently of the
    # Newton State objects. Clear it so a reset starts from the authored pose.
    # Contact history and Dahl cable friction are disabled in this environment;
    # their buffers may be allocated inside CUDA graph capture and must not be
    # touched from the eager reset path.
    for attribute_name in (
        "joint_lambda_lin",
        "joint_lambda_ang",
        "joint_C0_lin",
        "joint_C0_ang",
    ):
        value = getattr(solver, attribute_name, None)
        if value is not None:
            value.zero_()

    manager_type._mark_transforms_dirty()


def _get_cable_vbd_manager_type() -> type:
    """Return the VBD manager subclass that owns this environment's builder hooks."""
    global _CABLE_VBD_MANAGER_TYPE
    if _CABLE_VBD_MANAGER_TYPE is None:
        from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

        class NewtonCableRoutingVBDManager(NewtonVBDManager):
            """VBD manager that injects and renders the procedural cable."""

            @classmethod
            def initialize(cls, sim_context) -> None:
                cls._cable_body_state_cache = None
                _install_cable_builder_hooks()
                super().initialize(sim_context)

            @classmethod
            def start_simulation(cls) -> None:
                super().start_simulation()
                _initialize_yam_vbd_rest_pose(cls)
                _restore_cable_initial_body_state(cls)

            @classmethod
            def _build_solver(cls, model, solver_cfg) -> None:
                super()._build_solver(model, solver_cfg)
                for joint_index, joint_label in enumerate(model.joint_label):
                    if isinstance(joint_label, str) and "/Cable/Joint_" in joint_label:
                        cls._solver.set_joint_constraint_mode(joint_index, False)
                cls._yam_body_ids = wp.array(
                    [body_index for body_index, body_label in enumerate(model.body_label) if _is_yam_label(body_label)],
                    dtype=wp.int32,
                    device=model.device,
                )
                gravity = model.gravity.numpy()[0]
                cls._gravity = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))

            @classmethod
            def forward(cls) -> None:
                # Newton 1.2's base forward() runs FK for every articulation. That
                # reconstructs procedural cable joints from zero generalized
                # coordinates and straightens the curved cable at every env reset.
                # The newer manager used upstream applies FK only to articulations
                # dirtied by asset writes, which leaves maximal-coordinate cable
                # bodies untouched.
                from newton import eval_fk

                eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, cls._fk_reset_mask)
                _reset_vbd_history_from_current_state(cls)
                cls._world_reset_mask.zero_()
                cls._fk_reset_mask.zero_()

            @classmethod
            def _initialize_contacts(cls) -> None:
                # Follow Newton 1.2's cable reference with an explicit pipeline
                # and stable contact matching. Contact history is disabled for
                # this soft-contact configuration, but deterministic identities
                # keep this path compatible if history is tuned later.
                from isaaclab_newton.physics import NewtonManager
                from newton import CollisionPipeline

                if cls._collision_pipeline is None:
                    NewtonManager._collision_pipeline = CollisionPipeline(
                        cls._model,
                        broad_phase="explicit",
                        contact_matching="latest",
                    )
                if cls._contacts is None:
                    NewtonManager._contacts = cls._collision_pipeline.contacts()

            @classmethod
            def _run_solver_substeps(cls, contacts) -> None:
                # Arena's Newton 1.2 manager normally reuses one collision query
                # for every solver substep. The upstream task refreshes its VBD
                # proxy contacts every other 1/1200 s substep; reproduce that
                # cadence for this full-scene compatibility solver.
                from isaaclab.physics import PhysicsManager
                from isaaclab_newton.physics import NewtonManager
                from newton import eval_ik

                cfg = PhysicsManager._cfg
                need_copy_on_last = cfg is not None and cfg.use_cuda_graph and cls._num_substeps % 2 == 1
                for substep in range(cls._num_substeps):
                    wp.launch(
                        kernel=_apply_yam_gravity_compensation,
                        dim=cls._yam_body_ids.shape[0],
                        inputs=[cls._yam_body_ids, cls._model.body_mass, cls._gravity, cls._state_0.body_f],
                        device=cls._model.device,
                    )
                    cls._step_solver(cls._state_0, cls._state_1, cls._control, contacts, cls._solver_dt)
                    if need_copy_on_last and substep == cls._num_substeps - 1:
                        cls._state_0.assign(cls._state_1)
                    else:
                        NewtonManager._state_0, NewtonManager._state_1 = cls._state_1, cls._state_0
                    cls._state_0.clear_forces()
                    if (
                        contacts is not None
                        and (substep + 1) % _COLLISION_SUBSTEP_INTERVAL == 0
                        and substep + 1 < cls._num_substeps
                    ):
                        cls._collision_pipeline.collide(cls._state_0, contacts)

                # VBD advances maximal body coordinates. Recover generalized
                # joint coordinates so the next actuator update, observations,
                # and relative policy actions see the robot's actual state.
                eval_ik(cls._model, cls._state_0, cls._state_0.joint_q, cls._state_0.joint_qd)
                cls._state_1.joint_q.assign(cls._state_0.joint_q)
                cls._state_1.joint_qd.assign(cls._state_0.joint_qd)

            @classmethod
            def _simulate_physics_only(cls) -> None:
                # Newton 1.2's contrib manager calls rebuild_bvh() even for a
                # rigid rod model with no particles. SolverVBD does not create
                # its particle self-contact fields in that case.
                if cls._model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
                    cls._solver.rebuild_bvh(cls._state_0)

                from isaaclab_newton.physics import NewtonManager

                NewtonManager._simulate_physics_only.__func__(cls)

            @classmethod
            def _solver_specific_clear(cls) -> None:
                cls._cable_body_state_cache = None
                _clear_cable_builder_hooks()
                super()._solver_specific_clear()

        _CABLE_VBD_MANAGER_TYPE = NewtonCableRoutingVBDManager
    return _CABLE_VBD_MANAGER_TYPE


def _configure_cable_routing_newton_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure upstream timing with the Newton 1.2 VBD compatibility path."""
    from isaaclab.utils.configclass import configclass
    from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
    from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    assert env_cfg.scene.num_envs == 1, "The initial cable-routing compatibility environment supports one env only."

    @configclass
    class CableRoutingVBDSolverCfg(VBDSolverCfg):
        """Expose Newton 1.2 contact buffers and select the cable-aware manager."""

        class_type: type = _get_cable_vbd_manager_type()
        rigid_contact_hard: bool = False
        rigid_contact_history: bool = False
        rigid_body_contact_buffer_size: int = 512
        rigid_body_particle_contact_buffer_size: int = 256

    @configclass
    class CableRoutingShapeCfg(NewtonShapeCfg):
        """Forward the upstream contact material values to Newton's ShapeConfig."""

        ke: float = _CONTACT_STIFFNESS
        kd: float = _CONTACT_DAMPING
        mu: float = _CABLE_CONTACT_FRICTION

    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.sim.render_interval = 4
    env_cfg.sim.use_newton_actuators = True
    env_cfg.sim.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=_FIXTURE_CONTACT_FRICTION,
        dynamic_friction=_FIXTURE_CONTACT_FRICTION,
        restitution=0.0,
    )
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=CableRoutingVBDSolverCfg(iterations=12),
        default_shape_cfg=CableRoutingShapeCfg(margin=0.0, gap=_CONTACT_GAP),
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
    from isaaclab_newton.sim.schemas import NewtonArticulationRootPropertiesCfg, NewtonCollisionPropertiesCfg

    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path="/Geometry/arm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_YAM_USD_PATH),
            copy_from_source=False,
            articulation_props=NewtonArticulationRootPropertiesCfg(self_collision_enabled=False),
            collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=_CONTACT_GAP),
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
                stiffness=40.0,
                damping=2.5 / 40.0,
            ),
            "arm_joint_4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=10.0,
                stiffness=20.0,
                damping=0.5 / 20.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-6]"],
                effort_limit_sim=10.0,
                stiffness=10.0,
                damping=1.0 / 10.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_finger", "right_finger"],
                stiffness=100.0,
                damping=10.0 / 100.0,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )


def _make_bimanual_yam_embodiment(enable_cameras: bool = False):
    """Create the two-YAM Arena embodiment and its 14-dimensional action interface."""
    import torch

    import isaaclab.envs.mdp as mdp
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs.mdp.actions import BinaryJointPositionAction, RelativeJointPositionAction
    from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    class FiniteRelativeJointPositionAction(RelativeJointPositionAction):
        """Resolve one finite, limit-clamped absolute target per policy step."""

        def process_actions(self, actions: torch.Tensor) -> None:
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            super().process_actions(actions)
            current = self._asset.data.joint_pos.torch[:, self._joint_ids]
            default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
            current = torch.where(torch.isfinite(current), current, default)
            limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
            target = current + self._processed_actions
            target = torch.where(torch.isfinite(target), target, default)
            self._processed_actions = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])

        def apply_actions(self) -> None:
            self._asset.set_joint_position_target_index(target=self.processed_actions, joint_ids=self._joint_ids)

    class FiniteBinaryJointPositionAction(BinaryJointPositionAction):
        """Prevent non-finite gripper commands from reaching physics."""

        def process_actions(self, actions: torch.Tensor) -> None:
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            super().process_actions(actions)

    @configclass
    class FiniteRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
        class_type: type = FiniteRelativeJointPositionAction

    @configclass
    class FiniteBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
        class_type: type = FiniteBinaryJointPositionAction

    def arm_action_cfg(asset_name: str) -> FiniteRelativeJointPositionActionCfg:
        return FiniteRelativeJointPositionActionCfg(
            asset_name=asset_name,
            joint_names=["joint[1-6]"],
            scale=0.04,
            use_zero_offset=True,
            preserve_order=True,
        )

    def gripper_action_cfg(asset_name: str) -> FiniteBinaryJointPositionActionCfg:
        return FiniteBinaryJointPositionActionCfg(
            asset_name=asset_name,
            joint_names=["left_finger", "right_finger"],
            open_command_expr={
                "left_finger": _YAM_GRIPPER_OPEN_POS,
                "right_finger": -_YAM_GRIPPER_OPEN_POS,
            },
            close_command_expr={
                "left_finger": _YAM_GRIPPER_CLOSED_POS,
                "right_finger": -_YAM_GRIPPER_CLOSED_POS,
            },
        )

    @configclass
    class BimanualYamSceneCfg:
        robot: ArticulationCfg | None = None
        yam_right: ArticulationCfg | None = None

    @configclass
    class BimanualYamActionsCfg:
        left_arm = arm_action_cfg("robot")
        left_gripper = gripper_action_cfg("robot")
        right_arm = arm_action_cfg("yam_right")
        right_gripper = gripper_action_cfg("yam_right")

    @configclass
    class BimanualYamObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            left_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            left_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            right_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("yam_right")})
            right_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("yam_right")})
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
                robot=_make_yam_cfg(
                    "{ENV_REGEX_NS}/YamLeft",
                    (_YAM_FRONT_X, _YAM_LATERAL_OFFSET, _YAM_BASE_Z),
                    0.0,
                ),
                yam_right=_make_yam_cfg(
                    "{ENV_REGEX_NS}/YamRight",
                    (_YAM_FRONT_X, -_YAM_LATERAL_OFFSET, _YAM_BASE_Z),
                    0.0,
                ),
            )
            self.action_config = BimanualYamActionsCfg()
            self.observation_config = BimanualYamObservationsCfg()

        def get_command_body_name(self) -> str:
            return "link_6"

        def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
            return "link_6"

    return BimanualYamEmbodiment()


def _make_cable_routing_task():
    """Create the minimal scene, reset, route-success, and timeout task."""
    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
        table: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * _TABLE_TOP_Z)),
            spawn=sim_utils.CuboidCfg(
                size=(1.10, 0.80, _TABLE_TOP_Z),
                rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
                collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=_CONTACT_GAP),
                physics_material=fixture_material(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.23, 0.16)),
            ),
        )
        board: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Board",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _TABLE_TOP_Z + 0.5 * _BOARD_THICKNESS)),
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
            spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.85, 0.85, 0.85)),
        )

    @configclass
    class CableRoutingEventsCfg:
        reset_scene = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )
        reset_cable = EventTerm(func=_reset_cable_to_initial_state, mode="reset")

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
            return ViewerCfg(eye=(1.25, -1.10, 1.55), lookat=(0.0, 0.0, _BOARD_TOP_Z))

    return CableRoutingTask()


@dataclass
class CableRoutingYamNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the single-environment YAM cable-routing prototype."""


@register_environment
class CableRoutingYamNewtonEnvironment(ArenaEnvironmentFactory[CableRoutingYamNewtonEnvironmentCfg]):
    """Build the self-contained two-YAM Newton cable-routing prototype."""

    name = "cable_routing_yam_newton"
    _legacy_argparse_cfg_type = CableRoutingYamNewtonEnvironmentCfg

    def build(self, cfg: CableRoutingYamNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment using the locally bundled upstream assets."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        for asset_path in (_YAM_USD_PATH, _BOARD_USD_PATH, _ROUND_PEG_USD_PATH):
            assert asset_path.is_file(), f"Cable-routing asset not found: {asset_path}"

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=_make_bimanual_yam_embodiment(enable_cameras=cfg.enable_cameras),
            scene=Scene(assets=[]),
            task=_make_cable_routing_task(),
            env_cfg_callback=_configure_cable_routing_newton_physics,
        )
