# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current CAP cable physics using Arena's native cable import path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import TYPE_CHECKING, Literal

import warp as wp
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.coupling import CouplerProxyCfg, CouplerProxyMappingCfg, NewtonCouplerManager
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, VBDSolverCfg

from .scene import CableBuilderPhysics, CableCouplerProxy, CablePhysics, CableSolverExtensions

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

_ACTIVE_PHYSICS: CablePhysics | None = None
_ACTIVE_BUILDER_PHYSICS: CableBuilderPhysics | None = None


@configclass
class CableRoutingVBDSolverCfg(VBDSolverCfg):
    """VBD controls used by CAP but not yet declared by Isaac Lab's config."""

    rigid_avbd_beta: float = 1.0e2
    rigid_contact_history: bool = False
    rigid_body_contact_buffer_size: int = 256


@configclass
class CableRoutingMJWarpSolverCfg(MJWarpSolverCfg):
    """Expose CAP's sparse-Jacobian option on the pinned Isaac Lab API."""

    jacobian: str = "sparse"


@configclass
class CableRoutingCollisionPipelineCfg(NewtonCollisionPipelineCfg):
    """Expose CAP's contact-matching option on the pinned Isaac Lab API."""

    contact_matching: Literal["disabled", "latest", "sticky"] = "latest"


def _configure_native_cable_builder(builder, world_index: int, *_unused) -> None:
    """Apply CAP's cable friction and fixed-start constraint after native import."""
    physics = _ACTIVE_PHYSICS
    builder_physics = _ACTIVE_BUILDER_PHYSICS
    assert physics is not None, "Cable physics must be selected before Newton imports the scene."
    assert builder_physics is not None, "Cable builder physics must be selected before Newton imports the scene."
    active_world = builder.current_world
    assert active_world in (
        -1,
        world_index,
    ), f"Expected Newton world {world_index}, but the active builder world is {active_world}."

    cable_shapes = []
    for shape_index, (label, shape_world) in enumerate(zip(builder.shape_label, builder.shape_world)):
        if not isinstance(label, str) or int(shape_world) != active_world:
            continue
        cable_path, separator, suffix = label.rpartition("_edge_capsule_")
        if separator and suffix.isdigit() and cable_path.endswith("/Cable/geometry/mesh"):
            cable_shapes.append((int(suffix), shape_index, cable_path))
    cable_shapes.sort()
    assert cable_shapes, f"Unable to find the Arena cable shapes in Newton world {world_index}."
    cable_path = cable_shapes[0][2]
    assert all(
        path == cable_path for _, _, path in cable_shapes
    ), f"Expected one Arena cable in Newton world {world_index}."

    for _, shape_index, _ in cable_shapes:
        builder.shape_material_mu[shape_index] = physics.cable_friction
        builder.shape_material_ke[shape_index] = physics.contact_stiffness
        builder.shape_material_kd[shape_index] = physics.contact_damping
        builder.shape_gap[shape_index] = physics.contact_gap
        builder.shape_margin[shape_index] = 0.0

    # CableMaterialCfg carries elastic moduli but has no damping fields. Restore
    # CAP's direct per-joint rod constants after Arena's native USD import.
    cable_joint_prefix = f"{cable_path}_cable_"
    cable_joint_indices = [
        index
        for index, label in enumerate(builder.joint_label)
        if isinstance(label, str)
        and label.startswith(cable_joint_prefix)
        and int(builder.joint_world[index]) == active_world
    ]
    assert (
        len(cable_joint_indices) == len(cable_shapes) - 1
    ), f"Expected {len(cable_shapes) - 1} Arena cable joints, got {len(cable_joint_indices)}."
    stiffness = (
        physics.stretch_stiffness,
        physics.stretch_stiffness,
        physics.bend_stiffness,
        physics.bend_stiffness,
    )
    damping = (
        physics.stretch_damping,
        physics.stretch_damping,
        physics.bend_damping,
        physics.bend_damping,
    )
    for joint_index in cable_joint_indices:
        assert tuple(builder.joint_dof_dim[joint_index]) == (
            2,
            2,
        ), f"Unexpected Arena cable joint layout: {builder.joint_dof_dim[joint_index]}."
        dof_start = int(builder.joint_qd_start[joint_index])
        builder.joint_target_ke[dof_start : dof_start + 4] = stiffness
        builder.joint_target_kd[dof_start : dof_start + 4] = damping

    if not builder_physics.pin_start:
        return
    first_shape_index = cable_shapes[0][1]
    first_body_index = int(builder.shape_body[first_shape_index])
    pose = builder.body_q[first_body_index]
    anchor_label = f"{cable_path.rsplit('/geometry/mesh', 1)[0]}/Anchor"
    if anchor_label in builder.joint_label:
        return
    builder.add_joint_fixed(
        parent=-1,
        child=first_body_index,
        parent_xform=wp.transform(
            wp.vec3(*[float(value) for value in pose[:3]]),
            wp.quat(*[float(value) for value in pose[3:7]]),
        ),
        child_xform=wp.transform_identity(),
        label=anchor_label,
    )


class NewtonArenaCableRoutingCouplerManager(NewtonCouplerManager):
    """Install only the task-specific pin/material hook around Arena Cable."""

    _headless_contact_history_warmed = False

    @classmethod
    def initialize(cls, sim_context) -> None:
        from isaaclab_newton.physics import NewtonManager

        cls._headless_contact_history_warmed = False
        if _configure_native_cable_builder not in NewtonManager._per_world_builder_hooks:
            NewtonManager._per_world_builder_hooks.append(_configure_native_cable_builder)
        super().initialize(sim_context)

    @classmethod
    def _capture_or_defer_graph(cls) -> None:
        """Warm VBD contact history before immediate headless graph capture."""
        from isaaclab.physics import PhysicsManager

        cfg = PhysicsManager._cfg
        device = PhysicsManager._device
        needs_headless_warmup = (
            not cls._headless_contact_history_warmed
            and cfg is not None
            and cfg.use_cuda_graph
            and device is not None
            and "cuda" in device
            and cls._usdrt_stage is None
        )
        if needs_headless_warmup:
            # RTX capture already performs an eager solver warmup. Standard
            # headless capture does not, but SolverVBD requires its persistent
            # contact-history arrays to exist before capture begins.
            with wp.ScopedDevice(device):
                cls._simulate_physics_only()
            wp.synchronize_stream(wp.get_stream(device))
            cls._headless_contact_history_warmed = True
        super()._capture_or_defer_graph()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        from isaaclab_newton.physics import NewtonManager

        global _ACTIVE_PHYSICS, _ACTIVE_BUILDER_PHYSICS
        cls._headless_contact_history_warmed = False
        if hasattr(NewtonManager, "_per_world_builder_hooks"):
            NewtonManager._per_world_builder_hooks = [
                hook for hook in NewtonManager._per_world_builder_hooks if hook is not _configure_native_cable_builder
            ]
        _ACTIVE_PHYSICS = None
        _ACTIVE_BUILDER_PHYSICS = None
        super()._solver_specific_clear()


def configure_cable_routing_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    physics: CablePhysics,
    builder_physics: CableBuilderPhysics,
    solver_extensions: CableSolverExtensions,
    coupler_proxy: CableCouplerProxy,
    env_cfg_override: dict,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply declarative physics and install the remaining procedural-cable hook."""
    from isaaclab_arena.environment_spec.env_cfg_override import apply_env_cfg_override

    assert env_cfg.scene.num_envs == 1, "The current cable-routing compatibility environment supports one env only."
    apply_env_cfg_override(env_cfg, env_cfg_override)

    newton_cfg = env_cfg.sim.physics
    assert isinstance(newton_cfg, NewtonCfg), "Cable routing requires declarative Newton physics."
    coupler_cfg = newton_cfg.solver_cfg
    assert isinstance(coupler_cfg, CouplerProxyCfg), "Cable routing requires a proxy coupler."
    base_collision_cfg = newton_cfg.collision_cfg
    assert isinstance(
        base_collision_cfg, NewtonCollisionPipelineCfg
    ), "Cable routing requires a Newton collision pipeline."
    collision_values = {
        field.name: deepcopy(getattr(base_collision_cfg, field.name))
        for field in fields(NewtonCollisionPipelineCfg)
        if field.init
    }
    collision_cfg = CableRoutingCollisionPipelineCfg(
        **collision_values,
        contact_matching=solver_extensions.contact_matching,
    )
    newton_cfg.collision_cfg = collision_cfg
    # CouplerProxyMappingCfg currently cannot be materialized through Arena's
    # YAML override resolver because Isaac Lab's ModelView type is import-only.
    coupler_cfg.proxies = [
        CouplerProxyMappingCfg(
            source=coupler_proxy.source,
            destination=coupler_proxy.destination,
            bodies=list(coupler_proxy.bodies),
            mode=coupler_proxy.mode,
            mass_scale=coupler_proxy.mass_scale,
            collide_interval=coupler_proxy.collide_interval,
            collision_pipeline=collision_cfg,
        )
    ]
    cable_entries = [entry for entry in coupler_cfg.entries if entry.name == "cable"]
    assert len(cable_entries) == 1, "Cable routing requires exactly one cable solver entry."
    base_vbd_cfg = cable_entries[0].solver_cfg
    assert isinstance(base_vbd_cfg, VBDSolverCfg), "The cable solver entry must use VBD."
    base_values = {
        field.name: deepcopy(getattr(base_vbd_cfg, field.name)) for field in fields(VBDSolverCfg) if field.init
    }
    cable_entries[0].solver_cfg = CableRoutingVBDSolverCfg(
        **base_values,
        rigid_avbd_beta=builder_physics.rigid_avbd_beta,
        rigid_contact_history=builder_physics.rigid_contact_history,
        rigid_body_contact_buffer_size=builder_physics.rigid_body_contact_buffer_size,
    )
    rigid_entries = [entry for entry in coupler_cfg.entries if entry.name == "rigid"]
    assert len(rigid_entries) == 1, "Cable routing requires exactly one rigid solver entry."
    base_rigid_cfg = rigid_entries[0].solver_cfg
    assert isinstance(base_rigid_cfg, MJWarpSolverCfg), "The rigid solver entry must use MuJoCo Warp."
    rigid_values = {
        field.name: deepcopy(getattr(base_rigid_cfg, field.name))
        for field in fields(MJWarpSolverCfg)
        if field.init and field.name != "class_type"
    }
    rigid_entries[0].solver_cfg = CableRoutingMJWarpSolverCfg(
        **rigid_values,
        jacobian=solver_extensions.rigid_jacobian,
    )

    # The custom manager only installs and removes the procedural Cable import hook.
    coupler_cfg.class_type = NewtonArenaCableRoutingCouplerManager
    newton_cfg.class_type = NewtonArenaCableRoutingCouplerManager
    global _ACTIVE_PHYSICS, _ACTIVE_BUILDER_PHYSICS
    _ACTIVE_PHYSICS = physics
    _ACTIVE_BUILDER_PHYSICS = builder_physics
    return env_cfg


__all__ = ["configure_cable_routing_physics"]
