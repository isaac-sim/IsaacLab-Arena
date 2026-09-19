# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP USB-C solver, full-hand contacts, and actuator tuning."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonMJWarpManager

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


# TODO(xinjieyao, 09/17/2026): Remove this customization once the MR to apply physics to spawn cfg is merged.


def _set_contact_attributes(builder, shapes, *, condim=None) -> None:
    import warp as wp
    from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
    from newton._src.solvers.mujoco.solver_mujoco import vec5

    attributes = {
        "mujoco:solref": wp.vec2(0.004, 1.0),
        "mujoco:solref_mode": SOLREF_MODE_RAW,
        "mujoco:geom_solimp": vec5(0.95, 0.999, 0.0005, 0.5, 2.0),
    }
    if condim is not None:
        attributes["mujoco:condim"] = condim
    for name, value in attributes.items():
        attribute = builder.custom_attributes[name]
        if attribute.values is None:
            attribute.values = {}
        for index in shapes:
            attribute.values[index] = value


def _configure_contacts(_event_payload=None) -> None:
    """Apply CAP's matched contacts and passive-jaw coupling before model creation."""
    import newton
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    assert builder is not None, "USB-C contacts require a Newton builder."
    equality_joints = builder.custom_attributes["mujoco:equality_constraint_joint1"].values
    equality_solref = builder.custom_attributes["mujoco:eq_solref"]
    if equality_solref.values is None:
        equality_solref.values = {}
    coupled_fingers = 0
    for index, joint in enumerate(equality_joints):
        if int(joint) >= 0 and "finger" in str(builder.joint_label[int(joint)]).casefold():
            equality_solref.values[index] = wp.vec2(0.004, 1.0)
            coupled_fingers += 1
    assert coupled_fingers, "USB-C contact rig found no YAM finger equalities."

    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for robot_token in ("leftrobot", "rightrobot"):
        finger_bodies = {
            index
            for index, label in enumerate(builder.body_label)
            if robot_token in str(label).casefold() and "finger" in str(label).casefold()
        }
        fingers = [index for index, body in enumerate(builder.shape_body) if int(body) in finger_bodies]
        references = [index for index in fingers if int(builder.shape_flags[index]) & collide]
        assert references, f"USB-C contact rig found no colliding fingers for {robot_token}."
        reference = references[0]
        excluded = {
            other
            for pair in builder._shape_collision_filter_pairs
            if reference in pair
            for other in pair
            if other != reference
        }
        housings = []
        for index, body in enumerate(builder.shape_body):
            if int(body) < 0:
                continue
            label = str(builder.body_label[int(body)]).casefold()
            if robot_token not in label or label.rsplit("/", 1)[-1] != "link_6" or builder.shape_source[index] is None:
                continue
            builder.shape_flags[index] |= collide
            builder.shape_collision_group[index] = builder.shape_collision_group[reference]
            for other in {reference, *excluded}:
                if other != index:
                    builder.add_shape_collision_filter_pair(index, other)
            housings.append(index)
        for index in fingers:
            builder.shape_material_mu[index] = 8.0
            builder.shape_material_mu_torsional[index] = 0.002
            builder.shape_material_mu_rolling[index] = 0.0001
            builder.shape_gap[index] = 0.0002
        _set_contact_attributes(builder, fingers, condim=4)
        _set_contact_attributes(builder, housings, condim=3)

    connectors = []
    for index, raw_label in enumerate(builder.shape_label):
        label = str(raw_label).casefold()
        path_components = set(label.split("/"))
        if path_components & {"plug", "port", "bulkhead"}:
            builder.shape_material_ke[index] = 62500.0
            builder.shape_material_kd[index] = 500.0
            builder.shape_material_mu[index] = 2.5 if "bulkhead" in path_components else 0.35
            connectors.append(index)
        elif "bench" in path_components:
            builder.shape_material_mu[index] = 0.4
        elif "table" in path_components:
            builder.shape_material_mu[index] = 0.35
    assert len(connectors) >= 2, "USB-C contact rig could not find both connector meshes."
    _set_contact_attributes(builder, connectors)


class NewtonUsbcManager(NewtonMJWarpManager):
    """Install USB-C contact tuning only when this task's manager is initialized."""

    @classmethod
    def initialize(cls, sim_context) -> None:
        from isaaclab.physics import PhysicsEvent
        from isaaclab_newton.physics import NewtonManager

        NewtonManager.register_callback(
            _configure_contacts, PhysicsEvent.MODEL_INIT, name="arena_usbc_contacts", wrap_weak_ref=False
        )
        super().initialize(sim_context)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        from .cables import _remove_connector_cable_builder_hooks

        _remove_connector_cable_builder_hooks()
        super()._solver_specific_clear()


def configure_usbc_runtime(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    apply_graph_override: Callable[[IsaacLabArenaManagerBasedRLEnvCfg], IsaacLabArenaManagerBasedRLEnvCfg],
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply the graph override, then install USB-C runtime-only tuning."""
    env_cfg = apply_graph_override(env_cfg)
    env_cfg.sim.physics.class_type = NewtonUsbcManager
    env_cfg.scene.replicate_physics = False
    for robot in (env_cfg.scene.left_robot, env_cfg.scene.right_robot):
        robot.init_state.joint_pos.update(joint2=1.047, joint3=1.047)
        for name, actuator in robot.actuators.items():
            if name.startswith("arm_"):
                actuator.stiffness = 1600.0
                actuator.damping = 70.0
                actuator.effort_limit_sim = 28.0 if name == "arm_joints_1_3" else 10.0
        robot.actuators["gripper"].stiffness = 40000.0
        robot.actuators["gripper"].damping = 40.0
        robot.actuators["gripper"].effort_limit_sim = 160.0
    return env_cfg
