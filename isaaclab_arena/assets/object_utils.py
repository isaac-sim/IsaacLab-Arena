# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from pxr import Usd

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.usd.helpers import get_prim_depth, is_articulation_root, is_rigid_body, open_usd_file
from isaaclab_arena.utils.usd.variants import apply_usd_variant_selections


def detect_object_type(
    usd_path: str | None = None,
    stage: Usd.Stage | None = None,
    variants: object | dict[str, str] | None = None,
) -> ObjectType:
    """Detect the object type of the asset

    Goes through the USD tree and detects the object type. The detection is based
    on the presence of a RigidBodyAPI or ArticulationRootAPI at the shallowest depth
    in which one of these APIs is present.

    Note that if more than one API is present on that shallowest depth, we raise an error.

    Args:
        usd_path: The path to the USD file to inspect. Either this or stage must be provided.
        stage: The stage to inspect. Either this or usd_path must be provided.
        variants: A dictionary or config class mapping variant set names to the variants to select before
            looking at the asset. SimReady props need ``{"Physics": "physics"}``, or they have no physics at all.

    Returns:
        The object type of the asset.
    """
    assert usd_path is not None or stage is not None, "Either usd_path or stage must be provided"
    assert usd_path is None or stage is None, "Either usd_path or stage must be provided"
    if usd_path is not None:
        # Open a stage to inspect the USD.
        stage = open_usd_file(usd_path)
    # Select variants in the session layer, so the asset's layer, which other open stages share, stays unedited.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        apply_usd_variant_selections(stage, variants)
    # We do a Breadth First Search (BFS) through the prims, until we find either
    # a rigid body or an articulation root. At that point, we continue searching
    # the rest of the prims at that depth, to ensure that there's nothing else.
    # If we find more than one, we raise an error.
    open_prims = [stage.GetPseudoRoot()]
    found = False
    found_depth = -1
    interesting_prim = None
    while len(open_prims) > 0:
        # Update the DFS list
        prim = open_prims.pop(0)
        open_prims.extend(prim.GetChildren())
        if found and get_prim_depth(prim) > found_depth:
            break
        # Check if we found an interesting prim on this level
        if is_articulation_root(prim) or is_rigid_body(prim):
            if found:
                raise ValueError(f"Found multiple rigid body or articulation roots at depth {get_prim_depth(prim)}")
            found_depth = get_prim_depth(prim)
            found = True
            interesting_prim = prim
    if not found:
        return ObjectType.BASE
    # PhysX simulates a rigid body that is also an articulation root as the articulation's root link.
    if is_articulation_root(interesting_prim):
        return ObjectType.ARTICULATION
    if is_rigid_body(interesting_prim):
        return ObjectType.RIGID
    else:
        raise ValueError("This should not happen. There is an unknown USD type in the tree.")


# Predefined rigid body property configurations for assembly tasks
# High iteration count for precision tasks (peg/hole insertion)
RIGID_BODY_PROPS_HIGH_PRECISION = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)

# Standard iteration count for gear mesh tasks
RIGID_BODY_PROPS_MEDIUM_PRECISION = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=32,
    max_contact_impulse=1e32,
)

# Initial state configuration for articulations without joints (e.g., rigid bodies treated as articulations).
# We explicitly set joint_pos and joint_vel to empty dicts to avoid the default pattern {".*": 0.0} in ArticulationCfg.InitialStateCfg,
# which would fail to match when there are no joints in the articulation.
EMPTY_ARTICULATION_INIT_STATE_CFG = ArticulationCfg.InitialStateCfg(
    joint_pos={},
    joint_vel={},
)


def get_passive_joint_actuators(
    joint_name: str,
    actuator_name: str = "passive_joint",
    armature: float = 0.001,
    friction: float = 0.1,
) -> dict[str, ImplicitActuatorCfg]:
    """Return an actuator group that leaves a joint undriven and frictional, for an articulation's ``actuators``.

    Armature adds joint-space inertia so contacts do not fling a light link, and static friction holds the joint
    where it is left.

    Args:
        joint_name: Name or regular expression of the joint.
        actuator_name: Key of the actuator group.
        armature: Added joint inertia in kg m^2 for a revolute joint.
        friction: Static and dynamic joint friction in N m for a revolute joint.

    Returns:
        A mapping from actuator_name to the actuator config.
    """
    actuator = ImplicitActuatorCfg(
        joint_names_expr=[joint_name],
        stiffness=0.0,
        damping=0.0,
        armature=armature,
        friction=friction,
        dynamic_friction=friction,
    )
    return {actuator_name: actuator}


def get_joint_drive_props(
    joint_name: str,
    stiffness: float = 20.0,
    damping: float = 2.0,
    max_force: float = 10.0,
) -> dict[str, list[sim_utils.UsdPhysicsDriveCfg]]:
    """Return spawner joint_drive_props that author a force drive towards the joint's zero position.

    The drive is written to the spawned USD, so it also applies when the source asset authors no drive, or authors
    it on an instance PhysX ignores.

    Args:
        joint_name: Name of the joint, matched anywhere below the spawned prim.
        stiffness: Drive stiffness in N m/rad for a revolute joint.
        damping: Drive damping in N m s/rad for a revolute joint.
        max_force: Maximum drive torque in N m for a revolute joint.

    Returns:
        A mapping from the joint's prim path pattern to its drive config.
    """
    drive = sim_utils.UsdPhysicsDriveCfg(drive_type="force", max_force=max_force, stiffness=stiffness, damping=damping)
    return {f"/.*/{joint_name}": [drive]}
