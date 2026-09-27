# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# TODO(alexmillane) [physics-parameters-overrides-missing-feature]: Remove this file once we can
# control the physics parameters in the yaml files.

"""Native Newton coupling configuration for Isaac Cap cable routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

CONTACT_STIFFNESS = 4.0e4
CONTACT_DAMPING = 1.0e-5
CONTACT_GAP = 0.001
CABLE_CONTACT_FRICTION = 0.1
FIXTURE_CONTACT_FRICTION = 0.5
COLLISION_SUBSTEP_INTERVAL = 2


def _replace_easy_cable_capsules_with_boxes(builder, world_index: int, *_unused) -> None:
    """Give one replicated easy cable square collision links.

    ***IMPORTANT: This is added by Isaac Cap to replace the cable with square collision links.
    This is not generally supported by Lab and so will not be added to Arena.
    It's added here to re-create the cap env to sure it works in Arena v0.3.1.
    """
    from newton import GeoType, ShapeFlags

    active_world = builder.current_world
    assert active_world in (
        -1,
        world_index,
    ), f"Expected Newton world {world_index}, but the active builder world is {active_world}."

    capsule_shapes = []
    for shape_index, (label, shape_world) in enumerate(zip(builder.shape_label, builder.shape_world)):
        if not isinstance(label, str) or int(shape_world) != active_world:
            continue
        cable_path, separator, suffix = label.rpartition("_edge_capsule_")
        if separator and suffix.isdigit() and cable_path.endswith("/Cable/geometry/mesh"):
            capsule_shapes.append((int(suffix), shape_index, cable_path))

    capsule_shapes.sort()
    assert capsule_shapes, f"Unable to find cable capsule shapes in Newton world {world_index}."
    assert [segment for segment, _, _ in capsule_shapes] == list(
        range(len(capsule_shapes))
    ), f"Cable capsule labels in Newton world {world_index} are not contiguous."
    assert (
        len({cable_path for _, _, cable_path in capsule_shapes}) == 1
    ), f"Expected one cable in Newton world {world_index}."

    # NOTE: CableCfg currently generates round capsule colliders. Those let the
    # easy variant's initially straight cable coherently roll off the board. Keep
    # the capsules for native cable rendering, but replace only their collision
    # role with the square links used by the original Isaac Cap environment.
    for _, shape_index, _ in capsule_shapes:
        assert (
            builder.shape_type[shape_index] == GeoType.CAPSULE
        ), f"Expected cable shape {shape_index} to be a capsule."
        builder.shape_flags[shape_index] &= ~int(ShapeFlags.COLLIDE_SHAPES)

    box_cfg = builder.default_shape_cfg.copy()
    box_cfg.density = 0.0
    box_cfg.is_visible = False
    box_cfg.collision_filter_parent = True
    for segment, shape_index, cable_path in capsule_shapes:
        radius, half_length, _ = builder.shape_scale[shape_index]
        assert radius > 0.0 and half_length > 0.0, f"Cable shape {shape_index} has invalid dimensions."
        builder.add_shape_box(
            body=builder.shape_body[shape_index],
            xform=builder.shape_transform[shape_index],
            hx=float(radius),
            hy=float(radius),
            hz=float(half_length),
            cfg=box_cfg,
            label=f"{cable_path}_edge_box_{segment}",
        )


class NewtonEasyCableRoutingCouplerManager(NewtonCouplerManager):
    """Install the easy cable's Cap-compatible collision geometry."""

    @classmethod
    def initialize(cls, sim_context) -> None:
        """Install the builder hook before Newton imports and replicates the scene."""
        from isaaclab_newton.physics import NewtonManager

        if _replace_easy_cable_capsules_with_boxes not in NewtonManager._per_world_builder_hooks:
            NewtonManager._per_world_builder_hooks.append(_replace_easy_cable_capsules_with_boxes)
        super().initialize(sim_context)


def make_fixture_material():
    """Create the Newton material shared by cable-routing fixtures."""
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    return NewtonMaterialPropertiesCfg(
        static_friction=FIXTURE_CONTACT_FRICTION,
        dynamic_friction=FIXTURE_CONTACT_FRICTION,
        restitution=0.0,
        contact_stiffness=CONTACT_STIFFNESS,
        contact_damping=CONTACT_DAMPING,
    )


def configure_cable_routing_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure public MJWarp-to-VBD proxy coupling for the task."""
    return _configure_cable_routing_physics(env_cfg, use_square_cable_colliders=False)


def configure_easy_cable_routing_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure coupling and retain Cap's stable square easy-cable colliders."""
    return _configure_cable_routing_physics(env_cfg, use_square_cable_colliders=True)


def _configure_cable_routing_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    use_square_cable_colliders: bool,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure MJWarp-to-VBD proxy coupling for one cable variant."""
    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg, VBDSolverCfg

    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.sim.render_interval = 4
    env_cfg.sim.use_newton_actuators = True
    env_cfg.sim.physics_material = make_fixture_material()
    coupler_cfg = CouplerProxyCfg(
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
                    r"/World/envs/env_.*/LeftRobot",
                    r"/World/envs/env_.*/RightRobot",
                    r"/World/envs/env_.*/Board",
                    r"/World/envs/env_.*/Peg(0|1|2)",
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
                    r"/World/envs/env_.*/(LeftRobot|RightRobot)/Geometry/arm/"
                    r"link_1/link_2/link_3/link_4/link_5/link_6",
                    r"/World/envs/env_.*/Board",
                    r"/World/envs/env_.*/Peg(0|1|2)",
                ],
                mode="lagged",
                mass_scale=1.0,
                collide_interval=COLLISION_SUBSTEP_INTERVAL,
            )
        ],
        iterations=1,
    )
    if use_square_cable_colliders:
        coupler_cfg.class_type = NewtonEasyCableRoutingCouplerManager
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=coupler_cfg,
        default_shape_cfg=NewtonShapeCfg(
            ke=CONTACT_STIFFNESS,
            kd=CONTACT_DAMPING,
            mu=CABLE_CONTACT_FRICTION,
            margin=0.0,
            gap=CONTACT_GAP,
        ),
        num_substeps=10,
        use_cuda_graph=True,
        debug_mode=False,
    )
    env_cfg.decimation = 4
    return env_cfg
