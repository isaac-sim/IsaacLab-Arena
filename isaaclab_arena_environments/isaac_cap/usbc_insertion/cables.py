# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedural cable loads attached to the USB-C connectors.

Why not use Isaac Lab's ``CableObject``? It requires a standalone, unwelded cable
articulation stepped by the VBD solver. This task instead runs in MJWarp and must
joint the first cable link directly to the dynamic plug or bulkhead so cable loads
act on the connector. That attached topology requires this custom implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonMJWarpManager

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_asset

_PLUG_LINKS = 16
_PLUG_LENGTH = 0.12 * 2.0
_PLUG_REAR_Z = -0.034
_BULKHEAD_LINKS = 8
_BULKHEAD_CABLE_LENGTH = 0.05 * 2.0
_CABLE_RADIUS = 0.0016 * 2.0
_CABLE_DENSITY = 450.0

_CABLE_MU = 0.4
_CABLE_ARMATURE = 2.0e-5
_CABLE_JOINT_FRICTION = 2.0e-3
_CABLE_CONTACT_GAP = 2.0e-4
_CABLE_COLOR = (0.90, 0.90, 0.90)
_CABLE_CONTACT_STIFFNESS = 1.0e3
_CABLE_CONTACT_DAMPING = 300.0

_BULKHEAD_CABLE_ANCHOR_POSITION = (0.0615, 0.0, 0.0)
_BULKHEAD_CABLE_ANCHOR_ROTATION = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)


def _body_index(builder, prim_path: str) -> int:
    """Resolve exactly one connector body within its environment subtree."""
    hits = [
        index
        for index, label in enumerate(builder.body_label)
        if str(label) == prim_path or str(label).startswith(f"{prim_path}/")
    ]
    assert len(hits) == 1, f"USB-C cable connector {prim_path!r} matched {len(hits)} bodies."
    return hits[0]


def _spawn_connector_cable(prim_path, cfg, translation=None, orientation=None):
    """Register this asset's deferred Newton construction and author its root prim."""
    from isaaclab.sim import get_current_stage
    from isaaclab_newton.physics import NewtonManager
    from pxr import UsdGeom

    NewtonManager._per_world_builder_hooks.append(partial(_add_connector_cable, cfg=cfg))
    return UsdGeom.Xform.Define(get_current_stage(), prim_path).GetPrim()


def _remove_connector_cable_builder_hooks() -> None:
    """Remove deferred USB-C cable construction hooks during manager teardown."""
    from isaaclab_newton.physics import NewtonManager

    if hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = [
            hook
            for hook in NewtonManager._per_world_builder_hooks
            if not (isinstance(hook, partial) and hook.func is _add_connector_cable)
        ]


@configclass
class UsbcCableSpawnCfg(SpawnerCfg):
    """Describe an attached hinge chain built after the connector rigid body."""

    func = _spawn_connector_cable
    cable_prim_path: str = ""
    connector_prim_path: str = ""
    attachment: str = "plug"


@register_asset
class UsbcConnectorCable(Asset):
    """An attached MJWarp hinge chain with asset-owned construction and reset."""

    name = "usbc_insertion_connector_cable"
    tags = ["cable", "usbc_insertion"]

    def __init__(self, *, instance_name: str, prim_path: str, connector_prim_path: str, attachment: str) -> None:
        """Configure a cable attached to a plug or bulkhead.

        Args:
            instance_name: Unique scene asset name.
            prim_path: Cable root path beginning with {ENV_REGEX_NS}/.
            connector_prim_path: Parent connector path beginning with {ENV_REGEX_NS}/.
            attachment: Authored cable geometry and anchor, either plug or bulkhead.
        """
        assert attachment in ("plug", "bulkhead"), f"Unknown cable attachment: {attachment}"
        assert prim_path.startswith("{ENV_REGEX_NS}/"), "Cable path must be environment-local."
        assert connector_prim_path.startswith("{ENV_REGEX_NS}/"), "Connector path must be environment-local."
        super().__init__(name=instance_name, tags=list(type(self).tags))
        self.prim_path = prim_path
        self.object_cfg = AssetBaseCfg(
            prim_path=prim_path,
            spawn=UsbcCableSpawnCfg(
                cable_prim_path=prim_path,
                connector_prim_path=connector_prim_path,
                attachment=attachment,
            ),
        )
        self._reset_event = EventTermCfg(
            func=reset_connector_cable,
            mode="reset",
            params={"prim_path": prim_path, "links": _PLUG_LINKS if attachment == "plug" else _BULKHEAD_LINKS},
        )

    def get_object_cfg(self) -> tuple[str, AssetBaseCfg]:
        """Return the scene configuration that installs this cable's builder hook."""
        return self.name, self.object_cfg

    def get_event_cfg(self) -> tuple[str, EventTermCfg]:
        """Return this cable's reset event for normal Arena scene composition."""
        return self.name, self._reset_event

    def validate_simulation_cfg(self, sim_cfg) -> None:
        """Require the MJWarp backend used by the attached hinge-chain model."""
        from isaaclab_newton.physics import NewtonCfg

        assert isinstance(sim_cfg.physics, NewtonCfg) and issubclass(
            sim_cfg.physics.class_type, NewtonMJWarpManager
        ), "USB-C connector cables require the Newton MJWarp backend."


def _author_cable_visuals(root_path: str, links: int, segment_half_length: float) -> list[str]:
    """Author visual-only capsules for Newton/Fabric transform synchronization."""
    from isaaclab.sim import get_current_stage
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    paths = []
    for index in range(links):
        link_path = f"{root_path}/link{index}"
        link = UsdGeom.Xform.Define(stage, link_path)
        capsule = UsdGeom.Capsule.Define(stage, f"{link_path}/Geometry")
        capsule.CreateAxisAttr(UsdGeom.Tokens.z)
        capsule.CreateRadiusAttr(_CABLE_RADIUS)
        capsule.CreateHeightAttr(2.0 * segment_half_length)
        capsule.CreateDisplayColorAttr([Gf.Vec3f(*_CABLE_COLOR)])
        link.GetPrim().SetCustomDataByKey("isaaclabArena:visualOnly", True)
        paths.append(link_path)
    return paths


def _add_cable_chain(
    builder,
    *,
    parent: int,
    root_xform,
    root_anchor,
    links: int,
    segment_half_length: float,
    root_path: str,
) -> None:
    """Attach one alternating-hinge capsule chain to a connector body."""
    import warp as wp

    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.mu = _CABLE_MU
    shape_cfg.gap = _CABLE_CONTACT_GAP
    shape_cfg.density = _CABLE_DENSITY
    shape_cfg.mu_torsional = 0.0
    shape_cfg.mu_rolling = 0.0
    shape_cfg.ke = _CABLE_CONTACT_STIFFNESS
    shape_cfg.kd = _CABLE_CONTACT_DAMPING

    body_paths = _author_cable_visuals(root_path, links, segment_half_length)
    segment_length = 2.0 * segment_half_length
    joints = []
    for index in range(links):
        xform = wp.transform_multiply(
            root_xform,
            wp.transform(
                wp.vec3(0.0, 0.0, segment_half_length + index * segment_length),
                wp.quat_identity(),
            ),
        )
        body = builder.add_link(xform=xform, label=body_paths[index])
        builder.add_shape_capsule(
            body,
            radius=_CABLE_RADIUS,
            half_height=segment_half_length,
            cfg=shape_cfg,
            color=_CABLE_COLOR,
            label=f"{body_paths[index]}/Geometry",
        )
        parent_xform = (
            root_anchor if index == 0 else wp.transform(wp.vec3(0.0, 0.0, segment_half_length), wp.quat_identity())
        )
        axis = (1.0, 0.0, 0.0) if index % 2 == 0 else (0.0, 1.0, 0.0)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=body,
                parent_xform=parent_xform,
                child_xform=wp.transform(
                    wp.vec3(0.0, 0.0, -segment_half_length),
                    wp.quat_identity(),
                ),
                axis=axis,
                limit_lower=-2.0,
                limit_upper=2.0,
                armature=_CABLE_ARMATURE,
                friction=_CABLE_JOINT_FRICTION,
                label=f"{root_path}/bend{index}",
            )
        )
        parent = body
    builder.add_articulation(joints, label=f"{root_path}/articulation")


def _add_connector_cable(builder, world_index: int, _world_position, _world_quaternion, *, cfg) -> None:
    """Build one asset's attached hinge chain in the selected Newton world."""
    import warp as wp

    env_path = f"/World/envs/env_{world_index}"
    parent = _body_index(builder, cfg.connector_prim_path.replace("{ENV_REGEX_NS}", env_path))
    if cfg.attachment == "plug":
        anchor = wp.transform(
            wp.vec3(0.0, 0.0, _PLUG_REAR_Z),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi),
        )
        links, length = _PLUG_LINKS, _PLUG_LENGTH
    else:
        anchor = wp.transform(
            wp.vec3(*_BULKHEAD_CABLE_ANCHOR_POSITION),
            wp.quat(*_BULKHEAD_CABLE_ANCHOR_ROTATION),
        )
        links, length = _BULKHEAD_LINKS, _BULKHEAD_CABLE_LENGTH
    _add_cable_chain(
        builder,
        parent=parent,
        root_xform=wp.transform_multiply(builder.body_q[parent], anchor),
        root_anchor=anchor,
        links=links,
        segment_half_length=0.5 * length / links,
        root_path=cfg.cable_prim_path.replace("{ENV_REGEX_NS}", env_path),
    )


def reset_connector_cable(env, env_ids: Sequence[int] | None, *, prim_path: str, links: int) -> None:
    """Restore one cable asset's default bend state in selected environments."""
    import numpy as np

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    assert model is not None, "USB-C cable reset ran before Newton finalized its model."

    if env_ids is None:
        selected_worlds = set(range(env.num_envs))
    elif hasattr(env_ids, "detach"):
        selected_worlds = set(env_ids.detach().cpu().tolist())
    else:
        selected_worlds = set(env_ids)
    if not selected_worlds:
        return
    selected_prefixes = tuple(
        f"{prim_path.replace('{ENV_REGEX_NS}', f'/World/envs/env_{world}')}/bend" for world in selected_worlds
    )
    joint_indices = [index for index, label in enumerate(model.joint_label) if str(label).startswith(selected_prefixes)]
    expected = links * len(selected_worlds)
    assert len(joint_indices) == expected, f"USB-C reset found {len(joint_indices)} cable joints, expected {expected}."

    q_starts = np.asarray(model.joint_q_start.numpy())
    qd_starts = np.asarray(model.joint_qd_start.numpy())
    states = []
    for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
        if state is not None and all(state is not existing for existing in states):
            states.append(state)
    for state in states:
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        for joint_index in joint_indices:
            joint_q[int(q_starts[joint_index]) : int(q_starts[joint_index + 1])] = 0.0
            joint_qd[int(qd_starts[joint_index]) : int(qd_starts[joint_index + 1])] = 0.0
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
    NewtonManager.invalidate_fk()
