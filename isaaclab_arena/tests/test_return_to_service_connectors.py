# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject geometrically invalid socket capture without starting a physics simulation."""

import math
import torch
from types import SimpleNamespace

import pytest
from isaaclab.utils.math import quat_apply, quat_mul

_SKID_BOUNDS = ((-0.027, -0.009, 0.0), (0.027, 0.009, 0.003))
_CAPTURE_BOUNDS = ((0.0705, -0.0099, 0.03545), (0.1395, 0.0099, 0.0392))


class _Scene(dict):
    env_regex_ns = "/World/envs/env_.*"
    env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]


def _connector(local_position, local_rotation=(0.0, 0.0, 0.0, 1.0), *, capture=True, position_tolerance_m=0.003):
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena_environments.return_to_service.connectors import RetainedSocket, Socket

    stage = Usd.Stage.CreateInMemory()
    scene = _Scene()
    for name in ("body", "filter"):
        scene[name] = SimpleNamespace(cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/" + name))
        for env_path in scene.env_prim_paths:
            prim = UsdGeom.Xform.Define(stage, f"{env_path}/{name}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
    parent = torch.tensor(((0, 0, 0, 0, 0, 0, 1), (2, -3, 0.5, 0, 0, math.sin(0.4), math.cos(0.4))))
    local_position = parent.new_tensor(local_position).expand(2, -1)
    local_rotation = parent.new_tensor(local_rotation).expand(2, -1)
    child = torch.cat(
        (parent[:, :3] + quat_apply(parent[:, 3:], local_position), quat_mul(parent[:, 3:], local_rotation)), dim=-1
    )
    poses = {"body": parent, "filter": child}
    for name, pose in poses.items():
        scene[name].data = SimpleNamespace(root_com_pos_w=SimpleNamespace(torch=pose[:, :3].clone()))
    world = SimpleNamespace(
        get_pose_w=lambda name: poses[name],
        get_root_linear_velocity_w=lambda name: torch.zeros((2, 3)),
        get_root_angular_velocity_w=lambda name: torch.zeros((2, 3)),
    )
    env = SimpleNamespace(num_envs=2, device="cpu", scene=scene, sim=SimpleNamespace(stage=stage), arena_world=world)
    specification = Socket(
        "filter",
        "body",
        ("filter",),
        (0.103, 0.0, 0.036),
        position_tolerance_m=position_tolerance_m,
        capture_bounds=_CAPTURE_BOUNDS if capture else None,
        candidate_bounds={"filter": _SKID_BOUNDS} if capture else {},
    )
    return RetainedSocket(env, specification)


@pytest.mark.parametrize("height", (0.036, 0.0355))
def test_capture_accepts_nominal_and_supported_filter_in_each_parent_frame(height):
    connector = _connector((0.103, 0.0, height))
    assert connector.seated_candidates() == ["filter", "filter"]


@pytest.mark.parametrize(
    "position,rotation",
    (
        ((0.103, 0.0012, 0.036), (0, 0, 0, 1)),
        ((0.103, 0.0005, 0.036), (0, 0, math.sin(0.015), math.cos(0.015))),
        ((0.103, 0.0, 0.037), (0, 0, 0, 1)),
        ((0.103, 0.0, 0.0352), (0, 0, 0, 1)),
        ((0.103, 0.0, 0.036), (0, math.sin(0.02), 0, math.cos(0.02))),
    ),
)
def test_capture_rejects_crossed_guides_and_lost_vertical_engagement(position, rotation):
    # Every example passes the former independent 3 mm / 5 degree pose gates.
    assert _connector(position, rotation, capture=False).seated_candidates() == ["filter", "filter"]
    connector = _connector(position, rotation)
    assert connector.seated_candidates() == [None, None]
    assert connector.seated_candidates(require_stationary=False) == [None, None]


def test_capture_geometry_does_not_replace_the_insertion_speed_gate():
    connector = _connector((0.103, 0.0, 0.036))
    connector.env.arena_world.get_root_linear_velocity_w = lambda name: torch.full(
        (2, 3), 0.1 if name == "filter" else 0.0
    )
    assert connector.seated_candidates() == [None, None]
    assert connector.seated_candidates(require_stationary=False) == ["filter", "filter"]


def test_capture_contract_requires_matching_parent_and_candidate_geometry():
    from isaaclab_arena_environments.return_to_service.connectors import Socket

    with pytest.raises(AssertionError, match="every candidate"):
        Socket("test", "body", ("filter",), (0, 0, 0), capture_bounds=_CAPTURE_BOUNDS)
    with pytest.raises(AssertionError, match="parent capture"):
        Socket("test", "body", ("filter",), (0, 0, 0), candidate_bounds={"filter": _SKID_BOUNDS})


@pytest.mark.parametrize("offset", ((0.0, 0.0, 0.0025), (0.000337, 0.000995, 0.002555)))
def test_socket_specific_pose_tolerance_rejects_raised_entry(offset):
    # Nominal raised entry and the recorded pre-capture pose both satisfy the
    # former 3 mm envelope. Narrowing this socket leaves the default unchanged.
    position = (0.103 + offset[0], offset[1], 0.036 + offset[2])
    assert _connector(position, capture=False).seated_candidates() == ["filter", "filter"]
    connector = _connector(position, capture=False, position_tolerance_m=0.001)
    assert connector.seated_candidates() == [None, None]
    assert connector.seated_candidates(require_stationary=False) == [None, None]


def test_socket_specific_pose_tolerance_accepts_measured_final_seat_and_retains_speed_gate():
    connector = _connector((0.103112, -0.000012, 0.036153), capture=False, position_tolerance_m=0.001)
    assert connector.seated_candidates() == ["filter", "filter"]
    connector.env.arena_world.get_root_linear_velocity_w = lambda name: torch.full(
        (2, 3), 0.1 if name == "filter" else 0.0
    )
    assert connector.seated_candidates() == [None, None]


@pytest.mark.parametrize("tolerance", (0.0, -0.001, float("nan"), float("inf")))
def test_socket_requires_finite_positive_pose_tolerance(tolerance):
    from isaaclab_arena_environments.return_to_service.connectors import Socket

    with pytest.raises(AssertionError):
        Socket("test", "body", ("filter",), (0, 0, 0), position_tolerance_m=tolerance)


@pytest.mark.parametrize("omega,relative_speed,accepted", ((1.0, 0.055, False), (3.0, 0.0, True), (0.0, 0.055, False)))
def test_capture_transports_both_offset_com_velocities_to_socket_points(omega, relative_speed, accepted):
    connector = _connector((0.103, 0.0, 0.036), capture=False)
    world = connector.env.arena_world
    rotations = world.get_pose_w("body")[:, 3:]

    def rotate(vector):
        return quat_apply(rotations, rotations.new_tensor(vector).expand(2, -1))

    for name, offset in (("body", 0.04), ("filter", 0.02)):
        connector.env.scene[name].data.root_com_pos_w.torch = world.get_pose_w(name)[:, :3] + rotate((offset, 0, 0))
    # Both bodies rotate about the parent root; candidate translation adds an
    # independently specified speed relative to the coincident socket point.
    velocities = {
        "body": rotate((0.0, omega * 0.04, 0.0)),
        "filter": rotate((0.0, omega * 0.123 + relative_speed, 0.0)),
    }
    world.get_root_linear_velocity_w = velocities.__getitem__
    world.get_root_angular_velocity_w = lambda name: rotate((0.0, 0.0, omega))
    _, _, measured_speed = connector.candidate_errors("filter")
    torch.testing.assert_close(measured_speed, torch.full((2,), relative_speed), atol=1e-6, rtol=1e-6)
    assert connector.seated_candidates() == (["filter", "filter"] if accepted else [None, None])
    assert connector.seated_candidates(require_stationary=False) == ["filter", "filter"]
