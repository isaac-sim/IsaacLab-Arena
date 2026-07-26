# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab_arena.integrations.cap_barrier.grocery_close_guard import (
    GroceryCloseAuthorizationError,
)
from isaaclab_arena.integrations.cap_barrier.grocery_collision_runtime import (
    GroceryCollisionRuntimeContract,
    configure_grocery_ground_collision_contract,
)
from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_GROUND_COLLISION_PRIM_PATH,
    CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
    CAP_GROCERY_OBJECT_ASSET,
    CAP_GROCERY_SUPPORT_INSTANCE,
    CAP_GROCERY_SUPPORT_POSE,
    CAP_GROCERY_SUPPORT_SIZE,
)

_ENV = "/World/envs/env_0"
_ROBOT = f"{_ENV}/Robot"
_CAN = f"{_ENV}/{CAP_GROCERY_OBJECT_ASSET}"
_BIN = f"{_ENV}/{CAP_GROCERY_BIN_ASSET}"
_SUPPORT = f"{_ENV}/{CAP_GROCERY_SUPPORT_INSTANCE}"
_IDENTITY_INERTIA = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


class _NumpyLike:
    def __init__(self, value) -> None:
        self._value = value

    def tolist(self):
        return self._value


class _WarpArray:
    """Minimal installed-Warp shape: convertible, but not Python-iterable."""

    def __init__(self, value) -> None:
        self._value = value

    def numpy(self):
        return _NumpyLike(self._value)


class _View:
    def __init__(
        self,
        path: str,
        *,
        shapes: int,
        material: tuple[float, float, float],
        contact_offset_m: float = 0.00025,
        warp_backed: bool = False,
    ) -> None:
        self._backend = object()
        self.count = 1
        self.max_shapes = shapes
        self.prim_paths = (path,)
        self.contacts = [[contact_offset_m] * shapes]
        self.rests = [[0.0] * shapes]
        self.materials = [[list(material) for _ in range(shapes)]]
        self._warp_backed = warp_backed
        self.contact_reads = 0
        self.rest_reads = 0
        self.material_reads = 0

    def _value(self, value):
        return _WarpArray(value) if self._warp_backed else value

    def get_contact_offsets(self):
        self.contact_reads += 1
        return self._value(self.contacts)

    def get_rest_offsets(self):
        self.rest_reads += 1
        return self._value(self.rests)

    def get_material_properties(self):
        self.material_reads += 1
        return self._value(self.materials)


class _SimulationView:
    def __init__(self, *, warp_backed: bool = False) -> None:
        self.is_valid = True
        self.views: dict[str, _View] = {}
        self._warp_backed = warp_backed

    def create_rigid_body_view(self, path: str) -> _View:
        if path.endswith("/base_link"):
            shapes, material = 1, (0.5, 0.5, 0.0)
        elif path.endswith("/left_inner_finger") or path.endswith(
            "/right_inner_finger"
        ):
            shapes, material = 2, (2.0, 2.0, 0.0)
        elif path == _CAN:
            shapes, material = 1, (2.0, 2.0, 0.1)
        elif path == _BIN:
            shapes, material = 5, (0.5, 0.5, 0.0)
        elif path == _SUPPORT:
            shapes, material = 1, (0.5, 0.5, 0.0)
        else:
            raise AssertionError(f"unexpected view path {path}")
        view = _View(
            path,
            shapes=shapes,
            material=material,
            contact_offset_m=0.001 if path == _BIN else 0.00025,
            warp_backed=self._warp_backed,
        )
        if path == _SUPPORT:
            view.contacts = [[0.005]]
        self.views[path] = view
        return view


class _Scene(dict):
    def __init__(self, *args, stage, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage = stage


def _fixture_stage(*, ground_offsets: bool = True):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    support = UsdGeom.Xform.Define(stage, _SUPPORT)
    xform = UsdGeom.Xformable(support)
    xform.AddTranslateOp().Set(Gf.Vec3d(*CAP_GROCERY_SUPPORT_POSE[0]))
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    rigid = UsdPhysics.RigidBodyAPI.Apply(support.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)

    support_collision = UsdGeom.Cube.Define(stage, f"{_SUPPORT}/geometry/mesh")
    support_collision.GetSizeAttr().Set(min(CAP_GROCERY_SUPPORT_SIZE))
    UsdGeom.Xformable(support_collision).AddScaleOp().Set(
        Gf.Vec3d(
            *(
                value / min(CAP_GROCERY_SUPPORT_SIZE)
                for value in CAP_GROCERY_SUPPORT_SIZE
            )
        )
    )
    UsdPhysics.CollisionAPI.Apply(
        support_collision.GetPrim()
    ).CreateCollisionEnabledAttr(True)
    support_collision.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
    support_collision.GetPrim().CreateAttribute(
        "physxCollision:contactOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(0.005)
    support_collision.GetPrim().CreateAttribute(
        "physxCollision:restOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(0.0)

    ground = UsdGeom.Plane.Define(stage, CAP_GROCERY_GROUND_COLLISION_PRIM_PATH)
    ground.GetAxisAttr().Set(UsdGeom.Tokens.z)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)
    if ground_offsets:
        configure_grocery_ground_collision_contract(stage)
    return stage


def _environment(
    *,
    body_names: tuple[str, ...] = (
        "base_link",
        "left_inner_finger",
        "right_inner_finger",
    ),
    warp_backed: bool = False,
    ground_offsets: bool = True,
):
    simulation_view = _SimulationView(warp_backed=warp_backed)
    robot_data = SimpleNamespace(
        _sim_timestamp=1.25,
        body_mass=[[1.0, 1.0, 1.0]],
        body_inertia=[
            [
                _IDENTITY_INERTIA,
                _IDENTITY_INERTIA,
                _IDENTITY_INERTIA,
            ]
        ],
        body_com_pose_b=[[_IDENTITY_POSE] * 3],
        body_link_pose_w=[[_IDENTITY_POSE] * 3],
    )
    can_data = SimpleNamespace(
        _sim_timestamp=1.25,
        body_mass=[[0.5]],
        body_inertia=[[_IDENTITY_INERTIA]],
        body_com_pose_b=[[_IDENTITY_POSE]],
        root_link_pose_w=[(0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
    )
    bin_data = SimpleNamespace(
        _sim_timestamp=1.25,
        body_mass=[[1.0]],
        body_inertia=[[_IDENTITY_INERTIA]],
        body_com_pose_b=[[_IDENTITY_POSE]],
        root_link_pose_w=[(0.46, -0.15, 0.01, 0.0, 0.0, 0.0, 1.0)],
    )
    support_data = SimpleNamespace(
        _sim_timestamp=1.25,
        root_link_pose_w=[(*CAP_GROCERY_SUPPORT_POSE[0], *CAP_GROCERY_SUPPORT_POSE[1])],
    )
    robot = SimpleNamespace(body_names=body_names, data=robot_data)
    can = SimpleNamespace(data=can_data)
    bin_asset = SimpleNamespace(data=bin_data)
    support = SimpleNamespace(data=support_data)
    scene = _Scene(
        {
            "robot": robot,
            CAP_GROCERY_OBJECT_ASSET: can,
            CAP_GROCERY_BIN_ASSET: bin_asset,
            CAP_GROCERY_SUPPORT_INSTANCE: support,
        },
        stage=_fixture_stage(ground_offsets=ground_offsets),
    )
    unwrapped = SimpleNamespace(
        scene=scene,
        sim=SimpleNamespace(
            physics_manager=SimpleNamespace(
                get_physics_sim_view=lambda: simulation_view
            )
        ),
    )
    return (
        SimpleNamespace(unwrapped=unwrapped),
        simulation_view,
        robot,
        can,
        bin_asset,
    )


def _contract(environment) -> GroceryCollisionRuntimeContract:
    return GroceryCollisionRuntimeContract(
        environment,
        robot_prim_path=_ROBOT,
        can_prim_path=_CAN,
        bin_prim_path=_BIN,
        support_prim_path=_SUPPORT,
    )


def test_runtime_contract_captures_same_timestamp_geometry_and_offsets() -> None:
    environment, _, _, _, _ = _environment()

    capture = _contract(environment).capture()

    assert capture.simulation_timestamp_s == 1.25
    assert capture.can_pose.position_m == (0.13, 0.0, 0.0)
    assert capture.bin_pose.position_m == (0.46, -0.15, 0.01)
    assert capture.support_pose.position_m == CAP_GROCERY_SUPPORT_POSE[0]
    assert capture.gripper_base_pose.orientation_xyzw == (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert capture.collision_offsets.palm.contact_m == pytest.approx(0.00025)
    assert capture.collision_offsets.can.rest_m == 0.0
    assert capture.collision_offsets.bin.contact_m == pytest.approx(0.001)
    assert capture.collision_offsets.bin.rest_m == 0.0
    assert capture.collision_offsets.support.contact_m == pytest.approx(0.005)
    assert capture.collision_offsets.ground.contact_m == pytest.approx(
        CAP_GROCERY_GROUND_CONTACT_OFFSET_M
    )
    ground = environment.unwrapped.scene.stage.GetPrimAtPath(
        CAP_GROCERY_GROUND_COLLISION_PRIM_PATH
    )
    assert ground.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(
        CAP_GROCERY_GROUND_CONTACT_OFFSET_M
    )


def test_runtime_contract_converts_installed_warp_frontend_arrays() -> None:
    environment, _, _, _, _ = _environment(warp_backed=True)

    capture = _contract(environment).capture()

    assert capture.collision_offsets.palm.contact_m == pytest.approx(0.00025)
    assert capture.collision_offsets.can.contact_m == pytest.approx(0.00025)


def test_runtime_contract_rejects_composed_support_pose_or_size_drift() -> None:
    from pxr import Gf

    environment, _, _, _, _ = _environment()
    support = environment.unwrapped.scene.stage.GetPrimAtPath(_SUPPORT)
    support.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.01, 0.0, 0.0))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="support pose/size drifted",
    ):
        _contract(environment)

    environment, _, _, _, _ = _environment()
    support_collision = environment.unwrapped.scene.stage.GetPrimAtPath(
        f"{_SUPPORT}/geometry/mesh"
    )
    support_collision.GetAttribute("size").Set(0.03)

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="support pose/size drifted",
    ):
        _contract(environment)


def test_runtime_contract_rejects_composed_ground_plane_drift() -> None:
    from pxr import Gf, UsdGeom

    environment, _, _, _, _ = _environment()
    ground = environment.unwrapped.scene.stage.GetPrimAtPath(
        CAP_GROCERY_GROUND_COLLISION_PRIM_PATH
    )
    UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.01))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="ground plane is not world z=0",
    ):
        _contract(environment)


def test_runtime_contract_rejects_default_or_late_ground_offsets() -> None:
    environment, _, _, _, _ = _environment(ground_offsets=False)

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="ground collision offsets are missing",
    ):
        _contract(environment)

    environment, _, _, _, _ = _environment()
    ground = environment.unwrapped.scene.stage.GetPrimAtPath(
        CAP_GROCERY_GROUND_COLLISION_PRIM_PATH
    )
    ground.GetAttribute("physxCollision:contactOffset").Set(0.006)

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="ground collision offsets drifted",
    ):
        _contract(environment)


def test_runtime_contract_rejects_close_without_dynamics_certificate() -> None:
    environment, _, _, _, _ = _environment()
    contract = _contract(environment)

    assert contract.dynamics_certified is False
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="stock-equivalent mass/COM/inertia certificate is not installed",
    ):
        contract.require_dynamics_certificate()


def test_runtime_contract_rejects_missing_or_duplicate_exact_bodies() -> None:
    environment, _, _, _, _ = _environment(
        body_names=("base_link", "left_inner_finger", "left_inner_finger")
    )

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="exactly one 'left_inner_finger'",
    ):
        _contract(environment)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda simulation, robot, can, bin_asset: setattr(
                simulation, "is_valid", False
            ),
            "simulation view is invalid",
        ),
        (
            lambda simulation, robot, can, bin_asset: setattr(
                simulation.views[f"{_ROBOT}/Gripper/Robotiq_2F_85/base_link"],
                "max_shapes",
                2,
            ),
            "shape count drifted",
        ),
        (
            lambda simulation, robot, can, bin_asset: setattr(
                can.data,
                "_sim_timestamp",
                1.3,
            ),
            "not from the same simulation timestamp",
        ),
        (
            lambda simulation, robot, can, bin_asset: setattr(
                can.data,
                "root_link_pose_w",
                [(0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
            ),
            "must be normalized",
        ),
        (
            lambda simulation, robot, can, bin_asset: setattr(
                simulation.views[_BIN],
                "max_shapes",
                4,
            ),
            "bin PhysX shape count drifted",
        ),
        (
            lambda simulation, robot, can, bin_asset: setattr(
                bin_asset.data,
                "_sim_timestamp",
                1.3,
            ),
            "not from the same simulation timestamp",
        ),
    ],
)
def test_capture_revalidates_dynamic_runtime_handles_fail_closed(
    mutation,
    match: str,
) -> None:
    environment, simulation, robot, can, bin_asset = _environment()
    contract = _contract(environment)
    mutation(simulation, robot, can, bin_asset)

    with pytest.raises(GroceryCloseAuthorizationError, match=match):
        contract.capture()


def test_capture_rejects_support_timestamp_skew() -> None:
    environment, _, _, _, _ = _environment()
    contract = _contract(environment)
    environment.unwrapped.scene[CAP_GROCERY_SUPPORT_INSTANCE].data._sim_timestamp = 1.3

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="not from the same simulation timestamp",
    ):
        contract.capture()


def test_capture_does_not_refetch_immutable_backend_properties() -> None:
    environment, simulation, _, _, _ = _environment()
    contract = _contract(environment)
    views = tuple(simulation.views.values())
    assert all(
        (view.contact_reads, view.rest_reads, view.material_reads) == (1, 1, 1)
        for view in views
    )

    for _ in range(10):
        contract.capture()

    assert all(
        (view.contact_reads, view.rest_reads, view.material_reads) == (1, 1, 1)
        for view in views
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda simulation: setattr(
                simulation.views[f"{_ROBOT}/Gripper/Robotiq_2F_85/left_inner_finger"],
                "materials",
                [[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]],
            ),
            "material drifted",
        ),
        (
            lambda simulation: setattr(
                simulation.views[_CAN],
                "contacts",
                [[0.01]],
            ),
            "collision offsets drifted",
        ),
        (
            lambda simulation: setattr(
                simulation.views[_BIN],
                "contacts",
                [[0.00025] * 5],
            ),
            "bin shape 0 collision offsets drifted",
        ),
        (
            lambda simulation: setattr(
                simulation.views[_SUPPORT],
                "contacts",
                [[0.001]],
            ),
            "support shape 0 collision offsets drifted",
        ),
    ],
)
def test_constructor_rejects_invalid_immutable_backend_properties(
    mutation,
    match: str,
) -> None:
    environment, simulation, _, _, _ = _environment()
    # Build the views so the mutation targets the same objects construction reads.
    for path in (
        f"{_ROBOT}/Gripper/Robotiq_2F_85/base_link",
        f"{_ROBOT}/Gripper/Robotiq_2F_85/left_inner_finger",
        f"{_ROBOT}/Gripper/Robotiq_2F_85/right_inner_finger",
        _CAN,
        _BIN,
        _SUPPORT,
    ):
        simulation.create_rigid_body_view(path)
    mutation(simulation)

    original_create = simulation.create_rigid_body_view
    simulation.create_rigid_body_view = lambda path: simulation.views[path]
    try:
        with pytest.raises(GroceryCloseAuthorizationError, match=match):
            _contract(environment)
    finally:
        simulation.create_rigid_body_view = original_create


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda robot, can, bin_asset: robot.data.body_mass[0].__setitem__(0, 0.0),
            "palm mass must be positive",
        ),
        (
            lambda robot, can, bin_asset: robot.data.body_inertia[0].__setitem__(
                1,
                (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            ),
            "left inertia is not symmetric",
        ),
        (
            lambda robot, can, bin_asset: can.data.body_inertia[0].__setitem__(
                0,
                (0.0,) * 9,
            ),
            "can inertia is not positive definite",
        ),
        (
            lambda robot, can, bin_asset: bin_asset.data.body_mass[0].__setitem__(
                0, 0.0
            ),
            "bin mass must be positive",
        ),
    ],
)
def test_constructor_rejects_invalid_runtime_mass_properties(
    mutation,
    match: str,
) -> None:
    environment, _, robot, can, bin_asset = _environment()
    mutation(robot, can, bin_asset)

    with pytest.raises(GroceryCloseAuthorizationError, match=match):
        _contract(environment)
