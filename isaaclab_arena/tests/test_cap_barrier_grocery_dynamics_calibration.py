# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from isaaclab_arena.integrations.cap_barrier import (
    grocery_dynamics_calibration as calibration_module,
)
from isaaclab_arena.integrations.cap_barrier.grocery_dynamics_calibration import (
    GroceryDynamicsCalibrationEnvironment,
    _ROBOT_BODY_ROSTER,
    _attest_articulation_root_view,
    _attest_rigid_asset_root_view,
    _attest_simulation_view,
    _attested_body_record,
    _body_record,
    _canonical_sha256,
    _capture_verified_source_snapshots,
    _collision_manifest_for_root,
    _physical_device_identity,
    _require_source_identities_unchanged,
    _require_source_snapshots_unchanged,
    _validate_and_capture_collision_manifest,
    _validate_stock_collision_contract,
    _used_layer_closure,
    _write_verified_snapshot,
    calibration_recipe,
    canonical_payload_bytes,
    capture_dynamics_payload,
    capture_source_identity,
    mode_contract,
    write_calibration_artifact,
)
from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_OBJECT_ASSET,
)

_ENV = "/World/envs/env_0"
_COM = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
_INERTIA = (1.0, 0.1, 0.2, 0.1, 2.0, 0.3, 0.2, 0.3, 3.0)
_RUNTIME_VERSIONS = {
    "isaac_sim_app": "6.0.0.1",
    "isaaclab_physx": "1.1.3",
}


class _Stage:
    def GetTimeCodesPerSecond(self):
        return 60.0


class _WarpFrontend:
    def __init__(self, device="cuda:0", device_ordinal=0):
        self.device = device
        self.device_ordinal = device_ordinal


class _SimulationBackend:
    is_valid = True

    def __init__(self, device_ordinal=0):
        self.device_ordinal = device_ordinal


class _ArticulationBackend:
    pass


class _RigidBodyBackend:
    pass


class _RigidBodyView:
    count = 1

    def __init__(self, path, *, frontend, param_device=-1):
        self._backend = _RigidBodyBackend()
        self._frontend = frontend
        self._param_device = param_device
        self.prim_paths = (path,)

    def get_masses(self):
        return [[1.0]]

    def get_coms(self):
        return [[*_COM]]

    def get_inertias(self):
        return [[*_INERTIA]]


class _ArticulationView:
    count = 1

    def __init__(self, root_path, link_paths, *, frontend, param_device=-1):
        self._backend = _ArticulationBackend()
        self._frontend = frontend
        self._param_device = param_device
        self.prim_paths = (root_path,)
        self.link_paths = (tuple(link_paths),)


class _SimulationView:
    is_valid = True

    def __init__(self, *, device="cuda:0", actual_path=None):
        ordinal = 0 if device.startswith("cuda:") else -1
        self._backend = _SimulationBackend(ordinal)
        self._frontend = _WarpFrontend(device, ordinal)
        self._param_device_ordinal = -1
        self.actual_path = actual_path
        self.requested_paths = []

    @property
    def device(self):
        return self._frontend.device

    @property
    def device_ordinal(self):
        return self._backend.device_ordinal

    @property
    def param_device_ordinal(self):
        return self._param_device_ordinal

    def create_rigid_body_view(self, path):
        self.requested_paths.append(path)
        return _RigidBodyView(
            self.actual_path or path,
            frontend=self._frontend,
            param_device=self._param_device_ordinal,
        )


for _class, _name, _module in (
    (_WarpFrontend, "FrontendWarp", "omni.physics.tensors.frontend_warp"),
    (_SimulationView, "SimulationView", "omni.physics.tensors.api"),
    (_ArticulationView, "ArticulationView", "omni.physics.tensors.api"),
    (_RigidBodyView, "RigidBodyView", "omni.physics.tensors.api"),
    (
        _SimulationBackend,
        "SimulationView",
        "omni.physics.tensors.bindings._physicsTensors",
    ),
    (
        _ArticulationBackend,
        "ArticulationView",
        "omni.physics.tensors.bindings._physicsTensors",
    ),
    (
        _RigidBodyBackend,
        "RigidBodyView",
        "omni.physics.tensors.bindings._physicsTensors",
    ),
):
    _class.__name__ = _name
    _class.__qualname__ = _name
    _class.__module__ = _module


PhysxManager = type("PhysxManager", (), {})
PhysxManager.__module__ = "isaaclab_physx.physics.physx_manager"
PhysxManager.__qualname__ = "PhysxManager"

_PHYSICAL_DEVICE_IDENTITY = {
    "logical_device": "cuda:0",
    "warp_device_type": "warp.context.Device",
    "alias": "cuda:0",
    "ordinal": 0,
    "name": "Test GPU",
    "is_cuda": True,
    "uuid": "GPU-test",
    "pci_bus_id": "0000:01:00.0",
    "architecture": 89,
    "total_memory_bytes": 24 * 1024**3,
    "cuda_driver_version": [13, 0],
    "cuda_toolkit_version": [12, 9],
}


def _fake_tensor_types() -> dict[str, type]:
    return {
        "simulation_view": _SimulationView,
        "articulation_view": _ArticulationView,
        "rigid_body_view": _RigidBodyView,
        "warp_frontend": _WarpFrontend,
        "simulation_backend": _SimulationBackend,
        "articulation_backend": _ArticulationBackend,
        "rigid_body_backend": _RigidBodyBackend,
    }


def _calibration(
    *,
    mode="stock",
    body_names=None,
    close_error: Exception | None = None,
    physics_step_count=0,
    environment_step_count=0,
    timestamp=0.005,
    manager=None,
    actual_device="cuda:0",
):
    if body_names is None:
        body_names = tuple(item[1] for item in _ROBOT_BODY_ROSTER)
    robot_body_count = len(body_names)
    robot_data = SimpleNamespace(
        _sim_timestamp=timestamp,
        body_mass=[[float(index + 1) for index in range(robot_body_count)]],
        body_com_pose_b=[[_COM for _ in range(robot_body_count)]],
        body_inertia=[[_INERTIA for _ in range(robot_body_count)]],
    )
    object_data = SimpleNamespace(
        _sim_timestamp=timestamp,
        body_mass=[[0.5]],
        body_com_pose_b=[[_COM]],
        body_inertia=[[_INERTIA]],
    )

    class _Environment:
        closed = False

        def close(self):
            self.closed = True
            if close_error is not None:
                raise close_error

    simulation_view = _SimulationView(device=actual_device)
    relative_paths = {
        body_name: relative_path for _, body_name, relative_path in _ROBOT_BODY_ROSTER
    }
    robot_link_paths = tuple(
        f"{_ENV}/Robot/{relative_paths.get(body_name, body_name)}"
        for body_name in body_names
    )
    contents = {
        "robot": SimpleNamespace(
            body_names=body_names,
            data=robot_data,
            root_view=_ArticulationView(
                f"{_ENV}/Robot",
                robot_link_paths,
                frontend=simulation_view._frontend,
            ),
            _physics_sim_view=simulation_view,
            device=actual_device,
        ),
        CAP_GROCERY_OBJECT_ASSET: SimpleNamespace(
            data=object_data,
            root_view=_RigidBodyView(
                f"{_ENV}/{CAP_GROCERY_OBJECT_ASSET}",
                frontend=simulation_view._frontend,
            ),
            _physics_sim_view=simulation_view,
            device=actual_device,
        ),
        CAP_GROCERY_BIN_ASSET: SimpleNamespace(
            data=object_data,
            root_view=_RigidBodyView(
                f"{_ENV}/{CAP_GROCERY_BIN_ASSET}",
                frontend=simulation_view._frontend,
            ),
            _physics_sim_view=simulation_view,
            device=actual_device,
        ),
    }

    class _Scene:
        env_prim_paths = (_ENV,)
        stage = _Stage()

        def __getitem__(self, name):
            return contents[name]

    if manager is None:
        manager = PhysxManager()
        manager.get_physics_sim_view = lambda: simulation_view
        manager.get_physics_sim_device = lambda: actual_device
    sim = SimpleNamespace(
        physics_manager=manager,
        device=actual_device,
        backend="torch",
        get_physics_step_count=lambda: physics_step_count,
    )
    environment = _Environment()
    environment.unwrapped = SimpleNamespace(
        num_envs=1,
        scene=_Scene(),
        sim=sim,
        _sim_step_counter=environment_step_count,
    )
    recipe = calibration_recipe(device=actual_device)
    mode_identity = mode_contract(mode)
    collision_manifest = {
        "mode": mode,
        "analytic_cylinder_setting_override": None,
        "roots": {"robot": [], "can": [], "bin": []},
    }
    return GroceryDynamicsCalibrationEnvironment(
        environment=environment,
        mode=mode,
        source_uris={
            "robot": "robot.usd",
            "can": "can.usd",
            "bin": "bin.usd",
        },
        snapshot_uris={
            "robot": "/snapshot/robot.usd",
            "can": "/snapshot/can.usd",
            "bin": "/snapshot/bin.usd",
        },
        source_identities={
            "robot": {"raw_sha256": "a" * 64},
            "can": {"raw_sha256": "b" * 64},
            "bin": {"raw_sha256": "c" * 64},
        },
        simulation_dt_s=0.005,
        recipe=recipe,
        recipe_sha256=_canonical_sha256(recipe),
        mode_contract=mode_identity,
        mode_contract_sha256=_canonical_sha256(mode_identity),
        collision_manifest=collision_manifest,
        collision_manifest_sha256=_canonical_sha256(collision_manifest),
        _owned_resources=(),
    )


def _patch_capture_dependencies(monkeypatch, calibration) -> None:
    def body_record(**kwargs):
        record = _body_record(
            label=kwargs["label"],
            path=kwargs["expected_path"],
            body_index=kwargs["asset_body_index"],
            masses=kwargs["asset_masses"],
            com_poses=kwargs["asset_com_poses"],
            inertias=kwargs["asset_inertias"],
        )
        record["attestation"] = {
            "method": "omni.physics.tensors.RigidBodyView",
            "count": 1,
            "prim_paths": [kwargs["expected_path"]],
            "isaaclab_tensor_crosscheck": True,
        }
        return record

    monkeypatch.setattr(calibration_module, "_attested_body_record", body_record)
    monkeypatch.setattr(calibration_module, "_physx_manager_type", lambda: PhysxManager)
    monkeypatch.setattr(calibration_module, "_tensor_api_types", _fake_tensor_types)
    monkeypatch.setattr(
        calibration_module,
        "_physical_device_identity",
        lambda _device: dict(_PHYSICAL_DEVICE_IDENTITY),
    )
    monkeypatch.setattr(
        calibration_module,
        "_validate_and_capture_collision_manifest",
        lambda *args, **kwargs: calibration.collision_manifest,
    )
    monkeypatch.setattr(
        calibration_module,
        "_require_source_snapshots_unchanged",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "pxr.UsdGeom.GetStageMetersPerUnit",
        lambda _stage: 1.0,
    )
    monkeypatch.setattr(
        "pxr.UsdGeom.GetStageUpAxis",
        lambda _stage: "Z",
    )
    monkeypatch.setattr(
        "pxr.UsdPhysics.GetStageKilogramsPerUnit",
        lambda _stage: 1.0,
    )


def test_recipe_is_common_while_mode_contract_is_distinct() -> None:
    recipe = calibration_recipe(device="cuda:0")
    mutated = json.loads(json.dumps(recipe))
    mutated["environment"]["seed"] = 43

    assert _canonical_sha256(recipe) == _canonical_sha256(
        calibration_recipe(device="cuda:0")
    )
    assert _canonical_sha256(recipe) != _canonical_sha256(mutated)
    assert _canonical_sha256(mode_contract("stock")) != _canonical_sha256(
        mode_contract("proxy")
    )
    assert mode_contract("stock")["mode"] == "stock"
    assert mode_contract("proxy")["mode"] == "proxy"
    assert recipe["scene"]["object"]["scale"] == (1.0, 1.0, 1.0)
    assert recipe["scene"]["bin"]["scale"] == (0.007, 0.007, 0.007)


def test_object_spawn_scale_requires_asset_and_spawn_identity() -> None:
    asset = SimpleNamespace(
        scale=(0.007, 0.007, 0.007),
        object_cfg=SimpleNamespace(spawn=SimpleNamespace(scale=(0.007, 0.007, 0.007))),
    )
    calibration_module._require_object_spawn_scale(
        asset,
        label="bin",
        expected=(0.007, 0.007, 0.007),
    )

    asset.object_cfg.spawn.scale = (0.009, 0.009, 0.009)
    with pytest.raises(RuntimeError, match="scale drifted"):
        calibration_module._require_object_spawn_scale(
            asset,
            label="bin",
            expected=(0.007, 0.007, 0.007),
        )


def test_capture_emits_exact_attested_eleven_body_roster(monkeypatch) -> None:
    calibration = _calibration()
    _patch_capture_dependencies(monkeypatch, calibration)

    payload = capture_dynamics_payload(
        calibration,
        mode="stock",
        device="cuda:0",
        runtime_versions=_RUNTIME_VERSIONS,
    )

    assert payload["runner_commanded_physics_steps"] == 0
    assert payload["physics_step_count_before_capture"] == 0
    assert payload["physics_step_count_after_capture"] == 0
    assert payload["environment_step_count_before_capture"] == 0
    assert payload["environment_step_count_after_capture"] == 0
    assert payload["sample_phase"] == (
        "after_physx_warmup_and_scene_buffer_prime_before_environment_step"
    )
    assert payload["recipe_sha256"] == calibration.recipe_sha256
    assert payload["mode_contract_sha256"] == calibration.mode_contract_sha256
    assert payload["engine"] == {
        "engine_id": "physx",
        "engine_version": "isaac-sim:6.0.0.1;isaaclab-physx:1.1.3",
        "physics_manager_class": ("isaaclab_physx.physics.physx_manager.PhysxManager"),
        "isaaclab_tensor_backend": "torch",
        "actual_device": "cuda:0",
        "physical_device": _PHYSICAL_DEVICE_IDENTITY,
        "simulation_view": {
            "view_type": "omni.physics.tensors.api.SimulationView",
            "backend_type": (
                "omni.physics.tensors.bindings._physicsTensors.SimulationView"
            ),
            "frontend_type": "omni.physics.tensors.frontend_warp.FrontendWarp",
            "manager_device": "cuda:0",
            "view_device": "cuda:0",
            "device_ordinal": 0,
            "frontend_device": "cuda:0",
            "frontend_device_ordinal": 0,
            "param_device_ordinal": -1,
        },
    }
    assert set(payload["isaaclab_asset_views"]) == {"robot", "can", "bin"}
    assert [record["label"] for record in payload["bodies"]] == [
        *[item[0] for item in _ROBOT_BODY_ROSTER],
        "can",
        "bin",
    ]
    assert len(payload["bodies"]) == 11
    assert all(
        record["attestation"]["method"] == "omni.physics.tensors.RigidBodyView"
        for record in payload["bodies"]
    )


def test_capture_rejects_mode_relabel_before_reading_scene() -> None:
    with pytest.raises(RuntimeError, match="mode relabel is forbidden"):
        capture_dynamics_payload(
            _calibration(mode="stock"),
            mode="proxy",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )


@pytest.mark.parametrize(
    ("physics_steps", "environment_steps"),
    ((1, 0), (0, 1)),
)
def test_capture_rejects_any_prior_environment_step(
    physics_steps,
    environment_steps,
) -> None:
    with pytest.raises(RuntimeError, match="before an environment physics step"):
        capture_dynamics_payload(
            _calibration(
                physics_step_count=physics_steps,
                environment_step_count=environment_steps,
            ),
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )


def test_capture_rejects_scene_prime_timestamp_drift(monkeypatch) -> None:
    calibration = _calibration(timestamp=0.01)
    _patch_capture_dependencies(monkeypatch, calibration)

    with pytest.raises(RuntimeError, match="exactly one scene-prime update"):
        capture_dynamics_payload(
            calibration,
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )


def test_capture_rejects_hidden_step_during_evidence_collection(monkeypatch) -> None:
    step_count = 0
    calibration = _calibration()
    calibration.environment.unwrapped.sim.get_physics_step_count = lambda: step_count
    _patch_capture_dependencies(monkeypatch, calibration)

    def advance_while_rechecking_sources(*_args, **_kwargs):
        nonlocal step_count
        step_count = 1

    monkeypatch.setattr(
        calibration_module,
        "_require_source_snapshots_unchanged",
        advance_while_rechecking_sources,
    )

    with pytest.raises(RuntimeError, match="advanced while collecting evidence"):
        capture_dynamics_payload(
            calibration,
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )


def test_capture_rejects_manager_and_device_drift(monkeypatch) -> None:
    monkeypatch.setattr(calibration_module, "_physx_manager_type", lambda: PhysxManager)
    wrong_manager = SimpleNamespace(get_physics_sim_view=lambda: _SimulationView())
    with pytest.raises(RuntimeError, match="requires isaaclab_physx"):
        capture_dynamics_payload(
            _calibration(manager=wrong_manager),
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )

    with pytest.raises(RuntimeError, match="resolved to"):
        capture_dynamics_payload(
            _calibration(actual_device="cpu"),
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )


def test_capture_rejects_same_named_manager_impostor(monkeypatch) -> None:
    impostor = type("PhysxManager", (), {})()
    type(impostor).__module__ = "isaaclab_physx.physics.physx_manager"
    type(impostor).__qualname__ = "PhysxManager"
    monkeypatch.setattr(calibration_module, "_physx_manager_type", lambda: PhysxManager)

    with pytest.raises(RuntimeError, match="requires isaaclab_physx"):
        capture_dynamics_payload(
            _calibration(manager=impostor),
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )


def test_capture_rejects_incomplete_runtime_identity(monkeypatch) -> None:
    monkeypatch.setattr(calibration_module, "_physx_manager_type", lambda: PhysxManager)
    with pytest.raises(RuntimeError, match="runtime version identity is incomplete"):
        capture_dynamics_payload(
            _calibration(),
            mode="stock",
            device="cuda:0",
            runtime_versions={"isaac_sim_app": "6.0.0.1"},
        )


def test_simulation_view_attestation_requires_exact_types_and_devices(
    monkeypatch,
) -> None:
    monkeypatch.setattr(calibration_module, "_tensor_api_types", _fake_tensor_types)
    simulation_view = _SimulationView()
    manager = SimpleNamespace(
        get_physics_sim_view=lambda: simulation_view,
        get_physics_sim_device=lambda: "cuda:0",
    )

    returned, identity = _attest_simulation_view(
        manager,
        SimpleNamespace(device="cuda:0"),
        actual_device="cuda:0",
    )

    assert returned is simulation_view
    assert identity["backend_type"].endswith("_physicsTensors.SimulationView")
    assert identity["frontend_device_ordinal"] == 0

    impostor_class = type(
        "SimulationView",
        (_SimulationView,),
        {"__module__": "omni.physics.tensors.api"},
    )
    manager.get_physics_sim_view = lambda: impostor_class()
    with pytest.raises(RuntimeError, match="simulation view type drifted"):
        _attest_simulation_view(
            manager,
            SimpleNamespace(device="cuda:0"),
            actual_device="cuda:0",
        )

    simulation_view = _SimulationView()
    simulation_view._backend = object()
    manager.get_physics_sim_view = lambda: simulation_view
    with pytest.raises(RuntimeError, match="backend type drifted"):
        _attest_simulation_view(
            manager,
            SimpleNamespace(device="cuda:0"),
            actual_device="cuda:0",
        )

    simulation_view = _SimulationView()
    simulation_view._frontend.device_ordinal = 1
    manager.get_physics_sim_view = lambda: simulation_view
    with pytest.raises(RuntimeError, match="view device drifted"):
        _attest_simulation_view(
            manager,
            SimpleNamespace(device="cuda:0"),
            actual_device="cuda:0",
        )


def test_asset_view_attestation_rejects_foreign_ownership_and_path(
    monkeypatch,
) -> None:
    monkeypatch.setattr(calibration_module, "_tensor_api_types", _fake_tensor_types)
    simulation_view = _SimulationView()
    robot_paths = tuple(
        f"{_ENV}/Robot/{relative_path}" for _, _, relative_path in _ROBOT_BODY_ROSTER
    )
    robot = SimpleNamespace(
        _physics_sim_view=simulation_view,
        device="cuda:0",
        root_view=_ArticulationView(
            f"{_ENV}/Robot",
            robot_paths,
            frontend=simulation_view._frontend,
        ),
    )
    rigid = SimpleNamespace(
        _physics_sim_view=simulation_view,
        device="cuda:0",
        root_view=_RigidBodyView(
            f"{_ENV}/{CAP_GROCERY_OBJECT_ASSET}",
            frontend=simulation_view._frontend,
        ),
    )

    robot_identity = _attest_articulation_root_view(
        robot,
        simulation_view=simulation_view,
        actual_device="cuda:0",
        expected_root_path=f"{_ENV}/Robot",
        expected_body_count=len(robot_paths),
        expected_indexed_paths={index: path for index, path in enumerate(robot_paths)},
    )
    rigid_identity = _attest_rigid_asset_root_view(
        rigid,
        label="can",
        simulation_view=simulation_view,
        actual_device="cuda:0",
        expected_path=f"{_ENV}/{CAP_GROCERY_OBJECT_ASSET}",
    )
    assert robot_identity["link_paths"] == list(robot_paths)
    assert rigid_identity["prim_paths"] == [f"{_ENV}/{CAP_GROCERY_OBJECT_ASSET}"]

    robot._physics_sim_view = _SimulationView()
    with pytest.raises(RuntimeError, match="not owned by the active"):
        _attest_articulation_root_view(
            robot,
            simulation_view=simulation_view,
            actual_device="cuda:0",
            expected_root_path=f"{_ENV}/Robot",
            expected_body_count=len(robot_paths),
            expected_indexed_paths={},
        )

    rigid.device = "cuda:1"
    with pytest.raises(RuntimeError, match="device differs"):
        _attest_rigid_asset_root_view(
            rigid,
            label="can",
            simulation_view=simulation_view,
            actual_device="cuda:0",
            expected_path=f"{_ENV}/{CAP_GROCERY_OBJECT_ASSET}",
        )

    rigid.device = "cuda:0"
    rigid.root_view.prim_paths = (f"{_ENV}/wrong",)
    with pytest.raises(RuntimeError, match="root view path drifted"):
        _attest_rigid_asset_root_view(
            rigid,
            label="can",
            simulation_view=simulation_view,
            actual_device="cuda:0",
            expected_path=f"{_ENV}/{CAP_GROCERY_OBJECT_ASSET}",
        )


def test_physical_device_identity_is_stable_and_fail_closed(monkeypatch) -> None:
    warp_module = ModuleType("warp")
    device = SimpleNamespace(
        alias="cuda:0",
        ordinal=0,
        name="NVIDIA Test GPU",
        is_cuda=True,
        uuid="GPU-1234",
        pci_bus_id="0000:01:00.0",
        arch=89,
        total_memory=24 * 1024**3,
    )
    warp_module.get_device = lambda _name: device
    warp_module.get_cuda_driver_version = lambda: (13, 0)
    warp_module.get_cuda_toolkit_version = lambda: (12, 9)
    monkeypatch.setitem(sys.modules, "warp", warp_module)

    first = _physical_device_identity("cuda:0")
    second = _physical_device_identity("cuda:0")

    assert first == second
    assert first["alias"] == "cuda:0"
    assert first["ordinal"] == 0
    assert first["uuid"] == "GPU-1234"
    assert first["cuda_driver_version"] == [13, 0]
    assert first["cuda_toolkit_version"] == [12, 9]

    device.alias = "cuda:1"
    with pytest.raises(RuntimeError, match="differs from the physics device"):
        _physical_device_identity("cuda:0")

    device.alias = "cuda:0"
    warp_module.get_cuda_driver_version = lambda: (13,)
    with pytest.raises(RuntimeError, match="driver version"):
        _physical_device_identity("cuda:0")

    warp_module.get_cuda_driver_version = lambda: (13, 0)
    device.uuid = None
    with pytest.raises(RuntimeError, match="UUID or PCI bus identity is unavailable"):
        _physical_device_identity("cuda:0")


def test_capture_rejects_missing_or_duplicate_robot_body(monkeypatch) -> None:
    roster = tuple(item[1] for item in _ROBOT_BODY_ROSTER)
    missing = _calibration(body_names=roster[:-1])
    _patch_capture_dependencies(monkeypatch, missing)
    with pytest.raises(RuntimeError, match="exactly one 'right_inner_finger'"):
        capture_dynamics_payload(
            missing,
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )

    duplicate = _calibration(body_names=(roster[0], *roster))
    _patch_capture_dependencies(monkeypatch, duplicate)
    with pytest.raises(RuntimeError, match="exactly one 'base_link'"):
        capture_dynamics_payload(
            duplicate,
            mode="stock",
            device="cuda:0",
            runtime_versions=_RUNTIME_VERSIONS,
        )


@pytest.mark.parametrize(
    ("mass", "com", "inertia", "match"),
    (
        (0.0, _COM, _INERTIA, "mass must be positive"),
        (
            1.0,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0),
            _INERTIA,
            "must be normalized",
        ),
        (
            1.0,
            _COM,
            (1.0, 0.2, 0.0, 0.1, 2.0, 0.0, 0.0, 0.0, 3.0),
            "must be symmetric",
        ),
        (
            1.0,
            _COM,
            (1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 3.0),
            "must be positive definite",
        ),
    ),
)
def test_body_record_rejects_nonphysical_values(mass, com, inertia, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        _body_record(
            label="body",
            path="/World/body",
            body_index=0,
            masses=[[mass]],
            com_poses=[[com]],
            inertias=[[inertia]],
        )


def _rigid_body_stage(path="/World/body"):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return stage


def test_attested_body_record_binds_exact_physx_path_and_tensors(
    monkeypatch,
) -> None:
    stage = _rigid_body_stage()
    simulation_view = _SimulationView()
    monkeypatch.setattr(calibration_module, "_tensor_api_types", _fake_tensor_types)

    record = _attested_body_record(
        stage=stage,
        simulation_view=simulation_view,
        label="body",
        expected_path="/World/body",
        asset_body_index=0,
        asset_masses=[[1.0]],
        asset_com_poses=[[_COM]],
        asset_inertias=[[_INERTIA]],
    )

    assert simulation_view.requested_paths == ["/World/body"]
    assert record["prim_path"] == "/World/body"
    assert record["attestation"]["isaaclab_tensor_crosscheck"] is True


def test_attested_body_record_rejects_path_and_tensor_drift(monkeypatch) -> None:
    stage = _rigid_body_stage()
    monkeypatch.setattr(calibration_module, "_tensor_api_types", _fake_tensor_types)
    with pytest.raises(RuntimeError, match="view path drifted"):
        _attested_body_record(
            stage=stage,
            simulation_view=_SimulationView(actual_path="/World/wrong"),
            label="body",
            expected_path="/World/body",
            asset_body_index=0,
            asset_masses=[[1.0]],
            asset_com_poses=[[_COM]],
            asset_inertias=[[_INERTIA]],
        )

    with pytest.raises(RuntimeError, match="PhysX/IsaacLab mass_kg mismatch"):
        _attested_body_record(
            stage=stage,
            simulation_view=_SimulationView(),
            label="body",
            expected_path="/World/body",
            asset_body_index=0,
            asset_masses=[[2.0]],
            asset_com_poses=[[_COM]],
            asset_inertias=[[_INERTIA]],
        )


def test_used_layer_closure_hashes_root_and_sublayer(tmp_path) -> None:
    child = tmp_path / "child.usda"
    child.write_text(
        '#usda 1.0\n\ndef Xform "Child" {}\n',
        encoding="utf-8",
    )
    root = tmp_path / "root.usda"
    root.write_text(
        '#usda 1.0\n(\n    subLayers = [@child.usda@]\n)\ndef Xform "Root" {}\n',
        encoding="utf-8",
    )

    closure = _used_layer_closure(str(root))

    assert len(closure["used_layers"]) == 2
    assert len(closure["used_layer_closure_sha256"]) == 64
    assert {
        record["canonical_sha256"] for record in closure["used_layers"]
    } == _used_layer_digests(root)


def test_verified_snapshot_requires_exactly_one_resolved_root_layer(tmp_path) -> None:
    child = tmp_path / "child.usda"
    child.write_text('#usda 1.0\n\ndef Xform "Child" {}\n', encoding="utf-8")
    root = tmp_path / "root.usda"
    root.write_text(
        '#usda 1.0\n(\n    subLayers = [@child.usda@]\n)\ndef Xform "Root" {}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exactly one non-session"):
        _used_layer_closure(
            str(root),
            identity_uri="omniverse://source/root.usda",
            require_single_non_session_layer=True,
        )

    root.write_text('#usda 1.0\n\ndef Xform "Root" {}\n', encoding="utf-8")
    closure = _used_layer_closure(
        str(root),
        identity_uri="omniverse://source/root.usda",
        require_single_non_session_layer=True,
    )
    assert closure["used_layers"][0]["identifier"] == ("omniverse://source/root.usda")
    assert closure["used_layers"][0]["resolved_path"] == (
        "omniverse://source/root.usda"
    )


def test_verified_snapshot_rejects_path_escape(tmp_path) -> None:
    raw = b"#usda 1.0\n"
    with pytest.raises(RuntimeError, match="cannot form an owned snapshot path"):
        _write_verified_snapshot(
            directory=tmp_path,
            source_id="robot",
            uri_suffix="/../../escape.usda",
            raw_bytes=raw,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
        )


def _snapshot_usda(default_prim: str) -> bytes:
    return (
        "#usda 1.0\n"
        f'(\n    defaultPrim = "{default_prim}"\n)\n'
        f'def Xform "{default_prim}" (\n'
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI"]\n'
        ")\n"
        "{\n"
        '    def Cube "collider" (\n'
        '        prepend apiSchemas = ["PhysicsCollisionAPI"]\n'
        "    )\n"
        "    {\n"
        "        bool physics:collisionEnabled = true\n"
        "        double size = 1\n"
        "    }\n"
        "}\n"
    ).encode()


def test_owned_snapshots_close_remote_toctou_and_clean_up(
    monkeypatch,
) -> None:
    records = tuple(
        {
            "source_id": source_id,
            "uri_suffix": f"/{source_id}.usda",
            "raw_sha256": "unused",
        }
        for source_id in ("robot", "can", "bin")
    )
    source_uris = {
        source_id: f"omniverse://source/{source_id}.usda"
        for source_id in ("robot", "can", "bin")
    }
    remote_bytes = {
        source_id: _snapshot_usda(source_id.capitalize()) for source_id in source_uris
    }

    def read_verified(source_id, source_uri):
        raw = remote_bytes[source_id]
        return (
            {
                "source_id": source_id,
                "uri": source_uri,
                "uri_suffix": f"/{source_id}.usda",
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_size_bytes": len(raw),
                "provider_version": "v1",
                "provider_hash": f"etag-{source_id}",
            },
            raw,
        )

    monkeypatch.setattr(calibration_module, "_source_pin_records", lambda: records)
    monkeypatch.setattr(calibration_module, "_read_verified_source", read_verified)

    identities, snapshots, resource = _capture_verified_source_snapshots(source_uris)
    snapshot_root = Path(resource.directory.name)
    try:
        original_can = Path(snapshots["can"]).read_bytes()
        assert original_can == remote_bytes["can"]
        assert oct(Path(snapshots["can"]).stat().st_mode & 0o777) == "0o444"
        assert not tuple(snapshot_root.rglob("*.tmp"))
        assert str(snapshot_root) not in json.dumps(identities, sort_keys=True)
        assert all(
            identity["snapshot_binding"]["method"]
            == "owned_content_addressed_local_file"
            for identity in identities.values()
        )

        remote_bytes["can"] = _snapshot_usda("Changed")
        assert Path(snapshots["can"]).read_bytes() == original_can
        remote_bytes["can"] = original_can
        _require_source_snapshots_unchanged(identities, snapshots)

        os.chmod(snapshots["can"], 0o644)
        Path(snapshots["can"]).write_bytes(_snapshot_usda("Tampered"))
        with pytest.raises(RuntimeError, match="snapshot bytes changed"):
            _require_source_snapshots_unchanged(identities, snapshots)
    finally:
        resource.close()
    assert not snapshot_root.exists()


def test_calibration_retains_snapshots_until_environment_close(
    monkeypatch,
) -> None:
    records = (
        {
            "source_id": "robot",
            "uri_suffix": "/robot.usda",
            "raw_sha256": "unused",
        },
    )
    source_uri = "omniverse://source/robot.usda"
    raw = _snapshot_usda("Robot")
    monkeypatch.setattr(calibration_module, "_source_pin_records", lambda: records)
    monkeypatch.setattr(
        calibration_module,
        "_read_verified_source",
        lambda source_id, uri: (
            {
                "source_id": source_id,
                "uri": uri,
                "uri_suffix": "/robot.usda",
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_size_bytes": len(raw),
                "provider_version": "v1",
                "provider_hash": "etag",
            },
            raw,
        ),
    )
    identities, snapshots, resource = _capture_verified_source_snapshots(
        {"robot": source_uri}
    )
    snapshot = Path(snapshots["robot"])
    observed = []

    class _Environment:
        def close(self):
            observed.append(snapshot.exists())

    calibration = replace(
        _calibration(),
        environment=_Environment(),
        source_uris={"robot": source_uri},
        snapshot_uris=snapshots,
        source_identities=identities,
        _owned_resources=(resource,),
    )

    calibration.close()

    assert observed == [True]
    assert not snapshot.exists()


def _used_layer_digests(root: Path) -> set[str]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(root))
    return {
        hashlib.sha256(layer.ExportToString().encode("utf-8")).hexdigest()
        for layer in stage.GetUsedLayers()
        if not layer.anonymous
    }


def test_source_identity_pins_raw_bytes_and_provider_version(monkeypatch) -> None:
    import omni.client

    from isaaclab_arena.integrations.cap_barrier import (
        grocery_dynamics_certificate,
    )

    raw = b"#usda 1.0\n"
    pin = SimpleNamespace(
        source_id="robot",
        uri_suffix="/robot.usd",
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        grocery_dynamics_certificate,
        "GROCERY_DYNAMICS_SOURCE_PINS",
        (pin,),
    )
    monkeypatch.setattr(
        omni.client,
        "read_file",
        lambda _uri: (omni.client.Result.OK, "v1", raw),
    )
    monkeypatch.setattr(
        omni.client,
        "stat",
        lambda _uri: (
            omni.client.Result.OK,
            SimpleNamespace(size=len(raw), version="v1", hash="etag"),
        ),
    )
    monkeypatch.setattr(
        calibration_module,
        "_used_layer_closure",
        lambda _uri: {
            "default_prim_path": "/robot",
            "used_layers": [],
            "used_layer_closure_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        calibration_module,
        "_source_collision_identity",
        lambda _uri: {
            "source_collision_manifest": [],
            "source_collision_manifest_sha256": "e" * 64,
        },
    )

    identity = capture_source_identity("robot", "https://host/robot.usd")

    assert identity["raw_sha256"] == pin.sha256
    assert identity["provider_version"] == "v1"
    assert identity["provider_hash"] == "etag"

    monkeypatch.setattr(
        omni.client,
        "stat",
        lambda _uri: (
            omni.client.Result.OK,
            SimpleNamespace(size=len(raw), version="v2", hash="etag"),
        ),
    )
    with pytest.raises(RuntimeError, match="provider version changed"):
        capture_source_identity("robot", "https://host/robot.usd")


def test_source_identity_recheck_rejects_toctou_drift(monkeypatch) -> None:
    expected = {
        "robot": {"raw_sha256": "a" * 64},
        "can": {"raw_sha256": "b" * 64},
        "bin": {"raw_sha256": "c" * 64},
    }
    monkeypatch.setattr(
        calibration_module,
        "_capture_source_identities",
        lambda _uris: {
            **expected,
            "can": {"raw_sha256": "d" * 64},
        },
    )

    with pytest.raises(RuntimeError, match="changed between construction and capture"):
        _require_source_identities_unchanged(
            expected,
            {"robot": "r", "can": "c", "bin": "b"},
        )


def test_collision_manifest_records_effective_composed_prim() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/object")
    collision = UsdGeom.Cube.Define(stage, "/World/object/collider")
    collision.CreateSizeAttr(0.25)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)

    manifest = _collision_manifest_for_root(stage, "/World/object")

    assert manifest == [
        {
            "relative_path": "collider",
            "type_name": "Cube",
            "api_schemas": ["PhysicsCollisionAPI"],
            "attributes": {
                "physics:collisionEnabled": True,
                "size": 0.25,
            },
            "xform_ops": [],
            "resets_xform_stack": False,
            "ancestor_xform_chain": [
                {
                    "relative_path": "collider",
                    "xform_ops": [],
                    "resets_xform_stack": False,
                }
            ],
            "root_effective_linear_identity": {
                "gram_a_at": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "determinant": 1.0,
            },
            "mesh_topology_sha256": None,
            "physics_materials": [],
        }
    ]


def test_collision_manifest_skips_unset_nonfinite_schema_fallback_but_not_authored() -> (
    None
):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/object")
    collision = UsdGeom.Cube.Define(stage, "/World/object/collider")
    collision.CreateSizeAttr(0.25)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(collision.GetPrim())
    center_of_mass = mass.GetCenterOfMassAttr()
    assert center_of_mass.HasAuthoredValueOpinion() is False
    assert tuple(center_of_mass.Get()) == (
        float("-inf"),
        float("-inf"),
        float("-inf"),
    )

    manifest = _collision_manifest_for_root(stage, "/World/object")

    assert manifest[0]["attributes"]["physics:collisionEnabled"] is True
    assert "physics:centerOfMass" not in manifest[0]["attributes"]
    assert "PhysicsMassAPI" in manifest[0]["api_schemas"]

    center_of_mass.Set(Gf.Vec3f(float("-inf"), 0.0, 0.0))
    assert center_of_mass.HasAuthoredValueOpinion() is True
    with pytest.raises(RuntimeError, match="must be finite"):
        _collision_manifest_for_root(stage, "/World/object")


def test_collision_manifest_hashes_mesh_topology_and_ordered_transform() -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/object")
    mesh = UsdGeom.Mesh.Define(stage, "/World/object/collider")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.AddTranslateOp().Set((1.0, 2.0, 3.0))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)

    first = _collision_manifest_for_root(stage, "/World/object")
    first_hash = first[0]["mesh_topology_sha256"]
    mesh.GetPointsAttr().Set(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(2.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    second = _collision_manifest_for_root(stage, "/World/object")

    assert isinstance(first_hash, str) and len(first_hash) == 64
    assert first[0]["xform_ops"] == [
        {
            "name": "xformOp:translate",
            "inverse": False,
            "value": [1.0, 2.0, 3.0],
        }
    ]
    assert first_hash != second[0]["mesh_topology_sha256"]


def _collision_identity_stage():
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/parent")
    root = UsdGeom.Xform.Define(stage, "/World/parent/object")
    collider = UsdGeom.Cube.Define(stage, "/World/parent/object/collider")
    collider.CreateSizeAttr(0.25)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    collider.GetPrim().CreateAttribute(
        "physxConvexDecompositionCollision:maxConvexHulls",
        Sdf.ValueTypeNames.Int,
    ).Set(256)
    material = UsdShade.Material.Define(
        stage,
        "/World/parent/object/physicsMaterial",
    )
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material.GetPrim().CreateAttribute(
        "physics:staticFriction",
        Sdf.ValueTypeNames.Float,
    ).Set(2.0)
    material.GetPrim().CreateAttribute(
        "physics:dynamicFriction",
        Sdf.ValueTypeNames.Float,
    ).Set(2.0)
    material.GetPrim().CreateAttribute(
        "physics:restitution",
        Sdf.ValueTypeNames.Float,
    ).Set(0.1)
    material.GetPrim().CreateAttribute(
        "physics:density",
        Sdf.ValueTypeNames.Float,
    ).Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(collider.GetPrim())
    collider.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        [material.GetPath()]
    )
    return stage, parent, root, collider, material


def test_collision_manifest_binds_cooking_and_resolved_material_values() -> None:
    stage, _parent, _root, collider, material = _collision_identity_stage()

    first = _collision_manifest_for_root(stage, "/World/parent/object")
    first_hash = _canonical_sha256(first)
    record = first[0]

    assert (
        record["attributes"]["physxConvexDecompositionCollision:maxConvexHulls"] == 256
    )
    assert record["physics_materials"] == [
        {
            "target": "physicsMaterial",
            "binding_relationship": "collider.material:binding:physics",
            "type_name": "Material",
            "api_schemas": ["PhysicsMaterialAPI"],
            "attributes": {
                "physics:density": 0.0,
                "physics:dynamicFriction": 2.0,
                "physics:restitution": pytest.approx(0.1),
                "physics:staticFriction": 2.0,
            },
        }
    ]

    collider.GetPrim().GetAttribute(
        "physxConvexDecompositionCollision:maxConvexHulls"
    ).Set(3)
    assert (
        _canonical_sha256(_collision_manifest_for_root(stage, "/World/parent/object"))
        != first_hash
    )

    cooking_hash = _canonical_sha256(
        _collision_manifest_for_root(stage, "/World/parent/object")
    )
    material.GetPrim().GetAttribute("physics:staticFriction").Set(1.25)
    assert (
        _canonical_sha256(_collision_manifest_for_root(stage, "/World/parent/object"))
        != cooking_hash
    )


def test_collision_manifest_resolves_inherited_physics_material() -> None:
    from pxr import UsdShade

    stage, _parent, root, collider, _material = _collision_identity_stage()
    collider.GetPrim().RemoveProperty("material:binding:physics")
    UsdShade.MaterialBindingAPI.Apply(root.GetPrim())
    root.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        ["/World/parent/object/physicsMaterial"]
    )

    record = _collision_manifest_for_root(stage, "/World/parent/object")[0]

    assert record["physics_materials"][0]["target"] == "physicsMaterial"
    assert record["physics_materials"][0]["binding_relationship"] == (
        ".material:binding:physics"
    )
    assert record["physics_materials"][0]["attributes"]["physics:staticFriction"] == 2.0


def test_collision_manifest_ignores_rigid_pose_but_binds_scale_shear_and_handedness() -> (
    None
):
    from pxr import Gf, UsdGeom

    stage, parent, root, _collider, _material = _collision_identity_stage()
    parent_xform = UsdGeom.Xformable(parent)
    root_xform = UsdGeom.Xformable(root)
    translate = root_xform.AddTranslateOp()
    rotate = root_xform.AddRotateXYZOp()
    translate.Set(Gf.Vec3d(0.1, 0.2, 0.3))
    rotate.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    baseline = _collision_manifest_for_root(stage, "/World/parent/object")

    translate.Set(Gf.Vec3d(4.0, -2.0, 8.0))
    rotate.Set(Gf.Vec3f(23.0, -17.0, 91.0))
    rigid_pose = _collision_manifest_for_root(stage, "/World/parent/object")
    assert rigid_pose == baseline

    scale = parent_xform.AddScaleOp()
    scale.Set(Gf.Vec3f(1.25, 0.8, 1.1))
    scaled = _collision_manifest_for_root(stage, "/World/parent/object")
    assert _canonical_sha256(scaled) != _canonical_sha256(baseline)

    transform = parent_xform.AddTransformOp()
    shear = Gf.Matrix4d(1.0)
    shear.SetRow(0, Gf.Vec4d(1.0, 0.2, 0.0, 0.0))
    transform.Set(shear)
    sheared = _collision_manifest_for_root(stage, "/World/parent/object")
    assert _canonical_sha256(sheared) != _canonical_sha256(scaled)

    scale.Set(Gf.Vec3f(-1.25, 0.8, 1.1))
    reflected = _collision_manifest_for_root(stage, "/World/parent/object")
    assert reflected[0]["root_effective_linear_identity"]["determinant"] < 0.0
    assert _canonical_sha256(reflected) != _canonical_sha256(sheared)


def test_stock_collision_contract_requires_exact_source_manifests(
    monkeypatch,
) -> None:
    from isaaclab_arena.integrations.cap_barrier import gripper_linkage_override
    from isaaclab_arena.integrations.cap_barrier import (
        grocery_bin_collision_override,
    )
    from isaaclab_arena.integrations.cap_barrier.gripper_linkage_override import (
        _ALL_ORIGINAL_COLLISION_SUBPATHS,
    )
    from isaaclab_arena.integrations.cap_barrier.grocery_bin_collision_override import (
        _BIN_SOURCE_COLLISION_SUBPATH,
    )
    from isaaclab_arena.integrations.cap_barrier.grocery_object_collision_override import (
        _CAN_SOURCE_COLLISION_SUBPATH,
    )

    roots = {
        "/World/Robot": [{"relative_path": "robot"}],
        "/World/can": [{"relative_path": "can"}],
        "/World/bin": [{"relative_path": "bin"}],
    }
    source_identities = {
        name: {"source_collision_manifest": roots[path]}
        for name, path in (
            ("robot", "/World/Robot"),
            ("can", "/World/can"),
            ("bin", "/World/bin"),
        )
    }
    monkeypatch.setattr(
        gripper_linkage_override,
        "_require_collision_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        grocery_bin_collision_override,
        "_require_source_collider",
        lambda *_args, **_kwargs: None,
    )

    def enabled_paths(_stage, root_path):
        if root_path.endswith("Robotiq_2F_85"):
            return {
                f"/World/Robot/{subpath}"
                for subpath in _ALL_ORIGINAL_COLLISION_SUBPATHS
            }
        if root_path == "/World/can":
            return {f"{root_path}/{_CAN_SOURCE_COLLISION_SUBPATH}"}
        return {f"{root_path}/{_BIN_SOURCE_COLLISION_SUBPATH}"}

    monkeypatch.setattr(calibration_module, "_enabled_collision_paths", enabled_paths)
    monkeypatch.setattr(
        calibration_module,
        "_collision_manifest_for_root",
        lambda _stage, root_path: roots[root_path],
    )

    _validate_stock_collision_contract(
        object(),
        robot_prim_path="/World/Robot",
        can_prim_path="/World/can",
        bin_prim_path="/World/bin",
        source_identities=source_identities,
    )

    roots["/World/can"] = [{"relative_path": "drift"}]
    with pytest.raises(RuntimeError, match="differs from the raw pinned source"):
        _validate_stock_collision_contract(
            object(),
            robot_prim_path="/World/Robot",
            can_prim_path="/World/can",
            bin_prim_path="/World/bin",
            source_identities=source_identities,
        )


def test_proxy_collision_manifest_runs_all_live_validators(monkeypatch) -> None:
    from isaaclab_arena.integrations.cap_barrier import gripper_linkage_override
    from isaaclab_arena.integrations.cap_barrier import (
        grocery_bin_collision_override,
        grocery_object_collision_override,
    )

    calls = []
    monkeypatch.setattr(
        grocery_object_collision_override,
        "validate_analytic_cylinder_collision_setting",
        lambda: calls.append("setting") or "convexHull",
    )
    monkeypatch.setattr(
        gripper_linkage_override,
        "validate_live_grocery_gripper_collision_contract",
        lambda *_args, **_kwargs: calls.append("gripper"),
    )
    monkeypatch.setattr(
        grocery_object_collision_override,
        "validate_live_grocery_can_collision_contract",
        lambda *_args, **_kwargs: calls.append("can"),
    )
    monkeypatch.setattr(
        grocery_bin_collision_override,
        "validate_live_grocery_bin_collision_contract",
        lambda *_args, **_kwargs: calls.append("bin"),
    )
    monkeypatch.setattr(
        calibration_module,
        "_collision_manifest_for_root",
        lambda _stage, root_path: [{"relative_path": root_path}],
    )

    manifest = _validate_and_capture_collision_manifest(
        object(),
        mode="proxy",
        robot_prim_path="/World/Robot",
        can_prim_path="/World/can",
        bin_prim_path="/World/bin",
        source_identities={},
    )

    assert calls == ["setting", "gripper", "can", "bin"]
    assert manifest["analytic_cylinder_setting_override"] == "convexHull"


def test_canonical_artifact_is_stable_and_atomic(tmp_path) -> None:
    payload = {"z": [3, 2, 1], "a": {"value": 1.25}}
    expected = b'{"a":{"value":1.25},"z":[3,2,1]}\n'

    destination, digest = write_calibration_artifact(
        tmp_path / "nested" / "calibration.json",
        payload,
    )

    assert canonical_payload_bytes(payload) == expected
    assert destination.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert json.loads(destination.read_text()) == payload
    assert not tuple(destination.parent.glob("*.tmp"))


def test_canonical_payload_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_payload_bytes({"bad": float("nan")})


def test_calibration_environment_closes_every_resource_after_environment() -> None:
    order = []

    class _Close:
        def __init__(self, name):
            self.name = name

        def close(self):
            order.append(self.name)

    calibration = replace(
        _calibration(),
        environment=_Close("environment"),
        _owned_resources=(_Close("first"), _Close("second")),
    )

    calibration.close()

    assert order == ["environment", "second", "first"]


def test_calibration_environment_continues_cleanup_after_close_failure() -> None:
    order = []

    class _Environment:
        def close(self):
            order.append("environment")
            raise RuntimeError("environment close failed")

    class _Resource:
        def close(self):
            order.append("resource")

    calibration = replace(
        _calibration(),
        environment=_Environment(),
        _owned_resources=(_Resource(),),
    )

    with pytest.raises(RuntimeError, match="environment close failed"):
        calibration.close()

    assert order == ["environment", "resource"]


def test_runner_writes_only_after_simulation_app_context_exits(monkeypatch) -> None:
    cli_module = ModuleType("isaaclab_arena.cli.isaaclab_arena_cli")
    cli_module.get_isaaclab_arena_cli_parser = lambda: None
    simulation_app_module = ModuleType(
        "isaaclab_arena.utils.isaaclab_utils.simulation_app"
    )
    order = []

    class _Context:
        def __init__(self, _args):
            pass

        def __enter__(self):
            order.append("simulation_app_enter")

        def __exit__(self, *_args):
            order.append("simulation_app_exit")

    simulation_app_module.SimulationAppContext = _Context
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_arena.cli.isaaclab_arena_cli",
        cli_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_arena.utils.isaaclab_utils.simulation_app",
        simulation_app_module,
    )
    runner_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_cap_barrier_grocery_dynamics_calibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_cap_dynamics_runner_test",
        runner_path,
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    monkeypatch.setattr(
        runner,
        "_capture",
        lambda *_args, **_kwargs: order.append("capture") or {"bodies": []},
    )
    monkeypatch.setattr(
        runner,
        "_write_and_report",
        lambda *_args, **_kwargs: order.append("write"),
    )

    runner._run_cli(
        SimpleNamespace(
            device="cuda:0",
            mode="stock",
            output="/tmp/artifact.json",
        )
    )

    assert order == [
        "simulation_app_enter",
        "capture",
        "simulation_app_exit",
        "write",
    ]
