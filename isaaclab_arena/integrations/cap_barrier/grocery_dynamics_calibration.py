# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture reproducible pre-command dynamics evidence for the CAP grocery scene."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

_SCHEMA = "cap.grocery_dynamics_calibration.v1"
_RECIPE_SCHEMA = "cap.grocery_dynamics_calibration_recipe.v1"
_MODE_CONTRACT_SCHEMA = "cap.grocery_dynamics_mode_contract.v1"
_MODES = ("stock", "proxy")
_EXPECTED_PHYSICS_MANAGER = "isaaclab_physx.physics.physx_manager.PhysxManager"
_EXPECTED_SIMULATION_VIEW = "omni.physics.tensors.api.SimulationView"
_EXPECTED_ARTICULATION_VIEW = "omni.physics.tensors.api.ArticulationView"
_EXPECTED_RIGID_BODY_VIEW = "omni.physics.tensors.api.RigidBodyView"
_EXPECTED_WARP_FRONTEND = "omni.physics.tensors.frontend_warp.FrontendWarp"
_EXPECTED_SIMULATION_BACKEND = (
    "omni.physics.tensors.bindings._physicsTensors.SimulationView"
)
_EXPECTED_ARTICULATION_BACKEND = (
    "omni.physics.tensors.bindings._physicsTensors.ArticulationView"
)
_EXPECTED_RIGID_BODY_BACKEND = (
    "omni.physics.tensors.bindings._physicsTensors.RigidBodyView"
)
_ENGINE_ID = "physx"
_SCENE_SEED = 42
_SIMULATION_DT_S = 0.005
_DECIMATION = 1
_NUM_ENVS = 1
_CAN_SCALE = (1.0, 1.0, 1.0)
_BIN_SCALE = (0.007, 0.007, 0.007)
_QUATERNION_NORM_TOLERANCE = 1.0e-6
_INERTIA_SYMMETRY_TOLERANCE = 1.0e-8
_EFFECTIVE_LINEAR_DECIMAL_PLACES = 15
_ROBOT_BODY_ROSTER = (
    ("base_link", "base_link", "Gripper/Robotiq_2F_85/base_link"),
    (
        "left_outer_knuckle",
        "left_outer_knuckle",
        "Gripper/Robotiq_2F_85/left_outer_knuckle",
    ),
    (
        "left_outer_finger",
        "left_outer_finger",
        "Gripper/Robotiq_2F_85/left_outer_finger",
    ),
    (
        "left_inner_knuckle",
        "left_inner_knuckle",
        "Gripper/Robotiq_2F_85/left_inner_knuckle",
    ),
    (
        "left_inner_finger",
        "left_inner_finger",
        "Gripper/Robotiq_2F_85/left_inner_finger",
    ),
    (
        "right_outer_knuckle",
        "right_outer_knuckle",
        "Gripper/Robotiq_2F_85/right_outer_knuckle",
    ),
    (
        "right_outer_finger",
        "right_outer_finger",
        "Gripper/Robotiq_2F_85/right_outer_finger",
    ),
    (
        "right_inner_knuckle",
        "right_inner_knuckle",
        "Gripper/Robotiq_2F_85/right_inner_knuckle",
    ),
    (
        "right_inner_finger",
        "right_inner_finger",
        "Gripper/Robotiq_2F_85/right_inner_finger",
    ),
)
_EXPECTED_BODY_COUNT = 11


@dataclass(frozen=True)
class GroceryDynamicsCalibrationEnvironment:
    """Own one calibration environment and every temporary override it references."""

    environment: Any
    mode: str
    source_uris: dict[str, str]
    snapshot_uris: dict[str, str]
    source_identities: dict[str, dict[str, object]]
    simulation_dt_s: float
    recipe: dict[str, object]
    recipe_sha256: str
    mode_contract: dict[str, object]
    mode_contract_sha256: str
    collision_manifest: dict[str, object]
    collision_manifest_sha256: str
    _owned_resources: tuple[Any, ...]

    def close(self) -> None:
        """Close the environment before deleting any referenced override layers."""
        first_error: BaseException | None = None
        try:
            self.environment.close()
        except BaseException as error:
            first_error = error
        for resource in reversed(self._owned_resources):
            try:
                resource.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


@dataclass
class _OwnedSourceSnapshots:
    """Own verified local source bytes for the full environment lifetime."""

    directory: tempfile.TemporaryDirectory[str]
    paths: dict[str, str]

    def close(self) -> None:
        self.directory.cleanup()


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value using the artifact's canonical encoding."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_mode(mode: str) -> str:
    if mode not in _MODES:
        raise ValueError(f"unsupported dynamics calibration mode {mode!r}")
    return mode


def _source_pin_records() -> tuple[dict[str, str], ...]:
    from .grocery_dynamics_certificate import GROCERY_DYNAMICS_SOURCE_PINS

    return tuple(
        {
            "source_id": pin.source_id,
            "uri_suffix": pin.uri_suffix,
            "raw_sha256": pin.sha256,
        }
        for pin in GROCERY_DYNAMICS_SOURCE_PINS
    )


def calibration_recipe(*, device: str) -> dict[str, object]:
    """Return the immutable stock/proxy common scene recipe."""
    from .grocery_scene_spec import (
        CAP_GROCERY_BIN_ASSET,
        CAP_GROCERY_BIN_POSE,
        CAP_GROCERY_DROID_HOME,
        CAP_GROCERY_GROUND_COLLISION_PRIM_PATH,
        CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
        CAP_GROCERY_GROUND_REST_OFFSET_M,
        CAP_GROCERY_OBJECT_ASSET,
        CAP_GROCERY_OBJECT_POSE,
        CAP_GROCERY_SUPPORT_ASSET,
        CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
        CAP_GROCERY_SUPPORT_INSTANCE,
        CAP_GROCERY_SUPPORT_POSE,
        CAP_GROCERY_SUPPORT_REST_OFFSET_M,
        CAP_GROCERY_SUPPORT_SIZE,
    )

    return {
        "schema": _RECIPE_SCHEMA,
        "environment": {
            "seed": _SCENE_SEED,
            "num_envs": _NUM_ENVS,
            "solve_relations": False,
            "enable_cameras": False,
            "initial_gripper_closed": False,
            "device": device,
        },
        "simulation": {
            "dt_s": _SIMULATION_DT_S,
            "decimation": _DECIMATION,
        },
        "embodiment": {
            "registry_name": "droid_abs_joint_pos",
            "base_pose": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "joint_home": CAP_GROCERY_DROID_HOME,
            "stand": {
                "type": "invisible_cuboid",
                "size_m": (0.01, 0.01, 0.01),
            },
        },
        "scene": {
            "object": {
                "asset_name": CAP_GROCERY_OBJECT_ASSET,
                "pose": CAP_GROCERY_OBJECT_POSE,
                "scale": _CAN_SCALE,
            },
            "bin": {
                "asset_name": CAP_GROCERY_BIN_ASSET,
                "pose": CAP_GROCERY_BIN_POSE,
                "scale": _BIN_SCALE,
            },
            "support": {
                "asset_name": CAP_GROCERY_SUPPORT_ASSET,
                "instance_name": CAP_GROCERY_SUPPORT_INSTANCE,
                "pose": CAP_GROCERY_SUPPORT_POSE,
                "size_m": CAP_GROCERY_SUPPORT_SIZE,
                "contact_offset_m": CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
                "rest_offset_m": CAP_GROCERY_SUPPORT_REST_OFFSET_M,
            },
            "ground": {
                "collision_prim_path": CAP_GROCERY_GROUND_COLLISION_PRIM_PATH,
                "contact_offset_m": CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
                "rest_offset_m": CAP_GROCERY_GROUND_REST_OFFSET_M,
            },
        },
        "sources": _source_pin_records(),
        "body_roster": tuple(
            {
                "label": label,
                "body_name": body_name,
                "relative_prim_path": relative_path,
            }
            for label, body_name, relative_path in _ROBOT_BODY_ROSTER
        )
        + (
            {"label": "can", "body_name": "can", "relative_prim_path": None},
            {"label": "bin", "body_name": "bin", "relative_prim_path": None},
        ),
    }


def mode_contract(mode: str) -> dict[str, object]:
    """Return the immutable collision contract for one calibration mode."""
    mode = _require_mode(mode)
    from .gripper_linkage_override import (
        _ALL_ORIGINAL_COLLISION_SUBPATHS,
        _PROXY_BOX_SPECS,
    )
    from .grocery_bin_collision_override import (
        _BIN_PROXY_BOX_SPECS,
        _BIN_SOURCE_COLLISION_SUBPATH,
    )
    from .grocery_object_collision_override import (
        _CAN_PROXY_SUBPATH,
        _CAN_SOURCE_COLLISION_SUBPATH,
    )

    if mode == "stock":
        contract = {
            "gripper": {
                "enabled_original": _ALL_ORIGINAL_COLLISION_SUBPATHS,
                "enabled_proxy": (),
            },
            "can": {
                "enabled_original": (_CAN_SOURCE_COLLISION_SUBPATH,),
                "enabled_proxy": (),
            },
            "bin": {
                "enabled_original": (_BIN_SOURCE_COLLISION_SUBPATH,),
                "enabled_proxy": (),
            },
            "analytic_cylinder_override": False,
            "effective_manifest_rule": (
                "exactly_equal_to_raw_pinned_source_composed_manifest"
            ),
        }
    else:
        contract = {
            "gripper": {
                "enabled_original": (),
                "enabled_proxy": tuple(spec.proxy_subpath for spec in _PROXY_BOX_SPECS),
            },
            "can": {
                "enabled_original": (),
                "enabled_proxy": (_CAN_PROXY_SUBPATH,),
            },
            "bin": {
                "enabled_original": (),
                "enabled_proxy": tuple(spec.subpath for spec in _BIN_PROXY_BOX_SPECS),
            },
            "analytic_cylinder_override": True,
            "effective_manifest_rule": "exact_CAP_proxy_validators",
        }
    return {
        "schema": _MODE_CONTRACT_SCHEMA,
        "mode": mode,
        "contract": contract,
    }


def _read_verified_source(
    source_id: str,
    source_uri: str,
) -> tuple[dict[str, object], bytes]:
    """Read and pin exact source bytes through the active OmniClient."""
    import omni.client

    from .grocery_dynamics_certificate import GROCERY_DYNAMICS_SOURCE_PINS

    pins = {pin.source_id: pin for pin in GROCERY_DYNAMICS_SOURCE_PINS}
    pin = pins.get(source_id)
    if pin is None:
        raise RuntimeError(f"unknown dynamics source id {source_id!r}")
    normalized_uri = source_uri.replace("\\", "/")
    if not normalized_uri.endswith(pin.uri_suffix):
        raise RuntimeError(
            f"{source_id} source URI does not match the pinned suffix: {source_uri!r}"
        )

    result, provider_version, content = omni.client.read_file(source_uri)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"failed to read exact {source_id} source bytes: {result!r}")
    raw_bytes = memoryview(content).tobytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha256 != pin.sha256:
        raise RuntimeError(
            f"{source_id} raw source SHA-256 drifted: {raw_sha256} != {pin.sha256}"
        )

    stat_result, entry = omni.client.stat(source_uri)
    if stat_result != omni.client.Result.OK:
        raise RuntimeError(f"failed to stat exact {source_id} source: {stat_result!r}")
    if int(entry.size) != len(raw_bytes):
        raise RuntimeError(
            f"{source_id} source size changed between stat and read: "
            f"{entry.size!r} != {len(raw_bytes)}"
        )
    stat_version = str(entry.version or "")
    provider_version = str(provider_version or "")
    if stat_version and provider_version and stat_version != provider_version:
        raise RuntimeError(
            f"{source_id} provider version changed between stat and read: "
            f"{stat_version!r} != {provider_version!r}"
        )
    return (
        {
            "source_id": source_id,
            "uri": source_uri,
            "uri_suffix": pin.uri_suffix,
            "raw_sha256": raw_sha256,
            "raw_size_bytes": len(raw_bytes),
            "provider_version": provider_version or stat_version,
            "provider_hash": str(entry.hash or ""),
        },
        raw_bytes,
    )


def _read_raw_source(source_id: str, source_uri: str) -> dict[str, object]:
    """Read the exact source identity while discarding the returned bytes."""

    identity, _ = _read_verified_source(source_id, source_uri)
    return identity


def _used_layer_closure(
    source_uri: str,
    *,
    identity_uri: str | None = None,
    require_single_non_session_layer: bool = False,
) -> dict[str, object]:
    """Hash the complete composed USD layer closure for one source."""
    from pxr import Usd

    stage = Usd.Stage.Open(source_uri)
    if stage is None:
        raise RuntimeError(f"failed to open source USD {source_uri!r}")
    session_layer = stage.GetSessionLayer()
    layers = []
    for layer in stage.GetUsedLayers():
        if layer == session_layer:
            continue
        if bool(layer.anonymous):
            raise RuntimeError(
                f"source USD uses an anonymous non-session layer: {layer.identifier!r}"
            )
        canonical_text = layer.ExportToString()
        if not canonical_text:
            raise RuntimeError(
                f"source USD layer did not export canonical text: {layer.identifier!r}"
            )
        layers.append(
            {
                "identifier": str(layer.identifier),
                "resolved_path": str(layer.resolvedPath),
                "canonical_sha256": hashlib.sha256(
                    canonical_text.encode("utf-8")
                ).hexdigest(),
            }
        )
    layers.sort(key=lambda item: (item["resolved_path"], item["identifier"]))
    if not layers:
        raise RuntimeError(f"source USD has no composed layers: {source_uri!r}")
    if require_single_non_session_layer and len(layers) != 1:
        raise RuntimeError(
            "verified dynamics source must contain exactly one non-session "
            f"USD layer, got {len(layers)} for {source_uri!r}"
        )
    if require_single_non_session_layer:
        expected_root = str(Path(source_uri).resolve())
        if layers[0]["resolved_path"] != expected_root:
            raise RuntimeError(
                "verified dynamics source root did not resolve to its owned "
                f"snapshot: {layers[0]['resolved_path']!r} != {expected_root!r}"
            )
    if identity_uri is not None:
        if len(layers) != 1:
            raise RuntimeError(
                "a normalized source identity requires exactly one non-session layer"
            )
        layers[0]["identifier"] = identity_uri
        layers[0]["resolved_path"] = identity_uri
    default_prim = stage.GetDefaultPrim()
    return {
        "default_prim_path": (
            str(default_prim.GetPath()) if default_prim.IsValid() else None
        ),
        "used_layers": layers,
        "used_layer_closure_sha256": _canonical_sha256(layers),
    }


def _source_collision_identity(source_uri: str) -> dict[str, object]:
    from pxr import Usd

    stage = Usd.Stage.Open(source_uri)
    if stage is None:
        raise RuntimeError(f"failed to open source USD {source_uri!r}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError(f"source USD has no default prim: {source_uri!r}")
    collision_manifest = _collision_manifest_for_root(
        stage,
        str(default_prim.GetPath()),
    )
    return {
        "source_collision_manifest": collision_manifest,
        "source_collision_manifest_sha256": _canonical_sha256(collision_manifest),
    }


def capture_source_identity(source_id: str, source_uri: str) -> dict[str, object]:
    """Capture exact raw and composed identities for one source USD."""
    return {
        **_read_raw_source(source_id, source_uri),
        **_used_layer_closure(source_uri),
        **_source_collision_identity(source_uri),
    }


def _write_verified_snapshot(
    *,
    directory: Path,
    source_id: str,
    uri_suffix: str,
    raw_bytes: bytes,
    raw_sha256: str,
) -> str:
    relative_path = Path(uri_suffix.lstrip("/"))
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise RuntimeError(
            f"{source_id} source suffix cannot form an owned snapshot path"
        )
    snapshot_root = directory.resolve()
    destination = (snapshot_root / source_id / raw_sha256 / relative_path).resolve()
    if not destination.is_relative_to(snapshot_root):
        raise RuntimeError(f"{source_id} source snapshot escaped its owned directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary.write(raw_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
        os.chmod(destination, 0o444)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != raw_sha256:
        raise RuntimeError(f"{source_id} verified source snapshot write drifted")
    return str(destination)


def _capture_verified_source_snapshots(
    source_uris: Mapping[str, str],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, str],
    _OwnedSourceSnapshots,
]:
    """Bind source identity and scene construction to the same immutable bytes."""

    expected_ids = tuple(record["source_id"] for record in _source_pin_records())
    if set(source_uris) != set(expected_ids):
        raise RuntimeError(
            "dynamics source roster drifted: "
            f"expected={sorted(expected_ids)!r}, actual={sorted(source_uris)!r}"
        )
    temporary = tempfile.TemporaryDirectory(prefix="cap-grocery-dynamics-sources-")
    resource = _OwnedSourceSnapshots(directory=temporary, paths={})
    try:
        identities: dict[str, dict[str, object]] = {}
        for source_id in expected_ids:
            raw_identity, raw_bytes = _read_verified_source(
                source_id,
                source_uris[source_id],
            )
            snapshot_uri = _write_verified_snapshot(
                directory=Path(temporary.name),
                source_id=source_id,
                uri_suffix=str(raw_identity["uri_suffix"]),
                raw_bytes=raw_bytes,
                raw_sha256=str(raw_identity["raw_sha256"]),
            )
            resource.paths[source_id] = snapshot_uri
            identities[source_id] = {
                **raw_identity,
                **_used_layer_closure(
                    snapshot_uri,
                    identity_uri=source_uris[source_id],
                    require_single_non_session_layer=True,
                ),
                **_source_collision_identity(snapshot_uri),
                "snapshot_binding": {
                    "method": "owned_content_addressed_local_file",
                    "raw_sha256": raw_identity["raw_sha256"],
                    "non_session_layer_count": 1,
                },
            }
        return identities, dict(resource.paths), resource
    except BaseException:
        resource.close()
        raise


def _capture_source_identities(
    source_uris: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    expected_ids = tuple(record["source_id"] for record in _source_pin_records())
    if set(source_uris) != set(expected_ids):
        raise RuntimeError(
            "dynamics source roster drifted: "
            f"expected={sorted(expected_ids)!r}, actual={sorted(source_uris)!r}"
        )
    return {
        source_id: capture_source_identity(source_id, source_uris[source_id])
        for source_id in expected_ids
    }


def _require_source_identities_unchanged(
    expected: Mapping[str, dict[str, object]],
    source_uris: Mapping[str, str],
) -> None:
    actual = _capture_source_identities(source_uris)
    if actual != dict(expected):
        raise RuntimeError(
            "dynamics source identity changed between construction and capture"
        )


def _require_source_snapshots_unchanged(
    expected: Mapping[str, dict[str, object]],
    snapshot_uris: Mapping[str, str],
) -> None:
    """Reject any mutation of the owned source bytes before tensor capture."""

    if set(expected) != set(snapshot_uris):
        raise RuntimeError("dynamics source snapshot roster changed before capture")
    for source_id, snapshot_uri in snapshot_uris.items():
        expected_identity = expected[source_id]
        raw_bytes = Path(snapshot_uri).read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != expected_identity["raw_sha256"]:
            raise RuntimeError(
                f"{source_id} dynamics source snapshot bytes changed before capture"
            )
        actual_layers = _used_layer_closure(
            snapshot_uri,
            identity_uri=str(expected_identity["uri"]),
            require_single_non_session_layer=True,
        )
        for key in (
            "default_prim_path",
            "used_layers",
            "used_layer_closure_sha256",
        ):
            if actual_layers[key] != expected_identity[key]:
                raise RuntimeError(
                    f"{source_id} dynamics source snapshot layer identity changed"
                )
        actual_collision = _source_collision_identity(snapshot_uri)
        for key in (
            "source_collision_manifest",
            "source_collision_manifest_sha256",
        ):
            if actual_collision[key] != expected_identity[key]:
                raise RuntimeError(
                    f"{source_id} dynamics source snapshot collision identity changed"
                )


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"{label} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite, got {value!r}")
    return result


def _numeric_row(value: object, *, length: int, label: str) -> list[float]:
    value = _to_builtin(value)
    if isinstance(value, (str, bytes)):
        raise RuntimeError(f"{label} must contain exactly {length} numbers")
    try:
        result = [
            _finite_float(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)  # type: ignore[arg-type]
        ]
    except TypeError as error:
        raise RuntimeError(f"{label} must contain exactly {length} numbers") from error
    if len(result) != length:
        raise RuntimeError(
            f"{label} must contain exactly {length} numbers, got {len(result)}"
        )
    return result


def _source_uri(asset: Any) -> str:
    source_uri = getattr(asset, "usd_path", None)
    if source_uri is None:
        source_uri = getattr(getattr(asset, "object_cfg", None), "spawn", None)
        source_uri = getattr(source_uri, "usd_path", None)
    if not isinstance(source_uri, str) or not source_uri:
        raise RuntimeError(f"asset has no source USD URI: {asset!r}")
    return source_uri


def _set_object_source_uri(asset: Any, source_uri: str) -> None:
    spawn = getattr(getattr(asset, "object_cfg", None), "spawn", None)
    if spawn is None or not hasattr(spawn, "usd_path"):
        raise RuntimeError(f"asset has no mutable USD spawn config: {asset!r}")
    asset.usd_path = source_uri
    spawn.usd_path = source_uri
    if hasattr(asset, "bounding_box"):
        asset.bounding_box = None


def _require_object_spawn_scale(
    asset: Any,
    *,
    label: str,
    expected: tuple[float, float, float],
) -> None:
    spawn = getattr(getattr(asset, "object_cfg", None), "spawn", None)
    asset_scale = tuple(
        _numeric_row(
            getattr(asset, "scale", None),
            length=3,
            label=f"{label} asset scale",
        )
    )
    spawn_scale = tuple(
        _numeric_row(
            getattr(spawn, "scale", None),
            length=3,
            label=f"{label} spawn scale",
        )
    )
    if asset_scale != expected or spawn_scale != expected:
        raise RuntimeError(
            f"{label} scale drifted: asset={asset_scale!r}, "
            f"spawn={spawn_scale!r}, expected={expected!r}"
        )


def _module_version(module_name: str, distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        value = getattr(module, "__version__", None)
        return str(value) if value is not None else "unavailable"


def collect_runtime_versions() -> dict[str, str]:
    """Collect the active Kit and Python package versions without shell probes."""
    import omni.client

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
        get_isaac_sim_version,
    )

    return {
        "isaac_sim_app": str(get_isaac_sim_version()),
        "isaac_sim_package": _module_version("isaacsim", "isaacsim"),
        "isaac_lab": _module_version("isaaclab", "isaaclab"),
        "isaaclab_arena": _module_version(
            "isaaclab_arena",
            "isaaclab-arena",
        ),
        "isaaclab_physx": _module_version(
            "isaaclab_physx",
            "isaaclab-physx",
        ),
        "omni_client": str(omni.client.get_version()),
        "python": platform.python_version(),
    }


def _require_unique_body_index(
    body_names: tuple[str, ...],
    body_name: str,
) -> int:
    matches = [
        index for index, candidate in enumerate(body_names) if candidate == body_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"robot body roster must contain exactly one {body_name!r}, got {matches!r}"
        )
    return matches[0]


def _positive_definite_inertia(
    inertia: list[float],
    *,
    label: str,
) -> None:
    if len(inertia) != 9:
        raise RuntimeError(f"{label} must contain a 3x3 inertia matrix")
    xx, xy, xz, yx, yy, yz, zx, zy, zz = inertia
    if not (
        math.isclose(
            xy,
            yx,
            rel_tol=0.0,
            abs_tol=_INERTIA_SYMMETRY_TOLERANCE,
        )
        and math.isclose(
            xz,
            zx,
            rel_tol=0.0,
            abs_tol=_INERTIA_SYMMETRY_TOLERANCE,
        )
        and math.isclose(
            yz,
            zy,
            rel_tol=0.0,
            abs_tol=_INERTIA_SYMMETRY_TOLERANCE,
        )
    ):
        raise RuntimeError(f"{label} must be symmetric")
    leading_two = xx * yy - xy * yx
    determinant = (
        xx * (yy * zz - yz * zy) - xy * (yx * zz - yz * zx) + xz * (yx * zy - yy * zx)
    )
    if xx <= 0.0 or leading_two <= 0.0 or determinant <= 0.0:
        raise RuntimeError(f"{label} must be positive definite")


def _body_record(
    *,
    label: str,
    path: str,
    body_index: int,
    masses: Any,
    com_poses: Any,
    inertias: Any,
) -> dict[str, object]:
    try:
        mass = _finite_float(masses[0][body_index], label=f"{label} mass")
        com_pose = _numeric_row(
            com_poses[0][body_index],
            length=7,
            label=f"{label} COM pose",
        )
        inertia = _numeric_row(
            inertias[0][body_index],
            length=9,
            label=f"{label} inertia",
        )
    except (IndexError, TypeError) as error:
        raise RuntimeError(f"{label} dynamics tensor shape drifted") from error
    if mass <= 0.0:
        raise RuntimeError(f"{label} mass must be positive")
    quaternion_norm = math.sqrt(
        sum(component * component for component in com_pose[3:])
    )
    if not math.isclose(
        quaternion_norm,
        1.0,
        rel_tol=0.0,
        abs_tol=_QUATERNION_NORM_TOLERANCE,
    ):
        raise RuntimeError(
            f"{label} COM orientation quaternion must be normalized, "
            f"got norm={quaternion_norm!r}"
        )
    _positive_definite_inertia(inertia, label=f"{label} inertia")
    return {
        "label": label,
        "prim_path": path,
        "mass_kg": mass,
        "com_pose_body_xyzw": com_pose,
        "com_orientation_norm": quaternion_norm,
        "inertia_body_kg_m2_row_major": inertia,
    }


def _require_dynamics_records_match(
    backend: Mapping[str, object],
    asset: Mapping[str, object],
    *,
    label: str,
) -> None:
    for field in (
        "mass_kg",
        "com_pose_body_xyzw",
        "inertia_body_kg_m2_row_major",
    ):
        backend_value = backend[field]
        asset_value = asset[field]
        backend_values = (
            [backend_value] if isinstance(backend_value, Real) else list(backend_value)  # type: ignore[arg-type]
        )
        asset_values = (
            [asset_value] if isinstance(asset_value, Real) else list(asset_value)  # type: ignore[arg-type]
        )
        if backend_values != asset_values:
            raise RuntimeError(
                f"{label} PhysX/IsaacLab {field} mismatch: "
                f"backend={backend_value!r}, asset={asset_value!r}"
            )


def _expected_device_ordinal(device: str) -> int:
    if device == "cpu":
        return -1
    if device.startswith("cuda:"):
        try:
            ordinal = int(device.removeprefix("cuda:"))
        except ValueError as error:
            raise RuntimeError(f"invalid CUDA device identity {device!r}") from error
        if ordinal < 0:
            raise RuntimeError(f"invalid CUDA device ordinal {ordinal!r}")
        return ordinal
    raise RuntimeError(f"unsupported physics device identity {device!r}")


def _physx_manager_type() -> type:
    from isaaclab_physx.physics.physx_manager import PhysxManager

    return PhysxManager


def _tensor_api_types() -> dict[str, type]:
    from omni.physics.tensors.api import (
        ArticulationView,
        RigidBodyView,
        SimulationView,
    )
    from omni.physics.tensors.bindings import _physicsTensors
    from omni.physics.tensors.frontend_warp import FrontendWarp

    return {
        "simulation_view": SimulationView,
        "articulation_view": ArticulationView,
        "rigid_body_view": RigidBodyView,
        "warp_frontend": FrontendWarp,
        "simulation_backend": _physicsTensors.SimulationView,
        "articulation_backend": _physicsTensors.ArticulationView,
        "rigid_body_backend": _physicsTensors.RigidBodyView,
    }


def _require_view_identity(
    view: Any,
    *,
    label: str,
    expected_type: str,
    expected_class: type,
    expected_backend_type: str,
    expected_backend_class: type,
    simulation_view: Any,
) -> dict[str, object]:
    actual_type = _qualified_type_name(view)
    if type(view) is not expected_class:
        raise RuntimeError(
            f"{label} type drifted: expected={expected_type!r}, actual={actual_type!r}"
        )
    backend = getattr(view, "_backend", None)
    backend_type = _qualified_type_name(backend)
    if type(backend) is not expected_backend_class:
        raise RuntimeError(
            f"{label} backend type drifted: "
            f"expected={expected_backend_type!r}, actual={backend_type!r}"
        )
    frontend = getattr(view, "_frontend", None)
    if frontend is not getattr(simulation_view, "_frontend", None):
        raise RuntimeError(f"{label} frontend is not the active simulation frontend")
    param_device = getattr(view, "_param_device", None)
    expected_param_device = getattr(simulation_view, "param_device_ordinal", None)
    if param_device != expected_param_device:
        raise RuntimeError(
            f"{label} parameter device drifted: "
            f"{param_device!r} != {expected_param_device!r}"
        )
    return {
        "view_type": actual_type,
        "backend_type": backend_type,
        "frontend_type": _qualified_type_name(frontend),
        "param_device_ordinal": param_device,
    }


def _attest_simulation_view(
    manager: Any,
    sim: Any,
    *,
    actual_device: str,
) -> tuple[Any, dict[str, object]]:
    manager_device = str(manager.get_physics_sim_device())
    if manager_device != actual_device:
        raise RuntimeError(
            "physics manager device differs from the IsaacLab simulation device: "
            f"{manager_device!r} != {actual_device!r}"
        )
    simulation_view = manager.get_physics_sim_view()
    if simulation_view is None:
        raise RuntimeError(
            "PhysX simulation view is unavailable after scene construction"
        )
    actual_type = _qualified_type_name(simulation_view)
    tensor_types = _tensor_api_types()
    if type(simulation_view) is not tensor_types["simulation_view"]:
        raise RuntimeError(
            "PhysX simulation view type drifted: "
            f"{actual_type!r} != {_EXPECTED_SIMULATION_VIEW!r}"
        )
    is_valid = getattr(simulation_view, "is_valid", False)
    if callable(is_valid):
        is_valid = is_valid()
    if is_valid is not True:
        raise RuntimeError("PhysX simulation view is invalid")
    backend = getattr(simulation_view, "_backend", None)
    backend_type = _qualified_type_name(backend)
    if type(backend) is not tensor_types["simulation_backend"]:
        raise RuntimeError(
            "PhysX simulation backend type drifted: "
            f"{backend_type!r} != {_EXPECTED_SIMULATION_BACKEND!r}"
        )
    frontend = getattr(simulation_view, "_frontend", None)
    frontend_type = _qualified_type_name(frontend)
    if type(frontend) is not tensor_types["warp_frontend"]:
        raise RuntimeError(
            "PhysX simulation frontend drifted: "
            f"{frontend_type!r} != {_EXPECTED_WARP_FRONTEND!r}"
        )
    expected_ordinal = _expected_device_ordinal(actual_device)
    view_device = str(simulation_view.device)
    view_ordinal = getattr(simulation_view, "device_ordinal", None)
    backend_ordinal = getattr(backend, "device_ordinal", None)
    frontend_device = str(getattr(frontend, "device", None))
    frontend_ordinal = getattr(frontend, "device_ordinal", None)
    param_device_ordinal = getattr(
        simulation_view,
        "param_device_ordinal",
        None,
    )
    if (
        view_device != actual_device
        or view_ordinal != expected_ordinal
        or backend_ordinal != expected_ordinal
        or frontend_device != actual_device
        or frontend_ordinal != expected_ordinal
    ):
        raise RuntimeError(
            "PhysX simulation view device drifted: "
            f"device={view_device!r}, view_ordinal={view_ordinal!r}, "
            f"backend_ordinal={backend_ordinal!r}, "
            f"frontend_device={frontend_device!r}, "
            f"frontend_ordinal={frontend_ordinal!r}, "
            f"expected={actual_device!r}"
        )
    if param_device_ordinal != -1:
        raise RuntimeError(
            "PhysX property tensors must remain CPU-backed: "
            f"param_device_ordinal={param_device_ordinal!r}"
        )
    return simulation_view, {
        "view_type": actual_type,
        "backend_type": backend_type,
        "frontend_type": frontend_type,
        "manager_device": manager_device,
        "view_device": view_device,
        "device_ordinal": view_ordinal,
        "frontend_device": frontend_device,
        "frontend_device_ordinal": frontend_ordinal,
        "param_device_ordinal": param_device_ordinal,
    }


def _flatten_single_instance_paths(value: object, *, label: str) -> tuple[str, ...]:
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise RuntimeError(f"{label} paths are not iterable") from error
    if (
        len(items) == 1
        and not isinstance(items[0], (str, bytes))
        and hasattr(items[0], "__iter__")
    ):
        items = tuple(items[0])
    return tuple(str(item) for item in items)


def _attest_articulation_root_view(
    robot: Any,
    *,
    simulation_view: Any,
    actual_device: str,
    expected_root_path: str,
    expected_body_count: int,
    expected_indexed_paths: Mapping[int, str],
) -> dict[str, object]:
    if getattr(robot, "_physics_sim_view", None) is not simulation_view:
        raise RuntimeError(
            "IsaacLab robot is not owned by the active physics simulation view"
        )
    if str(getattr(robot, "device", None)) != actual_device:
        raise RuntimeError(
            "IsaacLab robot device differs from the active physics device"
        )
    root_view = robot.root_view
    tensor_types = _tensor_api_types()
    identity = _require_view_identity(
        root_view,
        label="IsaacLab robot root view",
        expected_type=_EXPECTED_ARTICULATION_VIEW,
        expected_class=tensor_types["articulation_view"],
        expected_backend_type=_EXPECTED_ARTICULATION_BACKEND,
        expected_backend_class=tensor_types["articulation_backend"],
        simulation_view=simulation_view,
    )
    if getattr(root_view, "count", None) != 1:
        raise RuntimeError("IsaacLab robot root view count must equal one")
    root_paths = tuple(str(path) for path in root_view.prim_paths)
    if root_paths != (expected_root_path,):
        raise RuntimeError(f"IsaacLab robot root view path drifted: {root_paths!r}")
    link_paths = _flatten_single_instance_paths(
        root_view.link_paths,
        label="IsaacLab robot root view",
    )
    if len(link_paths) != expected_body_count:
        raise RuntimeError(
            "IsaacLab robot link path count drifted: "
            f"expected={expected_body_count!r}, actual={len(link_paths)!r}"
        )
    for body_index, expected_path in expected_indexed_paths.items():
        if link_paths[body_index] != expected_path:
            raise RuntimeError(
                "IsaacLab robot link path/order drifted: "
                f"index={body_index}, expected={expected_path!r}, "
                f"actual={link_paths[body_index]!r}"
            )
    return {
        **identity,
        "count": 1,
        "prim_paths": list(root_paths),
        "link_paths": list(link_paths),
    }


def _attest_rigid_asset_root_view(
    asset: Any,
    *,
    label: str,
    simulation_view: Any,
    actual_device: str,
    expected_path: str,
) -> dict[str, object]:
    if getattr(asset, "_physics_sim_view", None) is not simulation_view:
        raise RuntimeError(
            f"IsaacLab {label} is not owned by the active physics simulation view"
        )
    if str(getattr(asset, "device", None)) != actual_device:
        raise RuntimeError(
            f"IsaacLab {label} device differs from the active physics device"
        )
    root_view = asset.root_view
    tensor_types = _tensor_api_types()
    identity = _require_view_identity(
        root_view,
        label=f"IsaacLab {label} root view",
        expected_type=_EXPECTED_RIGID_BODY_VIEW,
        expected_class=tensor_types["rigid_body_view"],
        expected_backend_type=_EXPECTED_RIGID_BODY_BACKEND,
        expected_backend_class=tensor_types["rigid_body_backend"],
        simulation_view=simulation_view,
    )
    if getattr(root_view, "count", None) != 1:
        raise RuntimeError(f"IsaacLab {label} root view count must equal one")
    root_paths = tuple(str(path) for path in root_view.prim_paths)
    if root_paths != (expected_path,):
        raise RuntimeError(f"IsaacLab {label} root view path drifted: {root_paths!r}")
    return {
        **identity,
        "count": 1,
        "prim_paths": list(root_paths),
    }


def _physical_device_identity(actual_device: str) -> dict[str, object]:
    import warp as wp

    device = wp.get_device(actual_device)
    expected_ordinal = _expected_device_ordinal(actual_device)
    raw_alias = getattr(device, "alias", None)
    raw_name = getattr(device, "name", None)
    if raw_alias is None or raw_name is None:
        raise RuntimeError("Warp physical device alias or name is unavailable")
    alias = str(raw_alias).strip()
    ordinal = int(device.ordinal)
    name = str(raw_name).strip()
    if alias != actual_device or ordinal != expected_ordinal:
        raise RuntimeError(
            "Warp physical device identity differs from the physics device: "
            f"alias={alias!r}, ordinal={ordinal!r}, expected={actual_device!r}"
        )
    if not name:
        raise RuntimeError("Warp physical device name is empty")
    identity: dict[str, object] = {
        "logical_device": actual_device,
        "warp_device_type": _qualified_type_name(device),
        "alias": alias,
        "ordinal": ordinal,
        "name": name,
        "is_cuda": bool(device.is_cuda),
    }
    if actual_device.startswith("cuda:"):
        if not device.is_cuda:
            raise RuntimeError(f"Warp resolved {actual_device!r} to a non-CUDA device")
        raw_uuid = getattr(device, "uuid", None)
        raw_pci_bus_id = getattr(device, "pci_bus_id", None)
        if raw_uuid is None or raw_pci_bus_id is None:
            raise RuntimeError("Warp CUDA UUID or PCI bus identity is unavailable")
        uuid = str(raw_uuid).strip()
        pci_bus_id = str(raw_pci_bus_id).strip()
        architecture = int(device.arch)
        total_memory_bytes = int(device.total_memory)
        driver_version = tuple(wp.get_cuda_driver_version())
        toolkit_version = tuple(wp.get_cuda_toolkit_version())
        if not uuid or not pci_bus_id:
            raise RuntimeError("Warp CUDA UUID or PCI bus identity is empty")
        if architecture <= 0 or total_memory_bytes <= 0:
            raise RuntimeError(
                "Warp CUDA architecture or physical memory identity is invalid"
            )
        for label, version in (
            ("driver", driver_version),
            ("toolkit", toolkit_version),
        ):
            if (
                len(version) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in version
                )
                or any(item < 0 for item in version)
            ):
                raise RuntimeError(
                    f"Warp CUDA {label} version must be a non-negative integer pair"
                )
        identity.update(
            {
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "architecture": architecture,
                "total_memory_bytes": total_memory_bytes,
                "cuda_driver_version": list(driver_version),
                "cuda_toolkit_version": list(toolkit_version),
            }
        )
    elif bool(device.is_cuda) or not bool(getattr(device, "is_cpu", False)):
        raise RuntimeError("Warp CPU identity resolved to a non-CPU device")
    return identity


def _attested_body_record(
    *,
    stage: Any,
    simulation_view: Any,
    label: str,
    expected_path: str,
    asset_body_index: int,
    asset_masses: Any,
    asset_com_poses: Any,
    asset_inertias: Any,
) -> dict[str, object]:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(expected_path)
    if (
        not prim.IsValid()
        or not prim.IsDefined()
        or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        raise RuntimeError(
            f"{label} expected prim is not a defined rigid body: {expected_path!r}"
        )
    view = simulation_view.create_rigid_body_view(expected_path)
    tensor_types = _tensor_api_types()
    view_identity = _require_view_identity(
        view,
        label=f"{label} PhysX rigid-body view",
        expected_type=_EXPECTED_RIGID_BODY_VIEW,
        expected_class=tensor_types["rigid_body_view"],
        expected_backend_type=_EXPECTED_RIGID_BODY_BACKEND,
        expected_backend_class=tensor_types["rigid_body_backend"],
        simulation_view=simulation_view,
    )
    count = getattr(view, "count", None)
    if isinstance(count, bool) or count != 1:
        raise RuntimeError(f"{label} PhysX rigid-body view count drifted: {count!r}")
    actual_paths = tuple(getattr(view, "prim_paths", ()))
    if actual_paths != (expected_path,):
        raise RuntimeError(
            f"{label} PhysX rigid-body view path drifted: {actual_paths!r}"
        )

    backend = _body_record(
        label=label,
        path=actual_paths[0],
        body_index=0,
        masses=_to_builtin(view.get_masses()),
        com_poses=[_to_builtin(view.get_coms())],
        inertias=[_to_builtin(view.get_inertias())],
    )
    asset = _body_record(
        label=label,
        path=expected_path,
        body_index=asset_body_index,
        masses=asset_masses,
        com_poses=asset_com_poses,
        inertias=asset_inertias,
    )
    _require_dynamics_records_match(backend, asset, label=label)
    backend["attestation"] = {
        "method": "omni.physics.tensors.RigidBodyView",
        **view_identity,
        "count": 1,
        "prim_paths": list(actual_paths),
        "isaaclab_tensor_crosscheck": True,
    }
    return backend


def _json_compatible(value: Any) -> object:
    value = _to_builtin(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Real):
        return _finite_float(value, label="collision manifest number")
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    try:
        return [_json_compatible(item) for item in value]
    except TypeError:
        return str(value)


def _qualified_type_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _ordered_xform_ops(prim: Any) -> list[dict[str, object]]:
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    result = []
    for operation in xformable.GetOrderedXformOps():
        value = operation.Get()
        if value is not None:
            result.append(
                {
                    "name": operation.GetOpName(),
                    "inverse": bool(operation.IsInverseOp()),
                    "value": _json_compatible(value),
                }
            )
    return result


def _root_effective_linear_identity(root: Any) -> dict[str, object]:
    """Capture effective scale/shear/reflection while ignoring rigid world pose."""

    from pxr import UsdGeom

    def canonical(value: float, *, label: str) -> float:
        return _finite_float(
            round(value, _EFFECTIVE_LINEAR_DECIMAL_PLACES),
            label=label,
        )

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(root)
    linear = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    metric = [
        [
            canonical(
                sum(linear[row][axis] * linear[column][axis] for axis in range(3)),
                label=f"root linear metric[{row}][{column}]",
            )
            for column in range(3)
        ]
        for row in range(3)
    ]
    determinant = (
        linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
        - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
        + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0])
    )
    return {
        "gram_a_at": metric,
        "determinant": canonical(
            determinant,
            label="root effective linear determinant",
        ),
    }


def _ancestor_xform_chain(root: Any, prim: Any) -> list[dict[str, object]]:
    """Capture every local transform below root, excluding the scene root pose."""

    chain = []
    cursor = prim
    while cursor.IsValid() and cursor != root:
        chain.append(cursor)
        cursor = cursor.GetParent()
    if cursor != root:
        raise RuntimeError(
            f"collision prim {prim.GetPath()} is not below root {root.GetPath()}"
        )
    records = []
    for item in reversed(chain):
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(item)
        records.append(
            {
                "relative_path": str(item.GetPath().MakeRelativePath(root.GetPath())),
                "xform_ops": _ordered_xform_ops(item),
                "resets_xform_stack": bool(xformable.GetResetXformStack()),
            }
        )
    return records


def _is_collision_attribute(name: str) -> bool:
    if name in {"radius", "height", "axis", "size"}:
        return True
    if name.startswith("physxCookedData:"):
        return False
    return name.startswith("physics:") or (
        name.startswith("physx") and "Collision:" in name
    )


def _authored_attribute_record(
    prim: Any,
    *,
    predicate,
) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for attribute in prim.GetAttributes():
        name = str(attribute.GetName())
        if not predicate(name) or not attribute.HasAuthoredValueOpinion():
            continue
        value = attribute.Get()
        if value is not None:
            attributes[name] = _json_compatible(value)
    return dict(sorted(attributes.items()))


def _physics_material_records(
    _stage: Any,
    root: Any,
    collision_prim: Any,
) -> list[dict[str, object]]:
    from pxr import UsdShade

    material, relationship = UsdShade.MaterialBindingAPI(
        collision_prim
    ).ComputeBoundMaterial("physics")
    if not material or not material.GetPrim().IsValid():
        cursor = collision_prim
        has_authored_binding = False
        while cursor.IsValid() and cursor.GetPath().HasPrefix(root.GetPath()):
            candidate = cursor.GetRelationship("material:binding:physics")
            if candidate.IsValid() and candidate.GetTargets():
                has_authored_binding = True
                break
            cursor = cursor.GetParent()
        if has_authored_binding:
            raise RuntimeError(
                "physics material binding did not resolve for "
                f"{collision_prim.GetPath()}"
            )
        return []
    material_prim = material.GetPrim()
    target = material_prim.GetPath()
    if not material_prim.IsDefined():
        raise RuntimeError(f"physics material target is not defined: {str(target)!r}")
    relative_or_absolute = (
        str(target.MakeRelativePath(root.GetPath()))
        if target.HasPrefix(root.GetPath())
        else str(target)
    )
    relationship_path = relationship.GetPath()
    normalized_relationship = (
        str(relationship_path.MakeRelativePath(root.GetPath()))
        if relationship_path.HasPrefix(root.GetPath())
        else str(relationship_path)
    )
    return [
        {
            "target": relative_or_absolute,
            "binding_relationship": normalized_relationship,
            "type_name": str(material_prim.GetTypeName()),
            "api_schemas": sorted(
                str(item) for item in material_prim.GetAppliedSchemas()
            ),
            "attributes": _authored_attribute_record(
                material_prim,
                predicate=lambda name: (
                    name.startswith("physics:") or name.startswith("physxMaterial:")
                ),
            ),
        }
    ]


def _collision_manifest_for_root(stage: Any, root_path: str) -> list[dict[str, object]]:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid() or not root.IsDefined():
        raise RuntimeError(f"collision-manifest root is missing: {root_path!r}")
    records = []
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root.GetPath()) or not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue
        attributes = _authored_attribute_record(
            prim,
            predicate=_is_collision_attribute,
        )
        xformable = UsdGeom.Xformable(prim)
        xform_ops = _ordered_xform_ops(prim)
        mesh_topology_sha256 = None
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh_topology_sha256 = _canonical_sha256(
                {
                    "points": _json_compatible(mesh.GetPointsAttr().Get()),
                    "face_vertex_counts": _json_compatible(
                        mesh.GetFaceVertexCountsAttr().Get()
                    ),
                    "face_vertex_indices": _json_compatible(
                        mesh.GetFaceVertexIndicesAttr().Get()
                    ),
                    "hole_indices": _json_compatible(mesh.GetHoleIndicesAttr().Get()),
                    "subdivision_scheme": _json_compatible(
                        mesh.GetSubdivisionSchemeAttr().Get()
                    ),
                    "orientation": _json_compatible(mesh.GetOrientationAttr().Get()),
                }
            )
        relative_path = str(prim.GetPath().MakeRelativePath(root.GetPath()))
        records.append(
            {
                "relative_path": relative_path,
                "type_name": str(prim.GetTypeName()),
                "api_schemas": sorted(
                    str(item)
                    for item in (
                        prim.GetAppliedSchemas()
                        if hasattr(prim, "GetAppliedSchemas")
                        else ()
                    )
                ),
                "attributes": attributes,
                "xform_ops": xform_ops,
                "resets_xform_stack": bool(xformable.GetResetXformStack()),
                "ancestor_xform_chain": _ancestor_xform_chain(root, prim),
                "root_effective_linear_identity": (
                    _root_effective_linear_identity(root)
                ),
                "mesh_topology_sha256": mesh_topology_sha256,
                "physics_materials": _physics_material_records(
                    stage,
                    root,
                    prim,
                ),
            }
        )
    records.sort(key=lambda item: item["relative_path"])
    if not records:
        raise RuntimeError(f"collision-manifest root has no colliders: {root_path!r}")
    return records


def _enabled_collision_paths(stage: Any, root_path: str) -> set[str]:
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    return {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(root.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    }


def _validate_stock_collision_contract(
    stage: Any,
    *,
    robot_prim_path: str,
    can_prim_path: str,
    bin_prim_path: str,
    source_identities: Mapping[str, dict[str, object]],
) -> None:
    from .gripper_linkage_override import (
        _ALL_ORIGINAL_COLLISION_SUBPATHS,
        _require_collision_state,
    )
    from .grocery_bin_collision_override import _require_source_collider
    from .grocery_object_collision_override import _CAN_SOURCE_COLLISION_SUBPATH

    _require_collision_state(
        stage,
        prim_root=robot_prim_path,
        subpaths=_ALL_ORIGINAL_COLLISION_SUBPATHS,
        enabled=True,
        label="stock dynamics calibration gripper",
    )
    gripper_root = f"{robot_prim_path}/Gripper/Robotiq_2F_85"
    expected_gripper = {
        f"{robot_prim_path}/{subpath}" for subpath in _ALL_ORIGINAL_COLLISION_SUBPATHS
    }
    actual_gripper = _enabled_collision_paths(stage, gripper_root)
    if actual_gripper != expected_gripper:
        raise RuntimeError(
            "stock dynamics calibration gripper collision set drifted: "
            f"expected={sorted(expected_gripper)!r}, "
            f"actual={sorted(actual_gripper)!r}"
        )

    expected_can = {f"{can_prim_path}/{_CAN_SOURCE_COLLISION_SUBPATH}"}
    actual_can = _enabled_collision_paths(stage, can_prim_path)
    if actual_can != expected_can:
        raise RuntimeError(
            "stock dynamics calibration can collision set drifted: "
            f"{sorted(actual_can)!r}"
        )
    _require_source_collider(
        stage,
        prim_root=bin_prim_path,
        enabled=True,
        label="stock dynamics calibration bin",
    )
    from .grocery_bin_collision_override import _BIN_SOURCE_COLLISION_SUBPATH

    expected_bin = {f"{bin_prim_path}/{_BIN_SOURCE_COLLISION_SUBPATH}"}
    actual_bin = _enabled_collision_paths(stage, bin_prim_path)
    if actual_bin != expected_bin:
        raise RuntimeError(
            "stock dynamics calibration bin collision set drifted: "
            f"{sorted(actual_bin)!r}"
        )
    if any(
        "cap_collision_proxy" in path
        for path in (*actual_gripper, *actual_can, *actual_bin)
    ):
        raise RuntimeError("stock dynamics calibration contains a CAP collision proxy")
    for source_id, root_path in (
        ("robot", robot_prim_path),
        ("can", can_prim_path),
        ("bin", bin_prim_path),
    ):
        source_identity = source_identities.get(source_id)
        expected_manifest = (
            source_identity.get("source_collision_manifest")
            if source_identity is not None
            else None
        )
        if not isinstance(expected_manifest, list):
            raise RuntimeError(
                f"stock {source_id} source collision manifest is missing"
            )
        actual_manifest = _collision_manifest_for_root(stage, root_path)
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                f"stock {source_id} composed collision manifest differs from "
                "the raw pinned source"
            )


def _validate_and_capture_collision_manifest(
    stage: Any,
    *,
    mode: str,
    robot_prim_path: str,
    can_prim_path: str,
    bin_prim_path: str,
    source_identities: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    mode = _require_mode(mode)
    if mode == "stock":
        _validate_stock_collision_contract(
            stage,
            robot_prim_path=robot_prim_path,
            can_prim_path=can_prim_path,
            bin_prim_path=bin_prim_path,
            source_identities=source_identities,
        )
        analytic_cylinder_setting: str | None = None
    else:
        from .gripper_linkage_override import (
            validate_live_grocery_gripper_collision_contract,
        )
        from .grocery_bin_collision_override import (
            validate_live_grocery_bin_collision_contract,
        )
        from .grocery_object_collision_override import (
            validate_analytic_cylinder_collision_setting,
            validate_live_grocery_can_collision_contract,
        )

        analytic_cylinder_setting = validate_analytic_cylinder_collision_setting()
        validate_live_grocery_gripper_collision_contract(
            stage,
            robot_prim_path=robot_prim_path,
        )
        validate_live_grocery_can_collision_contract(
            stage,
            can_prim_path=can_prim_path,
        )
        validate_live_grocery_bin_collision_contract(
            stage,
            bin_prim_path=bin_prim_path,
        )
    return {
        "mode": mode,
        "analytic_cylinder_setting_override": analytic_cylinder_setting,
        "roots": {
            "robot": _collision_manifest_for_root(stage, robot_prim_path),
            "can": _collision_manifest_for_root(stage, can_prim_path),
            "bin": _collision_manifest_for_root(stage, bin_prim_path),
        },
    }


def capture_dynamics_payload(
    calibration: GroceryDynamicsCalibrationEnvironment,
    *,
    mode: str,
    device: str,
    runtime_versions: dict[str, str],
) -> dict[str, object]:
    """Capture all 11 body dynamics tensors without advancing the environment."""
    mode = _require_mode(mode)
    if mode != calibration.mode:
        raise RuntimeError(
            f"calibration mode relabel is forbidden: "
            f"constructed={calibration.mode!r}, requested={mode!r}"
        )
    unwrapped = calibration.environment.unwrapped
    if unwrapped.num_envs != _NUM_ENVS:
        raise RuntimeError(
            f"dynamics calibration requires num_envs=1, got {unwrapped.num_envs}"
        )
    sim = unwrapped.sim
    physics_step_count_before = sim.get_physics_step_count()
    environment_step_count_before = getattr(unwrapped, "_sim_step_counter", None)
    if physics_step_count_before != 0 or environment_step_count_before != 0:
        raise RuntimeError(
            "dynamics calibration must capture before an environment physics step: "
            f"physics={physics_step_count_before!r}, "
            f"environment={environment_step_count_before!r}"
        )

    manager = sim.physics_manager
    manager_type = manager if isinstance(manager, type) else type(manager)
    if manager_type is not _physx_manager_type():
        manager_class = f"{manager_type.__module__}.{manager_type.__qualname__}"
        raise RuntimeError(
            f"dynamics calibration requires {_EXPECTED_PHYSICS_MANAGER}, "
            f"got {manager_class!r}"
        )
    manager_class = f"{manager_type.__module__}.{manager_type.__qualname__}"
    actual_device = str(sim.device)
    if device == "cuda":
        device_matches = actual_device.startswith("cuda:")
    else:
        device_matches = device == actual_device
    if not device_matches:
        raise RuntimeError(
            f"requested calibration device {device!r} resolved to {actual_device!r}"
        )
    physical_device = _physical_device_identity(actual_device)
    expected_recipe = calibration_recipe(device=actual_device)
    expected_recipe_sha256 = _canonical_sha256(expected_recipe)
    if (
        calibration.recipe != expected_recipe
        or calibration.recipe_sha256 != expected_recipe_sha256
    ):
        raise RuntimeError("calibration common recipe identity drifted")
    expected_mode_contract = mode_contract(mode)
    expected_mode_contract_sha256 = _canonical_sha256(expected_mode_contract)
    if (
        calibration.mode_contract != expected_mode_contract
        or calibration.mode_contract_sha256 != expected_mode_contract_sha256
    ):
        raise RuntimeError("calibration mode contract identity drifted")

    required_versions = ("isaac_sim_app", "isaaclab_physx")
    missing_versions = [
        key
        for key in required_versions
        if not isinstance(runtime_versions.get(key), str)
        or runtime_versions[key] in ("", "unavailable")
    ]
    if missing_versions:
        raise RuntimeError(
            f"runtime version identity is incomplete: {missing_versions!r}"
        )
    engine_version = (
        f"isaac-sim:{runtime_versions['isaac_sim_app']};"
        f"isaaclab-physx:{runtime_versions['isaaclab_physx']}"
    )

    scene = unwrapped.scene
    robot = scene["robot"]
    from .grocery_scene_spec import CAP_GROCERY_BIN_ASSET, CAP_GROCERY_OBJECT_ASSET

    can = scene[CAP_GROCERY_OBJECT_ASSET]
    bin_asset = scene[CAP_GROCERY_BIN_ASSET]
    env_paths = tuple(scene.env_prim_paths)
    if len(env_paths) != 1:
        raise RuntimeError(
            f"dynamics calibration requires exactly one environment path, got {env_paths!r}"
        )
    env_path = env_paths[0]
    robot_prim_path = f"{env_path}/Robot"
    can_prim_path = f"{env_path}/{CAP_GROCERY_OBJECT_ASSET}"
    bin_prim_path = f"{env_path}/{CAP_GROCERY_BIN_ASSET}"

    body_names = tuple(robot.body_names)
    robot_masses = _to_builtin(robot.data.body_mass)
    robot_com_poses = _to_builtin(robot.data.body_com_pose_b)
    robot_inertias = _to_builtin(robot.data.body_inertia)
    simulation_view, simulation_view_attestation = _attest_simulation_view(
        manager,
        sim,
        actual_device=actual_device,
    )
    stage = scene.stage
    robot_body_indices = {
        label: _require_unique_body_index(body_names, body_name)
        for label, body_name, _ in _ROBOT_BODY_ROSTER
    }
    robot_expected_paths = {
        robot_body_indices[label]: f"{robot_prim_path}/{relative_path}"
        for label, _, relative_path in _ROBOT_BODY_ROSTER
    }
    asset_view_attestations = {
        "robot": _attest_articulation_root_view(
            robot,
            simulation_view=simulation_view,
            actual_device=actual_device,
            expected_root_path=robot_prim_path,
            expected_body_count=len(body_names),
            expected_indexed_paths=robot_expected_paths,
        ),
        "can": _attest_rigid_asset_root_view(
            can,
            label="can",
            simulation_view=simulation_view,
            actual_device=actual_device,
            expected_path=can_prim_path,
        ),
        "bin": _attest_rigid_asset_root_view(
            bin_asset,
            label="bin",
            simulation_view=simulation_view,
            actual_device=actual_device,
            expected_path=bin_prim_path,
        ),
    }
    records = [
        _attested_body_record(
            stage=stage,
            simulation_view=simulation_view,
            label=label,
            expected_path=f"{robot_prim_path}/{relative_path}",
            asset_body_index=robot_body_indices[label],
            asset_masses=robot_masses,
            asset_com_poses=robot_com_poses,
            asset_inertias=robot_inertias,
        )
        for label, body_name, relative_path in _ROBOT_BODY_ROSTER
    ]
    for label, asset, prim_path in (
        ("can", can, can_prim_path),
        ("bin", bin_asset, bin_prim_path),
    ):
        records.append(
            _attested_body_record(
                stage=stage,
                simulation_view=simulation_view,
                label=label,
                expected_path=prim_path,
                asset_body_index=0,
                asset_masses=_to_builtin(asset.data.body_mass),
                asset_com_poses=_to_builtin(asset.data.body_com_pose_b),
                asset_inertias=_to_builtin(asset.data.body_inertia),
            )
        )
    if len(records) != _EXPECTED_BODY_COUNT:
        raise RuntimeError(f"dynamics calibration body roster drifted: {len(records)}")

    from pxr import UsdGeom, UsdPhysics

    simulation_timestamps = {
        "robot": _finite_float(
            getattr(robot.data, "_sim_timestamp", None),
            label="robot simulation timestamp",
        ),
        "can": _finite_float(
            getattr(can.data, "_sim_timestamp", None),
            label="can simulation timestamp",
        ),
        "bin": _finite_float(
            getattr(bin_asset.data, "_sim_timestamp", None),
            label="bin simulation timestamp",
        ),
    }
    if len(set(simulation_timestamps.values())) != 1:
        raise RuntimeError(
            "dynamics calibration tensors are not from one simulation timestamp: "
            f"{simulation_timestamps!r}"
        )
    timestamp = next(iter(simulation_timestamps.values()))
    if not math.isclose(
        timestamp,
        calibration.simulation_dt_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "dynamics calibration buffers were not captured after exactly one "
            f"scene-prime update: {timestamp!r}"
        )

    current_collision_manifest = _validate_and_capture_collision_manifest(
        stage,
        mode=mode,
        robot_prim_path=robot_prim_path,
        can_prim_path=can_prim_path,
        bin_prim_path=bin_prim_path,
        source_identities=calibration.source_identities,
    )
    current_collision_manifest_sha256 = _canonical_sha256(current_collision_manifest)
    if (
        current_collision_manifest != calibration.collision_manifest
        or current_collision_manifest_sha256 != calibration.collision_manifest_sha256
    ):
        raise RuntimeError("live collision manifest changed before capture")

    _require_source_snapshots_unchanged(
        calibration.source_identities,
        calibration.snapshot_uris,
    )
    physics_step_count_after = sim.get_physics_step_count()
    environment_step_count_after = getattr(unwrapped, "_sim_step_counter", None)
    if physics_step_count_after != 0 or environment_step_count_after != 0:
        raise RuntimeError(
            "dynamics calibration advanced while collecting evidence: "
            f"physics={physics_step_count_after!r}, "
            f"environment={environment_step_count_after!r}"
        )
    return {
        "schema": _SCHEMA,
        "mode": mode,
        "recipe": calibration.recipe,
        "recipe_sha256": calibration.recipe_sha256,
        "mode_contract": calibration.mode_contract,
        "mode_contract_sha256": calibration.mode_contract_sha256,
        "collision_manifest": calibration.collision_manifest,
        "collision_manifest_sha256": calibration.collision_manifest_sha256,
        "device": actual_device,
        "requested_device": device,
        "sample_phase": (
            "after_physx_warmup_and_scene_buffer_prime_before_environment_step"
        ),
        "runner_commanded_physics_steps": 0,
        "physics_step_count_before_capture": physics_step_count_before,
        "physics_step_count_after_capture": physics_step_count_after,
        "environment_step_count_before_capture": environment_step_count_before,
        "environment_step_count_after_capture": environment_step_count_after,
        "scene_buffer_prime_count": 1,
        "simulation_timestamps_s": simulation_timestamps,
        "num_envs": _NUM_ENVS,
        "simulation_dt_s": _finite_float(
            calibration.simulation_dt_s,
            label="simulation dt",
        ),
        "units": {
            "stage_meters_per_unit": _finite_float(
                UsdGeom.GetStageMetersPerUnit(stage),
                label="stage meters per unit",
            ),
            "stage_kilograms_per_unit": _finite_float(
                UsdPhysics.GetStageKilogramsPerUnit(stage),
                label="stage kilograms per unit",
            ),
            "stage_time_codes_per_second": _finite_float(
                stage.GetTimeCodesPerSecond(),
                label="stage time codes per second",
            ),
            "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "com_pose_order": "x,y,z,qx,qy,qz,qw",
            "inertia_order": "xx,xy,xz,yx,yy,yz,zx,zy,zz",
        },
        "engine": {
            "engine_id": _ENGINE_ID,
            "engine_version": engine_version,
            "physics_manager_class": manager_class,
            "isaaclab_tensor_backend": str(sim.backend),
            "actual_device": actual_device,
            "physical_device": physical_device,
            "simulation_view": simulation_view_attestation,
        },
        "isaaclab_asset_views": asset_view_attestations,
        "runtime_versions": dict(sorted(runtime_versions.items())),
        "source_identities": calibration.source_identities,
        "bodies": records,
    }


def canonical_payload_bytes(payload: dict[str, object]) -> bytes:
    """Encode one calibration payload with deterministic JSON ordering."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_calibration_artifact(
    output_path: str | os.PathLike[str],
    payload: dict[str, object],
) -> tuple[Path, str]:
    """Atomically write one canonical calibration artifact."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_payload_bytes(payload)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination, hashlib.sha256(content).hexdigest()


def make_grocery_dynamics_calibration_environment(
    *,
    device: str,
    mode: str,
) -> GroceryDynamicsCalibrationEnvironment:
    """Build a stock or proxy grocery scene solely for dynamics calibration."""
    mode = _require_mode(mode)

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    from .franka_env import (
        _configure_cap_droid_embodiment,
        _configure_cap_environment_profile,
        _configure_cap_grocery_embodiment,
        _make_cap_grocery_assets,
        _validate_cap_fixed_robot_anchor,
    )
    from .gripper_linkage_override import (
        _open_and_validate_source as validate_robot_source,
    )
    from .gripper_linkage_override import apply_grocery_gripper_linkage_override
    from .grocery_bin_collision_override import (
        _open_and_validate_source as validate_bin_source,
    )
    from .grocery_bin_collision_override import (
        apply_grocery_bin_collision_override,
    )
    from .grocery_object_collision_override import (
        AnalyticCylinderCollisionSettingOverride,
        _require_can_source_contract,
        _require_can_source_identity,
        apply_grocery_can_collision_override,
    )
    from .grocery_scene_spec import (
        CAP_GROCERY_BIN_ASSET,
        CAP_GROCERY_OBJECT_ASSET,
    )

    with ExitStack() as construction:
        registry = AssetRegistry()
        embodiment = registry.get_asset_by_name("droid_abs_joint_pos")(
            enable_cameras=False
        )
        stand_spawn = sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            visible=False,
        )
        _configure_cap_grocery_embodiment(embodiment)
        _configure_cap_droid_embodiment(
            embodiment,
            stand_spawn=stand_spawn,
            initial_gripper_closed=False,
        )
        assets = _make_cap_grocery_assets(registry, sim_utils)
        assets_by_name = {asset.name: asset for asset in assets}
        can_asset = assets_by_name[CAP_GROCERY_OBJECT_ASSET]
        bin_asset = assets_by_name[CAP_GROCERY_BIN_ASSET]
        _require_object_spawn_scale(
            can_asset,
            label="CAP grocery can",
            expected=_CAN_SCALE,
        )
        _require_object_spawn_scale(
            bin_asset,
            label="CAP grocery bin",
            expected=_BIN_SCALE,
        )
        source_uris = {
            "robot": str(embodiment.scene_config.robot.spawn.usd_path),
            "can": _source_uri(can_asset),
            "bin": _source_uri(bin_asset),
        }
        owned_resources: list[Any] = []
        (
            source_identities,
            snapshot_uris,
            source_snapshots,
        ) = _capture_verified_source_snapshots(source_uris)
        construction.callback(source_snapshots.close)
        owned_resources.append(source_snapshots)
        embodiment.scene_config.robot.spawn.usd_path = snapshot_uris["robot"]
        _set_object_source_uri(can_asset, snapshot_uris["can"])
        _set_object_source_uri(bin_asset, snapshot_uris["bin"])

        validate_robot_source(snapshot_uris["robot"])
        validate_bin_source(snapshot_uris["bin"])
        _require_can_source_identity(snapshot_uris["can"])
        from pxr import Usd

        can_source_stage = Usd.Stage.Open(snapshot_uris["can"])
        if can_source_stage is None:
            raise RuntimeError(
                f"failed to open CAP grocery can USD {snapshot_uris['can']!r}"
            )
        _require_can_source_contract(can_source_stage)

        if mode == "proxy":
            cylinder_setting = AnalyticCylinderCollisionSettingOverride()
            construction.callback(cylinder_setting.close)
            owned_resources.append(cylinder_setting)
            for override in (
                apply_grocery_gripper_linkage_override(embodiment.scene_config),
                apply_grocery_can_collision_override(can_asset),
                apply_grocery_bin_collision_override(bin_asset),
            ):
                construction.callback(override.close)
                owned_resources.append(override)

        scene = Scene(assets=assets)

        def configure_profile(cfg):
            return _configure_cap_environment_profile(
                cfg,
                enable_cameras=False,
            )

        description = IsaacLabArenaEnvironment(
            name=f"CAP-Grocery-Dynamics-Calibration-{mode}-B1-v0",
            scene=scene,
            embodiment=embodiment,
            task=NoTask(),
            env_cfg_callback=configure_profile,
        )
        builder = ArenaEnvBuilder(
            description,
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                device=device,
            ),
        )
        environment, cfg = builder.make_registered_and_return_cfg()
        construction.callback(environment.close)
        if (
            cfg.seed != _SCENE_SEED
            or cfg.sim.dt != _SIMULATION_DT_S
            or cfg.decimation != _DECIMATION
            or environment.unwrapped.num_envs != _NUM_ENVS
        ):
            raise RuntimeError(
                "CAP calibration common recipe mismatch: "
                f"seed={cfg.seed}, num_envs={environment.unwrapped.num_envs}, "
                f"dt={cfg.sim.dt}, decimation={cfg.decimation}"
            )
        _validate_cap_fixed_robot_anchor(environment.unwrapped.scene["robot"])
        actual_device = str(environment.unwrapped.sim.device)
        if device == "cuda":
            device_matches = actual_device.startswith("cuda:")
        else:
            device_matches = actual_device == device
        if not device_matches:
            raise RuntimeError(
                f"requested calibration device {device!r} resolved to {actual_device!r}"
            )
        recipe = calibration_recipe(device=actual_device)
        mode_identity = mode_contract(mode)
        env_paths = tuple(environment.unwrapped.scene.env_prim_paths)
        if len(env_paths) != 1:
            raise RuntimeError(
                "dynamics calibration requires exactly one environment path, "
                f"got {env_paths!r}"
            )
        env_path = env_paths[0]
        collision_manifest = _validate_and_capture_collision_manifest(
            environment.unwrapped.scene.stage,
            mode=mode,
            robot_prim_path=f"{env_path}/Robot",
            can_prim_path=f"{env_path}/{CAP_GROCERY_OBJECT_ASSET}",
            bin_prim_path=f"{env_path}/{CAP_GROCERY_BIN_ASSET}",
            source_identities=source_identities,
        )
        result = GroceryDynamicsCalibrationEnvironment(
            environment=environment,
            mode=mode,
            source_uris=source_uris,
            snapshot_uris=snapshot_uris,
            source_identities=source_identities,
            simulation_dt_s=float(cfg.sim.dt),
            recipe=recipe,
            recipe_sha256=_canonical_sha256(recipe),
            mode_contract=mode_identity,
            mode_contract_sha256=_canonical_sha256(mode_identity),
            collision_manifest=collision_manifest,
            collision_manifest_sha256=_canonical_sha256(collision_manifest),
            _owned_resources=tuple(owned_resources),
        )
        construction.pop_all()
        return result
