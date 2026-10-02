# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Add simulation physics to the workstation assets authored in Blender."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

PhysicsKind = Literal["static", "rigid", "kinematic"]
_AUTHORING_VERSION = 4


def default_asset_root() -> Path:
    """Return the external cache containing the Blender asset manifest and USDs."""
    return Path.home() / ".cache/isaaclab_arena/return_to_service/assets"


@dataclass(frozen=True)
class PreparedAssets:
    """Resolve source geometry and cache independent USD physics overlays."""

    asset_root: Path
    cache_root: Path
    manifest: dict[str, Any]

    def record(self, source_name: str) -> dict[str, Any]:
        """Return one source asset's manifest record."""
        assert source_name in self.manifest["assets"], f"Missing Blender asset: {source_name}"
        return self.manifest["assets"][source_name]

    def source_path(self, source_name: str) -> Path:
        """Return the existing source USD for an asset."""
        path = (self.asset_root / self.record(source_name)["file"]).resolve()
        assert path.is_file(), f"Missing Blender export: {path}"
        return path

    def affordance(self, source_name: str, name: str) -> Any:
        """Read a Blender-authored socket, grasp, or joint frame."""
        record = self.record(source_name)
        assert name in record.get(
            "affordances", {}
        ), f"Asset {source_name} has no {name} affordance. Rebuild the Blender asset bundle from current source."
        return record["affordances"][name]

    def asset_usd(self, source_name: str, kind: PhysicsKind = "static") -> Path:
        """Return a cached USD referencing the source and adding the requested physics."""
        assert kind in ("static", "rigid", "kinematic"), f"Unsupported physics kind: {kind}"
        path = self._cache_path(source_name, [source_name], {"kind": kind})
        if not path.is_file():
            stage, root = _new_stage()
            _reference_source(self, root, source_name)
            if self.record(source_name).get("collision_count", 0):
                _configure_colliders(root)
            else:
                assert kind == "static", f"Dynamic source {source_name} must provide collision geometry"
            if kind != "static":
                _configure_rigid_body(root, self.record(source_name)["mass_kg"], kinematic=kind == "kinematic")
            _save_stage(stage, path)
        return path

    def button_usd(self) -> Path:
        """Return a fixed-base, spring-return button with a physical prismatic joint."""
        from pxr import Gf, UsdGeom, UsdPhysics

        settings = {"travel_m": 0.006, "cap_rest_xyz": (0.0, 0.0, 0.012)}
        settings.update(self.record("test_button").get("affordances", {}).get("button_joint", {}))
        path = self._cache_path("push_button", ["button_base", "test_button"], settings)
        if path.is_file():
            return path
        stage, root = _new_stage()
        _articulation_root(stage, root)
        base = _rigid_link(self, stage, "base", "button_base")
        cap = _rigid_link(self, stage, "cap", "test_button")
        cap_xyz = tuple(settings["cap_rest_xyz"])
        UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(*cap_xyz))
        _fix_base(stage, base)
        joint = UsdPhysics.PrismaticJoint.Define(stage, "/Asset/press")
        _connect_joint(joint, base, cap, cap_xyz, (0.0, 0.0, 0.0))
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(-float(settings["travel_m"]))
        joint.CreateUpperLimitAttr(0.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateStiffnessAttr(180.0)
        drive.CreateDampingAttr(1.5)
        drive.CreateMaxForceAttr(8.0)
        _save_stage(stage, path)
        return path

    def case_usd(self) -> Path:
        """Return an anchored case with independent passive lid and latch joints."""
        from pxr import Gf, UsdGeom, UsdPhysics

        affordances = self.record("case_base")["affordances"]
        lid_pose = affordances["lid_closed_pose"]
        hinge_settings = affordances["hinge"]
        latch_settings = affordances["latch"]
        assert all(
            "angle_limits_degrees" in settings for settings in (hinge_settings, latch_settings)
        ), "Rebuild the Blender asset bundle to include authored case joint limits."
        assert all(
            name in latch_settings for name in ("static_friction_effort_nm", "dynamic_friction_effort_nm")
        ), "Rebuild the Blender asset bundle to include the case latch's passive friction settings."
        assert tuple(lid_pose["rotation_xyzw"]) == (0, 0, 0, 1), "The case lid source must use its hinge frame"
        assert tuple(affordances["hinge"]["axis"]) == (1, 0, 0), "The case hinge must rotate about local X"
        assert tuple(affordances["latch"]["axis"]) == (1, 0, 0), "The case latch must rotate about local X"
        settings = {
            "hinge_xyz": tuple(lid_pose["position_xyz"]),
            "lid_translation_xyz": tuple(lid_pose["position_xyz"]),
            "latch_xyz": tuple(affordances["latch"]["position_xyz"]),
            "latch_translation_xyz": tuple(affordances["latch"]["position_xyz"]),
            "hinge_lower_deg": float(hinge_settings["angle_limits_degrees"][0]),
            "hinge_upper_deg": float(hinge_settings["angle_limits_degrees"][1]),
            "latch_lower_deg": float(latch_settings["angle_limits_degrees"][0]),
            "latch_upper_deg": float(latch_settings["angle_limits_degrees"][1]),
            "latch_static_friction_effort_nm": float(latch_settings["static_friction_effort_nm"]),
            "latch_dynamic_friction_effort_nm": float(latch_settings["dynamic_friction_effort_nm"]),
        }
        path = self._cache_path("service_case", ["case_base", "case_lid", "case_latch"], settings)
        if path.is_file():
            return path
        stage, root = _new_stage()
        _articulation_root(stage, root)
        base = _rigid_link(self, stage, "base", "case_base")
        lid = _rigid_link(self, stage, "lid", "case_lid")
        latch = _rigid_link(self, stage, "latch", "case_latch")
        _fix_base(stage, base)
        for link, prefix in ((lid, "lid"), (latch, "latch")):
            position = tuple(settings[f"{prefix}_translation_xyz"])
            UsdGeom.Xformable(link).AddTranslateOp().Set(Gf.Vec3d(*position))
        for name, child, translation_name in (("hinge", lid, "lid"), ("latch", latch, "latch")):
            pivot = tuple(settings[f"{name}_xyz"])
            translation = tuple(settings[f"{translation_name}_translation_xyz"])
            child_pivot = tuple(pivot[index] - translation[index] for index in range(3))
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/Joints/{name}")
            _connect_joint(joint, base, child, pivot, child_pivot)
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(float(settings[f"{name}_lower_deg"]))
            joint.CreateUpperLimitAttr(float(settings[f"{name}_upper_deg"]))
            if name == "latch":
                _configure_angular_friction(
                    joint.GetPrim(),
                    settings["latch_static_friction_effort_nm"],
                    settings["latch_dynamic_friction_effort_nm"],
                )
        _save_stage(stage, path)
        return path

    def _cache_path(self, name: str, sources: list[str], settings: dict[str, Any]) -> Path:
        dependencies = []
        for source in sources:
            source_path = self.source_path(source)
            stat = source_path.stat()
            dependencies.append((str(source_path), stat.st_size, stat.st_mtime_ns, self.record(source)))
        specification = {"version": _AUTHORING_VERSION, "sources": dependencies, "settings": settings}
        digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_root / f"{name}_{digest}.usda"


def prepare_assets(asset_root: str | Path | None = None, cache_root: str | Path | None = None) -> PreparedAssets:
    """Load the Blender manifest; author requested physics overlays lazily.

    Args:
        asset_root: Directory containing manifest.json and the exported USD assets.
        cache_root: Destination for derived physics-only USD layers.

    Returns:
        A library whose methods resolve source geometry and prepare simulation assets.
    """
    root = Path(asset_root).expanduser().resolve() if asset_root is not None else default_asset_root()
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file(), (
        f"Missing Blender asset manifest: {manifest_path}. "
        "Generate the workstation with Blender first; see return_to_service/asset_source/README.md."
    )
    manifest = json.loads(manifest_path.read_text())
    validate_asset_manifest(manifest, root)
    destination = Path(cache_root).expanduser().resolve() if cache_root is not None else root / "physics"
    destination.mkdir(parents=True, exist_ok=True)
    return PreparedAssets(root, destination, manifest)


def validate_asset_manifest(manifest: dict[str, Any], asset_root: Path) -> None:
    """Validate the asset coordinate contract once, without importing simulation packages.

    Args:
        manifest: Parsed Blender export manifest.
        asset_root: Directory containing all source USD files referenced by the manifest.
    """
    assert isinstance(manifest, dict), "The asset manifest must be an object."
    assert manifest.get("schema_version") == 1, "Unsupported return-to-service asset manifest version"
    assert manifest.get("units") == "meters" and manifest.get("up_axis") == "Z", "Assets must use meters and Z-up"
    records = manifest.get("assets")
    assert isinstance(records, dict) and records, "Asset manifest is empty"
    asset_root = asset_root.resolve()
    for name, record in records.items():
        assert isinstance(name, str) and name and isinstance(record, dict), "Asset entries require a name and record."
        filename = record.get("file")
        assert isinstance(filename, str) and filename, f"Asset {name} must specify its USD file."
        relative_path = Path(filename)
        source = (asset_root / relative_path).resolve()
        assert not relative_path.is_absolute() and source.is_relative_to(asset_root), f"Asset {name} escapes its root."
        assert source.is_file(), f"Missing Blender export: {source}"
        assert source.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}, f"Asset {name} is not a USD file."
        root_prim = record.get("root_prim")
        assert isinstance(root_prim, str) and re.fullmatch(
            r"(?:/[A-Za-z_]\w*)+", root_prim
        ), f"Invalid root prim: {name}"
        _validate_finite_values(record, name)
        geometry_fields = {"bounds_min", "bounds_max", "mass_kg", "collision_count"}
        if geometry_fields.intersection(record):
            assert geometry_fields.issubset(record), f"Incomplete geometry contract: {name}"
            _validate_bounds([record["bounds_min"], record["bounds_max"]], name, positive_volume=True)
            mass = record["mass_kg"]
            assert isinstance(mass, (int, float)) and not isinstance(mass, bool) and mass > 0, f"Invalid mass: {name}"
            count = record["collision_count"]
            assert (
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
            ), f"Invalid collider count: {name}"
        affordances = record.get("affordances", {})
        assert isinstance(affordances, dict), f"Affordances must be an object: {name}"
        _validate_affordances(affordances, name, root_prim)


def _validate_finite_values(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_values(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_values(child, context)
    elif isinstance(value, (int, float)):
        assert math.isfinite(value), f"Non-finite asset value: {context}"


def _validate_vector(value: Any, size: int, context: str) -> None:
    assert isinstance(value, (list, tuple)) and len(value) == size, f"Expected {size} coordinates: {context}"
    assert all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ), f"Coordinates must be numbers: {context}"


def _validate_bounds(value: Any, context: str, *, positive_volume: bool = False) -> None:
    assert isinstance(value, (list, tuple)) and len(value) == 2, f"Expected two bound corners: {context}"
    lower, upper = value
    _validate_vector(lower, 3, context)
    _validate_vector(upper, 3, context)
    assert all(low <= high for low, high in zip(lower, upper, strict=True)), f"Reversed asset bounds: {context}"
    if positive_volume:
        assert all(low < high for low, high in zip(lower, upper, strict=True)), f"Empty asset bounds: {context}"


def _validate_pose(value: Any, context: str) -> None:
    if isinstance(value, dict):
        assert "position_xyz" in value, f"A pose must declare position_xyz: {context}"
        _validate_vector(value["position_xyz"], 3, context)
        quaternion = value.get("rotation_xyzw", [0, 0, 0, 1])
        _validate_vector(quaternion, 4, context)
        assert math.isclose(
            sum(component * component for component in quaternion), 1.0, abs_tol=1e-4
        ), f"Pose quaternion must be normalized XYZW: {context}"
    else:
        _validate_vector(value, 3, context)


def _validate_affordances(values: dict[str, Any], context: str, root_prim: str) -> None:
    friction_fields = ("static_friction_effort_nm", "dynamic_friction_effort_nm")
    if any(name in values for name in friction_fields):
        assert all(name in values for name in friction_fields), f"Incomplete joint friction settings: {context}"
        static_effort, dynamic_effort = (values[name] for name in friction_fields)
        assert all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in (static_effort, dynamic_effort)
        ), f"Joint friction efforts must be numbers: {context}"
        assert 0 <= dynamic_effort <= static_effort, f"Expected 0 <= dynamic <= static joint friction: {context}"
    for key, value in values.items():
        label = f"{context}.{key}"
        if key.endswith("_bounds"):
            _validate_bounds(value, label, positive_volume=key != "usable_bounds")
        elif key.endswith("_socket") or key.endswith("_pose") or key == "grasp":
            _validate_pose(value, label)
        elif key.endswith("_poses"):
            assert isinstance(value, (dict, list, tuple)), f"Expected named or sequential poses: {label}"
            poses = value.values() if isinstance(value, dict) else value
            for pose in poses:
                _validate_pose(pose, label)
        elif key.endswith("_paths"):
            assert isinstance(value, dict), f"Expected named USD paths: {label}"
            for path in value.values():
                assert (
                    isinstance(path, str) and path.startswith(root_prim + "/") and ".." not in path.split("/")
                ), f"Instrument paths must lie beneath the asset root: {label}"
        elif isinstance(value, dict):
            if "position_xyz" in value:
                _validate_pose(value, label)
            _validate_affordances(value, label, root_prim)
        elif key in {"axis", "extraction_axis", "press_axis"}:
            _validate_vector(value, 3, label)
            assert math.isclose(
                sum(component * component for component in value), 1.0, abs_tol=1e-4
            ), f"Expected a normalized axis: {label}"
        elif key == "angle_limits_degrees":
            _validate_vector(value, 2, label)
            assert value[0] < value[1], f"Joint angle limits must be ordered: {label}"
        elif key == "x_range":
            _validate_vector(value, 2, label)
            assert value[0] < value[1], f"Cylinder endpoints must be ordered: {label}"
        elif key == "center_yz":
            _validate_vector(value, 2, label)
        elif key in {"radius", "hollow_bore_radius", "jaw_span", "travel", "insertion_depth", "socket_separation"}:
            assert (
                isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
            ), f"Physical dimensions must be positive: {label}"


def _configure_angular_friction(prim, static_effort_nm: float, dynamic_effort_nm: float) -> None:
    """Author passive revolute-joint friction in torque units for PhysX articulations."""
    assert prim.ApplyAPI(
        "PhysxJointAxisAPI", "angular"
    ), "Case latch friction requires the Isaac Sim 6 joint-axis schema"
    for name, value in (("staticFrictionEffort", static_effort_nm), ("dynamicFrictionEffort", dynamic_effort_nm)):
        attribute = prim.GetAttribute(f"physxJointAxis:angular:{name}")
        assert attribute and str(attribute.GetTypeName()) == "float", f"Missing PhysX joint friction attribute: {name}"
        attribute.Set(value)


def _new_stage():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    return stage, root


def _reference_source(library: PreparedAssets, target, source_name: str) -> None:
    record = library.record(source_name)
    target.GetReferences().AddReference(str(library.source_path(source_name)), record.get("root_prim", "/Asset"))
    assert not target.IsInstanceProxy(), "Physics authoring requires editable source references"


def _configure_rigid_body(prim, mass_kg: float, *, kinematic: bool = False) -> None:
    from pxr import PhysxSchema, UsdPhysics

    assert mass_kg > 0.0, f"Rigid body {prim.GetPath()} must have positive mass"
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(kinematic)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass_kg))
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateLinearDampingAttr(0.04)
    physx.CreateAngularDampingAttr(0.08)
    physx.CreateEnableCCDAttr(not kinematic)
    physx.CreateSolverPositionIterationCountAttr(12)
    physx.CreateSolverVelocityIterationCountAttr(4)


def _configure_colliders(root) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(root.GetStage(), root.GetPath().AppendChild("PhysicsMaterial"))
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(0.8)
    physics.CreateDynamicFrictionAttr(0.65)
    physics.CreateRestitutionAttr(0.0)
    collision_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_count += 1
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    assert collision_count > 0, f"Blender asset {root.GetPath()} has no authored collision proxies"


def _rigid_link(library: PreparedAssets, stage, name: str, source_name: str):
    from pxr import UsdGeom

    link = UsdGeom.Xform.Define(stage, f"/Asset/{name}").GetPrim()
    _reference_source(library, link, source_name)
    _configure_colliders(link)
    _configure_rigid_body(link, library.record(source_name)["mass_kg"])
    return link


def _articulation_root(stage, root) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.ArticulationRootAPI.Apply(root)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(root)
    articulation.CreateEnabledSelfCollisionsAttr(False)
    articulation.CreateSolverPositionIterationCountAttr(16)
    articulation.CreateSolverVelocityIterationCountAttr(4)


def _fix_base(stage, base) -> None:
    from pxr import UsdPhysics

    joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/root_joint")
    joint.CreateBody1Rel().SetTargets([base.GetPath()])


def _connect_joint(joint, parent, child, parent_pivot, child_pivot) -> None:
    from pxr import Gf

    joint.CreateBody0Rel().SetTargets([parent.GetPath()])
    joint.CreateBody1Rel().SetTargets([child.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*parent_pivot))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*child_pivot))
    joint.CreateCollisionEnabledAttr(False)


def _save_stage(stage, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.stem}_{uuid4().hex}.usda")
    try:
        assert stage.GetRootLayer().Export(str(temporary)), f"Could not export physics overlay: {temporary}"
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
