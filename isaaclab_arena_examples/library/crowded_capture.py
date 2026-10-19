# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture generic robot and rigid-object geometry with measured PhysX poses."""

from __future__ import annotations

import copy
import json
import math
import numpy as np
from pathlib import Path

from isaaclab_arena_examples.bento.capture import _mesh, _vertex_colors, pose_matrix


class GenericCapture:
    """Store robot links and arbitrary rigid objects for offline video rendering."""

    def __init__(self, sim, robot, objects, directory: Path, fps: float):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.robots = list(robot.values()) if isinstance(robot, dict) else [robot]
        self.objects = list(objects.values()) if hasattr(objects, "values") else list(objects)
        self.directory = Path(directory)
        self.stage = sim.stage
        self.dt = sim.get_physics_dt()
        assert (
            math.isfinite(fps) and 0 < fps <= 1 / self.dt
        ), "Capture FPS must be positive and within the physics rate."
        self.stride = max(1, round(1 / (self.dt * fps)))
        self.times, self.poses, self.phases, self.extras = [], [], [], []
        self.directory.mkdir(parents=True, exist_ok=True)
        assert not (self.directory / "scene.npz").exists(), "Choose a fresh capture directory."
        prims = list(Usd.PrimRange(self.stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
        self.body_paths = ["static"]
        for arm in self.robots:
            robot_root = str(arm.cfg.prim_path).rstrip("/")
            rigid_paths = {
                prim.GetName(): str(prim.GetPath())
                for prim in prims
                if str(prim.GetPath()).startswith(robot_root + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            }
            assert all(name in rigid_paths for name in arm.body_names), "Cannot resolve every robot rigid body."
            self.body_paths.extend(rigid_paths[name] for name in arm.body_names)
        self.body_paths.extend(str(obj.cfg.prim_path) for obj in self.objects)
        self.extra_paths = [
            str(prim.GetPath())
            for prim in prims
            if prim.GetName() in ("LibraryJawLeft", "LibraryJawRight") and prim.IsA(UsdGeom.Xform)
        ]
        self.extra_owners = []
        for path in self.extra_paths:
            parent = next(
                (
                    index
                    for index, body in sorted(enumerate(self.body_paths[1:], 1), key=lambda item: -len(item[1]))
                    if path.startswith(body + "/")
                ),
                None,
            )
            assert parent is not None, "No rigid parent for animated jaw."
            self.extra_owners.append(parent)
        self.body_paths.extend(self.extra_paths)
        assert len(set(self.body_paths)) == len(self.body_paths), "Capture body paths must be unique."
        xforms = UsdGeom.XformCache()
        inverses = [np.eye(4)]
        for path in self.body_paths[1:]:
            prim = self.stage.GetPrimAtPath(path)
            assert prim.IsValid(), f"Missing capture body: {path}"
            inverses.append(np.linalg.inv(np.asarray(xforms.GetLocalToWorldTransform(prim)).T))
        owners = sorted(enumerate(self.body_paths[1:], 1), key=lambda item: -len(item[1]))
        vertices, normals, colors, body_ids, triangles = [], [], [], [], []
        material_cache = {}
        offset = 0
        for prim in prims:
            path = str(prim.GetPath())
            if not path.startswith("/World/") or not prim.IsA(UsdGeom.Gprim):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            if prim.IsA(UsdGeom.Capsule):
                import trimesh

                shape = UsdGeom.Capsule(prim)
                mesh = trimesh.creation.capsule(height=shape.GetHeightAttr().Get(), radius=shape.GetRadiusAttr().Get())
                axis = shape.GetAxisAttr().Get()
                if axis != "Z":
                    direction = (1, 0, 0) if axis == "X" else (0, 1, 0)
                    mesh.apply_transform(trimesh.geometry.align_vectors((0, 0, 1), direction))
            else:
                mesh = _mesh(prim)
            if mesh is None or not len(mesh.faces):
                continue
            vertex_colors = _vertex_colors(prim, mesh, material_cache)
            owner = next((index for index, body in owners if path == body or path.startswith(body + "/")), 0)
            # USD geometry and its owner must share the same authored transform basis.
            # Measured initial poses can differ from USD, especially for driven fingers.
            transform = inverses[owner] @ np.asarray(xforms.GetLocalToWorldTransform(prim)).T
            local_vertices = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
            local_normals = np.asarray(mesh.vertex_normals) @ np.linalg.inv(transform[:3, :3])
            local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-8)
            vertices.append(local_vertices.astype(np.float32))
            normals.append(local_normals.astype(np.float32))
            colors.append(vertex_colors.astype(np.float32))
            body_ids.append(np.full(len(mesh.vertices), owner, dtype=np.int32))
            triangles.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
            offset += len(mesh.vertices)
        assert vertices, "No visible USD geometry found for capture."
        self.geometry = {
            "vertices": np.concatenate(vertices),
            "normals": np.concatenate(normals),
            "colors": np.concatenate(colors),
            "vertex_body_ids": np.concatenate(body_ids),
            "triangles": np.concatenate(triangles),
        }
        assert all(np.isfinite(value).all() for value in self.geometry.values()), "Nonfinite captured geometry."
        print(f"[capture] {offset} vertices, {len(self.geometry['triangles'])} triangles", flush=True)

    def read_poses(self) -> np.ndarray:
        """Read static, robot-link, and object world poses in archive body order."""
        import trimesh

        from pxr import UsdGeom

        poses = [np.asarray([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)]
        poses.extend(arm.data.body_link_pose_w.torch[0].detach().cpu().numpy() for arm in self.robots)
        poses.extend(obj.data.root_pose_w.torch[0].detach().cpu().numpy()[None] for obj in self.objects)
        base = np.concatenate(poses).astype(np.float32)
        for path, owner in zip(self.extra_paths, self.extra_owners):
            local = np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetLocalTransformation()).T
            world = pose_matrix(base[owner]) @ local
            quaternion = trimesh.transformations.quaternion_from_matrix(world)[[1, 2, 3, 0]]
            poses.append(np.r_[world[:3, 3], quaternion][None])
        result = np.concatenate(poses).astype(np.float32)
        assert result.shape == (len(self.body_paths), 7), "Capture pose count does not match body paths."
        assert np.isfinite(result).all(), "Nonfinite measured body pose."
        return result

    def record(self, step: int, phase: str, extra: dict | None = None, force: bool = False) -> None:
        """Record one measured state and optional diagnostic values at the sampling cadence."""
        if not force and step % self.stride:
            return
        time = step * self.dt
        if self.times and time == self.times[-1]:
            return
        assert not self.times or time > self.times[-1], "Capture times must increase."
        self.times.append(time)
        self.poses.append(self.read_poses())
        self.phases.append(phase)
        self.extras.append(copy.deepcopy(extra or {}))

    def save(self, report: dict) -> None:
        """Write the archive and metadata with per-frame diagnostics and final result."""
        assert self.times, "Record at least one state before saving."
        np.savez_compressed(
            self.directory / "scene.npz",
            **self.geometry,
            body_paths=np.asarray(self.body_paths),
            poses=np.asarray(self.poses),
            times=np.asarray(self.times),
            phases=np.asarray(self.phases),
            books=np.full(len(self.times), "crowded"),
            names=np.full(len(self.times), "crowded"),
            extras_json=np.asarray([json.dumps(extra) for extra in self.extras]),
        )
        metadata = {
            "body_paths": self.body_paths,
            "sampling_fps": 1 / (self.stride * self.dt),
            "frames": len(self.times),
            "duration_seconds": self.times[-1],
            "material_model": "USD display colors and face-bound material diffuse colors; textures omitted",
            "geometry_model": "USD owner-local geometry transformed by measured PhysX body poses",
            "frame_metadata": [
                {"time_seconds": time, "phase": phase, "extra": extra}
                for time, phase, extra in zip(self.times, self.phases, self.extras)
            ],
            "simulation_result": report,
        }
        (self.directory / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[capture] saved {len(self.times)} measured states to {self.directory}", flush=True)
