# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture visible USD geometry and measured body poses for offline video rendering."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from isaaclab_arena_examples.bento.capture import _mesh, _vertex_colors, pose_matrix
from isaaclab_arena_examples.library.scene import BOOKS


class SceneCapture:
    """Save actual visual geometry and body poses without enabling an RTX camera."""

    def __init__(self, sim, robot, books, directory: Path, fps: float, mobile_base=None):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.robot, self.books, self.directory = robot, books, directory
        self.mobile_base = mobile_base
        self.stage = sim.stage
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, round(1 / (sim.get_physics_dt() * fps)))
        self.dt = sim.get_physics_dt()
        self.times, self.poses, self.phases, self.book_names = [], [], [], []
        prims = list(Usd.PrimRange(sim.stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
        rigid_paths = {
            prim.GetName(): str(prim.GetPath())
            for prim in prims
            if str(prim.GetPath()).startswith("/World/Robot/") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        assert all(
            name in rigid_paths for name in robot.body_names
        ), "Cannot resolve every robot body in the USD stage."
        self.body_paths = [
            "static",
            *(rigid_paths[name] for name in robot.body_names),
            *(f"/World/Book_{book.name}" for book in BOOKS),
        ]
        ee_path = rigid_paths["link_6_t"]
        self.ee_body_index = robot.body_names.index("link_6_t")
        self.jaw_paths = [
            f"{ee_path}/{name}"
            for name in ("LibraryJawLeft", "LibraryJawRight")
            if sim.stage.GetPrimAtPath(f"{ee_path}/{name}").IsValid()
        ]
        self.body_paths.extend(self.jaw_paths)
        if mobile_base is not None:
            self.body_paths.append(str(mobile_base.cfg.prim_path))
        initial_poses = self.read_poses()
        inverses = [np.linalg.inv(pose_matrix(pose)) for pose in initial_poses]
        vertices, normals, colors, body_ids, triangles = [], [], [], [], []
        offset = 0
        xforms = UsdGeom.XformCache()
        material_cache = {}
        for prim in prims:
            path = str(prim.GetPath())
            if not path.startswith("/World/") or not prim.IsA(UsdGeom.Gprim):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            mesh = _mesh(prim)
            if mesh is None or not len(mesh.faces):
                continue
            vertex_colors = _vertex_colors(prim, mesh, material_cache)
            owner = 0
            for index, body_path in sorted(enumerate(self.body_paths[1:], 1), key=lambda item: -len(item[1])):
                if path == body_path or path.startswith(body_path + "/"):
                    owner = index
                    break
            transform = inverses[owner] @ np.asarray(xforms.GetLocalToWorldTransform(prim)).T
            local_vertices = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
            local_normals = np.asarray(mesh.vertex_normals) @ np.linalg.inv(transform[:3, :3])
            local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-8)
            vertices.append(local_vertices.astype(np.float32))
            normals.append(local_normals.astype(np.float32))
            colors.append(vertex_colors)
            body_ids.append(np.full(len(mesh.vertices), owner, dtype=np.int32))
            triangles.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
            offset += len(mesh.vertices)
        assert vertices, "No visible USD geometry found for video capture."
        self.geometry = {
            "vertices": np.concatenate(vertices),
            "normals": np.concatenate(normals),
            "colors": np.concatenate(colors),
            "vertex_body_ids": np.concatenate(body_ids),
            "triangles": np.concatenate(triangles),
        }
        print(f"[capture] {offset} vertices, {len(self.geometry['triangles'])} triangles", flush=True)

    def read_poses(self) -> np.ndarray:
        """Read robot link and book world poses in the archive's body order."""
        import trimesh

        from pxr import UsdGeom

        static = np.asarray([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        robot = self.robot.data.body_link_pose_w.torch[0].detach().cpu().numpy()
        books = np.stack([self.books[book.name].data.root_pose_w.torch[0].detach().cpu().numpy() for book in BOOKS])
        poses = [static, robot, books]
        for path in self.jaw_paths:
            local = np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetLocalTransformation()).T
            world = pose_matrix(robot[self.ee_body_index]) @ local
            quaternion = trimesh.transformations.quaternion_from_matrix(world)[[1, 2, 3, 0]]
            poses.append(np.concatenate((world[:3, 3], quaternion))[None])
        if self.mobile_base is not None:
            poses.append(self.mobile_base.data.root_pose_w.torch[0].detach().cpu().numpy()[None])
        return np.concatenate(poses).astype(np.float32)

    def record(self, step: int, controller, force: bool = False) -> None:
        """Sample one measured simulation state at the requested cadence."""
        if not force and step % self.stride:
            return
        time = step * self.dt
        if self.times and time == self.times[-1]:
            return
        self.times.append(time)
        self.poses.append(self.read_poses())
        self.phases.append(controller.phase)
        self.book_names.append("all" if controller.finished else BOOKS[controller.book_index].name)

    def save(self, report: dict) -> None:
        """Write the reusable scene archive and capture metadata."""
        np.savez_compressed(
            self.directory / "scene.npz",
            **self.geometry,
            body_paths=np.asarray(self.body_paths),
            poses=np.asarray(self.poses),
            times=np.asarray(self.times),
            phases=np.asarray(self.phases),
            books=np.asarray(self.book_names),
        )
        metadata = {
            "body_paths": self.body_paths,
            "sampling_fps": 1 / (self.stride * self.dt),
            "frames": len(self.times),
            "duration_seconds": self.times[-1],
            "material_model": "USD display colors and face-bound material diffuse colors; textures omitted",
            "simulation_result": report,
        }
        (self.directory / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[capture] saved {len(self.times)} measured states to {self.directory}", flush=True)
