# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture visible USD geometry and measured body poses for offline video rendering."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from isaaclab_arena_examples.bento.meal import FOODS


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert world XYZ and an XYZW quaternion into a homogeneous matrix."""
    import trimesh

    matrix = trimesh.transformations.quaternion_matrix(pose[[6, 3, 4, 5]])
    matrix[:3, 3] = pose[:3]
    return matrix


def _mesh(prim):
    """Triangulate one visible USD mesh or analytic primitive."""
    import trimesh

    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        usd_mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
        counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if not len(vertices) or not len(indices):
            return None
        faces, corners, face_ids, start = [], [], [], 0
        holes = set(usd_mesh.GetHoleIndicesAttr().Get() or ())
        for face_id, count in enumerate(counts):
            if face_id not in holes:
                for j in range(1, count - 1):
                    corner = (start, start + j, start + j + 1)
                    faces.append(indices[list(corner)])
                    corners.append(corner)
                    face_ids.append(face_id)
            start += count
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        corners = np.asarray(corners, dtype=np.int32).reshape(-1, 3)
        if usd_mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            faces = faces[:, ::-1]
            corners = corners[:, ::-1]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.metadata.update(usd_corners=corners, usd_face_ids=np.asarray(face_ids, dtype=np.int32))
        return mesh
    if prim.IsA(UsdGeom.Cube):
        mesh = trimesh.creation.box(extents=[UsdGeom.Cube(prim).GetSizeAttr().Get()] * 3)
        mesh.unmerge_vertices()
        return mesh
    if prim.IsA(UsdGeom.Sphere):
        return trimesh.creation.icosphere(subdivisions=3, radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get())
    if prim.IsA(UsdGeom.Cylinder):
        shape = UsdGeom.Cylinder(prim)
        mesh = trimesh.creation.cylinder(
            radius=shape.GetRadiusAttr().Get(), height=shape.GetHeightAttr().Get(), sections=32
        )
        axis = shape.GetAxisAttr().Get()
        if axis != "Z":
            direction = (1, 0, 0) if axis == "X" else (0, 1, 0)
            mesh.apply_transform(trimesh.geometry.align_vectors((0, 0, 1), direction))
        return mesh
    assert not prim.IsA(
        UsdGeom.Gprim
    ), f"Unsupported visible geometry for capture: {prim.GetPath()} ({prim.GetTypeName()})"
    return None


def _color(prim, cache: dict) -> np.ndarray:
    """Read constant diffuse color from a USD material or display color."""
    from pxr import UsdGeom, UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if material:
        key = str(material.GetPath())
        if key in cache:
            return cache[key]
        for shader_prim in material.GetPrim().GetChildren():
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                continue
            for name in ("diffuseColor", "diffuse_color_constant", "base_color"):
                shader_input = shader.GetInput(name)
                value = shader_input.Get() if shader_input else None
                if value is not None and hasattr(value, "__len__") and len(value) == 3:
                    cache[key] = np.asarray(value, dtype=np.float32)
                    return cache[key]
    display = UsdGeom.Gprim(prim).GetDisplayColorPrimvar().Get() if prim.IsA(UsdGeom.Gprim) else None
    return np.asarray(display[0] if display else (0.48, 0.53, 0.60), dtype=np.float32)


def _display_colors(prim, mesh, cache: dict) -> np.ndarray:
    """Read authored USD display colors and preserve discontinuities between faces."""
    from pxr import UsdGeom

    primvar = UsdGeom.Gprim(prim).GetDisplayColorPrimvar()
    values = primvar.ComputeFlattened() if primvar else None
    if not values:
        return np.broadcast_to(_color(prim, cache), mesh.vertices.shape).copy()
    values = np.asarray(values, dtype=np.float32)
    interpolation = primvar.GetInterpolation()
    if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        assert values.shape == mesh.vertices.shape, f"Vertex color count does not match {prim.GetPath()}"
        return values
    if interpolation in (UsdGeom.Tokens.uniform, UsdGeom.Tokens.faceVarying) and "usd_corners" in mesh.metadata:
        source_indices = np.asarray(mesh.faces).reshape(-1)
        normals = np.asarray(mesh.vertex_normals)[source_indices]
        if interpolation == UsdGeom.Tokens.uniform:
            colors = np.repeat(values[mesh.metadata["usd_face_ids"]], 3, axis=0)
        else:
            colors = values[mesh.metadata["usd_corners"].reshape(-1)]
        mesh.vertices = np.asarray(mesh.vertices)[source_indices]
        mesh.faces = np.arange(len(source_indices)).reshape(-1, 3)
        mesh.vertex_normals = normals
        return colors
    assert len(values) == 1, f"Unsupported display color interpolation on {prim.GetPath()}: {interpolation}"
    return np.broadcast_to(_color(prim, cache), mesh.vertices.shape).copy()


def _vertex_colors(prim, mesh, cache: dict) -> np.ndarray:
    """Preserve display colors and face-bound materials without blending across boundaries."""
    from pxr import UsdShade

    colors = _display_colors(prim, mesh, cache)
    if "usd_face_ids" not in mesh.metadata:
        return colors
    subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
    if not subsets:
        return colors
    faces = np.asarray(mesh.faces)
    triangle_colors = colors[faces].copy()
    for subset in subsets:
        if not UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]:
            continue
        indices = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int32)
        selected = np.isin(mesh.metadata["usd_face_ids"], indices)
        triangle_colors[selected] = _color(subset.GetPrim(), cache)
    # Shared vertices cannot represent a sharp material change, such as the blue logo
    # beside the white housing. Split triangle corners while retaining smooth normals.
    source_indices = faces.reshape(-1)
    normals = np.asarray(mesh.vertex_normals)[source_indices].copy()
    mesh.vertices = np.asarray(mesh.vertices)[source_indices].copy()
    mesh.faces = np.arange(len(source_indices)).reshape(-1, 3)
    mesh.vertex_normals = normals
    return triangle_colors.reshape(-1, 3)


class SceneCapture:
    """Save actual visual geometry and body poses without enabling an RTX camera."""

    def __init__(self, sim, robot, foods, directory: Path, fps: float):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.robot, self.foods, self.directory = robot, foods, directory
        self.stage = sim.stage
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, round(1 / (sim.get_physics_dt() * fps)))
        self.dt = sim.get_physics_dt()
        self.times, self.poses, self.phases, self.food_names = [], [], [], []
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
            *(f"/World/Food_{food.name}" for food in FOODS),
        ]
        ee_path = rigid_paths["link_6_t"]
        self.ee_body_index = robot.body_names.index("link_6_t")
        self.jaw_paths = [
            f"{ee_path}/{name}"
            for name in ("ToolJawLeft", "ToolJawRight")
            if sim.stage.GetPrimAtPath(f"{ee_path}/{name}").IsValid()
        ]
        self.body_paths.extend(self.jaw_paths)
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
        """Read robot link and food world poses in the archive's body order."""
        import trimesh

        from pxr import UsdGeom

        static = np.asarray([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        robot = self.robot.data.body_link_pose_w.torch[0].detach().cpu().numpy()
        foods = np.stack([self.foods[food.name].data.root_pose_w.torch[0].detach().cpu().numpy() for food in FOODS])
        poses = [static, robot, foods]
        for path in self.jaw_paths:
            local = np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetLocalTransformation()).T
            world = pose_matrix(robot[self.ee_body_index]) @ local
            quaternion = trimesh.transformations.quaternion_from_matrix(world)[[1, 2, 3, 0]]
            poses.append(np.concatenate((world[:3, 3], quaternion))[None])
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
        self.food_names.append("all" if controller.finished else FOODS[controller.food_index].name)

    def save(self, report: dict) -> None:
        """Write the reusable scene archive and capture metadata."""
        np.savez_compressed(
            self.directory / "scene.npz",
            **self.geometry,
            poses=np.asarray(self.poses),
            times=np.asarray(self.times),
            phases=np.asarray(self.phases),
            foods=np.asarray(self.food_names),
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
