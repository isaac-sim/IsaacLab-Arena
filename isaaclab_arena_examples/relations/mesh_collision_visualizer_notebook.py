# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false

"""Mesh SDF collision visualizer — standalone diagnostic notebook.

Runs the RelationSolver on two cylinder objects placed on a table,
then visualises the result with actual 3D meshes (not just AABBs),
SDF-coloured query points, a loss curve, and an optimization animation.

No Isaac Sim dependency — uses ``DummyObject`` + ``trimesh`` cylinders.

Run cells interactively, or execute the whole file::

    python isaaclab_arena_examples/relations/mesh_collision_visualizer_notebook.py
"""

# %%
from __future__ import annotations

import numpy as np
import torch
import trimesh

import plotly.graph_objects as go

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On, get_anchor_objects
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshManager
from isaaclab_arena.relations.warp_sdf_kernels import mesh_sdf
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_examples.relations.relation_solver_visualizer import COLORS, RelationSolverVisualizer

# ---------------------------------------------------------------------------
# Object factories
# ---------------------------------------------------------------------------


def _make_cylinder_object(name: str, radius: float = 0.033, height: float = 0.101) -> DummyObject:
    """Create a DummyObject backed by a trimesh cylinder (YCB tomato-soup-can dims)."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-radius, -radius, -height / 2),
            max_point=(radius, radius, height / 2),
        ),
        collision_mesh=mesh,
    )


def _make_box_object(name: str, sx: float, sy: float, sz: float) -> DummyObject:
    """Create a DummyObject backed by a trimesh box."""
    mesh = trimesh.creation.box(extents=(sx, sy, sz))
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=(-sx / 2, -sy / 2, -sz / 2),
            max_point=(sx / 2, sy / 2, sz / 2),
        ),
        collision_mesh=mesh,
    )


def _extract_trimesh_from_usd(usd_path: str, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> trimesh.Trimesh:
    """Load all mesh geometry from a USD file into a single trimesh.

    Handles world transforms per-prim, fan triangulation for non-triangle
    faces, and applies *scale* to the combined result.

    Requires ``pxr`` (OpenUSD) — available inside the Isaac Sim Python env.
    """
    from pxr import Usd, UsdGeom  # type: ignore[import-untyped]

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD: {usd_path}")

    all_verts: list[np.ndarray] = []
    all_faces: list[list[int]] = []
    offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_prim = UsdGeom.Mesh(prim)
        pts = mesh_prim.GetPointsAttr().Get()
        fvc = mesh_prim.GetFaceVertexCountsAttr().Get()
        fvi = mesh_prim.GetFaceVertexIndicesAttr().Get()
        if pts is None or fvc is None or fvi is None:
            continue

        xform = UsdGeom.Xformable(prim)
        world_tf = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

        verts = np.asarray(pts, dtype=np.float64)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts_world = (verts_h @ world_tf)[:, :3]
        verts_world[:, 0] *= scale[0]
        verts_world[:, 1] *= scale[1]
        verts_world[:, 2] *= scale[2]

        idx = 0
        for count in fvc:
            for k in range(1, count - 1):
                all_faces.append([fvi[idx] + offset, fvi[idx + k] + offset, fvi[idx + k + 1] + offset])
            idx += count

        all_verts.append(verts_world)
        offset += len(verts_world)

    if not all_verts:
        raise ValueError(f"No meshes in {usd_path}")
    return trimesh.Trimesh(vertices=np.vstack(all_verts), faces=np.array(all_faces, dtype=np.int32))


# Cached USD paths for YCB assets (populated after first simulation run).
_USD_CACHE_ROOT = "/tmp/Assets/Isaac/6.0/Isaac"
_USD_ASSETS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "mustard_bottle": (f"{_USD_CACHE_ROOT}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd", (1, 1, 1)),
    "cracker_box": (f"{_USD_CACHE_ROOT}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", (1, 1, 1)),
    "power_drill": (
        f"{_USD_CACHE_ROOT}/IsaacLab/Arena/assets/object_library/power_drill_physics/power_drill_physics.usd",
        (1, 1, 1),
    ),
    "tomato_soup_can": (f"{_USD_CACHE_ROOT}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd", (1, 1, 1)),
    "sugar_box": (f"{_USD_CACHE_ROOT}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd", (1, 1, 1)),
}


def make_usd_object(name: str) -> DummyObject:
    """Create a DummyObject from a cached USD simulation asset.

    The asset must have been downloaded during a prior simulation run
    (cached under ``/tmp/Assets/``).  Supports: mustard_bottle,
    cracker_box, power_drill, tomato_soup_can, sugar_box.
    """
    if name not in _USD_ASSETS:
        raise ValueError(f"Unknown asset '{name}'. Available: {list(_USD_ASSETS.keys())}")
    usd_path, scale = _USD_ASSETS[name]
    mesh = _extract_trimesh_from_usd(usd_path, scale=scale)
    # Centre the mesh so the AABB is symmetric around the origin
    centroid = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.vertices -= centroid
    bb = mesh.bounds
    return DummyObject(
        name=name,
        bounding_box=AxisAlignedBoundingBox(min_point=tuple(bb[0]), max_point=tuple(bb[1])),
        collision_mesh=mesh,
    )


# ---------------------------------------------------------------------------
# SDF helpers
# ---------------------------------------------------------------------------


def _compute_sdf_for_pair(
    child_obj: DummyObject,
    parent_obj: DummyObject,
    child_pos: tuple[float, float, float],
    parent_pos: tuple[float, float, float],
    mgr: WarpMeshManager,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Query child vertices against parent mesh and vice-versa.

    Returns (child_query_world, child_sdf, parent_query_world, parent_sdf).
    """
    c_pos = torch.tensor(child_pos, dtype=torch.float32, device=device)
    p_pos = torch.tensor(parent_pos, dtype=torch.float32, device=device)

    child_verts = mgr.get_query_vertices(child_obj, device)
    parent_verts = mgr.get_query_vertices(parent_obj, device)

    parent_wp = mgr.get_warp_mesh(parent_obj)
    child_wp = mgr.get_warp_mesh(child_obj)

    query_a = child_verts + c_pos - p_pos
    with torch.no_grad():
        sdf_a = mesh_sdf(query_a, parent_wp, mgr.device)

    query_b = parent_verts + p_pos - c_pos
    with torch.no_grad():
        sdf_b = mesh_sdf(query_b, child_wp, mgr.device)

    child_query_world = (child_verts + c_pos).detach().cpu().numpy()
    parent_query_world = (parent_verts + p_pos).detach().cpu().numpy()
    return child_query_world, sdf_a.detach().cpu().numpy(), parent_query_world, sdf_b.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Wireframe helper (standalone, no visualizer instance needed)
# ---------------------------------------------------------------------------


def _wireframe_trace(
    bbox: AxisAlignedBoundingBox,
    pos: tuple[float, float, float],
    color: str,
    name: str,
    dash: str | None = None,
    opacity: float = 1.0,
) -> go.Scatter3d:
    """Build a 3D wireframe box trace from an AABB + position."""
    mn = bbox.min_point[0].tolist()
    mx = bbox.max_point[0].tolist()
    x, y, z = pos
    corners = [
        [x + mn[0], y + mn[1], z + mn[2]],
        [x + mx[0], y + mn[1], z + mn[2]],
        [x + mx[0], y + mx[1], z + mn[2]],
        [x + mn[0], y + mx[1], z + mn[2]],
        [x + mn[0], y + mn[1], z + mx[2]],
        [x + mx[0], y + mn[1], z + mx[2]],
        [x + mx[0], y + mx[1], z + mx[2]],
        [x + mn[0], y + mx[1], z + mx[2]],
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    xc, yc, zc = [], [], []
    for s, e in edges:
        xc.extend([corners[s][0], corners[e][0], None])
        yc.extend([corners[s][1], corners[e][1], None])
        zc.extend([corners[s][2], corners[e][2], None])
    line: dict = dict(color=color, width=2)
    if dash:
        line["dash"] = dash
    return go.Scatter3d(x=xc, y=yc, z=zc, mode="lines", line=line, name=name, opacity=opacity, showlegend=True)


# %%
# ===========================================================================
# Demo 1 — Two overlapping cylinders on a table, separated by the solver
# ===========================================================================


def run_solver_with_mesh_viz():
    """Two cylinders start overlapping on a table; the solver pushes them apart.

    Produces three figures:
    1. Final 3D scene with meshes + AABB wireframes + SDF query points
    2. Loss curve
    3. Optimization animation with meshes
    """
    # --- Objects ---
    table = _make_box_object("table", 0.4, 0.4, 0.02)
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    can_a = _make_cylinder_object("can_a")
    can_b = _make_cylinder_object("can_b")

    can_a.add_relation(On(table, clearance_m=0.005))
    can_b.add_relation(On(table, clearance_m=0.005))

    all_objects = [table, can_a, can_b]

    # --- Initial positions: cans nearly overlapping on the table ---
    table_top_z = 0.01 + 0.005  # table half-height + clearance
    initial_positions = {
        table: (0.0, 0.0, 0.0),
        can_a: (-0.01, 0.0, table_top_z),
        can_b: (0.01, 0.0, table_top_z),
    }

    # --- Solve ---
    solver = RelationSolver(
        params=RelationSolverParams(verbose=True, save_position_history=True, max_iters=600, lr=0.01),
    )
    final_positions_list = solver.solve(objects=all_objects, initial_positions=[initial_positions])
    final_positions = final_positions_list[0]

    anchor_objects = get_anchor_objects(all_objects)
    positions_dict = {obj.name: final_positions[obj] for obj in all_objects}

    print("\nFinal positions:")
    for obj in all_objects:
        tag = " (anchor)" if obj in anchor_objects else ""
        p = final_positions[obj]
        print(f"  {obj.name}{tag}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

    # --- Figure 1: Final 3D scene with meshes + SDF ---
    fig1 = _build_scene_figure(all_objects, positions_dict, title="Final Scene: Meshes + AABB + SDF Query Points")

    # --- Figure 2: Loss curve ---
    loss_history = solver.last_loss_history
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=loss_history, mode="lines", name="Total loss"))
    fig2.update_layout(
        title="Loss Curve",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        yaxis_type="log",
        width=800,
        height=400,
    )

    # --- Figure 3: Optimization animation with meshes ---
    fig3 = _build_mesh_animation(all_objects, solver.last_position_history, anchor_objects)

    return fig1, fig2, fig3


def _build_scene_figure(
    objects: list[DummyObject],
    positions: dict[str, tuple[float, float, float]],
    title: str = "Scene",
) -> go.Figure:
    """Build a 3D scene figure with meshes, AABB wireframes, and SDF query points."""
    fig = go.Figure()
    mgr = WarpMeshManager(device="cpu")

    obj_by_name: dict[str, DummyObject] = {o.name: o for o in objects}
    non_anchor_names = [o.name for o in objects if not any(isinstance(r, IsAnchor) for r in o.get_relations())]

    bbox_color = "#888888"
    for idx, obj in enumerate(objects):
        pos = positions[obj.name]
        color = COLORS[idx % len(COLORS)]
        mesh = obj.get_collision_mesh()
        if mesh is not None:
            fig.add_trace(RelationSolverVisualizer.create_mesh_trace(mesh, pos, color=color, name=obj.name))
        fig.add_trace(
            _wireframe_trace(
                obj.get_bounding_box(), pos, color=bbox_color, name=f"{obj.name} bounding box", dash="dash"
            )
        )

    # SDF queries for all non-anchor pairs
    for i, a_name in enumerate(non_anchor_names):
        for b_name in non_anchor_names[i + 1 :]:
            a_obj, b_obj = obj_by_name[a_name], obj_by_name[b_name]
            a_pos, b_pos = positions[a_name], positions[b_name]
            cq, cs, pq, ps = _compute_sdf_for_pair(a_obj, b_obj, a_pos, b_pos, mgr)
            fig.add_trace(RelationSolverVisualizer.create_sdf_scatter(cq, cs, name=f"SDF: {a_name} vs {b_name}"))
            fig.add_trace(RelationSolverVisualizer.create_sdf_scatter(pq, ps, name=f"SDF: {b_name} vs {a_name}"))

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
        ),
        width=900,
        height=700,
    )
    return fig


def _build_mesh_animation(
    objects: list[DummyObject],
    position_history: list,
    anchor_objects: list,
) -> go.Figure:
    """Build a Plotly animation of mesh objects moving through the solver iterations."""
    if not position_history:
        fig = go.Figure()
        fig.add_annotation(text="No position history", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Compute global axis ranges across all frames
    all_x, all_y, all_z = [], [], []
    for positions in position_history:
        for idx, obj in enumerate(objects):
            pos = positions[idx]
            bbox = obj.get_bounding_box()
            half = (bbox.size[0] / 2).tolist()
            all_x.extend([pos[0] - half[0], pos[0] + half[0]])
            all_y.extend([pos[1] - half[1], pos[1] + half[1]])
            all_z.extend([pos[2] - half[2], pos[2] + half[2]])

    pad = 0.05
    x_range = [min(all_x) - pad, max(all_x) + pad]
    y_range = [min(all_y) - pad, max(all_y) + pad]
    z_range = [min(all_z) - pad, max(all_z) + pad]

    # Initial frame
    initial_pos = position_history[0]
    fig = go.Figure()

    bbox_color = "#888888"
    traces_per_obj = []
    for idx, obj in enumerate(objects):
        pos = (initial_pos[idx][0], initial_pos[idx][1], initial_pos[idx][2])
        color = COLORS[idx % len(COLORS)]
        mesh = obj.get_collision_mesh()

        obj_traces = []
        if mesh is not None:
            is_anchor = obj in anchor_objects
            opacity = 0.2 if is_anchor else 0.4
            fig.add_trace(
                RelationSolverVisualizer.create_mesh_trace(mesh, pos, color=color, name=obj.name, opacity=opacity)
            )
            obj_traces.append("mesh")

        fig.add_trace(
            _wireframe_trace(
                obj.get_bounding_box(),
                pos,
                color=bbox_color,
                name=f"{obj.name} bounding box",
                dash="dot" if obj in anchor_objects else "dash",
            )
        )
        obj_traces.append("wire")
        traces_per_obj.append(obj_traces)

    # Build animation frames
    frames = []
    frame_layout = dict(
        scene=dict(
            xaxis=dict(range=x_range, autorange=False),
            yaxis=dict(range=y_range, autorange=False),
            zaxis=dict(range=z_range, autorange=False),
        )
    )

    for frame_idx, positions in enumerate(position_history):
        frame_data = []
        for idx, obj in enumerate(objects):
            pos = (positions[idx][0], positions[idx][1], positions[idx][2])
            mesh = obj.get_collision_mesh()

            if "mesh" in traces_per_obj[idx] and mesh is not None:
                verts = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.faces, dtype=np.int32)
                frame_data.append(
                    go.Mesh3d(
                        x=verts[:, 0] + pos[0],
                        y=verts[:, 1] + pos[1],
                        z=verts[:, 2] + pos[2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                    )
                )

            bbox = obj.get_bounding_box()
            wt = _wireframe_trace(bbox, pos, color=bbox_color, name=f"{obj.name} bounding box")
            frame_data.append(go.Scatter3d(x=wt.x, y=wt.y, z=wt.z))

        frames.append(go.Frame(data=frame_data, layout=frame_layout, name=str(frame_idx)))

    fig.frames = frames

    fig.update_layout(
        title="Optimization Animation (meshes)",
        scene=dict(
            xaxis=dict(range=x_range, autorange=False, title="X (m)"),
            yaxis=dict(range=y_range, autorange=False, title="Y (m)"),
            zaxis=dict(range=z_range, autorange=False, title="Z (m)"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=0, y=-2.0, z=0.8), up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        width=900,
        height=700,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.0,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=50),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
    )
    return fig


# %%
# ===========================================================================
# Demo 2 — Dense clutter: AABB overlap but mesh clearance
# ===========================================================================


def run_dense_clutter_sdf_viz():
    """Two cylinders at the dense-clutter diagonal offset.

    Shows the 3D scene with SDF query points demonstrating that AABBs overlap
    while meshes have clearance.
    """
    can_a = _make_cylinder_object("can_a")
    can_c = _make_cylinder_object("can_c")

    offset = 0.055
    positions = {"can_a": (0.0, 0.0, 0.0), "can_c": (offset, offset, 0.0)}

    radius = 0.033
    diameter = radius * 2
    print("=== Dense Clutter SDF Demo ===")
    print(f"Can radius: {radius} m, AABB half-width: {radius} m")
    print(f"Diagonal offset: {offset} m each axis")
    print(f"Centre-to-centre: {offset * np.sqrt(2):.4f} m")
    print(f"Can diameter: {diameter} m")
    print(f"Mesh gap: {offset * np.sqrt(2) - diameter:.4f} m")

    fig = _build_scene_figure(
        [can_a, can_c],
        positions,
        title="Dense Clutter: AABBs overlap, meshes have clearance",
    )
    return fig


# %%
# ===========================================================================
# Demo 3 — Real USD simulation assets (requires cached assets in /tmp)
# ===========================================================================


def run_real_objects_demo():
    """Three real YCB objects start overlapping on a table; the solver separates them.

    Uses actual simulation meshes extracted from cached USD files (mustard bottle,
    power drill, cracker box).  Requires that these assets have been downloaded
    during a prior Isaac Sim run.
    """
    import os

    first_asset = list(_USD_ASSETS.values())[0][0]
    if not os.path.exists(first_asset):
        print(f"Skipping real-objects demo: cached USD not found at {first_asset}")
        print("Run a simulation first to populate /tmp/Assets/.")
        return None, None, None

    table = _make_box_object("table", 0.6, 0.6, 0.02)
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    drill = make_usd_object("power_drill")
    mustard = make_usd_object("mustard_bottle")
    cracker = make_usd_object("cracker_box")

    for obj in [drill, mustard, cracker]:
        obj.add_relation(On(table, clearance_m=0.005))

    all_objects = [table, drill, mustard, cracker]

    table_top_z = 0.01 + 0.005
    initial_positions = {
        table: (0.0, 0.0, 0.0),
        drill: (0.02, 0.0, table_top_z),
        mustard: (-0.02, 0.0, table_top_z),
        cracker: (0.0, 0.02, table_top_z),
    }

    solver = RelationSolver(
        params=RelationSolverParams(verbose=True, save_position_history=True, max_iters=600, lr=0.01),
    )
    final_positions_list = solver.solve(objects=all_objects, initial_positions=[initial_positions])
    final_positions = final_positions_list[0]

    anchor_objects = get_anchor_objects(all_objects)
    positions_dict = {obj.name: final_positions[obj] for obj in all_objects}

    print("\nFinal positions:")
    for obj in all_objects:
        tag = " (anchor)" if obj in anchor_objects else ""
        p = final_positions[obj]
        print(f"  {obj.name}{tag}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

    fig1 = _build_scene_figure(all_objects, positions_dict, title="Real YCB Objects: Solved Positions")

    loss_history = solver.last_loss_history
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=loss_history, mode="lines", name="Total loss"))
    fig2.update_layout(
        title="Loss Curve (real objects)",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        yaxis_type="log",
        width=800,
        height=400,
    )

    fig3 = _build_mesh_animation(all_objects, solver.last_position_history, anchor_objects)

    return fig1, fig2, fig3


# %%
if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Solver separates two overlapping cylinders")
    print("=" * 60)
    fig1, fig2, fig3 = run_solver_with_mesh_viz()
    fig1.show()
    fig2.show()
    fig3.show()

    print()
    print("=" * 60)
    print("Demo 3: Real YCB objects (drill, mustard, cracker box)")
    print("=" * 60)
    fig5, fig6, fig7 = run_real_objects_demo()
    if fig5 is not None:
        fig5.show()
        fig6.show()
        fig7.show()
