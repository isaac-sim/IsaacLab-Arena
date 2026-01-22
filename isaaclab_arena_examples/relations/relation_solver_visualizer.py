# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""3D visualization for RelationSolver results using Plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

# TYPE_CHECKING: Import Object for type hints without runtime Isaac Sim dependency.
# At runtime, duck typing allows Object to work as well.
if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object

# Color palette for objects
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


class RelationSolverVisualizer:
    """Plotly-based 3D visualization for RelationSolver results.

    Provides separate methods for visualizing:
    - Final object positions and bounding boxes
    - Optimization trajectories
    - Loss history
    """

    def __init__(
        self,
        result: dict[Object, tuple[float, float, float]],
        objects: list[Object],
        anchor_object: Object,
        loss_history: list[float] | None = None,
        position_history: list | None = None,
    ):
        """Initialize the visualizer.

        Args:
            result: Result dictionary mapping objects to final positions.
            objects: List of Object instances (same order as passed to solve())
            anchor_object: The anchor object that was fixed during optimization
            loss_history: Optional list of loss values during optimization.
            position_history: Optional list of position snapshots during optimization.
        """
        self.result = result
        self.objects = objects
        self.anchor_object = anchor_object
        self.position_history = position_history or []
        self.loss_history = loss_history or []

    def _get_color(self, idx: int) -> str:
        """Get color for object at index."""
        return COLORS[idx % len(COLORS)]

    def _create_wireframe_box(
        self,
        bbox: AxisAlignedBoundingBox,
        position: tuple[float, float, float],
        color: str,
        name: str,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> go.Scatter3d:
        """Create wireframe box trace for a 3D bounding box.

        Args:
            bbox: The axis-aligned bounding box (local coordinates relative to origin)
            position: (x, y, z) object origin position in world coordinates
            color: Line color
            name: Object name for legend
            opacity: Line opacity
            dash: Line dash style ("dash", "dot", etc.)

        Returns:
            Scatter3d trace forming the wireframe
        """
        # Compute world-space corners: world = position + local offset
        # This matches how the loss strategies compute world extents
        x, y, z = position
        x_min = x + bbox.min_point[0]
        x_max = x + bbox.max_point[0]
        y_min = y + bbox.min_point[1]
        y_max = y + bbox.max_point[1]
        z_min = z + bbox.min_point[2]
        z_max = z + bbox.max_point[2]

        # 8 corners of the box (same ordering as get_corners_at)
        corners = [
            [x_min, y_min, z_min],  # 0: Bottom-front-left
            [x_max, y_min, z_min],  # 1: Bottom-front-right
            [x_max, y_max, z_min],  # 2: Bottom-back-right
            [x_min, y_max, z_min],  # 3: Bottom-back-left
            [x_min, y_min, z_max],  # 4: Top-front-left
            [x_max, y_min, z_max],  # 5: Top-front-right
            [x_max, y_max, z_max],  # 6: Top-back-right
            [x_min, y_max, z_max],  # 7: Top-back-left
        ]

        # Define edges as pairs of corner indices (matching get_corners ordering)
        edges = [
            # Bottom face
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Top face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Vertical edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        # Create a single trace with all edges connected by None breaks
        x_coords = []
        y_coords = []
        z_coords = []

        for start, end in edges:
            x_coords.extend([corners[start][0], corners[end][0], None])
            y_coords.extend([corners[start][1], corners[end][1], None])
            z_coords.extend([corners[start][2], corners[end][2], None])

        line_dict = {"color": color, "width": 3}
        if dash:
            line_dict["dash"] = dash

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="lines",
            line=line_dict,
            name=name,
            opacity=opacity,
            showlegend=True,
        )

    def _add_trajectory_traces(
        self,
        fig: go.Figure,
        obj: Object,
        obj_idx: int,
        color: str,
    ) -> None:
        """Add trajectory line and start marker for an object.

        Args:
            fig: The Plotly figure to add traces to
            obj: The object to plot trajectory for
            obj_idx: Index of the object in the position history
            color: Color for the trajectory
        """
        xs = [self.position_history[i][obj_idx][0] for i in range(len(self.position_history))]
        ys = [self.position_history[i][obj_idx][1] for i in range(len(self.position_history))]
        zs = [self.position_history[i][obj_idx][2] for i in range(len(self.position_history))]

        # Trajectory line
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{obj.name} trajectory",
                opacity=0.5,
                hovertemplate=f"{obj.name}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
            )
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]],
                y=[ys[0]],
                z=[zs[0]],
                mode="markers",
                marker=dict(size=8, color=color, symbol="circle", opacity=0.7),
                name=f"{obj.name} start",
                hovertemplate=f"{obj.name} start<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
            )
        )

    def _add_object_traces(
        self,
        fig: go.Figure,
        obj: Object,
        position: tuple[float, float, float],
        color: str,
        is_anchor: bool,
    ) -> None:
        """Add bounding box for an object.

        Args:
            fig: The Plotly figure to add traces to
            obj: The object to visualize
            position: (x, y, z) position of the object
            color: Color for the object
            is_anchor: Whether this is the anchor object
        """
        bbox = obj.get_bounding_box()
        label = f"{obj.name} (anchor)" if is_anchor else obj.name
        dash = "dot" if is_anchor else None

        # Wireframe bounding box
        box_trace = self._create_wireframe_box(
            bbox=bbox,
            position=position,
            color=color,
            name=label,
            dash=dash,
        )
        fig.add_trace(box_trace)

    def plot_objects_3d(self) -> go.Figure:
        """Plot final object positions, bounding boxes, and optimization trajectories in 3D.

        Returns:
            Plotly Figure with 3D visualization of objects and trajectories
        """
        fig = go.Figure()

        if not self.position_history:
            fig.add_annotation(
                text="No position data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return fig

        final_positions = self.position_history[-1]

        for idx, obj in enumerate(self.objects):
            is_anchor = obj is self.anchor_object
            pos = (final_positions[idx][0], final_positions[idx][1], final_positions[idx][2])
            color = self._get_color(idx)

            if not is_anchor:
                self._add_trajectory_traces(fig, obj, idx, color)

            self._add_object_traces(fig, obj, pos, color, is_anchor)

        fig.update_layout(
            title=dict(text="Object Positions & Trajectories", font=dict(size=18)),
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
                domain=dict(x=[0, 0.85], y=[0, 1]),
            ),
            legend=dict(x=0.87, y=0.98),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig

    def plot_loss_history(self) -> go.Figure:
        """Plot loss over optimization iterations.

        Returns:
            Plotly Figure with loss history curve
        """
        fig = go.Figure()

        if not self.loss_history:
            fig.add_annotation(
                text="No loss history available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return fig

        iterations = list(range(len(self.loss_history)))

        # Main loss curve
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=self.loss_history,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                name="Loss",
                hovertemplate="Iter: %{x}<br>Loss: %{y:.6f}<extra></extra>",
            )
        )

        # Initial and final loss reference lines
        initial_loss = self.loss_history[0]
        final_loss = self.loss_history[-1]

        fig.add_hline(
            y=initial_loss,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text=f"Initial: {initial_loss:.4f}",
            annotation_position="right",
        )

        fig.add_hline(
            y=final_loss,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Final: {final_loss:.4f}",
            annotation_position="right",
        )

        # Configure layout
        fig.update_layout(
            title=dict(text="Optimization Loss History", font=dict(size=18)),
            xaxis_title="Iteration",
            yaxis_title="Loss",
            hovermode="x unified",
            margin=dict(l=60, r=100, t=60, b=60),
        )

        return fig

    def _get_wireframe_coords(
        self,
        bbox: AxisAlignedBoundingBox,
        position: tuple[float, float, float],
    ) -> tuple[list, list, list]:
        """Get wireframe coordinates for a bounding box at a position.

        Args:
            bbox: The axis-aligned bounding box (local coordinates relative to origin)
            position: (x, y, z) object origin position in world coordinates

        Returns:
            Tuple of (x_coords, y_coords, z_coords) lists for the wireframe
        """
        # Compute world-space corners: world = position + local offset
        x, y, z = position
        x_min = x + bbox.min_point[0]
        x_max = x + bbox.max_point[0]
        y_min = y + bbox.min_point[1]
        y_max = y + bbox.max_point[1]
        z_min = z + bbox.min_point[2]
        z_max = z + bbox.max_point[2]

        corners = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
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

        x_coords = []
        y_coords = []
        z_coords = []

        for start, end in edges:
            x_coords.extend([corners[start][0], corners[end][0], None])
            y_coords.extend([corners[start][1], corners[end][1], None])
            z_coords.extend([corners[start][2], corners[end][2], None])

        return x_coords, y_coords, z_coords

    def animate_optimization(self) -> go.Figure:
        """Create animated 3D visualization of the optimization process.

        Shows objects moving from their initial positions to final positions
        through each optimization step. Auto-plays through all frames.

        Returns:
            Plotly Figure with animation frames that auto-plays.
        """
        if not self.position_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No position history available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Compute axis ranges from all positions to keep scale fixed
        all_x, all_y, all_z = [], [], []
        for positions in self.position_history:
            for idx, obj in enumerate(self.objects):
                pos = positions[idx]
                bbox = obj.get_bounding_box()
                half_size = [s / 2 for s in bbox.size]
                all_x.extend([pos[0] - half_size[0], pos[0] + half_size[0]])
                all_y.extend([pos[1] - half_size[1], pos[1] + half_size[1]])
                all_z.extend([pos[2] - half_size[2], pos[2] + half_size[2]])

        padding = 0.1
        x_range = [min(all_x) - padding, max(all_x) + padding]
        y_range = [min(all_y) - padding, max(all_y) + padding]
        z_range = [min(all_z) - padding, max(all_z) + padding]

        # Build initial frame (first position snapshot)
        initial_positions = self.position_history[0]
        fig = go.Figure()

        # Add traces for each object: wireframe box only
        for idx, obj in enumerate(self.objects):
            is_anchor = obj is self.anchor_object
            pos = (initial_positions[idx][0], initial_positions[idx][1], initial_positions[idx][2])
            color = self._get_color(idx)
            bbox = obj.get_bounding_box()

            # Wireframe bounding box
            x_coords, y_coords, z_coords = self._get_wireframe_coords(bbox, pos)
            label = f"{obj.name} (anchor)" if is_anchor else obj.name
            line_dict = {"color": color, "width": 3}
            if is_anchor:
                line_dict["dash"] = "dot"

            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines",
                    line=line_dict,
                    name=label,
                    showlegend=True,
                )
            )

        # Create animation frames with fixed axis ranges
        frames = []
        frame_layout = dict(
            scene=dict(
                xaxis=dict(range=x_range, autorange=False),
                yaxis=dict(range=y_range, autorange=False),
                zaxis=dict(range=z_range, autorange=False),
            )
        )

        for frame_idx, positions in enumerate(self.position_history):
            frame_data = []

            for idx, obj in enumerate(self.objects):
                pos = (positions[idx][0], positions[idx][1], positions[idx][2])
                bbox = obj.get_bounding_box()

                # Wireframe box data
                x_coords, y_coords, z_coords = self._get_wireframe_coords(bbox, pos)
                frame_data.append(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords))

            frames.append(go.Frame(data=frame_data, layout=frame_layout, name=str(frame_idx)))

        fig.frames = frames

        # Configure layout with fixed axis ranges and proper camera orientation
        # Camera: looking along +Y axis, X to the right, Z up
        fig.update_layout(
            title=dict(text="Optimization Animation", font=dict(size=18)),
            scene=dict(
                xaxis=dict(
                    range=x_range,
                    autorange=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title="",
                ),
                yaxis=dict(
                    range=y_range,
                    autorange=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title="",
                ),
                zaxis=dict(
                    range=z_range,
                    autorange=False,
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    title="",
                ),
                bgcolor="rgba(0,0,0,0)",
                aspectmode="manual",
                aspectratio=dict(
                    x=(x_range[1] - x_range[0]),
                    y=(y_range[1] - y_range[0]),
                    z=(z_range[1] - z_range[0]),
                ),
                camera=dict(
                    eye=dict(x=0, y=-2.0, z=0.8),  # Looking along +Y, X to the right
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                ),
            ),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
            margin=dict(l=0, r=0, t=60, b=50),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0.0,
                    x=0.5,
                    xanchor="center",
                    yanchor="bottom",
                    direction="left",
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
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )

        return fig
