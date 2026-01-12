# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""3D visualization for RelationSolver results using Plotly."""

import torch

import plotly.graph_objects as go

from isaaclab_arena.assets.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

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
        result: dict,
        objects: list[DummyObject],
        anchor_object: DummyObject,
    ):
        """Initialize the visualizer.

        Args:
            result: Result dictionary from RelationSolver.solve()
            objects: List of DummyObject instances (same order as passed to solve())
            anchor_object: The anchor object that was fixed during optimization
        """
        self.result = result
        self.objects = objects
        self.anchor_object = anchor_object
        self.position_history = result.get("_position_history", [])
        self.loss_history = result.get("_loss_history", [])

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
            bbox: The axis-aligned bounding box
            position: (x, y, z) center position
            color: Line color
            name: Object name for legend
            opacity: Line opacity
            dash: Line dash style ("dash", "dot", etc.)

        Returns:
            Scatter3d trace forming the wireframe
        """
        # Get corners from bounding box API
        pos_tensor = torch.tensor(position, dtype=torch.float32)
        corners = bbox.get_corners(pos_tensor).tolist()

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
        obj: DummyObject,
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
        obj: DummyObject,
        position: tuple[float, float, float],
        color: str,
        is_anchor: bool,
    ) -> None:
        """Add bounding box and center marker for an object.

        Args:
            fig: The Plotly figure to add traces to
            obj: The object to visualize
            position: (x, y, z) position of the object
            color: Color for the object
            is_anchor: Whether this is the anchor object
        """
        bbox = obj.get_bounding_box()
        label = f"{obj.name} (anchor)" if is_anchor else obj.name
        dash = "dash" if is_anchor else None

        # Wireframe bounding box
        box_trace = self._create_wireframe_box(
            bbox=bbox,
            position=position,
            color=color,
            name=label,
            dash=dash,
        )
        fig.add_trace(box_trace)

        # Center marker
        fig.add_trace(
            go.Scatter3d(
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,
                    symbol="square" if is_anchor else "diamond",
                ),
                name=f"{obj.name} end" if not is_anchor else f"{obj.name} center",
                showlegend=not is_anchor,
                hovertemplate=f"{obj.name}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>",
            )
        )

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
            ),
            legend=dict(x=1.02, y=0.98),
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
