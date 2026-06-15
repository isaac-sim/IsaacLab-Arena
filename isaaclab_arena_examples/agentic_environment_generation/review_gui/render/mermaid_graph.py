# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphStateSpec

_MERMAID_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")


def render_mermaid_graph(spec: ArenaEnvInitialGraphSpec, state: ArenaEnvGraphStateSpec) -> str:
    """Emit a left-to-right mermaid graph of spatial and task constraints.

    Binary spatial constraints (reference is set) are drawn as solid edges:
        subject -->|kind| reference

    Unary spatial constraints (no reference) are omitted from the graph and
    listed to its right by :func:`render_unary_constraints` so their params are
    visible.

    Task constraints with a child are drawn as dashed edges:
        parent -.->|type| child

    object_reference nodes are drawn with a dotted edge to their parent node:
        ref_node -. ref .-> parent_node
    """
    lines = ["graph LR"]

    anchor_ids: set[str] = set()
    edge_nodes: set[str] = set()

    for constraint in state.spatial_constraints:
        kind = constraint.kind
        if kind == "is_anchor":
            anchor_ids.add(constraint.subject)
        if constraint.reference is not None:
            lines.append(
                f"  {_mermaid_id(constraint.subject)}[{_mermaid_label(constraint.subject)}]"
                f" -->|{kind}| "
                f"{_mermaid_id(constraint.reference)}[{_mermaid_label(constraint.reference)}]"
            )
            edge_nodes.add(constraint.subject)
            edge_nodes.add(constraint.reference)

    for task_constraint in state.task_constraints:
        if task_constraint.child is not None:
            lines.append(
                f"  {_mermaid_id(task_constraint.parent)}[{_mermaid_label(task_constraint.parent)}]"
                f" -.->|{_mermaid_label(task_constraint.type.value)}| "
                f"{_mermaid_id(task_constraint.child)}[{_mermaid_label(task_constraint.child)}]"
            )
            edge_nodes.add(task_constraint.parent)
            edge_nodes.add(task_constraint.child)

    for node in spec.nodes:
        if node.id not in edge_nodes:
            lines.append(f"  {_mermaid_id(node.id)}[{_mermaid_label(node.id)}]")

    nodes_by_id = spec.nodes_by_id
    for node in spec.nodes:
        if node.type.value == "object_reference" and node.parent is not None:
            if node.parent in nodes_by_id:
                lines.append(f"  {_mermaid_id(node.id)} -.->|ref| {_mermaid_id(node.parent)}")

    for anchor_id in anchor_ids:
        lines.append(f"  style {_mermaid_id(anchor_id)} fill:#3a7d44,color:#fff,stroke:#7fd17f,stroke-width:2px")

    type_palette = {
        "background": ("#3a4f7a", "#7aa0d8"),
        "embodiment": ("#7a3a3a", "#d87a7a"),
        "object": ("#7a6b3a", "#d8c47a"),
        "object_reference": ("#6b3a7a", "#c47ad8"),
        "lighting": ("#3a7a7a", "#7ad8d8"),
    }
    for node in spec.nodes:
        if node.id in anchor_ids:
            continue
        fill, stroke = type_palette.get(node.type.value, ("#3a3d44", "#888"))
        lines.append(f"  style {_mermaid_id(node.id)} fill:{fill},color:#fff,stroke:{stroke}")

    return "\n".join(lines)


def _mermaid_id(value: str) -> str:
    """Mermaid node identifiers must be alphanumeric / underscore."""
    return _MERMAID_ID_SAFE.sub("_", value)


def _mermaid_label(value: str) -> str:
    """Escape mermaid-significant characters inside node labels."""
    return value.replace('"', "&quot;").replace("|", "&#124;")
