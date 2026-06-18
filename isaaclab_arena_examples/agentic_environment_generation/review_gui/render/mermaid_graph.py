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

    # (fill, stroke) per node type — matches badge colors in styles.py / panels.py
    type_palette = {
        "background": ("#3a4f7a", "#7aa0d8"),  # blue
        "embodiment": ("#7a3a3a", "#d87a7a"),  # red
        "object": ("#7a6b3a", "#d8c47a"),  # gold
        "object_reference": ("#6b3a7a", "#c47ad8"),  # purple
        "lighting": ("#3a7a7a", "#7ad8d8"),  # teal
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
    """Escape characters that would break labels in *our* Mermaid output.

    Mermaid flowchart syntax has many metacharacters overall (``]``, ``#``,
    ``;``, arrow tokens, etc.) — this helper does **not** attempt to cover all
    of them. It only handles the subset relevant to **this call site's inputs**:
    node ids, relation kinds, and task-constraint type strings from a validated
    :class:`ArenaEnvInitialGraphSpec` (snake_case identifiers, not free-form
    user text). For those values we escape:

    * ``"`` — would terminate a Mermaid string literal
    * ``|`` — would split an edge label (``-->|label|``)

    Brackets and newlines are not expected in spec ids/kinds; we assert rather
    than silently strip them. If other Mermaid metacharacters ever appear in
    spec fields, extend this helper for those cases explicitly.
    """
    assert "\n" not in value and "]" not in value, f"Mermaid label contains unexpected characters: {value!r}"
    return value.replace('"', "&quot;").replace("|", "&#124;")
