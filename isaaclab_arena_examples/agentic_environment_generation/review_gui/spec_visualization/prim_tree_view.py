# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collapsible, searchable HTML view for the background prim tree."""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from isaaclab_arena.utils.usd_prim_tree import UsdPrimRecord


@dataclass
class _PrimNode:
    """One prim tree row with its nested children."""

    text: str
    search_text: str
    children: list[_PrimNode] = field(default_factory=list)


def _record_label(record: UsdPrimRecord, suffix: str) -> str:
    tag = record.object_type.value
    if record.joint_names:
        tag += " " + ",".join(record.joint_names)
    return f"{suffix} ({tag})"


def build_prim_nodes(prim_tree: list[UsdPrimRecord]) -> list[_PrimNode]:
    """Build nested prim nodes from sorted ``UsdPrimRecord`` entries."""
    records = sorted(prim_tree, key=lambda record: record.relative_path)
    roots: list[_PrimNode] = []
    stack: list[tuple[str, _PrimNode]] = []
    for record in records:
        path = record.relative_path
        while stack and not path.startswith(stack[-1][0] + "/"):
            stack.pop()
        parent_path = stack[-1][0] if stack else ""
        suffix = path[len(parent_path) + 1 :] if parent_path else path
        node = _PrimNode(
            text=_record_label(record, suffix),
            search_text=path.lower(),
        )
        if stack:
            stack[-1][1].children.append(node)
        else:
            roots.append(node)
        stack.append((path, node))
    return roots


def estimate_prim_tree_height_px(prim_tree: list[UsdPrimRecord]) -> int:
    """Heuristic iframe height so typical trees fit without excessive scrolling."""
    return max(160, min(560, 22 * max(len(prim_tree), 1) + 64))


def _render_nodes(nodes: list[_PrimNode]) -> str:
    parts: list[str] = []
    for node in nodes:
        has_children = bool(node.children)
        classes = "li" if has_children else "li leaf"
        text = html.escape(node.text)
        search_text = html.escape(node.search_text)
        children_html = f'<div class="children">{_render_nodes(node.children)}</div>' if has_children else ""
        parts.append(
            f'<div class="{classes}" data-text="{search_text}">'
            f'<div class="row"><span class="toggle"></span><span class="txt">{text}</span></div>'
            f"{children_html}</div>"
        )
    return "".join(parts)


def render_prim_tree_html(prim_tree: list[UsdPrimRecord]) -> str:
    """Render a self-contained HTML document with a collapsible, searchable prim tree.

    Groups collapse by parent/child structure, and the live search keeps a matching
    prim's descendants (and its ancestors, so the match stays reachable).
    """
    tree_html = _render_nodes(build_prim_nodes(prim_tree))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; padding: 8px; background: transparent;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; color: #d0d0d0; }}
  #search {{ width: 100%; box-sizing: border-box; margin-bottom: 8px; padding: 4px 6px;
            background: #1a1c22; color: #d0d0d0; border: 1px solid #3a3d44; border-radius: 4px; }}
  .li.collapsed > .children {{ display: none; }}
  .children {{ padding-left: 16px; }}
  .row {{ white-space: pre; line-height: 1.5; }}
  .toggle {{ display: inline-block; width: 12px; cursor: pointer; color: #7aa0d8; user-select: none; }}
  .toggle::before {{ content: '\\25BE'; }}
  .li.collapsed > .row > .toggle::before {{ content: '\\25B8'; }}
  .li.leaf > .row > .toggle {{ visibility: hidden; }}
  .hidden {{ display: none; }}
  #empty {{ color: #888; padding: 4px 0; }}
</style>
</head>
<body>
<input id="search" type="text" placeholder="Search prims (keeps children of matches)…" autocomplete="off">
<div id="tree">{tree_html}</div>
<div id="empty" class="hidden">(no matching prims)</div>
<script>
  const tree = document.getElementById('tree');
  const empty = document.getElementById('empty');
  const search = document.getElementById('search');

  tree.querySelectorAll('.li:not(.leaf) > .row > .toggle').forEach(function (toggle) {{
    toggle.addEventListener('click', function () {{
      toggle.closest('.li').classList.toggle('collapsed');
    }});
  }});

  function childItems(li) {{
    const box = li.querySelector(':scope > .children');
    return box ? Array.from(box.querySelectorAll(':scope > .li')) : [];
  }}

  function filter(li, query, forcedByAncestor) {{
    const text = li.dataset.text;
    const selfMatch = query === '' || text.indexOf(query) !== -1;
    let descMatch = false;
    childItems(li).forEach(function (child) {{
      if (filter(child, query, forcedByAncestor || selfMatch)) descMatch = true;
    }});
    const show = query === '' ? true : (selfMatch || descMatch || forcedByAncestor);
    li.classList.toggle('hidden', !show);
    if (query !== '' && (selfMatch || descMatch)) li.classList.remove('collapsed');
    return selfMatch || descMatch;
  }}

  search.addEventListener('input', function () {{
    const query = search.value.trim().toLowerCase();
    let any = false;
    Array.from(tree.querySelectorAll(':scope > .li')).forEach(function (li) {{
      if (filter(li, query, false)) any = true;
    }});
    empty.classList.toggle('hidden', query === '' || any);
  }});
</script>
</body>
</html>
"""
