# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dark-theme CSS for the review GUI embedded HTML dashboard."""

DASHBOARD_CSS = """
:root {
  /* page chrome */
  --bg: #15181d;           /* charcoal page background */
  --bg-elev: #1d2128;      /* panel background */
  --bg-elev2: #262b34;    /* nested surface (cards, code blocks) */
  --border: #2f343d;       /* panel / table borders */
  --fg: #e4e6eb;           /* primary text */
  --fg-muted: #8a9099;     /* secondary text */
  --accent: #7fd17f;       /* green highlight (task badges, anchors) */
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; font: 14px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--fg); }
main { display: flex; flex-direction: column; gap: 16px; }
.panel { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.panel h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
.panel h2 .muted { color: var(--fg-muted); font-weight: 400; font-size: 13px; }
.graph-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(220px, 300px); gap: 16px; align-items: start; }
.graph-mermaid { min-width: 0; }
.graph-unary { background: var(--bg-elev2); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
.unary-heading { margin: 0 0 10px; font-size: 13px; font-weight: 600; letter-spacing: 0.02em; }
.unary-heading .muted { font-weight: 400; }
.unary-list { margin: 0; padding-left: 18px; list-style: disc; color: var(--fg); }
.unary-list li { padding: 4px 0; font-size: 12px; }
.unary-empty { margin: 0; font-size: 12px; }
code { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
       background: var(--bg-elev2); padding: 1px 6px; border-radius: 4px; }
pre { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
      background: var(--bg-elev2); padding: 10px 12px; border-radius: 6px; margin: 0;
      white-space: pre-wrap; word-break: break-word; }
.muted { color: var(--fg-muted); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
         font-weight: 600; letter-spacing: 0.03em; background: var(--bg-elev2); color: var(--fg); }
/* node-type badge fills — same palette as mermaid_graph.type_palette */
.badge.type-background { background: #3a4f7a; }       /* blue */
.badge.type-embodiment { background: #7a3a3a; }       /* red */
.badge.type-object { background: #7a6b3a; }           /* gold */
.badge.type-object_reference { background: #6b3a7a; } /* purple */
.badge.type-lighting { background: #3a7a7a; }         /* teal */
.badge.type-is_anchor { background: #3a7d44; }          /* green (anchor nodes) */
.badge.type-position_limits, .badge.type-at_pose, .badge.type-at_position { background: #6b3a7a; } /* purple */
.badge.type-task { background: #2f343d; border: 1px solid #4a5; color: var(--accent); } /* dark gray + green border */
.mermaid { background: var(--bg-elev2); padding: 8px; border-radius: 6px; min-height: 220px;
           display: flex; align-items: center; justify-content: center; margin: 0; }
table.tasks { width: 100%; border-collapse: collapse; }
table.tasks th, table.tasks td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
                                  vertical-align: top; font-size: 12px; }
table.tasks th { color: var(--fg-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
table.tasks pre { padding: 6px 8px; font-size: 11px; }
"""
