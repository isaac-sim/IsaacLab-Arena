# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

DASHBOARD_CSS = """
:root {
  --bg: #15181d;
  --bg-elev: #1d2128;
  --bg-elev2: #262b34;
  --border: #2f343d;
  --fg: #e4e6eb;
  --fg-muted: #8a9099;
  --accent: #7fd17f;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; font: 14px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--fg); }
header { margin-bottom: 16px; }
header h1 { margin: 0; font-size: 28px; font-weight: 700; }
header .sub { margin: 4px 0 0; color: var(--fg-muted); font-size: 13px; }
main { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto;
       grid-template-areas: "graph nodes" "tasks nodes"; gap: 16px; }
.graph-panel { grid-area: graph; }
.tasks-panel { grid-area: tasks; }
.nodes-panel { grid-area: nodes; }
.panel { background: var(--bg-elev); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.panel h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; letter-spacing: 0.02em; }
.panel h2 .muted { color: var(--fg-muted); font-weight: 400; font-size: 13px; }
code { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
       background: var(--bg-elev2); padding: 1px 6px; border-radius: 4px; }
pre { font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
      background: var(--bg-elev2); padding: 10px 12px; border-radius: 6px; margin: 0;
      white-space: pre-wrap; word-break: break-word; }
.muted { color: var(--fg-muted); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
         font-weight: 600; letter-spacing: 0.03em; background: var(--bg-elev2); color: var(--fg); }
.badge.type-background { background: #3a4f7a; }
.badge.type-embodiment { background: #7a3a3a; }
.badge.type-object { background: #7a6b3a; }
.badge.type-object_reference { background: #6b3a7a; }
.badge.type-lighting { background: #3a7a7a; }
.badge.type-is_anchor { background: #3a7d44; }
.badge.type-position_limits, .badge.type-at_pose, .badge.type-at_position { background: #6b3a7a; }
.badge.type-task { background: #2f343d; border: 1px solid #4a5; color: var(--accent); }
.mermaid { background: var(--bg-elev2); padding: 8px; border-radius: 6px; min-height: 220px;
           display: flex; align-items: center; justify-content: center; }
.unary { margin-top: 12px; }
.unary summary { cursor: pointer; color: var(--fg-muted); font-size: 13px; padding: 4px 0; }
.unary ul { margin: 8px 0 0; padding-left: 20px; list-style: disc; color: var(--fg); }
.unary li { padding: 3px 0; }
table.tasks { width: 100%; border-collapse: collapse; }
table.tasks th, table.tasks td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border);
                                  vertical-align: top; font-size: 12px; }
table.tasks th { color: var(--fg-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
table.tasks pre { padding: 6px 8px; font-size: 11px; }
.node-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.node-card { background: var(--bg-elev2); border: 1px solid var(--border); border-radius: 8px;
             padding: 12px; display: flex; flex-direction: column; gap: 10px; }
.node-card .thumb { aspect-ratio: 1 / 1; background: linear-gradient(135deg, #2a2f37, #1c2026);
                    border-radius: 6px; display: flex; flex-direction: column;
                    align-items: center; justify-content: center; color: var(--fg-muted);
                    position: relative; overflow: hidden; }
.node-card .thumb-rendered { background: #0e1115; }
.node-card .thumb-rendered img { width: 100%; height: 100%; object-fit: contain; display: block; }
.node-card .thumb-rendered .thumb-name { position: absolute; bottom: 0; left: 0; right: 0;
                                         padding: 4px 6px; background: rgba(15, 17, 21, 0.78);
                                         color: var(--fg); margin: 0; }
.thumb-initial { font-size: 36px; font-weight: 700; color: var(--fg); opacity: 0.6;
                 font-family: ui-monospace, monospace; }
.thumb-name { font-size: 10px; margin-top: 6px; padding: 0 8px; text-align: center; word-break: break-word; }
.node-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.node-id { font-family: ui-monospace, monospace; font-size: 13px; font-weight: 600; word-break: break-all; }
.node-yaml { font-size: 11px; }
"""
