# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a browsable HTML gallery from a policy-runner output folder.

Scans a folder for the per-(env, camera, episode) mp4s written by
``CameraObsVideoRecorder`` and writes an ``index.html`` laying them out in a
grid: one row per (env_idx, episode_idx), one column per camera. Videos are
referenced by relative path, so the html must stay next to the mp4s.

Files that do not match the recorder's naming pattern (e.g. the kit-viewport
``rl-video-step-*.mp4``) are ignored.

Usage:
    python isaaclab_arena/visualization/gallery.py outputs/
    python isaaclab_arena/visualization/gallery.py outputs/ -o gallery.html --title "Run 42"

When running inside the dev container, the host browser cannot be launched
directly. Use ``--serve`` to host the gallery over HTTP; because the container
runs with ``--net=host``, the printed ``http://localhost:<port>`` URL opens
straight from the host browser.
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import os
import re
import socketserver
import webbrowser
from pathlib import Path

# Matches: <name_prefix>-env<N>-<camera_name>-episode-<E>.mp4
# See CameraObsVideoRecorder._flush_envs in camera_observation_video_recorder.py.
PATTERN = re.compile(r"^(?P<prefix>.+)-env(?P<env>\d+)-(?P<camera>.+)-episode-(?P<episode>\d+)\.mp4$")


def parse_folder(folder: Path) -> tuple[dict[tuple[int, int], dict[str, str]], list[str]]:
    """Scan a folder for recorder mp4s and group them by (env_idx, episode_idx).

    Args:
        folder: Directory containing the per-(env, camera, episode) mp4 files.

    Returns:
        A tuple ``(grid, cameras)`` where ``grid`` maps ``(env_idx, episode_idx)``
        to a ``{camera_name: filename}`` dict, and ``cameras`` is the ordered list
        of unique camera names (discovery order, then alphabetical).
    """
    grid: dict[tuple[int, int], dict[str, str]] = {}
    cameras: list[str] = []
    for name in sorted(os.listdir(folder)):
        if not (folder / name).is_file():
            continue
        match = PATTERN.match(name)
        if match is None:
            continue
        env_idx = int(match.group("env"))
        episode_idx = int(match.group("episode"))
        camera = match.group("camera")
        grid.setdefault((env_idx, episode_idx), {})[camera] = name
        if camera not in cameras:
            cameras.append(camera)
    cameras.sort()
    return grid, cameras


def render_html(
    grid: dict[tuple[int, int], dict[str, str]],
    cameras: list[str],
    output_dir: Path,
    folder: Path,
    title: str,
) -> str:
    """Render the gallery as a single self-contained HTML string.

    Args:
        grid: Mapping of ``(env_idx, episode_idx)`` to ``{camera_name: filename}``.
        cameras: Ordered list of camera column names.
        output_dir: Directory the html will be written to (used for relative paths).
        folder: Source folder that was scanned (shown in the summary).
        title: Page title and heading.

    Returns:
        The complete HTML document as a string.
    """
    keys = sorted(grid.keys())
    n_envs = len({env for env, _ in keys})
    n_episodes = len({episode for _, episode in keys})

    header_cells = "".join(f"<th>{html.escape(c)}</th>" for c in cameras)

    rows = []
    for env_idx, episode_idx in keys:
        cells = [f'<th class="rowlabel">env {env_idx}<br>episode {episode_idx}</th>']
        cams = grid[(env_idx, episode_idx)]
        for camera in cameras:
            filename = cams.get(camera)
            if filename is None:
                cells.append('<td class="missing">&mdash;</td>')
                continue
            src = html.escape(os.path.relpath(folder / filename, start=output_dir))
            cells.append(
                "<td>"
                '<video controls preload="metadata" muted playsinline>'
                f'<source src="{src}" type="video/mp4">'
                "</video>"
                "</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    summary = (
        f"{len(keys)} episode(s) &middot; {n_envs} env(s) &middot; "
        f"{n_episodes} episode index(es) &middot; {len(cameras)} camera(s) "
        f"&middot; source: <code>{html.escape(str(folder))}</code>"
    )
    body_rows = "\n".join(rows)
    escaped_title = html.escape(title)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 0.25rem; }}
  .summary {{ color: #555; font-size: 0.9rem; margin-bottom: 1rem; }}
  code {{ font-family: ui-monospace, monospace; background: #eee; padding: 0 0.25rem; border-radius: 3px; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem; vertical-align: top; }}
  thead th {{ position: sticky; top: 0; background: #f0f0f0; z-index: 2; font-size: 0.85rem; }}
  .rowlabel {{ position: sticky; left: 0; background: #f0f0f0; font-family: ui-monospace, monospace;
               font-size: 0.8rem; white-space: nowrap; text-align: left; z-index: 1; }}
  video {{ width: 320px; height: auto; display: block; background: #000; }}
  .missing {{ color: #bbb; text-align: center; }}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
<div class="summary">{summary}</div>
<table>
<thead><tr><th class="rowlabel">env / episode</th>{header_cells}</tr></thead>
<tbody>
{body_rows}
</tbody>
</table>
</body>
</html>
"""


def build_gallery(
    folder: str | Path, output: str | Path | None = None, title: str = "Evaluation Gallery"
) -> Path | None:
    """Scan ``folder`` for recorder mp4s and write the gallery HTML, returning its path.

    Args:
        folder: Directory containing the per-(env, camera, episode) mp4 files.
        output: Output HTML path. Defaults to ``<folder>/index.html``.
        title: Title and heading for the generated page.

    Returns:
        The path to the written HTML file, or ``None`` if no matching videos were found.
    """
    folder = Path(folder).resolve()
    assert folder.is_dir(), f"Not a directory: {folder}"
    output = Path(output).resolve() if output else folder / "index.html"

    grid, cameras = parse_folder(folder)
    if not grid:
        return None

    document = render_html(grid, cameras, output.parent, folder, title)
    output.write_text(document, encoding="utf-8")
    print(f"Wrote gallery with {len(grid)} episode(s) and {len(cameras)} camera(s) to: {output}")
    return output


def in_container() -> bool:
    """Best-effort detection of whether we are running inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as f:
            return any(marker in f.read() for marker in ("docker", "containerd", "kubepods"))
    except OSError:
        return False


def serve_forever(directory: Path, port: int, filename: str) -> None:
    """Serve ``directory`` over HTTP until interrupted, printing the URL for ``filename``.

    Binds to ``0.0.0.0`` so the page is reachable from the host browser at
    ``http://localhost:<port>`` (the dev container runs with ``--net=host``).
    On a host (non-container) machine the page is also opened automatically.

    Args:
        directory: Directory to serve as the web root.
        port: TCP port to listen on.
        filename: File within ``directory`` to point the URL at.
    """
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    url = f"http://localhost:{port}/{filename}"
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving {directory} at {url}")
        print("Open that URL in your host browser (Ctrl+C to stop).")
        if not in_container():
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=str, help="Folder of policy-runner output videos to scan.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output HTML path. Defaults to <folder>/index.html.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Evaluation Gallery",
        help="Title and heading for the generated page.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Host the gallery over HTTP instead of opening a file. Recommended inside the dev container "
            "(reachable from the host browser at http://localhost:<port> thanks to --net=host)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve. Defaults to 8000.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated page in a web browser.",
    )
    args = parser.parse_args()

    output = build_gallery(args.folder, args.output, args.title)
    assert (
        output is not None
    ), f"No recorder videos (matching '<prefix>-env<N>-<camera>-episode-<E>.mp4') found in {args.folder}"

    if args.serve:
        serve_forever(output.parent, args.port, output.name)
    elif not args.no_open:
        if in_container():
            print(
                "Detected a container: cannot launch the host browser directly. "
                f"Re-run with --serve to view it from your host browser, or open manually: {output}"
            )
        elif not webbrowser.open(output.as_uri()):
            print(f"Could not open a browser automatically. Open this file manually: {output}")


if __name__ == "__main__":
    main()
