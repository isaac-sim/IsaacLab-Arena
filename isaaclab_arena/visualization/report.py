# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and serve an HTML evaluation report of the per-camera per-episode rollout videos."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import html
import http.server
import pathlib
import re
import socketserver
import string

# Matches the recorder output filename: <name_prefix>-env<N>-<camera_name>-episode-<E>.mp4
# See CameraObsVideoRecorder._flush_envs in camera_observation_video_recorder.py.
_VIDEO_FILENAME_PATTERN = re.compile(r"^(?P<prefix>.+)-env(?P<env>\d+)-(?P<camera>.+)-episode-(?P<episode>\d+)\.mp4$")

_TEMPLATE_PATH = pathlib.Path(__file__).parent / "report_template.html"

_DEFAULT_TITLE = "Evaluation Report"
_DEFAULT_PORT = 8000


@dataclasses.dataclass
class EpisodeVideos:
    """The recorded camera videos for a single (group, env, episode)."""

    group: str
    """Sub-directory the videos live in, relative to the scanned root (the eval job name); "" when flat."""

    env_index: int
    """Index of the environment the episode ran in."""

    episode_index: int
    """Index of the episode within that environment."""

    video_by_camera: dict[str, str]
    """Camera name -> mp4 path, relative to the scanned root (and so to the report's index.html)."""


@dataclasses.dataclass
class VideoGrid:
    """Recorded episode videos laid out as a grid: one row per (group, env, episode), one column per camera."""

    episodes: list[EpisodeVideos]
    cameras: list[str]

    @property
    def is_empty(self) -> bool:
        """Whether any episode videos were found."""
        return not self.episodes


def scan_video_dir(root: pathlib.Path) -> VideoGrid:
    """Recursively scan ``root`` for recorder mp4s and group them into a VideoGrid.

    Files that do not match the recorder's naming pattern (e.g. the kit-viewport
    ``rl-video-step-*.mp4``) are ignored. The scan recurses so the per-job sub-directories
    written by the eval runner are picked up, with the sub-directory used as the row group.

    Args:
        root: Directory of recorded rollout videos to scan.
    """
    videos_by_key: dict[tuple[str, int, int], dict[str, str]] = {}
    cameras: list[str] = []
    for path in sorted(root.rglob("*.mp4")):
        match = _VIDEO_FILENAME_PATTERN.match(path.name)
        if match is None:
            continue
        relative = path.relative_to(root)
        group = "" if relative.parent == pathlib.Path(".") else str(relative.parent)
        env_index = int(match.group("env"))
        episode_index = int(match.group("episode"))
        camera = match.group("camera")
        # Collect one entry per (group, env, episode), filling in each camera column as it is found.
        videos_by_key.setdefault((group, env_index, episode_index), {})[camera] = str(relative)
        if camera not in cameras:
            cameras.append(camera)

    cameras.sort()
    episodes = [
        EpisodeVideos(group=group, env_index=env, episode_index=episode, video_by_camera=videos)
        for (group, env, episode), videos in sorted(videos_by_key.items())
    ]
    return VideoGrid(episodes=episodes, cameras=cameras)


def _render_video_cell(src: str) -> str:
    """Render a single grid cell containing an inline, controllable video."""
    return (
        "<td>"
        '<video controls preload="metadata" muted playsinline>'
        f'<source src="{html.escape(src)}" type="video/mp4">'
        "</video>"
        "</td>"
    )


def _render_row_label(episode: EpisodeVideos) -> str:
    """Render the sticky left-hand label identifying an episode row."""
    parts = []
    if episode.group:
        parts.append(html.escape(episode.group))
    parts.append(f"env {episode.env_index}")
    parts.append(f"episode {episode.episode_index}")
    return f'<th class="rowlabel">{"<br>".join(parts)}</th>'


def _render_row(episode: EpisodeVideos, cameras: list[str]) -> str:
    """Render one table row: the row label followed by one cell per camera column."""
    cells = [_render_row_label(episode)]
    for camera in cameras:
        src = episode.video_by_camera.get(camera)
        cells.append('<td class="missing">&mdash;</td>' if src is None else _render_video_cell(src))
    return "<tr>" + "".join(cells) + "</tr>"


def render_report(grid: VideoGrid, title: str) -> str:
    """Render ``grid`` into a self-contained HTML document using the report template.

    Args:
        grid: The episode videos to lay out.
        title: Page title and heading.
    """
    header_cells = "".join(f"<th>{html.escape(camera)}</th>" for camera in grid.cameras)
    body_rows = "\n".join(_render_row(episode, grid.cameras) for episode in grid.episodes)
    summary = f"{len(grid.episodes)} episode(s) &middot; {len(grid.cameras)} camera(s)"
    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        title=html.escape(title), summary=summary, header_cells=header_cells, body_rows=body_rows
    )


def build_report(video_dir: str | pathlib.Path, title: str = _DEFAULT_TITLE) -> pathlib.Path | None:
    """Scan ``video_dir`` for recorder mp4s and write the report ``index.html``, returning its path.

    Returns ``None`` (rather than raising) when ``video_dir`` does not exist or holds no recorder
    videos, so callers can print a hint instead of crashing.

    Args:
        video_dir: Directory of recorded rollout videos to scan (the report is written here).
        title: Title and heading for the generated page.
    """
    video_dir = pathlib.Path(video_dir).resolve()
    if not video_dir.is_dir():
        return None

    grid = scan_video_dir(video_dir)
    if grid.is_empty:
        return None

    output = video_dir / "index.html"
    output.write_text(render_report(grid, title), encoding="utf-8")
    print(
        f"Wrote evaluation report with {len(grid.episodes)} episode(s) and {len(grid.cameras)} camera(s) to: {output}"
    )
    return output


def serve_until_ctrl_c(directory: pathlib.Path, port: int, filename: str) -> None:
    """Serve ``directory`` over HTTP until interrupted (Ctrl+C), printing the URL for ``filename``.

    Binds to ``0.0.0.0`` so the page is reachable from the host browser at ``http://localhost:<port>``
    (the dev container runs with ``--net=host``).

    Args:
        directory: Directory to serve as the web root.
        port: TCP port to listen on.
        filename: File within ``directory`` to point the URL at.
    """
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    url = f"http://localhost:{port}/{filename}"
    # Avoid "Address already in use" when a previous server's socket is still in TIME_WAIT.
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving evaluation report at {url} (Ctrl+C to stop).")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")


def build_and_serve_report(video_dir: str | pathlib.Path, port: int = _DEFAULT_PORT) -> None:
    """Build the evaluation report for ``video_dir`` and serve it over HTTP until interrupted.

    Prints a hint and returns immediately when no recorder videos are found.

    Args:
        video_dir: Directory of recorded rollout videos to scan and serve.
        port: TCP port to serve the report on.
    """
    output = build_report(video_dir)
    if output is None:
        print(f"No per-camera videos found in {video_dir}; nothing to report (did you pass --record_camera_video?).")
        return
    serve_until_ctrl_c(output.parent, port, output.name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and serve an HTML evaluation report of the per-camera per-episode rollout videos in a folder."
            " The report (index.html) is written into the folder and served over HTTP; because the dev container"
            " runs with --net=host, the printed http://localhost:<port> URL opens straight from the host browser."
        )
    )
    parser.add_argument("video_dir", type=str, help="Folder of recorded rollout videos to scan.")
    parser.add_argument(
        "--port", type=int, default=_DEFAULT_PORT, help=f"Port to serve on. Defaults to {_DEFAULT_PORT}."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_and_serve_report(args.video_dir, args.port)


if __name__ == "__main__":
    main()
