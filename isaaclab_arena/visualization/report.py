# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and serve an HTML evaluation report of the per-camera per-episode rollout videos."""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import pathlib
import re
import socketserver
import string
from dataclasses import dataclass

from isaaclab_arena.video.camera_observation_video_recorder import parse_episode_video_filename

# Reverse-dated run directory written by ``timestamped_run_dir`` (e.g. ``2026-06-17_14-42-54``).
_RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

_TEMPLATE_PATH = pathlib.Path(__file__).parent / "report_template.html"

_DEFAULT_TITLE = "Evaluation Report"
_DEFAULT_PORT = 8000


@dataclass
class EpisodeVideos:
    """The recorded camera videos for a single (env, episode) of one job."""

    env_index: int
    """Index of the environment the episode ran in."""

    episode_index: int
    """Episode index within the (job, env)"""

    video_by_camera: dict[str, str]
    """Camera name -> mp4 path, relative to the scanned root (and so to the report's index.html)."""


@dataclass
class JobReport:
    """All recorded episode videos for a single eval job."""

    name: str
    """The job name."""

    cameras: list[str]
    """Ordered camera names recorded for this job."""

    episodes: list[EpisodeVideos]
    """All recorded episode videos for this job."""


@dataclass
class EvaluationReport:
    """A whole evaluation run: one or more jobs, each with its own grid of episode videos."""

    title: str
    jobs: list[JobReport]


def _scan_jobs(root: pathlib.Path) -> list[JobReport]:
    """Recursively scan ``root`` for recorder mp4s and group them into per-job reports.

    Intended for use with two different output folder structures:
    - The eval_runner.py writes one per-job sub-directory under ``root``.
    - The policy_runner.py writes directly under ``root``.

    Files that do not match the recorder's naming pattern are ignored.

    Args:
        root: Directory of evaluation results to scan.
    """
    # job -> env -> {(rebuild, recorder_episode): {camera: relative_path}}
    raw: dict[str, dict[int, dict[tuple[int, int], dict[str, str]]]] = {}
    cameras_by_job: dict[str, list[str]] = {}

    for path in sorted(root.rglob("*.mp4")):
        parsed = parse_episode_video_filename(path.name)
        if parsed is None:
            continue
        relative = path.relative_to(root)
        job = "" if relative.parent == pathlib.Path(".") else str(relative.parent)
        rebuild = parsed.rebuild_index if parsed.rebuild_index is not None else 0
        env_index = parsed.env_index
        recorder_episode = parsed.episode_index
        camera = parsed.camera_name

        envs = raw.setdefault(job, {})
        recordings = envs.setdefault(env_index, {})
        recordings.setdefault((rebuild, recorder_episode), {})[camera] = str(relative)

        cameras = cameras_by_job.setdefault(job, [])
        if camera not in cameras:
            cameras.append(camera)

    jobs = []
    for job in sorted(raw):
        episodes = []
        for env_index in sorted(raw[job]):
            # Renumber (rebuild, recorder_episode) pairs into a contiguous, rebuild-agnostic index.
            for episode_index, recording_key in enumerate(sorted(raw[job][env_index])):
                episodes.append(
                    EpisodeVideos(
                        env_index=env_index,
                        episode_index=episode_index,
                        video_by_camera=raw[job][env_index][recording_key],
                    )
                )
        jobs.append(JobReport(name=job, cameras=sorted(cameras_by_job[job]), episodes=episodes))
    return jobs


def _render_video_cell(src: str) -> str:
    """Render a single grid cell containing an inline, controllable video."""
    return (
        "<td>"
        '<video controls preload="metadata" muted playsinline>'
        f'<source src="{html.escape(src)}" type="video/mp4">'
        "</video>"
        "</td>"
    )


def _render_row(episode: EpisodeVideos, cameras: list[str]) -> str:
    """Render one table row: the env/episode label followed by one cell per camera column."""
    cells = [f'<th class="rowlabel">env {episode.env_index}<br>episode {episode.episode_index}</th>']
    for camera in cameras:
        src = episode.video_by_camera.get(camera)
        cells.append('<td class="missing">&mdash;</td>' if src is None else _render_video_cell(src))
    return "<tr>" + "".join(cells) + "</tr>"


def _render_job_section(job: JobReport) -> str:
    """Render one job as a heading (when named) followed by its env x episode video grid."""
    heading = f"<h2>{html.escape(job.name)}</h2>" if job.name else ""
    header_cells = "".join(f"<th>{html.escape(camera)}</th>" for camera in job.cameras)
    body_rows = "\n".join(_render_row(episode, job.cameras) for episode in job.episodes)
    return (
        f"<section>{heading}<table>"
        f'<thead><tr><th class="rowlabel">env / episode</th>{header_cells}</tr></thead>'
        f"<tbody>\n{body_rows}\n</tbody></table></section>"
    )


def render_report(report: EvaluationReport) -> str:
    """Render ``report`` into a self-contained HTML document using the report template."""
    num_episodes = sum(len(job.episodes) for job in report.jobs)
    summary = f"{len(report.jobs)} job(s) &middot; {num_episodes} episode(s)"
    sections = "\n".join(_render_job_section(job) for job in report.jobs) or "<p>No results recorded yet.</p>"
    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(title=html.escape(report.title), summary=summary, sections=sections)


def build_report(video_dir: str | pathlib.Path, title: str = _DEFAULT_TITLE) -> pathlib.Path:
    """Scan ``video_dir`` for results and write the report ``index.html`` into it, returning its path.

    The report is always written (the directory is created if missing); when no results are present the
    report is simply empty. Writing is independent of serving — see ``serve_until_ctrl_c``.

    Args:
        video_dir: Directory of recorded results to scan (the report is written here).
        title: Title and heading for the generated page.
    """
    video_dir = pathlib.Path(video_dir).resolve()
    video_dir.mkdir(parents=True, exist_ok=True)

    report = EvaluationReport(title=title, jobs=_scan_jobs(video_dir))
    output = video_dir / "index.html"
    output.write_text(render_report(report), encoding="utf-8")
    num_episodes = sum(len(job.episodes) for job in report.jobs)
    print(f"Wrote evaluation report with {len(report.jobs)} job(s) and {num_episodes} episode(s) to: {output}")
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
    # Avoid "Address already in use".
    socketserver.TCPServer.allow_reuse_address = True
    try:
        server = socketserver.TCPServer(("0.0.0.0", port), handler)
    except OSError as e:
        # The port is held by another process. The report is already written to disk, so fail
        # gracefully rather than crashing after a long run.
        print(
            f"Could not serve the evaluation report on port {port} ({e}). The report is written to"
            f" {directory / filename}; open it directly, or re-run with a different port."
        )
        return
    with server as httpd:
        print(f"Serving evaluation report at {url} (Ctrl+C to stop).")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")


def _resolve_results_dir(video_dir: pathlib.Path) -> pathlib.Path:
    """Return the directory to report on, descending into the most recent dated run dir when present.

    When ``video_dir`` is a parent that holds reverse-dated run sub-directories (as written by
    ``timestamped_run_dir``, e.g. ``isaaclab_arena/output``), the newest one is used so the user can
    point at the output root and get the latest results. Otherwise ``video_dir`` is returned unchanged.

    Args:
        video_dir: Directory the user pointed at.
    """
    if not video_dir.is_dir():
        return video_dir
    run_dirs = sorted(child for child in video_dir.iterdir() if child.is_dir() and _RUN_DIR_PATTERN.match(child.name))
    if not run_dirs:
        return video_dir
    # Names sort chronologically, so the last is the most recent run.
    most_recent = run_dirs[-1]
    print(f"Using most recent run directory in {video_dir}: {most_recent.name}")
    return most_recent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and serve an HTML evaluation report of evaluation results."
            " The report (index.html) is written alongside the evaluation data into the folder and served over HTTP"
        )
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        type=str,
        help=(
            "Folder of recorded rollout videos to scan. May also be a parent of the reverse-dated run"
            " directories (e.g. the output root), in which case the most recent run is reported on."
        ),
    )
    parser.add_argument(
        "--port", type=int, default=_DEFAULT_PORT, help=f"Port to serve on. Defaults to {_DEFAULT_PORT}."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_dir = _resolve_results_dir(pathlib.Path(args.video_dir))
    output = build_report(video_dir)
    serve_until_ctrl_c(output.parent, args.port, output.name)


if __name__ == "__main__":
    main()
