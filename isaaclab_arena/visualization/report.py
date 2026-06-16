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

# Matches the recorder output filename: <prefix>[-rebuild<R>]-env<N>-<camera>-episode-<E>.mp4
# See CameraObsVideoRecorder._flush_envs in camera_observation_video_recorder.py. The optional
# "-rebuild<R>" segment is added by the eval runner's per-rebuild prefix; the policy runner omits it.
_VIDEO_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:-rebuild(?P<rebuild>\d+))?-env(?P<env>\d+)-(?P<camera>.+)-episode-(?P<episode>\d+)\.mp4$"
)

_TEMPLATE_PATH = pathlib.Path(__file__).parent / "report_template.html"

_DEFAULT_TITLE = "Evaluation Report"
_DEFAULT_PORT = 8000


@dataclasses.dataclass
class EpisodeVideos:
    """The recorded camera videos for a single (env, episode) of one job."""

    env_index: int
    """Index of the environment the episode ran in."""

    episode_index: int
    """Contiguous episode index within the (job, env), spanning rebuilds (which are not surfaced)."""

    video_by_camera: dict[str, str]
    """Camera name -> mp4 path, relative to the scanned root (and so to the report's index.html)."""


@dataclasses.dataclass
class JobReport:
    """All recorded episode videos for a single eval job, laid out as an env x episode grid per camera."""

    name: str
    """The job name (its sub-directory under the run dir); "" for a single unnamed run (policy runner)."""

    cameras: list[str]
    """Ordered camera column names recorded for this job."""

    episodes: list[EpisodeVideos]
    """Episode rows, ordered by (env, episode)."""


@dataclasses.dataclass
class EvaluationReport:
    """A whole evaluation run: one or more jobs, each with its own grid of episode videos."""

    title: str
    jobs: list[JobReport]

    @property
    def is_empty(self) -> bool:
        """Whether any job recorded any episode videos."""
        return not any(job.episodes for job in self.jobs)


def _scan_jobs(root: pathlib.Path) -> list[JobReport]:
    """Recursively scan ``root`` for recorder mp4s and group them into per-job reports.

    Files that do not match the recorder's naming pattern (e.g. the kit-viewport ``rl-video-step-*.mp4``)
    are ignored. The scan recurses, so each per-job sub-directory written by the eval runner becomes a
    job; videos written directly under ``root`` (the policy runner) form a single unnamed job. The
    rebuild index encoded in the filename is used only to order and disambiguate episodes — episodes are
    renumbered into a contiguous per-(job, env) index and rebuilds are not surfaced.

    Args:
        root: Directory of recorded rollout videos to scan.
    """
    # job -> env -> {(rebuild, recorder_episode): {camera: relative_path}}
    raw: dict[str, dict[int, dict[tuple[int, int], dict[str, str]]]] = {}
    cameras_by_job: dict[str, list[str]] = {}

    for path in sorted(root.rglob("*.mp4")):
        match = _VIDEO_FILENAME_PATTERN.match(path.name)
        if match is None:
            continue
        relative = path.relative_to(root)
        job = "" if relative.parent == pathlib.Path(".") else str(relative.parent)
        rebuild = int(match.group("rebuild")) if match.group("rebuild") is not None else 0
        env_index = int(match.group("env"))
        recorder_episode = int(match.group("episode"))
        camera = match.group("camera")

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
    sections = "\n".join(_render_job_section(job) for job in report.jobs)
    num_episodes = sum(len(job.episodes) for job in report.jobs)
    summary = f"{len(report.jobs)} job(s) &middot; {num_episodes} episode(s)"
    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(title=html.escape(report.title), summary=summary, sections=sections)


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

    report = EvaluationReport(title=title, jobs=_scan_jobs(video_dir))
    if report.is_empty:
        return None

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
    # Avoid "Address already in use" when a previous server's socket is still in TIME_WAIT.
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving evaluation report at {url} (Ctrl+C to stop).")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")


def write_report(video_dir: str | pathlib.Path, serve: bool, port: int = _DEFAULT_PORT) -> pathlib.Path | None:
    """Build the evaluation report for ``video_dir`` and, when ``serve`` is set, serve it until interrupted.

    Prints a hint and returns ``None`` when no recorder videos are found. When not serving, prints the
    command to view the report later.

    Args:
        video_dir: Directory of recorded rollout videos to scan and report on.
        serve: Whether to serve the report over HTTP (blocks until Ctrl+C) once built.
        port: TCP port to serve the report on.
    """
    output = build_report(video_dir)
    if output is None:
        print(f"No per-camera videos found in {video_dir}; nothing to report (did you pass --record_camera_video?).")
        return None
    if serve:
        serve_until_ctrl_c(output.parent, port, output.name)
    else:
        print(f"To view it, run: python isaaclab_arena/visualization/report.py {video_dir}")
    return output


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
    write_report(args.video_dir, serve=True, port=args.port)


if __name__ == "__main__":
    main()
