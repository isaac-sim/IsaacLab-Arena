# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and serve a hierarchical HTML evaluation report of per-episode results and rollout videos.

The report has three levels: an overview of success rate by task and policy, a page per task showing
where its episodes got to, and a page per Run holding the episode videos. Only the Run pages
reference video, and they mount each player when it scrolls into view, so no page asks the browser
for more than a screenful of video at a time.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import re
import socketserver

from isaaclab_arena.visualization.report_data import (
    DEFAULT_POLICY_SUFFIXES,
    ExperimentSummary,
    RunExecutionReport,
    build_experiment_summary,
)
from isaaclab_arena.visualization.report_render import (
    PAGES_DIRNAME,
    episode_anchor,
    render_index,
    render_job_page,
    render_task_page,
    unique_slugs,
)

# Reverse-dated run directory written by ``timestamped_run_dir`` (e.g. ``2026-06-17_14-42-54``).
_RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

_DEFAULT_TITLE = "Evaluation Report"
_DEFAULT_PORT = 8000

__all__ = ["RunExecutionReport", "build_report", "serve_until_ctrl_c"]


def _write_pages(
    summary: ExperimentSummary,
    video_dir: pathlib.Path,
    task_hrefs: dict[str, str],
    job_page_hrefs: dict[str, list[str]],
    episode_hrefs: dict[tuple[str, int, int], str],
    episodes_per_page: int,
) -> int:
    pages_dir = video_dir / PAGES_DIRNAME
    pages_dir.mkdir(parents=True, exist_ok=True)
    # Drop pages from an earlier build so a re-run over changed results cannot leave stale, orphaned
    # pages behind. Only the two filename shapes this module writes are removed.
    for stale_page in [*pages_dir.glob("task_*.html"), *pages_dir.glob("job_*.html")]:
        stale_page.unlink()

    for task in summary.tasks:
        (pages_dir / task_hrefs[task.name]).write_text(
            render_task_page(
                summary,
                task,
                {job: pages[0] for job, pages in job_page_hrefs.items()},
                episode_hrefs,
                index_href="../index.html",
            ),
            encoding="utf-8",
        )
    num_pages = len(summary.tasks)
    for job in summary.jobs:
        pages = job_page_hrefs[job.name]
        for page_index, page_href in enumerate(pages):
            start = page_index * episodes_per_page
            page_episodes = job.episodes[start : start + episodes_per_page]
            (pages_dir / page_href).write_text(
                render_job_page(
                    summary,
                    job,
                    task_href=task_hrefs[job.task],
                    index_href="../index.html",
                    # Videos are recorded relative to the results root, one level above the pages.
                    video_prefix="../",
                    episodes=page_episodes,
                    page_index=page_index,
                    num_pages=len(pages),
                    page_hrefs=pages,
                ),
                encoding="utf-8",
            )
            num_pages += 1
    return num_pages


def build_report(
    video_dir: str | pathlib.Path,
    title: str = _DEFAULT_TITLE,
    *,
    run_executions: list[RunExecutionReport] | None = None,
    policy_suffixes: tuple[str, ...] = DEFAULT_POLICY_SUFFIXES,
    episodes_per_page: int = 100,
) -> pathlib.Path:
    """Scan ``video_dir`` for results and write the report ``index.html`` into it, returning its path.

    The task and Run pages are written into a ``report`` sub-directory beside it. The report is always
    written (the directory is created if missing); when no results are present the report is simply
    empty. Writing is independent of serving — see ``serve_until_ctrl_c``.

    Args:
        video_dir: Directory of recorded results to scan (the report is written here).
        title: Title and heading for the generated page.
        run_executions: Optional Run process results supplied by a distributed collector.
        policy_suffixes: Run-name suffixes that identify policies for task/policy grouping.
        episodes_per_page: Maximum number of episode cards per generated run page.
    """
    assert episodes_per_page > 0, "episodes_per_page must be greater than zero"
    video_dir = pathlib.Path(video_dir).resolve()
    video_dir.mkdir(parents=True, exist_ok=True)

    summary = build_experiment_summary(video_dir, title, run_executions, policy_suffixes=policy_suffixes)
    # Page filenames are derived once and threaded through both levels, so a link and the file it
    # points at can never disagree.
    task_hrefs = {name: f"task_{slug}.html" for name, slug in unique_slugs([t.name for t in summary.tasks]).items()}
    job_page_hrefs = _job_page_hrefs(summary, episodes_per_page)
    episode_hrefs = _episode_hrefs(summary, job_page_hrefs, episodes_per_page)

    index_html = render_index(
        summary,
        task_hrefs={name: f"{PAGES_DIRNAME}/{href}" for name, href in task_hrefs.items()},
    )
    output = video_dir / "index.html"
    output.write_text(index_html, encoding="utf-8")
    num_pages = (
        _write_pages(summary, video_dir, task_hrefs, job_page_hrefs, episode_hrefs, episodes_per_page)
        if summary.tasks
        else 0
    )

    print(
        f"Wrote evaluation report with {len(summary.tasks)} task(s), {len(summary.jobs)} run(s) and "
        f"{summary.num_episodes} episode(s) to: {output} (+{num_pages} linked page(s))"
    )
    if not summary.tasks and not summary.run_executions:
        print("[WARNING] No episode results or rollout videos were found; the report is empty.")
    elif summary.num_episodes == 0 and summary.run_executions:
        print("[INFO] No episodes were recorded; the report contains Run execution results only.")
    elif summary.num_videos == 0:
        print("[INFO] No rollout videos were recorded; the report contains episode results only.")
    return output


def _job_page_hrefs(summary: ExperimentSummary, episodes_per_page: int) -> dict[str, list[str]]:
    slugs = unique_slugs([job.name for job in summary.jobs])
    hrefs = {}
    for job in summary.jobs:
        num_pages = max(1, (len(job.episodes) + episodes_per_page - 1) // episodes_per_page)
        if num_pages == 1:
            hrefs[job.name] = [f"job_{slugs[job.name]}.html"]
        else:
            hrefs[job.name] = [f"job_{slugs[job.name]}_{index + 1}.html" for index in range(num_pages)]
    return hrefs


def _episode_hrefs(
    summary: ExperimentSummary,
    job_page_hrefs: dict[str, list[str]],
    episodes_per_page: int,
) -> dict[tuple[str, int, int], str]:
    hrefs = {}
    for job in summary.jobs:
        pages = job_page_hrefs[job.name]
        for index, episode in enumerate(job.episodes):
            page_href = pages[index // episodes_per_page]
            hrefs[(job.name, episode.env_index, episode.episode_index)] = f"{page_href}#{episode_anchor(episode)}"
    return hrefs


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
            "Build and serve a hierarchical HTML evaluation report of evaluation results."
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
        "--title", type=str, default=_DEFAULT_TITLE, help=f"Title for the report. Defaults to '{_DEFAULT_TITLE}'."
    )
    parser.add_argument(
        "--port", type=int, default=_DEFAULT_PORT, help=f"Port to serve on. Defaults to {_DEFAULT_PORT}."
    )
    parser.add_argument(
        "--policy_suffix",
        action="append",
        default=None,
        help="Run-name policy suffix used for grouping. May be supplied more than once.",
    )
    parser.add_argument(
        "--episodes_per_page",
        type=int,
        default=100,
        help="Maximum number of episode cards per run page. Defaults to 100.",
    )
    parser.add_argument("--no_serve", action="store_true", help="Write the report without serving it.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_dir = _resolve_results_dir(pathlib.Path(args.video_dir))
    output = build_report(
        video_dir,
        args.title,
        policy_suffixes=tuple(args.policy_suffix or DEFAULT_POLICY_SUFFIXES),
        episodes_per_page=args.episodes_per_page,
    )
    if not args.no_serve:
        serve_until_ctrl_c(output.parent, args.port, output.name)


if __name__ == "__main__":
    main()
