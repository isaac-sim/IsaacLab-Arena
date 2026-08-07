# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render an aggregated Experiment into static HTML report pages."""

from __future__ import annotations

import html
import pathlib
import re
import string
from urllib.parse import quote

from isaaclab_arena.visualization.episode_results_files import REPORT_DIRNAME, DataIssue
from isaaclab_arena.visualization.report_data import (
    ExperimentSummary,
    JobSummary,
    ObjectiveFunnel,
    RunExecutionReport,
    TaskSummary,
    is_completed_execution,
    is_failed_execution,
)

_TEMPLATE_PATH = pathlib.Path(__file__).parent / "report_template.html"

# Sub-directory holding the task and run pages, keeping them out of the results directory itself.
PAGES_DIRNAME = REPORT_DIRNAME

_NUM_RAMP_STEPS = 7
_MAX_FUNNEL_STAGE_STEP = 2
_UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")

_OUTCOME_GLYPHS = {"success": "&check;", "partial": "&#9680;", "fail": "&times;", "unknown": "&middot;"}
_OUTCOME_LABELS = {"success": "success", "partial": "partial", "fail": "no progress", "unknown": "not scored"}


def unique_slugs(names: list[str]) -> dict[str, str]:
    """Return a filesystem-safe, collision-free slug for each name."""
    slugs: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        base = _UNSAFE_FILENAME_CHARACTERS.sub("_", name).strip("_") or "unnamed"
        slug = base
        suffix = 2
        while slug in used:
            slug = f"{base}_{suffix}"
            suffix += 1
        used.add(slug)
        slugs[name] = slug
    return slugs


def episode_anchor(episode) -> str:
    return f"ep-{episode.env_index}-{episode.episode_index}"


def _percent(fraction: float | None) -> str:
    return "&mdash;" if fraction is None else f"{fraction * 100:.0f}%"


def _ramp_rank(fraction: float) -> int:
    return min(_NUM_RAMP_STEPS - 1, max(0, int(fraction * _NUM_RAMP_STEPS)))


def _episode_outcome(episode) -> str:
    if episode.success is True:
        return "success"
    if episode.success is None:
        return "unknown"
    progress = episode.progress_fraction
    return "partial" if progress is not None and progress > 0 else "fail"


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def _render_failed_runs_section(run_executions: list[RunExecutionReport]) -> str:
    failed = [execution for execution in run_executions if is_failed_execution(execution)]
    if not failed:
        return ""
    rows = "\n".join(
        f"<tr><th>{html.escape(execution.run_name)}</th>"
        f"<td><code>{html.escape(str(execution.process_exit_code))}</code></td></tr>"
        for execution in failed
    )
    return (
        f"<section><h2>Failed runs ({len(failed)})</h2>"
        '<p class="note">These runs did not complete and are excluded from episode results.</p>'
        "<table><thead><tr><th>run</th><th>process exit code</th></tr></thead>"
        f"<tbody>\n{rows}\n</tbody></table></section>"
    )


def _render_issues(issues: list[DataIssue]) -> str:
    if not issues:
        return ""
    rows = "\n".join(
        f"<tr><th>{html.escape(issue.path)}</th><td>{html.escape(issue.message)}</td></tr>" for issue in issues[:200]
    )
    more = "" if len(issues) <= 200 else f'<p class="note">Showing 200 of {len(issues):,} issue(s).</p>'
    return (
        f"<section><h2>Data issues ({len(issues):,})</h2>"
        '<p class="note">The report skipped malformed records and kept rendering usable artifacts.</p>'
        "<table><thead><tr><th>path</th><th>issue</th></tr></thead>"
        f"<tbody>\n{rows}\n</tbody></table>{more}</section>"
    )


def _render_ramp_legend() -> str:
    swatches = "".join(f'<span class="swatch cell" data-rank="{rank}"></span>' for rank in range(_NUM_RAMP_STEPS))
    return f'<div class="ramp-legend"><span>0%</span>{swatches}<span>100%</span><span>success rate</span></div>'


def _render_matrix(summary: ExperimentSummary, task_hrefs: dict[str, str]) -> str:
    header_cells = "".join(
        f'<th class="num" data-sort-col="{index}">{html.escape(policy)}</th>'
        for index, policy in enumerate(summary.policies)
    )
    rows = []
    for task in summary.tasks:
        cells = [
            f'<th data-sort-col="task" data-value="{html.escape(task.name)}">'
            f'<a href="{_href(task_hrefs[task.name])}">{html.escape(task.name)}</a></th>'
        ]
        for index, policy in enumerate(summary.policies):
            job = task.job_for_policy(policy)
            rate = job.success_rate if job is not None else None
            if job is None or rate is None:
                cells.append(f'<td class="cell missing" data-sort-col="{index}" data-value="">&mdash;</td>')
                continue
            cells.append(
                f'<td class="cell" data-rank="{_ramp_rank(rate)}" data-sort-col="{index}" data-value="{rate:.6f}">'
                f'<a href="{_href(task_hrefs[task.name])}"'
                f' title="{html.escape(job.name)}: {job.num_successes}/{job.num_scored_episodes} episodes">'
                f"{_percent(rate)}</a></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f"<section><h2>Success rate by task and policy</h2>{_render_ramp_legend()}"
        '<table class="matrix"><thead><tr><th data-sort-col="task">task</th>'
        f"{header_cells}</tr></thead><tbody>\n"
        + "\n".join(rows)
        + "</tbody></table>"
        '<p class="note">Click a task to see where its episodes failed. Click a column heading to sort.</p>'
        "</section>"
    )


def _render_ungrouped_job_list(summary: ExperimentSummary, task_hrefs: dict[str, str]) -> str:
    """Render a flat list of Runs, used when no task and policy labels could be established.

    Rows link to the Run's task page rather than straight to its episodes, so descending through the
    report always passes the level that shows where episodes got to, however the Runs are grouped.
    """
    rows = "\n".join(
        f'<tr><th><a href="{_href(task_hrefs[job.task])}">{html.escape(job.name or "results")}</a></th>'
        f'<td class="num">{job.num_episodes}</td>'
        f'<td class="num">{_percent(job.success_rate)}</td></tr>'
        for job in summary.jobs
    )
    return (
        "<section><h2>Runs</h2>"
        "<table><thead><tr><th>run</th><th>episodes</th><th>success rate</th></tr></thead>"
        f"<tbody>\n{rows}\n</tbody></table></section>"
    )


def render_index(summary: ExperimentSummary, task_hrefs: dict[str, str]) -> str:
    """Render the overview page, whose rows descend to a task page whether or not Runs are grouped."""
    tiles = [
        _tile("Tasks", str(len(summary.tasks))),
        _tile("Runs", str(len(summary.jobs))),
        _tile("Episodes", f"{summary.num_episodes:,}"),
        _tile("Success rate", _percent(summary.overall_success_rate)),
    ]
    if summary.is_grouped:
        tiles.insert(1, _tile("Policies", str(len(summary.policies))))
    for policy in summary.policies:
        tiles.append(
            _tile(
                policy,
                _percent(summary.success_rate_for_policy(policy)),
                f"{summary.num_episodes_for_policy(policy):,} episodes",
            )
        )

    body = (
        _render_matrix(summary, task_hrefs) if summary.is_grouped else _render_ungrouped_job_list(summary, task_hrefs)
    )
    content = (
        f'<div class="tiles">{"".join(tiles)}</div>'
        f"{_render_failed_runs_section(summary.run_executions)}"
        f"{_render_issues(summary.issues)}{body}"
    )
    if not summary.tasks and not summary.run_executions:
        content += "<p>No results recorded yet.</p>"
    content += f'<p class="note">{_grouping_note(summary)}</p>'

    return _render_page(
        title=html.escape(summary.title),
        heading=html.escape(summary.title),
        breadcrumb=f'<span class="current">{html.escape(summary.title)}</span>',
        summary=_experiment_summary_line(summary),
        content=content,
    )


def _grouping_note(summary: ExperimentSummary) -> str:
    if summary.grouping_source == "policy_suffixes":
        return "Tasks and policies were inferred from run-name policy suffixes."
    if summary.grouping_source == "run_names":
        return "Tasks and policies were inferred from repeated final tokens in run names."
    return "Runs could not be grouped into tasks and policies, so they are listed individually."


def _experiment_summary_line(summary: ExperimentSummary) -> str:
    if summary.run_executions:
        completed = sum(is_completed_execution(execution) for execution in summary.run_executions)
        failed = sum(is_failed_execution(execution) for execution in summary.run_executions)
        return (
            f"{len(summary.run_executions)} run(s) &middot; {completed} completed &middot; "
            f"{failed} failed &middot; {summary.num_episodes} episode(s)"
        )
    return (
        f"{len(summary.jobs)} run(s) &middot; {summary.num_episodes} episode(s) &middot; {summary.num_videos} video(s)"
    )


def _render_funnel(funnel: ObjectiveFunnel) -> str:
    if not funnel.stages:
        return ""
    rows = []
    for stage in funnel.stages:
        fraction = 0.0 if funnel.num_instances == 0 else stage.num_reached / funnel.num_instances
        step = min(stage.index, _MAX_FUNNEL_STAGE_STEP)
        rows.append(
            '<div class="funnel-row">'
            f'<div class="stage-label"><span class="name">{html.escape(stage.name)}</span>'
            f'<span class="value">{stage.num_reached:,} &middot; {_percent(fraction)}</span></div>'
            f'<div class="bar-track"><div class="bar" data-stage="{step}"'
            f' style="width: {fraction * 100:.1f}%"></div></div></div>'
        )
    return (
        f'<div class="funnel"><h3>{html.escape(funnel.name)}</h3>'
        + "".join(rows)
        + f'<p class="note">{funnel.num_instances:,} objective instance(s)</p></div>'
    )


def _render_job_funnels(job: JobSummary) -> str:
    funnels = "".join(_render_funnel(funnel) for funnel in job.funnels)
    if not funnels:
        return ""
    return (
        f'<div class="funnel"><h3>{html.escape(job.policy or job.name)}</h3>'
        f'<p class="note">success {_percent(job.success_rate)} &middot; mean progress {_percent(job.mean_progress)}</p>'
        "</div>"
        + funnels
    )


def _render_chip(episode, href: str) -> str:
    outcome = _episode_outcome(episode)
    progress = episode.progress_fraction
    progress_text = "" if progress is None else f", progress {progress * 100:.0f}%"
    tooltip = f"env {episode.env_index} episode {episode.episode_index}: {_OUTCOME_LABELS[outcome]}{progress_text}"
    return (
        f'<a class="chip {outcome}" href="{_href(href)}" title="{html.escape(tooltip)}">{_OUTCOME_GLYPHS[outcome]}</a>'
    )


def _render_legend() -> str:
    items = "".join(
        f'<span class="item"><span class="chip {outcome}">{_OUTCOME_GLYPHS[outcome]}</span>'
        f"{html.escape(_OUTCOME_LABELS[outcome])}</span>"
        for outcome in ("success", "partial", "fail", "unknown")
    )
    return f'<div class="legend">{items}</div>'


def render_task_page(
    summary: ExperimentSummary,
    task: TaskSummary,
    job_hrefs: dict[str, str],
    episode_hrefs: dict[tuple[str, int, int], str],
    index_href: str,
) -> str:
    """Render one task's page: each policy's funnel and episode outcome chips."""
    tiles = [_tile("Episodes", f"{task.num_episodes:,}")]
    for job in task.jobs:
        tiles.append(_tile(job.policy or job.name, _percent(job.success_rate), f"{job.num_successes:,} successes"))

    funnels = "".join(_render_job_funnels(job) for job in task.jobs)
    funnel_section = (
        f'<section><h2>Where episodes got to</h2><div class="funnels">{funnels}</div></section>' if funnels else ""
    )

    chip_sections = []
    for job in task.jobs:
        chips = "".join(
            _render_chip(episode, episode_hrefs[(job.name, episode.env_index, episode.episode_index)])
            for episode in job.episodes
        )
        chip_sections.append(
            f"<section><h2>{html.escape(job.policy or job.name)} episodes</h2>"
            f'<p class="note"><a href="{_href(job_hrefs[job.name])}">Open '
            f"{html.escape(job.name)} to watch the videos</a></p>"
            f'{_render_legend()}<div class="chips">{chips}</div></section>'
        )

    breadcrumb = (
        f'<a href="{_href(index_href)}">{html.escape(summary.title)}</a>'
        f'<span class="sep">/</span><span class="current">{html.escape(task.name)}</span>'
        + _render_up_button(index_href, "Back to the overview", extra_class="up")
    )
    context = [_render_pill("task", task.name)]
    policies = [job.policy for job in task.jobs if job.policy]
    if policies:
        context.append(_render_pill("comparing", ", ".join(policies), extra_class="policy"))
    footer = _render_footer_nav([_render_up_button(index_href, "Back to the overview")])

    return _render_page(
        title=f"{html.escape(task.name)} &mdash; {html.escape(summary.title)}",
        heading=html.escape(task.name),
        breadcrumb=breadcrumb,
        summary=f"{len(task.jobs)} run(s) &middot; {task.num_episodes} episode(s)",
        content=f'<div class="tiles">{"".join(tiles)}</div>{funnel_section}{"".join(chip_sections)}{footer}',
        context=_render_context(context),
    )


def _render_metadata_entry(key: str, value: object) -> str:
    if isinstance(value, dict):
        sub_rows = "".join(
            f'<div class="subitem"><span class="k">{html.escape(str(sub_key))}</span>'
            f" {html.escape(str(sub_value))}</div>"
            for sub_key, sub_value in value.items()
        )
        return f'<div><span class="k">{html.escape(key)}</span>{sub_rows}</div>'
    return f'<div><span class="k">{html.escape(key)}</span> {html.escape(str(value))}</div>'


def _render_signal(signal) -> str:
    if signal.triggered:
        state, glyph = "on", "&check;"
        suffix = "" if signal.step is None else f'<span class="step">step {signal.step}</span>'
    elif signal.blocked:
        state, glyph = "blocked", "&#9654;"
        suffix = '<span class="step">waiting</span>'
    else:
        state, glyph = "off", "&#9675;"
        suffix = ""
    tooltip = signal.detail or signal.name
    return (
        f'<span class="signal {state}" title="{html.escape(tooltip)}">'
        f'<span class="glyph">{glyph}</span>{html.escape(signal.name)}{suffix}</span>'
    )


def _render_objective(objective) -> str:
    track = "".join(_render_signal(signal) for signal in objective.signals)
    if objective.blocked_predicates:
        blocked = ", ".join(objective.blocked_predicates)
        track += f'<span class="signal blocked"><span class="glyph">&#9654;</span>{html.escape(blocked)}</span>'
    score = f"{round(objective.score, 2):g} / {round(objective.max_score, 2):g}"
    family = "" if objective.family == objective.name else f'<span class="score">{html.escape(objective.family)}</span>'
    return (
        '<div class="objective"><div class="objective-head">'
        f'<span class="name">{html.escape(objective.name)}</span>{family}'
        f'<span class="score">{html.escape(score)}</span></div>'
        f'<div class="track">{track}</div></div>'
    )


def _render_signals(objectives: list) -> str:
    if not objectives:
        return ""
    if len(objectives) == 1:
        return f'<div class="signals">{_render_objective(objectives[0])}</div>'

    num_triggered = sum(objective.num_triggered for objective in objectives)
    num_signals = sum(len(objective.signals) for objective in objectives)
    num_complete = sum(1 for objective in objectives if objective.is_complete)
    body = "".join(_render_objective(objective) for objective in objectives)
    return (
        f'<details class="signals"><summary>{num_triggered} of {num_signals} known signals triggered '
        f"across {len(objectives)} objectives &middot; {num_complete} complete</summary>{body}</details>"
    )


def _render_episode_card(episode, cameras: list[str], video_prefix: str, policy: str = "", objectives=None) -> str:
    outcome = _episode_outcome(episode)
    progress = episode.progress_fraction
    progress_text = "" if progress is None else f'<span class="sub">progress {progress * 100:.0f}%</span>'
    policy_text = "" if not policy else f'<span class="who">policy <strong>{html.escape(policy)}</strong></span>'

    slots = []
    for camera in cameras:
        source = episode.video_by_camera.get(camera)
        if source is None:
            body = '<div class="placeholder">not recorded</div>'
        else:
            body = f'<div class="placeholder" data-video-src="{_media_src(video_prefix, source)}">video</div>'
        slots.append(f'<div class="videoslot"><div class="camera">{html.escape(camera)}</div>{body}</div>')

    signals_html = _render_signals(objectives or [])
    if episode.outcome_disagrees_with_progress:
        reached = "every objective completed" if episode.all_objectives_complete else "objectives are incomplete"
        verdict = "succeeded" if episode.success else "did not succeed"
        signals_html += (
            f'<p class="disagree">&#9888; {reached}, but the task\'s success term says this episode {verdict}.</p>'
        )
    metadata = "".join(_render_metadata_entry(key, value) for key, value in episode.metadata.items())
    metadata_html = f'<div class="meta">{metadata}</div>' if metadata else ""
    return (
        f'<article class="episode" id="{episode_anchor(episode)}" data-outcome="{outcome}">'
        f'<div class="episode-head"><span class="id">env {episode.env_index} &middot; '
        f"episode {episode.episode_index}</span>"
        f'<span class="badge {outcome}">{html.escape(_OUTCOME_LABELS[outcome])}</span>'
        f"{progress_text}{policy_text}</div>"
        f'<div class="videos">{"".join(slots)}</div>{signals_html}{metadata_html}</article>'
    )


def render_job_page(
    summary: ExperimentSummary,
    job: JobSummary,
    task_href: str,
    index_href: str,
    video_prefix: str,
    episodes: list,
    page_index: int,
    num_pages: int,
    page_hrefs: list[str],
) -> str:
    """Render one Run page holding one page of episodes."""
    tiles = [
        _tile("Episodes", f"{job.num_episodes:,}"),
        _tile("Success rate", _percent(job.success_rate), f"{job.num_successes:,} successes"),
        _tile("Mean progress", _percent(job.mean_progress)),
    ]
    controls = (
        '<div class="controls"><span class="note">Show</span>'
        '<button data-filter="all" aria-pressed="true">all</button>'
        '<button data-filter="success" aria-pressed="false">successes</button>'
        '<button data-filter="partial" aria-pressed="false">partial</button>'
        '<button data-filter="fail" aria-pressed="false">no progress</button>'
        '<button data-filter="unknown" aria-pressed="false">not scored</button></div>'
    )
    cards = "".join(
        _render_episode_card(
            episode, job.cameras, video_prefix, policy=job.policy, objectives=job.objectives_for(episode)
        )
        for episode in episodes
    )
    issues = _render_issues(job.issues)

    breadcrumb = (
        f'<a href="{_href(index_href)}">{html.escape(summary.title)}</a><span class="sep">/</span>'
        f'<a href="{_href(task_href)}">{html.escape(job.task)}</a>'
        f'<span class="sep">/</span><span class="current">{html.escape(job.policy or job.name)}</span>'
        + _render_up_button(task_href, f"Back to {job.task}", extra_class="up")
    )
    context = [_render_pill("task", job.task)]
    if job.policy:
        context.append(_render_pill("policy", job.policy, extra_class="policy"))
    context.append(_render_pill("run", job.name or "results"))
    if num_pages > 1:
        context.append(_render_pill("page", f"{page_index + 1} of {num_pages}"))

    footer = _render_footer_nav([
        _render_page_nav(page_index, page_hrefs),
        _render_up_button(task_href, f"Back to {job.task}"),
        _render_up_button(index_href, "Back to the overview"),
    ])
    pager = _render_page_nav(page_index, page_hrefs)
    content = f'<div class="tiles">{"".join(tiles)}</div>{issues}{pager}{controls}{cards}{footer}'

    return _render_page(
        title=f"{html.escape(job.name)} &mdash; {html.escape(summary.title)}",
        heading=html.escape(job.policy or job.name or "results"),
        breadcrumb=breadcrumb,
        summary=f"{job.num_episodes} episode(s) &middot; {job.num_videos} video(s)",
        content=content,
        context=_render_context(context),
    )


def _render_page_nav(page_index: int, page_hrefs: list[str]) -> str:
    if len(page_hrefs) <= 1:
        return ""
    links = []
    if page_index > 0:
        links.append(f'<a class="upbutton" href="{_href(page_hrefs[page_index - 1])}">Previous</a>')
    links.append(f'<span class="note">Page {page_index + 1} of {len(page_hrefs)}</span>')
    if page_index + 1 < len(page_hrefs):
        links.append(f'<a class="upbutton" href="{_href(page_hrefs[page_index + 1])}">Next</a>')
    return f'<nav class="pagenav">{"".join(links)}</nav>'


def _render_page(title: str, heading: str, breadcrumb: str, summary: str, content: str, context: str = "") -> str:
    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        title=title, heading=heading, breadcrumb=breadcrumb, summary=summary, content=content, context=context
    )


def _render_pill(key: str, value: str, extra_class: str = "") -> str:
    classes = f"pill {extra_class}".strip()
    return (
        f'<span class="{classes}"><span class="key">{html.escape(key)}</span>'
        f'<span class="value">{html.escape(value)}</span></span>'
    )


def _render_context(pills: list[str]) -> str:
    return f'<div class="context">{"".join(pills)}</div>' if pills else ""


def _render_up_button(href: str, label: str, extra_class: str = "") -> str:
    wrapper_open = f'<span class="{extra_class}">' if extra_class else ""
    wrapper_close = "</span>" if extra_class else ""
    return f'{wrapper_open}<a class="upbutton" href="{_href(href)}">&uarr; {html.escape(label)}</a>{wrapper_close}'


def _render_footer_nav(buttons: list[str]) -> str:
    buttons = [button for button in buttons if button]
    return f'<div class="footernav">{"".join(buttons)}</div>' if buttons else ""


def _href(path: str) -> str:
    return quote(path, safe="/#._-~")


def _media_src(prefix: str, source: str) -> str:
    return quote(prefix + source, safe="/._-~")
