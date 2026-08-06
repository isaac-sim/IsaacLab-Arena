# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render an aggregated Experiment into the report's three levels of HTML pages.

The overview holds no video at all, and a run page emits video slots that mount only once scrolled
into view, so opening any single page stays cheap however large the Experiment is.
"""

from __future__ import annotations

import html
import pathlib
import re
import string

from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.visualization.report_data import ExperimentSummary, JobSummary, RunExecutionReport, TaskSummary

_TEMPLATE_PATH = pathlib.Path(__file__).parent / "report_template.html"

# Sub-directory holding the task and run pages, keeping them out of the results directory itself.
PAGES_DIRNAME = "report"

# Number of steps in the sequential ramp the overview's success-rate cells are bucketed into.
_NUM_RAMP_STEPS = 7

# Highest ordinal step defined for funnel bars; deeper stages reuse the darkest step.
_MAX_FUNNEL_STAGE_STEP = 2

_UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")

_OUTCOME_GLYPHS = {"success": "&check;", "partial": "&#9680;", "fail": "&times;", "unknown": "&middot;"}
_OUTCOME_LABELS = {"success": "success", "partial": "partial", "fail": "no progress", "unknown": "not scored"}


def unique_slugs(names: list[str]) -> dict[str, str]:
    """Return a filesystem-safe, collision-free slug for each name.

    Args:
        names: Names to slugify, such as task or Run names.
    """
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


def _percent(fraction: float | None) -> str:
    """Format a fraction as a whole-number percentage, or an em dash when unknown."""
    return "&mdash;" if fraction is None else f"{fraction * 100:.0f}%"


def _ramp_rank(fraction: float) -> int:
    """Return the sequential-ramp step index for a fraction in [0, 1]."""
    return min(_NUM_RAMP_STEPS - 1, max(0, int(fraction * _NUM_RAMP_STEPS)))


def _episode_outcome(episode) -> str:
    """Return the status bucket an episode falls into: success, partial, fail, or unknown."""
    if episode.success is True:
        return "success"
    if episode.success is None:
        return "unknown"
    progress = episode.progress_fraction
    return "partial" if progress is not None and progress > 0 else "fail"


def _tile(label: str, value: str, sub: str = "") -> str:
    """Render one headline figure, which needs no plot."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def _render_failed_runs_section(run_executions: list[RunExecutionReport]) -> str:
    """Render a compact table of Runs whose processes failed."""
    failed = [execution for execution in run_executions if execution.status is RunStatus.FAILED]
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


def _render_ramp_legend() -> str:
    """Render the sequential ramp's key, so the heatmap's direction is stated rather than guessed."""
    swatches = "".join(f'<span class="swatch cell" data-rank="{rank}"></span>' for rank in range(_NUM_RAMP_STEPS))
    return f'<div class="ramp-legend"><span>0%</span>{swatches}<span>100%</span><span>success rate</span></div>'


def _render_matrix(summary: ExperimentSummary, task_hrefs: dict[str, str]) -> str:
    """Render the task x policy success-rate heatmap that opens the report."""
    header_cells = "".join(
        f'<th class="num" data-sort="{html.escape(policy)}">{html.escape(policy or "run")}</th>'
        for policy in summary.policies
    )
    rows = []
    for task in summary.tasks:
        cells = [
            f'<th data-key="task" data-value="{html.escape(task.name)}">'
            f'<a href="{html.escape(task_hrefs[task.name])}">{html.escape(task.name)}</a></th>'
        ]
        for policy in summary.policies:
            job = task.job_for_policy(policy)
            rate = job.success_rate if job is not None else None
            if job is None or rate is None:
                cells.append(f'<td class="cell missing" data-key="{html.escape(policy)}" data-value="">&mdash;</td>')
                continue
            cells.append(
                f'<td class="cell" data-rank="{_ramp_rank(rate)}" data-key="{html.escape(policy)}"'
                f' data-value="{rate:.6f}">'
                f'<a href="{html.escape(task_hrefs[task.name])}"'
                f' title="{html.escape(job.name)}: {job.num_successes}/{job.num_scored_episodes} episodes">'
                f"{_percent(rate)}</a></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f"<section><h2>Success rate by task and policy</h2>{_render_ramp_legend()}"
        '<table class="matrix"><thead><tr>'
        '<th data-sort="task">task</th>'
        f"{header_cells}</tr></thead><tbody>\n"
        + "\n".join(rows)
        + "</tbody></table>"
        '<p class="note">Click a task to see where its episodes failed. Click a column heading to sort.</p>'
        "</section>"
    )


def _render_ungrouped_job_list(summary: ExperimentSummary, job_hrefs: dict[str, str]) -> str:
    """Render a flat list of Runs, used when no task and policy labels could be established."""
    rows = "\n".join(
        f'<tr><th><a href="{html.escape(job_hrefs[job.name])}">{html.escape(job.name or "results")}</a></th>'
        f'<td class="num">{job.num_episodes}</td>'
        f'<td class="num">{_percent(job.success_rate)}</td></tr>'
        for job in summary.jobs
    )
    return (
        "<section><h2>Runs</h2>"
        "<table><thead><tr><th>run</th><th>episodes</th><th>success rate</th></tr></thead>"
        f"<tbody>\n{rows}\n</tbody></table></section>"
    )


def render_index(summary: ExperimentSummary, task_hrefs: dict[str, str], job_hrefs: dict[str, str]) -> str:
    """Render the overview page: headline figures and the task x policy heatmap.

    Args:
        summary: Aggregated Experiment to render.
        task_hrefs: Task name -> link to its page, relative to the report root.
        job_hrefs: Run name -> link to its page, relative to the report root.
    """
    tiles = [
        _tile("Tasks", str(len(summary.tasks))),
        _tile("Runs", str(len(summary.jobs))),
        _tile("Episodes", f"{summary.num_episodes:,}"),
        _tile("Success rate", _percent(summary.overall_success_rate)),
    ]
    if summary.is_grouped:
        tiles.insert(1, _tile("Policies", str(len(summary.policies))))
    for policy in summary.policies:
        if policy:
            tiles.append(
                _tile(
                    policy,
                    _percent(summary.success_rate_for_policy(policy)),
                    f"{summary.num_episodes_for_policy(policy):,} episodes",
                )
            )

    body = _render_matrix(summary, task_hrefs) if summary.is_grouped else _render_ungrouped_job_list(summary, job_hrefs)
    content = f'<div class="tiles">{"".join(tiles)}</div>{_render_failed_runs_section(summary.run_executions)}{body}'
    if not summary.tasks and not summary.run_executions:
        content += "<p>No results recorded yet.</p>"
    content += f'<p class="note">{_grouping_note(summary)}</p>'

    return _render_page(
        title=html.escape(summary.title),
        heading=html.escape(summary.title),
        # The sticky bar keeps the experiment named while the matrix scrolls past.
        breadcrumb=f'<span class="current">{html.escape(summary.title)}</span>',
        summary=_experiment_summary_line(summary),
        content=content,
    )


def _grouping_note(summary: ExperimentSummary) -> str:
    """State where the task and policy labels came from, since it bounds how much they can be trusted."""
    if summary.grouping_source == "run_names":
        return (
            "Tasks and policies were inferred by factorizing the run names, because a run records no"
            " task or policy of its own."
        )
    return "Runs could not be grouped into tasks and policies, so they are listed individually."


def _experiment_summary_line(summary: ExperimentSummary) -> str:
    """Render the one-line count summary under the page heading."""
    if summary.run_executions:
        completed = sum(execution.status is RunStatus.COMPLETED for execution in summary.run_executions)
        failed = sum(execution.status is RunStatus.FAILED for execution in summary.run_executions)
        return (
            f"{len(summary.run_executions)} run(s) &middot; {completed} completed &middot; "
            f"{failed} failed &middot; {summary.num_episodes} episode(s)"
        )
    return (
        f"{len(summary.jobs)} run(s) &middot; {summary.num_episodes} episode(s) &middot; {summary.num_videos} video(s)"
    )


def _render_funnel(job: JobSummary) -> str:
    """Render one Run's progress funnel as ordinal stage bars.

    The bars count (episode, objective) pairs rather than episodes, because a multi-object task
    declares one objective per object and fires each predicate once per object.
    """
    stages = job.funnel
    if not stages:
        return ""
    total = job.num_objective_instances
    rows = []
    for stage in stages:
        fraction = 0.0 if total == 0 else stage.num_reached / total
        step = min(stage.index, _MAX_FUNNEL_STAGE_STEP)
        rows.append(
            '<div class="funnel-row">'
            f'<div class="stage-label"><span class="name">{html.escape(stage.name)}</span>'
            f'<span class="value">{stage.num_reached:,} &middot; {_percent(fraction)}</span></div>'
            f'<div class="bar-track"><div class="bar" data-stage="{step}"'
            f' style="width: {fraction * 100:.1f}%"></div></div></div>'
        )
    unit = "objective instances" if total != job.num_episodes else "episodes"
    return (
        f'<div class="funnel"><h3>{html.escape(job.policy or job.name)}</h3>'
        + "".join(rows)
        + f'<p class="note">{total:,} {unit} &middot; success {_percent(job.success_rate)}'
        f" &middot; mean progress {_percent(job.mean_progress)}</p></div>"
    )


def _render_chip(episode, href: str) -> str:
    """Render one episode as a status chip carrying a glyph, so state is never colour alone."""
    outcome = _episode_outcome(episode)
    progress = episode.progress_fraction
    progress_text = "" if progress is None else f", progress {progress * 100:.0f}%"
    tooltip = f"env {episode.env_index} episode {episode.episode_index}: {_OUTCOME_LABELS[outcome]}{progress_text}"
    return (
        f'<a class="chip {outcome}" href="{html.escape(href)}" title="{html.escape(tooltip)}">'
        f"{_OUTCOME_GLYPHS[outcome]}</a>"
    )


def _render_legend() -> str:
    """Render the status legend for the episode chips."""
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
    index_href: str,
) -> str:
    """Render one task's page: each policy's funnel and its grid of episode outcomes.

    Args:
        summary: Aggregated Experiment the task belongs to.
        task: Task to render.
        job_hrefs: Run name -> link to its page, relative to this page.
        index_href: Link back to the overview, relative to this page.
    """
    tiles = [_tile("Episodes", f"{task.num_episodes:,}")]
    for job in task.jobs:
        tiles.append(_tile(job.policy or job.name, _percent(job.success_rate), f"{job.num_successes:,} successes"))

    funnels = "".join(_render_funnel(job) for job in task.jobs)
    funnel_section = (
        f'<section><h2>Where episodes got to</h2><div class="funnels">{funnels}</div></section>' if funnels else ""
    )

    chip_sections = []
    for job in task.jobs:
        chips = "".join(
            _render_chip(episode, f"{job_hrefs[job.name]}#ep-{episode.env_index}-{episode.episode_index}")
            for episode in job.episodes
        )
        chip_sections.append(
            f"<section><h2>{html.escape(job.policy or job.name)} episodes</h2>"
            f'<p class="note"><a href="{html.escape(job_hrefs[job.name])}">Open '
            f"{html.escape(job.name)} to watch the videos</a></p>"
            f'{_render_legend()}<div class="chips">{chips}</div></section>'
        )

    breadcrumb = (
        f'<a href="{html.escape(index_href)}">{html.escape(summary.title)}</a>'
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
    """Render one metadata field as a labelled block."""
    if isinstance(value, dict):
        sub_rows = "".join(
            f'<div class="subitem"><span class="k">{html.escape(str(sub_key))}</span>'
            f" {html.escape(str(sub_value))}</div>"
            for sub_key, sub_value in value.items()
        )
        return f'<div><span class="k">{html.escape(key)}</span>{sub_rows}</div>'
    return f'<div><span class="k">{html.escape(key)}</span> {html.escape(str(value))}</div>'


def _render_signal(signal) -> str:
    """Render one success predicate as a chip stating whether it fired, and when."""
    if signal.triggered:
        state, glyph = "on", "&check;"
        suffix = "" if signal.step is None else f'<span class="step">step {signal.step}</span>'
    elif signal.blocked:
        state, glyph = "blocked", "&#9654;"
        suffix = '<span class="step">waiting</span>'
    else:
        state, glyph = "off", "&#9675;"
        suffix = ""
    # The full predicate text names the object on multi-object tasks, which the bare name does not.
    tooltip = signal.detail or signal.name
    return (
        f'<span class="signal {state}" title="{html.escape(tooltip)}">'
        f'<span class="glyph">{glyph}</span>{html.escape(signal.name)}{suffix}</span>'
    )


def _render_objective(objective) -> str:
    """Render one objective as its score plus the full sequence of its success predicates."""
    track = "".join(_render_signal(signal) for signal in objective.signals)
    score = f"{round(objective.score, 2):g} / {round(objective.max_score, 2):g}"
    return (
        '<div class="objective"><div class="objective-head">'
        f'<span class="name">{html.escape(objective.name)}</span>'
        f'<span class="score">{html.escape(score)}</span></div>'
        f'<div class="track">{track}</div></div>'
    )


def _render_signals(objectives: list) -> str:
    """Render an episode's progress breakdown, collapsing it when a task has many objectives.

    A single-objective task is a short row and is always shown; a multi-object task declares an
    objective per object, so its breakdown is folded behind a summary that states the totals.
    """
    if not objectives:
        return ""
    if len(objectives) == 1:
        return f'<div class="signals">{_render_objective(objectives[0])}</div>'

    num_triggered = sum(objective.num_triggered for objective in objectives)
    num_signals = sum(len(objective.signals) for objective in objectives)
    num_complete = sum(1 for objective in objectives if objective.is_complete)
    body = "".join(_render_objective(objective) for objective in objectives)
    return (
        f'<details class="signals"><summary>{num_triggered} of {num_signals} signals triggered '
        f"across {len(objectives)} objectives &middot; {num_complete} complete</summary>{body}</details>"
    )


def _render_episode_card(episode, cameras: list[str], video_prefix: str, policy: str = "", objectives=None) -> str:
    """Render one episode: its outcome, its videos as lazily mounted slots, and its metadata.

    Args:
        episode: Episode to render.
        cameras: Camera names to lay out a slot for, in order.
        video_prefix: Prefix turning a root-relative video path into one relative to the page.
        policy: Policy label repeated on the card, so it stays visible deep into a long run page.
        objectives: The episode's objectives and their per-predicate signals, when available.
    """
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
            body = f'<div class="placeholder" data-video-src="{html.escape(video_prefix + source)}">video</div>'
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
        f'<article class="episode" id="ep-{episode.env_index}-{episode.episode_index}" data-outcome="{outcome}">'
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
) -> str:
    """Render one Run's page: every episode, with videos that mount when scrolled into view.

    Args:
        summary: Aggregated Experiment the Run belongs to.
        job: Run to render.
        task_href: Link to the Run's task page, relative to this page.
        index_href: Link back to the overview, relative to this page.
        video_prefix: Prefix turning a root-relative video path into one relative to this page.
    """
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
        '<button data-filter="fail" aria-pressed="false">no progress</button></div>'
    )
    cards = "".join(
        _render_episode_card(
            episode, job.cameras, video_prefix, policy=job.policy, objectives=job.objectives_for(episode)
        )
        for episode in job.episodes
    )

    # The policy is named in the sticky bar, in a pill under the heading, and again on every episode
    # card, so it stays answerable however far into the episodes the reader has scrolled.
    breadcrumb = (
        f'<a href="{html.escape(index_href)}">{html.escape(summary.title)}</a><span class="sep">/</span>'
        f'<a href="{html.escape(task_href)}">{html.escape(job.task)}</a>'
        f'<span class="sep">/</span><span class="current">{html.escape(job.policy or job.name)}</span>'
        + _render_up_button(task_href, f"Back to {job.task}", extra_class="up")
    )
    context = [_render_pill("task", job.task)]
    if job.policy:
        context.append(_render_pill("policy", job.policy, extra_class="policy"))
    context.append(_render_pill("run", job.name or "results"))

    # Repeated at the end because this page is hundreds of episodes long: the sticky button stays
    # reachable while scrolling, and this one lands where the reader runs out of episodes.
    footer = _render_footer_nav([
        _render_up_button(task_href, f"Back to {job.task}"),
        _render_up_button(index_href, "Back to the overview"),
    ])

    return _render_page(
        title=f"{html.escape(job.name)} &mdash; {html.escape(summary.title)}",
        heading=html.escape(job.policy or job.name or "results"),
        breadcrumb=breadcrumb,
        summary=f"{job.num_episodes} episode(s) &middot; {job.num_videos} video(s)",
        content=f'<div class="tiles">{"".join(tiles)}</div>{controls}{cards}{footer}',
        context=_render_context(context),
    )


def _render_page(title: str, heading: str, breadcrumb: str, summary: str, content: str, context: str = "") -> str:
    """Fill the shared page shell.

    Args:
        title: Document title.
        heading: Page heading.
        breadcrumb: Links back up the hierarchy, rendered into the sticky bar.
        summary: One-line count summary under the heading.
        content: The page body.
        context: Pills naming what the page is showing, such as its task and policy.
    """
    template = string.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        title=title, heading=heading, breadcrumb=breadcrumb, summary=summary, content=content, context=context
    )


def _render_pill(key: str, value: str, extra_class: str = "") -> str:
    """Render one labelled pill naming a dimension of what the page is showing."""
    classes = f"pill {extra_class}".strip()
    return (
        f'<span class="{classes}"><span class="key">{html.escape(key)}</span>'
        f'<span class="value">{html.escape(value)}</span></span>'
    )


def _render_context(pills: list[str]) -> str:
    """Wrap pills into the context row shown under a page heading."""
    return f'<div class="context">{"".join(pills)}</div>' if pills else ""


def _render_up_button(href: str, label: str, extra_class: str = "") -> str:
    """Render a button that climbs one level of the hierarchy.

    Args:
        href: Link target, relative to the page the button is rendered on.
        label: Text naming where the button goes.
        extra_class: Extra class placed on the wrapping element, such as ``up`` for the sticky bar.
    """
    wrapper_open = f'<span class="{extra_class}">' if extra_class else ""
    wrapper_close = "</span>" if extra_class else ""
    return (
        f'{wrapper_open}<a class="upbutton" href="{html.escape(href)}">'
        f'<span class="arrow">&uarr;</span>{html.escape(label)}</a>{wrapper_close}'
    )


def _render_footer_nav(buttons: list[str]) -> str:
    """Render the navigation block closing a page, reachable without scrolling back to the top."""
    return f'<div class="footernav">{"".join(buttons)}</div>' if buttons else ""
