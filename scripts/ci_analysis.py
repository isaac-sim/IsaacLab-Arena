#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
CI Analysis Script for IsaacLab-Arena GitHub Actions.

Fetches workflow run and job data via the GitHub API and reports:
  - Per-job queue time (job created_at -> started_at)
  - Per-job execution duration (started_at -> completed_at)
  - Per-run total queue time (sum of queue times for all PR-gating jobs)
  - Failure rate breakdown: genuine vs. infrastructure failures

Usage:
    # Full year-to-date analysis with plots (recommended for regular runs)
    python3 scripts/ci_analysis.py --plot

    # Custom date range
    python3 scripts/ci_analysis.py --since 2026-03-01 --plot

    # Text report only, no plots
    python3 scripts/ci_analysis.py

Requires:
  - `gh` CLI authenticated (gh auth login), or GITHUB_TOKEN env var
  - matplotlib (pip install matplotlib) for --plot

Output:
  - Console summary table
  - ci_analysis_results.json     raw data for further analysis
  - ci_trends_queue_time.png     weekly queue time trend (with --plot)
  - ci_trends_duration.png       weekly duration trend (with --plot)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median, stdev

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = "isaac-sim/IsaacLab-Arena"

# Workflow IDs (from: gh api repos/{repo}/actions/workflows).
# ci_new.yml (id=238099976) was a short-lived rename of ci.yml in Feb 2026
# and no longer exists in the repo; excluded here to avoid double-counting runs.
WORKFLOW_IDS = {
    "ci.yml": 184771057,
}

# Jobs shown in plots — maps API job name -> short display label.
# Jobs NOT listed here are still counted in totals but not plotted individually.
PLOT_JOBS = {
    "Pre-commit": "Pre-commit",
    "Run tests": "Run tests",
    "Run policy-related tests with GR00T & cuda12_8 deps": "Policy tests (GR00T)",
    "Build the docs (pre-merge)": "Build docs",
    "Build & push NGC image (post-merge)": "Build & push image",
}

# Jobs that gate every PR (used to compute per-run total queue time).
# Build & push only runs post-merge, so excluded from the PR total.
PR_GATING_JOBS = {
    "Pre-commit",
    "Run tests",
    "Run policy-related tests with GR00T & cuda12_8 deps",
    "Build the docs (pre-merge)",
}

# Step-name / job-name patterns that indicate infrastructure failures.
# The API does not expose log text, so we match against step names only.
INFRA_PATTERNS = [
    # NGC / Docker image pull issues
    r"pull.*image",
    r"image.*pull",
    r"unauthorized.*registry",
    r"authentication.*required",
    r"manifest.*unknown",
    r"toomanyrequests",
    r"nvcr\.io",
    r"ngc.*api.*key",
    # GPU / runner issues
    r"nvidia-smi",
    r"no.*gpu",
    r"cuda.*error",
    r"runner.*lost",
    r"runner.*offline",
    r"lost.*communication",
    r"set up job",
    # Container / runner lifecycle
    r"initialize containers",
    r"complete runner",
    r"set up runner",
    # Transient network
    r"connection.*reset",
    r"connection.*refused",
    r"timed out",
    r"rate limit",
    r"502",
    r"503",
    r"504",
]

_INFRA_RE = re.compile("|".join(INFRA_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def get_token() -> str:
    """Return a GitHub API token from env or gh CLI config."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    # Try gh >= 2.x
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    # Fallback: read directly from gh hosts.yml (older gh versions)
    hosts_path = os.path.expanduser("~/.config/gh/hosts.yml")
    try:
        import yaml

        with open(hosts_path) as fh:
            data = yaml.safe_load(fh)
        return data["github.com"]["oauth_token"]
    except Exception:
        pass
    sys.exit("ERROR: Could not obtain GitHub token. Set GITHUB_TOKEN env var or run `gh auth login`.")
    return ""  # unreachable; satisfies R503


def gh_get(path: str, token: str, params: dict | None = None, retries: int = 3) -> dict | list:
    """Call the GitHub REST API and return parsed JSON, with retries."""
    url = f"https://api.github.com{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "ci-analysis-script")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise exc


def fetch_runs_since(wf_id: int, since_dt: datetime, token: str) -> list[dict]:
    """Fetch all workflow runs created on or after since_dt (paginated)."""
    runs = []
    page = 1
    while True:
        data = gh_get(
            f"/repos/{REPO}/actions/runs",
            token,
            {"workflow_id": wf_id, "per_page": 100, "page": page},
        )
        batch = data.get("workflow_runs", [])
        if not batch:
            break
        for run in batch:
            created = parse_dt(run.get("created_at"))
            if created and created < since_dt:
                return runs  # runs are newest-first; stop when past the cutoff
            runs.append(run)
        if len(batch) < 100:
            break
        page += 1
    return runs


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def to_minutes(td) -> float | None:
    if td is None:
        return None
    v = td.total_seconds() / 60
    return v if 0 <= v < 300 else None  # sanity cap: discard anything >= 5 h


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def classify_failure(jobs: list[dict]) -> str:
    """
    Classify a failed run as 'infra', 'genuine', or 'unknown'.
    Matched against step names (API does not expose log text).
    """
    failed_jobs = [j for j in jobs if j.get("conclusion") in ("failure", "timed_out")]
    if not failed_jobs:
        return "unknown"

    for job in failed_jobs:
        job_name = job.get("name", "")
        failed_steps = [s for s in job.get("steps", []) if s.get("conclusion") in ("failure", "timed_out")]

        if job.get("conclusion") == "timed_out":
            return "infra"

        for step in failed_steps:
            if _INFRA_RE.search(f"{job_name} {step.get('name', '')}"):
                return "infra"

        # Only setup/teardown steps failed — runner problem
        if failed_steps:
            non_setup = [
                s
                for s in failed_steps
                if not re.match(r"^(set up job|post |complete job)", s.get("name", ""), re.IGNORECASE)
            ]
            if not non_setup:
                return "infra"

    return "genuine"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze(workflow_ids: dict[str, int], since_dt: datetime, token: str) -> dict:
    all_runs: list[dict] = []

    for wf_name, wf_id in workflow_ids.items():
        print(f"  Fetching runs for {wf_name} (id={wf_id}) since {since_dt.date()}...", flush=True)
        runs = fetch_runs_since(wf_id, since_dt, token)
        for r in runs:
            r["_workflow_name"] = wf_name
        all_runs.extend(runs)
        print(f"    -> {len(runs)} runs", flush=True)

    print(f"\nTotal runs: {len(all_runs)}", flush=True)

    # Aggregate stats (all-time over the window)
    job_queue_times: dict[str, list[float]] = defaultdict(list)
    job_durations: dict[str, list[float]] = defaultdict(list)
    run_total_times: list[float] = []

    # Per-job records for trend plots:
    #   {job, week, run_id, queue_m, duration_m}
    job_records: list[dict] = []

    # Per-run totals (sum of PR-gating job queue/duration times):
    #   {week, total_queue_m}  /  {week, total_duration_m}
    run_queue_records: list[dict] = []
    run_duration_records: list[dict] = []

    # Per-week failure counts for trend plot: {week: {"infra": n, "genuine": n, "total": n}}
    week_failure_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"infra": 0, "genuine": 0, "total": 0})

    # Failure tracking
    failure_counts = {"infra": 0, "genuine": 0, "unknown": 0}
    failed_runs_detail: list[dict] = []
    total_completed = 0
    total_failed = 0
    status_counts: dict[str, int] = defaultdict(int)

    print("Fetching job details...", flush=True)
    for i, run in enumerate(all_runs):
        run_id = run["id"]
        conclusion = run.get("conclusion")
        status = run.get("status")
        status_counts[f"{status}/{conclusion}"] += 1

        try:
            jobs_data = gh_get(f"/repos/{REPO}/actions/runs/{run_id}/jobs", token, {"per_page": 100})
            jobs = jobs_data.get("jobs", [])
        except Exception as exc:
            print(f"  [WARN] run {run_id}: {exc}", flush=True)
            jobs = []

        # Determine the week from the run's created_at
        run_created = parse_dt(run.get("created_at"))
        if run_created:
            iso_year, iso_week, _ = run_created.isocalendar()
            run_week = f"{iso_year}-W{iso_week:02d}"
        else:
            run_week = None

        # Per-job metrics
        run_pr_queue_total = 0.0  # sum of queue minutes for PR-gating jobs this run
        run_pr_job_count = 0
        pr_gating_succeeded: set[str] = set()  # PR-gating jobs that completed successfully

        for job in jobs:
            created = parse_dt(job.get("created_at"))
            started = parse_dt(job.get("started_at"))
            completed = parse_dt(job.get("completed_at"))
            name = job.get("name", "unknown")
            job_conclusion = job.get("conclusion")

            # Queue time: count for all jobs that actually started (the wait was real
            # regardless of how the job ended).
            q_m = to_minutes(started - created) if (created and started) else None

            # Duration: only count jobs that completed successfully.  Cancelled or
            # failed jobs ran for less than their full duration and would bias the
            # median downward, masking real slowdowns.
            d_m = to_minutes(completed - started) if (started and completed and job_conclusion == "success") else None

            if q_m is not None:
                job_queue_times[name].append(q_m)
            if d_m is not None:
                job_durations[name].append(d_m)

            if run_week and (q_m is not None or d_m is not None):
                job_records.append({
                    "job": name,
                    "week": run_week,
                    "run_id": run_id,
                    "queue_m": q_m,
                    "duration_m": d_m,
                })

            if name in PR_GATING_JOBS:
                if q_m is not None:
                    run_pr_queue_total += q_m
                if q_m is not None or d_m is not None:
                    run_pr_job_count += 1
                if job_conclusion == "success":
                    pr_gating_succeeded.add(name)

        if run_week and run_pr_job_count > 0:
            run_queue_records.append({"week": run_week, "total_queue_m": run_pr_queue_total})

        # Run-level wall-clock time (created_at -> updated_at).
        # Used as "total duration" — only recorded when ALL PR-gating jobs
        # completed successfully, so we compare apples to apples.
        run_updated = parse_dt(run.get("updated_at"))
        if run_created and run_updated:
            t = to_minutes(run_updated - run_created)
            if t is not None:
                run_total_times.append(t)
                if run_week and pr_gating_succeeded == PR_GATING_JOBS:
                    run_duration_records.append({"week": run_week, "total_duration_m": t})

        # Failure classification
        if status == "completed":
            total_completed += 1
            if conclusion == "failure":
                total_failed += 1
                cat = classify_failure(jobs)
                failure_counts[cat] += 1
                if run_week:
                    week_failure_counts[run_week]["total"] += 1
                    if cat == "infra":
                        week_failure_counts[run_week]["infra"] += 1
                failed_runs_detail.append({
                    "run_id": run_id,
                    "run_number": run.get("run_number"),
                    "created_at": run.get("created_at"),
                    "head_branch": run.get("head_branch"),
                    "head_sha": run.get("head_sha", "")[:8],
                    "workflow": run.get("_workflow_name"),
                    "category": cat,
                    "failed_jobs": [
                        {
                            "job": j.get("name"),
                            "conclusion": j.get("conclusion"),
                            "failed_steps": [
                                s.get("name")
                                for s in j.get("steps", [])
                                if s.get("conclusion") in ("failure", "timed_out")
                            ],
                        }
                        for j in jobs
                        if j.get("conclusion") in ("failure", "timed_out")
                    ],
                })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_runs)} runs...", flush=True)

    return {
        "meta": {
            "repo": REPO,
            "since": since_dt.isoformat(),
            "num_runs_fetched": len(all_runs),
            "workflows_analyzed": list(workflow_ids.keys()),
        },
        "job_queue_times": dict(job_queue_times),
        "job_durations": dict(job_durations),
        "run_total_times": run_total_times,
        "job_records": job_records,
        "run_queue_records": run_queue_records,
        "run_duration_records": run_duration_records,
        "week_failure_counts": {k: dict(v) for k, v in week_failure_counts.items()},
        "status_counts": dict(status_counts),
        "total_completed": total_completed,
        "total_failed": total_failed,
        "failure_counts": failure_counts,
        "failed_runs_detail": failed_runs_detail,
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def fmt_stats(values: list[float]) -> str:
    if not values:
        return "N/A (no data)"
    sv = sorted(values)
    p90 = sv[int(len(sv) * 0.9)]
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"mean={mean(values):.1f}m  median={median(values):.1f}m  p90={p90:.1f}m  stdev={sd:.1f}m  n={len(values)}"


def print_report(data: dict) -> None:
    print("\n" + "=" * 70)
    print("CI ANALYSIS REPORT")
    print(f"Repo:      {data['meta']['repo']}")
    print(f"Since:     {data['meta']['since'][:10]}")
    print(f"Runs:      {data['meta']['num_runs_fetched']}")
    print(f"Workflows: {', '.join(data['meta']['workflows_analyzed'])}")
    print("=" * 70)

    print("\n--- Per-job queue time (job created_at -> started_at) ---")
    for name in sorted(data["job_queue_times"]):
        print(f"  [{name}]")
        print("    " + fmt_stats(data["job_queue_times"][name]))

    print("\n--- Per-job duration (started_at -> completed_at) ---")
    for name in sorted(data["job_durations"]):
        print(f"  [{name}]")
        print("    " + fmt_stats(data["job_durations"][name]))

    rqt = [r["total_queue_m"] for r in data["run_queue_records"]]
    print("\n--- Per-run total queue time (sum of PR-gating job queue times) ---")
    print("  " + fmt_stats(rqt))

    print("\n--- Run wall-clock time (created_at -> updated_at) ---")
    print("  " + fmt_stats(data["run_total_times"]))

    print("\n--- Run status breakdown ---")
    for status, count in sorted(data["status_counts"].items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    tc = data["total_completed"]
    tf = data["total_failed"]
    fc = data["failure_counts"]
    print(f"\n--- Failure analysis ({tc} completed runs) ---")
    if tc:
        print(f"  Total failed:   {tf} / {tc}  ({100*tf/tc:.1f}%)")
        if tf:
            print(f"  Infrastructure: {fc['infra']} ({100*fc['infra']/tf:.1f}% of failures)")
            print(f"  Genuine:        {fc['genuine']} ({100*fc['genuine']/tf:.1f}% of failures)")
            print(f"  Unknown:        {fc['unknown']} ({100*fc['unknown']/tf:.1f}% of failures)")

    if data["failed_runs_detail"]:
        print("\n--- Recent failed runs (up to 20) ---")
        for r in data["failed_runs_detail"][:20]:
            print(
                f"  #{r['run_number']}  {r['created_at'][:10]}  "
                f"branch={r['head_branch']}  sha={r['head_sha']}  [{r['category'].upper()}]"
            )
            for j in r["failed_jobs"]:
                steps = ", ".join(j["failed_steps"]) or "(none recorded)"
                print(f"    job={j['job']}  conclusion={j['conclusion']}  failing steps: {steps}")

    # ---- Weekly stats table ----
    records = data.get("job_records", [])
    run_queue_records = data.get("run_queue_records", [])
    run_duration_records = data.get("run_duration_records", [])
    if records:
        from collections import defaultdict as _dd

        wq: dict[str, dict[str, list[float]]] = _dd(lambda: _dd(list))
        wd: dict[str, dict[str, list[float]]] = _dd(lambda: _dd(list))
        for r in records:
            if r.get("queue_m") is not None:
                wq[r["job"]][r["week"]].append(r["queue_m"])
            if r.get("duration_m") is not None:
                wd[r["job"]][r["week"]].append(r["duration_m"])

        wtq: dict[str, list[float]] = _dd(list)
        for r in run_queue_records:
            wtq[r["week"]].append(r["total_queue_m"])
        wtd: dict[str, list[float]] = _dd(list)
        for r in run_duration_records:
            wtd[r["week"]].append(r["total_duration_m"])

        now_dt = datetime.now(timezone.utc)
        cur_y, cur_w, _ = now_dt.isocalendar()
        cur_week = f"{cur_y}-W{cur_w:02d}"
        all_weeks = sorted(
            w for w in ({r["week"] for r in records} | set(wtq.keys()) | set(wtd.keys())) if w != cur_week
        )

        def pct(vals, p):
            sv = sorted(vals)
            return sv[max(0, int(len(sv) * p) - 1)] if sv else float("nan")

        def row(vals):
            if not vals:
                return "   —"
            return (
                f"{median(vals):5.1f} / {pct(vals, 0.9):5.1f} / {stdev(vals) if len(vals) > 1 else 0:5.1f} "
                f" n={len(vals)}"
            )

        KEY_JOBS = [
            ("Pre-commit", "Pre-commit"),
            ("Run tests", "Run tests"),
            ("Run policy-related tests with GR00T & cuda12_8 deps", "Policy tests"),
            ("Build the docs (pre-merge)", "Build docs"),
            ("Build & push NGC image (post-merge)", "Build+push"),
        ]
        col_w = 32

        for metric, job_week_map, total_map, label in [
            ("QUEUE TIME (med/p90/σ, minutes)", wq, wtq, "Total queue"),
            ("DURATION   (med/p90/σ, minutes)", wd, wtd, "Total wall-clock"),
        ]:
            print(f"\n--- Weekly {metric} ---")
            header = f"  {'Job':<{col_w}}" + "".join(f"  {w:>28}" for w in all_weeks)
            print(header)
            print("  " + "-" * (col_w + 30 * len(all_weeks)))
            for api_name, display in KEY_JOBS:
                vals_by_week = job_week_map.get(api_name, {})
                line = f"  {display:<{col_w}}" + "".join(f"  {row(vals_by_week.get(w, [])):>28}" for w in all_weeks)
                print(line)
            # Total row
            line = f"  {'  ' + label:<{col_w}}" + "".join(f"  {row(total_map.get(w, [])):>28}" for w in all_weeks)
            print(line)

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Weekly trend plots
# ---------------------------------------------------------------------------


def plot_weekly_trends(data: dict, output_prefix: str = "ci_trends") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed — skipping plots. Run: pip install matplotlib")
        return

    records = data.get("job_records", [])
    run_queue_records = data.get("run_queue_records", [])
    run_duration_records = data.get("run_duration_records", [])
    if not records:
        print("No per-job records to plot.")
        return

    # Group by (display_name, week)
    week_queue: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    week_dur: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        display = PLOT_JOBS.get(r["job"])
        if display is None:
            continue
        if r.get("queue_m") is not None:
            week_queue[display][r["week"]].append(r["queue_m"])
        if r.get("duration_m") is not None:
            week_dur[display][r["week"]].append(r["duration_m"])

    # Total queue/duration per run, grouped by week
    week_total_queue: dict[str, list[float]] = defaultdict(list)
    for r in run_queue_records:
        week_total_queue[r["week"]].append(r["total_queue_m"])

    week_total_dur: dict[str, list[float]] = defaultdict(list)
    for r in run_duration_records:
        week_total_dur[r["week"]].append(r["total_duration_m"])

    # Exclude the current (partial) ISO week so incomplete data doesn't distort trends.
    now = datetime.now(timezone.utc)
    current_iso_year, current_iso_week, _ = now.isocalendar()
    current_week = f"{current_iso_year}-W{current_iso_week:02d}"

    all_weeks = sorted(
        w
        for w in (
            {r["week"] for r in records if PLOT_JOBS.get(r["job"])}
            | set(week_total_queue.keys())
            | set(week_total_dur.keys())
        )
        if w != current_week
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def weeks_and_medians(job_week_map: dict[str, list[float]]) -> tuple[list, list]:
        ws = [w for w in all_weeks if job_week_map.get(w)]
        ms = [median(job_week_map[w]) for w in ws]
        return ws, ms

    def finalize_ax(ax, title, ylabel):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("ISO Week")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="y", which="minor", linestyle=":", alpha=0.3)
        if len(all_weeks) > 8:
            plt.xticks(rotation=45, ha="right")
        ax.legend(loc="upper left", fontsize=9)

    def simple_plot(job_week_map_by_display, total_week_map, title, ylabel, filename, total_label=None):
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, display in enumerate(PLOT_JOBS.values()):
            ws, ms = weeks_and_medians(job_week_map_by_display[display])
            if ws:
                n = sum(len(job_week_map_by_display[display][w]) for w in ws)
                ax.plot(ws, ms, marker="o", label=f"{display} (n={n})", color=colors[i % len(colors)])
        if total_label is not None and total_week_map:
            ws_tot, ms_tot = weeks_and_medians(total_week_map)
            if ws_tot:
                n_tot = sum(len(total_week_map[w]) for w in ws_tot)
                ax.plot(
                    ws_tot,
                    ms_tot,
                    marker="s",
                    linestyle="--",
                    linewidth=2,
                    color="black",
                    label=f"{total_label} (n={n_tot})",
                )
        finalize_ax(ax, title, ylabel)
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"  Saved: {filename}")

    # Queue time: individual jobs only — "total queue" removed as median-of-sums
    # is misleading (see weekly stats table for per-job med/p90/σ breakdown).
    simple_plot(
        week_queue,
        {},
        title="CI Job Queue Time per Week\n(median minutes waiting for a runner)",
        ylabel="Queue time (minutes)",
        filename=f"{output_prefix}_queue_time.png",
    )
    # Duration: individual jobs + total wall-clock (run created → finished,
    # fully-successful runs only).
    simple_plot(
        week_dur,
        week_total_dur,
        title="CI Job Duration per Week\n(successful runs only — median minutes)",
        ylabel="Duration (minutes)",
        filename=f"{output_prefix}_duration.png",
        total_label="Total wall-clock (run created → finished)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    current_year = datetime.now(timezone.utc).year
    parser = argparse.ArgumentParser(
        description="Analyze IsaacLab-Arena CI timing and failures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--since",
        default=f"{current_year}-01-01",
        metavar="YYYY-MM-DD",
        help=f"Fetch runs created on or after this date (default: {current_year}-01-01)",
    )
    parser.add_argument(
        "--workflow",
        choices=list(WORKFLOW_IDS.keys()) + ["all"],
        default="all",
        help="Which workflow to analyze (default: all)",
    )
    parser.add_argument(
        "--output",
        default="ci_analysis_results.json",
        help="Path for JSON output (default: ci_analysis_results.json)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate weekly trend PNG plots (requires matplotlib)",
    )
    parser.add_argument(
        "--plot-prefix",
        default="ci_trends",
        help="Filename prefix for plot PNGs (default: ci_trends)",
    )
    args = parser.parse_args()

    since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    wf_ids = WORKFLOW_IDS if args.workflow == "all" else {args.workflow: WORKFLOW_IDS[args.workflow]}

    print("Authenticating with GitHub...", flush=True)
    token = get_token()

    print(f"Fetching runs since {args.since} from {REPO}...\n", flush=True)
    data = analyze(wf_ids, since_dt, token)

    print_report(data)

    if args.plot:
        print("\nGenerating weekly trend plots...", flush=True)
        plot_weekly_trends(data, output_prefix=args.plot_prefix)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nRaw data saved to: {args.output}")


if __name__ == "__main__":
    main()
