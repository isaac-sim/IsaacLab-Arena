#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and launch an Isaac Lab Arena OSMO workflow.

Usage examples:

    # Default evaluation (zero_action on kitchen_pick_and_place)
    python osmo/launch_arena.py --pool isaac-dev-l40-03

    # Custom command
    python osmo/launch_arena.py \
        --pool isaac-dev-l40-03 \
        --command '/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
            --policy_type zero_action --num_steps 500 --headless \
            kitchen_pick_and_place --object cracker_box --embodiment franka_ik'

    # Run tests instead
    python osmo/launch_arena.py \
        --pool isaac-dev-l40-03
        --command 'ISAACLAB_ARENA_SUBPROCESS_TIMEOUT=900 \
            /isaac-sim/python.sh -m pytest -sv --durations=0 -m with_subprocess \
            isaaclab_arena/tests/'

    # Override resources
    python osmo/launch_arena.py --gpus 2 --platform ovx-l40 --memory 128Gi \
        --pool isaac-dev-l40-03 \
        --command '/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
            --policy_type zero_action --num_steps 500 --headless \
            kitchen_pick_and_place --object cracker_box --embodiment franka_ik'

    # Dry run (print rendered YAML without submitting)
    python osmo/launch_arena.py --pool isaac-dev-l40-03 --dry-run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WORKFLOW_YAML = Path(__file__).parent / "arena_base.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and submit an Isaac Lab Arena OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    task = parser.add_argument_group("task")
    task.add_argument(
        "--command",
        default=None,
        help="Shell command to run inside the container (overrides the default-values in the YAML)",
    )

    resources = parser.add_argument_group("resources")
    resources.add_argument("--cpus", type=int, default=15)
    resources.add_argument("--gpus", type=int, default=1)
    resources.add_argument("--memory", default="64Gi")
    resources.add_argument("--storage", default="200Gi")
    resources.add_argument("--platform", default="ovx-l40")

    timeouts = parser.add_argument_group("timeouts")
    timeouts.add_argument("--exec_timeout", default="1d")
    timeouts.add_argument("--queue_timeout", default="2d")

    workflow = parser.add_argument_group("workflow")
    workflow.add_argument("--workflow_name", default="arena-evaluation", help="OSMO workflow name")
    workflow.add_argument("--pool", default=None, help="Target a specific OSMO compute pool")
    workflow.add_argument("--priority", default="NORMAL", choices=["HIGH", "NORMAL", "LOW"])
    workflow.add_argument("--yaml", type=Path, default=WORKFLOW_YAML, help="Path to the workflow YAML template")

    parser.add_argument("--dry-run", action="store_true", help="Render and validate without submitting")

    return parser


def render_yaml(template_path: Path, values: dict[str, str]) -> str:
    """Read the YAML template, strip the default-values block, and substitute {{ tokens }}."""
    raw = template_path.read_text()

    # Remove the default-values block (not needed once we render ourselves)
    raw = re.sub(r"\ndefault-values:.*", "", raw, flags=re.DOTALL)

    for key, value in values.items():
        raw = raw.replace("{{ " + key + " }}", value)
        raw = raw.replace("{{" + key + "}}", value)

    return raw


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.yaml.exists():
        print(f"Error: workflow YAML not found: {args.yaml}", file=sys.stderr)
        return 1

    command = None
    if args.command is not None:
        command = args.command.replace("\\\n", " ")
        command = " ".join(command.split())

    values: dict[str, str] = {
        "workflow_name": args.workflow_name,
        "cpus": str(args.cpus),
        "gpus": str(args.gpus),
        "memory": args.memory,
        "storage": args.storage,
        "platform": args.platform,
        "exec_timeout": args.exec_timeout,
        "queue_timeout": args.queue_timeout,
    }

    # Read the default command from the YAML default-values if not overridden
    if command is None:
        raw = args.yaml.read_text()
        match = re.search(r"command:\s*>-\n((?:\s+.*\n?)+)", raw)
        if match:
            command = " ".join(match.group(1).split())
    if command:
        values["command"] = command

    rendered = render_yaml(args.yaml, values)

    if args.dry_run:
        print("[dry-run] Rendered workflow YAML:\n")
        print(rendered)
        return 0

    # Write rendered YAML to a temp file and submit it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="arena_", delete=False) as f:
        f.write(rendered)
        rendered_path = f.name

    cmd = ["osmo", "workflow", "submit", rendered_path]
    if args.pool:
        cmd.extend(["--pool", args.pool])
    if args.priority:
        cmd.extend(["--priority", args.priority])

    print(f"Submitting workflow '{args.workflow_name}':")
    print(f"  {' '.join(cmd)}")
    print(f"  (rendered from {args.yaml})\n")

    result = subprocess.run(cmd)

    Path(rendered_path).unlink(missing_ok=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
