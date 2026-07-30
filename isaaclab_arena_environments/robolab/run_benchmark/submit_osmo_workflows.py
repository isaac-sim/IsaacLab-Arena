# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submit OSMO evaluation workflows for the first robolab scene specs."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

POLICIES = ("pi0", "gr00t")
NUM_ENVS_BY_POLICY = {
    "pi0": 10,
    "gr00t": 6,
}
PRIORITY = "NORMAL"
POOL = "isaac-dev-l40s-04"
PLATFORM = "ovx-l40s"
VARIATIONS = (
    "light.hdr_image.enabled=true "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.high=[0.05,0.05,0.05] "
    "droid_abs_joint_pos.camera_extrinsics_wrist_camera.sampler_cfg.low=[-0.05,-0.05,-0.05] "
    "light.hdr_image.hdr_names=[home_office_robolab]"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _num_robolab_environments(value: str) -> int:
    num_environments = int(value)
    if num_environments == -1 or num_environments > 0:
        return num_environments
    raise argparse.ArgumentTypeError("must be -1 to submit all robolab environments, or a positive integer")


def get_n_robolab_envs(num_environments: int) -> list[Path]:
    robolab_dir = Path(__file__).resolve().parents[1]
    robolab_envs = sorted(robolab_dir.glob("tasks/*.yaml"))
    if num_environments == -1:
        return robolab_envs
    return robolab_envs[:num_environments]


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(_repo_root()).as_posix()


def _workflow_name(prefix: str, policy: str, env_path: Path, run_id: str) -> str:
    stem = env_path.stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return f"{prefix}-{policy}-{slug}-{run_id}"


def _policy_runner_args(policy: str, num_episodes: int) -> str:
    return (
        f"--enable_cameras --num_envs {NUM_ENVS_BY_POLICY[policy]} "
        f"--num_episodes {num_episodes} --headless --record_camera_video"
    )


def _run_id() -> str:
    """Reverse datetime (year-first) shared by all workflows submitted in one run."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _build_command(args: argparse.Namespace, policy: str, env_path: Path, run_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "osmo.submit_evaluation_workflow",
        "--policy",
        policy,
        "--workflow_name",
        _workflow_name(args.workflow_name_prefix, policy, env_path, run_id),
        "--priority",
        PRIORITY,
        "--pool",
        POOL,
        "--platform",
        PLATFORM,
        "--policy_runner_args",
        _policy_runner_args(policy, args.num_episodes),
        "--arena_env",
        _repo_relative(env_path),
        "--variation_args",
        VARIATIONS,
    ]
    if args.dry_run:
        command.append("--dry_run")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_robolab_environments",
        type=_num_robolab_environments,
        default=2,
        help="Number of sorted robolab YAMLs to submit. Use -1 to submit all robolab environments.",
    )
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate per workflow")
    parser.add_argument("--workflow_name_prefix", default="robolab", help="Prefix for generated OSMO workflow names")
    parser.add_argument("--dry-run", action="store_true", help="Render workflows without submitting them")
    args = parser.parse_args()

    envs = get_n_robolab_envs(args.num_robolab_environments)
    if len(envs) < args.num_robolab_environments:
        print(f"Expected {args.num_robolab_environments} robolab YAMLs, found {len(envs)}.", file=sys.stderr)
        return 1

    run_id = _run_id()
    for env_path in envs:
        for policy in POLICIES:
            command = _build_command(args, policy, env_path, run_id)
            print(f"\n$ {shlex.join(command)}", flush=True)
            result = subprocess.run(command, cwd=_repo_root())
            if result.returncode != 0:
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
