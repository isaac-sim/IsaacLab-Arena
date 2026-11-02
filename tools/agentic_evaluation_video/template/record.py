# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run filming steps in the running Docker container mounted to this checkout."""

import argparse
import getpass
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTAINER_ROOT = Path("/workspaces/isaaclab_arena")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["captures", "evaluation", "thumbnails"])
    parser.add_argument("--container", help="Optional explicit choice among containers mounting this checkout")
    parser.add_argument("--stage", help="Record just this capture stem; useful after an interrupted stage")
    args = parser.parse_args()
    assert args.phase == "captures" or args.stage is None, "--stage applies only to captures"
    repo = Path(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True).strip())
    project_in_container = CONTAINER_ROOT / ROOT.relative_to(repo)
    candidates = subprocess.check_output(
        ["docker", "ps", "--filter", f"volume={repo}", "--format", "{{.Names}}"], text=True
    ).splitlines()
    if args.container:
        assert args.container in candidates, "Selected container does not mount this checkout"
        container = args.container
    else:
        assert len(candidates) == 1, f"Expected one running checkout container, found {candidates}; use --container"
        container = candidates[0]
    project = json.loads((ROOT / "project.json").read_text())
    assert len(project["stages"]) == len(project["capture_counts"])
    definitions = [
        *ROOT.glob("*.yaml"),
        ROOT / "capture_stages.py",
        ROOT / "film_camera.py",
        ROOT / "run_filmed_experiment.py",
        ROOT / "camera.json",
        ROOT / "project.json",
    ]
    if project["replay_placements"]:
        definitions.append(ROOT / project["replay_placements"])
    provenance = {
        "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip(),
        "tracked_diff": subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"], text=True),
        "runtime_container": container,
        "definition_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in definitions},
    }
    provenance_path = ROOT / "source_provenance.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        assert old == provenance, "Sources/runtime changed between filming phases; use a new project directory"
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)

    def run(script, arguments, label):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = logs / f"{stamp}_{label}.log"
        command = [
            "env",
            f"PYTHONPATH={CONTAINER_ROOT}",
            "PYTHONUNBUFFERED=1",
            str(CONTAINER_ROOT / "submodules/IsaacLab/isaaclab.sh"),
            "-p",
            str(project_in_container / script),
            *map(str, arguments),
        ]
        print(f"Running {label}; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "--workdir",
                    str(CONTAINER_ROOT),
                    container,
                    "su",
                    getpass.getuser(),
                    "-c",
                    shlex.join(command),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"Completed {label}", flush=True)

    if args.phase == "captures":
        assert args.stage is None or args.stage in project["stages"], args.stage
        for stage, count in zip(project["stages"], project["capture_counts"]):
            if args.stage is not None and stage != args.stage:
                continue
            output = ROOT / "captures" / stage
            assert not output.exists() or not list(output.iterdir()), f"Preserving existing capture: {output}"
            run(
                "capture_stages.py",
                [
                    project_in_container / f"{stage}.yaml",
                    "--count",
                    count,
                    "--output-root",
                    project_in_container / "captures",
                ],
                stage,
            )
        sources = {s: f"captures/{s}" for s in project["stages"]}
        if all((ROOT / directory / "captures.json").exists() for directory in sources.values()):
            (ROOT / "capture_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    elif args.phase == "evaluation":
        arguments = [
            "--experiment_config",
            project_in_container / "experiment.yaml",
            "--output",
            project_in_container / "evaluation",
        ]
        if project["replay_placements"]:
            arguments.extend(["--replay-placements", project_in_container / project["replay_placements"]])
        run("run_filmed_experiment.py", arguments, "evaluation")
        runtime = dict(source_provenance="source_provenance.json", evaluation_directory="evaluation")
        (ROOT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    else:
        run("render_asset_thumbnails.py", [], "thumbnails")


if __name__ == "__main__":
    main()
