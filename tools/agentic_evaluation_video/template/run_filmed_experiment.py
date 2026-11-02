# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run Arena's Experiment API with an editorial camera configured through a factory."""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--replay-placements", type=Path)
parser.add_argument(
    "--validate-config", action="store_true", help="Load typed configuration without evaluating the policy"
)
args, overrides = parser.parse_known_args()
output = Path(args.output)
assert not output.exists() or not list(output.iterdir()), "Output must be missing or empty"
output.mkdir(parents=True, exist_ok=True)
app = AppLauncher({"headless": True, "enable_cameras": True}).app

import yaml

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.experiment_runner import _write_arena_experiment_result
from isaaclab_arena.evaluation.run_execution import execute_experiment
from isaaclab_arena.visualization.report import build_report
from isaaclab_arena_environments.cli import ensure_environments_registered


@dataclass
class FilmEnvironmentCfg(ArenaEnvironmentCfg):
    spec_file: str = str(Path(__file__).with_name("source_environment.yaml"))


@register_environment
class FilmEnvironment(ArenaEnvironmentFactory[FilmEnvironmentCfg]):
    name = "film_kitchenbench_split_bowls"
    _legacy_argparse_cfg_type = FilmEnvironmentCfg

    def build(self, cfg: FilmEnvironmentCfg):
        data = yaml.safe_load(Path(cfg.spec_file).read_text())
        arena = ArenaEnvGraphSpec.from_dict(data).to_arena_env(enable_cameras=cfg.enable_cameras)
        arena.task.desired_subtask_success_state = [True, True]
        from film_camera import make_camera

        arena.embodiment.camera_config.external_camera_2 = make_camera()
        return arena


try:
    import isaaclab_arena

    print(f"ARENA_SOURCE {isaaclab_arena.__file__}", flush=True)
    checkout = Path(isaaclab_arena.__file__).resolve().parents[1]
    for variable, arguments in {
        "ARENA_GIT_SHA": ["rev-parse", "HEAD"],
        "ARENA_GIT_BRANCH": ["branch", "--show-current"],
    }.items():
        os.environ[variable] = subprocess.check_output(["git", "-C", str(checkout), *arguments], text=True).strip()
    ensure_environments_registered()
    experiment = load_arena_experiment_from_config_file(args.experiment_config, device="cuda:0", overrides=overrides)
    if args.replay_placements:
        for run_cfg in experiment.runs.values():
            assert not run_cfg.environment_builder.solve_relations, "Replay requires solve_relations: false"
            run_cfg.environment_builder.recorded_variation_samples_path = str(args.replay_placements.resolve())
    if args.validate_config:
        for name, run_cfg in experiment.runs.items():
            assert Path(run_cfg.environment.spec_file).is_file()
            replay = run_cfg.environment_builder.recorded_variation_samples_path
            assert replay is None or Path(replay).is_file()
            print(f"FILMED_CONFIG_VALID {name}: {run_cfg.environment.spec_file}; replay={replay}", flush=True)
    else:
        results = execute_experiment(experiment, output_dir=output, record_camera_video=True)
        _write_arena_experiment_result(experiment, results, output)
        report_path = build_report(output)
        recorded = json.loads((output / "arena_experiment_result.json").read_text())
        assert all(run["status"] == "completed" for run in recorded["runs"].values()), recorded
        print(f"FILMED_EXPERIMENT_COMPLETE {report_path}", flush=True)
finally:
    app.close()
