# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the sole eval-runner entry point."""

from isaaclab_arena.evaluation import eval_runner


def test_eval_runner_parses_once_and_delegates_to_local_evaluation(monkeypatch):
    cfg = object()
    parsed_arguments = None
    received_cfg = None

    def _parse(arguments):
        nonlocal parsed_arguments
        parsed_arguments = arguments
        return cfg

    def _run_local(local_cfg):
        nonlocal received_cfg
        received_cfg = local_cfg
        return 17

    monkeypatch.setattr(eval_runner, "parse_eval_runner_cfg", _parse)
    monkeypatch.setattr(eval_runner, "run_local_experiment", _run_local)

    arguments = ["--experiment-config", "experiment.yaml"]
    assert eval_runner.main(arguments) == 17
    assert parsed_arguments is arguments
    assert received_cfg is cfg
