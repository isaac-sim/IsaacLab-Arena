# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.evaluation import policy_runner


def test_preload_external_policy_module_imports_requested_module(monkeypatch):
    imported_modules = []
    monkeypatch.setattr(policy_runner, "import_module", imported_modules.append)

    policy_runner.preload_external_policy_module("example.policy.RemotePolicy")

    assert imported_modules == ["example.policy"]


def test_preload_external_policy_module_skips_registered_names(monkeypatch):
    imported_modules = []
    monkeypatch.setattr(policy_runner, "import_module", imported_modules.append)

    policy_runner.preload_external_policy_module("zero_action")
    policy_runner.preload_external_policy_module(None)

    assert imported_modules == []
