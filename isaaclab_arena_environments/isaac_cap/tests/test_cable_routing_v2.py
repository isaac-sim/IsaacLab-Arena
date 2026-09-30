# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the current CAP cable-routing environments."""

import math
from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess

pytestmark = pytest.mark.isaac_cap

_ENVIRONMENT_DIRECTORY = Path(__file__).parents[1] / "cable_routing_v2"
_BEHAVIOUR_DEMO_SCRIPT = _ENVIRONMENT_DIRECTORY / "cable_env_behaviour_demo.py"


def _test_cable_routing_v2_camera_calibration(_simulation_app) -> bool:
    """Fixed camera focal lengths match the native CAP cable-routing rig."""
    from isaaclab_arena_environments.isaac_cap.cable_routing_v2.embodiment.cameras import BimanualYamCameraCfg

    cameras = BimanualYamCameraCfg()
    cameras.use_cable_routing_rig()
    expected_focal_length = 4.8 / (2.0 * math.tan(math.radians(50.0 / 2.0)))

    assert cameras.top_camera.spawn.focal_length == expected_focal_length
    assert cameras.cable_camera.spawn.focal_length == expected_focal_length
    return True


def test_cable_routing_v2_camera_calibration() -> None:
    assert run_function_with_persistent_simulation_app(_test_cable_routing_v2_camera_calibration)


def test_cable_routing_v2_declarative_configuration() -> None:
    """Both variants inherit the shared workcell and standard Newton settings."""
    for variant in ("easy", "medium"):
        spec = load_env_graph_spec_dict(_ENVIRONMENT_DIRECTORY / f"cable_routing_{variant}.yaml")

        assert spec["default_physics_backend"] == "newton"
        assert spec["max_episode_steps"] == 30000
        assert spec["workcell"]["table"]["tabletop_z"] == 0.767
        assert spec["cable_physics"]["contact_stiffness"] == 10000.0

        override = spec["env_cfg_override"]
        assert override["num_rerenders_on_reset"] == 20
        assert override["decimation"] == 1
        assert override["scene"]["replicate_physics"] is True
        assert override["sim"]["dt"] == 1.0 / 60.0
        assert override["sim"]["render_interval"] == 4
        assert override["sim"]["physics"]["num_substeps"] == 16
        coupler = override["sim"]["physics"]["solver_cfg"]
        assert coupler["_target_"] == "isaaclab_contrib.coupling.CouplerProxyCfg"
        assert [entry["name"] for entry in coupler["entries"]] == ["rigid", "cable"]
        assert "proxies" not in coupler

        builder = spec["cable_builder"]
        assert builder["rigid_contact_history"] is True
        extensions = spec["solver_extensions"]
        assert extensions["rigid_jacobian"] == "sparse"
        assert extensions["contact_matching"] == "latest"

        proxy = spec["coupler_proxy"]
        assert proxy["source"] == "rigid" and proxy["destination"] == "cable"
        assert proxy["mode"] == "staggered"


@pytest.mark.with_subprocess
@pytest.mark.parametrize("variant", ("easy", "medium"))
def test_cable_routing_v2_behaviour_demo(variant: str) -> None:
    """Run one headless demo cycle and verify that success resets the environment."""
    result = run_subprocess(
        [
            TestConstants.python_path,
            str(_BEHAVIOUR_DEMO_SCRIPT),
            "--variant",
            variant,
            "--cycles",
            "1",
            "--pause-steps",
            "1",
            "--no-real-time",
            "--visualizer",
            "none",
        ],
        capture_output=True,
        timeout_sec=900,
    )

    assert result is not None
    expected = "[cable-behaviour-demo] cycle 1: success reset observed"
    assert expected in result.stdout, result.stdout + result.stderr
