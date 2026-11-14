# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shell helper that turns a GPU memory reservation into XLA_PYTHON_CLIENT_MEM_FRACTION."""

from __future__ import annotations

import os
import pathlib
import stat
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "server" / "gpu_memory_fraction.sh"
_FAKE_GPU_TOTAL_MIB = 49140


@pytest.fixture
def fake_nvidia_smi_path(tmp_path):
    """A PATH whose nvidia-smi reports a 48 GB GPU, so absolute sizes convert deterministically."""
    fake = tmp_path / "nvidia-smi"
    fake.write_text(f"#!/usr/bin/env bash\necho '{_FAKE_GPU_TOTAL_MIB}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return f"{tmp_path}{os.pathsep}{os.environ['PATH']}"


def _fraction(memory: str, path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_SCRIPT), memory], capture_output=True, text=True, env={**os.environ, "PATH": path}
    )


@pytest.mark.parametrize("memory", ["0.5", "0.75", ".75", "1", "1.0"])
def test_fractions_pass_through(memory, fake_nvidia_smi_path):
    result = _fraction(memory, fake_nvidia_smi_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == memory


@pytest.mark.parametrize(
    "memory, expected_mib",
    [
        ("12GB", 12 * 1024),
        ("12G", 12 * 1024),
        ("12GiB", 12 * 1024),
        ("1.5GB", 1536),
        ("6000MB", 6000),
        ("6000MiB", 6000),
        ("6000M", 6000),
    ],
)
def test_sizes_divide_by_the_gpu_total(memory, expected_mib, fake_nvidia_smi_path):
    result = _fraction(memory, fake_nvidia_smi_path)
    assert result.returncode == 0, result.stderr
    assert float(result.stdout) == pytest.approx(expected_mib / _FAKE_GPU_TOTAL_MIB, abs=5e-5)


@pytest.mark.parametrize("memory", ["0", "0.0", "1.5", "2", "12", "12TB", "abc", "", "0GB", "100GB"])
def test_rejects_values_outside_the_gpu(memory, fake_nvidia_smi_path):
    result = _fraction(memory, fake_nvidia_smi_path)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "GPU memory" in result.stderr


def test_rejects_an_unreadable_gpu_total(tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/usr/bin/env bash\necho 'N/A'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    result = _fraction("12GB", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert result.returncode != 0
    assert "nvidia-smi" in result.stderr
