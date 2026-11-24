# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the RuntimeResourceTracker and its ProcessResourceReader."""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from isaaclab_arena.utils.runtime_resource_tracker import ProcessResourceReader, ResourceSample, RuntimeResourceTracker

THREAD_NAME = "runtime-resource-tracker"


class FakeReader:
    """Returns scripted samples, repeating the last one, and counts calls."""

    def __init__(self, samples: list[ResourceSample]):
        self.samples = samples
        self.sample_count = 0
        self.closed = False

    def sample(self) -> ResourceSample:
        sample = self.samples[min(self.sample_count, len(self.samples) - 1)]
        self.sample_count += 1
        return sample

    def close(self) -> None:
        self.closed = True


def _sample(vram=100.0, ram=200.0, cpu=50.0, gpu=10.0) -> ResourceSample:
    return ResourceSample(ram_mb=ram, cpu_util_percent=cpu, vram_mb=vram, gpu_util_percent=gpu)


def _thread_names() -> list[str]:
    return [thread.name for thread in threading.enumerate()]


def _wait_for(condition, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for condition"
        time.sleep(0.005)


def test_peak_and_mean_match_scripted_samples():
    reader = FakeReader([_sample(vram=100, ram=300, cpu=None, gpu=10), _sample(vram=500, ram=200, cpu=20, gpu=30)])
    with RuntimeResourceTracker(interval_s=0.01, reader=reader) as tracker:
        _wait_for(lambda: reader.sample_count >= 5)
    usage = tracker.usage
    assert usage.vram.maximum == 500
    assert usage.ram.maximum == 300
    assert usage.gpu_util.maximum == 30
    assert 10 <= usage.gpu_util.mean <= 30
    assert usage.cpu_util.mean == 20
    assert reader.closed


def test_immediate_exit_has_start_and_stop_samples():
    reader = FakeReader([_sample()])
    with RuntimeResourceTracker(interval_s=10.0, reader=reader) as tracker:
        pass
    assert tracker.usage.vram.count >= 2


def test_first_cpu_sample_is_skipped():
    reader = FakeReader([_sample(cpu=None)])
    with RuntimeResourceTracker(interval_s=10.0, reader=reader) as tracker:
        pass
    assert tracker.usage.cpu_util.count == 0
    assert tracker.usage.vram.count == 2


def test_thread_is_gone_after_exit_and_after_exception():
    with RuntimeResourceTracker(interval_s=0.01, reader=FakeReader([_sample()])):
        assert THREAD_NAME in _thread_names()
    assert THREAD_NAME not in _thread_names()

    with pytest.raises(RuntimeError):
        with RuntimeResourceTracker(interval_s=0.01, reader=FakeReader([_sample()])):
            raise RuntimeError("boom")
    assert THREAD_NAME not in _thread_names()


def test_exit_twice_is_harmless():
    tracker = RuntimeResourceTracker(interval_s=0.01, reader=FakeReader([_sample()]))
    tracker.__enter__()
    tracker.__exit__(None, None, None)
    tracker.__exit__(None, None, None)


def test_sampling_error_stops_sampling_without_raising(capsys):
    class FailingReader(FakeReader):
        def sample(self):
            if self.sample_count >= 1:
                self.sample_count += 1
                raise OSError("sampling failed")
            return super().sample()

    reader = FailingReader([_sample()])
    with RuntimeResourceTracker(interval_s=0.01, reader=reader):
        _wait_for(lambda: reader.sample_count >= 2)
        _wait_for(lambda: THREAD_NAME not in _thread_names())
    assert "sampling failed" in capsys.readouterr().out


def _make_fake_pynvml(init_error: Exception | None = None):
    fake = types.SimpleNamespace()
    fake.nvmlMemory_v2 = object()
    fake.memory_calls = []
    fake.shutdown_calls = 0

    def init():
        if init_error is not None:
            raise init_error

    def get_memory_info(handle, version=None):
        fake.memory_calls.append((handle, version))
        return types.SimpleNamespace(used=3 * 2**20)

    def shutdown():
        fake.shutdown_calls += 1

    fake.nvmlInit = init
    fake.nvmlDeviceGetHandleByIndex = lambda index: ("handle", index)
    fake.nvmlDeviceGetMemoryInfo = get_memory_info
    fake.nvmlDeviceGetUtilizationRates = lambda handle: types.SimpleNamespace(gpu=42)
    fake.nvmlShutdown = shutdown
    return fake


def test_nvml_init_failure_propagates_from_enter_without_thread(monkeypatch):
    fake = _make_fake_pynvml(init_error=RuntimeError("NVML unavailable"))
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    with pytest.raises(RuntimeError, match="NVML unavailable"):
        with RuntimeResourceTracker(interval_s=0.01):
            pass
    assert THREAD_NAME not in _thread_names()


def test_reader_uses_v2_used_memory_and_proc_rss(monkeypatch):
    fake = _make_fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    reader = ProcessResourceReader(gpu_index=1)
    first = reader.sample()
    second = reader.sample()
    reader.close()
    assert fake.memory_calls[0] == (("handle", 1), fake.nvmlMemory_v2)
    assert first.vram_mb == 3.0
    assert first.gpu_util_percent == 42
    assert first.ram_mb > 0
    assert first.cpu_util_percent is None
    assert second.cpu_util_percent is not None and second.cpu_util_percent >= 0
    assert fake.shutdown_calls == 1


def test_real_nvml_reports_vram():
    pytest.importorskip("pynvml")
    try:
        reader = ProcessResourceReader()
    except Exception as error:
        pytest.skip(f"NVML cannot initialize: {error}")
    try:
        assert reader.sample().vram_mb > 0
    finally:
        reader.close()
