# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the ResourceUsageTrackerDaemon and its ProcessResourceReader."""

from __future__ import annotations

import threading
import time
import types

import pytest

import isaaclab_arena.utils.resource_usage_tracker as tracker_module
from isaaclab_arena.utils.resource_usage_tracker import (
    ProcessResourceReader,
    ResourceSample,
    ResourceUsageTrackerDaemon,
)

THREAD_NAME = "resource-usage-tracker-daemon"


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


def _patch_reader(monkeypatch, reader: FakeReader) -> None:
    """Make the tracker use the given reader instead of constructing a ProcessResourceReader."""
    monkeypatch.setattr(tracker_module, "ProcessResourceReader", lambda gpu_index: reader)


def _sample(vram=100.0, ram=200.0, cpu=50.0, gpu=10.0) -> ResourceSample:
    return ResourceSample(ram_mb=ram, cpu_util_percent=cpu, vram_mb=vram, gpu_util_percent=gpu)


def _thread_names() -> list[str]:
    return [thread.name for thread in threading.enumerate()]


def _wait_for(condition, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for condition"
        time.sleep(0.005)


def test_peak_and_mean_match_scripted_samples(monkeypatch):
    reader = FakeReader([_sample(vram=100, ram=300, cpu=None, gpu=10), _sample(vram=500, ram=200, cpu=20, gpu=30)])
    _patch_reader(monkeypatch, reader)
    with ResourceUsageTrackerDaemon(interval_s=0.01) as tracker:
        _wait_for(lambda: reader.sample_count >= 5)
    usage = tracker.get_usage()
    assert usage.vram.maximum == 500
    assert usage.ram.maximum == 300
    assert usage.gpu_util.maximum == 30
    assert 10 <= usage.gpu_util.get_mean() <= 30
    assert usage.cpu_util.get_mean() == 20
    assert reader.closed


def test_immediate_exit_has_start_and_stop_samples(monkeypatch):
    reader = FakeReader([_sample()])
    _patch_reader(monkeypatch, reader)
    with ResourceUsageTrackerDaemon(interval_s=10.0) as tracker:
        pass
    assert tracker.get_usage().vram.count >= 2


def test_first_cpu_sample_is_skipped(monkeypatch):
    reader = FakeReader([_sample(cpu=None)])
    _patch_reader(monkeypatch, reader)
    with ResourceUsageTrackerDaemon(interval_s=10.0) as tracker:
        pass
    assert tracker.get_usage().cpu_util.count == 0
    assert tracker.get_usage().vram.count == 2


def test_thread_is_gone_after_exit_and_after_exception(monkeypatch):
    _patch_reader(monkeypatch, FakeReader([_sample()]))
    with ResourceUsageTrackerDaemon(interval_s=0.01):
        assert THREAD_NAME in _thread_names()
    assert THREAD_NAME not in _thread_names()

    _patch_reader(monkeypatch, FakeReader([_sample()]))
    with pytest.raises(RuntimeError):
        with ResourceUsageTrackerDaemon(interval_s=0.01):
            raise RuntimeError("boom")
    assert THREAD_NAME not in _thread_names()


def test_exit_twice_is_harmless(monkeypatch):
    _patch_reader(monkeypatch, FakeReader([_sample()]))
    tracker = ResourceUsageTrackerDaemon(interval_s=0.01)
    tracker.__enter__()
    tracker.__exit__(None, None, None)
    tracker.__exit__(None, None, None)


def test_sampling_error_stops_sampling_without_raising(capsys, monkeypatch):
    class FailingReader(FakeReader):
        def sample(self):
            if self.sample_count >= 1:
                self.sample_count += 1
                raise OSError("sampling failed")
            return super().sample()

    reader = FailingReader([_sample()])
    _patch_reader(monkeypatch, reader)
    with ResourceUsageTrackerDaemon(interval_s=0.01):
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
    monkeypatch.setattr(tracker_module, "pynvml", fake)
    with pytest.raises(RuntimeError, match="NVML unavailable"):
        with ResourceUsageTrackerDaemon(interval_s=0.01):
            pass
    assert THREAD_NAME not in _thread_names()


def test_old_pynvml_without_v2_memory_info_fails_at_start_without_thread(monkeypatch):
    fake = _make_fake_pynvml()
    del fake.nvmlMemory_v2
    fake.__file__ = "/isaac-sim/pip_prebundle/pynvml/__init__.py"
    init_calls = []
    fake.nvmlInit = lambda: init_calls.append(1)
    monkeypatch.setattr(tracker_module, "pynvml", fake)
    with pytest.raises(AssertionError, match=r"nvidia-ml-py.*pip_prebundle/pynvml.*shadow"):
        with ResourceUsageTrackerDaemon(interval_s=0.01):
            pass
    assert init_calls == []
    assert fake.shutdown_calls == 0
    assert THREAD_NAME not in _thread_names()


def test_reader_shuts_down_nvml_when_handle_lookup_fails(monkeypatch):
    fake = _make_fake_pynvml()

    def fail_handle(index):
        raise RuntimeError("no such GPU")

    fake.nvmlDeviceGetHandleByIndex = fail_handle
    monkeypatch.setattr(tracker_module, "pynvml", fake)
    with pytest.raises(RuntimeError, match="no such GPU"):
        ProcessResourceReader(gpu_index=7)
    assert fake.shutdown_calls == 1


def test_reader_closed_even_if_stop_sample_is_interrupted(monkeypatch):
    reader = FakeReader([_sample()])

    def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _patch_reader(monkeypatch, reader)
        with ResourceUsageTrackerDaemon(interval_s=60.0):
            reader.sample = interrupted
    assert reader.closed
    assert THREAD_NAME not in _thread_names()


def test_reader_uses_v2_used_memory_and_proc_rss(monkeypatch):
    fake = _make_fake_pynvml()
    monkeypatch.setattr(tracker_module, "pynvml", fake)
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
