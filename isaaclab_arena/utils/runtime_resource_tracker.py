# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Samples the memory and utilization of the current process and GPU while a block of code runs."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from isaaclab_arena.utils.resource_usage import ResourceUsage

BYTES_PER_MB = 2**20


@dataclass
class ResourceSample:
    """One reading of the resources used by the process and the GPU."""

    ram_mb: float
    """Resident memory of the process in MB."""

    cpu_util_percent: float | None
    """CPU time of the process since the previous sample, in percent of one core. None on the first sample."""

    vram_mb: float
    """Device-wide GPU memory in use in MB."""

    gpu_util_percent: float
    """GPU utilization of the device in percent."""


class ProcessResourceReader:
    """Reads the resources used by the current process and a GPU through NVML and /proc."""

    def __init__(self, gpu_index: int = 0):
        """Initialize NVML and get the handle of the GPU; fails if NVML cannot initialize.

        Args:
            gpu_index: Index of the GPU to read.
        """
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._last_cpu_time_s: float | None = None
        self._last_wall_time_s: float | None = None

    def sample(self) -> ResourceSample:
        """Read the current resource usage.

        Returns:
            The sample. Its CPU utilization is None on the first call.
        """
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        ram_mb = resident_pages * self._page_size / BYTES_PER_MB

        cpu_time_s = time.process_time()
        wall_time_s = time.monotonic()
        cpu_util_percent = None
        if self._last_cpu_time_s is not None and wall_time_s > self._last_wall_time_s:
            cpu_util_percent = (cpu_time_s - self._last_cpu_time_s) / (wall_time_s - self._last_wall_time_s) * 100.0
        self._last_cpu_time_s = cpu_time_s
        self._last_wall_time_s = wall_time_s

        memory_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle, version=self._pynvml.nvmlMemory_v2)
        gpu_util_percent = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        return ResourceSample(
            ram_mb=ram_mb,
            cpu_util_percent=cpu_util_percent,
            vram_mb=memory_info.used / BYTES_PER_MB,
            gpu_util_percent=gpu_util_percent,
        )

    def close(self) -> None:
        """Shut down NVML."""
        self._pynvml.nvmlShutdown()


class RuntimeResourceTracker:
    """Context manager sampling process and GPU resources at a fixed interval while its block runs."""

    def __init__(self, interval_s: float = 0.1, gpu_index: int = 0, reader: ProcessResourceReader | None = None):
        """Create a tracker; NVML is initialized when the context is entered.

        Args:
            interval_s: Seconds between samples.
            gpu_index: Index of the GPU to read.
            reader: Replaces the NVML reader; it only needs sample() and close().
        """
        self._interval_s = interval_s
        self._gpu_index = gpu_index
        self._reader = reader
        self._usage = ResourceUsage()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = False

    @property
    def usage(self) -> ResourceUsage:
        """The statistics of the samples so far; complete after the context exits."""
        return self._usage

    def __enter__(self) -> RuntimeResourceTracker:
        """Take the first sample and start the sampling thread."""
        if self._reader is None:
            self._reader = ProcessResourceReader(self._gpu_index)
        try:
            self._record(self._reader.sample())
        except BaseException:
            self._reader.close()
            raise
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="runtime-resource-tracker", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop the sampling thread, take the last sample and close the reader."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        if not self._failed:
            try:
                self._record(self._reader.sample())
            except Exception as error:
                self._report_failure(error)
        self._reader.close()

    def _record(self, sample: ResourceSample) -> None:
        self._usage.vram.update(sample.vram_mb)
        self._usage.ram.update(sample.ram_mb)
        self._usage.gpu_util.update(sample.gpu_util_percent)
        if sample.cpu_util_percent is not None:
            self._usage.cpu_util.update(sample.cpu_util_percent)

    def _report_failure(self, error: Exception) -> None:
        if not self._failed:
            self._failed = True
            print(f"RuntimeResourceTracker stopped sampling: {error!r}")

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._record(self._reader.sample())
            except Exception as error:
                self._report_failure(error)
                return
