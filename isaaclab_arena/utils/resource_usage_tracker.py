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

import pynvml

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
    """Reads the resources used by the current process and a GPU through NVML and /proc.

    VRAM and GPU utilization come from the GPU with the NVML index given at construction (default 0),
    regardless of ``--device`` or ``CUDA_VISIBLE_DEVICES``. On a multi-GPU machine that runs on another
    GPU they describe a different GPU.
    """

    def __init__(self, gpu_index: int = 0):
        """Initialize NVML and get the handle of the GPU; fails if NVML cannot initialize.

        Args:
            gpu_index: Index of the GPU to read.
        """
        assert hasattr(pynvml, "nvmlMemory_v2"), (
            "The imported pynvml has no nvmlMemory_v2; the resource tracker needs the nvidia-ml-py package."
            f" The imported module is {getattr(pynvml, '__file__', None)!r}; an older pynvml may shadow nvidia-ml-py."
        )
        pynvml.nvmlInit()
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except BaseException:
            pynvml.nvmlShutdown()
            raise
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._last_cpu_time_s: float | None = None
        self._last_wall_time_s: float | None = None

    def sample(self) -> ResourceSample:
        """Read the current resource usage.

        Returns:
            The sample. Its CPU utilization is None on the first call.
        """
        # RAM: resident set size, the second field of /proc/self/statm in pages.
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        ram_mb = resident_pages * self._page_size / BYTES_PER_MB

        # CPU utilization: CPU time of this process (all threads) since the previous sample divided by the wall
        # time since then, in percent of one core, so it can exceed 100. The first sample has no previous one.
        cpu_time_s = time.process_time()
        wall_time_s = time.monotonic()
        cpu_util_percent = None
        if self._last_cpu_time_s is not None and wall_time_s > self._last_wall_time_s:
            cpu_util_percent = (cpu_time_s - self._last_cpu_time_s) / (wall_time_s - self._last_wall_time_s) * 100.0
        self._last_cpu_time_s = cpu_time_s
        self._last_wall_time_s = wall_time_s

        # VRAM: device-wide memory in use from NVML's v2 memory info. It includes every process on the GPU and
        # excludes the memory the driver reserves for itself.
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle, version=pynvml.nvmlMemory_v2)
        # GPU utilization: NVML's utilization of the device, the percent of time a kernel was running recently.
        gpu_util_percent = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        return ResourceSample(
            ram_mb=ram_mb,
            cpu_util_percent=cpu_util_percent,
            vram_mb=memory_info.used / BYTES_PER_MB,
            gpu_util_percent=gpu_util_percent,
        )

    def close(self) -> None:
        """Shut down NVML."""
        pynvml.nvmlShutdown()


class ResourceUsageTrackerDaemon:
    """Context manager sampling process and GPU resources at a fixed interval while its block runs.

    It samples cheaply in a background thread and so can stay on during every Run, unlike profilers such as
    Nsight Systems.

    VRAM and GPU utilization are read from NVML GPU ``gpu_index`` (default 0), regardless of ``--device`` or
    ``CUDA_VISIBLE_DEVICES``, so on a multi-GPU machine that runs on another GPU they describe a different GPU.
    """

    def __init__(self, interval_s: float = 1.0, gpu_index: int = 0):
        """Create a tracker; sampling starts when the context is entered.

        Args:
            interval_s: Seconds between samples. A Run shorter than this has only the sample taken at entry, with
                no CPU utilization.
            gpu_index: Index of the GPU to read.
        """
        self._interval_s = interval_s
        self._gpu_index = gpu_index
        self._reader: ProcessResourceReader | None = None
        self._usage = ResourceUsage()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = False

    def get_usage(self) -> ResourceUsage:
        """The statistics of the samples so far; complete after the context exits."""
        return self._usage

    def __enter__(self) -> ResourceUsageTrackerDaemon:
        """Take the first sample and start the sampling thread."""
        self._reader = ProcessResourceReader(self._gpu_index)
        try:
            self._record(self._reader.sample())
        except BaseException:
            self._reader.close()
            raise
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="resource-usage-tracker-daemon", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop the sampling thread and close the reader."""
        if self._thread is None:
            return
        try:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        finally:
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
            print(f"ResourceUsageTrackerDaemon stopped sampling: {error!r}")

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._record(self._reader.sample())
            except Exception as error:
                self._report_failure(error)
                return
