# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import warp as wp

from isaaclab_arena.environments.newton_buffer_monitor import NewtonBufferMonitor


def _make_monitor(device: str = "cpu") -> tuple[NewtonBufferMonitor, SimpleNamespace, SimpleNamespace]:
    data = SimpleNamespace(
        nacon=wp.array([0], dtype=int, device=device),
        nefc=wp.array([0, 0], dtype=int, device=device),
        overflow=wp.array([0, 0], dtype=int, device=device),
        naconmax=8,
        njmax=6,
        nworld=2,
    )
    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([0], dtype=int, device=device),
        rigid_contact_max=12,
    )
    return NewtonBufferMonitor(data), data, contacts


@pytest.mark.parametrize(
    ("num_contacts", "num_constraints", "flags", "message"),
    [
        (8, [6, 6], [0, 0], None),
        (9, [1, 1], [0, 0], "nconmax.*at least 5"),
        (1, [1, 7], [0, 0], "njmax.*at least 7"),
        (1, [1, 1], [0, 1], "overflow flags"),
        (1, [1, 1], [0, 16], "overflow flags"),
    ],
)
def test_newton_buffer_limits(
    num_contacts: int, num_constraints: list[int], flags: list[int], message: str | None
) -> None:
    monitor, data, contacts = _make_monitor()
    contacts.rigid_contact_count.fill_(num_contacts)
    data.nefc.assign(num_constraints)
    data.overflow.assign(flags)
    monitor.record_newton_contacts(contacts)
    monitor.record_solver_status()
    if message is None:
        monitor.check()
    else:
        with pytest.raises(AssertionError, match=message):
            monitor.check()


def test_newton_collision_pipeline_capacity() -> None:
    monitor, data, contacts = _make_monitor()
    contacts.rigid_contact_max = 4
    contacts.rigid_contact_count.fill_(5)
    monitor.record_newton_contacts(contacts)
    with pytest.raises(AssertionError, match="rigid_contact_max.*at least 5"):
        monitor.check()


def test_internal_mujoco_contacts() -> None:
    monitor, data, _ = _make_monitor()
    data.nacon.fill_(9)
    monitor.record_solver_status()
    with pytest.raises(AssertionError, match="nconmax"):
        monitor.check()


@pytest.mark.parametrize("failure", ["contacts", "constraints", "flags"])
def test_overflow_survives_later_substeps_in_cuda_graph(failure: str) -> None:
    monitor, data, contacts = _make_monitor("cuda:0")
    monitor.record_newton_contacts(contacts)  # Compile before capture.
    monitor.record_solver_status()  # Compile before capture.
    with wp.ScopedCapture(device="cuda:0") as capture:
        contacts.rigid_contact_count.fill_(9 if failure == "contacts" else 1)
        monitor.record_newton_contacts(contacts)
        data.nefc.fill_(7 if failure == "constraints" else 1)
        data.overflow.fill_(1 if failure == "flags" else 0)
        monitor.record_solver_status()
        contacts.rigid_contact_count.fill_(1)
        monitor.record_newton_contacts(contacts)
        data.nefc.fill_(1)
        data.overflow.fill_(0)
        monitor.record_solver_status()
    wp.capture_launch(capture.graph)
    with pytest.raises(AssertionError, match="Newton physics buffer overflow"):
        monitor.check()
    fresh_monitor = NewtonBufferMonitor(data)
    fresh_monitor.record_newton_contacts(contacts)
    fresh_monitor.record_solver_status()
    fresh_monitor.check()
