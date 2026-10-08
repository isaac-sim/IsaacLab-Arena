# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from mujoco_warp import Data
    from newton import Contacts


@wp.kernel
def _record_num_newton_contacts(num_contacts: wp.array(dtype=int), peaks: wp.array(dtype=int)) -> None:
    wp.atomic_max(peaks, 0, num_contacts[0])


@wp.kernel
def _record_solver_status(
    num_contacts: wp.array(dtype=int),
    num_constraints: wp.array(dtype=int),
    overflow_flags: wp.array(dtype=int),
    peaks: wp.array(dtype=int),
) -> None:
    world_index = wp.tid()
    if world_index == 0:
        wp.atomic_max(peaks, 0, num_contacts[0])
    wp.atomic_max(peaks, 1, num_constraints[world_index])
    wp.atomic_or(peaks, 2, overflow_flags[world_index])


class NewtonBufferMonitor:
    """Latch MuJoCo-Warp buffer failures across substeps and CUDA graph replay."""

    def __init__(self, data: "Data") -> None:
        """Monitor the allocated buffers of a MuJoCo-Warp solver.

        Args:
            data: Solver data containing buffer capacities, usage counts, and overflow flags.
        """
        self._data: Data = data
        """MuJoCo-Warp solver data whose buffer usage is monitored."""
        self._peaks: wp.array = wp.zeros(3, dtype=int, device=data.nefc.device)
        """Latched contact count, maximum constraint count, and overflow flags."""
        self._num_rigid_contacts_max: int | None = None
        """External Newton contact capacity, populated when external contacts are recorded."""

    def record_newton_contacts(self, contacts: "Contacts") -> None:
        """Record the raw Newton contact count before transfer can truncate it.

        Args:
            contacts: External collision contacts about to be passed to the solver.
        """
        self._num_rigid_contacts_max = contacts.rigid_contact_max
        wp.launch(
            _record_num_newton_contacts,
            dim=1,
            inputs=[contacts.rigid_contact_count, self._peaks],
            device=self._peaks.device,
        )

    def record_solver_status(self) -> None:
        """Record contact and constraint usage and overflow flags after a solver substep.

        Atomic maxima preserve any larger Newton contact count recorded before truncation.
        """
        wp.launch(
            _record_solver_status,
            dim=self._data.nworld,
            inputs=[self._data.nacon, self._data.nefc, self._data.overflow, self._peaks],
            device=self._peaks.device,
        )

    def check(self) -> None:
        """Abort on overflow after replay, synchronizing one small status array to the host."""
        num_contacts, num_constraints, flags = self._peaks.numpy().tolist()
        failures = []
        if num_contacts > self._data.naconmax:
            num_contacts_required = (num_contacts + self._data.nworld - 1) // self._data.nworld
            failures.append(
                f"nconmax: {num_contacts} contacts exceeded the pooled capacity {self._data.naconmax}; "
                f"increase sim.physics.solver_cfg.nconmax to at least {num_contacts_required} per world"
            )
        if self._num_rigid_contacts_max is not None and num_contacts > self._num_rigid_contacts_max:
            failures.append(
                f"rigid_contact_max: {num_contacts} contacts exceeded {self._num_rigid_contacts_max}; "
                f"increase sim.physics.collision_cfg.rigid_contact_max to at least {num_contacts}"
            )
        if num_constraints > self._data.njmax:
            failures.append(
                f"njmax: {num_constraints} constraint rows exceeded {self._data.njmax}; "
                f"increase sim.physics.solver_cfg.njmax to at least {num_constraints} per world"
            )
        if flags:
            failures.append(f"MuJoCo-Warp overflow flags: {flags:#x}")
        assert not failures, (
            "Newton physics buffer overflow; simulation results are invalid. "
            + "; ".join(failures)
            + ". Increase the indicated capacities with headroom and restart the run."
        )
