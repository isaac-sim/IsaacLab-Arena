# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory recording of variation samples.

A :class:`VariationLedger` collects every value drawn by an enabled variation's
sampler so downstream sensitivity-analysis tooling has the input factors that
produced each episode. The ledger is explicitly attached by the caller: there
is no singleton or global lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg


class VariationRecord:
    """Per-variation slice of a :class:`VariationLedger`.

    Bundles a variation's identity (:attr:`source_id`), the cfg driving its
    sampler at attach time (:attr:`cfg`), and the ordered sequence of samples
    drawn into the record (:attr:`samples`).
    """

    def __init__(self, source_id: str, cfg: VariationBaseCfg) -> None:
        self.source_id = source_id
        #: Cfg reference captured at :meth:`VariationLedger.attach` time. The ledger
        #: is attached after Hydra overrides have been applied, so the cfg is
        #: treated as finalized; deep-copy if a frozen archival snapshot is needed.
        self.cfg = cfg
        #: One entry per :meth:`~isaaclab_arena.variations.sampler.Sampler.sample`
        #: call. Each is a detached CPU tensor of shape ``(num_samples, *event_shape)``.
        self.samples: list[torch.Tensor] = []


class VariationLedger:
    """Records every sample drawn by attached variations, grouped per variation.

    Records are keyed by ``source_id`` (by convention ``"{asset}.{variation}"``).
    Serialisation is intentionally not handled here.
    """

    def __init__(self) -> None:
        self._records: dict[str, VariationRecord] = {}

    def attach(self, source_id: str, variation: VariationBase) -> None:
        """Subscribe this ledger to ``variation`` under ``source_id``.

        The listener is registered via
        :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`
        so it survives subsequent sampler swaps.

        Args:
            source_id: Identifier the record is stored under. By convention
                ``"{asset_name}.{variation.name}"``.
            variation: The variation to observe.
        """
        assert source_id not in self._records, (
            f"VariationLedger: source_id '{source_id}' is already attached. "
            "Re-attaching the same variation would create two independent records and "
            "double-record subsequent samples; detach or use a different source_id instead."
        )

        record = VariationRecord(source_id=source_id, cfg=variation.cfg)
        self._records[source_id] = record

        def on_sample(sample: torch.Tensor) -> None:
            record.samples.append(sample.detach().cpu())

        variation.add_sample_listener(on_sample)

    @property
    def records(self) -> list[VariationRecord]:
        """All per-variation records, in attach order."""
        return list(self._records.values())

    def __getitem__(self, source_id: str) -> VariationRecord:
        return self._records[source_id]

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._records
