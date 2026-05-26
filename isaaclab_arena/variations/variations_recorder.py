# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory recording of variation samples.

A :class:`VariationRecorder` collects every value drawn by an enabled variation's
sampler so downstream sensitivity-analysis tooling has the input factors that
produced each episode. The recorder is explicitly attached by the caller: there
is no singleton or global lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg


class VariationRecord:
    """Per-variation slice of a :class:`VariationRecorder`.

    Bundles a variation's identity (:attr:`source_id`), the cfg driving its
    sampler at attach time (:attr:`cfg`), and the ordered sequence of samples
    drawn into the record (:attr:`samples`).
    """

    def __init__(self, source_id: str, cfg: VariationBaseCfg) -> None:
        self.source_id = source_id
        #: Cfg reference captured at :meth:`VariationRecorder.attach` time. The recorder
        #: is attached after Hydra overrides have been applied, so the cfg is
        #: treated as finalized; deep-copy if a frozen archival snapshot is needed.
        self.cfg = cfg
        #: One entry per :meth:`~isaaclab_arena.variations.sampler_base.SamplerBase.sample`
        #: call. Tensor samples are stored detached on CPU; non-tensor samples
        #: (e.g. a list returned by a categorical sampler) are stored as-is.
        self.samples: list[Any] = []

    def summary(self) -> str:
        """Return a multi-line human-readable summary of this record."""
        lines = [f"--- {self.source_id} ---", "cfg:"]
        lines.append(OmegaConf.to_yaml(OmegaConf.structured(self.cfg)).rstrip())
        lines.append(f"sample calls: {len(self.samples)}")
        if self.samples:
            first = self.samples[0]
            if isinstance(first, torch.Tensor):
                stacked_shape = (len(self.samples), *tuple(first.shape))
                lines.append(f"stacked shape: {stacked_shape}")
                lines.append(f"first call:   {first.tolist()}")
                lines.append(f"last call:    {self.samples[-1].tolist()}")
            else:
                lines.append(f"first call:   {first!r}")
                lines.append(f"last call:    {self.samples[-1]!r}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class VariationRecorder:
    """Records every sample drawn by attached variations, grouped per variation.

    Records are keyed by ``source_id`` (by convention ``"{asset}.{variation}"``).
    Serialisation is intentionally not handled here.
    """

    def __init__(self) -> None:
        self._records: dict[str, VariationRecord] = {}

    def attach(self, source_id: str, variation: VariationBase) -> None:
        """Subscribe this recorder to ``variation`` under ``source_id``.

        The listener is registered via
        :meth:`~isaaclab_arena.variations.variation_base.VariationBase.add_sample_listener`
        so it survives subsequent sampler swaps.

        Args:
            source_id: Identifier the record is stored under. By convention
                ``"{asset_name}.{variation.name}"``.
            variation: The variation to observe.
        """
        assert source_id not in self._records, (
            f"VariationRecorder: source_id '{source_id}' is already attached. "
            "Re-attaching the same variation would create two independent records and "
            "double-record subsequent samples; detach or use a different source_id instead."
        )

        record = VariationRecord(source_id=source_id, cfg=variation.cfg)
        self._records[source_id] = record

        def on_sample(sample: Any) -> None:
            if isinstance(sample, torch.Tensor):
                record.samples.append(sample.detach().cpu())
            else:
                record.samples.append(sample)

        variation.add_sample_listener(on_sample)

    def attach_to_scene(self, scene: Scene) -> None:
        """Attach every enabled variation in ``scene`` under ``"{asset}.{variation}"``."""
        for asset_name, asset_variations in scene.get_asset_variations().items():
            for variation in asset_variations:
                if not variation.enabled:
                    continue
                self.attach(f"{asset_name}.{variation.name}", variation)

    @property
    def records(self) -> list[VariationRecord]:
        """All per-variation records, in attach order."""
        return list(self._records.values())

    def summary(self) -> str:
        """Return a multi-line human-readable summary of every attached record."""
        parts = [f"VariationRecorder: {len(self._records)} record(s)"]
        for record in self._records.values():
            parts.append("")
            parts.append(record.summary())
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.summary()

    def __getitem__(self, source_id: str) -> VariationRecord:
        return self._records[source_id]

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._records
