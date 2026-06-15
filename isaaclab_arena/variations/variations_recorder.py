# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory recording of variation samples.

A ``VariationRecorder`` collects every value drawn by an enabled variation's sampler so
downstream sensitivity-analysis tooling has the input factors that produced each episode.
The recorder is explicitly attached by the caller; there is no singleton or global lookup.
"""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg


class VariationRecord:
    """Per-variation slice of a ``VariationRecorder``: a variation's identity, cfg, and samples."""

    def __init__(self, source_id: str, cfg: VariationBaseCfg) -> None:
        self.source_id = source_id

        self.cfg = cfg
        """Cfg driving the sampler, captured at attach time (after Hydra overrides, so treated as finalized)."""

        self.samples: list[Any] = []
        """One entry per ``sample()`` call. Tensors stored detached on CPU; other values as-is."""

    def _header_lines(self) -> list[str]:
        """Return the shared preamble (identity, cfg, sample-call count) for renderers."""
        lines = [f"--- {self.source_id} ---", "cfg:"]
        lines.append(OmegaConf.to_yaml(OmegaConf.structured(self.cfg)).rstrip())
        lines.append(f"sample calls: {len(self.samples)}")
        if self.samples and isinstance(self.samples[0], torch.Tensor):
            stacked_shape = (len(self.samples), *tuple(self.samples[0].shape))
            lines.append(f"stacked shape: {stacked_shape}")
        return lines

    @staticmethod
    def _format_sample(sample: Any) -> str:
        """Render a single sample value (tensors as nested lists, others via ``repr``)."""
        return f"{sample.tolist()}" if isinstance(sample, torch.Tensor) else f"{sample!r}"

    def summary(self) -> str:
        """Return a multi-line human-readable summary of this record (first/last sample only)."""
        lines = self._header_lines()
        if self.samples:
            lines.append(f"first call:   {self._format_sample(self.samples[0])}")
            lines.append(f"last call:    {self._format_sample(self.samples[-1])}")
        return "\n".join(lines)

    def details(self) -> str:
        """Return a multi-line human-readable view of this record, listing every sample."""
        lines = self._header_lines()
        for i, sample in enumerate(self.samples):
            lines.append(f"call {i}: {self._format_sample(sample)}")
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

    def attach(self, variations: dict[str, list[VariationBase]]) -> None:
        """Attach every enabled variation in ``variations`` under ``"{asset}.{variation}"``.

        Used by ``ArenaEnvBuilder`` to attach variations sourced from both the scene and
        the embodiment. Disabled variations are skipped.
        """
        for asset_name, asset_variations in variations.items():
            for variation in asset_variations:
                if not variation.enabled:
                    continue
                self._attach(f"{asset_name}.{variation.name}", variation)

    def attach_to_scene(self, scene: Scene) -> None:
        """Attach every enabled variation in ``scene`` under ``"{asset}.{variation}"``."""
        self.attach(scene.get_asset_variations())

    def _attach(self, source_id: str, variation: VariationBase) -> None:
        """Subscribe this recorder to ``variation`` under ``source_id``.

        Registers the listener through ``variation.add_sample_listener`` so it survives
        subsequent sampler swaps.

        Args:
            source_id: Identifier the record is stored under, by convention ``"{asset}.{variation}"``.
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

    @property
    def records(self) -> list[VariationRecord]:
        """All per-variation records, in attach order."""
        return list(self._records.values())

    def _render(self, render_record: Callable[[VariationRecord], str]) -> str:
        """Join ``render_record`` applied to every attached record under a shared header."""
        parts = [f"VariationRecorder: {len(self._records)} record(s)"]
        for record in self._records.values():
            parts.append("")
            parts.append(render_record(record))
        return "\n".join(parts)

    def summary(self) -> str:
        """Return a multi-line human-readable summary of every attached record (first/last sample)."""
        return self._render(VariationRecord.summary)

    def details(self) -> str:
        """Return a multi-line human-readable view of every attached record, listing all samples."""
        return self._render(VariationRecord.details)

    def __str__(self) -> str:
        return self.summary()

    def __getitem__(self, source_id: str) -> VariationRecord:
        return self._records[source_id]

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._records
