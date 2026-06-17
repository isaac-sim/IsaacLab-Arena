# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase, VariationBaseCfg


class VariationRecord:
    """Per-variation record configuration and samples."""

    def __init__(self, name: str, cfg: VariationBaseCfg) -> None:
        self.name = name
        self.cfg = cfg
        self.samples: list[Any] = []

    def _header_lines(self) -> list[str]:
        """Return the shared preamble (identity, cfg, sample-call count) for renderers."""
        # Add the title for this variation
        lines = [f"--- {self.name} ---", "cfg:"]
        # Print the Cfg
        lines.append(OmegaConf.to_yaml(OmegaConf.structured(self.cfg)).rstrip())
        # Print basic information about the samples
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
            # Print the first and last sample
            lines.append(f"first call:   {self._format_sample(self.samples[0])}")
            lines.append(f"last call:    {self._format_sample(self.samples[-1])}")
        return "\n".join(lines)

    def details(self) -> str:
        """Return a multi-line human-readable view of this record, listing every sample."""
        lines = self._header_lines()
        # Print every sample
        for i, sample in enumerate(self.samples):
            lines.append(f"call {i}: {self._format_sample(sample)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class VariationRecorder:
    """Records samples drawn by attached variations."""

    def __init__(self) -> None:
        # Records are keyed by: "{asset_name}.{variation_name}"
        self.records: dict[str, VariationRecord] = {}

    def __getitem__(self, key: str) -> VariationRecord:
        """Return the record stored under "{asset_name}.{variation_name}"."""
        return self.records[key]

    def __contains__(self, key: str) -> bool:
        """Whether a record is stored under "{asset_name}.{variation_name}"."""
        return key in self.records

    def attach(self, variations: dict[str, list[VariationBase]]) -> None:
        """Attach every enabled variation in ``variations`` under "{asset_name}.{variation_name}"."""
        for asset_name, asset_variations in variations.items():
            for variation in asset_variations:
                if not variation.enabled:
                    continue
                variation_key = f"{asset_name}.{variation.name}"
                assert (
                    variation_key not in self.records
                ), f"VariationRecorder: asset_name '{variation_key}' is already attached."

                # Create a record for the variation
                record = VariationRecord(name=variation_key, cfg=variation.cfg)
                self.records[variation_key] = record

                def on_sample(sample: Any, record: VariationRecord = record) -> None:
                    if isinstance(sample, torch.Tensor):
                        record.samples.append(sample.detach().cpu())
                    else:
                        record.samples.append(sample)

                variation.add_sample_listener(on_sample)

    def _render(self, render_record: Callable[[VariationRecord], str]) -> str:
        """Join ``render_record`` applied to every attached record under a shared header."""
        parts = [f"VariationRecorder: {len(self.records)} record(s)"]
        for record in self.records.values():
            parts.append("")
            parts.append(render_record(record))
        return "\n".join(parts)

    def summary(self) -> str:
        """Return a multi-line human-readable summary of every attached record (first/last sample)."""
        return self._render(VariationRecord.summary)

    def details(self) -> str:
        """Return a multi-line human-readable view of every attached record, listing all samples."""
        return self._render(VariationRecord.details)
