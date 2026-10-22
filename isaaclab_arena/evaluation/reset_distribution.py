# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure realized scene states before the first policy action of each episode."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


class ResetDistributionRecorder:
    """Stream reset samples and accumulate per-component population statistics."""

    def __init__(self, output_dir: Path, run_name: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = output_dir / "reset_distribution.json"
        if self.summary_path.exists():
            raise FileExistsError(self.summary_path)
        self.stream = (output_dir / "reset_samples.jsonl").open("x", encoding="utf-8")
        self.run_name = run_name
        self.count = 0
        self.statistics = {}
        self.identities = set()

    def capture(self, env, env_ids, *, rebuild_index: int, seed: int) -> None:
        """Copy environment-local scene state for episodes about to receive their first action."""
        ids = env_ids.tolist()
        if not ids:
            return
        state = env.scene.get_state(is_relative=True)
        for env_id in ids:
            if hasattr(env, "active_episode_mask") and not bool(env.active_episode_mask[env_id]):
                continue
            episode_index = env.get_episode_index(env_id)
            identity = (rebuild_index, env_id, episode_index)
            if identity in self.identities:
                raise ValueError(f"Duplicate reset sample: {identity}")
            sample = {}
            measurements = {}
            for kind, assets in state.items():
                sample[kind] = {}
                for name, fields in assets.items():
                    sample[kind][name] = {}
                    for field, tensor in fields.items():
                        value = tensor[env_id].detach().cpu().numpy().astype(np.float64)
                        if not np.isfinite(value).all():
                            raise ValueError(f"Non-finite reset state: {kind}/{name}/{field}")
                        sample[kind][name][field] = value.tolist()
                        # Quaternion components are retained in samples, not averaged as Euclidean values.
                        if field.endswith("pose"):
                            value = value[..., :3]
                            field = field.removesuffix("pose") + "position"
                        key = f"{kind}/{name}/{field}"
                        if key in self.statistics and self.statistics[key]["mean"].shape != value.shape:
                            raise ValueError(f"Reset state shape changed across rebuilds: {key}")
                        measurements[key] = value
            record = {
                "schema_version": 1,
                "run_name": self.run_name,
                "rebuild_index": rebuild_index,
                "seed": seed,
                "env_id": env_id,
                "episode_in_env": episode_index,
                "coordinate_frame": "environment",
                "quaternion_order": "xyzw",
                "state": sample,
            }
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
            self.stream.flush()
            self.identities.add(identity)
            self.count += 1
            for key, value in measurements.items():
                self._update(key, value)

    def _update(self, key, value):
        stats = self.statistics.get(key)
        if stats is None:
            self.statistics[key] = {
                "count": 1,
                "mean": value.copy(),
                "m2": np.zeros_like(value),
                "min": value.copy(),
                "max": value.copy(),
            }
            return
        stats["count"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        stats["m2"] += delta * (value - stats["mean"])
        stats["min"] = np.minimum(stats["min"], value)
        stats["max"] = np.maximum(stats["max"], value)

    def close(self) -> None:
        """Write statistics for captured samples, including those from an interrupted rollout."""
        if self.stream.closed:
            return
        self.stream.close()
        fields = {}
        for key, stats in self.statistics.items():
            fields[key] = {name: stats[name].tolist() for name in ("mean", "min", "max")}
            fields[key].update(
                count=stats["count"],
                std=np.sqrt(np.maximum(stats["m2"], 0) / stats["count"]).tolist(),
            )
        summary = {
            "schema_version": 1,
            "run_name": self.run_name,
            "sample_count": self.count,
            "coordinate_frame": "environment",
            "std_convention": "population",
            "sampling_point": "before first policy action after reset",
            "orientation_statistics": "omitted; raw xyzw quaternions are in reset_samples.jsonl",
            "fields": fields,
        }
        with self.summary_path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
