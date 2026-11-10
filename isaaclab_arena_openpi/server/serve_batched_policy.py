# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve an openpi policy over websocket with batched inference.

Runs inside the openpi server image (see ``isaaclab_arena_openpi/docker/run_openpi_server.sh``), next to
openpi's own ``scripts/serve_policy.py`` whose arguments it accepts unchanged plus ``--max-batch-size`` and
``--batch-sizes``.

The wire format stays openpi's: a request that is one observation dict gets one response dict. A request of
the form ``{"batch": [obs_0, obs_1, ...]}`` runs the model once for all observations and returns
``{"batch": [out_0, out_1, ...]}`` with one openpi response dict per observation. The server advertises
``batch_inference`` and ``max_batch_size`` in the metadata it sends on connect, which is how clients
discover the batched form.

The JAX model is compiled once per batch shape, so a batch is padded (by repeating its last observation)
to one of a few sizes: the powers of two up to ``--max-batch-size`` by default, or exactly the sizes given
with ``--batch-sizes`` (for example ``--batch-sizes 8 16``), each compiled on its first use.
"""

from __future__ import annotations

import dataclasses
import logging
import numpy as np
import pathlib
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

BATCH_KEY = "batch"


def padded_batch_size(num_samples: int, max_batch_size: int) -> int:
    """Return the batch size a group of ``num_samples`` observations is padded to.

    The JAX model is compiled once per batch shape, so batches are padded to the next power of two
    (capped at ``max_batch_size``) to keep the number of compilations small.
    """
    assert 0 < num_samples <= max_batch_size, f"expected 0 < {num_samples} <= {max_batch_size}"
    size = 1
    while size < num_samples:
        size *= 2
    return min(size, max_batch_size)


def padded_batch_size_from_sizes(num_samples: int, batch_sizes: Sequence[int]) -> int:
    """Return the smallest of the fixed ``batch_sizes`` that holds ``num_samples`` observations.

    Args:
        num_samples: Observations in the group; at most ``max(batch_sizes)``.
        batch_sizes: The batch shapes the server compiles, in any order.
    """
    fitting_sizes = [size for size in batch_sizes if size >= num_samples]
    assert 0 < num_samples and fitting_sizes, f"expected 0 < {num_samples} <= one of {sorted(batch_sizes)}"
    return min(fitting_sizes)


def stack_samples(samples: list[dict[str, Any]], batch_size: int, stack: Callable[[list], Any]) -> dict[str, Any]:
    """Stack same-structured sample dicts along a new leading axis, padding to ``batch_size`` with the last sample.

    Args:
        samples: Transformed observations with identical nested keys and leaf shapes.
        batch_size: Leading dimension of the result; at least ``len(samples)``.
        stack: Function stacking a list of leaves (for example ``np.stack``).
    """
    assert samples, "stack_samples needs at least one sample"
    assert batch_size >= len(samples), f"batch_size {batch_size} smaller than {len(samples)} samples"
    padded = samples + [samples[-1]] * (batch_size - len(samples))

    def recurse(leaves: list[Any]) -> Any:
        if isinstance(leaves[0], dict):
            return {key: recurse([leaf[key] for leaf in leaves]) for key in leaves[0]}
        return stack(leaves)

    return recurse(padded)


def unstack_samples(
    batched: dict[str, Any], num_samples: int, select: Callable[[Any, int], Any]
) -> list[dict[str, Any]]:
    """Split a batched dict back into ``num_samples`` per-sample dicts, dropping any padding.

    Args:
        batched: Nested dict whose leaves have a leading batch axis.
        num_samples: Number of real (unpadded) samples to return.
        select: Function returning sample ``i`` of a batched leaf as a numpy array.
    """

    def recurse(node: Any, index: int) -> Any:
        if isinstance(node, dict):
            return {key: recurse(value, index) for key, value in node.items()}
        return select(node, index)

    return [recurse(batched, index) for index in range(num_samples)]


class BatchedPolicy:
    """Wraps an openpi ``Policy`` so one request can carry several observations.

    Single-observation requests are forwarded unchanged. Batched requests apply the policy's input
    transforms per observation, run ``sample_actions`` once on the stacked batch and apply the output
    transforms per observation. Only JAX models are supported.

    Args:
        policy: The openpi policy to wrap.
        max_batch_size: Most observations per model call; bigger requests are split. Ignored when
            ``batch_sizes`` is given, which sets it to the largest listed size.
        batch_sizes: Fixed batch shapes to pad to, or None to pad to the powers of two up to ``max_batch_size``.
    """

    def __init__(self, policy: Any, max_batch_size: int = 32, batch_sizes: Sequence[int] | None = None) -> None:
        import jax

        assert not getattr(policy, "_is_pytorch_model", False), "BatchedPolicy supports JAX openpi policies only"
        if batch_sizes is not None:
            assert batch_sizes and all(size > 0 for size in batch_sizes), f"batch sizes must be positive: {batch_sizes}"
            max_batch_size = max(batch_sizes)
        assert max_batch_size > 0, "max_batch_size must be positive"
        self._policy = policy
        self._max_batch_size = max_batch_size
        self._batch_sizes: list[int] | None = sorted(set(batch_sizes)) if batch_sizes is not None else None
        self._jax = jax
        self._compiled_batch_sizes: set[int] = set()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._policy.metadata,
            "batch_inference": True,
            "max_batch_size": self._max_batch_size,
            "batch_sizes": self._batch_sizes,
        }

    def padded_batch_size(self, num_samples: int) -> int:
        """Return the batch shape a group of ``num_samples`` observations is padded to."""
        if self._batch_sizes is not None:
            return padded_batch_size_from_sizes(num_samples, self._batch_sizes)
        return padded_batch_size(num_samples, self._max_batch_size)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if BATCH_KEY not in obs:
            return self._policy.infer(obs)
        observations = obs[BATCH_KEY]
        assert isinstance(observations, list) and observations, "batch request must hold a non-empty list"
        outputs: list[dict[str, Any]] = []
        model_ms = 0.0
        for start in range(0, len(observations), self._max_batch_size):
            group_outputs, group_ms = self._infer_group(observations[start : start + self._max_batch_size])
            outputs.extend(group_outputs)
            model_ms += group_ms
        return {BATCH_KEY: outputs, "policy_timing": {"infer_ms": model_ms, "batch_size": len(observations)}}

    def reset(self) -> None:
        self._policy.reset()

    def _infer_group(self, observations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
        jax, jnp = self._jax, self._jax.numpy
        policy = self._policy
        from openpi.models import model as _model

        transformed = [policy._input_transform(jax.tree.map(lambda x: x, observation)) for observation in observations]
        batch_size = self.padded_batch_size(len(transformed))
        if batch_size not in self._compiled_batch_sizes:
            logging.info("Batched inference: first request with batch size %d, compiling", batch_size)
            self._compiled_batch_sizes.add(batch_size)
        inputs = stack_samples(transformed, batch_size, lambda leaves: jnp.asarray(np.stack(leaves)))
        policy._rng, sample_rng = jax.random.split(policy._rng)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        batched_outputs = {
            "state": inputs["state"],
            "actions": policy._sample_actions(sample_rng, observation, **policy._sample_kwargs),
        }
        model_ms = (time.monotonic() - start_time) * 1000

        per_sample = unstack_samples(batched_outputs, len(transformed), lambda leaf, i: np.asarray(leaf[i, ...]))
        results = []
        for sample in per_sample:
            sample = policy._output_transform(sample)
            sample["policy_timing"] = {"infer_ms": model_ms / len(transformed)}
            results.append(sample)
        return results, model_ms


def _import_serve_policy_module() -> Any:
    """Import openpi's ``scripts/serve_policy.py`` from the openpi checkout this script runs in."""
    import openpi

    scripts_dir = pathlib.Path(openpi.__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import serve_policy  # type: ignore[import-not-found]

    return serve_policy


def main() -> None:
    import tyro
    from openpi.serving import websocket_policy_server

    serve_policy = _import_serve_policy_module()

    @dataclasses.dataclass
    class Args(serve_policy.Args):
        """serve_policy.py arguments plus the batching limit."""

        max_batch_size: int = 32
        """Largest number of observations run through the model in one call; bigger requests are split.
        Batches are padded to the next power of two. Ignored when ``batch_sizes`` is given."""

        batch_sizes: tuple[int, ...] | None = None
        """Fixed batch shapes to pad every batch to, for example ``--batch-sizes 8 16``; the largest is the
        batch limit. Each shape compiles once, on first use."""

    args = tyro.cli(Args)
    policy = BatchedPolicy(
        serve_policy.create_policy(args), max_batch_size=args.max_batch_size, batch_sizes=args.batch_sizes
    )
    logging.info(
        "Serving batched policy on port %d (max batch size %d, batch sizes %s)",
        args.port,
        policy.metadata["max_batch_size"],
        policy.metadata["batch_sizes"] or "powers of two",
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
