# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the batched openpi server script, which runs in the openpi image without Arena installed."""

from __future__ import annotations

import importlib.util
import numpy as np
import pathlib
import sys
import types

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "server" / "serve_batched_policy.py"


@pytest.fixture(scope="module")
def serve_batched_policy():
    spec = importlib.util.spec_from_file_location("serve_batched_policy", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_padded_batch_size_picks_the_smallest_fitting_size(serve_batched_policy):
    assert serve_batched_policy.padded_batch_size(1, [8, 16]) == 8
    assert serve_batched_policy.padded_batch_size(8, [16, 8]) == 8
    assert serve_batched_policy.padded_batch_size(9, [8, 16]) == 16
    with pytest.raises(AssertionError):
        serve_batched_policy.padded_batch_size(17, [8, 16])


def test_group_batch_sizes_fills_the_largest_shape_then_pads_the_rest_to_the_smallest_fitting_one(
    serve_batched_policy,
):
    assert serve_batched_policy.group_batch_sizes(1, [2, 8, 20]) == [2]
    assert serve_batched_policy.group_batch_sizes(7, [2, 8, 20]) == [8]
    assert serve_batched_policy.group_batch_sizes(20, [2, 8, 20]) == [20]
    assert serve_batched_policy.group_batch_sizes(46, [2, 8, 20]) == [20, 20, 8]
    assert serve_batched_policy.group_batch_sizes(100, [8]) == [8] * 13
    with pytest.raises(AssertionError):
        serve_batched_policy.group_batch_sizes(0, [8])


def test_stack_and_unstack_round_trip_with_padding(serve_batched_policy):
    samples = [
        {"state": np.full(3, i, dtype=np.float32), "image": {"base": np.full((2, 2), i, dtype=np.uint8)}}
        for i in range(3)
    ]
    batched = serve_batched_policy.stack_samples(samples, batch_size=4, stack=np.stack)
    assert batched["state"].shape == (4, 3)
    assert batched["image"]["base"].shape == (4, 2, 2)
    assert (batched["state"][3] == batched["state"][2]).all(), "padding repeats the last sample"

    restored = serve_batched_policy.unstack_samples(batched, 3, lambda leaf, i: leaf[i])
    assert len(restored) == 3
    for i, sample in enumerate(restored):
        assert (sample["state"] == i).all()
        assert (sample["image"]["base"] == i).all()


class _FakeJax(types.SimpleNamespace):
    """The jax surface BatchedPolicy touches, backed by numpy."""

    def __init__(self):
        super().__init__(
            numpy=types.SimpleNamespace(asarray=np.asarray),
            tree=types.SimpleNamespace(map=lambda fn, tree: _tree_map(fn, tree)),
            random=types.SimpleNamespace(split=lambda key: (key + 1, key + 1000)),
            block_until_ready=lambda x: x,
        )


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    return fn(tree)


class _FakeOpenpiPolicy:
    """Mimics openpi.policies.policy.Policy's private surface for a JAX model."""

    _is_pytorch_model = False
    metadata = {"model": "fake"}

    def __init__(self):
        self._rng = 0
        self._sample_kwargs = {}
        self.sample_batch_sizes: list[int] = []
        self.single_calls = 0

    def _input_transform(self, data):
        return {"state": np.asarray(data["observation/state"], dtype=np.float32) * 2}

    def _sample_actions(self, rng, observation, **kwargs):
        state = observation["state"]
        self.sample_batch_sizes.append(state.shape[0])
        # One action row per batch entry: (batch, horizon=2, action_dim=3), each equal to the state.
        return np.repeat(state[:, None, :], 2, axis=1)

    def _output_transform(self, data):
        return {"actions": data["actions"] + 1}

    def infer(self, obs):
        self.single_calls += 1
        return {"actions": np.zeros((2, 3))}


@pytest.fixture
def batched_policy(monkeypatch, serve_batched_policy):
    monkeypatch.setitem(sys.modules, "jax", _FakeJax())
    model_module = types.SimpleNamespace(Observation=types.SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setitem(sys.modules, "openpi", types.SimpleNamespace(models=types.SimpleNamespace(model=model_module)))
    monkeypatch.setitem(sys.modules, "openpi.models", types.SimpleNamespace(model=model_module))
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_module)
    inner = _FakeOpenpiPolicy()
    return serve_batched_policy.BatchedPolicy(inner, batch_sizes=[4]), inner


def test_single_observation_requests_pass_through(batched_policy):
    policy, inner = batched_policy
    policy.infer({"observation/state": np.ones(3)})
    assert inner.single_calls == 1
    assert inner.sample_batch_sizes == []


def test_batched_request_runs_the_model_once_and_returns_one_entry_per_observation(batched_policy):
    policy, inner = batched_policy
    observations = [{"observation/state": np.full(3, i, dtype=np.float32)} for i in range(3)]

    response = policy.infer({"batch": observations})

    assert inner.sample_batch_sizes == [4], "three observations pad to a batch of four"
    outputs = response["batch"]
    assert len(outputs) == 3
    for i, output in enumerate(outputs):
        assert output["actions"].shape == (2, 3)
        assert (output["actions"] == 2 * i + 1).all(), "input and output transforms apply per observation"
    assert response["policy_timing"]["batch_size"] == 3


def test_requests_above_the_largest_shape_are_split(batched_policy):
    policy, inner = batched_policy
    observations = [{"observation/state": np.full(3, i, dtype=np.float32)} for i in range(6)]
    response = policy.infer({"batch": observations})
    assert inner.sample_batch_sizes == [4, 4], "the remaining 2 observations pad to the only shape, 4"
    assert [int(o["actions"][0, 0]) for o in response["batch"]] == [2 * i + 1 for i in range(6)]


def test_compile_batch_sizes_runs_the_model_once_per_shape_on_fake_observations(monkeypatch, serve_batched_policy):
    monkeypatch.setitem(sys.modules, "jax", _FakeJax())
    inner = _FakeOpenpiPolicy()
    policy = serve_batched_policy.BatchedPolicy(inner, batch_sizes=[8, 2])
    policy.compile_batch_sizes(lambda batch_size: {"state": np.ones((batch_size, 3), np.float32)})
    assert inner.sample_batch_sizes == [2, 8], "every shape runs once, smallest first"
    assert inner.single_calls == 0


def test_metadata_advertises_batching(batched_policy):
    policy, _ = batched_policy
    assert policy.metadata == {"model": "fake", "batch_inference": True, "max_batch_size": 4, "batch_sizes": [4]}


def test_several_batch_sizes_pad_every_batch_to_the_smallest_fitting_shape(monkeypatch, serve_batched_policy):
    monkeypatch.setitem(sys.modules, "jax", _FakeJax())
    model_module = types.SimpleNamespace(Observation=types.SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setitem(sys.modules, "openpi", types.SimpleNamespace(models=types.SimpleNamespace(model=model_module)))
    monkeypatch.setitem(sys.modules, "openpi.models", types.SimpleNamespace(model=model_module))
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_module)
    inner = _FakeOpenpiPolicy()
    policy = serve_batched_policy.BatchedPolicy(inner, batch_sizes=[4, 2])

    assert policy.metadata["max_batch_size"] == 4, "the largest shape is the batch limit"
    assert policy.metadata["batch_sizes"] == [2, 4]
    observations = [{"observation/state": np.full(3, i, dtype=np.float32)} for i in range(6)]
    response = policy.infer({"batch": observations})
    assert inner.sample_batch_sizes == [4, 2], "4 observations fill a batch of 4; the remaining 2 fit the shape of 2"
    assert [int(o["actions"][0, 0]) for o in response["batch"]] == [2 * i + 1 for i in range(6)]

    inner.sample_batch_sizes.clear()
    response = policy.infer({"batch": observations[:3]})
    assert inner.sample_batch_sizes == [4], "3 observations pad to 4 rather than running 2 + 2"
    assert [int(o["actions"][0, 0]) for o in response["batch"]] == [2 * i + 1 for i in range(3)]
