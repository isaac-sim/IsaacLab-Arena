# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batch robots sharing a registered policy and assemble their action columns."""

import gymnasium as gym
import numpy as np
import torch
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class MultiRobotPolicyCfg(PolicyCfg):
    """Configure named policies and assign robot instance keys to them."""

    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Policy instance names mapped to registered type names and primitive parameters."""

    assignments: dict[str, str] = field(default_factory=dict)
    """Robot scene keys mapped to policy instance names."""


@register_policy
class MultiRobotPolicy(PolicyBase[MultiRobotPolicyCfg]):
    """Call each inner policy once using robot-major rows."""

    name = "multi_robot"

    def __init__(self, config: MultiRobotPolicyCfg):
        super().__init__(config)
        assert config.assignments, "Assign at least one robot to an inner policy"
        assert set(config.assignments.values()) == set(config.policies), "Every policy must have assigned robots"
        registry = PolicyRegistry()
        self.policies = {}
        with ExitStack() as cleanup:
            for name, definition in config.policies.items():
                assert (
                    set(definition) <= {"type", "params"} and "type" in definition
                ), "Inner policy needs type and params"
                assert definition["type"] != self.name, "Nested multi-robot policies are not supported"
                params = definition.get("params", {})
                assert isinstance(params, dict) and _primitive(
                    params
                ), "Inner policy parameters must be primitive values"
                policy_type = registry.get_policy(definition["type"])
                self.policies[name] = policy_type(registry.get_policy_cfg_type(policy_type)(**params))
                cleanup.callback(self.policies[name].close)
            cleanup.pop_all()
        self._layouts = None
        self._num_envs = None
        self._env = None

    def _bind(self, env):
        """Resolve action columns and construct each inner policy's environment view."""
        manager = env.unwrapped.action_manager
        layouts = {key: [] for key in self.config.assignments}
        column = 0
        for name, width in zip(manager.active_terms, manager.action_term_dim):
            term = manager.get_term(name)
            owner = term.cfg.asset_name
            assert owner in layouts, f"Action term '{name}' has no policy assignment for '{owner}'"
            layouts[owner].append((name, slice(column, column + width)))
            column += width
        assert column == env.action_space.shape[-1], "Action manager widths must cover the action space"
        assert all(layouts.values()), "Every assigned robot must own at least one action term"
        self._layouts = layouts
        self._num_envs = env.unwrapped.num_envs
        self._env = env
        self._views = {}
        for name in self.policies:
            keys = [key for key, assigned in self.config.assignments.items() if assigned == name]
            manager_view = _RobotActionManager(manager, keys, layouts)
            self._views[name] = (keys, _RobotEnvView(env.unwrapped, manager_view, len(keys)))

    def get_action(self, env, observation):
        """Batch each policy's robot observations and scatter its returned actions."""
        if self._layouts is None:
            self._bind(env)
        assert env is self._env, "Construct a new composite policy when the environment is rebuilt"
        output = torch.empty(env.action_space.shape, device=env.unwrapped.device)
        for name, policy in self.policies.items():
            keys, view = self._views[name]
            robot_observations = [_robot_observation(observation, key, tuple(self.config.assignments)) for key in keys]
            observations = _stack_observations(robot_observations, self._num_envs)
            actions = policy.get_action(view, observations)
            expected = (view.num_envs, view.action_manager.total_action_dim)
            assert (
                isinstance(actions, torch.Tensor) and tuple(actions.shape) == expected
            ), f"Policy '{name}' must return actions shaped {expected}"
            for key, rows in zip(keys, actions.split(self._num_envs)):
                term_actions = rows.split(view.action_manager.action_term_dim, dim=-1)
                for (_, columns), values in zip(self._layouts[key], term_actions):
                    output[:, columns] = values
        return output

    def reset(self, env_ids=None):
        """Expand environment indices to the corresponding rows of every shared policy."""
        if env_ids is None:
            for policy in self.policies.values():
                policy.reset(None)
            return
        assert self._num_envs is not None, "Indexed reset requires the first action call to establish row counts"
        assert env_ids.ndim == 1, "Reset indices must be a one-dimensional tensor"
        assert bool(((env_ids >= 0) & (env_ids < self._num_envs)).all()), "Reset indices are out of range"
        for name, policy in self.policies.items():
            count = len(self._views[name][0])
            policy.reset(torch.cat([env_ids + index * self._num_envs for index in range(count)]))

    def set_task_description(self, task_description):
        """Send the mission description to every inner policy."""
        super().set_task_description(task_description)
        for policy in self.policies.values():
            policy.set_task_description(task_description)
        return task_description

    def close(self):
        """Close each inner policy once."""
        with ExitStack() as cleanup:
            for policy in reversed(self.policies.values()):
                cleanup.callback(policy.close)

    @property
    def is_remote(self):
        """Report whether any inner policy uses a remote service."""
        return any(policy.is_remote for policy in self.policies.values())

    def has_length(self):
        """Require inner policies to agree whether they replay recorded actions."""
        lengths = {policy.has_length() for policy in self.policies.values()}
        assert len(lengths) == 1, "Inner policies must agree whether they have a recorded length"
        return lengths.pop()

    def length(self):
        """Return the common recording length, rejecting disagreement."""
        lengths = {policy.length() for policy in self.policies.values()}
        assert len(lengths) == 1, "Inner policies must agree on recorded length"
        return lengths.pop()


class _RobotEnvView:
    """Expose only the robot batch and action interface used by inner policies."""

    def __init__(self, env, action_manager, robot_count):
        self.num_envs = env.num_envs * robot_count
        self.device = env.device
        self.action_manager = action_manager
        lows = [
            np.concatenate([env.single_action_space.low[columns] for _, columns in layout])
            for layout in action_manager._layouts
        ]
        highs = [
            np.concatenate([env.single_action_space.high[columns] for _, columns in layout])
            for layout in action_manager._layouts
        ]
        assert all(
            np.array_equal(value, lows[0]) for value in lows
        ), "Shared action spaces must have equal lower bounds"
        assert all(
            np.array_equal(value, highs[0]) for value in highs
        ), "Shared action spaces must have equal upper bounds"
        self.single_action_space = gym.spaces.Box(lows[0], highs[0])
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.unwrapped = self


class _RobotActionManager:
    """Present corresponding action terms stacked over robots, then environments."""

    def __init__(self, manager, keys, layouts):
        self._manager = manager
        self._layouts = [layouts[key] for key in keys]
        self.active_terms = [_strip_prefix(name, keys[0]) for name, _ in self._layouts[0]]
        self.action_term_dim = [columns.stop - columns.start for _, columns in self._layouts[0]]
        for key, layout in zip(keys, self._layouts):
            assert [
                _strip_prefix(name, key) for name, _ in layout
            ] == self.active_terms, "Shared policies need matching action terms"
            assert [
                columns.stop - columns.start for _, columns in layout
            ] == self.action_term_dim, "Shared policies need matching action widths"
        self.total_action_dim = sum(self.action_term_dim)

    @property
    def action(self):
        """Read current actions in robot-major row order."""
        return self._robot_rows(self._manager.action)

    @property
    def prev_action(self):
        """Read previous actions in robot-major row order."""
        return self._robot_rows(self._manager.prev_action)

    def _robot_rows(self, actions):
        """Select owned columns and concatenate robot batches."""
        return torch.cat(
            [torch.cat([actions[:, columns] for _, columns in layout], dim=-1) for layout in self._layouts]
        )

    def get_term(self, name):
        """Expose the corresponding raw and processed action terms across robots."""
        index = self.active_terms.index(name)
        terms = [self._manager.get_term(layout[index][0]) for layout in self._layouts]
        return SimpleNamespace(
            raw_actions=torch.cat([term.raw_actions for term in terms]),
            processed_actions=torch.cat([term.processed_actions for term in terms]),
        )


def _strip_prefix(name, key):
    """Restore the term name expected by a single-robot policy."""
    return name.removeprefix(f"{key}_") if key != "robot" else name


def _robot_observation(observation, key, keys):
    """Select one robot's groups and cameras using the longest matching prefix."""

    def owner(name):
        matches = [other for other in keys if other != "robot" and name.startswith(f"{other}_")]
        return max(matches, key=len) if matches else "robot"

    result = {}
    for name, value in observation.items():
        if name == "camera_obs":
            cameras = {_strip_prefix(camera, key): image for camera, image in value.items() if owner(camera) == key}
            if cameras:
                result[name] = cameras
        elif owner(name) == key:
            result[_strip_prefix(name, key)] = value
    assert result, f"No observation groups found for robot '{key}'"
    return result


def _stack_observations(values, num_envs):
    """Stack matching nested observation dictionaries in robot-major row order."""
    first = values[0]
    if isinstance(first, dict):
        assert all(
            isinstance(value, dict) and set(value) == set(first) for value in values
        ), "Shared policies need matching observation groups and terms"
        return {name: _stack_observations([value[name] for value in values], num_envs) for name in first}
    assert all(
        isinstance(value, torch.Tensor) and value.shape[1:] == first.shape[1:] for value in values
    ), "Shared policies need matching observation tensor shapes"
    assert all(value.shape[0] == num_envs for value in values), "Observation rows must match the environment count"
    return torch.cat(values, dim=0)


def _primitive(value):
    """Check whether policy parameters contain only serializable scalar containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_primitive(child) for child in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive(child) for key, child in value.items())
    return False
