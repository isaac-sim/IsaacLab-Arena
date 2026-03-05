# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utility to load a pre-trained NaVILA low-level locomotion policy.

The low-level policy is an RSL-RL PPO policy trained to convert velocity
commands ``[vx, vy, yaw_rate]`` into joint-position targets for the H1
humanoid.  This module:

  1. Wraps the Isaac Lab environment for RSL-RL compatibility.
  2. Loads the PPO checkpoint.
  3. Returns a :class:`VLNEnvWrapper` that provides a high-level interface
     (velocity commands in, camera observations out).

Usage::

    env = arena_builder.make_registered()
    env = load_navila_low_level_policy(env, args_cli)
    obs, info = env.reset()
    # Now env.step() accepts velocity commands [vx, vy, yaw_rate]
"""

from __future__ import annotations

import os
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_arena.policy.vln.vln_env_wrapper import VLNEnvWrapper


def load_navila_low_level_policy(
    env,
    log_root_path: str,
    agent_cfg_yaml: str,
    policy_run_name: str,
    policy_checkpoint_id: int = 0,
    task_name: str = "h1",
    max_length: int = 10_000,
    high_level_obs_key: str = "camera_obs",
    history_length: int = 1,
) -> VLNEnvWrapper:
    """Load a pre-trained NaVILA locomotion policy and wrap the environment.

    Args:
        env: Isaac Lab ManagerBasedRLEnv (unwrapped).
        log_root_path: Root directory of RSL-RL training logs.
        agent_cfg_yaml: Path to the ``agent.yaml`` used during training.
        policy_run_name: Run folder name under the log root.
        policy_checkpoint_id: Checkpoint index to load.
        task_name: Robot identifier (``"h1"``, ``"g1"``, ``"go2"``).
        max_length: Maximum number of low-level steps per episode.
        high_level_obs_key: Observation key for the camera group.
        history_length: Number of proprioceptive history frames.

    Returns:
        A :class:`VLNEnvWrapper` ready for VLN evaluation.
    """
    # 1) Wrap for RSL-RL
    vec_env = RslRlVecEnvWrapper(env)

    # 2) Load agent config
    agent_cfg_dict = load_yaml(agent_cfg_yaml)

    # 3) Resolve checkpoint path
    log_dir = os.path.join(
        log_root_path,
        "rsl_rl",
        agent_cfg_dict.get("experiment_name", "default"),
        policy_run_name,
    )
    ckpt_path = os.path.join(log_dir, "models", f"model_{policy_checkpoint_id}.pt")
    print(f"[rslrl_loader] Loading checkpoint from: {ckpt_path}")

    # 4) Create RSL-RL runner and load the checkpoint
    device = agent_cfg_dict.get("device", "cuda")
    runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    # 5) Wrap for VLN
    vln_env = VLNEnvWrapper(
        env=vec_env,
        low_level_policy=policy,
        task_name=task_name,
        max_length=max_length,
        high_level_obs_key=high_level_obs_key,
        use_history_wrapper=False,  # standard RslRlVecEnvWrapper
    )
    return vln_env
