import sys
import types

import numpy as np
import torch


def _install_lightweight_stubs() -> None:
    """Install minimal module stubs so pure-logic tests can import VLN code."""

    def _identity_configclass(cls):
        return cls

    class _RecorderTerm:
        def __init__(self, cfg=None, env=None):
            self.cfg = cfg
            self._env = env

    class _RecorderTermCfg:
        class_type = None

    class _ManagerBasedEnv:
        pass

    class _DummyClientSidePolicy:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def add_remote_args_to_parser(parser):
            return parser

    class _DummyProtocol:
        default_duration = 1.0

    class _DummyRemotePolicyConfig:
        pass

    class _DummyRslRlVecEnvWrapper:
        pass

    class _DummyOnPolicyRunner:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.envs", types.ModuleType("isaaclab.envs"))
    sys.modules.setdefault("isaaclab.managers", types.ModuleType("isaaclab.managers"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules.setdefault("isaaclab.utils.io", types.ModuleType("isaaclab.utils.io"))
    sys.modules.setdefault("isaaclab_rl", types.ModuleType("isaaclab_rl"))
    sys.modules.setdefault("isaaclab_rl.rsl_rl", types.ModuleType("isaaclab_rl.rsl_rl"))
    sys.modules.setdefault("rsl_rl", types.ModuleType("rsl_rl"))
    sys.modules.setdefault("rsl_rl.runners", types.ModuleType("rsl_rl.runners"))

    env_mod = types.ModuleType("isaaclab.envs.manager_based_rl_env")
    env_mod.ManagerBasedEnv = _ManagerBasedEnv
    sys.modules["isaaclab.envs.manager_based_rl_env"] = env_mod

    recorder_mod = types.ModuleType("isaaclab.managers.recorder_manager")
    recorder_mod.RecorderTerm = _RecorderTerm
    recorder_mod.RecorderTermCfg = _RecorderTermCfg
    sys.modules["isaaclab.managers.recorder_manager"] = recorder_mod

    utils_mod = sys.modules["isaaclab.utils"]
    utils_mod.configclass = _identity_configclass

    io_mod = sys.modules["isaaclab.utils.io"]
    io_mod.load_yaml = lambda _path: {"device": "cpu"}

    rsl_wrapper_mod = sys.modules["isaaclab_rl.rsl_rl"]
    rsl_wrapper_mod.RslRlVecEnvWrapper = _DummyRslRlVecEnvWrapper

    runners_mod = sys.modules["rsl_rl.runners"]
    runners_mod.OnPolicyRunner = _DummyOnPolicyRunner

    client_policy_mod = types.ModuleType("isaaclab_arena.policy.client_side_policy")
    client_policy_mod.ClientSidePolicy = _DummyClientSidePolicy
    sys.modules["isaaclab_arena.policy.client_side_policy"] = client_policy_mod

    protocol_mod = types.ModuleType("isaaclab_arena.remote_policy.action_protocol")
    protocol_mod.VlnVelocityActionProtocol = _DummyProtocol
    sys.modules["isaaclab_arena.remote_policy.action_protocol"] = protocol_mod

    config_mod = types.ModuleType("isaaclab_arena.remote_policy.remote_policy_config")
    config_mod.RemotePolicyConfig = _DummyRemotePolicyConfig
    sys.modules["isaaclab_arena.remote_policy.remote_policy_config"] = config_mod


_install_lightweight_stubs()

from isaaclab_arena.metrics.vln_metrics import OracleSuccessMetric, PathLengthMetric, SPLMetric, SuccessMetric
from isaaclab_arena.policy.vln.vln_vlm_locomotion_policy import VlnVlmLocomotionPolicy


def test_success_and_spl_require_stop_signal():
    gt_waypoints = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    success_radius = [0.5]

    # Episode reaches the goal region but never emits STOP.
    no_stop_episode = [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0, 0.0],
                [1.1, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    ]
    success_metric = SuccessMetric(gt_waypoints, success_radius, require_stop_signal=True)
    spl_metric = SPLMetric(gt_waypoints, success_radius, require_stop_signal=True)
    assert success_metric.compute_metric_from_recording(no_stop_episode) == 0.0
    assert spl_metric.compute_metric_from_recording(no_stop_episode) == 0.0

    # Episode emits STOP near the goal and should count as successful.
    stop_success_episode = [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    ]
    assert success_metric.compute_metric_from_recording(stop_success_episode) == 1.0

    # Same shortest path (1m), but a longer actual path (3m) before STOP.
    stop_long_path_episode = [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    ]
    assert np.isclose(spl_metric.compute_metric_from_recording(stop_long_path_episode), 1.0 / 3.0)


def test_oracle_success_uses_best_trajectory_point_without_stop():
    gt_waypoints = [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    success_radius = [0.5]
    oracle_metric = OracleSuccessMetric(gt_waypoints, success_radius)

    # The agent reaches the goal area once, then walks away and never emits STOP.
    pass_then_leave_episode = [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    ]
    assert oracle_metric.compute_metric_from_recording(pass_then_leave_episode) == 1.0

    # The agent never gets close enough to the goal region.
    miss_episode = [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    ]
    assert oracle_metric.compute_metric_from_recording(miss_episode) == 0.0


def test_spl_prefers_dataset_geodesic_distance_when_provided():
    gt_waypoints = [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    success_radius = [1.1]
    spl_metric = SPLMetric(
        gt_waypoints,
        success_radius,
        shortest_path_distance_per_episode=[2.0],
        require_stop_signal=True,
    )

    stop_success_episode = [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    ]
    assert np.isclose(spl_metric.compute_metric_from_recording(stop_success_episode), 1.0)


def test_path_length_ignores_stop_flag_column():
    metric = PathLengthMetric()
    episode = [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
                [6.0, 8.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    ]
    assert metric.compute_metric_from_recording(episode) == 10.0


def test_low_level_warmup_does_not_step_environment(monkeypatch):
    class DummyVecEnv:
        def __init__(self):
            self.unwrapped = types.SimpleNamespace(device="cpu")
            self.step_calls = 0

        def get_observations(self):
            return {"policy": torch.zeros((1, 69), dtype=torch.float32)}

        def step(self, _actions):
            self.step_calls += 1
            raise AssertionError("Warmup should not step the environment")

    class DummyRunner:
        def __init__(self, vec_env, _cfg, log_dir=None, device="cpu"):
            self._vec_env = vec_env

        def load(self, _checkpoint_path):
            return None

        def get_inference_policy(self, device="cpu"):
            def _policy(obs_td):
                return torch.zeros((1, 19), dtype=torch.float32)

            return _policy

    monkeypatch.setattr("isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.load_yaml", lambda _path: {"device": "cpu"})
    monkeypatch.setattr("isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.OnPolicyRunner", DummyRunner)
    monkeypatch.setattr("isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.RslRlVecEnvWrapper", DummyVecEnv)

    dummy_vec_env = DummyVecEnv()
    dummy_policy = types.SimpleNamespace(
        _ll_agent_cfg="unused.yaml",
        _ll_checkpoint_path="unused.pt",
        _device="cpu",
        action_dim=3,
        _vel_cmd_indices=(9, 12),
        _warmup_steps=5,
        _ll_vec_env=None,
        _ll_policy=None,
        _ll_obs_td=None,
    )

    VlnVlmLocomotionPolicy._load_low_level_policy(dummy_policy, dummy_vec_env)

    assert dummy_vec_env.step_calls == 0
    assert dummy_policy._ll_vec_env is dummy_vec_env
    assert dummy_policy._ll_policy is not None
