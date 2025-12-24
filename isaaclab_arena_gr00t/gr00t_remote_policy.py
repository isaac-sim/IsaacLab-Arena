# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from gr00t.experiment.data_config import DATA_CONFIG_MAP, load_data_config
from gr00t.model.policy import Gr00tPolicy

from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena_gr00t.policy_config import Gr00tClosedloopPolicyConfig
from isaaclab_arena_gr00t.data_utils.io_utils import create_config_from_yaml


class Gr00tRemoteServerSidePolicy(ServerSidePolicy):
    """Server-side wrapper around Gr00tPolicy."""

    def __init__(self, policy_config_yaml_path: Path) -> None:
        print(f"[Gr00tRemoteServerSidePolicy] loading config from: {policy_config_yaml_path}")
        self._cfg = create_config_from_yaml(policy_config_yaml_path, Gr00tClosedloopPolicyConfig)
        print(
            "[Gr00tRemoteServerSidePolicy] config:\n"
            f"  model_path        = {self._cfg.model_path}\n"
            f"  embodiment_tag    = {self._cfg.embodiment_tag}\n"
            f"  task_mode_name    = {self._cfg.task_mode_name}\n"
            f"  data_config       = {self._cfg.data_config}\n"
            f"  action_horizon    = {self._cfg.action_horizon}\n"
            f"  action_chunk_len  = {self._cfg.action_chunk_length}\n"
            f"  pov_cam_name_sim  = {self._cfg.pov_cam_name_sim}\n"
            f"  policy_device     = {self._cfg.policy_device}"
        )
        self._policy = self._load_gr00t_policy()
        print("[Gr00tRemoteServerSidePolicy] Gr00tPolicy loaded successfully")

    def _load_gr00t_policy(self) -> Gr00tPolicy:
        print(f"[Gr00tRemoteServerSidePolicy] loading data_config={self._cfg.data_config}")
        if self._cfg.data_config in DATA_CONFIG_MAP:
            data_config = DATA_CONFIG_MAP[self._cfg.data_config]
        elif self._cfg.data_config == "unitree_g1_sim_wbc":
            data_config = load_data_config("isaaclab_arena_gr00t.data_config:UnitreeG1SimWBCDataConfig")
        else:
            raise ValueError(f"Invalid data config: {self._cfg.data_config}")

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        model_path = Path(self._cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        print(f"[Gr00tRemoteServerSidePolicy] loading checkpoint from: {model_path}")

        policy = Gr00tPolicy(
            model_path=str(model_path),
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self._cfg.embodiment_tag,
            denoising_steps=self._cfg.denoising_steps,
            device=self._cfg.policy_device,
        )
        return policy

    # ------------------------------------------------------------------ #
    # ServerSidePolicy interface
    # ------------------------------------------------------------------ #

    def get_action(
        self,
        observation: Dict[str, Any],
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        print("[Gr00tRemoteServerSidePolicy] get_action called")
        print(f"  observation keys: {list(observation.keys())}")
        if options is not None:
            print(f"  options keys: {list(options.keys())}")

        result = self._policy.get_action(observation)
        # Gr00tPolicy.get_action usually returns a dict; wrap it with empty info.
        if isinstance(result, tuple) and len(result) == 2:
            action, info = result
        else:
            action, info = result, {}

        print("[Gr00tRemoteServerSidePolicy] get_action done")
        return action, info

    def reset(self, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        print(f"[Gr00tRemoteServerSidePolicy] reset called: options={options}")
        if hasattr(self._policy, "reset"):
            self._policy.reset(options=options)
        return {}

