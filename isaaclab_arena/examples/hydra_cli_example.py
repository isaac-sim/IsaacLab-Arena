# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal Hydra example for CLI parameter experimentation.

This script has **no** Isaac Sim dependency — it is meant purely as a sandbox
for playing with Hydra's command-line override syntax before wiring it into a
real pipeline.

Run from the container (or any env with ``hydra-core`` installed):

.. code-block:: bash

    # 1. Print the default resolved config.
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_cli_example.py

    # 2. Override a leaf value using dotted paths.
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_cli_example.py \\
        training.lr=1e-4 training.epochs=50

    # 3. Override a nested field and a list element.
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_cli_example.py \\
        model.hidden_sizes='[256,256,128]' model.activation=gelu

    # 4. Sweep (multirun): runs the script once per combination.
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_cli_example.py -m \\
        training.lr=1e-3,1e-4 model.activation=relu,gelu

    # 5. Show the resolved config without running the function body.
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_cli_example.py --cfg job

    # 6. Add a new field that isn't in the schema (Hydra rejects this by
    #    default with a structured config — use the ``+`` prefix to add it).
    /isaac-sim/python.sh isaaclab_arena/examples/hydra_cli_example.py \\
        +experiment.note=baseline

Hydra will create a ``outputs/<date>/<time>/`` directory per run containing
the resolved config and any logs. That directory can be customized with
``hydra.run.dir=...``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Dummy model hyperparameters."""

    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Dummy training hyperparameters."""

    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    optimizer: str = "adam"


@dataclass
class ExperimentConfig:
    """Top-level config composed of nested sub-configs.

    Using a structured config (rather than plain YAML) gives us type checking
    on overrides: ``training.lr=abc`` will be rejected at parse time because
    ``lr`` is typed as ``float``.
    """

    name: str = MISSING
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


cs = ConfigStore.instance()
cs.store(name="experiment_schema", node=ExperimentConfig)


@hydra.main(version_base=None, config_name="experiment_schema")
def main(cfg: DictConfig) -> None:
    # ``cfg.name`` is ``MISSING`` by default; require the user to pass it on
    # the CLI (e.g. ``name=my_run``) so we exercise Hydra's missing-value check.
    if OmegaConf.is_missing(cfg, "name"):
        cfg.name = "default_run"

    print("=== Resolved config ===")
    print(OmegaConf.to_yaml(cfg))

    print("=== Accessing values ===")
    print(f"name         = {cfg.name}")
    print(f"seed         = {cfg.seed}")
    print(f"lr           = {cfg.training.lr}")
    print(f"epochs       = {cfg.training.epochs}")
    print(f"hidden_sizes = {list(cfg.model.hidden_sizes)}")
    print(f"activation   = {cfg.model.activation}")


if __name__ == "__main__":
    main()
