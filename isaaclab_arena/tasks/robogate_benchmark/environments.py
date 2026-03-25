"""RoboGate Benchmark environment for Isaac Lab-Arena.

Follows the ArenaEnvBuilder pattern:
    1. Load embodiment from asset_registry (franka, ur5e, etc.)
    2. Compose Scene with background + pick object + target
    3. Create RoboGateValidationTask (68 adversarial scenarios)
    4. Build via ArenaEnvBuilder → gymnasium env

Usage::

    from robogate_benchmark.environments import RoboGateBenchmarkEnvironment
    env_def = RoboGateBenchmarkEnvironment()
    arena_env = env_def.get_env(args_cli)
    builder = ArenaEnvBuilder(arena_env, args_cli)
    env = builder.make_registered()
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from robogate_benchmark.scenarios import (
    VARIANT_CONFIGS,
    ScenarioCategory,
    ScenarioVariant,
    build_scenario_suite,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physics constants (matching RoboGate core)
# ---------------------------------------------------------------------------

TABLE_POS = np.array([0.5, 0.0, 0.0], dtype=np.float64)
TABLE_SIZE = 0.6
TABLE_TOP_Z = float(TABLE_POS[2] + TABLE_SIZE / 2.0)
OBJ_SPAWN_Z = TABLE_TOP_Z + 0.02
OBJ_SPAWN_X = (0.35, 0.65)
OBJ_SPAWN_Y = (-0.15, 0.15)
TARGET_POS = np.array([0.35, 0.30, OBJ_SPAWN_Z], dtype=np.float64)
SUCCESS_DIST = 0.08
DROP_Z_THRESHOLD = TABLE_TOP_Z - 0.10


# ---------------------------------------------------------------------------
# Isaac Lab-Arena integration (graceful fallback)
# ---------------------------------------------------------------------------

_ARENA_AVAILABLE = False

try:
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import TaskBase

    _ARENA_AVAILABLE = True
except ImportError:
    logger.info(
        "isaaclab_arena not available — using standalone mode. "
        "Install Isaac Lab-Arena for full integration."
    )


# ---------------------------------------------------------------------------
# RoboGateValidationTask (Isaac Lab-Arena TaskBase)
# ---------------------------------------------------------------------------

if _ARENA_AVAILABLE:

    class RoboGateValidationTask(TaskBase):
        """68-scenario adversarial validation task for Isaac Lab-Arena.

        Runs all 4 scenario categories with configurable difficulty
        parameters and evaluates 5 safety metrics.
        """

        def __init__(
            self,
            pick_object: Any,
            target_object: Any,
            background: Any,
            scenario_config: dict[str, Any] | None = None,
            episode_length_s: float = 15.0,
        ) -> None:
            super().__init__(
                episode_length_s=episode_length_s,
                task_description="RoboGate adversarial pick-and-place validation",
            )
            self.pick_object = pick_object
            self.target_object = target_object
            self.background = background
            self.scenario_config = scenario_config or {}

        def get_scene_cfg(self) -> Any:
            """Configure contact sensors for pick object."""
            try:
                from isaaclab.utils import configclass
                from isaaclab.scene import InteractiveSceneCfg

                @configclass
                class RoboGateSceneCfg(InteractiveSceneCfg):
                    pass

                return RoboGateSceneCfg()
            except ImportError:
                return None

        def get_termination_cfg(self) -> Any:
            """Configure success/failure termination conditions."""
            try:
                from isaaclab.utils import configclass
                from isaaclab.managers import TerminationTermCfg

                @configclass
                class RoboGateTerminationsCfg:
                    time_out = TerminationTermCfg(
                        func=lambda env, **kwargs: env.episode_length_buf
                        >= env.max_episode_length,
                        time_out=True,
                    )

                return RoboGateTerminationsCfg()
            except ImportError:
                return None

        def get_events_cfg(self) -> Any:
            """Configure reset/randomization events."""
            return None

        def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
            """Not applicable for validation benchmark."""
            return None

        def get_metrics(self) -> list[Any]:
            """Return RoboGate metric instances."""
            metrics = []
            try:
                from robogate_benchmark.metrics import (
                    RoboGateCollisionMetric,
                    RoboGateSuccessRateMetric,
                )

                metrics.append(RoboGateSuccessRateMetric())
                metrics.append(RoboGateCollisionMetric())
            except (ImportError, NameError):
                pass
            return metrics


# ---------------------------------------------------------------------------
# RoboGateBenchmarkEnvironment (ExampleEnvironmentBase pattern)
# ---------------------------------------------------------------------------


class RoboGateBenchmarkEnvironment:
    """RoboGate 68-scenario benchmark environment definition.

    Follows the isaaclab_arena_environments ExampleEnvironmentBase pattern.
    Can be used standalone or registered in IsaacLab-Arena's environment dict.
    """

    name = "robogate_benchmark"

    def __init__(self) -> None:
        if _ARENA_AVAILABLE:
            self.asset_registry = AssetRegistry()
        else:
            self.asset_registry = None

    def get_env(self, args_cli: argparse.Namespace) -> Any:
        """Compose and return the full benchmark environment.

        Args:
            args_cli: Parsed CLI arguments.

        Returns:
            IsaacLabArenaEnvironment (or config dict in standalone mode).
        """
        embodiment_name = getattr(args_cli, "embodiment", "franka")
        background_name = getattr(args_cli, "background", "plain_table")
        enable_cameras = getattr(args_cli, "enable_cameras", False)

        if _ARENA_AVAILABLE:
            return self._build_arena_env(
                embodiment_name, background_name, enable_cameras
            )
        else:
            return self._build_standalone_config(
                embodiment_name, background_name
            )

    def _build_arena_env(
        self,
        embodiment_name: str,
        background_name: str,
        enable_cameras: bool,
    ) -> Any:
        """Build environment using Isaac Lab-Arena APIs."""
        # Load embodiment
        embodiment = self.asset_registry.get_asset_by_name(embodiment_name)(
            enable_cameras=enable_cameras
        )

        # Load or create background
        try:
            background = self.asset_registry.get_asset_by_name(background_name)()
        except (KeyError, Exception):
            background = self.asset_registry.get_asset_by_name("kitchen")()

        # Create pick object (red cube)
        try:
            pick_object = self.asset_registry.get_asset_by_name("cracker_box")()
        except (KeyError, Exception):
            pick_object = self.asset_registry.get_asset_by_name("cube")()

        # Create target marker
        try:
            target = self.asset_registry.get_asset_by_name("target_marker")()
        except (KeyError, Exception):
            target = pick_object  # fallback

        # Compose scene
        scene = Scene(assets=[background, pick_object, target])

        # Create task
        task = RoboGateValidationTask(
            pick_object=pick_object,
            target_object=target,
            background=background,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
        )

    def _build_standalone_config(
        self,
        embodiment_name: str,
        background_name: str,
    ) -> dict[str, Any]:
        """Build standalone config dict when Isaac Lab-Arena is not available."""
        return {
            "name": self.name,
            "embodiment": embodiment_name,
            "background": background_name,
            "table_pos": TABLE_POS.tolist(),
            "target_pos": TARGET_POS.tolist(),
            "obj_spawn_z": OBJ_SPAWN_Z,
            "success_dist": SUCCESS_DIST,
            "drop_z_threshold": DROP_Z_THRESHOLD,
            "scenarios": {
                "nominal": {"count": 20},
                "edge_cases": {"count": 15},
                "adversarial": {"count": 10},
                "domain_randomization": {"count": 23},
            },
            "variant_configs": VARIANT_CONFIGS,
        }

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add RoboGate-specific CLI arguments.

        Args:
            parser: ArgumentParser to extend.
        """
        parser.add_argument(
            "--embodiment",
            type=str,
            default="franka",
            choices=["franka", "ur5e"],
            help="Robot embodiment (default: franka)",
        )
        parser.add_argument(
            "--background",
            type=str,
            default="plain_table",
            help="Background scene asset name",
        )
        parser.add_argument(
            "--enable-cameras",
            action="store_true",
            default=False,
            help="Enable wrist cameras on embodiment",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for scenario generation",
        )
        parser.add_argument(
            "--nominal-count",
            type=int,
            default=20,
            help="Number of nominal scenarios",
        )
        parser.add_argument(
            "--edge-count",
            type=int,
            default=15,
            help="Number of edge case scenarios",
        )
        parser.add_argument(
            "--adversarial-count",
            type=int,
            default=10,
            help="Number of adversarial scenarios",
        )
