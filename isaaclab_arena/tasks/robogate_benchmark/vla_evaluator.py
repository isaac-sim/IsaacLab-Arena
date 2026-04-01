"""VLA (Vision-Language-Action) evaluation pipeline for Isaac Lab-Arena.

Evaluates VLA models (Octo, OpenVLA, RT-2, etc.) on the RoboGate
68-scenario benchmark using the ArenaEnvBuilder gym environment.

Supports two modes:
    1. arena: Full Isaac Lab-Arena gym env with real physics
    2. mock:  Synthetic episodes for pipeline testing

Usage::

    evaluator = VLAEvaluator(model_name="octo-small")
    evaluator.load_model()
    results = evaluator.run_evaluation(config_path="configs/robogate_68.yaml")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robogate_benchmark.confidence_scorer import compute_confidence_score
from robogate_benchmark.metrics import (
    CycleResult,
    collect_failure_evidence,
    compute_all_metrics,
    compute_scenario_summary,
    evaluate_all_metrics,
)
from robogate_benchmark.scenarios import (
    ScenarioVariant,
    build_scenario_suite,
    get_language_instructions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy protocol — any VLA model must implement this
# ---------------------------------------------------------------------------


class VLAPolicy(Protocol):
    """Protocol for VLA model inference."""

    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """Predict robot action from image and language instruction.

        Args:
            image: RGB image array (H, W, 3), uint8.
            instruction: Natural language task instruction.

        Returns:
            Action array of shape (7,): [dx, dy, dz, drx, dry, drz, gripper].
        """
        ...

    def reset(self) -> None:
        """Reset model state between episodes."""
        ...


# ---------------------------------------------------------------------------
# Supported VLA models
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "octo-small": {
        "hf_path": "hf://rail-berkeley/octo-small-1.5",
        "framework": "jax",
        "params": "27M",
        "image_size": 256,
    },
    "octo-base": {
        "hf_path": "hf://rail-berkeley/octo-base-1.5",
        "framework": "jax",
        "params": "93M",
        "image_size": 256,
    },
    "openvla-7b": {
        "hf_path": "openvla/openvla-7b",
        "framework": "pytorch",
        "params": "7B",
        "image_size": 224,
    },
}


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Result of a single VLA evaluation episode."""

    scenario_category: str
    scenario_variant: str
    seed: int
    success: bool
    failure_type: str | None
    cycle_time: float
    steps: int
    total_inference_time: float
    collision: bool = False
    drop: bool = False
    grasp_miss: bool = False


# ---------------------------------------------------------------------------
# VLA Evaluator
# ---------------------------------------------------------------------------


class VLAEvaluator:
    """Evaluates VLA models on the RoboGate 68-scenario benchmark.

    Args:
        model_name: Model identifier from SUPPORTED_MODELS.
        env: Optional gymnasium env from ArenaEnvBuilder.
        mock: If True, use synthetic episodes (no simulator needed).
    """

    def __init__(
        self,
        model_name: str = "octo-small",
        env: Any = None,
        mock: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_info = SUPPORTED_MODELS.get(model_name, SUPPORTED_MODELS["octo-small"])
        self.env = env
        self.mock = mock
        self._policy: VLAPolicy | None = None
        self._rng = np.random.default_rng(42)

    def load_model(self, policy: VLAPolicy | None = None) -> None:
        """Load or set the VLA policy.

        Args:
            policy: Pre-loaded VLAPolicy instance. If None, user must
                    provide one before calling run_evaluation().
        """
        if policy is not None:
            self._policy = policy
            logger.info("VLA policy set: %s", self.model_name)

    def run_evaluation(
        self,
        config_path: str | Path | None = None,
        output_path: str | Path | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run the full 68-scenario evaluation.

        Args:
            config_path: Path to robogate_68.yaml (optional, uses defaults).
            output_path: Path to write JSON results.
            seed: Random seed.

        Returns:
            Full evaluation results dict.
        """
        self._rng = np.random.default_rng(seed)
        scenarios = build_scenario_suite(seed=seed)
        instructions = get_language_instructions()

        logger.info(
            "Starting VLA evaluation: model=%s, scenarios=%d, mock=%s",
            self.model_name,
            len(scenarios),
            self.mock,
        )

        t0 = time.time()
        episodes: list[EpisodeResult] = []
        total_inferences = 0

        for i, scenario in enumerate(scenarios):
            instruction = instructions.get(
                scenario.category.value,
                "pick up the object and place it at the target",
            )

            if self.mock:
                ep = self._run_mock_episode(scenario)
            else:
                ep = self._run_arena_episode(scenario, instruction)

            episodes.append(ep)
            total_inferences += ep.steps

            if (i + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d episodes", i + 1, len(scenarios)
                )

        total_time = time.time() - t0

        # Convert to CycleResults for metric computation
        cycles = [
            CycleResult(
                scenario_category=ep.scenario_category,
                scenario_variant=ep.scenario_variant,
                success=ep.success,
                cycle_time=ep.cycle_time,
                collision=ep.collision,
                drop=ep.drop,
                grasp_miss=ep.grasp_miss,
            )
            for ep in episodes
        ]

        # Compute metrics
        raw_metrics = compute_all_metrics(cycles)
        evaluated = evaluate_all_metrics(raw_metrics)
        summaries = compute_scenario_summary(cycles)
        confidence = compute_confidence_score(evaluated, summaries)
        evidence = collect_failure_evidence(cycles)

        # Assemble results
        results = self._build_results(
            episodes=episodes,
            raw_metrics=raw_metrics,
            evaluated=evaluated,
            summaries=summaries,
            confidence=confidence,
            evidence=evidence,
            total_time=total_time,
            total_inferences=total_inferences,
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info("Results written to %s", output_path)

        return results

    def _run_arena_episode(
        self,
        scenario: ScenarioVariant,
        instruction: str,
    ) -> EpisodeResult:
        """Run a single episode using the Isaac Lab-Arena gym env."""
        if self.env is None:
            raise RuntimeError("No gym environment set. Use mock=True or provide env.")
        if self._policy is None:
            raise RuntimeError("No VLA policy loaded. Call load_model() first.")

        self._policy.reset()
        obs, info = self.env.reset()

        steps = 0
        total_inf_time = 0.0
        max_steps = 300
        done = False
        success = False

        while not done and steps < max_steps:
            # Extract image from observation
            image = self._extract_image(obs)

            # VLA inference
            t_inf = time.time()
            action = self._policy.predict_action(image, instruction)
            total_inf_time += time.time() - t_inf

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1

            if info.get("success", False):
                success = True
                break

        cycle_time = info.get("cycle_time", steps * 0.05)
        collision = info.get("collision", False)
        drop = info.get("drop", False)
        grasp_miss = not success and not collision and not drop

        failure_type = None
        if not success:
            if collision:
                failure_type = "collision"
            elif drop:
                failure_type = "drop"
            elif grasp_miss:
                failure_type = "grasp_miss"
            else:
                failure_type = "timeout"

        return EpisodeResult(
            scenario_category=scenario.category.value,
            scenario_variant=scenario.variant,
            seed=scenario.seed,
            success=success,
            failure_type=failure_type,
            cycle_time=cycle_time,
            steps=steps,
            total_inference_time=total_inf_time,
            collision=collision,
            drop=drop,
            grasp_miss=grasp_miss,
        )

    def _run_mock_episode(self, scenario: ScenarioVariant) -> EpisodeResult:
        """Run a synthetic mock episode for pipeline testing."""
        cat = scenario.category.value
        variant = scenario.variant

        # Probability model (VLA models typically struggle)
        base_sr = {
            "nominal": 0.15,
            "edge_cases": 0.05,
            "adversarial": 0.02,
            "domain_randomization": 0.10,
        }.get(cat, 0.10)

        success = self._rng.random() < base_sr
        collision = not success and self._rng.random() < 0.20
        drop = not success and not collision and self._rng.random() < 0.10
        grasp_miss = not success and not collision and not drop

        failure_type = None
        if not success:
            if collision:
                failure_type = "collision"
            elif drop:
                failure_type = "drop"
            elif grasp_miss:
                failure_type = "grasp_miss"
            else:
                failure_type = "timeout"

        steps = int(self._rng.integers(50, 300))
        cycle_time = steps * 0.05
        inf_time = steps * float(self._rng.uniform(0.08, 0.25))

        return EpisodeResult(
            scenario_category=cat,
            scenario_variant=variant,
            seed=scenario.seed,
            success=success,
            failure_type=failure_type,
            cycle_time=round(cycle_time, 2),
            steps=steps,
            total_inference_time=round(inf_time, 2),
            collision=collision,
            drop=drop,
            grasp_miss=grasp_miss,
        )

    def _extract_image(self, obs: Any) -> np.ndarray:
        """Extract RGB image from gym observation."""
        if isinstance(obs, dict):
            for key in ["image", "rgb", "pixels", "wrist_image", "policy"]:
                if key in obs:
                    img = obs[key]
                    if isinstance(img, dict) and "image" in img:
                        return np.asarray(img["image"])
                    return np.asarray(img)
        # Fallback: return dummy image
        size = self.model_info.get("image_size", 256)
        return np.zeros((size, size, 3), dtype=np.uint8)

    def _build_results(
        self,
        episodes: list[EpisodeResult],
        raw_metrics: dict[str, float | int],
        evaluated: dict[str, dict[str, Any]],
        summaries: dict[str, Any],
        confidence: dict[str, Any],
        evidence: list[Any],
        total_time: float,
        total_inferences: int,
    ) -> dict[str, Any]:
        """Build the full results JSON structure."""
        passed = sum(1 for ep in episodes if ep.success)

        return {
            "metadata": {
                "benchmark": "robogate",
                "version": "1.0.0",
                "model": self.model_name,
                "model_info": self.model_info,
                "mode": "mock" if self.mock else "arena",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_scenarios": len(episodes),
                "total_time_s": round(total_time, 1),
                "total_inferences": total_inferences,
            },
            "summary": {
                "total": len(episodes),
                "passed": passed,
                "failed": len(episodes) - passed,
                "success_rate": round(passed / len(episodes), 4) if episodes else 0,
                "confidence_score": confidence["score"],
                "verdict": confidence["verdict"],
            },
            "metrics": raw_metrics,
            "metrics_evaluated": evaluated,
            "scenario_summary": {
                cat: {
                    "total": s.total,
                    "passed": s.passed,
                    "failed": s.failed,
                    "success_rate": round(s.pass_rate, 4),
                }
                for cat, s in summaries.items()
            },
            "confidence": confidence,
            "failure_evidence": [
                {
                    "scenario": e.scenario,
                    "failure_type": e.failure_type,
                    "count": e.count,
                    "severity": e.severity,
                    "description": e.description,
                }
                for e in evidence
            ],
            "episodes": [
                {
                    "category": ep.scenario_category,
                    "variant": ep.scenario_variant,
                    "seed": ep.seed,
                    "success": ep.success,
                    "failure_type": ep.failure_type,
                    "cycle_time": ep.cycle_time,
                    "steps": ep.steps,
                    "inference_time": round(ep.total_inference_time, 3),
                }
                for ep in episodes
            ],
        }
