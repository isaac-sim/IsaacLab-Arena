"""RoboGate 68-scenario adversarial test suite.

Defines 4 categories x 16 variants = 68 scenarios for pick-and-place
validation. Maps directly to configs/default_test.yaml from the
RoboGate core project.

Categories:
    nominal (20):             Standard operating conditions
    edge_cases (15):          Boundary conditions (small/heavy/edge/occluded/transparent)
    adversarial (10):         Hostile conditions (low light/clutter/slippery/disturbance)
    domain_randomization (23): Visual/physical variations (lighting/color/position/camera)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ScenarioCategory(str, Enum):
    """Scenario difficulty category."""

    NOMINAL = "nominal"
    EDGE_CASES = "edge_cases"
    ADVERSARIAL = "adversarial"
    DOMAIN_RANDOMIZATION = "domain_randomization"


@dataclass
class ScenarioVariant:
    """Single scenario variant definition."""

    category: ScenarioCategory
    variant: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


# ---------------------------------------------------------------------------
# Variant physics/difficulty configs (from isaac_connector.py)
# ---------------------------------------------------------------------------

TABLE_TOP_Z = 0.3
OBJ_SPAWN_Z = TABLE_TOP_Z + 0.02

OBSTACLE_LAYOUT = [
    [-0.08, 0.05, TABLE_TOP_Z + 0.08],
    [0.05, -0.04, TABLE_TOP_Z + 0.06],
    [0.0, 0.18, TABLE_TOP_Z + 0.07],
]

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    # === Nominal (target 95-100%) ===
    "standard_objects": {"obj_scale": 0.04, "obj_mass": 0.1},
    "standard_lighting": {"obj_scale": 0.04, "obj_mass": 0.1},
    "centered_placement": {"obj_scale": 0.04, "obj_mass": 0.1, "center": True},
    # === Edge cases (target 70-85%) ===
    "small_objects": {
        "obj_scale": 0.02,
        "obj_mass": 0.02,
        "approach_noise": 0.025,
        "grasp_tol": 0.04,
    },
    "heavy_objects": {
        "obj_scale": 0.05,
        "obj_mass": 2.0,
        "grasp_break_prob": 0.004,
    },
    "edge_placement": {
        "obj_scale": 0.04,
        "obj_mass": 0.1,
        "edge": True,
        "approach_noise": 0.02,
        "grasp_tol": 0.055,
    },
    "occluded_objects": {
        "obj_scale": 0.04,
        "obj_mass": 0.1,
        "approach_noise": 0.025,
        "grasp_tol": 0.055,
    },
    "transparent_objects": {
        "obj_scale": 0.04,
        "obj_mass": 0.1,
        "approach_noise": 0.028,
        "grasp_tol": 0.045,
    },
    # === Adversarial (target 40-60%) ===
    "low_lighting": {
        "obj_scale": 0.04,
        "obj_mass": 0.1,
        "approach_noise": 0.04,
        "grasp_tol": 0.035,
    },
    "cluttered_scene": {
        "obj_scale": 0.04,
        "obj_mass": 0.1,
        "obstacle_offsets": OBSTACLE_LAYOUT,
        "approach_noise": 0.015,
        "grasp_tol": 0.06,
    },
    "slippery_surface": {
        "obj_scale": 0.04,
        "obj_mass": 0.1,
        "release_slip": 0.07,
        "approach_noise": 0.01,
    },
    "moving_disturbance": {
        "obj_scale": 0.04,
        "obj_mass": 0.1,
        "obj_perturbation": 0.022,
        "approach_noise": 0.025,
        "grasp_tol": 0.05,
    },
    # === Domain randomization (mild, target 85-95%) ===
    "lighting": {"obj_scale": 0.04, "obj_mass": 0.1, "approach_noise": 0.015},
    "object_color": {"obj_scale": 0.04, "obj_mass": 0.1, "approach_noise": 0.012},
    "position": {"obj_scale": 0.04, "obj_mass": 0.1, "approach_noise": 0.008},
    "camera": {"obj_scale": 0.04, "obj_mass": 0.1, "approach_noise": 0.015},
}


# ---------------------------------------------------------------------------
# Scenario distributions
# ---------------------------------------------------------------------------

NOMINAL_VARIANTS = ["standard_objects", "standard_lighting", "centered_placement"]
EDGE_CASE_VARIANTS = [
    "small_objects",
    "heavy_objects",
    "edge_placement",
    "occluded_objects",
    "transparent_objects",
]
ADVERSARIAL_VARIANTS = [
    "low_lighting",
    "cluttered_scene",
    "slippery_surface",
    "moving_disturbance",
]
DOMAIN_RAND_VARIANTS = ["lighting", "object_color", "position", "camera"]
DOMAIN_RAND_COUNTS = {
    "lighting": 10,
    "object_color": 5,
    "position": 5,
    "camera": 3,
}


def build_scenario_suite(
    seed: int = 42,
    nominal_count: int = 20,
    edge_count: int = 15,
    adversarial_count: int = 10,
) -> list[ScenarioVariant]:
    """Build the full 68-scenario RoboGate test suite.

    Args:
        seed: Random seed for reproducible domain randomization.
        nominal_count: Number of nominal scenarios.
        edge_count: Number of edge case scenarios.
        adversarial_count: Number of adversarial scenarios.

    Returns:
        List of 68 ScenarioVariant objects.
    """
    scenarios: list[ScenarioVariant] = []
    rng = np.random.default_rng(seed)

    # Nominal (20)
    for i in range(nominal_count):
        variant = NOMINAL_VARIANTS[i % len(NOMINAL_VARIANTS)]
        params = dict(VARIANT_CONFIGS[variant])
        params["cycle_index"] = i
        scenarios.append(
            ScenarioVariant(
                category=ScenarioCategory.NOMINAL,
                variant=variant,
                params=params,
                seed=int(rng.integers(0, 2**31)),
            )
        )

    # Edge cases (15)
    for i in range(edge_count):
        variant = EDGE_CASE_VARIANTS[i % len(EDGE_CASE_VARIANTS)]
        params = dict(VARIANT_CONFIGS[variant])
        params["cycle_index"] = i
        scenarios.append(
            ScenarioVariant(
                category=ScenarioCategory.EDGE_CASES,
                variant=variant,
                params=params,
                seed=int(rng.integers(0, 2**31)),
            )
        )

    # Adversarial (10)
    for i in range(adversarial_count):
        variant = ADVERSARIAL_VARIANTS[i % len(ADVERSARIAL_VARIANTS)]
        params = dict(VARIANT_CONFIGS[variant])
        params["cycle_index"] = i
        scenarios.append(
            ScenarioVariant(
                category=ScenarioCategory.ADVERSARIAL,
                variant=variant,
                params=params,
                seed=int(rng.integers(0, 2**31)),
            )
        )

    # Domain randomization (23)
    for variant, count in DOMAIN_RAND_COUNTS.items():
        params_base = dict(VARIANT_CONFIGS[variant])
        for i in range(count):
            params = dict(params_base)
            params["cycle_index"] = i
            params["variation_type"] = variant
            # Per-episode randomization parameters
            params["light_intensity"] = float(rng.uniform(0.3, 1.5))
            params["color_jitter"] = float(rng.uniform(0.0, 0.3))
            params["position_offset"] = rng.normal(0, 0.01, size=3).tolist()
            params["camera_noise_std"] = float(rng.uniform(0.0, 0.02))
            scenarios.append(
                ScenarioVariant(
                    category=ScenarioCategory.DOMAIN_RANDOMIZATION,
                    variant=variant,
                    params=params,
                    seed=int(rng.integers(0, 2**31)),
                )
            )

    return scenarios


def get_language_instructions() -> dict[str, str]:
    """Get language instructions for VLA evaluation per category.

    Returns:
        Mapping from scenario category to natural language instruction.
    """
    return {
        "nominal": "pick up the red cube and place it at the green target",
        "edge_cases": "pick up the object and place it at the target location",
        "adversarial": "grasp the object on the table and move it to the goal",
        "domain_randomization": "pick up the red cube and place it at the green target",
    }
