#!/usr/bin/env python3
"""Run VLA model evaluation on the RoboGate benchmark.

Evaluates Vision-Language-Action models on 68 adversarial pick-and-place
scenarios using Isaac Lab-Arena or mock mode.

Usage:
    # Mock mode (pipeline testing, no GPU needed)
    python scripts/run_vla_eval.py --model octo-small --mock

    # With Isaac Lab-Arena (requires GPU + Isaac Sim)
    python scripts/run_vla_eval.py --model octo-small --embodiment franka

    # OpenVLA with 4-bit quantization
    python scripts/run_vla_eval.py --model openvla-7b --mock

Supported models:
    octo-small   : 27M params, JAX, 256x256
    octo-base    : 93M params, JAX, 256x256
    openvla-7b   : 7B params, PyTorch 4-bit, 224x224

Exit codes:
    0: PASS (confidence >= 76)
    1: FAIL (confidence < 76)
    2: ERROR
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robogate_benchmark.report_generator import (
    generate_json_report,
    generate_text_report,
)
from robogate_benchmark.vla_evaluator import SUPPORTED_MODELS, VLAEvaluator

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for VLA evaluation."""
    parser = argparse.ArgumentParser(
        description="RoboGate VLA Benchmark — Evaluate VLA models on 68 adversarial scenarios"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="octo-small",
        choices=list(SUPPORTED_MODELS.keys()),
        help="VLA model to evaluate (default: octo-small)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock evaluation (no simulator or model needed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/robogate_68.yaml",
        help="Benchmark config path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default="franka",
        choices=["franka", "ur5e"],
        help="Robot embodiment",
    )
    parser.add_argument(
        "--enable-cameras",
        action="store_true",
        default=False,
        help="Enable wrist cameras",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress text report",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Default output path
    if args.output is None:
        model_safe = args.model.replace("-", "_").replace("/", "_")
        mode = "mock" if args.mock else "arena"
        args.output = f"results/vla_{model_safe}_{mode}.json"

    # Build environment (if not mock)
    env = None
    if not args.mock:
        try:
            from robogate_benchmark.environments import RoboGateBenchmarkEnvironment
            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

            env_def = RoboGateBenchmarkEnvironment()
            arena_env = env_def.get_env(args)
            builder = ArenaEnvBuilder(arena_env, args)
            env = builder.make_registered()
            logger.info("Isaac Lab-Arena environment ready")
        except ImportError as e:
            logger.error(
                "Isaac Lab-Arena not available: %s. Use --mock for testing.", e
            )
            sys.exit(2)

    # Load VLA policy (if not mock)
    policy = None
    if not args.mock:
        policy = _load_vla_policy(args.model)

    # Run evaluation
    evaluator = VLAEvaluator(
        model_name=args.model,
        env=env,
        mock=args.mock,
    )
    evaluator.load_model(policy=policy)

    results = evaluator.run_evaluation(
        config_path=args.config,
        output_path=args.output,
        seed=args.seed,
    )

    # Print report
    if not args.quiet:
        print(generate_text_report(results))

    # Cleanup
    if env is not None:
        env.close()

    # Exit code
    verdict = results.get("summary", {}).get("verdict", "FAIL")
    logger.info("Verdict: %s, Confidence: %s/100",
                verdict, results["summary"]["confidence_score"])

    if verdict == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


def _load_vla_policy(model_name: str):
    """Load a VLA policy for real evaluation.

    Args:
        model_name: Model identifier.

    Returns:
        VLAPolicy instance.
    """
    model_info = SUPPORTED_MODELS[model_name]

    if model_info["framework"] == "jax":
        return _load_octo_policy(model_name, model_info)
    elif model_info["framework"] == "pytorch":
        return _load_openvla_policy(model_name, model_info)
    else:
        raise ValueError(f"Unsupported framework: {model_info['framework']}")


def _load_octo_policy(model_name: str, model_info: dict):
    """Load Octo model as VLAPolicy."""
    try:
        from octo.model.octo_model import OctoModel
        import jax
    except ImportError:
        logger.error(
            "Octo not installed. Run: pip install octo-model jax[cuda12_pip]"
        )
        sys.exit(2)

    class OctoPolicy:
        def __init__(self, model, rng):
            self._model = model
            self._rng = rng
            self._task = None

        def predict_action(self, image, instruction):
            import jax.numpy as jnp

            if self._task is None:
                self._task = self._model.create_tasks(texts=[instruction])

            obs = {
                "image_primary": jnp.expand_dims(
                    jnp.expand_dims(jnp.array(image), axis=0), axis=0
                ),
                "timestep_pad_mask": jnp.array([[True]]),
            }

            self._rng, key = jax.random.split(self._rng)
            actions = self._model.sample_actions(
                obs, self._task, rng=key
            )
            action = actions[0, 0]
            return action.tolist()[:7]

        def reset(self):
            self._task = None

    logger.info("Loading Octo model: %s", model_info["hf_path"])
    model = OctoModel.load_pretrained(model_info["hf_path"])
    rng = jax.random.PRNGKey(42)

    return OctoPolicy(model, rng)


def _load_openvla_policy(model_name: str, model_info: dict):
    """Load OpenVLA model as VLAPolicy."""
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    except ImportError:
        logger.error(
            "Transformers not installed. Run: pip install transformers accelerate bitsandbytes"
        )
        sys.exit(2)

    class OpenVLAPolicy:
        PROMPT = "In: What action should the robot take to {instruction}?\nOut:"

        def __init__(self, model, processor):
            self._model = model
            self._processor = processor

        def predict_action(self, image, instruction):
            from PIL import Image as PILImage

            prompt = self.PROMPT.format(instruction=instruction)
            if not isinstance(image, PILImage.Image):
                image = PILImage.fromarray(image)

            inputs = self._processor(prompt, image).to(
                self._model.device, dtype=torch.bfloat16
            )
            action = self._model.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False
            )
            action_list = action.tolist()
            while len(action_list) < 7:
                action_list.append(0.0)
            return action_list[:7]

        def reset(self):
            pass

    logger.info("Loading OpenVLA model: %s (4-bit NF4)", model_info["hf_path"])
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_info["hf_path"],
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_info["hf_path"],
        trust_remote_code=True,
    )

    return OpenVLAPolicy(model, processor)


if __name__ == "__main__":
    main()
