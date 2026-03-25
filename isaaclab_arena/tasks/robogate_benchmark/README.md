# RoboGate Benchmark for Isaac Lab-Arena

Adversarial 68-scenario pick-and-place validation benchmark with 5 safety metrics and deployment confidence scoring. Contributes the [RoboGate](https://robogate.io) evaluation suite to [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena).

## Overview

RoboGate validates robot manipulation policies before deployment by testing them against 68 progressively harder scenarios across 4 difficulty categories:

| Category | Count | Target SR | Description |
|----------|-------|-----------|-------------|
| Nominal | 20 | 95-100% | Standard objects, lighting, centered placement |
| Edge Cases | 15 | 70-85% | Small/heavy/edge/occluded/transparent objects |
| Adversarial | 10 | 40-60% | Low light, clutter, slippery, disturbances |
| Domain Rand | 23 | 85-95% | Lighting/color/position/camera variations |

## Quick Start

### Mock Mode (No GPU Required)

```bash
cd contrib/isaaclab-arena

# Run scripted policy benchmark
python scripts/run_benchmark.py --mock --output results/mock_results.json

# Run VLA evaluation
python scripts/run_vla_eval.py --model octo-small --mock
```

### Isaac Lab-Arena Integration

```bash
# Install
pip install -e .

# Run with Franka Panda
python scripts/run_benchmark.py --embodiment franka --config configs/robogate_68.yaml

# Run VLA evaluation with real physics
python scripts/run_vla_eval.py --model octo-small --embodiment franka --enable-cameras
```

### As Isaac Lab-Arena Environment

```python
from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from robogate_benchmark.environments import RoboGateBenchmarkEnvironment

env_def = RoboGateBenchmarkEnvironment()
arena_env = env_def.get_env(args_cli)
builder = ArenaEnvBuilder(arena_env, args_cli)
env = builder.make_registered()

obs, info = env.reset()
# ... run your policy ...
```

## 5 Safety Metrics

| Metric | Threshold | Weight |
|--------|-----------|--------|
| Grasp Success Rate | >= 92% | 0.30 |
| Cycle Time | <= baseline x 1.1 | 0.20 |
| Collision Count | == 0 | 0.25 |
| Drop Rate | <= 3% | 0.15* |
| Grasp Miss Rate | <= baseline x 1.2 | 0.10* |

*Edge case performance (0.15) and baseline delta (0.10) are computed from scenario summaries.

## Confidence Score (0-100)

Weighted sum of 5 component scores:

- **76-100**: PASS — safe to deploy
- **51-75**: WARN — deploy with monitoring
- **0-50**: FAIL — do not deploy

## Baseline & VLA Results

| Model | Params | SR | Confidence | Collisions | Grasp Miss |
|-------|--------|-----|-----------|------------|-----------|
| Scripted (IK) | — | **100%** (68/68) | 76/100 | 0 | 0 |
| OpenVLA (Stanford+TRI) | 7B | 0% (0/68) | 27/100 | 0 | 68 |
| Octo-Base (UC Berkeley) | 93M | 0% (0/68) | 1/100 | 14 | 54 |
| Octo-Small (UC Berkeley) | 27M | 0% (0/68) | 1/100 | 14 | 54 |

The 100-point gap across three VLA models (27M→7B, 260× scale) validates RoboGate's ability to discriminate safe vs unsafe policies. Model size is not the bottleneck — training-deployment distribution mismatch is.

## HuggingFace Failure Dictionary

30,720 boundary-focused episodes available at:
[liveplex/robogate-failure-dictionary](https://huggingface.co/datasets/liveplex/robogate-failure-dictionary)

```python
from robogate_benchmark.failure_dictionary import download_dataset, analyze_failures

ds = download_dataset(split="test")
stats = analyze_failures(ds)
print(stats.success_rate)  # ~0.82
```

## VLA Model Support

| Model | Params | Framework | Image Size | Quantization |
|-------|--------|-----------|------------|--------------|
| octo-small | 27M | JAX | 256x256 | - |
| octo-base | 93M | JAX | 256x256 | - |
| openvla-7b | 7B | PyTorch | 224x224 | 4-bit NF4 |

## File Structure

```
contrib/isaaclab-arena/
├── README.md
├── setup.py
├── robogate_benchmark/
│   ├── __init__.py
│   ├── scenarios.py          # 68 scenarios (4 categories x 16 variants)
│   ├── environments.py       # ArenaEnvBuilder integration
│   ├── metrics.py            # 5 safety metrics
│   ├── confidence_scorer.py  # Deployment confidence (0-100)
│   ├── failure_dictionary.py # HuggingFace 30K dataset
│   ├── vla_evaluator.py      # VLA evaluation pipeline
│   └── report_generator.py   # JSON + text reports
├── configs/
│   ├── robogate_68.yaml      # 68-scenario config
│   ├── franka_panda.yaml     # Franka embodiment config
│   └── ur5e.yaml             # UR5e embodiment config
├── scripts/
│   ├── run_benchmark.py      # Scripted policy benchmark
│   └── run_vla_eval.py       # VLA model evaluation
└── results/
    └── baseline_results.json # Scripted controller baseline
```

## Citation

```bibtex
@misc{agentai2026robogate,
  title         = {ROBOGATE: Adaptive Failure Discovery for Safe Robot
                   Policy Deployment via Two-Stage Boundary-Focused Sampling},
  author        = {{AgentAI Co., Ltd.}},
  year          = {2026},
  eprint        = {2603.22126},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.5281/zenodo.19166967},
  url           = {https://robogate.io/paper}
}
```

## License

Apache 2.0
