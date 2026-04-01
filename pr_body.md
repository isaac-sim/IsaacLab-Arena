## Summary

Adds **RoboGate**, a 68-scenario adversarial pick-and-place validation benchmark with deployment confidence scoring, a 30,000-experiment failure dictionary, and VLA model evaluation support.

**Paper:** [arXiv:2603.22126](https://arxiv.org/abs/2603.22126) (cs.RO)
**Dataset:** [liveplex/robogate-failure-dictionary](https://huggingface.co/datasets/liveplex/robogate-failure-dictionary) (30K experiments)
**Website:** [robogate.io](https://robogate.io)

## What This Adds

### 68-Scenario Benchmark Suite
- **4 difficulty categories:** Nominal (20), Edge Cases (15), Adversarial (10), Domain Randomization (23)
- **5 safety metrics:** grasp success rate, cycle time, collision count, drop rate, grasp miss rate
- **Deployment Confidence Score (0-100):** weighted composite with PASS/WARN/FAIL thresholds
- **Two-Stage Adaptive Sampling:** uniform exploration (20K) + boundary-focused (10K) → AUC 0.780

### 30,000-Experiment Failure Dictionary
- Cross-robot validation: Franka Panda (7-DOF) + UR5e (6-DOF)
- 4 universal danger zones (mass > 0.935 kg → both robots SR < 40%)
- Closed-form failure boundary: μ*(m) = (1.469 + 0.419m) / (3.691 - 1.400m)
- Available on HuggingFace via `from robogate_benchmark.failure_dictionary import download_dataset`

### VLA Model Evaluation (3 Models Tested)

| Model | Params | SR | Confidence | Failure Pattern |
|-------|--------|-----|-----------|-----------------|
| Scripted Controller | — | **100%** (68/68) | 76/100 | — |
| OpenVLA (Stanford+TRI) | 7B | 0% (0/68) | 27/100 | grasp_miss only |
| Octo-Base (UC Berkeley) | 93M | 0% (0/68) | 1/100 | grasp_miss 79%, collision 21% |
| Octo-Small (UC Berkeley) | 27M | 0% (0/68) | 1/100 | grasp_miss 79%, collision 21% |

**Key finding:** 260× parameter scaling (27M→7B) yields zero improvement — the bottleneck is training-deployment distribution mismatch, not model capacity.

## File Structure

```
isaaclab_arena/tasks/robogate_benchmark/
├── README.md
├── __init__.py
├── scenarios.py          # 68 scenarios (4 categories)
├── environments.py       # ArenaEnvBuilder integration
├── metrics.py            # 5 safety metrics + thresholds
├── confidence_scorer.py  # Deployment confidence (0-100)
├── failure_dictionary.py # HuggingFace 30K dataset loader
├── vla_evaluator.py      # VLA evaluation pipeline (ZMQ)
├── report_generator.py   # JSON + text reports
├── configs/
│   ├── robogate_68.yaml
│   ├── franka_panda.yaml
│   └── ur5e.yaml
├── scripts/
│   ├── run_benchmark.py
│   └── run_vla_eval.py
└── results/
    └── baseline_results.json
```

## Quick Start

```bash
# Mock mode (no GPU required)
cd isaaclab_arena/tasks/robogate_benchmark
python scripts/run_benchmark.py --mock --output results/mock_results.json

# VLA evaluation
python scripts/run_vla_eval.py --model octo-small --mock
```

## Integration Points

- Follows `ArenaEnvBuilder` pattern for environment registration
- Compatible with existing `pick_and_place_task.py` infrastructure
- Reports output in standard Arena benchmark JSON format
- Mock mode for CI/testing without GPU

## Citation

```bibtex
@misc{agentai2026robogate,
  title         = {ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy
                   Deployment via Two-Stage Boundary-Focused Sampling},
  author        = {{AgentAI Co., Ltd.}},
  year          = {2026},
  eprint        = {2603.22126},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.5281/zenodo.19166967}
}
```

## Test Plan

- [ ] `python scripts/run_benchmark.py --mock` passes with 68/68 scenarios
- [ ] `python scripts/run_vla_eval.py --model octo-small --mock` completes
- [ ] Confidence score matches expected 76/100 for scripted baseline
- [ ] JSON report includes all 5 metrics + scenario breakdown
- [ ] No import errors without Isaac Sim installed (mock mode)
