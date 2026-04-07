<div align="center">

# Isaac Lab-Arena

### Composable Environment Creation and Policy Evaluation for Robotics Simulation

[![Alpha](https://img.shields.io/badge/status-alpha-e8912d.svg)](#%EF%B8%8F-project-status)
[![Version](https://img.shields.io/badge/version-0.2.x-blue.svg)](https://github.com/isaac-sim/IsaacLab-Arena/tree/main)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-silver.svg)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE.md)

[Documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) · [NVIDIA Blog Post](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/) · [Report a Bug](https://github.com/isaac-sim/IsaacLab-Arena/issues) · [Discussions](https://github.com/isaac-sim/IsaacLab-Arena/discussions)

</div>

---

> [!WARNING]
> **Alpha Software — Not an Early Access or General Availability Release.**
> Isaac Lab-Arena `v0.2.x` is an early code release intended to give the community a practical starting point to experiment, provide feedback, and influence future design direction. APIs are unstable and will change. Features are incomplete. Documentation is evolving. **Do not use this in production.** See [Project Status](#%EF%B8%8F-project-status) for details.

> [!NOTE]
> Changes on `main` contains an in development version of v0.2.0.
> As of March 16th 2026 (GTC San Jose 2026), `main` contains most of the features for the v0.2.0 release,
> however, is based on Isaac Lab 2.3 (rather than Isaac Lab 3.0) and has not been SQA tested.*
---

## Overview

**Isaac Lab-Arena** is an open-source extension to [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) for simplified task curation and robotic policy evaluation at scale. It provides a composable architecture where environments are assembled on-the-fly from independent, reusable building blocks — eliminating the redundant boilerplate that plagues traditional task library development.

Instead of hand-writing and maintaining a separate configuration for every combination of robot, object, and scenario, Arena lets you **compose** environments from three independent primitives:

| Primitive | Description |
|-----------|-------------|
| **Scene** | The physical environment layout — a collection of objects, furniture, fixtures |
| **Embodiment** | The robot and its observations, actions, sensors, and controllers |
| **Task** | The objective — what the robot should accomplish (pick-and-place, open door, etc.) |

The `ArenaEnvBuilder` composes these primitives into a standard `ManagerBasedRLEnvCfg` that runs natively in Isaac Lab.

## Why Isaac Lab-Arena?

With the rise of generalist robot policies (e.g., [GR00T N](https://developer.nvidia.com/isaac/gr00t), [pi0](https://www.physicalintelligence.company/), [SmolVLA](https://huggingface.co/docs/lerobot/smolvla)), there is an urgent need to evaluate these policies across many diverse tasks and environments. Traditional approaches suffer from:

- **Code duplication** — each task variation (different object, different robot) requires a near-copy of the same configuration
- **Maintenance burden** — N robots × M objects × K scenes = an explosion of configs to keep in sync
- **Slow iteration** — researchers spend more time wrangling configs than running experiments

Arena solves this by making environment variation a first-class concept. Swap an object, change a robot, or modify a scene — all without duplicating a single line of task logic.

## Key Features

- **LEGO-like Composable Environments** — Mix and match scenes,  embodiments, and tasks independently
- **On-the-fly Assembly** — Environments are built at runtime; no duplicate config files to maintain.
- **New Sequential Task Chaining** — Chain atomic skills (e.g. Pick + Walk + Place + …) to create complex long-horizon tasks.
- **New Natural Language Object Placement** — Define scene layouts using semantic relationships like "on" or "next to", instead of manually specified coordinates.
- **Integrated Evaluation** — Extensible metrics and evaluation pipelines for policy benchmarking
- **New Large-scale Parallel Evaluations with Heterogeneous Objects** — Evaluate policy on multiple parallel environments, each with different objects, to maximize evaluation throughput.
- **New RL Workflow Support and Seamless Interoperation with Isaac Lab: Plug Isaac Lab** - Arena environments into Isaac Lab workflows for Reinforcement learning and Data generation for imitation learning.

## Quick Start

### Prerequisites

- Linux (Ubuntu 22.04+)
- NVIDIA GPU (see [Isaac Sim hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html))
- Docker and NVIDIA Container Toolkit
- Git

### Installation

Isaac Lab-Arena currently supports **installation from source inside a Docker container**.

```bash
# 1. Clone the repository
git clone git@github.com:isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena
git submodule update --init --recursive

# 2. Launch the Docker container
#    Base container (recommended for development):
./docker/run_docker.sh

#    Or with GR00T dependencies (for policy training/evaluation):
./docker/run_docker.sh -g

# 3. Verify the installation
/isaac-sim/python.sh -m pytest -sv -m "not with_cameras" isaaclab_arena/tests/
```

> **Note:** The Docker script automatically mounts `$HOME/datasets`, `$HOME/models`, and `$HOME/eval` from your host into the container.

For detailed setup instructions (including server-client mode for GR00T), see the [Installation Guide](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html).

## Usage Example

Compose a Franka arm in a kitchen scene with a couple of objects:

```python
from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene

asset_registry = AssetRegistry()

# Select building blocks
background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("franka_ik")()
cracker_box = asset_registry.get_asset_by_name("cracker_box")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()

# Compose the environment
scene = Scene(assets=[background, cracker_box, tomato_soup_can])
env_cfg = IsaacLabArenaEnvironment(
    name="franka_kitchen_example",
    embodiment=embodiment,
    scene=scene,
)

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(env_cfg, args_cli)
env = env_builder.make_registered()
env.reset()
```

Explore more examples in the [documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html), including:

| Example | Description |
|---------|-------------|
| [G1 Loco-Manipulation](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/locomanipulation/index.html) | Unitree G1 humanoid locomotion + manipulation |
| [GR1 Open Microwave Door](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/static_manipulation/index.html) | GR1 humanoid interacting with articulated objects |
| [GR1 Sequential Pick & Place](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/sequential_static_manipulation/index.html) | Multi-step manipulation with GR1 |
| [Franka Lift Object](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/reinforcement_learning/index.html) | Classic Franka pick-up task |

## Project Structure

```
IsaacLab-Arena/
├── isaaclab_arena/                    # Core framework (environments, tasks, scenes, embodiments)
├── isaaclab_arena_environments/       # Concrete environment definitions
├── isaaclab_arena_examples/           # Policy and relation examples
├── isaaclab_arena_g1/                 # Unitree G1 humanoid embodiment + examples
├── isaaclab_arena_gr00t/              # GR00T policy integration
├── docker/                            # Docker configurations and launch scripts
├── docs/                      # Sphinx documentation source
├── osmo/                      # Cloud deployment configs (OSMO)
├── submodules/                # Git submodules (Isaac Lab, etc.)
├── setup.py                   # Package installation
├── CONTRIBUTING.md            # Contribution guidelines
└── LICENSE.md                 # Apache 2.0 license
```

## Version Compatibility

| Isaac Lab-Arena | Isaac Lab | Isaac Sim | Python |
|-----------------|-----------|-----------|--------|
| `main` branch   | 3.0.0     | 6.0.0     | ≥ 3.12 |
| `feature/arena_v0.2_on_lab_2.3`   | 2.3.0     | 5.1.0     | ≥ 3.10 |
| `release/0.1.1` | 2.3.0     | 5.0.0     | ≥ 3.10 |

## ⚠️ Project Status

Isaac Lab-Arena is in **alpha** (`v0.2.x`). This is important to understand:

| What This Means | Details |
|-----------------|---------|
| **Not EA / GA** | This is not an Early Access or General Availability release. It is a very early community code drop. |
| **APIs will break** | Public interfaces are under active development and will change without deprecation warnings. |
| **Features are incomplete** | Core capabilities like agentic task generation, non-sequential long horizon tasks, easy-to-configure sensitivity analysis, enhanced heterogeneity across parallel evaluations and pip install support are planned but not yet implemented. |
| **Docker-only install** | Source installation in a Docker container is the only supported method. |
| **Limited testing** | The `main` branch contains the latest code but may not be fully tested. Use `release/0.1.1` for the most stable experience. |


## Ecosystem

Isaac Lab-Arena is part of a growing ecosystem of tools and benchmarks. NVIDIA and partners are building industrial and academic benchmarks on the unified Isaac Lab-Arena core, so you can reuse building blocks (tasks, scenes, metrics, and datasets) for your custom evaluations.

### Published Benchmarks

- **[Lightwheel RoboFinals](https://lightwheel.ai/robofinals)** — High-fidelity industrial benchmarks.
- **[Lightwheel RoboCasa Tasks](https://github.com/LightwheelAI/LW-BenchHub)** — 138+ open-source tasks, 50 datasets per task, 7+ robots.
- **[Lightwheel LIBERO Tasks](https://github.com/LightwheelAI/LW-BenchHub)** — Adapted LIBERO benchmarks.
- **[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena)** — Extended simulation benchmarks using Arena; [Arxiv](https://arxiv.org/abs/2603.01229).
- **[LeRobot Environment Hub](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot)** — Share and discover Arena environments on Hugging Face.

NIST Board 1, NVIDIA Isaac GR00T Industrial Benchmarks, NVIDIA DexBench, NVIDIA RoboLab, and more benchmarks are coming soon.

### Publishing Your Own Benchmark

We encourage the community to build and publish benchmarks on Isaac Lab-Arena. The recommended workflow:

1. **Maintain your benchmark in your own repository.** Create a branch or package that integrates with Isaac Lab-Arena (e.g. an `IsaacLab-Arena` branch). See [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena) for a reference example.
2. **Reference your benchmark and Isaac Lab-Arena in publications.** When publishing on ArXiv or elsewhere, cite both your benchmark (by name, with a link to your repository) and Isaac Lab-Arena as the underlying evaluation framework.
3. **List it here.** Open a PR to add your benchmark to the [Published Benchmarks](#published-benchmarks) list above. This README serves as the single source of truth for the Arena benchmark ecosystem so that community can discover and reuse.


## Contributing

We welcome contributions — bug reports, feature suggestions, and code. This is a pre-alpha project, so community input directly shapes the framework's direction.

1. Read the [Contribution Guidelines](CONTRIBUTING.md)
2. Sign off your commits (DCO required — see `CONTRIBUTING.md`)
3. Open a [Pull Request](https://github.com/isaac-sim/IsaacLab-Arena/pulls)

Areas where contributions are especially valuable:
- New task definitions and benchmark suites
- Additional robot embodiments and scene assets
- Sim-to-real validated evaluation methods
- Documentation improvements and tutorials

## Support

- **Questions & Ideas** — [GitHub Discussions](https://github.com/isaac-sim/IsaacLab-Arena/discussions)
- **Bug Reports** — [GitHub Issues](https://github.com/isaac-sim/IsaacLab-Arena/issues)
- **Isaac Sim Questions** — [NVIDIA Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67)
- **Community Chat** — [Omniverse Discord](https://discord.com/invite/nvidiaomniverse)

## License

Isaac Lab-Arena is released under the [Apache 2.0 License](LICENSE.md).

Note that Isaac Lab-Arena requires Isaac Sim, which includes components under proprietary licensing terms. See the [Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html) for details.

## Citation

If you use Isaac Lab-Arena in your research, please cite:

```bibtex
@misc{isaaclab-arena2025,
    title   = {Isaac Lab-Arena: Composable Environment Creation and Policy Evaluation for Robotics},
    author  = {{NVIDIA Isaac Lab-Arena Contributors}},
    year    = {2025},
    url     = {https://github.com/isaac-sim/IsaacLab-Arena}
}
```

If you use Isaac Lab (the underlying framework), please also cite the [Isaac Lab paper](https://arxiv.org/abs/2511.04831).

## Acknowledgements

Isaac Lab-Arena builds on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab), with the evaluation and task layers designed in close collaboration with Lightwheel. We thank the Isaac Lab team and the broader robotics community for their foundational work.

---

<div align="center">

**Isaac Lab-Arena** · Alpha · [Documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) · [GitHub](https://github.com/isaac-sim/IsaacLab-Arena)

Made with ❤️ by the NVIDIA Robotics Team

</div>
