<div align="center">

# Isaac Lab Arena

### Composable Environment Creation and Policy Evaluation for Robotics Simulation

[![Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-e8912d.svg)](#%EF%B8%8F-project-status)
[![Version](https://img.shields.io/badge/version-0.1.x-blue.svg)](https://github.com/isaac-sim/IsaacLab-Arena/tree/release/0.1.1)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-silver.svg)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE.md)

[Documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) · [NVIDIA Blog Post](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/) · [Report a Bug](https://github.com/isaac-sim/IsaacLab-Arena/issues) · [Discussions](https://github.com/isaac-sim/IsaacLab-Arena/discussions)

</div>

---

> [!WARNING]
> **Pre-Alpha Software — Not an Early Access or General Availability Release.**
> Isaac Lab Arena `v0.1.x` is an early code release intended to give the community a practical starting point to experiment, provide feedback, and influence future design direction. APIs are unstable and will change. Features are incomplete. Documentation is evolving. **Do not use this in production.** See [Project Status](#%EF%B8%8F-project-status) for details.

---

## Overview

**Isaac Lab Arena** is an open-source extension to [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) that simplifies the creation of large-scale task and environment libraries for robotic policy evaluation. Co-developed with [Lightwheel](https://www.lightwheel.ai/), it provides a composable architecture where environments are assembled on-the-fly from independent, reusable building blocks — eliminating the redundant boilerplate that plagues traditional task library development.

Instead of hand-writing and maintaining a separate configuration for every combination of robot, object, and scenario, Arena lets you **compose** environments from three independent primitives:

| Primitive | Description |
|-----------|-------------|
| **Scene** | The physical environment layout — a collection of objects, furniture, fixtures |
| **Embodiment** | The robot and its observations, actions, sensors, and controllers |
| **Task** | The objective — what the robot should accomplish (pick-and-place, open door, etc.) |

The `ArenaEnvBuilder` composes these primitives into a standard `ManagerBasedRLEnvCfg` that runs natively in Isaac Lab.

## Why Isaac Lab Arena?

With the rise of generalist robot policies (e.g., [GR00T N](https://developer.nvidia.com/gr00t), [pi0](https://www.physicalintelligence.company/), [SmolVLA](https://huggingface.co/HuggingFaceTB/SmolVLA-base)), there is an urgent need to evaluate these policies across many diverse tasks and environments. Traditional approaches suffer from:

- **Code duplication** — each task variation (different object, different robot) requires a near-copy of the same configuration
- **Maintenance burden** — N robots × M objects × K scenes = an explosion of configs to keep in sync
- **Slow iteration** — researchers spend more time wrangling configs than running experiments

Arena solves this by making environment variation a first-class concept. Swap an object, change a robot, or modify a scene — all without duplicating a single line of task logic.

## Key Features

- **Composable Environments** — Mix and match scenes, embodiments, and tasks independently
- **On-the-fly Assembly** — Environments are built at runtime; no duplicate config files to maintain
- **Asset Registry** — Centralized management of robots, objects, and scenes with affordance annotations
- **Integrated Evaluation** — Built-in metrics and evaluation pipelines for policy benchmarking
- **Teleoperation Support** — Data collection via keyboard, VR, or other input devices
- **GR00T Integration** — First-class support for NVIDIA GR00T N policy training and evaluation
- **LeRobot Hub** — Publish and share environments on the [Hugging Face LeRobot Environment Hub](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot)

## Quick Start

### Prerequisites

- Linux (Ubuntu 22.04+)
- NVIDIA GPU (see [Isaac Sim hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html))
- Docker and NVIDIA Container Toolkit
- Git

### Installation

Isaac Lab Arena currently supports **installation from source inside a Docker container**.

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
pytest -sv -m "not with_cameras" isaaclab_arena/tests/
```

> **Note:** The Docker script automatically mounts `$HOME/datasets`, `$HOME/models`, and `$HOME/eval` from your host into the container.

For detailed setup instructions (including server-client mode for GR00T), see the [Installation Guide](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html).

## Usage Example

Build a pick-and-place environment with a Franka arm in a kitchen scene:

```python
from isaaclab_arena import (
    IsaacLabArenaEnvironment, ArenaEnvBuilder, Scene,
    ObjectReference, ObjectType, PickAndPlaceTask,
    asset_registry, device_registry,
)

# Select building blocks
embodiment = asset_registry.get_asset_by_name("franka")(enable_cameras=True)
background = asset_registry.get_asset_by_name("kitchen")()
tomato_soup_can = asset_registry.get_asset_by_name("tomato_soup_can")()
destination = ObjectReference(
    name="destination_location",
    prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
    parent_asset=background,
    object_type=ObjectType.RIGID,
)
teleop_device = device_registry.get_device_by_name("keyboard")()

# Compose the environment
scene = Scene([background, tomato_soup_can])

env_cfg = IsaacLabArenaEnvironment(
    name="franka_kitchen_pickup",
    embodiment=embodiment,
    scene=scene,
    task=PickAndPlaceTask(tomato_soup_can, destination, background),
    teleop_device=teleop_device,
)

env_builder = ArenaEnvBuilder(env_cfg, args_cli)
env = env_builder.make_registered()  # Registers with the gym registry
```

Explore more examples in the [documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html), including:

| Example | Description |
|---------|-------------|
| [G1 Loco-Manipulation](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) | Unitree G1 humanoid locomotion + manipulation |
| [GR1 Open Microwave Door](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) | GR1 humanoid interacting with articulated objects |
| [GR1 Sequential Pick & Place](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) | Multi-step manipulation with GR1 |
| [Franka Lift Object](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) | Classic Franka pick-up task |

## Project Structure

```
IsaacLab-Arena/
├── isaaclab_arena/            # Core framework (environments, tasks, scenes, embodiments)
├── isaaclab_arena_g1/         # Unitree G1 humanoid embodiment + examples
├── isaaclab_arena_gr00t/      # GR00T policy integration
├── docker/                    # Docker configurations and launch scripts
├── docs/                      # Sphinx documentation source
├── osmo/                      # Cloud deployment configs (OSMO)
├── submodules/                # Git submodules (Isaac Lab, etc.)
├── setup.py                   # Package installation
├── CONTRIBUTING.md            # Contribution guidelines
└── LICENSE.md                 # Apache 2.0 license
```

## Version Compatibility

| Isaac Lab Arena | Isaac Lab | Isaac Sim | Python |
|-----------------|-----------|-----------|--------|
| `main` branch   | 2.3.0     | 5.1.0     | ≥ 3.10 |
| `release/0.1.1` | 2.3.0     | 5.1.0     | ≥ 3.10 |

## ⚠️ Project Status

Isaac Lab Arena is in **pre-alpha** (`v0.1.x`). This is important to understand:

| What This Means | Details |
|-----------------|---------|
| **Not EA / GA** | This is not an Early Access or General Availability release. It is a very early community code drop. |
| **APIs will break** | Public interfaces are under active development and will change without deprecation warnings. |
| **Features are incomplete** | Core capabilities like natural-language object placement, composite task chaining, RL task setup, and heterogeneous parallel evaluation are planned but not yet implemented. |
| **Docker-only install** | Source installation in a Docker container is the only supported method in `v0.1.x`. |
| **Limited testing** | The `main` branch contains the latest code but may not be fully tested. Use `release/0.1.1` for the most stable experience. |

### Planned Enhancements (Post `v0.1.x`)

- Object placement via natural language
- Composite tasking by chaining atomic skills
- Reinforcement learning task setup
- Parallel heterogeneous evaluations (different objects per parallel environment)

Track progress and upcoming features in [GitHub Issues](https://github.com/isaac-sim/IsaacLab-Arena/issues).

## Ecosystem

Isaac Lab Arena is part of a growing ecosystem of tools and benchmarks:

- **[Lightwheel RoboCasa Tasks](https://github.com/lightwheel-ai)** — 250+ open-source tasks built on Arena
- **[Lightwheel LIBERO Tasks](https://github.com/lightwheel-ai)** — Adapted LIBERO benchmarks
- **[LeRobot Environment Hub](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot)** — Share and discover Arena environments on Hugging Face
- **[RoboTwin 2.0](https://robotwin-benchmark.github.io/dex-robot/)** — Extended simulation benchmarks using Arena
- **[Isaac Lab Teleop](https://github.com/isaac-sim/IsaacLab)** — Demonstration collection
- **[Isaac Lab Mimic](https://github.com/isaac-sim/IsaacLab)** — Synthetic data generation

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

Isaac Lab Arena is released under the [Apache 2.0 License](LICENSE.md).

Note that Isaac Lab Arena requires Isaac Sim, which includes components under proprietary licensing terms. See the [Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html) for details.

## Citation

If you use Isaac Lab Arena in your research, please cite:

```bibtex
@misc{isaaclab-arena2025,
    title   = {Isaac Lab Arena: Composable Environment Creation and Policy Evaluation for Robotics},
    author  = {{NVIDIA Isaac Lab Arena Contributors}},
    year    = {2025},
    url     = {https://github.com/isaac-sim/IsaacLab-Arena}
}
```

If you use Isaac Lab (the underlying framework), please also cite the [Isaac Lab paper](https://arxiv.org/abs/2511.04831).

## Acknowledgements

Isaac Lab Arena is co-developed with [Lightwheel](https://www.lightwheel.ai/) and builds on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab). We thank the Isaac Lab team and the broader robotics community for their foundational work.

---

<div align="center">

**Isaac Lab Arena** · Pre-Alpha · [Documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/index.html) · [GitHub](https://github.com/isaac-sim/IsaacLab-Arena)

</div>
