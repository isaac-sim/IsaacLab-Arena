<div align="center">

# Isaac Lab-Arena

### Composable Environment Creation and Policy Evaluation for Robotics Simulation

[![Alpha](https://img.shields.io/badge/status-alpha-e8912d.svg)](#%EF%B8%8F-project-status)
[![Version](https://img.shields.io/badge/version-0.3-blue.svg)](https://github.com/isaac-sim/IsaacLab-Arena/tree/main)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-3.0.0-silver.svg)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/python-≥3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE.md)

[Documentation](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/index.html) · [NVIDIA Blog Post](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/) · [Report a Bug](https://github.com/isaac-sim/IsaacLab-Arena/issues) · [Discussions](https://github.com/isaac-sim/IsaacLab-Arena/discussions)

</div>

---

> [!WARNING]
> **Alpha Software — Not an Early Access or General Availability Release.**
> Isaac Lab-Arena `v0.3` is an early code release intended to give the community a practical starting point to experiment, provide feedback, and influence future design direction. APIs are unstable and will change. Features are incomplete. Documentation is evolving. **Do not use this in production.** See [Project Status](#%EF%B8%8F-project-status) for details.

> [!NOTE]
> Changes on `main` contain an in-development version based on v0.3.0 and Isaac Lab 3.0.
---

## Overview

**Isaac Lab-Arena** is an open-source framework for scalable benchmark authoring and robot policy evaluation in simulation. It extends [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) with reusable APIs to author benchmarks, execute evaluations at scale, and analyze results for actionable feedback.

Instead of hand-writing and maintaining a separate configuration for every combination of robot,
object, and scenario, Arena composes environments from three independent primitives: a **scene**,
which defines the physical layout and its objects, furniture, and fixtures; an **embodiment**, which
defines the robot, observations, actions, sensors, and controllers; and a **task**, which defines what
the robot must accomplish. `ArenaEnvBuilder` combines them into a standard `ManagerBasedRLEnvCfg`
that runs natively in Isaac Lab.

Building on that foundation, Arena provides three connected capabilities across the benchmark and policy-evaluation workflow:

| Workflow | What Arena provides |
|----------|---------------------|
| **Author** | Build reusable benchmark environments through modular composition, relational placement, prompt-driven generation, and controlled variations. Register Arena environments with Isaac Lab for learning and data generation. |
| **Execute** | Evaluate one policy concurrently across thousands of heterogeneous environments on a GPU. Package multiple tasks and policies as experiments that run locally or across nodes through OSMO and a common policy client. |
| **Analyze** | Collect aggregate and per-episode metrics, trace predicate-based subtask progress, and run sensitivity analysis over controlled conditions to see where and why policies fail. |

## Why Isaac Lab-Arena?

See the [documentation overview](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/index.html#why-isaac-lab-arena) for the motivation behind Arena and how it addresses evaluation scale, reproducibility, and failure diagnosis.

## Key Features

### Author Scalable Benchmarks

- **Composable environments** — Configure scenes, embodiments, and tasks independently, with reusable objects composed into each scene. `ArenaEnvBuilder` assembles them into an Isaac Lab environment without duplicating task logic.
- **Relational object placement** — Define layouts with spatial relationships such as "on" and "next to" instead of hard-coded coordinates.
- **Sequential task chaining** — Chain atomic skills (pick, walk, place, …) into long-horizon composite tasks.
- **Agentic environment generation (experimental)** — Describe a task in natural language; an agent infers constraints, creates a reviewable specification, finds SimReady USD assets, and builds a family of Arena environments. Initial examples cover composite pick-and-place.
- **Controlled environment variations** — Turn one environment into a perturbation sweep over lighting, backgrounds, camera parameters, object mass, and other configurable ranges and distributions.
- **Isaac Lab interoperability** — Register Arena-authored environments with Isaac Lab workflows for reinforcement learning and data generation.

### Execute Evaluations at Scale

- **GPU-accelerated parallel evaluation** — Evaluate one policy across thousands of heterogeneous environments concurrently instead of running sequential rollouts.
- **Large-scale multi-node evaluation** — Define experiments with multiple tasks and policies, run locally or distribute them with an orchestrator such as OSMO, and collect aggregate metrics plus per-episode results.
- **Policy client-server architecture** — Evaluate GR00T, π0.5, or a custom policy behind a server through a common observation-and-action contract.
- **Streamlined setup and agent skills** — Use the native `uv` installation path and reusable agent skills for key workflows.

### Analyze Policy Robustness

- **Subtask predicates** — Track milestones such as grasp, lift, transport, and place to pinpoint where a policy fails.
- **Sensitivity analysis** — Perturb environment factors to reveal robustness gaps and generate actionable feedback for targeted policy learning.

## Quick Start

### Prerequisites

- Linux (Ubuntu 22.04+)
- NVIDIA GPU (see [Isaac Sim hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html))
- [uv](https://docs.astral.sh/uv/) (for the native install), or Docker and the NVIDIA Container Toolkit (for the container install)
- Git

### Installation

**Native developer setup with uv:**

```bash
# 1. Clone the repository
git clone --recurse-submodules git@github.com:isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena

# 2. Create the locked environment (Isaac Lab from source, plus the Isaac Sim, PyTorch, and Newton wheels)
uv sync

# 3. Activate the environment and accept the Isaac Sim EULA
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

# 4. Verify the installation with a short zero-action rollout
python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 20 cube_goal_pose

# 4b. (Optional) Watch the rollout in the GUI visualizer
python isaaclab_arena/evaluation/policy_runner.py \
  --viz kit --policy_type zero_action --num_steps 200 cube_goal_pose
```

> **Note:** See our
> [installation docs](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/installation.html)
> for more details and installation flavors.

**Source install inside Docker:**

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

# 3. Verify the installation with a short zero-action rollout
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 20 cube_goal_pose

# 3b. (Optional) Watch the rollout in the GUI visualizer
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --viz kit --policy_type zero_action --num_steps 200 cube_goal_pose
```

> **Note:** The Docker script automatically mounts `$HOME/datasets`, `$HOME/models`, and `$HOME/eval` from your host into the container.

For detailed setup instructions (including server-client mode for GR00T), see the [Installation Guide](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/installation.html).

## Usage Example

Compose a Franka arm in a kitchen scene with a couple of objects:

```python
from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder, ArenaEnvBuilderCfg
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

builder_cfg = ArenaEnvBuilderCfg()
env_builder = ArenaEnvBuilder(env_cfg, builder_cfg)
env = env_builder.make_registered()
env.reset()
```

Python callers set builder options directly on `ArenaEnvBuilderCfg`. Runner scripts
continue to accept the same options as CLI flags, such as `--num_envs 4 --seed 7`,
and translate them into an `ArenaEnvBuilderCfg` before building the environment.

### Continue in the Documentation

#### Getting Started

Choose a guide based on what you want to do:

- [First Arena Environment](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/arena_env.html) — Compose a scene, embodiment, and task into a reusable environment.
- [First Arena Experiment](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/arena_experiment.html) — Define and run multiple evaluation configurations as one experiment.
- [Exploring Environment Variations](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/environment_variations.html) — Sample controlled changes to lighting, cameras, and backgrounds.
- [Running a Real Policy](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/running_a_real_policy/index.html) — Evaluate a pretrained policy from a saved configuration.

#### Ready-to-Use Environments

- [Example Environments](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/example_workflows/example_environments.html) — Browse Python-registered environments, RoboLab-inspired tasks, and Kitchen Benchmark specifications.

#### Example Workflows

Explore complete workflows for:

- [Evaluation](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/example_workflows/analysis/index.html) — Run controlled sweeps and analyze the conditions associated with policy success or failure.
- [Agentic Environment Generation](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/example_workflows/agentic_env_gen/index.html) — Generate Arena environment specifications from natural-language prompts.
- [Imitation Learning](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/example_workflows/imitation_learning/index.html) — Collect data, post-train a policy, and run closed-loop evaluation.
- [Reinforcement Learning](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/example_workflows/reinforcement_learning_workflows/index.html) — Set up an environment, train a policy, and run closed-loop evaluation.

## Project Structure

```
IsaacLab-Arena/
├── isaaclab_arena/                    # Core framework (environments, tasks, scenes, embodiments)
├── isaaclab_arena_environments/       # Concrete environment definitions
├── isaaclab_arena_examples/           # Policy and relation examples
├── isaaclab_arena_g1/                 # Unitree G1 humanoid embodiment + examples
├── isaaclab_arena_dreamzero/          # DreamZero policy integration
├── isaaclab_arena_gr00t/              # GR00T policy integration
├── isaaclab_arena_openpi/             # OpenPi (pi0 / pi05) policy integration
├── docker/                            # Docker configurations and launch scripts
├── docs/                              # Sphinx documentation source
├── osmo/                              # Cloud deployment configs (OSMO)
├── submodules/                        # Git submodules (Isaac Lab, etc.)
├── pyproject.toml                     # Package metadata, dependencies, and uv config
├── CONTRIBUTING.md                    # Contribution guidelines
└── LICENSE.md                         # Apache 2.0 license
```

## Version Compatibility

| Isaac Lab-Arena                      | Isaac Lab | Isaac Sim | Python |
|--------------------------------------|-----------|-----------|--------|
| `main`                               | 3.0.0     | 6.0.0     | ≥ 3.12 |
| `release/0.3.0`                      | 3.0.0     | 6.0.0     | ≥ 3.12 |
| `release/0.2.1`                      | 3.0.0     | 6.0.0     | ≥ 3.12 |
| `release/0.2.0`                      | 3.0.0     | 6.0.0     | ≥ 3.12 |
| `feature/arena_v0.2_on_lab_2.3`      | 2.3.0     | 5.1.0     | ≥ 3.10 |
| `release/0.1.1`                      | 2.3.0     | 5.0.0     | ≥ 3.10 |
| `release/0.1.0`                      | 2.3.0     | 5.0.0     | ≥ 3.10 |

## ⚠️ Project Status

Isaac Lab-Arena is in **alpha** (`v0.3`). This is important to understand:

| What This Means | Details |
|-----------------|---------|
| **Not EA / GA** | This is not an Early Access or General Availability release. It is a very early community code drop. |
| **APIs will break** | Public interfaces are under active development and will change without deprecation warnings. |
| **Features are evolving** | Agentic environment generation is experimental, performance is not yet hardened for production-scale workloads, and benchmark and analysis coverage continues to expand. |
| **Limited testing** | The `main` branch contains the latest code but may not be fully tested. Use `release/0.3.0` for the most stable experience. |


## Ecosystem

Isaac Lab-Arena is part of a growing ecosystem of tools and benchmarks. NVIDIA is working with benchmark authors and model developers to build, run, and open-source benchmarks on Arena.

The ecosystem extends beyond pick-and-place to contact-rich, dexterous, and deformable benchmarks for industry and academia. Arena's modular foundation lets you reuse them as low-cost readiness gates or adapt their building blocks—tasks, scenes, robots, and evaluation methods—for custom evaluations.

### Published Benchmarks

- **[Lightwheel RoboFinals](https://lightwheel.ai/robofinals)** — High-fidelity industrial benchmarks.
- **[Lightwheel RoboCasa Tasks](https://github.com/LightwheelAI/LW-BenchHub)** — 138+ open-source tasks, 50 datasets per task, 7+ robots.
- **[Lightwheel LIBERO Tasks](https://github.com/LightwheelAI/LW-BenchHub)** — Adapted LIBERO benchmarks.
- [**RoboTwin 2.0**](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena) — Extended simulation benchmarks using Arena; [Arxiv](https://arxiv.org/abs/2603.01229).
- **[LeRobot Environment Hub](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot)** — Share and discover Arena environments on Hugging Face.
- **[Isaac for Healthcare RHEO Workflows](https://github.com/isaac-for-healthcare/i4h-workflows/tree/main/workflows/rheo)** — Healthcare robotics benchmark workflows.

### Coming Soon

Coming soon: support for the full RoboTwin and RoboDojo task suites, plus benchmark integrations
from ecosystem partners including RLWRLD (DexBench), UC Berkeley, X Square, Sharpa, and NVIDIA
GEAR (G1 Factory), with more partner benchmarks to follow.

### Publishing Your Own Benchmark

We encourage the community to build and publish benchmarks on Isaac Lab-Arena. The recommended workflow:

1. **Maintain your benchmark in your own repository.** Create a branch or package that integrates with Isaac Lab-Arena (e.g. an `IsaacLab-Arena` branch). For detailed setup instructions—including repository layout, Dockerfile setup, and how to register custom environments, robots, and tasks—see the [Arena in Your Repository](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/arena_in_your_repo/index.html) guide.
2. **Reference your benchmark and Isaac Lab-Arena in publications.** When publishing on ArXiv or elsewhere, cite both your benchmark (by name, with a link to your repository) and Isaac Lab-Arena as the underlying evaluation framework.
3. **List it here.** Open a PR to add your benchmark to the [Published Benchmarks](#published-benchmarks) list above. This README serves as the single source of truth for the Arena benchmark ecosystem so that community can discover and reuse.


## Contributing

We welcome contributions — bug reports, feature suggestions, and code. This is an alpha project, so community input directly shapes the framework's direction.

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

Isaac Lab-Arena was built in collaboration with the authors of Robolab ([website](https://research.nvidia.com/labs/srl/projects/robolab/), [paper](https://arxiv.org/abs/2604.09860)).

---

<div align="center">

**Isaac Lab-Arena** · Alpha · [Documentation](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/index.html) · [GitHub](https://github.com/isaac-sim/IsaacLab-Arena)

Made with ❤️ by the NVIDIA Robotics Team

</div>
