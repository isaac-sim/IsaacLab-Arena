# Isaac Arena

**A scalable environment creation and evaluation framework for robotics simulations built on top of NVIDIA Isaac Lab**

</div>

## 🌟 Overview

Isaac Arena is a comprehensive robotics simulation framework that enhances NVIDIA Isaac Lab by providing a composable, scalable system for creating diverse simulation environments and evaluating robot learning policies. The framework enables researchers and developers to rapidly prototype and test robotic tasks with various robot embodiments, objects, and environments.

### Key Features

- 🤖 **Multi-Robot Support**: Compatible with various robot embodiments (Franka Panda, GR1T2, and more)
- 🏗️ **Modular Architecture**: Composable system with interchangeable backgrounds, objects, and tasks
- 📦 **Asset Registry**: Centralized management system for simulation assets with tagging
- 🎯 **Task Framework**: Composable task system. Currently we support only pick and place task.
- 📊 **Evaluation Framework**: _Coming up!!!
- 🔧 **Isaac Lab Integration**: Seamless integration with NVIDIA's Isaac Lab ecosystem.

## 🏗️ Architecture

Isaac Arena follows a modular architecture with six core components:

### Core Components

1. **Embodiments**: Robot configurations including Franka Panda and GR1T2 humanoid robot
2. **Tasks**: Specific objectives like pick-and-place
3. **Scene**: Specific parts defined for a task like pick-and-place
4. **Asset Registry**: Centralized system for managing and selecting simulation assets. This contains manipulatable objects and the backgrounds.
5. **Gym Environment Compilation**: Environmental setups which does real time compilation of specified background, objects and task to create an environment.
6. **Metrics**: Coming up!!!

The architecture enables flexible composition of environments by mixing and matching different:
- Robot embodiments (Franka, GR1T2, etc.)
- Background scenes (kitchen, warehouse, etc.)
- Manipulation objects (fruits, kitchen objects, etc)
- Task objectives (pick-and-place etc.)

## 🚀 Installation

### Prerequisites

- **CUDA-compatible GPU**
- **Docker** (for containerized deployment)

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd isaac_arena
   ```

2. **Initialize submodules**:
   ```bash
   git submodule update --init --recursive
   ```

## 📖 Usage

### Quick Start

Start the docker to before running any commands.

### Docker

1. **Build and run the Docker container**:
   ```bash
   .docker/run_docker.sh
   ```

The Docker setup is based on NVIDIA Isaac Lab base image and includes all necessary dependencies.

Run a simple pick-and-place simulation with zero actions:

```bash
python isaac_arena/examples/zero_action_runner.py \
    --background kitchen_pick_and_place \
    --object tomato_soup_can \
    --embodiment franka \
```

For using a scene with actions one can use the teleop interface from IsaacLab

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --background kitchen_pick_and_place \
    --object tomato_soup_can \
    --embodiment franka \
    --teleop_device franka \
    --device cpu
```

The following are example commands used for the mimic gen pipeline.

For recording demos with the gr1 robot and the Apple Vision Pro:
Launch the [CloudXR runtime as explained here](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#:~:text=container%20with%20Docker-,Isaac%20Lab,-can%20be%20run) in a separate terminal.
The environment variables `XDG_RUNTIME_DIR` and `XR_RUNTIME_JSON` are already set within the Isaac_arena docker.

```bash
python isaac_arena/scripts/record_demos.py \
    --teleop_device avp_handtracking \
    --embodiment gr1 \
    --background packing_table_pick_and_place \
    --task PickPlace-GR1T2 \
    --object tomato_soup_can \
    --dataset_file /tmp/gr1_table.hdf5 \
    --num_demos 1 \
    --mimic \
    --enable_pinocchio \
    --num_success_steps 1 \
    --device cpu
```

For replaying the recorded demos
```bash
python submodules/IsaacLab/scripts/tools/replay_demos.py \
    --embodiment gr1 \
    --background packing_table_pick_and_place \
    --task PickPlace-GR1T2 \
    --object tomato_soup_can \
    --dataset_file /tmp/gr1_table.hdf5 \
    --mimic \
    --enable_pinocchio \
    --device cpu
```

For annotating them. We only support manual annotation for now.
```bash
python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --embodiment gr1 \
    --background packing_table_pick_and_place \
    --task PickPlace-GR1T2 \
    --object tomato_soup_can \
    --input_file /tmp/gr1_table.hdf5 \
    --output_file /tmp/gr1_annotated.hdf5 \
    --mimic \
    --enable_pinocchio \
    --device cpu
```

For generating a dataset
```bash
python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --embodiment gr1 \
    --background packing_table_pick_and_place \
    --task PickPlace-GR1T2 \
    --object tomato_soup_can \
    --input_file /tmp/gr1_annotated.hdf5 \
    --output_file /tmp/gr1_generated.hdf5 \
    --mimic \
    --enable_pinocchio \
    --generation_num_trials 10 \
    --num_envs 5 \
    --device cpu
```

### Command Line Interface

Isaac Arena provides a comprehensive CLI for environment configuration:

```bash
# Run with specific configuration
python your_script.py \
    --background <background_name> \
    --object <object_name> \
    --task <task_name> \
    --embodiment <embodiment_name> \
    --num_envs 4 \
    --disable_fabric \
    --enable_pinocchio
```

#### Available CLI Arguments

**Isaac Lab Arguments:**
- `--num_envs`: Number of parallel environments (default: 1)
- `--mimic`: Enable mimic environment for imitation learning

**Isaac Arena Arguments:**
- `--background`: Name of the background environment
- `--object`: Name of the pick-up object
- `--task`: Name of the task to execute
- `--embodiment`: Robot embodiment to use (franka, gr1)

### Programming Interface

#### Creating Custom Environments

```python
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.embodiments.franka.franka import FrankaEmbodiment
from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene
from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask

# Create environment configuration
env_config = IsaacArenaEnvironment(
    name="custom_pick_place",
    embodiment=FrankaEmbodiment(),
    scene=PickAndPlaceScene(background_scene=my_background, pick_up_object=my_object),
    task=PickAndPlaceTask()
)
```

#### Using the Asset Registry

```python
from isaac_arena.assets.asset_registry import AssetRegistry

# Get registry instance
registry = AssetRegistry()

# Get specific asset
my_object = registry.get_asset_by_name("sugar_box")

# Get random asset by tag
random_background = registry.get_random_asset_by_tag("background")
random_object = registry.get_random_asset_by_tag("object")

# List all assets with specific tag
all_objects = registry.get_assets_by_tag("object")
```

#### Custom Asset Creation

```python
from isaac_arena.assets.asset import Asset
from isaac_arena.assets.register import register

@register_asset
class MyCustomObject(Asset):
    def __init__(self):
        super().__init__()
        self.name = "my_custom_object"
        self.tags = ["object", "custom"]

    def get_rigid_object_cfg(self):
        # Define your object configuration
        pass
```

## 🎯 Examples

### Custom Policy Evaluation

```python
import torch
import gymnasium as gym
from isaac_arena.environments.compile_env import get_arena_env_cfg

# Create environment
env_cfg, env_name = get_arena_env_cfg(args)
env = gym.make(env_name, cfg=env_cfg)

# Reset environment
obs = env.reset()

# Run policy
for step in range(1000):
    with torch.inference_mode():
        # Replace with your policy
        actions = your_policy(obs)
        obs = env.step(actions)

        if dones.any():
            obs = env.reset()
```

## 🤖 Supported Embodiments

### Franka Panda
- 7-DOF manipulator arm
- Parallel gripper
- Full joint control and observation

### GR1T2 Humanoid Robot
- Full humanoid embodiment
- Upper body manipulation
- Bi-manual capabilities

### Adding Custom Embodiments

All embodiments should contain observation, action and reset configuration defined.

```python
from isaac_arena.embodiments.embodiment_base import EmbodimentBase

class MyRobotEmbodiment(EmbodimentBase):
    def __init__(self):
        super().__init__()
        self.name = "my_robot"
        # Configure robot-specific settings
        self.event_config = MyRobotEventConfig()
        self.action_config = MyRobotActionConfig()
        self.observation_config = MyRobotObservationConfig()
```

## 📊 Tasks and Evaluation

Currently for our proof of concept we only support pick and place tasks. We are working on bringing in more tasks in the future.

### Pick and Place Task

The built-in pick-and-place task includes:

- **Objective**: Pick up specified object and place it in target location
- **Termination Conditions**: Success (object placed), timeout, or failure
- **Metrics**: Coming up!!!

### Custom Task Creation

A task brings in the termination conditions along with mimic configurations.

```python
from isaac_arena.tasks.task import TaskBase

class MyCustomTask(TaskBase):
    def get_termination_cfg(self):
        # Define when the task ends
        pass

    def get_prompt(self):
        # Return natural language description
        return "Place the pick up object on the destination object"

    def get_mimic_env_cfg(self, embodiment_name: str):
        # Return configuration for imitation learning
        pass
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest isaac_arena/tests/
```

## 🛠️ Development

### Code Style

The project follows these conventions:

- **Python 3.10+** type hints
- **Black** code formatting
- **isort** import sorting (configured for 120 character lines)
- **Pyright** type checking
- **Pre-commit hooks** for code quality

one can run all the linters and pre commit checks by running
```
pre-commit run --all-files
```

## 📄 License

This project is proprietary software owned by NVIDIA Corporation. All rights reserved.

```
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

<div align="center">

**Isaac Arena** - Scaling robotic simulation and evaluation for the future

Made with ❤️ by the NVIDIA Robotics Team

</div>
