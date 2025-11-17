# IsaacLab Arena for ORCA development

This is a fork of the IsaacLab Arena repository for ORCA development.

It is used to develop and test the IsaacLab Arena framework for ORCA.

## Environment Setup

```
git clone git@github.com:isaac-for-healthcare/IsaacLab-Arena-Orca.git
cd IsaacLab-Arena-Orca
git submodule update --init --recursive
```

## Development Guide

```bash
mkdir -p ~/datasets
mkdir -p ~/models
mkdir -p ~/eval
```

Set up huggingface `pip install huggingface-hub`
```bash
pip install huggingface-hub
```

Login to huggingface
```bash
hf auth login
```

Download the dataset

```bash
hf download nvidia/orca-dev-test --repo-type=dataset --local-dir ~/datasets/orca-dev-test
```

**Contributing**

For new files, please add `orca` to the filename. For example, `run_docker.orca.sh` to avoid conflicts with the main repository.

## Usage

```bash
./docker/run_docker.orca.sh -g
```

### Data Collection

Collect
```bash
python isaaclab_arena/scripts/record_demos_keyboard_23d_orca.py \
  --enable_cameras \
  --dataset_file /datasets/orca-dev-test/keyboard23d/g1_demo_test.hdf5 \
  --num_demos 1 \
  --pos_sensitivity 0.1 \
  --vel_sensitivity 0.2 \
  orca_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

Replay
```bash
python isaaclab_arena/scripts/replay_demos.py \
  --enable_cameras \
  --dataset_file /datasets/orca-dev-test/keyboard23d/g1_demo_test_3.hdf5 \
  orca_g1_locomanip_pick_and_place \
  --embodiment g1_wbc_pink
```

Inspect
```bash
# No need to run Isaac Sim
python isaaclab_arena/scripts/inspect_dataset.py \
  /datasets/orca-dev-test/keyboard23d/g1_demo_test_3.hdf5 \
  --episode 0
```

### Policy Running

```bash
python isaaclab_arena/examples/policy_runner.py \
    --policy_type gr00t_closedloop \
    --policy_config_yaml_path isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
    --num_steps 1200 \
    --enable_cameras \
    orca_g1_locomanip_pick_and_place \
    --object brown_box \
    --embodiment g1_wbc_joint
```
