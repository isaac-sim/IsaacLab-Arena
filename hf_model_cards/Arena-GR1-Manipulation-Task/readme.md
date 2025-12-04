---
license: cc-by-4.0
task_categories:
- robotics
tags:
- robotics
---
## Dataset Description:

This dataset is multimodal collections of trajectories generated in Isaac Lab. It supports humanoid (GR1) manipulation task in IsaacLab-Arena enviorment. Each entry provides the full context (state, vision, language, action) needed to train and evaluate generalist robot policies for opening microwave task.

| Dataset Name          | # Trajectories |
|-----------------------|----------------|
| GR1 Manipulation Task | 50             |

This dataset is ideal for behavior cloning, policy learning, and generalist robotic manipulation research. It has been for post-training GR00T N1.5 model.

This dataset is ready for commercial use.

## Dataset Owner
NVIDIA Corporation

## Dataset Creation Date:
10/10/2025

## License/Terms of Use:
This dataset is governed by the Creative Commons Attribution 4.0 International License (CC-BY-4.0).

## Intended Usage:
This dataset is intended for:

- Training robot manipulation policies using behavior cloning.
- Research in generalist robotics and task-conditioned agents.
- Sim-to-real / Sim-to-Sim transfer studies.

## Dataset Characterization:
### Data Collection Method

- Automated
- Automatic/Sensors
- Synthetic
  
10 human teleoperated demonstrations are collected through a depth camera and keyboard in Isaac Lab. All 50 demos are generated automatically using a synthetic motion trajectory generation framework, Mimicgen [1]. Each demo is generated at 50 Hz.

### Labeling Method

Not Applicable

## Dataset Format:
We provide a few dataset files, including

- a human-annoated 10 demonstrations in HDF5 dataset file (`arena_gr1_manipulation_dataset_annotated.hdf5`)
- a Mimic-generated 50 demonstrations in HDF5 dataset file (`arena_gr1_manipulation_dataset_generated.hdf5`)
- a GR00T-Lerobot formatted dataset converted from the Mimic-generated HDF5 dataset file (`lerobot`)

Each demo in GR00T-Lerobot datasets consists of a time-indexed sequence of the following modalities:

### Actions
- action (FP64): joint desired positions for all body joints (36 DoF)

### Observations
- observation.state (FP64): joint positions for all body joints (54 DoF)

### Task-specific
- timestamp (FP64): simulation time in seconds of each recorded data entry.
- annotation.human.action.task_description (INT64): index referring to the language instruction recorded in the metadata
- annotation.human.action.valid (INT64): index indicating validity of annotaion recorded in the metadata
- episode_index (INT64): index indicating the order of each demo
- task_index (INT64): index used in multi-task data loader. Not applicable to Gr00t-N1 post training, always set to 0.


### Videos
- 512 x 512 RGB videos in mp4 format from first-person-view camera

In additional, a set of metadata describing the followings is provided,
- `episodes.jsonl` contains a list of all the episodes in the entire dataset. Each episode contains a list of tasks and the length of the episode.
- `tasks.jsonl` contains a list of all the tasks in the entire dataset.
- `modality.json` contains the modality configuration.
- `info.json` contains the dataset information.


## Dataset Quantification:

### Record Count

#### GR1 Manipulation Task
- Number of demonstrations/trajectories: 50
- Number of RGB videos: 50

### Total Storage

5.16 GB

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Reference(s):
[1] @inproceedings{mandlekar2023mimicgen,
    title={MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations},
    author={Mandlekar, Ajay and Nasiriany, Soroush and Wen, Bowen and Akinola, Iretiayo and Narang, Yashraj and Fan, Linxi and Zhu, Yuke and Fox, Dieter},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023}
    }