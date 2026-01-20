# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import collections
import json
import numpy as np
import yaml
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar, Union

# Generic type variable for configuration classes
ConfigType = TypeVar("ConfigType")


def dump_jsonl(data, file_path):
    """
    Write a sequence of data to a file in JSON Lines format.

    Args:
        data: Sequence of items to write, one per line.
        file_path: Path to the output file.

    Returns:
        None
    """
    assert isinstance(data, collections.abc.Sequence) and not isinstance(data, str)
    if isinstance(data, (np.ndarray, np.number)):
        data = data.tolist()
    with open(file_path, "w") as fp:
        for line in data:
            print(json.dumps(line), file=fp, flush=True)


def dump_json(data, file_path, **kwargs):
    """
    Write data to a file in standard JSON format.

    Args:
        data: Data to write to the file.
        file_path: Path to the output file.
        **kwargs: Additional keyword arguments for json.dump.

    Returns:
        None
    """
    if isinstance(data, (np.ndarray, np.number)):
        data = data.tolist()
    with open(file_path, "w") as fp:
        json.dump(data, fp, **kwargs)


def load_json(file_path: str | Path, **kwargs) -> dict[str, Any]:
    """
    Load a JSON file.

    Args:
        file_path: Path to the JSON file to load.
        **kwargs: Additional keyword arguments for the JSON loader.

    Returns:
        Dictionary loaded from the JSON file.
    """
    with open(file_path) as fp:
        return json.load(fp, **kwargs)


def load_dict_from_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Load dictionary from YAML file"""
    assert yaml_path.exists(), f"{yaml_path} does not exist"
    # also return empty dict if the file is empty
    if yaml_path.stat().st_size == 0:
        return {}
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_robot_joints_config_from_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Load robot joint configuration from YAML file"""
    config = load_dict_from_yaml(yaml_path)
    return config.get("joints", {})


def convert_yaml_value(field_type: type, value: Any) -> Any:
    """Convert YAML value to the appropriate type based on field type annotation."""
    # Handle Path fields
    if field_type == Path or (
        hasattr(field_type, "__origin__") and field_type.__origin__ is Union and Path in field_type.__args__
    ):
        if isinstance(value, str):
            return Path(value)
        return value

    # Handle tuple fields (like image size)
    if hasattr(field_type, "__origin__") and field_type.__origin__ is tuple:
        if isinstance(value, list):
            return tuple(value)
        return value

    # Handle basic types
    if field_type in (str, int, float, bool):
        return field_type(value)

    return value


def load_config_from_yaml(yaml_path: str | Path, config_class: type[ConfigType]) -> dict[str, Any]:
    """
    Load configuration from a YAML file for any dataclass.

    Args:
        yaml_path: Path to the YAML configuration file
        config_class: The dataclass type to load configuration for

    Returns:
        dict: Dictionary of converted configuration data

    Example:
        >>> data = load_config_from_yaml("my_config.yaml", Gr00tDatasetConfig)
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    yaml_data = load_dict_from_yaml(yaml_path)

    # Get field information from dataclass
    field_types = {field.name: field.type for field in fields(config_class)}

    # Convert YAML values to appropriate types
    converted_data = {}
    for field_name, value in yaml_data.items():
        if field_name in field_types:
            try:
                converted_data[field_name] = convert_yaml_value(field_types[field_name], value)
            except Exception as e:
                print(f"Warning: Failed to convert field '{field_name}' with value '{value}': {e}")
                converted_data[field_name] = value
        else:
            print(f"Warning: Unknown field '{field_name}' in YAML config")
    return converted_data


def create_config_from_yaml(yaml_path: str | Path, config_class: type[ConfigType]) -> ConfigType:
    """Create a configuration object from a YAML file using any dataclass.

    Args:
        yaml_path (str | Path): Path to the YAML configuration file
        config_class (Type[ConfigType]): The dataclass type to instantiate

    Returns:
        ConfigType: Instance of the specified configuration class

    Example:
        >>> dataset_config = create_config_from_yaml("dataset.yaml", Gr00tDatasetConfig)
        >>> policy_config = create_config_from_yaml("policy.yaml", LerobotReplayActionPolicyConfig)
    """
    try:
        converted_data = load_config_from_yaml(yaml_path, config_class)
        config = config_class(**converted_data)
    except Exception as e:
        print(f"Error creating {config_class.__name__}: {e}")
        print("Available fields:")
        for field in fields(config_class):
            print(f"  - {field.name}: {field.type}")
        raise

    return config

def load_gr00t_modality_config_from_file(modality_config_path: str | Path, embodiment_tag: str):
    """Load the modality configs using GR00T's pattern.
    1. Import the config module (registers it globally)
    2. Retrieve from the global registry using embodiment_tag
    Args:
        modality_config_path: Path to the modality configuration file
        embodiment_tag: Embodiment tag to use to load modality configurations from `submodules/Isaac-GR00T/gr00t/data/embodiment_tags.py`.
    Returns:
        modality_configs: Modality configurations
    """
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.launch_finetune import load_modality_config

    if modality_config_path:
        # Import module for side-effect registration
        load_modality_config(modality_config_path)

    # Get the embodiment tag from policy config and convert to EmbodimentTag enum
    # Handle case-insensitive lookup (e.g., "NEW_EMBODIMENT" or "new_embodiment" both work)
    try:
        embodiment_tag_enum = EmbodimentTag[embodiment_tag.upper()]
    except KeyError:
        embodiment_tag_str = embodiment_tag.upper()
        matching_tags = [tag for tag in EmbodimentTag if tag.name == embodiment_tag_str]
        if not matching_tags:
            available_tags = [tag.name for tag in EmbodimentTag]
            raise ValueError(
                f"Invalid embodiment tag '{embodiment_tag}'. "
                f"Available tags: {available_tags}"
            )
        embodiment_tag_enum = matching_tags[0]

    # Use the enum's value (lowercase string) to look up in MODALITY_CONFIGS
    embodiment_tag_key = embodiment_tag_enum.value

    if embodiment_tag_key not in MODALITY_CONFIGS:
        raise ValueError(
            f"Embodiment tag '{embodiment_tag_enum.name}' (value: '{embodiment_tag_key}') not found in MODALITY_CONFIGS. "
            f"Available tags: {list(MODALITY_CONFIGS.keys())}. "
            f"Make sure {modality_config_path} calls register_modality_config()."
        )

    modality_configs = MODALITY_CONFIGS[embodiment_tag_key]
    return modality_configs
