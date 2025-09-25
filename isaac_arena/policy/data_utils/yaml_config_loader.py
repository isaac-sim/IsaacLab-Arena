# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import yaml
from pathlib import Path
from typing import Any, Union
from dataclasses import fields

from isaac_arena.policy.config.dataset_config import Gr00tDatasetConfig


def convert_yaml_value(field_type: type, value: Any) -> Any:
    """Convert YAML value to the appropriate type based on field type annotation."""
    # Handle Path fields
    if field_type == Path or (hasattr(field_type, '__origin__') and field_type.__origin__ is Union and Path in field_type.__args__):
        if isinstance(value, str):
            return Path(value)
        return value

    # Handle tuple fields (like image size)
    if hasattr(field_type, '__origin__') and field_type.__origin__ is tuple:
        if isinstance(value, list):
            return tuple(value)
        return value

    # Handle basic types
    if field_type in (str, int, float, bool):
        return field_type(value)

    return value


def load_config_from_yaml(yaml_path: Union[str, Path]) -> Gr00tDatasetConfig:
    """
    Load Gr00tDatasetConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Gr00tDatasetConfig: Initialized configuration object

    Example:
        >>> config = load_config_from_yaml("my_config.yaml")
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    # Load YAML content
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    if yaml_data is None:
        yaml_data = {}

    # Get field information from dataclass
    field_types = {field.name: field.type for field in fields(Gr00tDatasetConfig)}

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

    # Create the config object with converted values
    try:
        config = Gr00tDatasetConfig(**converted_data)
    except Exception as e:
        print(f"Error creating Gr00tDatasetConfig: {e}")
        print("Available fields:")
        for field in fields(Gr00tDatasetConfig):
            print(f"  - {field.name}: {field.type}")
        raise

    return config
