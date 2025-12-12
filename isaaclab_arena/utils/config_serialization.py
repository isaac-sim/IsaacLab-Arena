# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for serializing and deserializing IsaacLab Arena environment configs."""

import builtins
import importlib
import inspect
import numpy as np
import pkgutil
import yaml
from dataclasses import fields
from pathlib import Path
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import (
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderManagerBaseCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, VisualMaterialCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, SphereCfg
from isaaclab.utils.string import string_to_callable

# IsaacLab Arena imports
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.arena_env_reconstruction import reconstruct_embodiment, reconstruct_scene, reconstruct_task
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose


def _get_config_class_patterns(class_type_str):
    """Generate potential config class patterns from a class_type string."""
    if ":" not in class_type_str:
        return []

    module_path, class_name = class_type_str.rsplit(":", 1)
    config_patterns = []

    # Pattern 1: same module + Cfg (e.g., joint_actions:JointPositionActionCfg)
    config_patterns.append(f"{module_path}:{class_name}Cfg")

    # Pattern 2: module + _cfg + Cfg (e.g., rigid_object_cfg:RigidObjectCfg)
    config_patterns.append(f"{module_path}_cfg:{class_name}Cfg")

    # Pattern 3: For nested modules like actions.joint_actions, try parent.parent_cfg
    # e.g., isaaclab.envs.mdp.actions.joint_actions -> isaaclab.envs.mdp.actions.actions_cfg
    if "." in module_path:
        parent_parts = module_path.rsplit(".", 1)
        if len(parent_parts) == 2:
            parent_module, last_module = parent_parts
            parent_name = parent_module.split(".")[-1]
            config_patterns.append(f"{parent_module}.{parent_name}_cfg:{class_name}Cfg")

            # Pattern 4: For actuators specifically, try replacing last module with 'actuator_cfg'
            # e.g., isaaclab.actuators.actuator_pd:ImplicitActuator -> isaaclab.actuators.actuator_cfg:ImplicitActuatorCfg
            if last_module.startswith("actuator"):
                config_patterns.append(f"{parent_module}.actuator_cfg:{class_name}Cfg")

    return config_patterns


def register_yaml_constructors():
    """Register custom YAML constructors for numpy types and Python builtins."""

    def slice_constructor(loader, node):
        """Construct Python slice objects from YAML."""
        args = loader.construct_sequence(node, deep=True)
        return builtins.slice(*args)

    def numpy_scalar_constructor(loader, node):
        """Construct numpy scalar objects from YAML binary data."""
        dtype_node, data_node = node.value
        dtype = loader.construct_object(dtype_node, deep=True)
        data = loader.construct_object(data_node, deep=True)
        scalar = np.frombuffer(data, dtype=dtype, count=1)[0]
        return scalar

    def numpy_dtype_constructor(loader, node):
        """Construct numpy dtype objects from YAML."""
        if isinstance(node, yaml.MappingNode):
            mapping = loader.construct_mapping(node, deep=True)
            if "args" in mapping:
                return np.dtype(*mapping["args"])
        raise yaml.YAMLError(f"Could not reconstruct numpy.dtype from node: {node}")

    # Register constructors
    yaml.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice", slice_constructor, Loader=yaml.FullLoader
    )
    yaml.add_constructor(
        "tag:yaml.org,2002:python/object/apply:numpy.dtype", numpy_dtype_constructor, Loader=yaml.FullLoader
    )
    yaml.add_constructor(
        "tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar",
        numpy_scalar_constructor,
        Loader=yaml.FullLoader,
    )


def _discover_metric_classes():
    """Discover all MetricBase subclasses in the metrics module.

    Returns:
        Dict mapping metric class names to their classes
    """

    try:
        import isaaclab_arena.metrics as metrics_module
        from isaaclab_arena.metrics.metric_base import MetricBase
    except ImportError as e:
        print(f"[WARNING] Could not import metrics module: {e}")
        return {}

    metric_classes = {}

    # Get the metrics module path
    metrics_path = Path(metrics_module.__file__).parent

    # Iterate through all .py files in the metrics directory
    for module_info in pkgutil.iter_modules([str(metrics_path)]):
        if module_info.name.startswith("_") or module_info.name == "metric_base":
            continue

        try:
            # Import the module
            module = importlib.import_module(f"isaaclab_arena.metrics.{module_info.name}")

            # Find all MetricBase subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, MetricBase) and obj is not MetricBase:
                    metric_classes[name] = obj
                    print(f"[DEBUG] Discovered metric class: {name} from module {module_info.name}")
        except Exception as e:
            print(f"[WARNING] Could not import metric module {module_info.name}: {e}")
            import traceback

            traceback.print_exc()

    return metric_classes


def _reconstruct_asset_from_dict(asset_dict):
    """Reconstruct an Asset object from a dictionary.

    Args:
        asset_dict: Dictionary containing asset data

    Returns:
        Asset instance
    """
    from isaaclab_arena.assets.asset import Asset

    return Asset(name=asset_dict.get("_name", asset_dict.get("name")), tags=asset_dict.get("tags"))


def _reconstruct_metrics_from_yaml(metrics_list):
    """Reconstruct metric objects from YAML data by auto-discovering metric classes.

    This function automatically discovers all MetricBase subclasses in the metrics folder
    and attempts to match YAML data to their __init__ signatures.

    Args:
        metrics_list: List of metric dictionaries from YAML

    Returns:
        List of MetricBase objects
    """

    if not metrics_list:
        return []

    # Discover all available metric classes
    metric_classes = _discover_metric_classes()

    if not metric_classes:
        print("[WARNING] No metric classes discovered")
        return []

    print(f"[INFO] Discovered {len(metric_classes)} metric classes: {list(metric_classes.keys())}")

    reconstructed_metrics = []

    for idx, metric_data in enumerate(metrics_list):
        metric_created = False
        print(f"[DEBUG] Processing metric {idx}: {metric_data}")

        # Empty dict - try metrics with no required parameters
        if not metric_data or metric_data == {}:
            # note (xinjieyao, 2025-12-08): empty init params are allowed for metrics, e.g. SuccessRateMetric.
            for metric_name, metric_class in metric_classes.items():
                try:
                    # Try to instantiate with no arguments
                    metric_instance = metric_class()
                    reconstructed_metrics.append(metric_instance)
                    metric_created = True
                    break
                except Exception as e:
                    print(f"[DEBUG] Failed to instantiate {metric_name}: {type(e).__name__}: {e}, skipping")
                    continue
        else:
            # Try to match metric data to metric class signatures
            for metric_name, metric_class in metric_classes.items():
                try:
                    sig = inspect.signature(metric_class.__init__)
                    params = {name: p for name, p in sig.parameters.items() if name != "self"}

                    # Prepare arguments for the metric class
                    args = {}

                    # Handle 'object' parameter specially - reconstruct Asset
                    if "object" in params and "object" in metric_data:
                        args["object"] = _reconstruct_asset_from_dict(metric_data["object"])

                    # Handle other parameters by direct mapping
                    for param_name, param in params.items():
                        if param_name == "object":
                            continue  # Already handled

                        # Try to find matching key in metric_data
                        # TODO(xinjieyao, 2025-12-08): check other metrics for required parameters.
                        if param_name in metric_data:
                            args[param_name] = metric_data[param_name]
                        elif param.default == inspect.Parameter.empty:
                            # Required parameter not found, can't use this class
                            print(f"[DEBUG] Required parameter {param_name} not found for {metric_name}, skipping")
                            continue

                    # Try to instantiate the metric
                    metric_instance = metric_class(**args)
                    reconstructed_metrics.append(metric_instance)
                    metric_created = True
                    break

                except Exception as e:
                    # This metric class doesn't match, try next one
                    print(f"[DEBUG] Failed to match {metric_name}: {e}")
                    continue

        if not metric_created:
            print(f"[WARNING] Could not match metric data to any known metric class: {metric_data}")

    return reconstructed_metrics


def _reconstruct_isaaclab_arena_env(arena_env_dict, cfg):
    """Reconstruct isaaclab_arena_env using already-deserialized top-level configs.

    This function constructs proper typed objects for embodiment, scene, and task
    from the YAML dictionary data, using the already-deserialized top-level configs.

    Args:
        arena_env_dict: The YAML dictionary for isaaclab_arena_env (for metadata like 'name')
        cfg: The main IsaacLabArenaManagerBasedRLEnvCfg with deserialized sections

    Returns:
        IsaacLabArenaEnvironment instance or None if arena_env_dict is empty
    """
    if not arena_env_dict:
        return None

    arena_env = IsaacLabArenaEnvironment()

    # Populate name from YAML
    if "name" in arena_env_dict:
        arena_env.name = arena_env_dict["name"]

    # Reconstruct embodiment
    if "embodiment" in arena_env_dict:
        arena_env.embodiment = reconstruct_embodiment(arena_env_dict["embodiment"], cfg)

    # Reconstruct scene
    if "scene" in arena_env_dict:
        arena_env.scene = reconstruct_scene(arena_env_dict["scene"], cfg)

    # Reconstruct task
    if "task" in arena_env_dict:
        arena_env.task = reconstruct_task(arena_env_dict["task"], cfg)

    # Handle teleop_device and orchestrator - keep as dicts for now since they're optional
    # TODO(xinjieyao, 2025-12-08): test on teleop device
    if "teleop_device" in arena_env_dict:
        arena_env.teleop_device = arena_env_dict["teleop_device"]

    if "orchestrator" in arena_env_dict:
        arena_env.orchestrator = arena_env_dict["orchestrator"]

    if "env_cfg_callback" in arena_env_dict:
        arena_env.env_cfg_callback = arena_env_dict["env_cfg_callback"]

    return arena_env


def load_env_cfg_from_yaml(yaml_path: str):
    """Load an IsaacLab Arena environment config from a YAML file.

    This function deserializes a config saved using dump_yaml(), handling complex
    nested structures, numpy types, and configclass objects.

    Metrics are automatically discovered from the isaaclab_arena/metrics folder and
    reconstructed by matching YAML data to their __init__ signatures. This supports
    custom user-defined metrics - just add them to the metrics folder.

    Args:
        yaml_path: Path to the YAML config file

    Returns:
        IsaacLabArenaManagerBasedRLEnvCfg: The deserialized environment configuration
            with metrics auto-discovered and reconstructed
    """
    # Register YAML constructors
    register_yaml_constructors()

    # Load YAML
    with open(yaml_path, encoding="utf-8") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Create base config
    cfg = IsaacLabArenaManagerBasedRLEnvCfg()

    # Extract and handle sections with dynamic configs
    for section_name, create_func in [
        ("scene", _create_scene_config),
        ("recorders", _create_recorders_config),
        ("actions", _create_actions_config),
        ("events", _create_events_config),
        ("observations", _create_observations_config),
    ]:
        section_dict = cfg_dict.pop(section_name, None)
        if section_dict is not None:
            setattr(cfg, section_name, create_func(section_dict))

    # Default observations if not present
    if not hasattr(cfg, "observations") or cfg.observations is None:
        cfg.observations = ObservationGroupCfg()

    # Handle terminations
    terminations_dict = cfg_dict.pop("terminations", None)
    if terminations_dict is not None:
        _create_terminations_config(cfg, terminations_dict)

    # Handle rewards, curriculum, commands - create dynamic configs
    for config_name, term_cfg_class in [
        ("rewards", RewardTermCfg),
        ("curriculum", CurriculumTermCfg),
        ("commands", CommandTermCfg),
    ]:
        config_dict = cfg_dict.pop(config_name, None)
        if config_dict is not None:
            if config_dict:  # Non-empty dict
                result_cfg = _create_dynamic_manager_config(
                    config_dict, None, term_cfg_class, f"{config_name.capitalize()}Cfg"
                )
            else:  # Empty dict
                EmptyCfg = make_configclass(f"{config_name.capitalize()}Cfg", [])
                result_cfg = EmptyCfg()
            setattr(cfg, config_name, result_cfg)

    # Handle metrics - reconstruct metric objects from YAML data
    metrics_list = cfg_dict.pop("metrics", None)
    if metrics_list is not None:
        cfg.metrics = _reconstruct_metrics_from_yaml(metrics_list)
    else:
        cfg.metrics = []

    # Handle XR config
    xr_dict = cfg_dict.pop("xr", None)
    if xr_dict is not None:
        xr_cfg = XrCfg()
        _populate_from_dict(xr_cfg, xr_dict)
        cfg.xr = xr_cfg

    # Reconstruct isaaclab_arena_env using already-deserialized top-level configs
    arena_env_dict = cfg_dict.pop("isaaclab_arena_env", None)
    if arena_env_dict is not None:
        cfg.isaaclab_arena_env = _reconstruct_isaaclab_arena_env(arena_env_dict, cfg)

    return cfg


def _convert_markers_to_spawner_configs(markers_dict):
    """Convert markers dictionary to proper spawner config objects (SphereCfg, etc.).

    Markers are spawner configs identified by their func field, not class_type.

    Args:
        markers_dict: Dictionary where each value is a marker dict with 'func' field

    Returns:
        Dictionary with each marker converted to its spawner config object
    """
    result_cfg = {}
    for marker_name, marker_dict in markers_dict.items():
        if isinstance(marker_dict, dict) and "func" in marker_dict:
            # Recursively convert nested configs first
            marker_dict_converted = _convert_funcs_in_dict(marker_dict)

            # Convert visual_material dict to PreviewSurfaceCfg if present
            if "visual_material" in marker_dict_converted and isinstance(
                marker_dict_converted["visual_material"], dict
            ):
                visual_mat = PreviewSurfaceCfg()
                _populate_from_dict(visual_mat, marker_dict_converted["visual_material"])
                marker_dict_converted["visual_material"] = visual_mat

            # Determine spawner config type from func
            func = marker_dict_converted.get("func")
            marker_cfg = None

            if func:
                func_name = func.__name__ if callable(func) else str(func)
                # Map func name to config class
                if "sphere" in func_name.lower():
                    marker_cfg = SphereCfg()
                elif "cone" in func_name.lower():
                    marker_cfg = ConeCfg()
                elif "cylinder" in func_name.lower():
                    marker_cfg = CylinderCfg()
                elif "cuboid" in func_name.lower() or "cube" in func_name.lower():
                    marker_cfg = CuboidCfg()
                elif "capsule" in func_name.lower():
                    marker_cfg = CapsuleCfg()

                if marker_cfg:
                    _populate_from_dict(marker_cfg, marker_dict_converted, skip_conversion=True)
                    result_cfg[marker_name] = marker_cfg
                else:
                    # Unknown marker type, keep as converted dict
                    result_cfg[marker_name] = marker_dict_converted
            else:
                result_cfg[marker_name] = marker_dict_converted
        else:
            result_cfg[marker_name] = marker_dict

    return result_cfg


def _convert_class_type_dict_to_config(items_dict, context_name=""):
    """Convert a dict of items with class_type to a dict of config objects.

    Args:
        items_dict: Dictionary where each value is a dict with 'class_type' field
        context_name: Context name for error messages (e.g. 'Actuator', 'Marker')

    Returns:
        Dictionary with each value converted to its config object
    """
    result_cfg = {}
    for item_name, item_dict in items_dict.items():
        if isinstance(item_dict, dict) and "class_type" in item_dict:
            # Recursively convert this item dict first (but NOT actuators, to avoid infinite recursion)
            item_dict_converted = item_dict.copy()
            for k, v in item_dict.items():
                if k == "class_type" and isinstance(v, str):
                    try:
                        item_dict_converted[k] = string_to_callable(v)
                    except (ImportError, AttributeError, ValueError):
                        item_dict_converted[k] = v
                elif k != "actuators" and isinstance(v, (dict, list)):
                    item_dict_converted[k] = _convert_funcs_in_dict(v)
                else:
                    item_dict_converted[k] = v

            class_type = item_dict_converted["class_type"]

            # Try to get the Cfg class directly by appending 'Cfg' to class name
            item_cfg = None
            if not isinstance(class_type, str):
                # class_type is already a class object
                # Try multiple approaches to get the config class
                class_module = class_type.__module__
                class_name = class_type.__name__
                cfg_class_name = class_name + "Cfg"

                # Approach 1: Try direct string_to_callable
                try:
                    cfg_class = string_to_callable(f"{class_module}:{cfg_class_name}")
                    item_cfg = cfg_class()
                    _populate_from_dict(item_cfg, item_dict_converted, skip_conversion=True)
                    result_cfg[item_name] = item_cfg
                except (ImportError, AttributeError, ValueError) as e:
                    print(
                        f"[DEBUG] Failed to instantiate {item_name} with {class_module}:{cfg_class_name}:"
                        f" {type(e).__name__}: {e}, skipping"
                    )
                    # Approach 2: Try patterns
                    config_patterns = _get_config_class_patterns(f"{class_module}:{class_name}")
                    for config_class_str in config_patterns:
                        try:
                            config_class = string_to_callable(config_class_str)
                            item_cfg = config_class()
                            _populate_from_dict(item_cfg, item_dict_converted, skip_conversion=True)
                            result_cfg[item_name] = item_cfg
                            break
                        except (ImportError, AttributeError, ValueError) as e2:
                            print(
                                f"[DEBUG] Failed to instantiate {item_name} with {config_class_str}:"
                                f" {type(e2).__name__}: {e2}, skipping"
                            )
                            continue
            else:
                # class_type is still a string
                config_patterns = _get_config_class_patterns(class_type)
                for config_class_str in config_patterns:
                    try:
                        config_class = string_to_callable(config_class_str)
                        item_cfg = config_class()
                        _populate_from_dict(item_cfg, item_dict_converted, skip_conversion=True)
                        result_cfg[item_name] = item_cfg
                        break
                    except (ImportError, AttributeError, ValueError) as e:
                        print(
                            f"[DEBUG] Failed to instantiate {item_name} with {config_class_str}:"
                            f" {type(e).__name__}: {e}, skipping"
                        )
                        continue

            if item_cfg is None:
                # If conversion failed, keep as converted dict
                result_cfg[item_name] = item_dict_converted
        else:
            # Not a dict with class_type, just convert funcs
            result_cfg[item_name] = _convert_funcs_in_dict(item_dict) if isinstance(item_dict, dict) else item_dict

    return result_cfg


def _convert_funcs_in_dict(data):
    """Recursively convert all 'func' and '*_class_type' string fields to callables and actuators to config objects in nested dicts."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if (key == "func" or key == "class_type" or key.endswith("_class_type")) and isinstance(value, str):
                try:
                    result[key] = string_to_callable(value)
                except (ImportError, AttributeError, ValueError) as e:
                    # If conversion fails, keep as string
                    print(f"Warning: Could not convert {key}='{value}' to callable: {e}")
                    result[key] = value
            elif key == "actuators" and isinstance(value, dict):
                # Convert actuators dict to proper config objects
                result[key] = _convert_class_type_dict_to_config(value, "Actuator")
            elif key == "markers" and isinstance(value, dict):
                # Convert markers dict (used in visualizer_cfg) to proper spawner config objects
                # Markers don't have class_type, they're spawner configs determined by func
                result[key] = _convert_markers_to_spawner_configs(value)
            elif isinstance(value, (dict, list)):
                result[key] = _convert_funcs_in_dict(value)
            else:
                result[key] = value
        return result
    elif isinstance(data, list):
        return [_convert_funcs_in_dict(item) for item in data]
    else:
        return data


def _populate_from_dict(obj, data_dict, skip_conversion=False):
    """Recursively populate an object from a dictionary for nested config structures.

    Args:
        obj: The object to populate
        data_dict: The dictionary with values
        skip_conversion: If True, skip the _convert_funcs_in_dict call (assume already converted)
    """
    if not isinstance(data_dict, dict):
        return

    # Convert all func strings to callables first (recursively)
    # This also handles actuators conversion
    if not skip_conversion:
        data_dict = _convert_funcs_in_dict(data_dict)

    # Replace all MISSING fields with None first
    _replace_all_missing(obj)

    for key, value in data_dict.items():
        if not hasattr(obj, key):
            # Dynamically add the attribute - just set the value
            setattr(obj, key, value)
        else:
            existing_attr = getattr(obj, key)
            is_missing = type(existing_attr).__name__ == "_MISSING_TYPE"

            if is_missing or existing_attr is None:
                # Just set the value directly
                object.__setattr__(obj, key, value)
            elif isinstance(value, dict) and hasattr(existing_attr, "__dict__") and not isinstance(existing_attr, dict):
                # Recurse into nested objects (only if it's a proper object, not a dict)
                _populate_from_dict(existing_attr, value)
            else:
                # Simple assignment
                setattr(obj, key, value)


def _replace_all_missing(obj):
    """Replace all MISSING fields in a dataclass with None."""
    if hasattr(obj, "__dataclass_fields__"):
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)
            if type(field_value).__name__ == "_MISSING_TYPE":
                object.__setattr__(obj, field_name, None)


def _create_dynamic_manager_config(cfg_dict, base_fields, term_cfg_class, config_name):
    """Create a dynamic manager config (terminations, rewards, etc.).

    Args:
        cfg_dict: Dictionary of term configurations
        base_fields: Base fields from the parent class (if any)
        term_cfg_class: The term config class (e.g., TerminationTermCfg, RewardTermCfg)
        config_name: Name for the dynamic config class

    Returns:
        Dynamic configclass instance
    """
    if not cfg_dict:
        return None

    # Create term instances
    term_instances = {}
    for key, term_dict in cfg_dict.items():
        # Convert funcs and scene entities in params
        # e.g. Event with pose in params is converted to Pose objects (params: pose: position_xyz and rotation_wxyz)
        # e.g. Termination with object_cfg in params is converted to SceneEntityCfg
        term_dict = _convert_funcs_in_dict(term_dict)
        if "params" in term_dict and isinstance(term_dict["params"], dict):
            term_dict["params"] = _convert_term_params_to_objects(term_dict["params"])

        term_cfg = term_cfg_class()
        _populate_from_dict(term_cfg, term_dict, skip_conversion=True)
        term_instances[key] = term_cfg

    # Create dynamic config class
    term_fields = [(key, term_cfg_class, inst) for key, inst in term_instances.items()]
    DynamicCfg = make_configclass(config_name, term_fields, bases=base_fields if base_fields else ())
    return DynamicCfg()


def _convert_spawn_to_config(asset_dict):
    """Convert spawn dictionary to proper spawn config object (UsdFileCfg, etc.).

    Also converts nested config objects within spawn (rigid_props, articulation_props, etc.)

    Args:
        asset_dict: Asset dictionary that may contain a 'spawn' dict

    Returns:
        Asset dictionary with spawn dict converted to proper config object
    """
    if not isinstance(asset_dict, dict) or "spawn" not in asset_dict:
        return asset_dict

    spawn_dict = asset_dict.get("spawn")
    if spawn_dict is None or not isinstance(spawn_dict, dict):
        return asset_dict

    # Convert nested config dicts to proper config objects
    spawn_dict_copy = spawn_dict.copy()

    # Handle rigid_props
    if "rigid_props" in spawn_dict_copy and isinstance(spawn_dict_copy["rigid_props"], dict):
        rigid_cfg = RigidBodyPropertiesCfg()
        _populate_from_dict(rigid_cfg, spawn_dict_copy["rigid_props"])
        spawn_dict_copy["rigid_props"] = rigid_cfg

    # Handle collision_props
    if "collision_props" in spawn_dict_copy and isinstance(spawn_dict_copy["collision_props"], dict):
        collision_cfg = CollisionPropertiesCfg()
        _populate_from_dict(collision_cfg, spawn_dict_copy["collision_props"])
        spawn_dict_copy["collision_props"] = collision_cfg

    # Handle mass_props
    if "mass_props" in spawn_dict_copy and isinstance(spawn_dict_copy["mass_props"], dict):
        mass_cfg = MassPropertiesCfg()
        _populate_from_dict(mass_cfg, spawn_dict_copy["mass_props"])
        spawn_dict_copy["mass_props"] = mass_cfg

    # Handle articulation_props
    if "articulation_props" in spawn_dict_copy and isinstance(spawn_dict_copy["articulation_props"], dict):
        articulation_cfg = ArticulationRootPropertiesCfg()
        _populate_from_dict(articulation_cfg, spawn_dict_copy["articulation_props"])
        spawn_dict_copy["articulation_props"] = articulation_cfg

    # Handle visual_material
    if "visual_material" in spawn_dict_copy and isinstance(spawn_dict_copy["visual_material"], dict):
        visual_mat_cfg = VisualMaterialCfg()
        _populate_from_dict(visual_mat_cfg, spawn_dict_copy["visual_material"])
        spawn_dict_copy["visual_material"] = visual_mat_cfg

    # Check if this is a camera spawn config (has spawn_camera func)
    func_str = spawn_dict_copy.get("func", "")
    is_camera_spawn = "spawn_camera" in str(func_str)

    # Remove fields that are invalid for camera spawn configs
    if is_camera_spawn:
        INVALID_CAMERA_SPAWN_FIELDS = {
            "activate_contact_sensors",
            "activateContactSensors",
            "rigid_props",
            "articulation_props",
            "collision_props",
        }
        for field in INVALID_CAMERA_SPAWN_FIELDS:
            spawn_dict_copy.pop(field, None)

    # Create the UsdFileCfg with converted nested configs
    spawn_cfg = UsdFileCfg()
    _populate_from_dict(spawn_cfg, spawn_dict_copy)

    # Create a copy of asset_dict with the converted spawn
    result = asset_dict.copy()
    result["spawn"] = spawn_cfg
    return result


def _convert_term_params_to_objects(params_dict):
    """Convert term parameter dictionaries to typed objects (Pose, SceneEntityCfg).

    Event and termination functions expect typed parameters, not plain dicts.
    This function pattern-matches parameter dictionaries and converts them to proper types:
    - Pose objects (dicts with 'position_xyz' and 'rotation_wxyz')
    - SceneEntityCfg objects (dicts with 'name' and 'joint_names')

    Note:
        If the parameter is not a Pose or SceneEntityCfg, it is returned as is.
    Args:
        params_dict: Dictionary of parameters that may contain Pose or SceneEntityCfg dicts

    Returns:
        Dictionary with Pose/SceneEntityCfg dicts converted to proper objects
    """
    result = {}
    for key, value in params_dict.items():
        # Check if this dict looks like a Pose (has position_xyz and rotation_wxyz)
        if isinstance(value, dict) and "position_xyz" in value and "rotation_wxyz" in value:
            # Convert to Pose
            result[key] = Pose(**value)
        # Check if this dict looks like a SceneEntityCfg (has 'name' and other typical fields)
        elif isinstance(value, dict) and "name" in value and "joint_names" in value:
            # Convert to SceneEntityCfg
            result[key] = SceneEntityCfg(**value)
        else:
            # e.g. isaaclab_observation_config: params: link_name: "right_wrist_yaw_link"
            # This is not a Pose or SceneEntityCfg, so we just return the value as is
            result[key] = value

    return result


def _create_config_from_class_type(item_name, item_dict, context="item"):
    """Create a config object from a dict with class_type field.

    E.g.for actuator:
        isaaclab.actuators.actuator_pd:ImplicitActuator -> isaaclab.actuators.actuator_cfg:ImplicitActuatorCfg
    E.g.for asset:
        isaaclab.assets.rigid_object.rigid_object:RigidObject -> isaaclab.assets.rigid_object_cfg:RigidObjectCfg
    E.g.for metric in recorder manager config:
        isaaclab_arena.metrics.success_rate:SuccessRecorder -> isaaclab_arena.metrics.success_rate_cfg:SuccessRecorderCfg

    Args:
        item_name: Name of the item
        item_dict: Dictionary containing class_type and other fields
        context: Context for error messages (e.g., 'asset', 'recorder term')

    Returns:
        Configured object matching the class_type
    """
    class_type = item_dict.get("class_type")
    if class_type is None:
        raise ValueError(f"{context} '{item_name}' has no class_type")

    # Get class_type string
    if not isinstance(class_type, str):
        class_type_str = f"{class_type.__module__}:{class_type.__name__}"
    else:
        class_type_str = class_type

    # Try to find and instantiate the config class
    config_patterns = _get_config_class_patterns(class_type_str)

    for config_class_str in config_patterns:
        try:
            # class type conversion
            config_class = string_to_callable(config_class_str)
            config_obj = config_class()
            # populate data fields
            _populate_from_dict(config_obj, item_dict, skip_conversion=True)
            return config_obj
        except (ImportError, AttributeError, ValueError):
            continue

    # All patterns failed
    raise ValueError(
        f"Failed to find config class for {context} '{item_name}' with class_type '{class_type_str}'. "
        f"Tried patterns: {config_patterns}"
    )


def _create_asset_config(
    asset_name: str, asset_dict: dict[str, Any]
) -> AssetBaseCfg | RigidObjectCfg | ArticulationCfg:
    """Create a single asset config object from dictionary.

    If no class_type, it's a AssetBaseCfg. If class_type is provided, it could be RigidObjectCfg, ArticulationCfg, AssetBaseCfg. Assets all come with
    spawn configuration, which is converted to proper config objects in the _convert_spawn_to_config function.

    Args:
        asset_name: Name of the asset
        asset_dict: Dictionary containing asset configuration

    Returns:
        Configured asset object (RigidObjectCfg, ArticulationCfg, or AssetBaseCfg)
    """
    # Convert funcs and spawn to proper configs
    # With conversion: spawn_cfg.func = <function spawn_from_usd>
    asset_dict = _convert_funcs_in_dict(asset_dict)
    # The spawn field has a known, specific structure with standard sub-properties (rigid_props, articulation_props, etc.) that need specific config classes.
    asset_dict = _convert_spawn_to_config(asset_dict)

    # If no class_type, it's a BASE asset
    if asset_dict.get("class_type") is None:
        asset_cfg = AssetBaseCfg()
        # string callables have been converted to callables by now, only poulating the items
        _populate_from_dict(asset_cfg, asset_dict, skip_conversion=True)
        return asset_cfg

    # Otherwise, use class_type to create the appropriate config
    return _create_config_from_class_type(asset_name, asset_dict, "asset")


def _create_scene_config(scene_dict: dict[str, Any]):
    """Create scene config with dynamic assets.

    If no assets are listed, a default InteractiveSceneCfg is returned.
    If assets are listed, an InteractiveSceneCfg is returned with those dynamically created assets added to the scene.

    Args:
        scene_dict: Dictionary containing scene configuration

    Returns:
        Dynamic SceneCfg with all assets properly configured
    """
    # Default scene config
    scene_cfg = InteractiveSceneCfg()
    if not scene_dict:
        return scene_cfg

    # Separate base fields from dynamic assets
    # e.g. num_envs, env_spacing, replicate_physics, filter_collisions, clone_in_fabric, etc.
    base_scene_fields = {f.name for f in fields(InteractiveSceneCfg)}
    base_fields_dict = {k: v for k, v in scene_dict.items() if k in base_scene_fields}
    # background/objects/destinations/robots/sensors etc. are added dynamically to the scene
    dynamic_assets_dict = {k: v for k, v in scene_dict.items() if k not in base_scene_fields}

    # Create asset instances
    asset_instances = {}
    for asset_name, asset_dict in dynamic_assets_dict.items():
        if isinstance(asset_dict, dict):
            asset_instances[asset_name] = _create_asset_config(asset_name, asset_dict)
        else:
            asset_instances[asset_name] = asset_dict

    # Create dynamic SceneCfg
    if asset_instances:
        asset_fields = [(name, type(inst), inst) for name, inst in asset_instances.items()]
        SceneCfg = make_configclass("SceneCfg", asset_fields, bases=(InteractiveSceneCfg,))
        scene_cfg = SceneCfg()

    # Populate base fields
    if base_fields_dict:
        _populate_from_dict(scene_cfg, base_fields_dict)

    return scene_cfg


def _create_recorders_config(recorders_dict: dict[str, Any]):
    """Create recorders config with dynamic terms. If no metrics are listed, a default
    RecorderManagerCfg is returned. If metrics are listed, a RecorderManagerCfg is returned with
    those metrics recorder. Recorder terms are added dynamically to the recorder manager config.

    Args:
        recorders_dict: Dictionary containing recorder configuration

    Returns:
        Dynamic RecorderManagerCfg with metircs recorder terms added if any
    """
    recorder_cfg = RecorderManagerBaseCfg()
    if not recorders_dict:
        return recorder_cfg

    # Separate base fields from dynamic terms
    base_recorder_fields = {f.name for f in fields(RecorderManagerBaseCfg)}
    base_fields_dict = {k: v for k, v in recorders_dict.items() if k in base_recorder_fields}
    dynamic_terms_dict = {k: v for k, v in recorders_dict.items() if k not in base_recorder_fields}

    # Create recorder term configs using class_type
    recorder_term_instances = {}
    # recorder manager config contains metrics listed in the recorder manager config if any
    for term_name, term_dict in dynamic_terms_dict.items():
        if isinstance(term_dict, dict) and "class_type" in term_dict and term_dict["class_type"] is not None:
            term_dict = _convert_funcs_in_dict(term_dict)
            recorder_term_instances[term_name] = _create_config_from_class_type(term_name, term_dict, "recorder term")
        else:
            raise ValueError(f"Recorder term '{term_name}' is not a dictionary with a 'class_type' field")

    # Create dynamic RecorderManagerCfg
    if recorder_term_instances:
        recorder_fields = [(name, type(inst), inst) for name, inst in recorder_term_instances.items()]
        RecorderManagerCfg = make_configclass("RecorderManagerCfg", recorder_fields, bases=(RecorderManagerBaseCfg,))
        recorder_cfg = RecorderManagerCfg()

    # Populate base fields
    if base_fields_dict:
        _populate_from_dict(recorder_cfg, base_fields_dict)

    return recorder_cfg


def _create_actions_config(actions_dict: dict[str, Any]):
    """Create actions config with dynamic terms.

    Args:
        actions_dict: Dictionary containing action terms

    Returns:
        Dynamic ActionsCfg with all terms properly configured
    """
    actions_cfg = None
    if not actions_dict:
        return actions_cfg

    # Create action term configs using class_type
    action_terms = {}
    for term_name, term_dict in actions_dict.items():
        if isinstance(term_dict, dict) and "class_type" in term_dict and term_dict["class_type"] is not None:
            term_dict = _convert_funcs_in_dict(term_dict)
            action_terms[term_name] = _create_config_from_class_type(term_name, term_dict, "action term")
        else:
            raise ValueError(f"Action term '{term_name}' is not a dictionary with a 'class_type' field")

    # Create dynamic ActionsCfg
    if action_terms:
        action_fields = [(name, type(term), term) for name, term in action_terms.items()]
        ActionsCfg = make_configclass("ActionsCfg", action_fields)
        actions_cfg = ActionsCfg()

    return actions_cfg


def _create_events_config(events_dict: dict[str, Any]):
    """Create events config with dynamic terms.

    Args:
        events_dict: Dictionary containing event terms

    Returns:
        Dynamic EventsCfg with all terms properly configured
    """
    if not events_dict:
        return None

    return _create_dynamic_manager_config(events_dict, None, EventTermCfg, "EventsCfg")


def _create_observations_config(observations_dict: dict[str, Any]):
    """Create observations config with proper structure.

    Args:
        observations_dict: Dictionary containing observation groups

    Returns:
        Dynamic ObservationCfg with all groups properly configured

    If no observation groups are listed, a default ObservationGroupCfg is returned.
    If observation groups are listed, an ObservationCfg is returned with those observation groups added to the observation config.
    """
    if not observations_dict:
        return ObservationGroupCfg()

    # Get base fields of ObservationGroupCfg
    base_group_field_names = {f.name for f in fields(ObservationGroupCfg)}

    # Create observation groups
    obs_groups = {}
    for group_name, group_dict in observations_dict.items():
        if not isinstance(group_dict, dict):
            raise ValueError(f"Observation group '{group_name}' is not a dictionary")
            # obs_groups[group_name] = group_dict
            # continue

        # Separate base attributes from observation terms
        group_base_attrs = {k: v for k, v in group_dict.items() if k in base_group_field_names}
        obs_terms_dict = {k: v for k, v in group_dict.items() if k not in base_group_field_names}

        # Create observation term configs
        obs_term_instances = {}
        for term_name, term_dict in obs_terms_dict.items():
            if isinstance(term_dict, dict) and "func" in term_dict:
                term_dict = _convert_funcs_in_dict(term_dict)
                if "params" in term_dict and isinstance(term_dict["params"], dict):
                    term_dict["params"] = _convert_term_params_to_objects(term_dict["params"])
                term_cfg = ObservationTermCfg()
                _populate_from_dict(term_cfg, term_dict, skip_conversion=True)
                obs_term_instances[term_name] = term_cfg

        # Create dynamic group config or use base
        if obs_term_instances:
            obs_term_fields = [(name, ObservationTermCfg, cfg) for name, cfg in obs_term_instances.items()]
            ObsGroupCfg = make_configclass(
                f"{group_name.capitalize()}ObsGroupCfg", obs_term_fields, bases=(ObservationGroupCfg,)
            )
            obs_group_cfg = ObsGroupCfg()
        else:
            obs_group_cfg = ObservationGroupCfg()

        # Populate base attributes
        if group_base_attrs:
            _populate_from_dict(obs_group_cfg, group_base_attrs)

        obs_groups[group_name] = obs_group_cfg

    # Create dynamic ObservationCfg with all groups
    if obs_groups:
        obs_fields = [(name, type(group), group) for name, group in obs_groups.items()]
        ObservationCfg = make_configclass("ObservationCfg", obs_fields)
        return ObservationCfg()

    return ObservationGroupCfg()


def _create_terminations_config(cfg, terminations_dict):
    """Create terminations config."""
    cfg.terminations = _create_dynamic_manager_config(terminations_dict, None, TerminationTermCfg, "TerminationCfg")
