# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for reconstructing IsaacLabArenaEnvironment components from YAML."""

# IsaacLab Arena imports
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.generic_task import GenericTask
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose


def _extract_config_from_merged_cfg(embodiment_dict, dict_key, cfg, cfg_attr_name):
    """Extract a config from the merged cfg by looking up the first key in the dict.

    This helper function handles the common pattern of:
    - Check if dict_key exists in embodiment_dict
    - Iterate through the keys in embodiment_dict[dict_key]
    - Use getattr to extract the corresponding config from cfg.cfg_attr_name

    Args:
        embodiment_dict: Dictionary containing embodiment configuration
        dict_key: Key to look up in embodiment_dict (e.g., 'scene_config')
        cfg: The main IsaacLabArenaManagerBasedRLEnvCfg
        cfg_attr_name: Attribute name in cfg to extract from (e.g., 'scene')

    Returns:
        The extracted config object, or None if not found
    """
    if dict_key in embodiment_dict and embodiment_dict[dict_key] is not None:
        #TODO (xinjieyao, 2025-12-08): Handle mimic_env
        if dict_key == 'mimic_env':
            return None
        for config_name, _ in embodiment_dict[dict_key].items():
            cfg_object = getattr(cfg, cfg_attr_name, None)
            if cfg_object is not None:
                return getattr(cfg_object, config_name, None)
    return None


def reconstruct_embodiment(embodiment_dict, cfg):
    """Reconstruct EmbodimentBase from YAML dictionary.

    Stores embodiment-specific configs from embodiment_dict. These are the individual
    contributions from the embodiment that will be combined with scene and task configs.
    Uses deserialization functions to convert YAML dicts to proper config objects.

    Args:
        embodiment_dict: Dictionary containing embodiment configuration and metadata
        cfg: The main IsaacLabArenaManagerBasedRLEnvCfg with deserialized top-level configs

    Returns:
        EmbodimentBase instance populated with embodiment-specific configs
    """
    # Extract initialization parameters from YAML
    enable_cameras = embodiment_dict.get('enable_cameras', False)
    initial_pose_dict = embodiment_dict.get('initial_pose')
    initial_pose = None
    if initial_pose_dict:
        initial_pose = Pose(
            position_xyz=tuple(initial_pose_dict.get('position_xyz', (0, 0, 0))),
            rotation_wxyz=tuple(initial_pose_dict.get('rotation_wxyz', (1, 0, 0, 0)))
        )

    # Create embodiment instance
    embodiment = EmbodimentBase(
        enable_cameras=enable_cameras,
        initial_pose=initial_pose
    )

    # Camera config needs special handling
    if 'camera_config' in embodiment_dict:
        embodiment.camera_config = _deserialize_camera_config(embodiment_dict['camera_config'])

    # Define config mappings: (dict_key, cfg_attr, embodiment_attr)
    config_mappings = [
        ('scene_config', 'scene', 'scene_config'),
        ('action_config', 'actions', 'action_config'),
        ('observation_config', 'observations', 'observation_config'),
        ('event_config', 'events', 'event_config'),
        ('reward_config', 'rewards', 'reward_config'),
        ('curriculum_config', 'curriculum', 'curriculum_config'),
        ('command_config', 'commands', 'command_config'),
        ('termination_cfg', 'terminations', 'termination_cfg'),
        ('xr', 'xr', 'xr'),
        ('mimic_env', 'mimic_env', 'mimic_env'),
    ]

    # Extract all configs using a loop
    for dict_key, cfg_attr, embodiment_attr in config_mappings:
        config = _extract_config_from_merged_cfg(embodiment_dict, dict_key, cfg, cfg_attr)
        if config is not None:
            setattr(embodiment, embodiment_attr, config)

    return embodiment


def _extract_asset_metadata(asset_data):
    """Extract common asset metadata from YAML dictionary.

    Args:
        asset_data: Dictionary containing asset data

    Returns:
        Tuple of (tags, prim_path, usd_path, scale, object_type, initial_pose)
    """
    tags = asset_data.get('tags')
    prim_path = asset_data.get('prim_path')
    usd_path = asset_data.get('usd_path')
    scale = tuple(asset_data.get('scale', (1.0, 1.0, 1.0)))

    # Extract object type
    asset_type_enum_dict = asset_data.get('object_type', {})
    object_type_value = asset_type_enum_dict.get('_value_', 'base')
    object_type = {
        'base': ObjectType.BASE,
        'rigid': ObjectType.RIGID,
        'articulation': ObjectType.ARTICULATION,
        'spawner': ObjectType.SPAWNER,
    }.get(object_type_value, ObjectType.BASE)

    # Extract initial pose
    initial_pose_dict = asset_data.get('initial_pose')
    initial_pose = None
    if initial_pose_dict:
        initial_pose = Pose(
            position_xyz=tuple(initial_pose_dict.get('position_xyz', (0, 0, 0))),
            rotation_wxyz=tuple(initial_pose_dict.get('rotation_wxyz', (1, 0, 0, 0)))
        )

    return tags, prim_path, usd_path, scale, object_type, initial_pose


def reconstruct_scene(scene_dict, cfg):
    """Reconstruct Scene from YAML dictionary.

    Reconstructs Asset objects and stores scene-specific configs from scene_dict.
    These are the individual contributions from the scene that will be combined.

    Args:
        scene_dict: Dictionary containing scene metadata, assets, and configs
        cfg: The main IsaacLabArenaManagerBasedRLEnvCfg (for reference, not directly used)

    Returns:
        Scene instance populated with assets and scene-specific configs
    """
    # Create scene instance
    scene = Scene()

    # Reconstruct assets from the metadata in scene_dict and configs from cfg.scene
    assets_dict = scene_dict.get('assets', {})

    # First pass: Create regular assets (Object, Background)
    for asset_name, asset_data in assets_dict.items():
        if not isinstance(asset_data, dict) or 'parent_asset' in asset_data:
            continue

        asset_cfg = getattr(cfg.scene, asset_name, None)
        if asset_cfg is None:
            continue

        # Extract metadata
        tags, prim_path, usd_path, scale, object_type, initial_pose = _extract_asset_metadata(asset_data)

        try:
            # Create the appropriate Asset type
            if tags and 'background' in tags:
                asset = Background(
                    name=asset_name,
                    usd_path=usd_path,
                    object_min_z=asset_data.get('object_min_z', -0.2),
                    prim_path=prim_path,
                    initial_pose=initial_pose,
                    scale=scale,
                    tags=tags
                )
            else:
                asset = Object(
                    name=asset_name,
                    prim_path=prim_path,
                    object_type=object_type,
                    usd_path=usd_path,
                    scale=scale,
                    initial_pose=initial_pose,
                    tags=tags
                )

            asset.object_cfg = asset_cfg
            scene.add_asset(asset)
        except Exception as exc:
            raise Exception(f"Failed to create asset {asset_name}: {exc}") from exc

    # Second pass: Create ObjectReference assets (they need parent assets to exist first)
    for asset_name, asset_data in assets_dict.items():
        if not isinstance(asset_data, dict) or 'parent_asset' not in asset_data:
            continue

        asset_cfg = getattr(cfg.scene, asset_name, None)
        if asset_cfg is None:
            continue

        try:
            # Get parent asset from scene
            parent_asset_name = asset_data.get('parent_asset', {}).get('_name')
            if parent_asset_name is None:
                continue

            parent_asset = scene.assets.get(parent_asset_name)
            if parent_asset is None:
                raise ValueError(f"Parent asset '{parent_asset_name}' not found for '{asset_name}'")

            # Extract metadata
            tags, prim_path, _, _, object_type, _ = _extract_asset_metadata(asset_data)

            # Create ObjectReference
            asset_ref = ObjectReference(
                parent_asset=parent_asset,
                name=asset_name,
                prim_path=prim_path,
                object_type=object_type,
                tags=tags
            )

            asset_ref.object_cfg = asset_cfg
            scene.add_asset(asset_ref)
        except Exception as exc:
            raise Exception(f"Failed to create ObjectReference {asset_name}: {exc}") from exc

    # Define config mappings: (dict_key, cfg_attr, scene_attr)
    config_mappings = [
        ('observation_cfg', 'observations', 'observation_cfg'),
        ('events_cfg', 'events', 'events_cfg'),
        ('termination_cfg', 'terminations', 'termination_cfg'),
        ('rewards_cfg', 'rewards', 'rewards_cfg'),
        ('curriculum_cfg', 'curriculum', 'curriculum_cfg'),
        ('commands_cfg', 'commands', 'commands_cfg'),
    ]

    # Extract all configs using a loop
    for dict_key, cfg_attr, scene_attr in config_mappings:
        config = _extract_config_from_merged_cfg(scene_dict, dict_key, cfg, cfg_attr)
        if config is not None:
            setattr(scene, scene_attr, config)

    return scene


def reconstruct_task(task_dict, cfg):
    """Reconstruct TaskBase from YAML dictionary.

    Stores task-specific configs from task_dict. These are the individual
    contributions from the task that will be combined with embodiment and scene.
    Uses deserialization functions to convert YAML dicts to proper config objects.

    Args:
        task_dict: Dictionary containing task metadata and configs
        cfg: The main IsaacLabArenaManagerBasedRLEnvCfg with deserialized top-level configs

    Returns:
        TaskBase-like object with task-specific data and configs
    """
    # Extract task-specific configs using the helper function
    # Define config mappings: (dict_key, cfg_attr)
    config_keys = [
        ('scene_config', 'scene'),
        ('events_cfg', 'events'),
        ('termination_cfg', 'terminations'),
        ('observation_cfg', 'observations'),
        ('rewards_cfg', 'rewards'),
        ('curriculum_cfg', 'curriculum'),
        ('commands_cfg', 'commands'),
    ]

    # Extract all configs
    extracted_configs = {}
    for dict_key, cfg_attr in config_keys:
        config = _extract_config_from_merged_cfg(task_dict, dict_key, cfg, cfg_attr)
        extracted_configs[dict_key] = config

    return GenericTask(
        scene_cfg=extracted_configs.get('scene_config'),
        events_cfg=extracted_configs.get('events_cfg'),
        termination_cfg=extracted_configs.get('termination_cfg'),
        observation_cfg=extracted_configs.get('observation_cfg'),
        rewards_cfg=extracted_configs.get('rewards_cfg'),
        curriculum_cfg=extracted_configs.get('curriculum_cfg'),
        commands_cfg=extracted_configs.get('commands_cfg'),
        episode_length_s=task_dict.get('episode_length_s'),
    )


def _deserialize_camera_config(camera_config_dict):
    """Deserialize camera config dictionary into proper camera sensor config objects.

    Camera configs contain sensor configurations (e.g., TiledCamera) that need to be
    properly instantiated with the correct config classes.

    Args:
        camera_config_dict: Dictionary containing camera configurations
    Returns:
        Dynamically created config class with camera sensor configs
    """
    # Import here to avoid circular dependency
    from isaaclab_arena.utils.config_serialization import _create_asset_config

    # Fields that should NOT be in camera configs (these are for spawn configs of robots/objects)
    INVALID_CAMERA_FIELDS = {
        'activateContactSensors', 'activate_contact_sensors',
        'rigid_props', 'articulation_props', 'collision_props',
        'actuators', 'joint_names', 'fixed_tendons',
    }

    def _filter_dict_recursive(data):
        """Recursively filter out invalid fields from nested dicts."""
        if not isinstance(data, dict):
            return data
        
        filtered = {}
        for k, v in data.items():
            # Skip invalid fields
            if k in INVALID_CAMERA_FIELDS:
                continue
            # Recursively filter nested dicts
            if isinstance(v, dict):
                filtered[k] = _filter_dict_recursive(v)
            else:
                filtered[k] = v
        return filtered

    # Deserialize each camera sensor configuration
    camera_configs = {}
    for camera_name, camera_data in camera_config_dict.items():
        if camera_name.startswith('_'):
            # Skip metadata fields like _is_tiled_camera, _camera_offset
            continue

        if isinstance(camera_data, dict):
            # Filter out invalid fields recursively (including in spawn config)
            filtered_camera_data = _filter_dict_recursive(camera_data)
            
            # Deserialize camera sensor config using _create_asset_config
            camera_configs[camera_name] = _create_asset_config(camera_name, filtered_camera_data)
        else:
            # Keep as-is if not a dict
            camera_configs[camera_name] = camera_data

    # Create dynamic config class containing all cameras
    if camera_configs:
        camera_fields = [(name, type(cfg), cfg) for name, cfg in camera_configs.items()]
        CameraConfigClass = make_configclass('CameraConfig', camera_fields)
        return CameraConfigClass()

    return None

