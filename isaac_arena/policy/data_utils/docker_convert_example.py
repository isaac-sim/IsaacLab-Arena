#!/usr/bin/env python3
"""
Docker-friendly example for running HDF5 to LeRobot conversion.

This script shows how to properly configure paths and handle data conversion
when running inside a Docker container.
"""

import os
from pathlib import Path
import sys

# Add isaac_arena to Python path
isaac_arena_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(isaac_arena_path))

from isaac_arena.policy.config.dataset_config import Gr00tDatasetConfig
from isaac_arena.policy.data_utils.convert_hdf5_to_lerobot import convert_hdf5_dataset_to_lerobot


def create_docker_config(
    data_root: str = "/workspaces/data",  # Docker mount point
    hdf5_name: str = "demo_data.hdf5",
    task_name: str = "manipulation_task",
    language_instruction: str = "",
    robot_type: str = "unitree_g1"
) -> Gr00tDatasetConfig:
    """
    Create a Docker-friendly configuration.
    
    Args:
        data_root: Path where data is mounted in Docker (default: /workspaces/data)
        hdf5_name: Name of the HDF5 file
        task_name: Task identifier
        language_instruction: Natural language task description
        robot_type: Robot type identifier
    
    Returns:
        Configured Gr00tDatasetConfig instance
    """
    
    # Use Docker-friendly paths
    config = Gr00tDatasetConfig(
        data_root=Path(data_root),
        hdf5_name=hdf5_name,
        language_instruction=language_instruction,
        robot_type=robot_type,
    )
    
    return config


def run_docker_conversion():
    """Run HDF5 to LeRobot conversion with Docker-friendly settings."""
    
    print("🐳 Running HDF5 to LeRobot conversion in Docker")
    print("=" * 60)
    
    # Get configuration from environment variables or defaults
    data_root = os.environ.get("DATA_ROOT", "/workspaces/data")
    hdf5_name = os.environ.get("HDF5_NAME", "demo_data.hdf5") 
    task_name = os.environ.get("TASK_NAME", "manipulation_task")
    language_instruction = os.environ.get("LANGUAGE_INSTRUCTION", "")
    robot_type = os.environ.get("ROBOT_TYPE", "unitree_g1")
    
    print(f"Data Root: {data_root}")
    print(f"HDF5 File: {hdf5_name}")
    print(f"Task: {task_name}")
    print(f"Robot Type: {robot_type}")
    print(f"Instruction: {language_instruction}")
    print("-" * 60)
    
    try:
        # Create Docker-friendly configuration
        config = create_docker_config(
            data_root=data_root,
            hdf5_name=hdf5_name,
            task_name=task_name,
            language_instruction=language_instruction,
            robot_type=robot_type
        )
        
        print("✅ Configuration created successfully")
        
        # Validate paths exist
        if not config.data_root.exists():
            print(f"❌ Error: Data root does not exist: {config.data_root}")
            print("💡 Make sure to mount your data directory to Docker:")
            print(f"   docker run -v /host/path/to/data:{data_root} ...")
            return False
            
        if not config.hdf5_file_path.exists():
            print(f"❌ Error: HDF5 file does not exist: {config.hdf5_file_path}")
            return False
        
        # Run the conversion
        print("🚀 Starting conversion...")
        convert_hdf5_dataset_to_lerobot(config)
        print("✅ Conversion completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("🔍 Common Docker issues:")
        print("   1. Mount data directory: -v /host/data:/workspaces/data")
        print("   2. Check file permissions in mounted volumes")
        print("   3. Ensure ffmpeg is installed (should be fixed in Dockerfile)")
        print("   4. Verify config files exist in the container")
        return False


def print_docker_usage():
    """Print Docker usage instructions."""
    
    print("\n🐳 DOCKER USAGE INSTRUCTIONS")
    print("=" * 60)
    print("1. Build the Docker image:")
    print("   docker build -f docker/Dockerfile.isaac_arena -t isaac_arena .")
    print()
    print("2. Run with data mounted:")
    print("   docker run -it \\")
    print("     -v /path/to/your/data:/workspaces/data \\")
    print("     -e HDF5_NAME=your_file.hdf5 \\")
    print("     -e LANGUAGE_INSTRUCTION='Your task instruction' \\")
    print("     isaac_arena \\")
    print("     python isaac_arena/policy/data_utils/docker_convert_example.py")
    print()
    print("3. Environment variables you can set:")
    print("   - DATA_ROOT: Path to data in container (default: /workspaces/data)")
    print("   - HDF5_NAME: Name of HDF5 file (required)")
    print("   - TASK_NAME: Task identifier (default: manipulation_task)")  
    print("   - LANGUAGE_INSTRUCTION: Task description (default: empty)")
    print("   - ROBOT_TYPE: Robot type (default: unitree_g1)")
    print()
    print("4. Alternative - direct script usage:")
    print("   python isaac_arena/policy/data_utils/convert_hdf5_to_lerobot.py \\")
    print("     --data_root /workspaces/data \\")
    print("     --hdf5_name your_file.hdf5 \\")
    print("     --language_instruction 'Your task instruction'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker-friendly HDF5 to LeRobot conversion")
    parser.add_argument("--help-docker", action="store_true", help="Show Docker usage instructions")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without running conversion")
    
    args = parser.parse_args()
    
    if args.help_docker:
        print_docker_usage()
    elif args.dry_run:
        config = create_docker_config()
        print("Configuration (dry run):")
        print(f"  Data root: {config.data_root}")
        print(f"  HDF5 file: {config.hdf5_file_path}")
        print(f"  Output dir: {config.lerobot_data_dir}")
    else:
        success = run_docker_conversion()
        if not success:
            print_docker_usage()
            sys.exit(1)
