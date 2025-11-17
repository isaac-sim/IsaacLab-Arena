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

"""Script to inspect HDF5 dataset files."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.utils.datasets import HDF5DatasetFileHandler


def print_dict_structure(data, prefix="", max_depth=3, current_depth=0):
    """Recursively print dictionary structure with shape information."""
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                print_dict_structure(value, prefix + "  ", max_depth, current_depth + 1)
            else:
                shape = list(value.shape) if hasattr(value, "shape") else "N/A"
                dtype = value.dtype if hasattr(value, "dtype") else type(value).__name__
                print(f"{prefix}{key}: shape={shape}, dtype={dtype}")
    else:
        shape = list(data.shape) if hasattr(data, "shape") else "N/A"
        dtype = data.dtype if hasattr(data, "dtype") else type(data).__name__
        print(f"{prefix}shape={shape}, dtype={dtype}")


def find_camera_images(data, prefix=""):
    """Recursively find camera image data in the episode data structure."""
    cameras = {}
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                cameras.update(find_camera_images(value, current_path))
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                # Check if this looks like an image (has 3-4 dimensions and last dim is 1-4)
                if hasattr(value, "shape") and len(value.shape) >= 3:
                    shape = value.shape
                    # Skip transformation matrices (4x4 matrices) and other non-image data
                    # Images should have: (T, H, W, C) or (H, W, C) where H, W are typically > 32
                    # and C is 1-4 (for grayscale, RGB, RGBA)
                    if len(shape) >= 3:
                        # Check if it's likely a transformation matrix (small square matrices)
                        if len(shape) == 3 and shape[-2] == 4 and shape[-1] == 4:
                            # Skip 4x4 transformation matrices
                            continue
                        
                        # Check if it looks like image data
                        # For shape (T, H, W, C) or (H, W, C), check that H and W are reasonable image sizes
                        if len(shape) == 4:
                            # (T, H, W, C) format
                            h, w, c = shape[1], shape[2], shape[3]
                        elif len(shape) == 3:
                            # (H, W, C) or (T, H, W) format
                            if shape[-1] <= 4:
                                # (H, W, C) format
                                h, w, c = shape[0], shape[1], shape[2]
                            else:
                                # (T, H, W) format - grayscale
                                h, w, c = shape[1], shape[2], 1
                        else:
                            continue

                        # Check if this is likely a camera image by key name first
                        is_camera_key = "cam" in key.lower() or "cam" in current_path.lower()
                        if is_camera_key or h * w > 10000:
                            cameras[current_path] = {
                                "shape": shape,
                                "dtype": value.dtype if hasattr(value, "dtype") else type(value).__name__,
                                "data": value.cpu().numpy() if isinstance(value, torch.Tensor) else value,
                            }
    return cameras


def display_image(image_data, title="Image", save_path=None):
    """Display an image from tensor/array data.
    
    Args:
        image_data: Image data as numpy array or torch tensor
        title: Title for the image
        save_path: Optional path to save the image
    """
    # Convert to numpy if needed
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.cpu().numpy()
    
    # Handle different image shapes
    if len(image_data.shape) == 4:
        # (T, H, W, C) - take first frame
        image_data = image_data[0]
    elif len(image_data.shape) == 3:
        # (H, W, C) or (H, W) - use as is
        pass
    elif len(image_data.shape) == 2:
        # (H, W) - add channel dimension
        image_data = image_data[..., None]
    else:
        print(f"Warning: Unexpected image shape {image_data.shape}, cannot display")
        return
    
    # Normalize to 0-1 range if needed
    if image_data.dtype == np.uint8:
        image_data = image_data.astype(np.float32) / 255.0
    elif image_data.max() > 1.0:
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    
    # Handle different channel counts
    if image_data.shape[-1] == 1:
        # Grayscale
        plt.imshow(image_data[..., 0], cmap="gray")
    elif image_data.shape[-1] == 3:
        # RGB
        plt.imshow(image_data)
    elif image_data.shape[-1] == 4:
        # RGBA - show RGB only
        plt.imshow(image_data[..., :3])
    else:
        print(f"Warning: Unexpected number of channels {image_data.shape[-1]}, cannot display")
        return
    
    plt.title(title)
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        print(f"Image saved to: {save_path}")
    else:
        plt.show()


def inspect_dataset(
    dataset_file: str,
    episode_index: int | None = None,
    show_structure: bool = False,
    show_images: bool = False,
    camera_name: str | None = None,
    time_step: int | None = None,
    save_image: str | None = None,
):
    """Inspect an HDF5 dataset file.

    Args:
        dataset_file: Path to the HDF5 dataset file.
        episode_index: Optional index of episode to inspect in detail.
        show_structure: Whether to show the data structure of episodes.
        show_images: Whether to display camera images.
        camera_name: Name of camera to display (if None, list all available cameras).
        time_step: Time step to display (if None, display first frame).
        save_image: Optional path to save the displayed image.
    """
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"The dataset file {dataset_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(dataset_file)

    # Basic information
    env_name = dataset_file_handler.get_env_name()
    num_episodes = dataset_file_handler.get_num_episodes()
    episode_names = list(dataset_file_handler.get_episode_names())

    print("=" * 80)
    print("Dataset Information")
    print("=" * 80)
    print(f"File: {dataset_file}")
    print(f"Environment Name: {env_name}")
    print(f"Number of Episodes: {num_episodes}")
    print()

    if num_episodes == 0:
        print("No episodes found in the dataset.")
        dataset_file_handler.close()
        return

    # List episode names
    print("Episode Names:")
    print("-" * 80)
    for i, name in enumerate(episode_names[:20]):  # Show first 20
        print(f"  [{i:4d}] {name}")
    if num_episodes > 20:
        print(f"  ... and {num_episodes - 20} more episodes")
    print()

    # Show episode attributes summary
    print("Episode Summary:")
    print("-" * 80)
    success_count = 0
    failed_count = 0
    total_steps = 0

    for i, episode_name in enumerate(episode_names):
        episode = dataset_file_handler.load_episode(episode_name, device="cpu")
        if episode.success is not None:
            if episode.success:
                success_count += 1
            else:
                failed_count += 1

        if "actions" in episode.data:
            num_steps = len(episode.data["actions"])
            total_steps += num_steps
        else:
            num_steps = 0

        if i < 5:  # Show details for first 5 episodes
            success_str = f"success={episode.success}" if episode.success is not None else "success=N/A"
            seed_str = f"seed={episode.seed}" if episode.seed is not None else "seed=N/A"
            print(f"  [{i:4d}] {episode_name}: steps={num_steps:4d}, {success_str}, {seed_str}")

    print()
    print(f"Total Steps: {total_steps}")
    if success_count + failed_count > 0:
        print(f"Successful Episodes: {success_count}")
        print(f"Failed Episodes: {failed_count}")
        print(f"Success Rate: {success_count / (success_count + failed_count) * 100:.2f}%")
    print()

    # Inspect specific episode if requested
    if episode_index is not None:
        if episode_index < 0 or episode_index >= num_episodes:
            print(f"Error: Episode index {episode_index} is out of range (0-{num_episodes-1})")
        else:
            episode_name = episode_names[episode_index]
            episode = dataset_file_handler.load_episode(episode_name, device="cpu")

            print("=" * 80)
            print(f"Episode Details: [{episode_index}] {episode_name}")
            print("=" * 80)
            print(f"Success: {episode.success}")
            print(f"Seed: {episode.seed}")
            print(f"Environment ID: {episode.env_id}")
            print()

            if show_structure:
                print("Data Structure:")
                print("-" * 80)
                print_dict_structure(episode.data, max_depth=5)
                print()
            else:
                print("Data Keys:")
                print("-" * 80)

                def print_keys(data, prefix=""):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            print(f"{prefix}{key}/")
                            print_keys(value, prefix + "  ")
                        else:
                            try:
                                shape = list(value.shape) if hasattr(value, "shape") else "N/A"
                                dtype = value.dtype if hasattr(value, "dtype") else type(value).__name__
                                print(f"{prefix}{key}: shape={shape}, dtype={dtype}")
                            except Exception:
                                print(f"{prefix}{key}: {type(value).__name__}")

                print_keys(episode.data)
            
            # Find and report camera images (even without --show_images)
            cameras = find_camera_images(episode.data)
            if cameras:
                print()
                print("Camera Images Found:")
                print("-" * 80)
                for cam_path, cam_info in cameras.items():
                    print(f"  - {cam_path}: shape={cam_info['shape']}, dtype={cam_info['dtype']}")
                print()
                print("  (Use --show_images to display these images)")
                print()
            
            # Handle image display
            if show_images:
                print()
                print("=" * 80)
                print("Camera Images")
                print("=" * 80)
                
                # Use cameras already found above, or find them if not already found
                if not cameras:
                    cameras = find_camera_images(episode.data)
                
                if not cameras:
                    print("No camera images found in episode data.")
                    print("Looking for keys containing 'camera' or image-like tensors...")
                else:
                    print(f"Found {len(cameras)} camera(s):")
                    for cam_path, cam_info in cameras.items():
                        print(f"  - {cam_path}: shape={cam_info['shape']}, dtype={cam_info['dtype']}")
                    
                    print()
                    
                    # Display images
                    if camera_name:
                        # Find matching camera
                        matching_cams = {k: v for k, v in cameras.items() if camera_name in k}
                        if not matching_cams:
                            print(f"Error: Camera '{camera_name}' not found.")
                            print(f"Available cameras: {list(cameras.keys())}")
                        else:
                            for cam_path, cam_info in matching_cams.items():
                                image_data = cam_info["data"]
                                
                                # Handle time step selection
                                if time_step is not None:
                                    if len(image_data.shape) == 4:
                                        if time_step >= image_data.shape[0]:
                                            print(f"Error: Time step {time_step} out of range (0-{image_data.shape[0]-1})")
                                            continue
                                        image_data = image_data[time_step]
                                    else:
                                        print(f"Warning: Time step specified but image data is not time-series")
                                
                                display_image(
                                    image_data,
                                    title=f"{cam_path} (t={time_step if time_step is not None else 0})",
                                    save_path=save_image,
                                )
                    else:
                        # Display all cameras (first frame)
                        num_cams = len(cameras)
                        if num_cams == 1:
                            cam_path, cam_info = next(iter(cameras.items()))
                            image_data = cam_info["data"]
                            if time_step is not None and len(image_data.shape) == 4:
                                if time_step < image_data.shape[0]:
                                    image_data = image_data[time_step]
                            display_image(
                                image_data,
                                title=f"{cam_path} (t={time_step if time_step is not None else 0})",
                                save_path=save_image,
                            )
                        else:
                            # Display in a grid
                            cols = min(3, num_cams)
                            rows = (num_cams + cols - 1) // cols
                            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
                            if num_cams == 1:
                                axes = [axes]
                            else:
                                axes = axes.flatten()
                            
                            for idx, (cam_path, cam_info) in enumerate(cameras.items()):
                                image_data = cam_info["data"]
                                if time_step is not None and len(image_data.shape) == 4:
                                    if time_step < image_data.shape[0]:
                                        image_data = image_data[time_step]
                                
                                # Convert to numpy if needed
                                if isinstance(image_data, torch.Tensor):
                                    image_data = image_data.cpu().numpy()
                                
                                # Handle different image shapes
                                if len(image_data.shape) == 4:
                                    image_data = image_data[0]
                                elif len(image_data.shape) == 2:
                                    image_data = image_data[..., None]
                                
                                # Normalize
                                if image_data.dtype == np.uint8:
                                    image_data = image_data.astype(np.float32) / 255.0
                                elif image_data.max() > 1.0:
                                    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
                                
                                # Display
                                ax = axes[idx]
                                if image_data.shape[-1] == 1:
                                    ax.imshow(image_data[..., 0], cmap="gray")
                                elif image_data.shape[-1] == 3:
                                    ax.imshow(image_data)
                                elif image_data.shape[-1] == 4:
                                    ax.imshow(image_data[..., :3])
                                ax.set_title(cam_path, fontsize=8)
                                ax.axis("off")
                            
                            # Hide unused subplots
                            for idx in range(num_cams, len(axes)):
                                axes[idx].axis("off")
                            
                            plt.tight_layout()
                            if save_image:
                                plt.savefig(save_image, bbox_inches="tight", dpi=100)
                                print(f"Images saved to: {save_image}")
                            else:
                                plt.show()

    dataset_file_handler.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 dataset files.")
    parser.add_argument(
        "dataset_file",
        type=str,
        help="Path to the HDF5 dataset file to inspect.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Index of specific episode to inspect in detail.",
    )
    parser.add_argument(
        "--show_structure",
        action="store_true",
        help="Show detailed data structure for the specified episode.",
    )
    parser.add_argument(
        "--show_images",
        action="store_true",
        help="Display camera images from the episode (requires --episode).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Name of camera to display (partial match supported). If not specified, all cameras are shown.",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=None,
        help="Time step to display (default: first frame).",
    )
    parser.add_argument(
        "--save_image",
        type=str,
        default=None,
        help="Path to save the displayed image(s) instead of showing interactively.",
    )

    args = parser.parse_args()

    if args.show_images and args.episode is None:
        print("Warning: --show_images requires --episode. Showing images from episode 0.")
        args.episode = 0

    inspect_dataset(
        args.dataset_file,
        args.episode,
        args.show_structure,
        args.show_images,
        args.camera,
        args.time_step,
        args.save_image,
    )


if __name__ == "__main__":
    main()

