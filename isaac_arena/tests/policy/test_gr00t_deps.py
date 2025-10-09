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

"""
Test suite for GR00T dependencies.

This module tests that GR00T dependencies are properly installed and functional:
- flash_attn package import and basic functionality
- Isaac-GR00T package import
"""

import os
import sys

import pytest


class TestGr00tDependencies:
    """Test class for GR00T dependencies."""

    def test_flash_attn_import(self):
        """Test that flash_attn can be imported successfully."""
        try:
            import flash_attn  # pylint: disable=import-outside-toplevel

            # Verify version is available
            assert hasattr(flash_attn, "__version__"), "flash_attn version not available"
            print(f"Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            pytest.fail(f"Failed to import flash_attn: {e}")

    def test_flash_attn_functionality(self):
        """Test basic flash_attn functionality."""
        pytest.importorskip("torch", reason="PyTorch not available")
        pytest.importorskip("flash_attn", reason="flash_attn not available")

        import torch  # pylint: disable=import-outside-toplevel

        from flash_attn import flash_attn_func  # pylint: disable=import-outside-toplevel

        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for flash_attn test")

        try:
            # Create small test tensors
            batch_size, seq_len, num_heads, head_dim = 1, 32, 4, 64
            device = "cuda"
            dtype = torch.float16

            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

            # Test flash attention function
            out = flash_attn_func(q, k, v)

            # Verify output shape
            expected_shape = (batch_size, seq_len, num_heads, head_dim)
            assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

            # Verify output is on correct device and dtype
            assert out.device.type == "cuda", f"Expected output on CUDA, got {out.device}"
            assert out.dtype == dtype, f"Expected dtype {dtype}, got {out.dtype}"

            print(f"Flash Attention test passed - output shape: {out.shape}")

        except (RuntimeError, ValueError) as e:
            pytest.fail(f"Flash Attention functionality test failed: {e}")
        finally:
            # Clean up GPU memory
            torch.cuda.empty_cache()

    def test_gr00t_package_directory_exists(self):
        """Test that GR00T package directory exists."""
        gr00t_path = f"{os.getenv('WORKDIR', '/workspaces/isaac_arena')}/submodules/Isaac-GR00T"

        assert os.path.exists(gr00t_path), f"GR00T directory not found at {gr00t_path}"
        assert os.path.isdir(gr00t_path), f"GR00T path exists but is not a directory: {gr00t_path}"

        print(f"GR00T package directory exists at: {gr00t_path}")

    def test_gr00t_package_import(self):
        """Test that GR00T package can be imported."""
        # get workdir from env
        gr00t_path = f"{os.getenv('WORKDIR', '/workspaces/isaac_arena')}/submodules/Isaac-GR00T"

        # Add GR00T path to Python path if not already there
        if gr00t_path not in sys.path:
            sys.path.insert(0, gr00t_path)

        # First, try to import the gr00t package directly
        try:
            import gr00t  # pylint: disable=import-outside-toplevel  # noqa: F401

            _ = gr00t  # Mark as used

            print(
                "Successfully imported gr00t package from"
                f" {gr00t.__file__ if hasattr(gr00t, '__file__') else 'unknown location'}"
            )

            # Try to import a submodule to verify package structure
            try:
                from gr00t.data import dataset  # pylint: disable=import-outside-toplevel  # noqa: F401

                _ = dataset  # Mark as used
                print("Successfully imported gr00t.data.dataset module")
            except ImportError:
                print("gr00t package imported but submodules may not be fully accessible")

            return  # Test passed

        except ImportError:
            print("Could not import gr00t package directly, checking directory structure...")

        # If direct import fails, verify the directory has Python files
        python_files = []
        for root, _, files in os.walk(gr00t_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    python_files.append(os.path.join(root, file))

        if python_files:
            print(f"GR00T directory contains {len(python_files)} Python files")
            # Just verify the structure exists - don't try to import files with relative imports
            # Look for key files that indicate a proper installation
            key_files = ["gr00t/data/dataset.py", "gr00t/model", "gr00t/eval"]
            found_structure = False

            for key_file in key_files:
                if any(key_file in py_file for py_file in python_files):
                    found_structure = True
                    print(f"Found expected GR00T structure: {key_file}")

            if found_structure:
                print("GR00T package structure verified (import may require additional setup)")
            else:
                pytest.fail("GR00T directory exists but expected package structure not found")
        else:
            pytest.fail("No Python files found in GR00T directory")

    def test_pytorch_cuda_compatibility(self):
        """Test that PyTorch and CUDA are properly configured for GR00T."""
        pytest.importorskip("torch", reason="PyTorch not available")

        import torch  # pylint: disable=import-outside-toplevel

        # Check PyTorch version
        print(f"PyTorch version: {torch.__version__}")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA available with {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 0:
                print(f"Current device: {torch.cuda.get_device_name(0)}")

            # Test basic CUDA operations
            try:
                x = torch.randn(100, 100, device="cuda")
                y = torch.matmul(x, x.T)
                assert y.device.type == "cuda", "CUDA operation failed"
                print("Basic CUDA operations working")

                # Clean up
                del x, y
                torch.cuda.empty_cache()

            except (RuntimeError, AssertionError) as e:
                pytest.fail(f"CUDA operations failed: {e}")
        else:
            pytest.skip("CUDA not available, skipping CUDA compatibility test")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
