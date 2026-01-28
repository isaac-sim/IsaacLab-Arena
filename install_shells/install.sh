#!/usr/bin/env bash
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installation utilities for Isaac Lab Arena extensions

# Check if input directory is a python extension and install the module
install_isaaclab_arena_extension() {
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    pip_command=$(extract_pip_command)

    # if the directory contains setup.py then install the python module
    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        $pip_command --editable "$1"
    fi
}

# Install all Isaac Lab Arena extensions and dependencies
install_extensions() {
    local framework_name="$1"
    
    echo "[INFO] Installing extensions inside the Isaac Lab Arena repository..."
    python_exe=$(extract_python_exe)
    pip_command=$(extract_pip_command)
    pip_uninstall_command=$(extract_pip_uninstall_command)
    
    export -f extract_python_exe
    export -f extract_pip_command
    export -f extract_pip_uninstall_command
    
    # Install IsaacLab
    $ISAACLAB_PATH/isaaclab.sh -i $framework_name
    
    # Install Isaac-GR00T
    ${pip_command} -e "${ISAACLAB_ARENA_PATH}/submodules/Isaac-GR00T"
    
    # Install IsaacLab Arena
    ${pip_command} -e .
    
    # unset local variables
    unset extract_python_exe
    unset extract_pip_command
    unset extract_pip_uninstall_command
}
