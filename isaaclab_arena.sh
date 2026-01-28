#!/usr/bin/env bash
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Path to this repository
export ISAACLAB_ARENA_PATH="$(pwd)"

# Path to Isaac Lab (assumed to be in submodules)
export ISAACLAB_PATH="$ISAACLAB_ARENA_PATH/submodules/IsaacLab"

# Source utility scripts
INSTALL_DIR="${ISAACLAB_ARENA_PATH}/install_shells"
source "${INSTALL_DIR}/python.sh"
source "${INSTALL_DIR}/uv.sh"
source "${INSTALL_DIR}/pycharm.sh"
source "${INSTALL_DIR}/install.sh"

# Print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-c] [-u] [-y] -- Utility to manage Isaac Lab Arena."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install [LIB]  Install the extensions inside Isaac Lab Arena and learning frameworks as extra dependencies. Default is 'all'."
    echo -e "\t-c, --conda [NAME]   Create the conda environment for Isaac Lab Arena. Default name is 'env_isaaclab_arena'."
    echo -e "\t-u, --uv [NAME]      Create the uv environment for Isaac Lab Arena. Default name is '.venv'."
    echo -e "\t-y, --pycharm        Generate the PyCharm settings files from templates."
    echo -e "\n" >&2
}

# Check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 0
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--install)
            # Determine framework name
            if [ -z "$2" ]; then
                framework_name="all"
            elif [ "$2" = "none" ]; then
                framework_name="none"
                shift
            else
                framework_name=$2
                shift
            fi
            # Install extensions
            install_extensions "$framework_name"
            shift
            ;;
        -c|--conda)
            # Use default name if not provided
            if [ -z "$2" ]; then
                conda_env_name="env_isaaclab_arena"
            else
                conda_env_name=$2
                shift
            fi
            # Setup the conda environment for Isaac Lab
            $ISAACLAB_PATH/isaaclab.sh --conda ${conda_env_name}
            shift
            ;;
        -u|--uv)
            # Use default name if not provided
            if [ -z "$2" ]; then
                echo "[INFO] Using default uv environment name: .venv"
                uv_env_name=".venv"
            else
                echo "[INFO] Using uv environment name: $2"
                uv_env_name=$2
                shift
            fi
            # Setup the uv environment for Isaac Lab
            setup_uv_env ${uv_env_name}
            shift
            ;;
        -y|--pycharm)
            # Update the PyCharm settings
            update_pycharm_settings
            shift
            break
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "[Error] Unknown argument: $1" >&2
            print_help
            exit 1
            ;;
    esac
done
