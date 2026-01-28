#!/usr/bin/env bash
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# UV environment setup utilities for Isaac Lab Arena

# Setup uv environment for Isaac Lab Arena
setup_uv_env() {
    # get environment name from input
    local env_name="$1"
    local python_path="${2:-python3.11}"

    # check uv is installed
    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv could not be found. Please install uv and try again."
        echo "[ERROR] uv can be installed here:"
        echo "[ERROR] https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    # check if _isaac_sim symlink exists and isaacsim-rl is not installed via pip
    if [ ! -L "${ISAACLAB_PATH}/_isaac_sim" ] && ! python -m pip list | grep -q 'isaacsim-rl'; then
        echo -e "[WARNING] _isaac_sim symlink not found at ${ISAACLAB_PATH}/_isaac_sim"
        echo -e "\tThis warning can be ignored if you plan to install Isaac Sim via pip."
        echo -e "\tIf you are using a binary installation of Isaac Sim, please ensure the symlink is created before setting up the conda environment."
    fi

    # check if the environment exists
    local env_path="${ISAACLAB_ARENA_PATH}/${env_name}"
    if [ ! -d "${env_path}" ]; then
        echo -e "[INFO] Creating uv environment named '${env_name}'..."
        uv venv --clear --python "${python_path}" "${env_path}"
        # Install pip so isaaclab.sh can use python -m pip
        echo -e "[INFO] Installing pip in uv environment..."
        uv pip install --python "${env_path}/bin/python" pip
    else
        echo "[INFO] uv environment '${env_name}' already exists."
    fi

    # define root path for activation hooks
    local isaaclab_root="${ISAACLAB_ARENA_PATH}"

    # cache current paths for later
    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH

    # ensure activate file exists
    touch "${env_path}/bin/activate"

     # add variables to environment during activation
    cat >> "${env_path}/bin/activate" <<EOF
export ISAACLAB_PATH="${ISAACLAB_PATH}"
alias isaaclab="${ISAACLAB_PATH}/isaaclab.sh"
alias isaaclab_arena="${ISAACLAB_ARENA_PATH}/isaaclab_arena.sh"
export RESOURCE_NAME="IsaacSim"

if [ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]; then
    . "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
fi
EOF

    # add information to the user about alias
    echo -e "[INFO] Added 'isaaclab' alias to uv environment for 'isaaclab.sh' script."
    echo -e "[INFO] Created uv environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                source ${env_name}/bin/activate."
    echo -e "\t\t2. To install Isaac Lab Arena extensions, run:      isaaclab_arena -i"
    echo -e "\t\t3. To perform formatting, run:                      isaaclab -f"
    echo -e "\t\t4. To deactivate the environment, run:              deactivate"
    echo -e "\n"
}
