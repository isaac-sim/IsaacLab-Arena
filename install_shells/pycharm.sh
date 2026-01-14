#!/usr/bin/env bash
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# PyCharm IDE configuration utilities for Isaac Lab Arena

# Update the PyCharm settings from template and Isaac Sim settings
update_pycharm_settings() {
    echo "[INFO] Setting up PyCharm settings..."
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    # path to setup_pycharm.py
    setup_pycharm_script="${ISAACLAB_ARENA_PATH}/.idea/tools/setup_pycharm.py"
    # check if the file exists before attempting to run it
    if [ -f "${setup_pycharm_script}" ]; then
        ${python_exe} "${setup_pycharm_script}"
    else
        echo "[WARNING] Unable to find the script 'setup_pycharm.py'. Aborting PyCharm settings setup."
    fi

    # Update run configurations with environment variables
    update_run_configs_script="${ISAACLAB_ARENA_PATH}/.idea/tools/update_run_configs.py"
    if [ -f "${update_run_configs_script}" ]; then
        echo "[INFO] Updating PyCharm run configurations with environment variables..."
        ${python_exe} "${update_run_configs_script}"
    else
        echo "[WARNING] Unable to find the script 'update_run_configs.py'. Run configurations may not have proper environment variables."
    fi
}
