#!/bin/bash
set -euo pipefail

# Install dependencies from project metadata before copying Arena source.
# The later editable install uses --no-deps, keeping this step cached on source edits.
# The Python helper turns project and developer dependencies into a requirements file for pip.
/isaac-sim/python.sh /tmp/export_requirements.py /tmp/arena-pyproject.toml > /tmp/arena-requirements.txt
/isaac-sim/python.sh -m pip install -r /tmp/arena-requirements.txt
rm /tmp/arena-requirements.txt

# simready-search declares an AWS stack conflicting with Isaac Sim's bundle.
# Restore compatible versions without letting pip change their other dependencies.
/isaac-sim/python.sh -m pip install --force-reinstall --no-deps \
    boto3==1.40.61 botocore==1.40.61 s3transfer==0.14.0 requests==2.32.3

# HuggingFace CLI for downloading datasets and models.
# Use pipx so the hf binary gets an isolated venv with all its deps (e.g. requests),
# without touching system Python packages.
apt-get update
apt-get install -y pipx
PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install "huggingface-hub[cli]"
