#!/bin/bash
# Install the lightweight GR00T policy client and its dependencies.

set -euo pipefail
# Preserve group write access inherited from the shared Isaac Lab venv.
umask 0002

# Install GR00T's video and transport dependencies without its full training stack.
uv pip install --no-deps \
    av==18.0.0 msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1
# Make the GR00T checkout importable without installing its full training/inference dependencies.
# Its metadata requires Python 3.10; Arena's client runs on the Lab venv's Python 3.12.
"${VIRTUAL_ENV}/bin/python" -m pip install \
    --no-deps --ignore-requires-python --editable "${WORKDIR}/submodules/Isaac-GR00T/"
# Package caches can preserve read-only modes despite the umask, so repair them after installation.
find "${VIRTUAL_ENV}" \( -type f -o -type d \) \
    \( ! -group isaac-sim -o ! -perm -g+w \) \
    -exec chgrp isaac-sim {} + -exec chmod g+rwX {} +
