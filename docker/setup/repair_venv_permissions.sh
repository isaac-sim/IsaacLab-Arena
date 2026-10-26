#!/bin/bash
# uv can preserve restrictive ownership and modes from its package cache, leaving
# the shared virtual environment unwritable by runtime users in the isaac-sim group.
# Repair only mismatches in each install layer to avoid copying the full venv later.

set -euo pipefail

find "${VIRTUAL_ENV}" \( -type f -o -type d \) \
    \( ! -group isaac-sim -o ! -perm -g+w \) \
    -exec chgrp isaac-sim {} + -exec chmod g+rwX {} +
