#!/bin/bash
# Repair shared-group access after installing packages into the virtual environment.

set -euo pipefail

find "${VIRTUAL_ENV}" \( -type f -o -type d \) \
    \( ! -group isaac-sim -o ! -perm -g+w \) \
    -exec chgrp isaac-sim {} + -exec chmod g+rwX {} +
