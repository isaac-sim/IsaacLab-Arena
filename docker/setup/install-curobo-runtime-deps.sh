#!/bin/bash
set -euo pipefail

# Read the installed versions of packages that must stay compatible with Isaac Sim.
# The Python block writes them as pip constraints, preventing the dependency install
# below from replacing them with different versions. Packages not yet installed are skipped.
/isaac-sim/python.sh - <<'PYTHON' > /tmp/curobo-constraints.txt
from importlib.metadata import PackageNotFoundError, version
for name in ('torch', 'torchvision', 'torchaudio', 'numpy', 'warp-lang',
             'boto3', 'botocore', 's3transfer', 'requests'):
    try:
        installed_version = version(name)
    except PackageNotFoundError:
        continue
    print(f'{name}=={installed_version}')
PYTHON

# Install the dependencies listed by the wheel builder (-r), respecting the protected versions (-c).
# The Dockerfile installs the cuRobo wheel itself afterward.
/isaac-sim/python.sh -m pip install -r /wheels/runtime-requirements.txt -c /tmp/curobo-constraints.txt
rm /tmp/curobo-constraints.txt
