# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Download the two reference NVIDIA Sans fonts and verify their recorded hashes."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "assets/branding_sources.json").read_text())
    folder = root / "assets/fonts"
    folder.mkdir(exist_ok=True)
    for name, entry in manifest["fonts"].items():
        path = folder / name
        if path.exists():
            data = path.read_bytes()
        else:
            with urlopen(entry["source"], timeout=60) as response:
                data = response.read()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], f"Font source changed: {name}"
        path.write_bytes(data)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
