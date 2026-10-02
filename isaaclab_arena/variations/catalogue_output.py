# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Emit variation catalogues independently of simulation console messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_variations_output(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject a catalogue output path without a discovery request before simulation starts.

    Args:
        args: Parsed runner arguments.
        parser: Parser reporting invalid option combinations.
    """
    if args.variations_output is not None and not args.list_variations:
        parser.error("--variations_output requires --list_variations")


def emit_variations_catalogue(catalogue: str | dict[str, Any], output_path: str | Path | None = None) -> None:
    """Print a text or JSON catalogue and optionally write the same content to a clean UTF-8 file.

    Args:
        catalogue: Formatted text or a JSON-serializable catalogue dictionary.
        output_path: Optional file to replace, creating missing parent directories.
    """
    content = json.dumps(catalogue, indent=2) if isinstance(catalogue, dict) else catalogue
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
    print(content, flush=True)
