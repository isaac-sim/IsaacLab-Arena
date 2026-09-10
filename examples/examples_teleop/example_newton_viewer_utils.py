"""Shared Newton viewer helpers for standalone examples."""

from __future__ import annotations

import ast
from pathlib import Path

import warp as wp

import newton.viewer


def arena_root() -> Path:
    return Path(__file__).resolve().parents[2]


def init_viewer(args):
    for entry in args.warp_config:
        if "=" not in entry:
            raise ValueError(f"Invalid --warp-config format {entry!r}; expected KEY=VALUE")
        key, value_str = entry.split("=", 1)
        if not hasattr(wp.config, key):
            raise ValueError(f"Invalid --warp-config key {key!r}")
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        setattr(wp.config, key, value)

    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)

    if args.viewer == "gl":
        return newton.viewer.ViewerGL(headless=args.headless)
    if args.viewer == "usd":
        return newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=args.num_frames)
    if args.viewer == "viser":
        return newton.viewer.ViewerViser()
    raise ValueError(f"Invalid viewer: {args.viewer}")
