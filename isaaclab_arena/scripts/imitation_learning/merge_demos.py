# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Merge multiple Arena teleop HDF5 demonstration files into a single training-ready dataset.

Teleoperators often collect 100+ demonstrations across several sessions rather than in one sitting,
producing one HDF5 per session via :mod:`isaaclab_arena.scripts.imitation_learning.record_demos`.
This script combines those files into a single dataset suitable for downstream consumers
(:mod:`isaaclab_arena.scripts.imitation_learning.replay_demos`,
:mod:`isaaclab_arena.scripts.imitation_learning.annotate_demos`,
``convert_hdf5_to_lerobot.py``).

Compared to upstream ``submodules/IsaacLab/scripts/tools/merge_hdf5_datasets.py``, this script:

- Preserves the root ``format_version`` attribute (otherwise the loader assumes legacy WXYZ
  quaternions and silently corrupts ``root_pose`` data).
- Recomputes ``data.attrs["total"]`` from the per-demo ``num_samples`` sum.
- Validates structural compatibility across inputs (``format_version``, dataset paths, per-key
  shapes/dtypes) and warns on ``env_args`` mismatches.
- Drops empty demos (no ``actions`` dataset or ``actions.shape[0] == 0``) automatically,
  since they break replay and post-training. Dropped demos are listed in the merge summary.
- Logs an operator-friendly per-file summary and aggregate report.
- Uses a recursive ``h5py.Group.copy`` so new recorder terms added by future Isaac Lab versions
  (new ``obs/*`` keys, new sensor groups, new metadata attrs) round-trip unchanged.

The script has zero simulation dependency and only requires ``h5py``. A single input file is
supported and equivalent to a copy with empty demos dropped and per-demo ``num_samples`` totals
recomputed.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/merge_demos.py \\
        -o $DATASET_DIR/combined.hdf5 \\
        $DATASET_DIR/session_a.hdf5 $DATASET_DIR/session_b.hdf5 $DATASET_DIR/session_c.hdf5
"""

from __future__ import annotations

import argparse
import contextlib
import h5py
import json
import os
import sys
from dataclasses import dataclass, field

_TABLE_SEP_CHAR = "-"


@dataclass
class _FileInfo:
    """Summary of one input HDF5 file, populated by :func:`_inspect_file`."""

    path: str
    format_version: int
    env_args: dict
    num_demos: int
    total_steps: int
    size_bytes: int
    schema_fingerprint: dict[str, tuple[tuple[int, ...], str]] = field(default_factory=dict)
    success_count: int = 0
    no_success_attr_count: int = 0
    failed_count: int = 0
    # Demos that won't replay or train: either no `actions` dataset or `actions.shape[0] == 0`.
    empty_demo_names: list[str] = field(default_factory=list)
    # Demos within this file whose dataset key/shape/dtype set differs from demo_0's. This
    # surfaces single-recording corruption that the cross-file schema check (which samples
    # one demo per input) cannot see.
    intra_file_schema_mismatches: list[str] = field(default_factory=list)


def _format_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def _format_int(n: int) -> str:
    """Format an integer with thousands separators."""
    return f"{n:,}"


def _build_schema_fingerprint(demo_group: h5py.Group) -> dict[str, tuple[tuple[int, ...], str]]:
    """Walk a demo group recursively and produce a ``path -> (shape[1:], dtype_str)`` map.

    The leading time dimension is dropped because episode lengths legitimately vary across demos.
    Everything else (action_dim, obs feature dim, image HxWxC, ...) must match for the merged
    dataset to be usable by downstream consumers.
    """
    fingerprint: dict[str, tuple[tuple[int, ...], str]] = {}

    def _visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            fingerprint[name] = (tuple(obj.shape[1:]), obj.dtype.str)

    demo_group.visititems(_visit)
    return fingerprint


def _sorted_demo_names(data_group: h5py.Group) -> list[str]:
    """Return ``demo_*`` group names sorted by numeric suffix when possible."""
    demo_names = [k for k in data_group.keys() if k.startswith("demo_")]

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len("demo_") :]
        if suffix.isdigit():
            return (0, f"{int(suffix):020d}")
        return (1, name)

    return sorted(demo_names, key=_key)


def _inspect_file(path: str) -> _FileInfo:
    """Open an input HDF5 file read-only and build a :class:`_FileInfo` summary."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file does not exist: {path}")
    size_bytes = os.path.getsize(path)

    # Prepend the path to h5py's bare OSError on non-HDF5/truncated/locked inputs.
    try:
        h5_file = h5py.File(path, "r")
    except OSError as e:
        raise ValueError(f"{path}: cannot open as HDF5 ({e})") from e

    with h5_file as f:
        if "data" not in f:
            raise ValueError(f"{path}: missing top-level 'data' group; not a record_demos HDF5 file")

        format_version = int(f.attrs["format_version"]) if "format_version" in f.attrs else 0

        data_group = f["data"]
        env_args: dict = {}
        raw_env_args = data_group.attrs.get("env_args")
        if raw_env_args is not None:
            try:
                env_args = json.loads(raw_env_args)
            except (json.JSONDecodeError, TypeError):
                env_args = {"_raw": str(raw_env_args)}

        demo_names = _sorted_demo_names(data_group)
        num_demos = len(demo_names)
        if num_demos == 0:
            raise ValueError(f"{path}: contains no 'demo_*' groups; nothing to merge")

        # Sample the file-level reference schema from the first demo. record_demos.py writes
        # a uniform recorder stack within a single file, so demo_0 is a reasonable reference;
        # the per-demo loop below catches the rare case where a single demo in the file
        # diverges (e.g. partial recording, manual edit).
        schema_fingerprint = _build_schema_fingerprint(data_group[demo_names[0]])

        total_steps = 0
        success_count = 0
        no_success_attr_count = 0
        failed_count = 0
        empty_demo_names: list[str] = []
        intra_file_schema_mismatches: list[str] = []
        for name in demo_names:
            demo = data_group[name]
            actions_len = int(demo["actions"].shape[0]) if "actions" in demo else 0

            if "num_samples" in demo.attrs:
                total_steps += int(demo.attrs["num_samples"])
            else:
                total_steps += actions_len

            # "Empty" = unusable for replay / training. Either there is no actions dataset at
            # all, or it has zero timesteps. record_demos.py shouldn't normally export these,
            # but partial recordings and interrupted sessions can produce them.
            if "actions" not in demo or actions_len == 0:
                empty_demo_names.append(name)

            # Intra-file schema mismatch: any demo whose key set / shape / dtype differs from
            # the file's reference (demo_0). Skip demo_0 itself.
            if name != demo_names[0]:
                demo_fp = _build_schema_fingerprint(demo)
                if demo_fp != schema_fingerprint:
                    intra_file_schema_mismatches.append(name)

            if "success" in demo.attrs:
                if bool(demo.attrs["success"]):
                    success_count += 1
                else:
                    failed_count += 1
            else:
                no_success_attr_count += 1

    return _FileInfo(
        path=path,
        format_version=format_version,
        env_args=env_args,
        num_demos=num_demos,
        total_steps=total_steps,
        size_bytes=size_bytes,
        schema_fingerprint=schema_fingerprint,
        success_count=success_count,
        no_success_attr_count=no_success_attr_count,
        failed_count=failed_count,
        empty_demo_names=empty_demo_names,
        intra_file_schema_mismatches=intra_file_schema_mismatches,
    )


def _diff_schema(
    ref: dict[str, tuple[tuple[int, ...], str]], other: dict[str, tuple[tuple[int, ...], str]]
) -> list[str]:
    """Return human-readable list of differences between two schema fingerprints."""
    out: list[str] = []
    ref_keys = set(ref.keys())
    other_keys = set(other.keys())
    for k in sorted(ref_keys - other_keys):
        out.append(f"only in first: {k} (shape={ref[k][0]}, dtype={ref[k][1]})")
    for k in sorted(other_keys - ref_keys):
        out.append(f"only in other: {k} (shape={other[k][0]}, dtype={other[k][1]})")
    for k in sorted(ref_keys & other_keys):
        if ref[k] != other[k]:
            out.append(
                f"shape/dtype differs at {k}: shape={ref[k][0]} dtype={ref[k][1]}"
                f" vs shape={other[k][0]} dtype={other[k][1]}"
            )
    return out


@dataclass
class _ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    schema_status: str = "OK"
    format_version_status: str = "OK"
    env_args_status: str = "OK"


def _validate_compatibility(infos: list[_FileInfo]) -> _ValidationReport:
    """Compare input files for compatibility and produce a :class:`_ValidationReport`.

    Empty demos (no ``actions`` dataset or ``actions.shape[0] == 0``) are always dropped from
    the merged output, so intra-file schema mismatches whose only divergence is the empty
    demo itself are not surfaced as errors here.
    """
    report = _ValidationReport()

    versions = {i.format_version for i in infos}
    if len(versions) > 1:
        report.format_version_status = "MISMATCH"
        report.errors.append(
            f"format_version mismatch across inputs: {sorted(versions)}. Legacy WXYZ files "
            "must be converted first via HDF5DatasetFileHandler.convert_dataset_to_xyzw()."
        )

    if len(infos) > 1:
        ref = infos[0]
        any_schema_diff = False
        for other in infos[1:]:
            diffs = _diff_schema(ref.schema_fingerprint, other.schema_fingerprint)
            if not diffs:
                continue
            any_schema_diff = True
            msg = f"Schema mismatch between {ref.path} and {other.path}:\n  " + "\n  ".join(diffs)
            report.errors.append(msg)
        if any_schema_diff:
            report.schema_status = "MISMATCH"

    env_names = {i.env_args.get("env_name", "") for i in infos}
    sim_args_set = {json.dumps(i.env_args.get("sim_args", {}), sort_keys=True) for i in infos}
    if len(env_names) > 1:
        report.env_args_status = "WARN"
        report.warnings.append(
            f"env_args.env_name differs across inputs: {sorted(env_names)}. "
            "record_demos.py typically writes an empty env_name, so this is often expected."
        )
    if len(sim_args_set) > 1:
        report.env_args_status = "WARN"
        report.warnings.append(
            "env_args.sim_args differ across inputs (e.g. dt, decimation, render_interval, "
            "num_envs). The first file's values will be written to the merged output."
        )

    for i in infos:
        if i.failed_count > 0:
            report.warnings.append(
                f"{i.path}: {i.failed_count} demo(s) with success=False. record_demos.py "
                "normally exports only successful demos; included as-is in the merged file."
            )
        if i.no_success_attr_count > 0:
            report.info.append(
                f"{i.path}: {i.no_success_attr_count} demo(s) without @success attribute (legacy format)."
            )
        if i.empty_demo_names:
            shown = ", ".join(i.empty_demo_names[:3])
            suffix = ", ..." if len(i.empty_demo_names) > 3 else ""
            report.warnings.append(
                f"{i.path}: {len(i.empty_demo_names)} empty demo(s) ({shown}{suffix}) — missing"
                " 'actions' or actions.shape[0] == 0. These break replay and post-training and"
                " will be dropped from the merged output."
            )
        if i.intra_file_schema_mismatches:
            empty_set = set(i.empty_demo_names)
            # Demos that diverge from the file's reference schema solely because they are
            # empty will be dropped before being written, so they cannot break downstream
            # consumers. Real intra-file inconsistency (e.g. a missing camera obs on an
            # otherwise valid demo) still surfaces as a hard error.
            actionable = [name for name in i.intra_file_schema_mismatches if name not in empty_set]
            if actionable:
                shown = ", ".join(actionable[:3])
                suffix = ", ..." if len(actionable) > 3 else ""
                report.schema_status = "MISMATCH"
                report.errors.append(
                    f"{i.path}: {len(actionable)} demo(s) ({shown}{suffix}) have a key/shape/dtype"
                    " set that differs from demo_0 in the same file (e.g. a missing camera"
                    " observation). Inspect the offending demos and either drop them upstream"
                    " or re-record the session."
                )

    return report


def _print_summary(
    infos: list[_FileInfo],
    output_path: str,
    *,
    output_total_steps: int | None = None,
    output_num_demos: int | None = None,
    output_size_bytes: int | None = None,
    report: _ValidationReport | None = None,
    dry_run: bool = False,
    dropped: list[str] | None = None,
) -> None:
    """Print an operator-friendly per-file table, aggregate row, and validation summary."""
    n = len(infos)
    if n == 0:
        return

    max_name_len = max(len(os.path.basename(i.path)) for i in infos)
    max_name_len = max(max_name_len, len(os.path.basename(output_path)) + len(" (output)"), 25)

    print()
    for idx, i in enumerate(infos, start=1):
        env_name = i.env_args.get("env_name", "")
        empty_str = f"  empty={len(i.empty_demo_names)}" if i.empty_demo_names else ""
        print(
            f"[{idx}/{n}] {os.path.basename(i.path):<{max_name_len}}  "
            f"demos={i.num_demos:>4d}  "
            f"steps={_format_int(i.total_steps):>10s}  "
            f"size={_format_bytes(i.size_bytes):>10s}  "
            f'env="{env_name}"  '
            f"v={i.format_version}  "
            f"keys={len(i.schema_fingerprint)}"
            f"{empty_str}"
        )

    print(_TABLE_SEP_CHAR * (max_name_len + 65))

    total_demos = output_num_demos if output_num_demos is not None else sum(i.num_demos for i in infos)
    total_steps = output_total_steps if output_total_steps is not None else sum(i.total_steps for i in infos)
    if output_size_bytes is not None:
        size_str = _format_bytes(output_size_bytes)
    else:
        size_str = "~" + _format_bytes(sum(i.size_bytes for i in infos))

    label = f"{os.path.basename(output_path)} ({'dry-run' if dry_run else 'output'})"
    print(
        f"        {label:<{max_name_len}}  "
        f"demos={total_demos:>4d}  "
        f"steps={_format_int(total_steps):>10s}  "
        f"size={size_str:>10s}"
    )

    if report is not None:
        print(
            "Validation: "
            f"format_version {report.format_version_status}, "
            f"schema {report.schema_status}, "
            f"env_args {report.env_args_status}"
        )
        for msg in report.info:
            print(f"INFO: {msg}")
        for msg in report.warnings:
            print(f"WARNING: {msg}")
        for msg in report.errors:
            print(f"ERROR: {msg}")

    if dropped:
        shown = ", ".join(dropped[:5])
        suffix = ", ..." if len(dropped) > 5 else ""
        print(f"Dropped {len(dropped)} empty demo(s): {shown}{suffix}")

    if total_demos > 0:
        numbering_note = "input order preserved"
        if dropped:
            numbering_note += f"; {len(dropped)} empty demo(s) skipped"
        print(f"Demo numbering: demo_0..demo_{total_demos - 1} ({numbering_note})")
    print()


def _merge(infos: list[_FileInfo], output_path: str) -> tuple[int, int, int, list[str]]:
    """Write a merged HDF5 dataset from validated inputs.

    Empty demos (no ``actions`` dataset or ``actions.shape[0] == 0``) are always skipped:
    they cannot be replayed or used for training, so including them would only break
    downstream consumers.

    Args:
        infos: Per-input file summaries produced by :func:`_inspect_file`.
        output_path: Destination HDF5 path.

    Returns:
        ``(output_size_bytes, total_steps_written, total_demos_written, dropped_log)`` where
        ``dropped_log`` is a list of ``"{source_path}::{demo_name}"`` strings naming every
        empty demo that was skipped. Empty when no input demo was empty.
    """
    format_version = infos[0].format_version
    merged_env_args = dict(infos[0].env_args)
    for other in infos[1:]:
        for k, v in other.env_args.items():
            if k not in merged_env_args:
                merged_env_args[k] = v

    total_demos_written = 0
    total_steps_written = 0
    dropped: list[str] = []

    with h5py.File(output_path, "w") as out:
        out.attrs["format_version"] = format_version
        data_out = out.create_group("data")
        data_out.attrs["env_args"] = json.dumps(merged_env_args)

        for info in infos:
            skip_set: set[str] = set(info.empty_demo_names)
            with h5py.File(info.path, "r") as src:
                src_data = src["data"]
                for src_demo_name in _sorted_demo_names(src_data):
                    if src_demo_name in skip_set:
                        dropped.append(f"{info.path}::{src_demo_name}")
                        continue
                    dst_demo_name = f"demo_{total_demos_written}"
                    try:
                        src.copy(src_data[src_demo_name], data_out, name=dst_demo_name)
                    except (OSError, RuntimeError, ValueError) as e:
                        # h5py error messages from deep recursive copies are notoriously opaque;
                        # surface the offending source file and demo so the operator can
                        # repair / drop the bad input without guessing.
                        raise RuntimeError(
                            f"Failed to copy {info.path}::{src_demo_name} into the merged"
                            f" dataset as {dst_demo_name}: {e}"
                        ) from e
                    dst_demo = data_out[dst_demo_name]
                    if "num_samples" in dst_demo.attrs:
                        total_steps_written += int(dst_demo.attrs["num_samples"])
                    elif "actions" in dst_demo:
                        total_steps_written += int(dst_demo["actions"].shape[0])
                    total_demos_written += 1

        data_out.attrs["total"] = total_steps_written

    output_size = os.path.getsize(output_path)
    return output_size, total_steps_written, total_demos_written, dropped


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Arena teleop HDF5 demo files into a single training-ready dataset."
            " Validates schema compatibility across inputs, renumbers demos sequentially, and"
            " preserves all recorder data (forward-compatible to new recorder terms)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="One or more input HDF5 demo files to merge. Files are merged in the given order.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to write the merged HDF5 dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print the merge report without writing the output file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_abs = os.path.abspath(args.output_file)
    tmp_abs = output_abs + ".tmp"
    for inp in args.input_files:
        if not os.path.exists(inp):
            continue
        inp_abs = os.path.abspath(inp)
        if inp_abs == output_abs:
            print(
                f"ERROR: --output_file path equals an input path: {inp}",
                file=sys.stderr,
            )
            return 2
        if inp_abs == tmp_abs:
            print(
                f"ERROR: input path {inp} collides with the internal temp path "
                f"{args.output_file}.tmp. Rename the input or choose a different --output_file.",
                file=sys.stderr,
            )
            return 2
    if not args.dry_run and os.path.exists(args.output_file) and not args.overwrite:
        print(
            f"ERROR: output file {args.output_file} already exists. Pass --overwrite to replace.",
            file=sys.stderr,
        )
        return 2

    infos: list[_FileInfo] = []
    for path in args.input_files:
        try:
            infos.append(_inspect_file(path))
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    report = _validate_compatibility(infos)

    # Pre-flight: if every input demo is empty, the merged output would have a useless
    # zero-demo /data group that downstream loaders can't consume. Refuse rather than
    # write a structurally empty file.
    total_input_demos = sum(i.num_demos for i in infos)
    total_empty_demos = sum(len(i.empty_demo_names) for i in infos)
    if total_input_demos > 0 and total_empty_demos == total_input_demos:
        _print_summary(infos, args.output_file, report=report, dry_run=True)
        print(
            "ERROR: every input demo is empty (no 'actions' or actions.shape[0] == 0). Nothing to merge.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        _print_summary(infos, args.output_file, report=report, dry_run=True)
        return 1 if report.errors else 0

    if report.errors:
        _print_summary(infos, args.output_file, report=report, dry_run=True)
        print(
            "Merge aborted due to validation errors. Re-run with --dry_run to inspect.",
            file=sys.stderr,
        )
        return 1

    output_dir = os.path.dirname(output_abs) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Atomic write: stream into <output>.tmp and rename on success so a mid-merge
    # failure does not leave a partial output or, under --overwrite, destroy the
    # prior file.
    tmp_path = args.output_file + ".tmp"
    success = False
    try:
        output_size, total_steps, total_demos, dropped = _merge(infos, tmp_path)
        os.replace(tmp_path, args.output_file)
        success = True
    except Exception as e:
        print(f"ERROR: Merge failed: {e}", file=sys.stderr)
        return 1
    finally:
        if not success and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    _print_summary(
        infos,
        args.output_file,
        output_total_steps=total_steps,
        output_num_demos=total_demos,
        output_size_bytes=output_size,
        report=report,
        dry_run=False,
        dropped=dropped,
    )
    print(f"Merged dataset written to: {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
