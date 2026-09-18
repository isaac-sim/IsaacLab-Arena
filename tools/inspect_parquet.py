# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Print the format, schema, metadata, and contents of a Parquet file.

Typical usage:

    # Preview the first 10 rows (default).
    .venv/bin/python tools/inspect_parquet.py path/to/episode.parquet

    # Preview the first 20 rows.
    .venv/bin/python tools/inspect_parquet.py --num-rows 20 path/to/episode.parquet

    # Print every row.
    .venv/bin/python tools/inspect_parquet.py --all path/to/episode.parquet
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the Parquet file.")
    rows_group = parser.add_mutually_exclusive_group()
    rows_group.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=10,
        help="Number of rows to print (default: 10).",
    )
    rows_group.add_argument("--all", action="store_true", help="Print every row.")
    return parser.parse_args()


def print_key_values(items: list[tuple[str, Any]]) -> None:
    """Print key-value pairs with aligned separators."""
    key_width = max(len(key) for key, _ in items)
    for key, value in items:
        print(f"{key:<{key_width}} : {value}")


def format_value(value: Any) -> str:
    """Format arrays as their size and scalar values as text."""
    if hasattr(value, "size") and hasattr(value, "shape") and not isinstance(value, (str, bytes)):
        return f"array(size={value.size})"
    if isinstance(value, (list, tuple)):
        return f"array(size={len(value)})"
    if pd.isna(value):
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def inspect_parquet(path: Path, num_rows: int | None) -> None:
    """Print metadata and up to ``num_rows`` rows from a Parquet file."""
    assert path.is_file(), f"Parquet file does not exist: {path}"
    assert num_rows is None or num_rows >= 0, "--num-rows must be non-negative"

    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    table = parquet_file.read()
    dataframe = table.to_pandas()

    compressions = sorted(
        {
            metadata.row_group(row_group_index).column(column_index).compression
            for row_group_index in range(metadata.num_row_groups)
            for column_index in range(metadata.num_columns)
        }
    )

    print("=== File format ===")
    print_key_values(
        [
            ("Format", "Apache Parquet"),
            ("Path", path.resolve()),
            ("Size", f"{path.stat().st_size:,} bytes"),
            ("Created by", metadata.created_by),
            ("Rows", f"{metadata.num_rows:,}"),
            ("Columns", metadata.num_columns),
            ("Row groups", metadata.num_row_groups),
            ("Compression", ", ".join(compressions)),
        ]
    )

    print("\n=== Columns ===")
    column_rows = [
        (field.name, str(field.type), str(dataframe.dtypes.iloc[index]))
        for index, field in enumerate(table.schema)
    ]
    name_width = max(len("Name"), *(len(row[0]) for row in column_rows))
    arrow_width = max(len("Arrow type"), *(len(row[1]) for row in column_rows))
    pandas_width = max(len("Pandas dtype"), *(len(row[2]) for row in column_rows))
    print(f"{'Name':<{name_width}}  {'Arrow type':<{arrow_width}}  {'Pandas dtype':<{pandas_width}}")
    print(f"{'-' * name_width}  {'-' * arrow_width}  {'-' * pandas_width}")
    for name, arrow_type, pandas_type in column_rows:
        print(f"{name:<{name_width}}  {arrow_type:<{arrow_width}}  {pandas_type:<{pandas_width}}")

    displayed = dataframe if num_rows is None else dataframe.head(num_rows)
    print(f"\n=== Contents ({len(displayed):,}/{len(dataframe):,} rows) ===")
    if displayed.empty:
        print("<no rows>")
        return

    field_width = max(len(column) for column in dataframe.columns)
    for row_index, row in displayed.iterrows():
        print(f"\n--- Row {row_index} ---")
        for field, value in row.items():
            print(f"{field:<{field_width}} : {format_value(value)}")


def main() -> None:
    """Inspect the requested Parquet file."""
    args = parse_args()
    inspect_parquet(args.path, None if args.all else args.num_rows)


if __name__ == "__main__":
    main()
