#!/usr/bin/env python3
"""Resolve row-count and shard layout for Zarr-backed batch jobs."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import zarr


DEFAULT_ROW_KEYS = ("flux", "continuum", "params", "global_index", "teff", "logg", "feh")


def _infer_row_count(zarr_path: str, row_keys: Sequence[str]) -> int:
    root = zarr.open_group(os.path.abspath(zarr_path), mode="r")
    for key in row_keys:
        if key in root and len(root[key].shape) >= 1:
            return int(root[key].shape[0])
    raise RuntimeError(
        f"Could not infer row count from {zarr_path!r} using keys {list(row_keys)!r}"
    )


def _positive_int(raw: str | None, name: str) -> int | None:
    if raw in (None, ""):
        return None
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", default=None, help="Zarr store used to infer row_count")
    parser.add_argument("--row-count", type=int, default=None, help="Explicit row_count override")
    parser.add_argument(
        "--row-keys",
        default=",".join(DEFAULT_ROW_KEYS),
        help="Comma-separated keys to probe for row_count inference",
    )
    parser.add_argument("--rows-per-shard", default=None, help="Target rows per shard")
    parser.add_argument("--shard-count", default=None, help="Target number of shards")
    parser.add_argument(
        "--default-rows-per-shard",
        default=None,
        help="Fallback rows_per_shard when neither rows_per_shard nor shard_count is provided",
    )
    args = parser.parse_args()

    row_keys = tuple(key.strip() for key in str(args.row_keys).split(",") if key.strip())
    if not row_keys:
        raise ValueError("row_keys must contain at least one key")

    if args.row_count is not None:
        row_count = int(args.row_count)
    elif args.zarr_path:
        row_count = _infer_row_count(args.zarr_path, row_keys)
    else:
        raise ValueError("Provide either --row-count or --zarr-path")

    if row_count < 1:
        raise ValueError(f"row_count must be positive, got {row_count!r}")

    rows_per_shard = _positive_int(args.rows_per_shard, "rows_per_shard")
    shard_count = _positive_int(args.shard_count, "shard_count")
    default_rows_per_shard = _positive_int(args.default_rows_per_shard, "default_rows_per_shard")

    if shard_count is not None and rows_per_shard is None:
        rows_per_shard = (row_count + shard_count - 1) // shard_count
    elif rows_per_shard is not None and shard_count is None:
        shard_count = (row_count + rows_per_shard - 1) // rows_per_shard
    elif rows_per_shard is not None and shard_count is not None:
        coverage = shard_count * rows_per_shard
        if coverage < row_count:
            # shard_count is the harder constraint (e.g. PBS array size);
            # auto-adjust rows_per_shard upward so all rows are covered.
            old_rps = rows_per_shard
            rows_per_shard = (row_count + shard_count - 1) // shard_count
            print(
                f"WARN: shard_count={shard_count} and rows_per_shard={old_rps} only cover "
                f"{coverage} rows, but row_count={row_count}. "
                f"Auto-adjusting rows_per_shard to {rows_per_shard}.",
                file=sys.stderr,
            )
    elif default_rows_per_shard is not None:
        rows_per_shard = default_rows_per_shard
        shard_count = (row_count + rows_per_shard - 1) // rows_per_shard
    else:
        raise ValueError(
            "Provide rows_per_shard or shard_count, or set default_rows_per_shard"
        )

    if rows_per_shard is None or shard_count is None:
        raise AssertionError("rows_per_shard and shard_count must both be resolved")

    array_range = f"0-{shard_count - 1}"

    print(f"ROW_COUNT={row_count}")
    print(f"ROWS_PER_SHARD={rows_per_shard}")
    print(f"SHARD_COUNT={shard_count}")
    print(f"ARRAY_RANGE={array_range}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(str(exc))
