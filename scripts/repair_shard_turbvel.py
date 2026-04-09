#!/usr/bin/env python3
"""Backfill missing ``turbvel`` (and ``t_value``) columns in shard Zarr stores.

Older pipeline runs silently dropped string-dtype columns (turbvel, t_value,
output_mode, …) when writing shard Zarr outputs because ``_write_array`` only
handled numeric dtypes and the ``except Exception: pass`` swallowed the error.

This script recovers turbvel for those shards using two strategies (in order):

1. **Grid Zarr lookup** (preferred) – read ``turbvel`` from the grid Zarr using
   the shard's ``global_index`` array.
2. **base_name parsing** (fallback) – extract the ``_t{XX}_`` segment from the
   ``base_name`` column stored in the shard.

Usage
-----
    # Single shard, grid zarr available:
    python repair_shard_turbvel.py \\
        --shard-zarr outputs/shards/spectra_shard_0.zarr \\
        --grid-zarr  outputs/grids/regular_parameter_grid.zarr

    # Multiple shards via glob:
    python repair_shard_turbvel.py \\
        --shard-zarr 'outputs/shards/spectra_shard_*.zarr' \\
        --grid-zarr  outputs/grids/regular_parameter_grid.zarr

    # Without grid zarr (parse from base_name):
    python repair_shard_turbvel.py \\
        --shard-zarr outputs/shards/spectra_shard_0.zarr

    # Dry-run (report what would be written):
    python repair_shard_turbvel.py \\
        --shard-zarr outputs/shards/spectra_shard_0.zarr \\
        --grid-zarr  outputs/grids/regular_parameter_grid.zarr \\
        --dry-run
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from typing import List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Regex to extract turbvel from a Turbospectrum output stem.
# Matches  _t{digits}_  or  _t{digits}. at end of stem component.
# Example: p7000_g+5.0_m0.0_t05_st_z-4.00_... → "05"
_TURBVEL_RE = re.compile(r"_t(\d{2,})_")


def _open_zarr_group(path: str, mode: str = "r"):
    """Open a Zarr group, compatible with both v2 and v3."""
    import zarr  # type: ignore

    try:
        return zarr.open_group(path, mode=mode)
    except Exception:
        pass
    # zarr v3 with LocalStore
    from zarr import storage as zstorage  # type: ignore
    if hasattr(zstorage, "LocalStore"):
        store = zstorage.LocalStore(path)
        return zarr.open_group(store=store, mode=mode)
    raise RuntimeError(f"Cannot open Zarr store at {path}")


def _write_string_column(root, name: str, values: List[str], chunks: int = 2048):
    """Write a 1D string column into an open zarr group (v2/v3 compatible)."""
    vals = ["" if v is None else str(v) for v in values]
    if hasattr(root, "create_array"):
        import zarr.codecs as zc  # type: ignore
        from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore
        arr = root.create_array(
            name,
            shape=(len(vals),),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            chunks=min(int(chunks), len(vals)) if len(vals) else 1,
        )
        arr[:] = vals
        return

    # zarr v2
    try:
        from numcodecs import VLenUTF8  # type: ignore
        root.array(
            name, vals, dtype=object, object_codec=VLenUTF8(),
            chunks=min(int(chunks), len(vals)) if len(vals) else 1,
        )
    except Exception:
        arr = np.asarray(vals, dtype="U256")
        try:
            root.create_dataset(name, data=arr)
        except TypeError:
            root.create_dataset(name, shape=arr.shape, dtype=arr.dtype, data=arr)


def _turbvel_from_grid(grid_path: str, global_indices: np.ndarray) -> Optional[List[str]]:
    """Look up turbvel values from the grid zarr using global indices."""
    try:
        grid = _open_zarr_group(grid_path, mode="r")
    except Exception as exc:
        logger.warning("Cannot open grid zarr %s: %s", grid_path, exc)
        return None

    for col_name in ("turbvel", "t_value"):
        if col_name in grid:
            all_values = np.asarray(grid[col_name][:])
            max_idx = int(global_indices.max()) if len(global_indices) else 0
            if max_idx >= len(all_values):
                logger.warning(
                    "global_index max %d exceeds grid %s length %d",
                    max_idx, col_name, len(all_values),
                )
                return None
            selected = [str(all_values[i]) for i in global_indices]
            logger.info("Recovered turbvel from grid column %r", col_name)
            return selected

    logger.warning("Grid zarr has no turbvel or t_value column")
    return None


def _turbvel_from_base_names(base_names: List[str]) -> Optional[List[str]]:
    """Parse turbvel from base_name strings (e.g. p7000_g+5.0_m0.0_t05_st_...)."""
    result = []
    failures = 0
    for bn in base_names:
        m = _TURBVEL_RE.search(str(bn))
        if m:
            result.append(m.group(1))
        else:
            result.append("")
            failures += 1

    if failures == len(base_names):
        logger.warning("Could not extract turbvel from any base_name")
        return None

    if failures > 0:
        logger.warning(
            "Could not extract turbvel from %d/%d base_names",
            failures, len(base_names),
        )
    return result


def repair_shard(shard_path: str, grid_path: Optional[str], dry_run: bool = False) -> bool:
    """Backfill turbvel into a single shard zarr. Returns True if repaired."""
    logger.info("Checking shard: %s", shard_path)

    shard = _open_zarr_group(shard_path, mode="r")
    available = set(shard.keys()) if hasattr(shard, "keys") else set(shard.array_keys())

    if "turbvel" in available:
        logger.info("  turbvel already present — skipping")
        return False

    if "global_index" not in available:
        logger.error("  No global_index in shard — cannot repair")
        return False

    global_indices = np.asarray(shard["global_index"][:])
    n_rows = len(global_indices)
    logger.info("  %d rows, global_index range [%d, %d]", n_rows, global_indices.min(), global_indices.max())

    # Strategy 1: grid zarr lookup
    turbvel_values = None
    if grid_path:
        turbvel_values = _turbvel_from_grid(grid_path, global_indices)

    # Strategy 2: parse from base_name
    if turbvel_values is None and "base_name" in available:
        base_names = [str(v) for v in np.asarray(shard["base_name"][:])]
        turbvel_values = _turbvel_from_base_names(base_names)

    if turbvel_values is None:
        logger.error(
            "  Cannot recover turbvel: no grid zarr and no base_name column. "
            "If .spec files are still on disk, use --spec-dir to scan filenames."
        )
        return False

    unique_values = sorted(set(turbvel_values) - {""})
    logger.info("  Recovered turbvel values: %s (%d unique)", unique_values, len(unique_values))

    if dry_run:
        logger.info("  [DRY RUN] Would write turbvel to shard")
        return True

    # Write back — only turbvel.
    # t_value is a legacy alias that _resolve_synthesis_request checks *before*
    # turbvel when choosing model_turb_str (the _t{XX}_ segment of the
    # atmosphere filename).  In the regular grid both are identical, so writing
    # turbvel alone is sufficient: synthesis_turb_raw is read from turbvel
    # (line 1214 of run_turbospectrum.py) and model_turb_str falls through to
    # turbvel when t_value is absent (line 1211).
    shard_rw = _open_zarr_group(shard_path, mode="r+")
    _write_string_column(shard_rw, "turbvel", turbvel_values, chunks=max(1, min(2048, n_rows)))
    logger.info("  Wrote turbvel (%d values)", n_rows)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing turbvel column in shard Zarr stores."
    )
    parser.add_argument(
        "--shard-zarr", required=True,
        help="Path to shard zarr (supports shell glob patterns for multiple shards)",
    )
    parser.add_argument(
        "--grid-zarr", default=None,
        help="Path to grid zarr (preferred source for turbvel recovery)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be written without modifying files",
    )
    args = parser.parse_args()

    # Expand glob patterns
    shard_paths = sorted(glob.glob(args.shard_zarr))
    if not shard_paths:
        logger.error("No shard zarr files matched: %s", args.shard_zarr)
        sys.exit(1)

    logger.info("Found %d shard(s) to check", len(shard_paths))
    repaired = 0
    for path in shard_paths:
        try:
            if repair_shard(path, args.grid_zarr, dry_run=args.dry_run):
                repaired += 1
        except Exception:
            logger.exception("Failed to repair %s", path)

    logger.info("Done: %d/%d shards repaired", repaired, len(shard_paths))


if __name__ == "__main__":
    main()
