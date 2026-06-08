#!/usr/bin/env python3
"""Take a subsample of rows from a spectra (or grid) .zarr store.

Copies a subset of rows into a new, self-contained zarr store. Arrays whose
first dimension matches the source row count (flux, continuum, teff, feh,
status, mu_selected, ...) are sliced to the chosen rows; arrays that are not
row-aligned (e.g. ``wavelength``) are copied verbatim. Group-level attrs and
nested subgroups (e.g. ``provenance``) are preserved.

This is a diagnostic / dev convenience tool: the output is *not* a fresh
synthesis product and does not recompute provenance hashes. The sampled
source row indices are recorded in the output array ``source_row_index`` and
in the ``subsample`` group attrs so the subset can be traced back.

Usage:
    python3 scripts/subsample_spectra.py --input spectra.zarr --output sub.zarr
    python3 scripts/subsample_spectra.py -i spectra.zarr -o sub.zarr -n 1000 --seed 0
    python3 scripts/subsample_spectra.py -i spectra.zarr -o sub.zarr --head   # first N rows
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zarr_compat import (  # noqa: E402
    zarr_store,
    open_root_group,
    create_root_group,
    create_array,
    compression_kwargs,
)


def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _row_count(group) -> int:
    """Infer the number of rows from common row-indexed arrays."""
    for name in ("flux", "teff", "global_index", "status"):
        if name in group:
            return int(group[name].shape[0])
    # Fall back to the most common leading dimension across arrays.
    lengths: dict[int, int] = {}
    for name in group.array_keys():
        shp = group[name].shape
        if shp:
            lengths[shp[0]] = lengths.get(shp[0], 0) + 1
    if not lengths:
        raise ValueError("Could not infer row count: store has no arrays.")
    return max(lengths.items(), key=lambda kv: kv[1])[0]


def _copy_group_attrs(src, dst) -> None:
    for key, value in dict(src.attrs).items():
        dst.attrs[key] = value


def _copy_subgroups(src, dst) -> None:
    """Recursively copy non-row-aligned subgroups (e.g. provenance) verbatim."""
    for name in src.group_keys():
        sub_src = src[name]
        sub_dst = dst.create_group(name)
        _copy_group_attrs(sub_src, sub_dst)
        for arr_name in sub_src.array_keys():
            arr = sub_src[arr_name]
            data = arr[...]
            new = create_array(sub_dst, arr_name, data=data, chunks=arr.chunks)
            _copy_group_attrs(arr, new)
        _copy_subgroups(sub_src, sub_dst)


def subsample(input_path: str, output_path: str, n: int, *, seed: int, head: bool) -> None:
    input_path = _norm(input_path)
    output_path = _norm(output_path)
    if os.path.exists(output_path):
        raise FileExistsError(f"Output already exists, refusing to overwrite: {output_path}")

    src = open_root_group(input_path, mode="r")
    n_rows = _row_count(src)
    if n >= n_rows:
        print(f"Requested {n} rows but store has only {n_rows}; copying all rows.")
        rows = np.arange(n_rows, dtype=np.int64)
    elif head:
        rows = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n_rows, size=n, replace=False)).astype(np.int64)

    comp = compression_kwargs()
    dst = create_root_group(zarr_store(output_path), overwrite=True)
    _copy_group_attrs(src, dst)

    copied, sliced = [], []
    for name in src.array_keys():
        arr = src[name]
        if arr.shape and arr.shape[0] == n_rows:
            data = arr[...][rows]
            sliced.append(name)
        else:
            data = arr[...]
            copied.append(name)
        # Re-chunk the row dimension to the new size; keep trailing chunks.
        chunks = None
        if arr.chunks is not None and data.ndim >= 1:
            chunks = (min(data.shape[0], arr.chunks[0]),) + tuple(arr.chunks[1:])
        new = create_array(dst, name, data=data, chunks=chunks, **comp)
        _copy_group_attrs(arr, new)

    # Record provenance of the subsample itself.
    create_array(dst, "source_row_index", data=rows, chunks=(len(rows),))
    _copy_subgroups(src, dst)
    meta = dst.create_group("subsample")
    meta.attrs["source_store"] = input_path
    meta.attrs["source_row_count"] = int(n_rows)
    meta.attrs["sampled_row_count"] = int(len(rows))
    meta.attrs["mode"] = "head" if head else "random"
    meta.attrs["seed"] = int(seed)

    print(f"Wrote {len(rows)} rows -> {output_path}")
    print(f"  sliced arrays ({len(sliced)}): {', '.join(sorted(sliced))}")
    print(f"  copied arrays ({len(copied)}): {', '.join(sorted(copied))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="Source spectra/grid .zarr store")
    ap.add_argument("-o", "--output", required=True, help="Destination .zarr store (must not exist)")
    ap.add_argument("-n", "--rows", type=int, default=1000, help="Number of rows to sample (default 1000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for random sampling (default 0)")
    ap.add_argument("--head", action="store_true", help="Take the first N rows instead of a random sample")
    args = ap.parse_args(argv)

    subsample(args.input, args.output, args.rows, seed=args.seed, head=args.head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
