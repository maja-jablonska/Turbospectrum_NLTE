#!/usr/bin/env python3
"""Take a subsample of rows from a spectra (or grid) .zarr store.

Copies a subset of rows into a new, self-contained zarr store. Arrays whose
first dimension matches the source row count (flux, continuum, teff, feh,
status, mu_selected, ...) are sliced to the chosen rows; arrays that are not
row-aligned (e.g. ``wavelength``) are copied verbatim. Group-level attrs and
nested subgroups (e.g. ``provenance``) are preserved.

This is a diagnostic / dev convenience tool. ``source_row_index`` maps each
output row back to the original store, root attrs get an updated ``n_models``
and a ``lineage`` entry, and the parameters of the draw are in the
``subsample`` group attrs (see zarr_rowstore.write_row_subset).

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

from zarr_compat import open_root_group  # noqa: E402
from zarr_rowstore import row_count, write_row_subset  # noqa: E402


def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def subsample(input_path: str, output_path: str, n: int, *, seed: int, head: bool) -> None:
    input_path = _norm(input_path)
    output_path = _norm(output_path)
    if os.path.exists(output_path):
        raise FileExistsError(f"Output already exists, refusing to overwrite: {output_path}")

    src = open_root_group(input_path, mode="r")
    n_rows = row_count(src)
    if n >= n_rows:
        print(f"Requested {n} rows but store has only {n_rows}; copying all rows.")
        rows = np.arange(n_rows, dtype=np.int64)
    elif head:
        rows = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n_rows, size=n, replace=False)).astype(np.int64)

    written = write_row_subset(
        src, output_path, rows,
        stage="subsample", source_path=input_path,
        stage_attrs={
            "source_store": input_path,
            "source_row_count": int(n_rows),
            "sampled_row_count": int(len(rows)),
            "mode": "head" if head else "random",
            "seed": int(seed),
        },
    )
    print(f"Wrote {len(rows)} rows -> {output_path}")
    print(f"  sliced arrays ({len(written['sliced'])}): {', '.join(written['sliced'])}")
    print(f"  copied arrays ({len(written['copied'])}): {', '.join(written['copied'])}")


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
