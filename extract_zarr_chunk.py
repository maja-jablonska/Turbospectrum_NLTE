#!/usr/bin/env python3
"""
Extract a chunk of samples from a spectra Zarr store.

Example:
    python extract_zarr_chunk.py \
        --input synthesized_spectra.zarr \
        --output chunk_100.zarr \
        --start 0 \
        --count 100
"""

import argparse
import zarr
import numpy as np


SAMPLE_ARRAYS = [
    "flux",
    "global_index",
    "model_id",
    "mu_selected",
    "mu_selected_index",
    "params",
]

STATIC_ARRAYS = [
    "wavelength",
    "param_names",
]

SCALAR_ARRAYS = [
    "schema_version",
    "physics_hash",
]


def copy_provenance(src, dst):
    if "provenance" not in src:
        return

    prov_src = src["provenance"]
    prov_dst = dst.create_group("provenance")

    for k in prov_src.array_keys():
        prov_dst.create_dataset(
            k,
            data=prov_src[k][()],
            shape=prov_src[k].shape,
            dtype=prov_src[k].dtype,
            overwrite=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    src = zarr.open_group(args.input, mode="r")
    dst = zarr.open_group(args.output, mode="w")

    start = args.start
    end = start + args.count

    n_total = src["flux"].shape[0]
    if end > n_total:
        raise ValueError(f"Requested end={end} exceeds dataset size={n_total}")

    print(f"Copying samples [{start}:{end})")

    # ------------------------------
    # Copy sample-dependent arrays
    # ------------------------------
    for name in SAMPLE_ARRAYS:
        arr = src[name]

        data = arr[start:end]

        dst.create_dataset(
            name,
            data=data,
            chunks=arr.chunks,
            compressor=arr.compressor,
            dtype=arr.dtype,
            overwrite=True,
        )

        print(f"Copied {name}: {data.shape}")

    # ------------------------------
    # Copy static arrays
    # ------------------------------
    for name in STATIC_ARRAYS:
        arr = src[name]
        dst.create_dataset(
            name,
            data=arr[:],
            chunks=arr.chunks,
            compressor=arr.compressor,
            dtype=arr.dtype,
            overwrite=True,
        )
        print(f"Copied static {name}")

    # ------------------------------
    # Copy scalar arrays
    # ------------------------------
    for name in SCALAR_ARRAYS:
        arr = src[name]
        dst.create_dataset(
            name,
            data=arr[()],
            shape=arr.shape,
            dtype=arr.dtype,
            overwrite=True,
        )

    # ------------------------------
    # Copy provenance group
    # ------------------------------
    copy_provenance(src, dst)

    # copy root attributes
    dst.attrs.update(src.attrs)

    print("Done.")


if __name__ == "__main__":
    main()