#!/usr/bin/env python3
"""
Extract a chunk of samples from a spectra Zarr store.

Zarr v2 + v3 compatible.
HPC-safe (atomic write).

Example:
python extract_zarr_chunk.py \
    --input synthesized_spectra.zarr \
    --output chunk_100.zarr \
    --start 0 \
    --count 100
"""

import argparse
import os
import shutil
import zarr
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

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

BATCH = 32  # copy in chunks (memory-safe)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def copy_provenance(src, dst):
    if "provenance" not in src:
        return

    prov_src = src["provenance"]
    prov_dst = dst.create_group("provenance")

    for k in prov_src.array_keys():
        arr = prov_src[k]
        prov_dst.create_dataset(
            k,
            data=arr[()],
            overwrite=True,
        )


def copy_sample_array(arr, dst_group, name, start, end):
    n = end - start

    # create empty output array with SAME metadata
    out = dst_group.create_dataset(
        name,
        shape=(n,) + arr.shape[1:],
        like=arr,   # <- v3 safe magic
        overwrite=True,
    )

    # batch copy (avoids loading entire flux block)
    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        out[i:j] = arr[start + i:start + j]

    return out


def atomic_finalize(tmp_path, final_path):
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    os.rename(tmp_path, final_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    src = zarr.open_group(args.input, mode="r")

    start = args.start
    end = start + args.count

    n_total = src["flux"].shape[0]
    if end > n_total:
        raise ValueError(f"Requested [{start}:{end}] exceeds {n_total}")

    print(f"Copying samples [{start}:{end})")

    # --------------------------------------------------
    # atomic write
    # --------------------------------------------------
    tmp_output = args.output + ".tmp"

    if os.path.exists(tmp_output):
        shutil.rmtree(tmp_output)

    dst = zarr.open_group(tmp_output, mode="w")

    # --------------------------------------------------
    # copy sample arrays
    # --------------------------------------------------
    for name in SAMPLE_ARRAYS:
        arr = src[name]
        out = copy_sample_array(arr, dst, name, start, end)
        print(f"Copied {name} -> {out.shape}")

    # --------------------------------------------------
    # copy static arrays
    # --------------------------------------------------
    for name in STATIC_ARRAYS:
        arr = src[name]
        dst.create_dataset(
            name,
            data=arr[:],
            like=arr,
            overwrite=True,
        )
        print(f"Copied static {name}")

    # --------------------------------------------------
    # copy scalars
    # --------------------------------------------------
    for name in SCALAR_ARRAYS:
        arr = src[name]
        dst.create_dataset(
            name,
            data=arr[()],
            overwrite=True,
        )

    # --------------------------------------------------
    # provenance
    # --------------------------------------------------
    copy_provenance(src, dst)

    # copy root attrs
    dst.attrs.update(src.attrs)

    # --------------------------------------------------
    # validation (VERY IMPORTANT)
    # --------------------------------------------------
    gi = dst["global_index"][:]
    if not np.all(np.diff(gi) >= 0):
        print("⚠ WARNING: global_index not sorted")

    print("Validation done.")

    # --------------------------------------------------
    # atomic finalize
    # --------------------------------------------------
    atomic_finalize(tmp_output, args.output)

    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()