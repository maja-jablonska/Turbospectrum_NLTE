#!/usr/bin/env python3
"""
Extract chunk from Zarr (v2/v3 compatible).
Designed for HPC pipelines.
"""

import argparse
import os
import shutil
import zarr
import numpy as np


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

BATCH = 32


# --------------------------------------------------
def safe_array_kwargs(arr, for_data=False):
    """
    Extract safe kwargs across zarr versions.

    for_data=True -> omit dtype (required by newer Zarr)
    """

    kwargs = {}

    if not for_data:
        kwargs["dtype"] = arr.dtype

    if arr.chunks is not None:
        kwargs["chunks"] = arr.chunks

    # v3
    if hasattr(arr, "compressors"):
        try:
            kwargs["compressors"] = arr.compressors
        except Exception:
            pass

    # v2
    elif hasattr(arr, "compressor"):
        try:
            kwargs["compressor"] = arr.compressor
        except Exception:
            pass

    return kwargs


def copy_provenance(src, dst):
    if "provenance" not in src:
        return

    psrc = src["provenance"]
    pdst = dst.create_group("provenance")

    for k in psrc.array_keys():
        arr = psrc[k]
        pdst.create_array(k, data=arr[()])


def copy_sample_array(arr, dst_group, name, start, end):
    n = end - start

    kwargs = safe_array_kwargs(arr)

    out = dst_group.create_array(
        name,
        shape=(n,) + arr.shape[1:],
        **kwargs,
    )

    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        out[i:j] = arr[start + i:start + j]

    return out


def atomic_finalize(tmp_path, final_path):
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    os.rename(tmp_path, final_path)


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
        raise ValueError(f"{end=} exceeds dataset size {n_total}")

    print(f"Copying samples [{start}:{end})")

    tmp_output = args.output + ".tmp"
    if os.path.exists(tmp_output):
        shutil.rmtree(tmp_output)

    dst = zarr.open_group(tmp_output, mode="w")

    # ---------------- SAMPLE ARRAYS ----------------
    for name in SAMPLE_ARRAYS:
        arr = src[name]
        out = copy_sample_array(arr, dst, name, start, end)
        print(f"Copied {name} -> {out.shape}")

    # ---------------- STATIC ARRAYS ----------------
    for name in STATIC_ARRAYS:
        arr = src[name]
        kwargs = safe_array_kwargs(arr, for_data=True)

        dst.create_array(
            name,
            data=arr[:],
            **kwargs,
        )
        print(f"Copied static {name}")

    # ---------------- SCALARS ----------------
    for name in SCALAR_ARRAYS:
        arr = src[name]
        dst.create_array(name, data=arr[()])

    # ---------------- PROVENANCE ----------------
    copy_provenance(src, dst)

    dst.attrs.update(src.attrs)

    # ---------------- VALIDATION ----------------
    gi = dst["global_index"][:]
    if not np.all(np.diff(gi) >= 0):
        print("⚠ global_index not sorted")

    atomic_finalize(tmp_output, args.output)

    print("Done.")


if __name__ == "__main__":
    main()