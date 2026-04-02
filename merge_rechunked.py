=#!/usr/bin/env python3

import sys
import glob
import zarr

INPUT_DIR = sys.argv[1]
OUTPUT = sys.argv[2]

shards = sorted(glob.glob(f"{INPUT_DIR}/shard_*.zarr"))

if not shards:
    raise RuntimeError("No shards found")

print(f"[INFO] Found {len(shards)} shards")

arrays = [zarr.open(s, mode="r") for s in shards]
out = zarr.open(OUTPUT, mode="w")

# ---------- helper ----------
def concat_array(name):
    print(f"[MERGE] {name}")

    src_arrays = [a[name] for a in arrays]
    total_rows = sum(a.shape[0] for a in src_arrays)

    first = src_arrays[0]

    out_arr = out.create_array(
        name,
        shape=(total_rows,) + first.shape[1:],
        dtype=first.dtype,
        chunks=first.chunks,
        compressors=first.compressors,
    )

    offset = 0
    step = 512  # streaming block size

    for a in src_arrays:
        n = a.shape[0]

        for i in range(0, n, step):
            i_end = min(i + step, n)
            out_arr[offset + i : offset + i_end] = a[i:i_end]

        offset += n

    print(f"[DONE] {name}")


# ---------- 2D ----------
for name in ["flux", "continuum"]:
    concat_array(name)

# ---------- params ----------
concat_array("params")

# ---------- 1D ----------
for name in [
    "global_index",
    "model_id",
    "mu_selected",
    "mu_selected_index",
]:
    concat_array(name)

# ---------- parameter_columns ----------
if "parameter_columns" in arrays[0]:
    print("[MERGE] parameter_columns")

    pc_out = out.create_group("parameter_columns")
    keys = arrays[0]["parameter_columns"].keys()

    for key in keys:
        print(f"  → {key}")

        src_arrays = [a["parameter_columns"][key] for a in arrays]
        total_rows = sum(a.shape[0] for a in src_arrays)

        first = src_arrays[0]

        out_arr = pc_out.create_array(
            key,
            shape=(total_rows,),
            dtype=first.dtype,
            chunks=first.chunks,
            compressors=first.compressors,
        )

        offset = 0
        for a in src_arrays:
            n = a.shape[0]
            out_arr[offset:offset + n] = a[:]  # safe for 1D
            offset += n


# ---------- wavelength (copy once) ----------
if "wavelength" in arrays[0]:
    print("[COPY] wavelength")
    wl = arrays[0]["wavelength"]
    out.create_array(
        "wavelength",
        data=wl[:]
    )

# ---------- param_names ----------
if "param_names" in arrays[0]:
    print("[COPY] param_names")
    pn = arrays[0]["param_names"]
    out.create_array(
        "param_names",
        data=pn[:]
    )

# ---------- scalars ----------
for name in ["physics_hash", "schema_version"]:
    if name in arrays[0]:
        print(f"[COPY] {name}")
        out.create_array(name, data=arrays[0][name][()])

# ---------- provenance ----------
if "provenance" in arrays[0]:
    print("[COPY] provenance")

    prov_src = arrays[0]["provenance"]
    prov_out = out.create_group("provenance")

    for k in prov_src.keys():
        v = prov_src[k]
        print(f"  → {k}")
        prov_out.create_array(k, data=v[()])

print("✅ Merge complete")