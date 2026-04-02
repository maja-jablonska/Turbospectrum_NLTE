#!/usr/bin/env python3

import sys
import glob
import zarr
import numpy as np

INPUT_DIR = sys.argv[1]
OUTPUT = sys.argv[2]

shards = sorted(glob.glob(f"{INPUT_DIR}/shard_*.zarr"))
if not shards:
    raise RuntimeError("No shards found")

arrays = [zarr.open(s, mode="r") for s in shards]
out = zarr.open(OUTPUT, mode="w")

def concat(name):
    print(f"[MERGE] {name}")

    srcs = [a[name] for a in arrays]
    total = sum(a.shape[0] for a in srcs)
    first = srcs[0]

    out_arr = out.create_array(
        name,
        shape=(total,) + first.shape[1:],
        dtype=first.dtype,
        chunks=first.chunks,
        compressors=first.compressors,
    )

    offset = 0
    step = 512

    for a in srcs:
        n = a.shape[0]
        for i in range(0, n, step):
            i_end = min(i + step, n)
            out_arr[offset+i:offset+i_end] = a[i:i_end]
        offset += n

for k in ["flux", "continuum", "params"]:
    concat(k)

for k in ["global_index", "model_id", "mu_selected", "mu_selected_index"]:
    concat(k)

# parameter_columns
if "parameter_columns" in arrays[0]:
    pc_out = out.create_group("parameter_columns")
    for key in arrays[0]["parameter_columns"].keys():
        srcs = [a["parameter_columns"][key] for a in arrays]
        total = sum(a.shape[0] for a in srcs)
        first = srcs[0]

        out_arr = pc_out.create_array(
            key,
            shape=(total,),
            dtype=first.dtype,
            chunks=first.chunks,
            compressors=first.compressors,
        )

        offset = 0
        for a in srcs:
            n = a.shape[0]
            out_arr[offset:offset+n] = a[:]
            offset += n

# metadata
if "wavelength" in arrays[0]:
    out.create_array("wavelength", data=arrays[0]["wavelength"][:])

if "param_names" in arrays[0]:
    out.create_array("param_names", data=arrays[0]["param_names"][:])

# scalars
for name in ["physics_hash", "schema_version"]:
    if name in arrays[0]:
        val = arrays[0][name][()]
        if isinstance(val, str):
            val = np.array(val, dtype=object)
        out.create_array(name, data=val)

# provenance
if "provenance" in arrays[0]:
    prov_src = arrays[0]["provenance"]
    prov_out = out.create_group("provenance")

    for k in prov_src.keys():
        val = prov_src[k][()]
        if isinstance(val, str):
            val = np.array(val, dtype=object)
        prov_out.create_array(k, data=val)

print("✅ Merge complete")