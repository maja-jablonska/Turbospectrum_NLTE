#!/usr/bin/env python3

import sys
import glob
import zarr
import zarr.codecs as zc
import numpy as np
from zarr.core.dtype.npy.string import VariableLengthUTF8

INPUT_DIR = sys.argv[1]
OUTPUT = sys.argv[2]

shards = sorted(glob.glob(f"{INPUT_DIR}/shard_*.zarr"))
if not shards:
    raise RuntimeError("No shards found")

arrays = [zarr.open(s, mode="r") for s in shards]
out = zarr.open(OUTPUT, mode="w")


def write_string_array(group, name, values, chunk_len=128):
    vals = [str(value) for value in np.asarray(values).tolist()]
    arr = group.create_array(
        name,
        shape=(len(vals),),
        dtype=VariableLengthUTF8(),
        serializer=zc.VLenUTF8Codec(),
        chunks=(min(len(vals), chunk_len) if vals else 1,),
    )
    arr[:] = vals


def write_string_scalar(group, name, value):
    text = str(value)
    try:
        arr = group.create_array(
            name,
            shape=(),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
        )
        arr[...] = text
    except Exception:
        arr = group.create_array(
            name,
            shape=(1,),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            chunks=(1,),
        )
        arr[0] = text

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


# ---------- main ----------
for k in ["flux", "continuum", "params"]:
    concat(k)

for k in ["global_index", "model_id", "mu_selected", "mu_selected_index"]:
    if k in arrays[0]:
        concat(k)

# ---------- parameter_columns ----------
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

# ---------- metadata ----------
if "wavelength" in arrays[0]:
    out.create_array("wavelength", data=arrays[0]["wavelength"][:])

if "param_names" in arrays[0]:
    write_string_array(out, "param_names", arrays[0]["param_names"][:])

# ---------- scalars ----------
for name in ["physics_hash", "schema_version"]:
    if name in arrays[0]:
        val = arrays[0][name][()]

        if isinstance(val, (str, np.str_)):
            write_string_scalar(out, name, val)
        else:
            out.create_array(name, data=val)

# ---------- provenance ----------
if "provenance" in arrays[0]:
    prov_src = arrays[0]["provenance"]
    prov_out = out.create_group("provenance")

    for k in prov_src.keys():
        val = prov_src[k][()]

        if isinstance(val, (str, np.str_)):
            write_string_scalar(prov_out, k, val)
        else:
            prov_out.create_array(k, data=val)

print("✅ Merge complete")
