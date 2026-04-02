#!/usr/bin/env python3

import sys
import os
import zarr
import zarr.codecs as zc
import numpy as np
from zarr.core.dtype.npy.string import VariableLengthUTF8

INPUT = sys.argv[1]
OUTDIR = sys.argv[2]
START = int(sys.argv[3])
END = int(sys.argv[4])
JOB_ID = int(sys.argv[5])

ROW_CHUNK = 128
LAMBDA_CHUNK = 8192
PARAM_CHUNK = 512
ONE_D_CHUNK = 1024

COMPRESSORS = [zarr.codecs.Blosc(cname="zstd", clevel=3)]

os.makedirs(OUTDIR, exist_ok=True)
out_path = os.path.join(OUTDIR, f"shard_{JOB_ID:03d}.zarr")

print(f"[INFO] Opening source: {INPUT}")
src = zarr.open(INPUT, mode="r")

N = src["flux"].shape[0]
END = min(END, N)

print(f"[INFO] Writing shard {JOB_ID}: rows [{START}:{END}]")

dst = zarr.open(out_path, mode="w")


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

# ---------- 2D ----------
def copy_2d(name):
    arr = src[name]
    shape = (END - START, arr.shape[1])
    chunks = (ROW_CHUNK, min(LAMBDA_CHUNK, arr.shape[1]))

    print(f"[2D] {name} → {shape}, chunks={chunks}")

    out = dst.create_array(
        name,
        shape=shape,
        dtype=arr.dtype,
        chunks=chunks,
        compressors=COMPRESSORS,
    )

    for i in range(START, END, ROW_CHUNK):
        i_end = min(i + ROW_CHUNK, END)
        li0 = i - START
        li1 = i_end - START

        for j in range(0, arr.shape[1], chunks[1]):
            j_end = min(j + chunks[1], arr.shape[1])
            out[li0:li1, j:j_end] = arr[i:i_end, j:j_end]


for name in ["flux", "continuum"]:
    copy_2d(name)

# ---------- params ----------
params = src["params"]

out = dst.create_array(
    "params",
    shape=(END - START, params.shape[1]),
    dtype=params.dtype,
    chunks=(PARAM_CHUNK, params.shape[1]),
    compressors=COMPRESSORS,
)

for i in range(START, END, PARAM_CHUNK):
    i_end = min(i + PARAM_CHUNK, END)
    out[i - START:i_end - START] = params[i:i_end]

# ---------- 1D ----------
for name in ["global_index", "model_id", "mu_selected", "mu_selected_index"]:
    arr = src[name]

    out = dst.create_array(
        name,
        shape=(END - START,),
        dtype=arr.dtype,
        chunks=(ONE_D_CHUNK,),
        compressors=COMPRESSORS,
    )
    out[:] = arr[START:END]

# ---------- parameter_columns ----------
if "parameter_columns" in src:
    pc_src = src["parameter_columns"]
    pc_dst = dst.create_group("parameter_columns")

    for name in pc_src.keys():
        arr = pc_src[name]

        out = pc_dst.create_array(
            name,
            shape=(END - START,),
            dtype=arr.dtype,
            chunks=(ONE_D_CHUNK,),
            compressors=COMPRESSORS,
        )
        out[:] = arr[START:END]

# ---------- metadata ----------
if "wavelength" in src:
    dst.create_array("wavelength", data=src["wavelength"][:])

if "param_names" in src:
    write_string_array(dst, "param_names", src["param_names"][:])

# ---------- scalars ----------
for name in ["physics_hash", "schema_version"]:
    if name in src:
        val = src[name][()]

        if isinstance(val, (str, np.str_)):
            write_string_scalar(dst, name, val)
        else:
            dst.create_array(name, data=val)

# ---------- provenance ----------
if "provenance" in src:
    prov_src = src["provenance"]
    prov_dst = dst.create_group("provenance")

    for k in prov_src.keys():
        val = prov_src[k][()]

        if isinstance(val, (str, np.str_)):
            write_string_scalar(prov_dst, k, val)
        else:
            prov_dst.create_array(k, data=val)

print(f"[DONE] shard {JOB_ID}")
