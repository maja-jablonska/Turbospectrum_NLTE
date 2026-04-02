#!/usr/bin/env python3

import sys
import os
import zarr

INPUT = sys.argv[1]
OUTDIR = sys.argv[2]
START = int(sys.argv[3])
END = int(sys.argv[4])
JOB_ID = int(sys.argv[5])

# -------- CHUNK CONFIG --------
ROW_CHUNK = 128
LAMBDA_CHUNK = 8192
PARAM_CHUNK = 512
ONE_D_CHUNK = 1024

COMPRESSORS = [zarr.codecs.Blosc(cname="zstd", clevel=3)]

# -----------------------------

os.makedirs(OUTDIR, exist_ok=True)
out_path = os.path.join(OUTDIR, f"shard_{JOB_ID:03d}.zarr")

print(f"[INFO] Opening source: {INPUT}")
src = zarr.open(INPUT, mode="r")

N = src["flux"].shape[0]
END = min(END, N)

print(f"[INFO] Writing shard {JOB_ID}: rows [{START}:{END}] → {out_path}")

dst = zarr.open(out_path, mode="w")

# ---------- 2D arrays ----------
def copy_2d(name):
    arr = src[name]
    shape = (END - START, arr.shape[1])
    chunks = (ROW_CHUNK, min(LAMBDA_CHUNK, arr.shape[1]))

    print(f"[2D] {name} → shape={shape}, chunks={chunks}")

    out = dst.create_array(
        name,
        shape=shape,
        dtype=arr.dtype,
        chunks=chunks,
        compressors=COMPRESSORS,
    )

    for i in range(START, END, ROW_CHUNK):
        i_end = min(i + ROW_CHUNK, END)
        local_i0 = i - START
        local_i1 = i_end - START

        for j in range(0, arr.shape[1], chunks[1]):
            j_end = min(j + chunks[1], arr.shape[1])
            out[local_i0:local_i1, j:j_end] = arr[i:i_end, j:j_end]


for name in ["flux", "continuum"]:
    copy_2d(name)

# ---------- params ----------
params = src["params"]
print("[PARAMS]")

out = dst.create_array(
    "params",
    shape=(END - START, params.shape[1]),
    dtype=params.dtype,
    chunks=(PARAM_CHUNK, params.shape[1]),
    compressors=COMPRESSORS,
)

for i in range(START, END, PARAM_CHUNK):
    i_end = min(i + PARAM_CHUNK, END)
    out[i - START : i_end - START] = params[i:i_end]

# ---------- 1D arrays ----------
for name in [
    "global_index",
    "model_id",
    "mu_selected",
    "mu_selected_index",
]:
    arr = src[name]
    print(f"[1D] {name}")

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
    print("[GROUP] parameter_columns")

    pc_src = src["parameter_columns"]
    pc_dst = dst.create_group("parameter_columns")

    for name in pc_src.keys():
        arr = pc_src[name]
        print(f"  → {name}")

        out = pc_dst.create_array(
            name,
            shape=(END - START,),
            dtype=arr.dtype,
            chunks=(ONE_D_CHUNK,),
            compressors=COMPRESSORS,
        )
        out[:] = arr[START:END]

# ---------- wavelength ----------
if "wavelength" in src:
    print("[META] wavelength")
    wl = src["wavelength"]
    dst.create_array(
        "wavelength",
        data=wl[:],   # ✅ no shape here
        dtype=wl.dtype,
    )

# ---------- param_names ----------
if "param_names" in src:
    print("[META] param_names")
    dst.create_array(
        "param_names",
        data=src["param_names"][:],  # ✅ no shape
        dtype=src["param_names"].dtype,
    )

# ---------- scalars ----------
for name in ["physics_hash", "schema_version"]:
    if name in src:
        print(f"[SCALAR] {name}")
        dst.create_array(name, data=src[name][()])

# ---------- provenance ----------
if "provenance" in src:
    print("[GROUP] provenance")
    prov_src = src["provenance"]
    prov_dst = dst.create_group("provenance")

    for k in prov_src.keys():
        v = prov_src[k]
        print(f"  → {k}")
        prov_dst.create_array(k, data=v[()])

print(f"[DONE] shard {JOB_ID}")