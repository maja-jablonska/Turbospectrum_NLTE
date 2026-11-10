#!/usr/bin/env python3
"""Splice a blue re-synthesis (with molecular lines below 4200 A) into a released row store.

The released harps-interp-t02 spectra have no molecular lines below 4200 A (the GES molecular lists start there).
The blue campaign re-synthesised every row over 3780-4350 A with the Kurucz-derived CN/CH lists added. This writes a
new store whose flux and continuum below --splice (default 4250 A: keeps the red wings of the new lines, and stays
100 A inside the blue run's red end, where lines beyond its range are missing) come from the blue run and above it
from the release. Rows are matched by model_id; release rows without a blue counterpart (failed blue synthesis) are
dropped and listed. Everything else (parameters, ids, wavelength, provenance) is copied; a ``blue_splice`` provenance
entry and a new physics_hash record the change.

    python splice_blue_molecules.py RELEASE.zarr OUT.zarr BLUE_RUN_DIR/j0*/outputs/zarr/synthesized_spectra.zarr \
        --splice 4250 --note "molecules-blue-3760-4200A-kurucz"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merge_spectra_runs import _AlignedRowWriter  # noqa: E402
from zarr_compat import (compression_kwargs, create_array, create_root_group, write_fixed_string_scalar,  # noqa: E402
                         write_string_scalar, zarr_store)
from zarr_rowstore import atomic_output, copy_array, copy_attrs, copy_subgroups  # noqa: E402

ROW_ARRAYS = ("params", "model_id", "run_index", "source_physics_hash")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release"); ap.add_argument("out"); ap.add_argument("blue", nargs="+")
    ap.add_argument("--splice", type=float, default=4250.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--io-chunks", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    R = zarr.open_group(a.release, mode="r")
    wl = np.asarray(R["wavelength"][:], dtype=np.float64)
    n_rows, n_wl = R["flux"].shape
    chunk = int(R["flux"].chunks[0])
    rmid = np.asarray(R["model_id"][:])

    # blue rows by model_id
    blue, where = [], {}
    for s, path in enumerate(a.blue):
        B = zarr.open_group(path, mode="r")
        bw = np.asarray(B["wavelength"][:], dtype=np.float64)
        if not np.array_equal(bw, wl[: bw.size]):
            raise SystemExit(f"{path}: wavelength grid is not a prefix of the release grid")
        if bw[-1] < a.splice + 50:
            raise SystemExit(f"{path}: blue run ends at {bw[-1]} A, too close to the splice at {a.splice} A")
        blue.append(B)
        for r, m in enumerate(np.asarray(B["model_id"][:])):
            where[int(m)] = (s, r)
    ns = int(np.searchsorted(wl, a.splice))  # pixels [0, ns) come from the blue run
    keep = np.array([int(m) in where for m in rmid])
    dropped = rmid[~keep]
    print(f"release rows {n_rows}; blue rows {len(where)}; spliced {int(keep.sum())}; dropped {dropped.size}; "
          f"splice at {wl[ns]:.2f} A ({ns} blue pixels)", flush=True)

    kw = compression_kwargs({"cname": "zstd", "clevel": 5, "shuffle": True})
    n_out = int(keep.sum())
    with atomic_output(a.out) as tmp:
        dst = create_root_group(zarr_store(tmp), overwrite=True)
        copy_attrs(R, dst)
        flux = create_array(dst, "flux", shape=(n_out, n_wl), dtype=np.float32, chunks=(chunk, n_wl), **kw)
        cont = create_array(dst, "continuum", shape=(n_out, n_wl), dtype=np.float32, chunks=(chunk, n_wl), fill_value=np.nan, **kw)
        wf, wc = _AlignedRowWriter(flux, 0, a.io_chunks * chunk), _AlignedRowWriter(cont, 0, a.io_chunks * chunk)
        step = a.io_chunks * chunk
        max_jump = 0.0
        nonfinite = []  # (output row, model_id, non-finite continuum pixels) after the splice
        out_row = 0
        for r0 in range(0, n_rows, step):
            r1 = min(n_rows, r0 + step)
            k = keep[r0:r1]
            if not k.any():
                continue
            F = np.asarray(R["flux"][r0:r1], dtype=np.float32)[k]
            C = np.asarray(R["continuum"][r0:r1], dtype=np.float32)[k]
            src = [where[int(m)] for m in rmid[r0:r1][k]]
            for s in sorted({x[0] for x in src}):
                rows_here = np.array([i for i, x in enumerate(src) if x[0] == s])
                brows = np.array([src[i][1] for i in rows_here])
                order = np.argsort(brows)
                bf = np.asarray(blue[s]["flux"].oindex[brows[order], slice(0, ns)], dtype=np.float32)
                bc = np.asarray(blue[s]["continuum"].oindex[brows[order], slice(0, ns)], dtype=np.float32)
                inv = np.empty_like(order); inv[order] = np.arange(order.size)
                # continuity diagnostic at the splice: blue vs release normalised flux in the 1 A above the splice
                above = slice(ns, min(ns + 100, n_wl))
                bf_full = np.asarray(blue[s]["flux"].oindex[brows[order], above], dtype=np.float64)[inv]
                bc_full = np.asarray(blue[s]["continuum"].oindex[brows[order], above], dtype=np.float64)[inv]
                rel = F[rows_here][:, above].astype(np.float64) / C[rows_here][:, above]
                with np.errstate(invalid="ignore", divide="ignore"):
                    max_jump = max(max_jump, float(np.nanmax(np.abs(bf_full / bc_full - rel))))
                F[rows_here, :ns] = bf[inv]
                C[rows_here, :ns] = bc[inv]
            bad = (~np.isfinite(C) | (C <= 0)).sum(axis=1)
            for q in np.nonzero(bad)[0]:
                nonfinite.append((out_row + int(q), int(rmid[r0:r1][k][q]), int(bad[q])))
            out_row += int(k.sum())
            wf.append(F); wc.append(C)
            if (r0 // step) % 20 == 0:
                print(f"rows {r1}/{n_rows} {time.time() - t0:.0f} s", flush=True)
        assert wf.close() == n_out and wc.close() == n_out
        for name in ROW_ARRAYS:
            if name in R:
                arr = R[name]
                data = np.asarray(arr[...])[keep]
                if data.dtype.kind in "USOT":
                    from zarr_compat import write_string_array
                    write_string_array(dst, name, [str(x) for x in data.tolist()], compression_kw=kw)
                else:
                    create_array(dst, name, data=data, chunks=(chunk,) + data.shape[1:], **kw)
        for name in R.array_keys():
            if name in ("flux", "continuum", "physics_hash") or name in ROW_ARRAYS:
                continue
            copy_array(R, dst, name)
        copy_subgroups(R, dst)
        parent = str(np.asarray(R["physics_hash"][...]).ravel()[0]) if "physics_hash" in R else ""
        info = {"splice_angstrom": float(wl[ns]), "blue_pixels": ns, "blue_stores": a.blue, "note": a.note,
                "rows_in": int(n_rows), "rows_out": n_out, "dropped_model_ids": [int(x) for x in dropped],
                "max_abs_dnorm_blue_vs_release_first_A_above_splice": max_jump}
        new_hash = hashlib.sha256(json.dumps({"parent": parent, "blue_splice": info}, sort_keys=True).encode()).hexdigest()
        write_fixed_string_scalar(dst, "physics_hash", new_hash, min_width=64, compression_kw=kw)
        prov = dst["provenance"] if "provenance" in dst else dst.create_group("provenance")
        write_string_scalar(prov, "blue_splice.json", json.dumps(info, indent=1))
        dst.attrs.update({"n_models": n_out, "physics_hash": new_hash, "blue_splice": json.dumps(info)})
        names = [str(x) for x in np.asarray(R["param_names"][:]).tolist()]
        P = np.asarray(dst["params"][:], dtype=np.float64)
        kdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "known_issues"); os.makedirs(kdir, exist_ok=True)
        with open(os.path.join(kdir, "nonfinite_continuum_rows.csv"), "w") as fh:
            fh.write("row,model_id,nonfinite_continuum_pixels,teff,logg,feh,mu\n")
            for row, m, nb in nonfinite:
                fh.write(f"{row},{m},{nb}," + ",".join(f"{P[row, names.index(c)]:.6g}" for c in ("teff", "logg", "feh", "mu")) + "\n")
        info["nonfinite_continuum_rows"] = len(nonfinite)
        zarr.consolidate_metadata(zarr_store(tmp))
    print(json.dumps({k: v for k, v in info.items() if k != "dropped_model_ids"} | {"dropped": int(dropped.size),
                      "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
