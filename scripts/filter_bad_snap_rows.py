#!/usr/bin/env python3
"""Filter rows whose recorded MARCS atmosphere is not the correct snap.

Datasets synthesised before the nearest-atmosphere selection fixes can contain
rows whose atmosphere was clamped to logg 3.0 (spherical models were invisible
to the snapper) or forced to solar metallicity (hard same-turbulence filter on
a sparse t-subset). The per-row ``marcs_*`` columns record the atmosphere that
was actually used, so each row carries the evidence needed to detect this.

For every row this script re-derives the *correct* nearest atmosphere from the
requested (teff, logg, feh, t-label) using the current production selector
(``run_turbospectrum.ModelInterpolator.find_nearest_model``, vectorised) and
flags rows whose recorded (marcs_teff, marcs_logg, marcs_fe_h) disagree.
Optional ``--max-d*`` thresholds additionally drop rows whose atmosphere is far
from the request even though the snap was correct (MARCS coverage holes, e.g.
hot low-logg corners where no model exists).

Without ``--output`` it only reports counts. With ``--output`` it writes a new
schema-compliant store containing the kept rows (streamed in row blocks; the
row-aligned arrays are sliced, everything else copied verbatim, provenance
subgroups preserved). The write is atomic: data lands in ``--output-tmp`` (same
filesystem) and is renamed into place.

Usage:
    python3 scripts/filter_bad_snap_rows.py -i spectra.zarr -a /path/to/marcs_dir
    python3 scripts/filter_bad_snap_rows.py -i spectra.zarr -a ... -o clean.zarr
    python3 scripts/filter_bad_snap_rows.py -i ... -a ... --max-dlogg 0.5 --mask-output bad.npy
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_turbospectrum import ModelInterpolator, SNAP_SCALES  # noqa: E402
from subsample_spectra import _copy_group_attrs, _copy_subgroups, _norm  # noqa: E402
from zarr_compat import (  # noqa: E402
    zarr_store,
    open_root_group,
    create_root_group,
    create_array,
    compression_kwargs,
    write_string_array,
)

# Recorded params are float32; every MARCS grid value (integer teff, multiples
# of 0.25/0.5) is exactly representable, so a small epsilon suffices.
_MATCH_EPS = 0.01

_REQUIRED = ("teff", "logg", "feh", "marcs_teff", "marcs_logg", "marcs_fe_h")


def _load_columns(src, names):
    """Fetch named per-row columns from either store schema.

    Supports the merged schema (``params`` [N, P] + ``param_names`` [P]) and
    per-column arrays (shard-style stores).
    """
    out = {}
    if "params" in src and "param_names" in src:
        param_names = [str(n) for n in src["param_names"][...]]
        params = src["params"]
        for name in names:
            if name in param_names:
                out[name] = np.asarray(params[:, param_names.index(name)], dtype=np.float64)
    for name in names:
        if name not in out and name in src:
            out[name] = np.asarray(src[name][...], dtype=np.float64)
    return out


def _model_arrays(atmosphere_path: str):
    lib = ModelInterpolator(SimpleNamespace(model_atmosphere_path=atmosphere_path))
    models = lib.available_models
    if not models:
        raise SystemExit(f"No MARCS models found in {atmosphere_path}")
    arrays = {
        "teff": np.array([m["teff"] for m in models], dtype=np.float64),
        "logg": np.array([m["logg"] for m in models], dtype=np.float64),
        "feh": np.array([m["feh"] for m in models], dtype=np.float64),
        "vmicro": np.array([float(m["turb"]) for m in models], dtype=np.float64),
        "geom_penalty": np.array([m["geom"] != "p" for m in models], dtype=np.float64),
        "mass_penalty": np.array([abs(m["mass"] - 1.0) for m in models], dtype=np.float64),
    }
    return lib, arrays


def compute_correct_snap(model_arrays, teff, logg, feh, t_value, *, block=512):
    """Vectorised replica of ``find_nearest_model`` over many rows.

    Returns (teff, logg, feh) arrays of the correct nearest atmosphere. Rows
    with NaN inputs yield NaN. Near-ties (distance within 1e-9) are resolved
    with the same (distance, plane-parallel-first, mass-nearest-1) ordering the
    production selector uses.
    """
    n = len(teff)
    out = np.full((n, 3), np.nan)
    valid = ~(np.isnan(teff) | np.isnan(logg) | np.isnan(feh) | np.isnan(t_value))
    idx_valid = np.where(valid)[0]
    ma = model_arrays
    for lo in range(0, len(idx_valid), block):
        rows = idx_valid[lo:lo + block]
        d = ((ma["teff"][None, :] - teff[rows, None]) / SNAP_SCALES["teff"]) ** 2
        d += ((ma["logg"][None, :] - logg[rows, None]) / SNAP_SCALES["logg"]) ** 2
        d += ((ma["feh"][None, :] - feh[rows, None]) / SNAP_SCALES["feh"]) ** 2
        d += ((ma["vmicro"][None, :] - t_value[rows, None]) / SNAP_SCALES["vmicro"]) ** 2
        best = np.argmin(d, axis=1)
        # Resolve near-ties exactly like the production lexicographic key.
        dmin = d[np.arange(len(rows)), best]
        for j in np.where((d <= dmin[:, None] + 1e-9).sum(axis=1) > 1)[0]:
            tied = np.where(d[j] <= dmin[j] + 1e-9)[0]
            order = np.lexsort((tied, ma["mass_penalty"][tied], ma["geom_penalty"][tied], d[j][tied]))
            best[j] = tied[order[0]]
        out[rows, 0] = ma["teff"][best]
        out[rows, 1] = ma["logg"][best]
        out[rows, 2] = ma["feh"][best]
    return out[:, 0], out[:, 1], out[:, 2]


def _spot_check(lib, model_arrays, cols, sample_rows) -> None:
    """Verify the vectorised snap against the production selector row-by-row."""
    for i in sample_rows:
        t_label = f"{int(round(cols['t_value'][i])):02d}"
        best, _ = lib.find_nearest_model(cols["teff"][i], cols["logg"][i], cols["feh"][i], t_label)
        vec = compute_correct_snap(
            model_arrays,
            *(np.array([cols[k][i]]) for k in ("teff", "logg", "feh", "t_value")),
        )
        got = (vec[0][0], vec[1][0], vec[2][0])
        want = (float(best["teff"]), float(best["logg"]), float(best["feh"]))
        if got != want:
            raise SystemExit(
                f"Vectorised snap disagrees with find_nearest_model on row {i}: "
                f"{got} != {want}. Refusing to filter."
            )


def classify_rows(src, atmosphere_path: str, *, max_dteff=None, max_dlogg=None, max_dfeh=None):
    """Return (keep_mask, stats, details) for the store's rows.

    ``details`` carries per-row evidence for reject reporting: ``reason``
    (empty string for kept rows; ``nan_marcs`` takes precedence over
    ``wrong_snap`` over ``out_of_tolerance``), the re-derived correct snap
    (``correct_teff``/``correct_logg``/``correct_feh``), and the loaded
    ``requested`` columns.
    """
    cols = _load_columns(src, _REQUIRED + ("t_value", "vmicro", "turbvel"))
    missing = [name for name in _REQUIRED if name not in cols]
    if missing:
        raise SystemExit(f"Store lacks required per-row columns: {', '.join(missing)}")
    n = len(cols["teff"])

    # The t-label actually targeted at synthesis time. t_value records it
    # directly; fall back to the requested vmicro (the selector treats the
    # label numerically, so the value is what matters).
    if "t_value" not in cols:
        fallback = cols.get("vmicro", cols.get("turbvel"))
        if fallback is None:
            raise SystemExit("Store has none of t_value/vmicro/turbvel; cannot re-derive the snap.")
        cols["t_value"] = fallback

    lib, model_arrays = _model_arrays(atmosphere_path)
    correct_teff, correct_logg, correct_feh = compute_correct_snap(
        model_arrays, cols["teff"], cols["logg"], cols["feh"], cols["t_value"]
    )
    rng = np.random.default_rng(0)
    finite = np.where(~np.isnan(correct_teff))[0]
    if len(finite):
        _spot_check(lib, model_arrays, cols, rng.choice(finite, size=min(100, len(finite)), replace=False))

    nan_rows = (
        np.isnan(cols["marcs_teff"]) | np.isnan(cols["marcs_logg"]) | np.isnan(cols["marcs_fe_h"])
        | np.isnan(correct_teff)
    )
    wrong_snap = ~nan_rows & (
        (np.abs(cols["marcs_teff"] - correct_teff) > _MATCH_EPS)
        | (np.abs(cols["marcs_logg"] - correct_logg) > _MATCH_EPS)
        | (np.abs(cols["marcs_fe_h"] - correct_feh) > _MATCH_EPS)
    )
    out_of_tolerance = np.zeros(n, dtype=bool)
    for threshold, requested, recorded in (
        (max_dteff, "teff", "marcs_teff"),
        (max_dlogg, "logg", "marcs_logg"),
        (max_dfeh, "feh", "marcs_fe_h"),
    ):
        if threshold is not None:
            out_of_tolerance |= ~nan_rows & (
                np.abs(cols[recorded] - cols[requested]) > float(threshold)
            )

    keep = ~(nan_rows | wrong_snap | out_of_tolerance)
    stats = {
        "n_rows": int(n),
        "n_kept": int(keep.sum()),
        "n_wrong_snap": int(wrong_snap.sum()),
        "n_out_of_tolerance": int((out_of_tolerance & ~wrong_snap).sum()),
        "n_nan": int(nan_rows.sum()),
    }
    reason = np.full(n, "", dtype=object)
    reason[out_of_tolerance] = "out_of_tolerance"
    reason[wrong_snap] = "wrong_snap"
    reason[nan_rows] = "nan_marcs"
    details = {
        "reason": reason,
        "correct_teff": correct_teff,
        "correct_logg": correct_logg,
        "correct_feh": correct_feh,
        "requested": cols,
    }
    return keep, stats, details


def collect_rejects(src, run_path: str, keep: np.ndarray, details: dict):
    """Return one dict per rejected row: lineage, reason, the full requested
    parameter vector, and the re-derived correct snap target."""
    idx = np.where(~keep)[0]
    if not len(idx):
        return []
    if "params" in src and "param_names" in src:
        names = [str(n) for n in src["param_names"][...]]
        data = np.asarray(src["params"][...], dtype=np.float64)[idx]
        param_rows = [dict(zip(names, map(float, data[j]))) for j in range(len(idx))]
    else:
        req = details["requested"]
        param_rows = [{k: float(v[i]) for k, v in req.items()} for i in idx]
    rows = []
    for j, i in enumerate(idx):
        row = {
            "run_path": run_path,
            "source_row": int(i),
            "reason": str(details["reason"][i]),
            "correct_marcs_teff": float(details["correct_teff"][i]),
            "correct_marcs_logg": float(details["correct_logg"][i]),
            "correct_marcs_feh": float(details["correct_feh"][i]),
        }
        row.update(param_rows[j])
        rows.append(row)
    return rows


def _axis_summary(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    if not len(values):
        return {}
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    hist, edges = np.histogram(values, bins=10)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "quantiles": {f"p{int(q * 100)}": float(v) for q, v in zip((0.05, 0.25, 0.5, 0.75, 0.95), quantiles)},
        "histogram": {"bin_edges": [float(e) for e in edges], "counts": [int(c) for c in hist]},
    }


def summarize_rejects(reject_rows, total_rows: int) -> dict:
    """Aggregate rejected rows into the parameter-space view needed to plan a
    re-run: counts by reason and run, per-axis coverage of the rejected
    requests, and the number of unique physical points (mu excluded)."""
    reasons = {}
    per_run = {}
    for row in reject_rows:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        per_run[row["run_path"]] = per_run.get(row["run_path"], 0) + 1

    lineage_keys = {"run_path", "source_row", "reason",
                    "correct_marcs_teff", "correct_marcs_logg", "correct_marcs_feh"}
    axes = {}
    if reject_rows:
        param_keys = [k for k in reject_rows[0] if k not in lineage_keys]
        for key in ("teff", "logg", "feh", "vmicro", "turbvel", "t_value"):
            if key in param_keys:
                axes[key] = _axis_summary(np.array([row.get(key, np.nan) for row in reject_rows], dtype=float))
        point_keys = [k for k in param_keys if k != "mu" and not k.startswith("marcs_")]
        unique_points = {
            tuple(round(float(row.get(k, float("nan"))), 6) for k in point_keys)
            for row in reject_rows
        }
    else:
        unique_points = set()

    return {
        "n_rows_total": int(total_rows),
        "n_rejected": len(reject_rows),
        "rejected_fraction": len(reject_rows) / max(total_rows, 1),
        "rejects_by_reason": reasons,
        "rejects_by_run": per_run,
        "n_unique_rejected_points_excl_mu": len(unique_points),
        "rejected_request_axes": axes,
    }


def write_rejects(csv_path: str, reject_rows, total_rows: int) -> str:
    """Write the rejected rows to CSV and an aggregate summary JSON next to it.

    Returns the summary path. Column order: lineage first, then the union of
    parameter columns in first-seen order.
    """
    import csv as _csv
    import json as _json

    lineage = ["run_path", "source_row", "reason",
               "correct_marcs_teff", "correct_marcs_logg", "correct_marcs_feh"]
    header = list(lineage)
    for row in reject_rows:
        for key in row:
            if key not in header:
                header.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(reject_rows)

    summary_path = (csv_path[:-4] if csv_path.endswith(".csv") else csv_path) + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        _json.dump(summarize_rejects(reject_rows, total_rows), fh, indent=2, sort_keys=True)
    return summary_path


def write_filtered(src, dst_path: str, tmp_path: str, keep: np.ndarray, stats: dict,
                   atmosphere_path: str, thresholds: dict) -> None:
    if os.path.exists(dst_path):
        raise FileExistsError(f"Output already exists, refusing to overwrite: {dst_path}")
    if os.path.exists(tmp_path):
        raise FileExistsError(f"Temp output already exists (stale run?): {tmp_path}")

    n_rows = len(keep)
    rows = np.where(keep)[0].astype(np.int64)
    comp = compression_kwargs()
    dst = create_root_group(zarr_store(tmp_path), overwrite=False)
    _copy_group_attrs(src, dst)

    def _write_small(name, arr, data):
        """Write a fully-materialised array; strings go through the vlen writer."""
        if data.dtype.kind in "USO":
            if data.ndim != 1:
                raise ValueError(f"Unsupported multi-dimensional string array: {name}")
            return write_string_array(dst, name, [str(v) for v in data], compression_kw=comp)
        return create_array(dst, name, data=data, chunks=arr.chunks, **comp)

    sliced, copied = [], []
    for name in src.array_keys():
        arr = src[name]
        row_aligned = bool(arr.shape) and arr.shape[0] == n_rows
        if row_aligned and arr.dtype.kind not in "USO":
            chunks = None
            if arr.chunks is not None:
                chunks = (min(len(rows), arr.chunks[0]) or 1,) + tuple(arr.chunks[1:])
            new = create_array(
                dst, name, shape=(len(rows),) + tuple(arr.shape[1:]),
                dtype=arr.dtype, chunks=chunks, **comp,
            )
            # Stream in source-chunk-aligned row blocks so a 300k x 300k-lambda
            # flux array is never materialised in memory.
            block = arr.chunks[0] if arr.chunks else 4096
            cursor = 0
            for lo in range(0, n_rows, block):
                hi = min(lo + block, n_rows)
                local = rows[(rows >= lo) & (rows < hi)] - lo
                if len(local):
                    data = arr[lo:hi]
                    new[cursor:cursor + len(local)] = data[local]
                    cursor += len(local)
            sliced.append(name)
        elif row_aligned:
            new = _write_small(name, arr, arr[...][rows])
            sliced.append(name)
        else:
            new = _write_small(name, arr, arr[...])
            copied.append(name)
        _copy_group_attrs(arr, new)

    create_array(dst, "source_row_index", data=rows, chunks=(max(len(rows), 1),))
    _copy_subgroups(src, dst)
    meta = dst.create_group("snap_filter")
    meta.attrs["atmosphere_path"] = atmosphere_path
    meta.attrs["criteria"] = "recorded marcs_(teff|logg|fe_h) must equal the re-derived nearest atmosphere"
    meta.attrs["thresholds"] = {k: v for k, v in thresholds.items() if v is not None}
    for key, value in stats.items():
        meta.attrs[key] = value

    os.rename(tmp_path, dst_path)
    print(f"Wrote {len(rows)} rows -> {dst_path}")
    print(f"  sliced arrays ({len(sliced)}): {', '.join(sorted(sliced))}")
    print(f"  copied arrays ({len(copied)}): {', '.join(sorted(copied))}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="Source spectra .zarr store")
    ap.add_argument("-a", "--atmosphere-path", required=True,
                    help="MARCS model directory the dataset was synthesised against")
    ap.add_argument("-o", "--output", help="Destination .zarr store for kept rows (omit to only report)")
    ap.add_argument("--output-tmp", help="Temp path for the atomic write (default: <output>.tmp, same filesystem)")
    ap.add_argument("--mask-output", help="Optional .npy path for the boolean keep-mask")
    ap.add_argument("--rejects-csv", help="Optional CSV path for rejected rows (a *_summary.json lands next to it)")
    ap.add_argument("--max-dteff", type=float, help="Also drop rows with |marcs_teff - teff| above this (K)")
    ap.add_argument("--max-dlogg", type=float, help="Also drop rows with |marcs_logg - logg| above this (dex)")
    ap.add_argument("--max-dfeh", type=float, help="Also drop rows with |marcs_fe_h - feh| above this (dex)")
    args = ap.parse_args(argv)

    input_path = _norm(args.input)
    atmosphere_path = _norm(args.atmosphere_path)
    src = open_root_group(input_path, mode="r")
    keep, stats, details = classify_rows(
        src, atmosphere_path,
        max_dteff=args.max_dteff, max_dlogg=args.max_dlogg, max_dfeh=args.max_dfeh,
    )

    print(f"rows:              {stats['n_rows']}")
    print(f"  wrong snap:      {stats['n_wrong_snap']}")
    print(f"  out of tolerance:{stats['n_out_of_tolerance']}")
    print(f"  NaN marcs cols:  {stats['n_nan']}")
    print(f"  kept:            {stats['n_kept']} ({stats['n_kept'] / max(stats['n_rows'], 1) * 100:.1f}%)")

    if args.mask_output:
        np.save(_norm(args.mask_output), keep)
        print(f"Keep-mask written to {args.mask_output}")

    if args.rejects_csv:
        reject_rows = collect_rejects(src, input_path, keep, details)
        summary_path = write_rejects(_norm(args.rejects_csv), reject_rows, stats["n_rows"])
        print(f"Rejects written to {args.rejects_csv} (summary: {summary_path})")

    if args.output:
        output_path = _norm(args.output)
        tmp_path = _norm(args.output_tmp) if args.output_tmp else output_path + ".tmp"
        thresholds = {"max_dteff": args.max_dteff, "max_dlogg": args.max_dlogg, "max_dfeh": args.max_dfeh}
        write_filtered(src, output_path, tmp_path, keep, stats, atmosphere_path, thresholds)
    elif not args.mask_output and not args.rejects_csv:
        print("(report only; pass --output to write the filtered store)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
