"""Extended acceptance check for interpolated-atmosphere test runs (beyond check_interp_atmosphere_run.py).

For each run (merged store under RUN_ROOT/outputs/zarr/synthesized_spectra.zarr):
  rows        every input row present exactly once (global_index == CSV row order), none dropped
  inputs      teff, logg, feh, vmicro, a, c, n, o, r, s, mu equal the rows CSV
  bracket     marcs_interp == the production bracket search for that row (0 node file, 1 + widening steps)
  wavelength  identical to the reference store's grid (the previous training grid) and strictly increasing
  spectrum    flux and continuum finite and > 0, normalised flux within [0, 1.05]
Across runs (same rows, different I/O setup): flux and continuum must agree to --max-rel-diff.

    python scripts/compare_interp_test_runs.py --rows-csv ROWS --reference-store REF.zarr \
        --atmosphere-path STORE --report OUT.json RUN_A [RUN_B ...]
Exit status 1 if any check fails.
"""
import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_turbospectrum as RT  # noqa: E402

INPUTS = {"teff": "teff", "logg": "logg", "feh": "feh", "vmicro": "turbvel", "a": "a", "c": "c", "n": "n",
          "o": "o", "r": "r", "s": "s", "mu": "mu"}


def load_run(root):
    g = zarr.open(os.path.join(root, "outputs", "zarr", "synthesized_spectra.zarr"), mode="r")
    names = [str(n) for n in g["param_names"][:]]
    P = np.asarray(g["params"][:], float)
    # flux/continuum stay lazy zarr arrays: a production store (25k rows x 313k px) does not fit in memory.
    return dict(gidx=np.asarray(g["global_index"][:]), cols={n: P[:, i] for i, n in enumerate(names)},
                flux=g["flux"], cont=g["continuum"],
                wl=np.asarray(g["wavelength"][:]), dropped_idx=np.asarray(g["dropped_global_index"][:], dtype=np.int64),
                dropped=int(np.asarray(g["dropped_global_index"][:]).size))


CHUNK = 500


def chunks(n):
    for i in range(0, n, CHUNK):
        yield slice(i, min(n, i + CHUNK))


COOL_CONTINUUM_TEFF = 3200.0  # below this the reconstructed continuum (flux / 5-decimal normalised flux) can be inf


def check_run(name, run, rows, ref_wl, expected_interp, keep):
    res, ok = {}, True
    n = len(keep)
    order = np.argsort(run["gidx"])
    gs = run["gidx"][order]
    missing = np.setdiff1d(keep, gs)
    res["rows"] = dict(merged=int(len(gs)), expected=n, dropped=run["dropped"],
                       global_index_is_csv_order=bool(np.array_equal(gs, keep)),
                       unique=bool(np.all(np.diff(gs) > 0)), subset_of_csv=bool(np.isin(gs, keep).all()),
                       missing_equals_dropped=bool(np.array_equal(np.sort(missing), np.sort(run["dropped_idx"]))),
                       missing_global_index=[int(x) for x in missing[:50]])
    ok &= res["rows"]["unique"] and res["rows"]["subset_of_csv"] and res["rows"]["missing_equals_dropped"]
    # inputs and brackets on every row that is present (dropped rows are accounted for above)
    present = [rows[i] for i in gs]
    res["inputs_max_abs_diff"] = {}
    for store_name, csv_name in INPUTS.items():
        want = np.array([float(r[csv_name]) for r in present])
        got = run["cols"][store_name][order]
        d = float(np.max(np.abs(got - want)))
        res["inputs_max_abs_diff"][store_name] = d
        ok &= np.allclose(got, want, rtol=1e-5, atol=1e-4)
    got_interp = run["cols"]["marcs_interp"][order]
    mism = int(np.sum(got_interp != expected_interp[gs]))
    res["bracket"] = dict(mismatches=mism, counts={str(int(k)): int(v) for k, v in zip(*np.unique(got_interp, return_counts=True))})
    ok &= mism == 0
    res["wavelength"] = dict(n=int(run["wl"].size), equal_to_reference=bool(np.array_equal(run["wl"], ref_wl)),
                             increasing=bool(np.all(np.diff(run["wl"]) > 0)),
                             range=[float(run["wl"][0]), float(run["wl"][-1])])
    ok &= res["wavelength"]["equal_to_reference"] and res["wavelength"]["increasing"]
    n_good, norm_min, norm_max, n_above, bad_rows, bad_flux, cont_rows = 0, np.inf, -np.inf, 0, [], [], []
    teff = run["cols"]["teff"]
    for sl in chunks(run["flux"].shape[0]):
        f = np.asarray(run["flux"][sl], float)
        c = np.asarray(run["cont"][sl], float)
        flux_ok = np.isfinite(f).all(1) & (f > 0).all(1)
        cont_ok = np.isfinite(c).all(1) & (c > 0).all(1)
        n_good += int((flux_ok & cont_ok).sum())
        bad_rows += [int(run["gidx"][sl.start + k]) for k in np.where(~(flux_ok & cont_ok))[0]]
        bad_flux += [int(run["gidx"][sl.start + k]) for k in np.where(~flux_ok)[0]]
        cont_rows += [(int(run["gidx"][sl.start + k]), float(teff[sl.start + k])) for k in np.where(flux_ok & ~cont_ok)[0]]
        cf = np.where(np.isfinite(c) & (c > 0), c, np.nan)
        norm = f / cf
        norm_min, norm_max = min(norm_min, float(np.nanmin(norm))), max(norm_max, float(np.nanmax(norm)))
        n_above += int(np.sum(np.nanmax(norm, axis=1) > 1.05))
    cont_hot = [g for g, t in cont_rows if t >= COOL_CONTINUUM_TEFF]
    res["spectrum"] = dict(rows_finite_positive=n_good, bad_rows_global_index=bad_rows[:50], norm_min=norm_min,
                           norm_max=norm_max, rows_norm_above_1_05=n_above,
                           bad_flux_rows=len(bad_flux), bad_flux_global_index=bad_flux[:50],
                           nonfinite_continuum_rows=len(cont_rows),
                           nonfinite_continuum_max_teff=max((t for _, t in cont_rows), default=None),
                           nonfinite_continuum_above_cool_limit=cont_hot[:50],
                           note=f"rows with a non-finite continuum below {COOL_CONTINUUM_TEFF:.0f} K are the known "
                                "5-decimal normalised-flux limitation (training preflight drops them)")
    ok &= not bad_flux and not cont_hot and norm_min >= 0 and n_above == 0
    res["pass"] = bool(ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rows-csv", required=True)
    ap.add_argument("--reference-store", required=True)
    ap.add_argument("--atmosphere-path", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-rel-diff", type=float, default=1e-6)
    ap.add_argument("--turb-label", default="02")
    ap.add_argument("--max-widen", type=int, default=2)
    ap.add_argument("--skip-rows", default="",
                    help="comma-separated CSV row indices deliberately not synthesised (e.g. a job stopped first)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.rows_csv)))
    ref_wl = np.asarray(zarr.open(a.reference_store, mode="r")["wavelength"][:])
    lib = RT.ModelInterpolator(SimpleNamespace(model_atmosphere_path=a.atmosphere_path))
    expected = []
    for r in rows:
        b = lib.find_interpolation_bracket(float(r["teff"]), float(r["logg"]), float(r["feh"]), a.turb_label,
                                           max_widen=a.max_widen)
        expected.append(0.0 if b["node_path"] else 1.0 + b["widen"])
    expected = np.array(expected)

    skip = {int(x) for x in a.skip_rows.split(",") if x.strip()}
    keep = np.array([i for i in range(len(rows)) if i not in skip])
    report, all_ok, runs = {"rows_csv": a.rows_csv, "reference_store": a.reference_store, "skip_rows": sorted(skip)}, True, {}
    for root in a.runs:
        runs[root] = load_run(root)
        report[root] = check_run(root, runs[root], rows, ref_wl, expected, keep)
        all_ok &= report[root]["pass"]
    roots = list(runs)
    for other in roots[1:]:
        A, B = runs[roots[0]], runs[other]
        if A["flux"].shape != B["flux"].shape:
            report[f"{roots[0]} vs {other}"] = {"pass": False, "error": "shape mismatch"}
            all_ok = False
            continue
        oa, ob = np.argsort(A["gidx"]), np.argsort(B["gidx"])
        fmax = cmax = 0.0
        identical = True
        for sl in chunks(len(oa)):
            fa, fb = np.asarray(A["flux"].oindex[oa[sl], :], float), np.asarray(B["flux"].oindex[ob[sl], :], float)
            ca, cb = np.asarray(A["cont"].oindex[oa[sl], :], float), np.asarray(B["cont"].oindex[ob[sl], :], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                fmax = max(fmax, float(np.nanmax(np.abs(fa - fb) / np.maximum(np.abs(fa), 1e-300))))
                cmax = max(cmax, float(np.nanmax(np.abs(ca - cb) / np.maximum(np.abs(ca), 1e-300))))
            identical &= bool(np.array_equal(fa, fb) and np.array_equal(ca, cb, equal_nan=True))
        cmp = dict(flux_max_rel_diff=fmax, cont_max_rel_diff=cmax, bitwise_identical=identical)
        cmp["pass"] = cmp["flux_max_rel_diff"] <= a.max_rel_diff and cmp["cont_max_rel_diff"] <= a.max_rel_diff
        report[f"{roots[0]} vs {other}"] = cmp
        all_ok &= cmp["pass"]
    report["pass"] = bool(all_ok)
    json.dump(report, open(a.report, "w"), indent=1)
    print(json.dumps(report, indent=1))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
