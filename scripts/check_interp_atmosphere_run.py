"""Acceptance check for a synthesis run made with model_atmosphere.selection = "interpolate".

Reads the merged store of RUN_ROOT and verifies, per row: the atmosphere columns describe the
requested point (marcs_teff/logg/fe_h == teff/logg/feh), marcs_turb == the fixed t-label,
marcs_interp is present (0 = node file, 1 + k = k widening steps), and the spectrum is finite and
positive. Shard stores are scanned for failed rows and their messages. Writes
RUN_ROOT/interp_check.json and exits 1 when fewer than --min-success of the expected rows passed.

    python scripts/check_interp_atmosphere_run.py RUN_ROOT --expected-rows 96
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--expected-rows", type=int, required=True)
    ap.add_argument("--min-success", type=float, default=0.95)
    ap.add_argument("--turb", type=float, default=2.0)
    a = ap.parse_args()

    report = {"run_root": a.run_root, "expected_rows": a.expected_rows}
    failures = collections.Counter()
    for shard in sorted(glob.glob(os.path.join(a.run_root, "**", "*shard_*.zarr"), recursive=True)):
        g = zarr.open(shard, mode="r")
        if "status" not in g:
            continue
        status = [str(s) for s in np.asarray(g["status"][:]).tolist()]
        messages = [str(m) for m in np.asarray(g["message"][:]).tolist()] if "message" in g else [""] * len(status)
        for s, m in zip(status, messages):
            if s.lower() != "success":
                failures[f"{s}: {m[:160]}"] += 1
    report["failed_rows_in_shards"] = sum(failures.values())
    report["failure_messages"] = dict(failures.most_common(20))

    merged = os.path.join(a.run_root, "outputs", "zarr", "synthesized_spectra.zarr")
    if not os.path.isdir(merged):
        report["error"] = f"merged store missing: {merged}"
        print(json.dumps(report, indent=1))
        json.dump(report, open(os.path.join(a.run_root, "interp_check.json"), "w"), indent=1)
        sys.exit(1)
    g = zarr.open(merged, mode="r")
    names = [str(n) for n in g["param_names"][:]]
    P = np.asarray(g["params"][:], float)
    col = {n: P[:, i] for i, n in enumerate(names)}
    n = len(P)
    checks = {}
    for req, atm in (("teff", "marcs_teff"), ("logg", "marcs_logg"), ("feh", "marcs_fe_h")):
        checks[f"{atm}_matches_{req}"] = bool(atm in col and np.allclose(col[atm], col[req], atol=1e-3))
    checks["marcs_turb_fixed"] = bool("marcs_turb" in col and np.allclose(col["marcs_turb"], a.turb))
    checks["marcs_interp_present"] = bool("marcs_interp" in col and np.isfinite(col["marcs_interp"]).all())
    good_flux = np.zeros(n, dtype=bool)
    for i in range(0, n, 500):  # chunked: a production store does not fit in memory
        f = np.asarray(g["flux"][i:i + 500], float)
        good_flux[i:i + 500] = np.isfinite(f).all(axis=1) & (f > 0).all(axis=1)
    checks["flux_finite_positive_rows"] = int(good_flux.sum())
    report["checks"] = checks
    report["merged_rows"] = n
    if "marcs_interp" in col:
        report["marcs_interp_counts"] = {str(int(k)): int(v) for k, v in zip(*np.unique(col["marcs_interp"], return_counts=True))}
    if "logg" in col:
        report["giant_rows"] = int((col["logg"] < 3.0).sum())
        report["cool_rows_below_4000K"] = int((col["teff"] < 4000).sum())
    ok_rows = int(good_flux.sum()) if all(v for k, v in checks.items() if isinstance(v, bool)) else 0
    report["passed_rows"] = ok_rows
    report["pass"] = ok_rows >= a.min_success * a.expected_rows
    print(json.dumps(report, indent=1))
    json.dump(report, open(os.path.join(a.run_root, "interp_check.json"), "w"), indent=1)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
