#!/usr/bin/env python3
"""List the released rows whose continuum has non-finite pixels and add them to the dataset card.

Reads the per-array extended reports (``<fleet>/jXXX/extended_check.json``), maps each flagged row to its row in
the released merged store through ``model_id``, re-counts the non-finite pixels there, writes
``<release>/known_issues/nonfinite_continuum_rows.csv`` and inserts a bullet under "## Known gaps" in README.md.

    python scripts/release_known_issues.py FLEET_ROOT RELEASE_DIR
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np
import zarr


def main() -> None:
    fleet, release = sys.argv[1], sys.argv[2]
    store = zarr.open_group(os.path.join(release, "harps_interp_t02.zarr"), mode="r")
    names = [str(n) for n in store["param_names"][:]]
    params = np.asarray(store["params"][:], dtype=np.float64)
    model_id = np.asarray(store["model_id"][:])
    row_of = {int(m): i for i, m in enumerate(model_id)}
    rows = []
    for report in sorted(glob.glob(os.path.join(fleet, "j0??", "extended_check.json"))):
        array_dir = os.path.dirname(report)
        r = json.load(open(report))
        key = [k for k in r if k.startswith("/")][0]
        spec = r[key]["spectrum"]
        flagged = spec.get("bad_rows_global_index", [])
        if len(flagged) >= 50:
            raise SystemExit(f"{report}: flagged list truncated at 50; extend the report before using it")
        if not flagged:
            continue
        a = zarr.open_group(os.path.join(array_dir, "outputs", "zarr", "synthesized_spectra.zarr"), mode="r")
        gidx = np.asarray(a["global_index"][:]); mids = np.asarray(a["model_id"][:])
        for g in flagged:
            m = int(mids[int(np.nonzero(gidx == g)[0][0])])
            i = row_of[m]
            cont = np.asarray(store["continuum"][i]); flux = np.asarray(store["flux"][i])
            if not (np.isfinite(flux).all() and (flux > 0).all()):
                raise SystemExit(f"row {i} (model_id {m}) has bad FLUX, not only continuum")
            p = {n: float(params[i, names.index(n)]) for n in ("teff", "logg", "feh", "mu")}
            rows.append(dict(row=i, model_id=m, array=os.path.basename(array_dir), array_global_index=int(g),
                             nonfinite_continuum_pixels=int((~np.isfinite(cont)).sum()), **p))
    rows.sort(key=lambda x: x["row"])
    out_dir = os.path.join(release, "known_issues"); os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "nonfinite_continuum_rows.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(rows); px = [r["nonfinite_continuum_pixels"] for r in rows]; te = [r["teff"] for r in rows]
    bullet = (f"* {n} spectra ({100 * n / params.shape[0]:.2f}%), all with Teff {min(te):.0f}-{max(te):.0f} K, have non-finite values in "
              f"`continuum` at {min(px)}-{max(px)} of {store['continuum'].shape[1]:,} pixels (their `flux` is finite and positive). "
              "In saturated pixels Turbospectrum's 5-decimal normalised flux rounds to 0, so continuum = flux / normalised flux "
              "is infinite. Their row numbers are listed in `known_issues/nonfinite_continuum_rows.csv`; mask the non-finite "
              "pixels or skip these rows.\n")
    readme = os.path.join(release, "README.md")
    text = open(readme, encoding="utf-8").read()
    if "known_issues/nonfinite_continuum_rows.csv" not in text:
        marker = "## Known gaps\n\n"
        if marker not in text:
            raise SystemExit("README has no '## Known gaps' section")
        text = text.replace(marker, marker + bullet, 1)
        open(readme, "w", encoding="utf-8").write(text)
    print(json.dumps({"rows": n, "teff": [min(te), max(te)], "pixels": [min(px), max(px)], "csv": out_csv}))


if __name__ == "__main__":
    main()
