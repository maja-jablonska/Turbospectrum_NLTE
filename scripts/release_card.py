#!/usr/bin/env python3
"""Write the Hugging Face dataset card (README.md) for a merged interpolated-atmosphere grid store.

Everything quantitative (row count, parameter ranges, wavelength grid, atmosphere bracket statistics,
attrs) is read from the store itself; the physics description matches the harps-interp-t02 campaign.

    python scripts/release_card.py STORE.zarr README.md --repo-id mjablonska/harps-interp-t02-grid
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import zarr

DESCRIPTIONS = {
    "teff": "effective temperature (K); also the atmosphere's (interpolated) Teff",
    "logg": "surface gravity log g (cgs dex); also the atmosphere's",
    "feh": "[Fe/H]: metallicity of the line formation and of the (interpolated) atmosphere",
    "vmicro": "microturbulence of the line formation (km/s); the atmosphere is always the xi = 2 km/s MARCS model",
    "a": "[alpha/Fe] (bsyn ALPHA/Fe keyword: O Ne Mg Si S Ar Ca Ti)",
    "c": "[C/Fe]: A(C) = 8.56 + [Fe/H] + c",
    "n": "[N/Fe]: A(N) = 7.98 + [Fe/H] + n",
    "o": "oxygen relative to the alpha enhancement: A(O) = 8.77 + [Fe/H] + a + o  (so [O/Fe] = a + o)",
    "r": "[r/Fe] (bsyn R-PROCESS keyword)",
    "s": "[s/Fe] (bsyn S-PROCESS keyword)",
    "mu": "cosine of the viewing angle of the emergent intensity (one exact mu per spectrum)",
    "marcs_teff": "atmosphere Teff (= teff: interpolated at the requested point)",
    "marcs_logg": "atmosphere log g (= logg)",
    "marcs_fe_h": "atmosphere [Fe/H] (= feh)",
    "marcs_a_fe": "[alpha/Fe] of the MARCS standard composition at this [Fe/H] (blended between nodes)",
    "marcs_turb": "atmosphere microturbulence label (always 2 km/s)",
    "marcs_interp": "atmosphere bracket: 0 = MARCS node used as is, 1 + k = interpolated with k widening steps",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store"); ap.add_argument("readme")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--pending-rows", type=int, default=0, help="rows to be added later (re-run of failed rows)")
    a = ap.parse_args()

    g = zarr.open_group(a.store, mode="r")
    names = [str(n) for n in np.asarray(g["param_names"][:]).tolist()]
    P = np.asarray(g["params"][:], dtype=np.float64)
    col = {n: P[:, i] for i, n in enumerate(names)}
    wl = np.asarray(g["wavelength"][:], dtype=np.float64)
    n_rows = P.shape[0]
    attrs = dict(g.attrs)
    rows = []
    for n in names:
        v = col[n]; fin = np.isfinite(v)
        if not fin.any():
            continue
        lo, hi = float(v[fin].min()), float(v[fin].max())
        rows.append(f"| `{n}` | {lo:.4g} | {hi:.4g} | {DESCRIPTIONS.get(n, '(bookkeeping)')} |")
    interp = ""
    if "marcs_interp" in col:
        k, c = np.unique(col["marcs_interp"][np.isfinite(col["marcs_interp"])], return_counts=True)
        interp = ", ".join(f"{int(x)}: {int(y)}" for x, y in zip(k, c))
    giants = int((col["logg"] < 3.0).sum()) if "logg" in col else 0

    card = f"""---
pretty_name: HARPS-range Turbospectrum intensities on interpolated MARCS atmospheres
tags:
- astronomy
- stellar-spectra
- synthetic-spectra
- turbospectrum
- marcs
- zarr
size_categories:
- 100K<n<1M
---

# HARPS-range synthetic intensity grid on interpolated MARCS atmospheres (`harps-interp-t02`)

{n_rows:,} LTE Turbospectrum spectra of the emergent specific intensity I(lambda, mu), {wl[0]:.2f}-{wl[-1]:.2f} A at
{np.median(np.diff(wl)):.3f} A sampling ({wl.size:,} pixels), not convolved with any instrumental or rotational profile.
Training set for the SPICE spectral emulator (https://arxiv.org/abs/2511.10998).

## How the spectra were made

* **Parameters**: a Latin hypercube over Teff 3000-8000 K, log g -0.5 to 5.25, [Fe/H] -5 to +1, vmicro 1-5 km/s and six
  abundance offsets (table below), one random exact mu per spectrum; points without a complete MARCS bracket were removed.
* **Atmospheres**: MARCS standard-composition models interpolated to each (Teff, log g, [Fe/H]) with Masseron's
  `interpol_modeles` (linear in Teff, log g, [Fe/H] between the 8 bracketing models) instead of the nearest node. Fixed
  microturbulence label t02 (2 km/s) for all atmospheres; spherical 1 Msun models below log g 3, plane-parallel at and
  above. Brackets with missing (unconverged) MARCS models are widened by up to 2 node steps; `marcs_interp` records it
  (counts: {interp}). {giants:,} spectra have log g < 3 (spherical).
* **Synthesis**: Turbospectrum (babsma + bsyn) in Intensity mode, LTE; solar abundances Magg et al. (2022). Line lists:
  Gaia-ESO atomic list (`nlte_ges_linelist_jmg16jan2026_I_II`, used in LTE), VALD 3700-4200 A, hydrogen lines
  (`Hlinedata`) and the Gaia-ESO molecular lists (C2, CN, CH, NH, OH, MgH, SiH, CaH, TiO, VO, ZrO), which start at
  4200 A: the spectra contain no molecular lines between 3780 and 4200 A.
* **Continuum**: `continuum` = flux / normalised flux as printed by Turbospectrum (normalised flux has 5 decimals, so
  below ~3150 K the continuum is coarse in saturated pixels).

## Contents

One Zarr v3 store, `harps_interp_t02.zarr`:

| array | shape | |
|---|---|---|
| `flux` | ({n_rows}, {wl.size}) float32 | emergent intensity at `mu` ({attrs.get('flux_definition', '')}, {attrs.get('flux_unit', '')}) |
| `continuum` | ({n_rows}, {wl.size}) float32 | continuum intensity |
| `wavelength` | ({wl.size},) float32 | A |
| `params` / `param_names` | ({n_rows}, {len(names)}) float32 | per-spectrum parameters (below) |
| `model_id`, `run_index`, `source_physics_hash` | ({n_rows},) | row identity and lineage |
| `provenance/` | | configs, line-list / atmosphere / software manifests of the synthesis runs |

Chunks hold 128 spectra x all wavelengths (Blosc zstd). Read with e.g.
`zarr.open_group("harps_interp_t02.zarr", mode="r")["flux"][i0:i1]`.

| parameter | min | max | meaning |
|---|---|---|---|
""" + "\n".join(rows) + f"""

## Known gaps

* No molecular lines below 4200 A: the molecular line lists start at 4200 A, so the CN violet bands (3883 A and most of
  the 4216 A band, which extends blueward of its head) and CH/NH/OH lines at 3780-4200 A are missing. Atomic and
  hydrogen lines cover the full range. Molecular features at 4200-6910 A (G band, C2 Swan, MgH, TiO, ...) are included.
* Rows whose Turbospectrum chemical equilibrium did not converge (cool stars near C/O = 1) are absent (a handful).
""" + (f"* {a.pending_rows} rows that failed in an early version of the atmosphere interpolation will be added in a later commit.\n"
       if a.pending_rows else "") + f"""
## Provenance

Generated by the Turbospectrum_NLTE pipeline (git {str(attrs.get('git_commit', ''))[:12]}), schema {attrs.get('schema_version', '')};
see `provenance/` and the store attributes for the full configuration. Contact: Maja Jablonska.
"""
    with open(a.readme, "w", encoding="utf-8") as fh:
        fh.write(card)
    print(json.dumps({"rows": n_rows, "params": names, "wavelength": [float(wl[0]), float(wl[-1]), int(wl.size)],
                      "marcs_interp": interp, "readme": a.readme}))


if __name__ == "__main__":
    main()
