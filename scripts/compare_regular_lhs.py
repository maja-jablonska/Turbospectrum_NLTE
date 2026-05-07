#!/usr/bin/env python3
"""End-to-end check that the regular-grid and LHS-grid generators emit the
same row at a matched physical point, and that downstream synthesis is
identical.

Picks one solar-like point with t_value_options spanning the available
MARCS labels {00, 01, 02, 05} so the snap-to-nearest logic is exercised.
"""
from __future__ import annotations

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from synthesize_regular_grid import _build_regular_columns
import generate_grid as gg
from spectrum_api import synthesize_spectrum

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
LINELIST = os.path.join(
    PROJECT_ROOT, "input_files/linelists/nlte_ges_linelist_jmg16jan2026_I_II"
)
ATMOSPHERES = os.path.join(
    PROJECT_ROOT, "input_files/model_atmospheres/1D/marcs_standard_comp/marcs_standard_comp"
)


# Point chosen so that turbvel=03 is genuinely *between* available atmosphere
# labels — exposes the snap behavior on both sides of an interior gap.
TEFF, LOGG, FEH = 5000, 4.0, 0.0
TURBVEL = "03"  # falls in the gap of {00, 01, 02, 05}; snaps to 02
T_VALUE_OPTIONS = ["00", "01", "02", "05"]
LAM_MIN, LAM_MAX, LAM_STEP = 6700.0, 6710.0, 0.05
OUTPUT_MODE = "Intensity"


def _regular_grid_row():
    abund = {k: np.asarray(["+0.00"], dtype=object) for k in ("a", "c", "n", "o", "r", "s")}
    cols = _build_regular_columns(
        teff_axis=np.asarray([TEFF], dtype=np.int64),
        logg_axis=np.asarray([LOGG], dtype=np.float64),
        feh_axis=np.asarray([FEH], dtype=np.float64),
        turbvel_axis=np.asarray([TURBVEL], dtype=object),
        mu_axis=None,
        grid_version="regular-test",
        lam_min=LAM_MIN,
        lam_max=LAM_MAX,
        lam_step=LAM_STEP,
        output_mode=OUTPUT_MODE,
        mode="1D",
        calculation_mode="LTE",
        abundances=abund,
        max_rows=10,
        t_value_options=T_VALUE_OPTIONS,
    )
    return {k: cols[k][0] for k in cols if cols[k].size}


def _lhs_grid_row():
    # Bounds collapsed to a near-zero-width interval around the test point so
    # the LHS sampler returns the exact same (teff, logg, feh, abundance)
    # values after rounding. turbvel is forced to '03' via a single-option
    # turbvel_options list.
    config = {
        "bounds": {
            "teff": {"min": TEFF, "max": TEFF + 1e-3},
            "logg": {"min": LOGG, "max": LOGG + 1e-4},
            "feh": {"min": FEH, "max": FEH + 1e-4},
        },
        "abundances": {k: "+0.00" for k in ("a", "c", "n", "o", "r", "s")},
        "sample_turbvel": True,
        "turbvel_options": [TURBVEL],
        "t_value_options": T_VALUE_OPTIONS,
        "num_samples": 1,
        "synthesis": {
            "lam_min": LAM_MIN,
            "lam_max": LAM_MAX,
            "lam_step": LAM_STEP,
            "output_mode": OUTPUT_MODE,
        },
        "mode": "1D",
        "calculation_mode": "LTE",
        "grid_version": "lhs-test",
    }
    rng = np.random.default_rng(12345)
    cols = gg._resolve_ml_sampling(config, rng=rng)
    return {k: cols[k][0] for k in cols if cols[k].size}


def _compare_rows(reg, lhs):
    # grid_version is a human-readable label, not a physics input — emit it
    # for inspection but exclude from the pass/fail decision.
    label_only = {"grid_version"}
    keys = sorted(set(reg) | set(lhs))
    rows = []
    mismatches = []
    for k in keys:
        rv = reg.get(k, "<missing>")
        lv = lhs.get(k, "<missing>")
        match = "yes"
        if k in label_only:
            match = "label-only"
            rows.append((k, str(rv), str(lv), match))
            continue
        if k in reg and k in lhs:
            try:
                if isinstance(rv, (np.floating, float)) or isinstance(lv, (np.floating, float)):
                    same = float(rv) == float(lv)
                else:
                    same = str(rv) == str(lv)
            except Exception:
                same = rv == lv
            if not same:
                match = "NO"
                mismatches.append(k)
        else:
            match = "only-one-side"
            mismatches.append(k)
        rows.append((k, str(rv), str(lv), match))
    return rows, mismatches


def main() -> int:
    print(f"Test point: teff={TEFF} logg={LOGG} feh={FEH} turbvel={TURBVEL!r}")
    print(f"t_value_options={T_VALUE_OPTIONS}\n")

    reg = _regular_grid_row()
    lhs = _lhs_grid_row()

    rows, mismatches = _compare_rows(reg, lhs)

    width = max(len(k) for k, *_ in rows) + 2
    print(f"{'column'.ljust(width)}{'regular'.ljust(20)}{'lhs'.ljust(20)}match")
    print("-" * (width + 44))
    for k, rv, lv, match in rows:
        print(f"{k.ljust(width)}{rv.ljust(20)}{lv.ljust(20)}{match}")

    print()
    if mismatches:
        print(f"FAIL: column mismatches: {mismatches}")
        return 1
    print("OK: every shared column agrees row-by-row.")
    print(f"    t_value snapped to {reg.get('t_value')!r} for turbvel={TURBVEL!r} (expected '02').")

    if reg.get("t_value") != "02":
        print("FAIL: expected t_value to snap to '02' (nearest available label).")
        return 1

    if not (os.path.isdir(ATMOSPHERES) and os.path.isfile(LINELIST)):
        print("\nSkipping synthesis: atmosphere/linelist data not present.")
        return 0

    def _synthesize_from_row(row, label):
        # Drive `synthesize_spectrum` with parameters pulled out of the row
        # dict that each generator actually emits — including the snapped
        # t_value, even though run_turbospectrum re-snaps internally.
        print(f"  [{label}] teff={row['teff']} logg={row['logg']} feh={row['feh']} "
              f"turbvel={row['turbvel']!r} t_value={row['t_value']!r}")
        return synthesize_spectrum(
            teff=int(row["teff"]),
            logg=float(row["logg"]),
            feh=float(row["feh"]),
            vmicro=float(row["turbvel"]),  # synthesis microturb (line formation)
            lambda_min=float(row["lam_min"]),
            lambda_max=float(row["lam_max"]),
            lambda_step=float(row["lam_step"]),
            output_mode=str(row["output_mode"]),
            abundances={k: float(row[k]) for k in ("a", "c", "n", "o", "r", "s") if k in row},
            linelist_files=[LINELIST],
            model_atmosphere_path=ATMOSPHERES,
        )

    print("\nSynthesizing from each generator's row dict (Intensity mode)...")
    res_reg = _synthesize_from_row(reg, "regular")
    res_lhs = _synthesize_from_row(lhs, "lhs    ")

    # Wavelength axis sanity
    if res_reg.wavelength.shape != res_lhs.wavelength.shape:
        print(f"FAIL: wavelength shape mismatch {res_reg.wavelength.shape} vs {res_lhs.wavelength.shape}")
        return 1
    max_dlam = float(np.max(np.abs(res_reg.wavelength - res_lhs.wavelength)))
    print(f"\nwavelength: {res_reg.wavelength.size} points, max |Δλ| = {max_dlam:.3e} Å")

    # mu axis
    if res_reg.mu_angles is None or res_lhs.mu_angles is None:
        print("FAIL: Intensity mode but mu_angles missing on one side")
        return 1
    if res_reg.mu_angles.shape != res_lhs.mu_angles.shape:
        print(f"FAIL: mu shape mismatch {res_reg.mu_angles.shape} vs {res_lhs.mu_angles.shape}")
        return 1
    max_dmu = float(np.max(np.abs(res_reg.mu_angles - res_lhs.mu_angles)))
    print(f"mu axis: {res_reg.mu_angles.size} points, max |Δmu| = {max_dmu:.3e}")
    print("mu values: " + ", ".join(f"{m:.4f}" for m in res_reg.mu_angles))

    # Intensity arrays — shape (n_mu, n_lambda)
    if res_reg.intensity.shape != res_lhs.intensity.shape:
        print(f"FAIL: intensity shape mismatch {res_reg.intensity.shape} vs {res_lhs.intensity.shape}")
        return 1

    print(f"\nIntensity comparison (shape = {res_reg.intensity.shape}):")
    print(f"{'mu':>8}  {'I_max [reg]':>14}  {'I_max [lhs]':>14}  {'max |ΔI|':>12}  {'max |ΔI|/I':>12}  "
          f"{'max |ΔI_norm|':>14}")
    print("-" * 90)
    overall_max_abs = 0.0
    overall_max_rel = 0.0
    overall_max_norm_abs = 0.0
    for i, mu in enumerate(res_reg.mu_angles):
        ir = res_reg.intensity[i]
        il = res_lhs.intensity[i]
        nr = res_reg.intensity_normalized[i]
        nl = res_lhs.intensity_normalized[i]
        diff = np.abs(ir - il)
        ndiff = np.abs(nr - nl)
        scale = np.maximum(np.abs(ir), 1e-30)
        rel = float(np.max(diff / scale))
        max_abs = float(np.max(diff))
        max_norm_abs = float(np.max(ndiff))
        overall_max_abs = max(overall_max_abs, max_abs)
        overall_max_rel = max(overall_max_rel, rel)
        overall_max_norm_abs = max(overall_max_norm_abs, max_norm_abs)
        print(f"{mu:>8.4f}  {float(ir.max()):>14.4e}  {float(il.max()):>14.4e}  "
              f"{max_abs:>12.3e}  {rel:>12.3e}  {max_norm_abs:>14.3e}")

    print()
    print(f"Overall max |ΔI|        = {overall_max_abs:.3e}")
    print(f"Overall max |ΔI|/|I|    = {overall_max_rel:.3e}")
    print(f"Overall max |ΔI_norm|   = {overall_max_norm_abs:.3e}")

    # Hard pass condition: bitwise identical (the row dicts agreed and
    # synthesis is deterministic, so this should hold).
    bitwise_identical = (
        np.array_equal(res_reg.intensity, res_lhs.intensity)
        and np.array_equal(res_reg.intensity_normalized, res_lhs.intensity_normalized)
    )
    if bitwise_identical:
        print("\nOK: regular and LHS intensities are bitwise-identical across all mu.")
        return 0

    # Soft pass: floating-point tolerance (in case of any non-determinism).
    tol_rel = 1e-6
    tol_norm = 1e-6
    if overall_max_rel <= tol_rel and overall_max_norm_abs <= tol_norm:
        print(f"\nOK: regular vs LHS intensities match within tolerance "
              f"(rel ≤ {tol_rel}, |ΔI_norm| ≤ {tol_norm}).")
        return 0

    print("\nFAIL: intensities differ beyond tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
