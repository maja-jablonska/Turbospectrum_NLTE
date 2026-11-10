"""Continuity scans for interpolated-atmosphere synthesis.

Synthesises fine one-parameter scans across the places where interpolated atmospheres could jump: a Teff node,
the spherical/plane-parallel switch at log g 3, [Fe/H] nodes, the synthesis microturbulence, and boundaries
where the interpol_modeles bracket widens around unconverged MARCS models. For every spectrum it records the
continuum at the window centre and the mean line depth (1 - mean normalised flux); a step between
neighbouring points is compared with the local trend (median of the neighbouring steps), and the largest
excess per scan is reported. Uses the production line lists (window-trimmed molecules).

    python scripts/scan_interp_continuity.py OUT_DIR [--workers 7]
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_turbospectrum as RT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINELISTS = ["nlte_ges_linelist_jmg16jan2026_I_II", "Hlinedata", "vald-3700-3800-for-grid-nlte-16jan2026",
             "vald-3800-4200-for-grid-nlte-16jan2026", "molecules-3760-6930A/*.bsyn"]
WINDOW = (5025.0, 5075.0, 0.01)
BASE = dict(teff=5800, logg=4.4, feh=0.0, turbvel=1.5, mu=1.0)
SCANS = {
    "teff_node_5750": ("teff", np.arange(5600, 6001, 10), dict(logg=4.4, feh=0.0)),
    "logg_geometry_switch": ("logg", np.round(np.arange(2.80, 3.201, 0.02), 3), dict(teff=5000, feh=0.0, turbvel=2.0)),
    "feh_nodes": ("feh", np.round(np.arange(-0.40, 0.101, 0.025), 3), dict(teff=5800, logg=4.4)),
    "vmicro_dcep": ("turbvel", np.round(np.arange(1.0, 5.01, 0.25), 2), dict(teff=6000, logg=2.0, feh=0.0)),
    "widen_logg1.2_feh-0.3": ("teff", np.arange(5400, 6151, 15), dict(logg=1.2, feh=-0.3, turbvel=3.0)),
    "widen_logg1.6_feh0": ("teff", np.arange(6150, 6401, 10), dict(logg=1.6, feh=0.0, turbvel=3.0)),
}


def run_point(job):
    scan, i, params, out_dir = job
    d = os.path.join(out_dir, "work", f"{scan}_{i:03d}")
    out = os.path.join(out_dir, "points", f"{scan}_{i:03d}.npz")
    if os.path.exists(out):
        return out
    shutil.rmtree(d, ignore_errors=True)
    for s in ("spectra", "logs", "tmp", "opac"):
        os.makedirs(os.path.join(d, s))
    cfg = RT.TurbospectrumConfig(project_root=ROOT, lambda_min=WINDOW[0], lambda_max=WINDOW[1], lambda_step=WINDOW[2],
                                 output_mode="Intensity", linelist_files=LINELISTS, resolution=115000.0,
                                 output_dir=f"{d}/spectra", log_dir=f"{d}/logs", tmp_dir=f"{d}/tmp", model_opac_dir=f"{d}/opac",
                                 force=True, atmosphere_selection="interpolate", atmosphere_max_widen=2)
    cfg.linelist_file_path = RT.create_linelist_file(cfg)
    r = RT.run_single_synthesis((params, cfg))
    if r["status"] != "success":
        np.savez(out, ok=False, message=r["message"][:500])
    else:
        a = np.loadtxt(r["output_path"])
        cont = a[:, 2] / a[:, 1]
        np.savez(out, ok=True, cont=float(np.median(cont[len(cont) // 2 - 50: len(cont) // 2 + 50])),
                 depth=float(1 - np.mean(a[:, 1])), interp=float(r.get("marcs_interp", np.nan)))
    shutil.rmtree(d, ignore_errors=True)
    return out


def step_excess(y):
    """Largest |step - local median step| relative to the series' typical value (fraction)."""
    d = np.diff(y)
    exc = np.zeros_like(d)
    for k in range(len(d)):
        nb = np.concatenate([d[max(0, k - 3):k], d[k + 1:k + 4]])
        exc[k] = abs(d[k] - np.median(nb))
    k = int(np.argmax(exc))
    return k, float(exc[k] / np.median(np.abs(y))), float(np.median(np.abs(d)) / np.median(np.abs(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "points"), exist_ok=True)
    jobs = []
    for scan, (key, values, fixed) in SCANS.items():
        for i, v in enumerate(values):
            p = {**BASE, **fixed, key: float(v)}
            p["teff"] = int(round(p["teff"]))
            jobs.append((scan, i, p, a.out_dir))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(run_point, jobs)):
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)

    report = {}
    for scan, (key, values, fixed) in SCANS.items():
        pts = [dict(np.load(os.path.join(a.out_dir, "points", f"{scan}_{i:03d}.npz"))) for i in range(len(values))]
        failed = [float(values[i]) for i, p in enumerate(pts) if not bool(p["ok"])]
        ok = [i for i, p in enumerate(pts) if bool(p["ok"])]
        x = np.array([values[i] for i in ok], float)
        cont = np.array([float(pts[i]["cont"]) for i in ok])
        depth = np.array([float(pts[i]["depth"]) for i in ok])
        interp = np.array([float(pts[i]["interp"]) for i in ok])
        kc, exc_c, typ_c = step_excess(cont)
        kd, exc_d, typ_d = step_excess(depth)
        report[scan] = dict(parameter=key, fixed=fixed, n=len(values), failed=failed,
                            marcs_interp_changes=[float(x[i + 1]) for i in range(len(x) - 1) if interp[i + 1] != interp[i]],
                            continuum=dict(typical_step=typ_c, max_excess_step=exc_c, at=[float(x[kc]), float(x[kc + 1])]),
                            line_depth=dict(typical_step=typ_d, max_excess_step=exc_d, at=[float(x[kd]), float(x[kd + 1])]),
                            series=dict(x=x.tolist(), cont=cont.tolist(), depth=depth.tolist(), interp=interp.tolist()))
        print(f"{scan:24s} cont: typical step {100*typ_c:.3f}%  worst excess {100*exc_c:.3f}% at {x[kc]:g}->{x[kc+1]:g} | "
              f"depth: typical {100*typ_d:.3f}%  worst excess {100*exc_d:.3f}% at {x[kd]:g}->{x[kd+1]:g} | "
              f"bracket changes at {report[scan]['marcs_interp_changes']} | failed {failed}")
    json.dump(report, open(os.path.join(a.out_dir, "continuity_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
