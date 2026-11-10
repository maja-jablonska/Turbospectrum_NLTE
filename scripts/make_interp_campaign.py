"""Build the harps-interp-t02 synthesis campaign: rows, per-job configs and PBS scripts.

Input: the prefiltered LHS grid (pipeline_from_config --skip-synthesis on config_harps_ml_interp.json).
Output directory layout (uploaded to Gadi as is):
  rows/test_rows.csv                     96 rows spanning dwarfs/giants/M stars/widened brackets
  rows/rows_jNNN.csv                     the N_ROWS campaign rows, shuffled, ROWS_PER_SUBJOB * 10 per array
  configs/config_harps_interp_t02_{test,jNNN}.json
  pbs/<test-tag>.pbs, pbs/array_jNNN.pbs, pbs/finalize_jNNN.pbs, pbs/submit_fleet.sh

Synthesis settings are copied from the cno_v2 wave config (same window, Intensity + exact mu, R, line
lists), plus model_atmosphere.selection = "interpolate" (t02, max_widen 2) and the Linux interpolator. The
molecular lists are the window-trimmed copy (scripts/trim_bsyn_linelist.py, 3760-6930 A; spectra identical,
~30% less parsing per row). Every synthesis job stages its line lists to $PBS_JOBFS (TS_LINELIST_ROOT) and
keeps its per-row scratch there (SCRATCH_TMP_ROOT); shard temp outputs stay on /scratch so the final rename
is atomic. test.pbs builds the trimmed lists on Gadi first and verifies them against MOLECULE_SHA256.

    python scripts/make_interp_campaign.py GRID_ZARR CNO_V2_W00_CONFIG OUT_DIR [--rows 300000]
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

GADI_ROOT = "/scratch/mk27/mj8805/harps_interp_t02"
TS_ROOT = "/scratch/mk27/mj8805/Turbospectrum_NLTE"
PYTHON_BIN = "/g/data/y89/mj8805/miniforge/envs/astro/bin/python"
COLUMNS = ["teff", "logg", "feh", "turbvel", "t_value", "a", "c", "n", "o", "r", "s", "mu"]
ATMOSPHERE = {"selection": "interpolate", "turb_label": "02", "max_widen": 2, "spherical_below_logg": 3.0}
MOLECULES_FULL, MOLECULES_TRIMMED = "molecules-420-920nm", "molecules-3760-6930A"
MOLECULE_SHA256 = f"{GADI_ROOT}/molecules-3760-6930A.sha256"
LINELIST_DIR = f"{TS_ROOT}/input_files/linelists"

PBS_HEAD = """#!/bin/bash
#PBS -P mk27
#PBS -q {queue}
#PBS -N {name}
#PBS -r y
#PBS -l ncpus={ncpus}
#PBS -l mem={mem}
#PBS -l jobfs={jobfs}
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -m a
#PBS -l storage=scratch/mk27+scratch/y89+gdata/y89
"""


def load_grid(path):
    g = zarr.open(path, mode="r")
    cols = {}
    for name in COLUMNS:
        arr = np.asarray(g[name][:])
        cols[name] = arr.astype(str) if arr.dtype.kind in "OUS" else arr
    return cols


def fmt_row(cols, i):
    out = []
    for name in COLUMNS:
        v = cols[name][i]
        if name == "teff":
            out.append(str(int(round(float(v)))))
        elif name == "t_value":
            out.append(str(v))
        else:
            out.append(f"{float(v):.6g}")
    return out


def pick_test_rows(cols, idx, lib, rng):
    """96 rows: 24 dwarfs >= 4000 K, 24 giants >= 4000 K, 16 below 4000 K, 16 widened brackets, 16 random."""
    T, G, Z = (np.asarray(cols[k], float)[idx] for k in ("teff", "logg", "feh"))
    widen = np.array([lib.find_interpolation_bracket(T[i], G[i], Z[i], "02", max_widen=2)["widen"] for i in range(len(idx))])
    groups = [((G >= 3) & (T >= 4000), 24), ((G < 3) & (T >= 4000), 24), (T < 4000, 16), (widen > 0, 16)]
    chosen = []
    for mask, k in groups:
        pool = np.setdiff1d(np.where(mask)[0], chosen)
        chosen += rng.choice(pool, min(k, len(pool)), replace=False).tolist()
    pool = np.setdiff1d(np.arange(len(idx)), chosen)
    chosen += rng.choice(pool, 96 - len(chosen), replace=False).tolist()
    return [idx[i] for i in chosen], widen[chosen]


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for i in rows:
            w.writerow(fmt_row(cols, i))


def make_config(base, tag, rows_csv):
    c = json.loads(json.dumps(base))
    c["_comment"] = (f"harps-interp-t02 campaign ({tag}): cno_v2 synthesis settings on interpol_modeles MARCS atmospheres "
                     "(t02 fixed, spherical 1 Msun below logg 3, max_widen 2). Rows prefiltered from the LHS of "
                     "configs/pipeline/config_harps_ml_interp.json. Built by scripts/make_interp_campaign.py.")
    c["grid"]["grid_version"] = f"harps-interp-t02-{tag}"
    c["grid"]["rows_csv"] = rows_csv
    c["turbospectrum"]["executables"]["interpol_exec"] = "interpol_modeles_linux"
    c["turbospectrum"]["model_atmosphere"] = dict(ATMOSPHERE)
    c["turbospectrum"]["paths"]["linelist_files"] = [f.replace(MOLECULES_FULL + "/", MOLECULES_TRIMMED + "/")
                                                     for f in c["turbospectrum"]["paths"]["linelist_files"]]
    c["outputs"]["run_root"] = f"{GADI_ROOT}/{tag}"
    return c


def stage_block(config):
    """Shell lines that copy the config's line lists to $PBS_JOBFS and keep per-row scratch there."""
    tops = sorted({f.split("/", 1)[0] for f in config["turbospectrum"]["paths"]["linelist_files"]})
    return (f'mkdir -p "$PBS_JOBFS/linelists"\n'
            f'for x in {" ".join(tops)}; do cp -rp "{LINELIST_DIR}/$x" "$PBS_JOBFS/linelists/" || exit 4; done\n'
            'export TS_LINELIST_ROOT="$PBS_JOBFS/linelists" SCRATCH_TMP_ROOT="$PBS_JOBFS/ai" '
            'TMP_OUTPUT_DIR="$RUN_ROOT/tmp/out_$PBS_JOBID"\n')


TRIM_BLOCK = f"""T={LINELIST_DIR}/{MOLECULES_TRIMMED}
if [[ ! -d "$T" ]]; then
  W="$T.tmp_$PBS_JOBID"
  "$PYTHON_BIN" "${{PBS_O_WORKDIR}}/scripts/trim_bsyn_linelist.py" {LINELIST_DIR}/{MOLECULES_FULL} "$W" --lo 3760 --hi 6930 || exit 3
  (cd "$W" && sha256sum -c --quiet {MOLECULE_SHA256}) || {{ echo "trimmed line lists do not match the reference checksums"; exit 3; }}
  mv -n "$W" "$T"
fi
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_zarr")
    ap.add_argument("base_config")
    ap.add_argument("out_dir")
    ap.add_argument("--rows", type=int, default=300000)
    ap.add_argument("--rows-per-subjob", type=int, default=2500)
    ap.add_argument("--ncpus", type=int, default=104)
    ap.add_argument("--queue", default="rsaa")
    ap.add_argument("--mem", default="480GB")
    ap.add_argument("--jobfs", default="200GB")
    ap.add_argument("--test-ncpus", type=int, default=48)
    ap.add_argument("--test-tag", default="test")
    ap.add_argument("--walltime", default="10:00:00")
    ap.add_argument("--atmosphere-path", required=True, help="local MARCS store (identical to Gadi's) for the test-row picks")
    a = ap.parse_args()

    rng = np.random.default_rng(20261027)
    cols = load_grid(a.grid_zarr)
    n_all = len(cols["teff"])
    if n_all < a.rows:
        raise SystemExit(f"grid has {n_all} rows < requested {a.rows}")
    order = rng.permutation(n_all)[: a.rows]  # seeded random subset, shuffled so every job samples the whole box
    base = json.load(open(a.base_config))
    for sub in ("rows", "configs", "pbs"):
        os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)

    lib = RT.ModelInterpolator(SimpleNamespace(model_atmosphere_path=a.atmosphere_path))
    test_rows, test_widen = pick_test_rows(cols, order[:20000], lib, rng)
    write_csv(os.path.join(a.out_dir, "rows", "test_rows.csv"), cols, test_rows)
    t = a.test_tag
    test_cfg = make_config(base, t, f"{GADI_ROOT}/rows/test_rows.csv")
    json.dump(test_cfg, open(os.path.join(a.out_dir, "configs", f"config_harps_interp_t02_{t}.json"), "w"), indent=1)

    per_array = a.rows_per_subjob * 10
    n_arrays = -(-a.rows // per_array)
    jobs = []
    for j in range(n_arrays):
        tag = f"j{j:03d}"
        rows = order[j * per_array:(j + 1) * per_array]
        write_csv(os.path.join(a.out_dir, "rows", f"rows_{tag}.csv"), cols, rows)
        cfg = make_config(base, tag, f"{GADI_ROOT}/rows/rows_{tag}.csv")
        json.dump(cfg, open(os.path.join(a.out_dir, "configs", f"config_harps_interp_t02_{tag}.json"), "w"), indent=1)
        env = (f"export CONFIG_PATH=configs/pipeline/config_harps_interp_t02_{tag}.json PYTHON_BIN={PYTHON_BIN} "
               f"RUN_ROOT={GADI_ROOT}/{tag}\n")
        with open(os.path.join(a.out_dir, "pbs", f"array_{tag}.pbs"), "w") as fh:
            fh.write(PBS_HEAD.format(queue=a.queue, name=f"ai-{tag}", ncpus=a.ncpus, mem=a.mem, jobfs=a.jobfs, walltime=a.walltime))
            fh.write("#PBS -J 0-9\n")
            fh.write(env + stage_block(cfg))
            fh.write(f"export WORKERS={a.ncpus} SHARD_COUNT=10\n")
            fh.write('exec bash "${PBS_O_WORKDIR}/turbospectrum_pipeline_example.pbs"\n')
        with open(os.path.join(a.out_dir, "pbs", f"finalize_{tag}.pbs"), "w") as fh:
            fh.write(PBS_HEAD.format(queue=a.queue, name=f"ai-fin-{tag}", ncpus=8, mem="64GB", jobfs="50GB", walltime="03:00:00"))
            fh.write(env + "export WORKERS=8 FINALIZE_ONLY=1 SHARD_COUNT=10\n")
            fh.write('exec bash "${PBS_O_WORKDIR}/turbospectrum_pipeline_example.pbs"\n')
        jobs.append((tag, len(rows)))

    with open(os.path.join(a.out_dir, "pbs", f"{t}.pbs"), "w") as fh:
        fh.write(PBS_HEAD.format(queue=a.queue, name=f"ai-{t}", ncpus=a.test_ncpus, mem="190GB", jobfs=a.jobfs, walltime="03:00:00"))
        fh.write(f"export CONFIG_PATH=configs/pipeline/config_harps_interp_t02_{t}.json PYTHON_BIN={PYTHON_BIN} "
                 f"RUN_ROOT={GADI_ROOT}/{t} WORKERS={a.test_ncpus} CLEANUP_SHARDS=0\n")
        fh.write(TRIM_BLOCK + stage_block(test_cfg))
        fh.write('bash "${PBS_O_WORKDIR}/turbospectrum_pipeline_example.pbs"; rc=$?\n')
        fh.write('echo "pipeline rc=$rc"\n')
        fh.write(f'"$PYTHON_BIN" "${{PBS_O_WORKDIR}}/scripts/check_interp_atmosphere_run.py" {GADI_ROOT}/{t} --expected-rows 96; crc=$?\n')
        fh.write('exit $(( rc != 0 ? rc : crc ))\n')

    with open(os.path.join(a.out_dir, "pbs", "submit_fleet.sh"), "w") as fh:
        fh.write("#!/bin/bash\n# Submit every campaign array up front (10 subjobs each, NCI limit), each followed by its own\n"
                 "# finalize (afterany). Optional arg: a job id every array waits on (afterok). No polling.\n"
                 "set -euo pipefail\nPATH=$PATH:/opt/pbs/default/bin\n"
                 f"cd {TS_ROOT}\ndep=${{1:-}}\n")
        fh.write(f"for tag in {' '.join(t for t, _ in jobs)}; do\n")
        fh.write('  args=(); [[ -n "$dep" ]] && args+=(-W depend=afterok:$dep)\n')
        fh.write('  A=$(qsub "${args[@]}" pbs_interp_t02/array_$tag.pbs)\n')
        fh.write('  F=$(qsub -W depend=afterany:$A pbs_interp_t02/finalize_$tag.pbs)\n')
        fh.write(f'  echo "$A array_$tag $(date)" >> {GADI_ROOT}/jobids.txt; echo "$F finalize_$tag $(date)" >> {GADI_ROOT}/jobids.txt\n')
        fh.write('  echo "$tag array $A finalize $F"\ndone\n')

    summary = dict(grid_rows=n_all, campaign_rows=int(a.rows), arrays=n_arrays, rows_per_subjob=a.rows_per_subjob,
                   ncpus=a.ncpus, queue=a.queue, test_rows=len(test_rows),
                   test_widen_counts={str(k): int(v) for k, v in zip(*np.unique(test_widen, return_counts=True))})
    json.dump(summary, open(os.path.join(a.out_dir, "campaign.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
