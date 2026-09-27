"""Failure-policy tests for turbospectrum_pipeline_example.pbs (no Turbospectrum needed)."""
import json
import os
import subprocess
import sys

import numpy as np
import zarr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PBS_SCRIPT = os.path.join(REPO_ROOT, "turbospectrum_pipeline_example.pbs")
SYNTHESIS_CONFIG = os.path.join(REPO_ROOT, "configs", "synthesis", "config_sample_comprehensive.json")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from pbs_array_range import array_range, parse_range  # noqa: E402
from zarr_compat import create_array, write_string_array  # noqa: E402


def _config(tmp_path, runtime):
    run_root = tmp_path / "run"
    grids = run_root / "outputs" / "grids"
    cfg = {
        "grid": {
            "grid_version": "test-policy",
            "axes": {"teff": "5000,5250", "logg": "4.0,4.5", "feh": "-0.5,0.0", "turbvel": "01"},
            "synthesis": {"lam_min": 5000.0, "lam_max": 5000.2, "lam_step": 0.1, "output_mode": "Flux",
                          "mode": "1D", "calculation_mode": "LTE"},
            "limits": {"max_rows": 128},
        },
        "turbospectrum": {"config": SYNTHESIS_CONFIG},
        "outputs": {
            "run_root": str(run_root),
            "grid_csv": str(grids / "grid.csv"),
            "grid_zarr": str(grids / "grid.zarr"),
            "grid_index_parquet": str(grids / "index.parquet"),
            "spectra_zarr": str(run_root / "outputs" / "zarr" / "spectra.zarr"),
            "spectra_shard_template": str(run_root / "outputs" / "shards" / "spectra_shard_{shard_index}.zarr"),
        },
        "runtime": {"workers": 1, **runtime},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


def _run(config_path, run_root, **env_extra):
    env = os.environ.copy()
    env.pop("PBS_ARRAY_INDEX", None)
    env.pop("PBS_ARRAY_ID", None)
    env.update({"PBS_O_WORKDIR": REPO_ROOT, "PBS_JOBID": "999.test", "CONFIG_PATH": str(config_path),
                "PYTHON_BIN": sys.executable, "RUN_ROOT": str(run_root)})
    env.update(env_extra)
    return subprocess.run(["bash", PBS_SCRIPT], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def test_parse_range():
    assert parse_range("0-99") == (0, 99, 1)
    assert parse_range("0-18:2") == (0, 18, 2)
    assert array_range({"PBS_JOBID": "123[0-9].gadi-pbs"}) == (0, 9, 1)
    assert array_range({"PBS_JOBID": "123[5].gadi-pbs"}) is None


def test_array_task_without_resolvable_shard_count_fails(tmp_path):
    config_path, cfg = _config(tmp_path, {})
    proc = _run(config_path, cfg["outputs"]["run_root"], PBS_ARRAY_INDEX="0", PBS_JOBID="123[0].gadi-pbs")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "pass SHARD_COUNT explicitly" in proc.stderr


def test_array_range_must_start_at_zero(tmp_path):
    config_path, cfg = _config(tmp_path, {})
    # Parent-style id carries the range (as when PBS_JOBID holds it).
    proc = _run(config_path, cfg["outputs"]["run_root"], PBS_ARRAY_INDEX="1", PBS_JOBID="123[1-4].gadi-pbs")
    assert proc.returncode == 1
    assert "must be 0-(N-1)" in proc.stderr


def _write_shard(path, gidx, failed=()):
    root = zarr.open_group(str(path), mode="w")
    flux = np.ones((len(gidx), 3), dtype=np.float32)
    status = ["success"] * len(gidx)
    for i in failed:
        flux[i] = np.nan
        status[i] = "error"
    create_array(root, "wavelength", data=np.asarray([5000.0, 5000.1, 5000.2], dtype=np.float32))
    create_array(root, "global_index", data=np.asarray(gidx, dtype=np.int64))
    create_array(root, "flux", data=flux)
    write_string_array(root, "status", status)
    write_string_array(root, "message", ["bsyn died" if i in failed else "" for i in range(len(gidx))])


def _prepare_grid(config_path, run_root):
    proc = _run(config_path, run_root, SKIP_SYNTHESIS="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_finalize_partial_merge_exits_3_and_keeps_shards(tmp_path):
    config_path, cfg = _config(tmp_path, {"shard_count": 2})
    run_root = cfg["outputs"]["run_root"]
    _prepare_grid(config_path, run_root)
    shard_dir = os.path.join(run_root, "outputs", "shards")
    os.makedirs(shard_dir, exist_ok=True)
    _write_shard(os.path.join(shard_dir, "spectra_shard_0.zarr"), [0, 2, 4, 6])  # shard 1 missing

    proc = _run(config_path, run_root, FINALIZE_ONLY="1", ALLOW_INCOMPLETE_PROVENANCE="1")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert os.path.isdir(os.path.join(shard_dir, "spectra_shard_0.zarr"))
    merged = zarr.open_group(cfg["outputs"]["spectra_zarr"], mode="r")
    assert int(merged.attrs["missing_models"]) == 4


def test_finalize_without_partial_allowed_aborts(tmp_path):
    config_path, cfg = _config(tmp_path, {"shard_count": 2})
    run_root = cfg["outputs"]["run_root"]
    _prepare_grid(config_path, run_root)
    shard_dir = os.path.join(run_root, "outputs", "shards")
    os.makedirs(shard_dir, exist_ok=True)
    _write_shard(os.path.join(shard_dir, "spectra_shard_0.zarr"), [0, 2, 4, 6])

    proc = _run(config_path, run_root, FINALIZE_ONLY="1", ALLOW_INCOMPLETE_PROVENANCE="1", ALLOW_PARTIAL_MERGE="0")
    assert proc.returncode == 1
    assert not os.path.exists(cfg["outputs"]["spectra_zarr"])


def test_finalize_complete_merge_cleans_shards(tmp_path):
    config_path, cfg = _config(tmp_path, {"shard_count": 2})
    run_root = cfg["outputs"]["run_root"]
    _prepare_grid(config_path, run_root)
    shard_dir = os.path.join(run_root, "outputs", "shards")
    os.makedirs(shard_dir, exist_ok=True)
    _write_shard(os.path.join(shard_dir, "spectra_shard_0.zarr"), [0, 2, 4, 6])
    _write_shard(os.path.join(shard_dir, "spectra_shard_1.zarr"), [1, 3, 5, 7])

    proc = _run(config_path, run_root, FINALIZE_ONLY="1", ALLOW_INCOMPLETE_PROVENANCE="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not os.path.exists(shard_dir)
    merged = zarr.open_group(cfg["outputs"]["spectra_zarr"], mode="r")
    assert int(merged.attrs["missing_models"]) == 0


def _two_shards_one_failed_row(tmp_path):
    config_path, cfg = _config(tmp_path, {"shard_count": 2})
    run_root = cfg["outputs"]["run_root"]
    _prepare_grid(config_path, run_root)
    shard_dir = os.path.join(run_root, "outputs", "shards")
    os.makedirs(shard_dir, exist_ok=True)
    _write_shard(os.path.join(shard_dir, "spectra_shard_0.zarr"), [0, 2, 4, 6])
    _write_shard(os.path.join(shard_dir, "spectra_shard_1.zarr"), [1, 3, 5, 7], failed=(1,))
    return config_path, cfg, run_root, shard_dir


def test_finalize_drops_failed_rows_by_default(tmp_path):
    config_path, cfg, run_root, shard_dir = _two_shards_one_failed_row(tmp_path)
    proc = _run(config_path, run_root, FINALIZE_ONLY="1", ALLOW_INCOMPLETE_PROVENANCE="1", ALLOW_PARTIAL_MERGE="0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 failed row(s) dropped" in proc.stderr
    merged = zarr.open_group(cfg["outputs"]["spectra_zarr"], mode="r")
    assert int(merged.attrs["missing_models"]) == 0
    assert int(merged.attrs["dropped_failed_rows"]) == 1
    np.testing.assert_array_equal(merged["dropped_global_index"][:], [3])
    assert merged["flux"].shape[0] == 7
    assert not os.path.exists(shard_dir)


def test_drop_failed_rows_0_rejects_failed_shard(tmp_path):
    config_path, cfg, run_root, shard_dir = _two_shards_one_failed_row(tmp_path)
    proc = _run(config_path, run_root, FINALIZE_ONLY="1", ALLOW_INCOMPLETE_PROVENANCE="1", DROP_FAILED_ROWS="0")
    assert proc.returncode == 3, proc.stdout + proc.stderr  # partial: shard 1 skipped
    merged = zarr.open_group(cfg["outputs"]["spectra_zarr"], mode="r")
    assert int(merged.attrs["missing_models"]) == 4
    assert os.path.isdir(shard_dir)
