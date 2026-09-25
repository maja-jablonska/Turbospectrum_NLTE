import os
import subprocess
import sys

import numpy as np
import zarr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from zarr_compat import create_array, write_string_array  # noqa: E402
from zarr_rowstore import check_shard, expected_shard_indices  # noqa: E402

SCRIPT = os.path.join(REPO_ROOT, "scripts", "shard_status.py")
WL = np.asarray([5000.0, 5000.2], dtype=np.float32)


def _write_shard(path, gidx, *, flux=None, statuses=None, attrs=None):
    gidx = np.asarray(gidx, dtype=np.int64)
    root = zarr.open_group(str(path), mode="w")
    create_array(root, "wavelength", data=WL)
    create_array(root, "global_index", data=gidx)
    create_array(root, "flux", data=np.ones((gidx.size, WL.size), np.float32) if flux is None else np.asarray(flux, np.float32))
    write_string_array(root, "status", statuses or ["success"] * gidx.size)
    write_string_array(root, "message", [""] * gidx.size)
    if attrs:
        root.attrs.update(attrs)


def test_expected_shard_indices_is_strided():
    np.testing.assert_array_equal(expected_shard_indices(1, 3, 8), [1, 4, 7])


def test_check_shard_accepts_valid(tmp_path):
    p = tmp_path / "s.zarr"
    _write_shard(p, [1, 4, 7])
    assert check_shard(str(p), expected_global_indices=expected_shard_indices(1, 3, 8)) is None


def test_check_shard_reasons(tmp_path):
    p = tmp_path / "s.zarr"
    _write_shard(p, [1, 4])
    assert check_shard(str(p), expected_global_indices=np.array([1, 4, 7])) == "layout_mismatch"

    _write_shard(p, [0, 1], flux=[[1, 1], [np.nan, 1]])
    assert check_shard(str(p)) == "non-finite flux"

    _write_shard(p, [0, 1], flux=[[1, 1], [np.nan, np.nan]], statuses=["success", "failed"])
    assert check_shard(str(p)).startswith("non-success statuses")
    assert check_shard(str(p), drop_failed_rows=True) is None

    _write_shard(p, [0], attrs={"config_hash": "abc"})
    assert check_shard(str(p), expected_attrs={"config_hash": "abc"}) is None
    assert check_shard(str(p), expected_attrs={"config_hash": "xyz"}) == "config_hash_mismatch"
    # Shards written before an attr existed are accepted (legacy runs keep resuming).
    assert check_shard(str(p), expected_attrs={"input_hash": "new"}) is None
    assert check_shard(str(tmp_path / "nope.zarr")).startswith("unreadable")


def test_cli_reports_missing_and_prunes_stale(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # 8 rows, 3 shards: shard 0 valid, shard 1 from an old 2-shard layout, shard 2 absent,
    # shard 5 out of range.
    _write_shard(shard_dir / "spectra_shard_0.zarr", [0, 3, 6])
    _write_shard(shard_dir / "spectra_shard_1.zarr", [1, 3, 5, 7])
    _write_shard(shard_dir / "spectra_shard_5.zarr", [5])
    out = tmp_path / "missing.txt"
    cmd = [sys.executable, SCRIPT, "--shard-dir", str(shard_dir), "--shard-count", "3",
           "--row-count", "8", "--missing-out", str(out)]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out.read_text().split() == ["1", "2"]
    assert "layout_mismatch" in res.stderr
    assert (shard_dir / "spectra_shard_1.zarr").exists()

    subprocess.run(cmd + ["--prune-stale"], capture_output=True, text=True, check=True)
    assert not (shard_dir / "spectra_shard_1.zarr").exists()
    assert not (shard_dir / "spectra_shard_5.zarr").exists()
    assert (shard_dir / "spectra_shard_0.zarr").exists()
