import os
import sys

import numpy as np
import pytest
import zarr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import rechunk_zarr  # noqa: E402
from zarr_compat import create_array, write_fixed_string_scalar, write_string_array  # noqa: E402

N, L = 2500, 300


@pytest.fixture()
def merged(tmp_path):
    path = tmp_path / "merged.zarr"
    root = zarr.open_group(str(path), mode="w")
    create_array(root, "wavelength", data=np.linspace(5000, 5030, L))
    create_array(root, "flux", data=np.random.default_rng(0).random((N, L), dtype=np.float32), chunks=(128, L))
    create_array(root, "continuum", data=np.ones((N, L), np.float32), chunks=(128, L))
    create_array(root, "params", data=np.arange(N * 3, dtype=np.float32).reshape(N, 3), chunks=(128, 3))
    create_array(root, "global_index", data=np.arange(N, dtype=np.int64))
    write_string_array(root, "param_names", ["teff", "logg", "feh"])
    write_fixed_string_scalar(root, "physics_hash", "p" * 64, min_width=64)
    root.create_group("provenance").attrs["x"] = 1
    root.attrs.update({"n_models": N, "config_hash": "c" * 64})
    return str(path)


def _check(merged, out):
    src = zarr.open_group(merged, mode="r")
    dst = zarr.open_group(out, mode="r")
    np.testing.assert_array_equal(dst["flux"][...], src["flux"][...])
    np.testing.assert_array_equal(dst["params"][...], src["params"][...])
    np.testing.assert_array_equal(dst["global_index"][...], src["global_index"][...])
    assert dst["flux"].chunks == (rechunk_zarr.ROW_CHUNK, L)
    assert dst.attrs["config_hash"] == "c" * 64
    assert dst.attrs["lineage"][-1]["stage"] == "rechunk"
    assert "provenance" in dst
    assert not os.path.exists(out + ".building")


def test_single_job(merged, tmp_path):
    out = str(tmp_path / "ml.zarr")
    rechunk_zarr.main(["all", "-i", merged, "-o", out, "--tasks", "3"])
    _check(merged, out)


def test_array_tasks_and_finalize_requires_all(merged, tmp_path):
    out = str(tmp_path / "ml.zarr")
    n_tasks = len(rechunk_zarr.task_ranges(N, 3))
    assert n_tasks == 3
    rechunk_zarr.main(["init", "-i", merged, "-o", out, "--tasks", "3"])
    for k in (2, 0):  # out of order, task 1 missing
        rechunk_zarr.main(["task", "-i", merged, "-o", out, "--tasks", "3", "--task-index", str(k)])
    with pytest.raises(SystemExit, match="incomplete"):
        rechunk_zarr.main(["finalize", "-i", merged, "-o", out, "--tasks", "3"])
    assert not os.path.exists(out)
    rechunk_zarr.main(["task", "-i", merged, "-o", out, "--tasks", "3", "--task-index", "1"])
    rechunk_zarr.main(["finalize", "-i", merged, "-o", out, "--tasks", "3"])
    _check(merged, out)


def test_task_ranges_are_chunk_aligned():
    for lo, _hi in rechunk_zarr.task_ranges(100_000, 7):
        assert lo % rechunk_zarr.ROW_CHUNK == 0 and lo % rechunk_zarr.ONE_D_CHUNK == 0
    ranges = rechunk_zarr.task_ranges(100_000, 7)
    assert ranges[0][0] == 0 and ranges[-1][1] == 100_000
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
