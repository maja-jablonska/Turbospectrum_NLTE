"""Tests for shard layout resolution: row counting, layout computation."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from scripts.zarr_compat import create_array as _zc_create_array, write_string_array as _zc_write_strings  # noqa: E402
import pytest
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from resolve_shard_layout import _infer_row_count, _positive_int


# -- _positive_int -------------------------------------------------------------


def test_positive_int_valid():
    assert _positive_int("10", "test") == 10
    assert _positive_int("1", "test") == 1


def test_positive_int_none():
    assert _positive_int(None, "test") is None
    assert _positive_int("", "test") is None


def test_positive_int_rejects_zero():
    with pytest.raises(ValueError, match="positive integer"):
        _positive_int("0", "test")


def test_positive_int_rejects_negative():
    with pytest.raises(ValueError, match="positive integer"):
        _positive_int("-5", "test")


def test_positive_int_rejects_non_numeric():
    with pytest.raises(ValueError, match="positive integer"):
        _positive_int("abc", "test")


# -- _infer_row_count ----------------------------------------------------------


def test_infer_row_count_from_flux():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "grid.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        _zc_create_array(root, "flux", data=np.zeros((42, 100), dtype=np.float32))
        count = _infer_row_count(zarr_path, ("flux", "params"))
        assert count == 42


def test_infer_row_count_from_params():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "grid.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        _zc_create_array(root, "params", data=np.zeros((15, 5), dtype=np.float32))
        count = _infer_row_count(zarr_path, ("flux", "params"))
        assert count == 15


def test_infer_row_count_from_teff():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "grid.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        _zc_create_array(root, "teff", data=np.array([5000, 5500, 6000], dtype=np.float32))
        count = _infer_row_count(zarr_path, ("teff",))
        assert count == 3


def test_infer_row_count_key_precedence():
    """First matching key wins."""
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "grid.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        _zc_create_array(root, "flux", data=np.zeros((10, 50), dtype=np.float32))
        _zc_create_array(root, "params", data=np.zeros((10, 3), dtype=np.float32))
        _zc_create_array(root, "teff", data=np.array([5000] * 10, dtype=np.float32))
        count = _infer_row_count(zarr_path, ("flux", "params", "teff"))
        assert count == 10


def test_infer_row_count_no_matching_key():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "grid.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        _zc_create_array(root, "unrelated", data=np.zeros((5, 3), dtype=np.float32))
        with pytest.raises(RuntimeError, match="Could not infer row count"):
            _infer_row_count(zarr_path, ("flux", "params"))


# -- Layout resolution via subprocess ------------------------------------------


def test_shard_layout_from_shard_count(tmp_path):
    """shard_count given, rows_per_shard computed."""
    import subprocess

    zarr_path = str(tmp_path / "grid.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    _zc_create_array(root, "flux", data=np.zeros((100, 10), dtype=np.float32))

    result = subprocess.run(
        [sys.executable, "-m", "resolve_shard_layout",
         "--zarr-path", zarr_path, "--shard-count", "10"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", "scripts"),
    )
    assert result.returncode == 0
    assert "ROW_COUNT=100" in result.stdout
    assert "SHARD_COUNT=10" in result.stdout
    assert "ROWS_PER_SHARD=10" in result.stdout
    assert "ARRAY_RANGE=0-9" in result.stdout


def test_shard_layout_from_rows_per_shard(tmp_path):
    """rows_per_shard given, shard_count computed."""
    import subprocess

    zarr_path = str(tmp_path / "grid.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    _zc_create_array(root, "flux", data=np.zeros((100, 10), dtype=np.float32))

    result = subprocess.run(
        [sys.executable, "-m", "resolve_shard_layout",
         "--zarr-path", zarr_path, "--rows-per-shard", "30"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", "scripts"),
    )
    assert result.returncode == 0
    assert "ROW_COUNT=100" in result.stdout
    assert "ROWS_PER_SHARD=30" in result.stdout
    assert "SHARD_COUNT=4" in result.stdout  # ceil(100/30)


def test_shard_layout_explicit_row_count(tmp_path):
    """Row count from --row-count instead of Zarr."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "resolve_shard_layout",
         "--row-count", "200", "--shard-count", "5"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", "scripts"),
    )
    assert result.returncode == 0
    assert "ROW_COUNT=200" in result.stdout
    assert "SHARD_COUNT=5" in result.stdout
    assert "ROWS_PER_SHARD=40" in result.stdout


def test_shard_layout_widens_rows_per_shard_when_coverage_is_short(tmp_path):
    """shard_count*rows_per_shard < row_count: shard_count (e.g. a fixed PBS
    array size) wins and rows_per_shard is raised so every row is covered."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "resolve_shard_layout",
         "--row-count", "100", "--shard-count", "2", "--rows-per-shard", "10"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", "scripts"),
    )
    assert result.returncode == 0, result.stderr
    assert "SHARD_COUNT=2" in result.stdout
    assert "ROWS_PER_SHARD=50" in result.stdout
    assert "Auto-adjusting rows_per_shard to 50" in result.stderr
