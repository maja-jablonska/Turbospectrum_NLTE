"""Tests for JAX spectra dataloader (non-JAX-dependent parts).

The JAX import is mocked so these tests run without a JAX installation.
We test: split logic, dataset wrapper, standardization, and key inference.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from unittest import mock

import numpy as np
import pytest
import zarr


# -- Mock JAX before importing the module -------------------------------------
# The module does `import jax` at the top level, so we must mock it first.

_jax_mock = types.ModuleType("jax")
_jax_mock.Array = np.ndarray
_jax_mock.Device = type(None)
_jax_mock.device_put = lambda x, d=None: x

_jnp_mock = types.ModuleType("jax.numpy")
_jnp_mock.asarray = np.asarray
_jax_mock.numpy = _jnp_mock

sys.modules.setdefault("jax", _jax_mock)
sys.modules.setdefault("jax.numpy", _jnp_mock)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from jax_spectra_dataloader import (
    SpectraZarrDataset,
    StandardizationStats,
    _compute_standardization,
    _infer_io_keys,
    _split_indices,
)


def _make_dataset_zarr(tmp_dir, n_rows=20, n_lambda=10, n_params=3):
    """Build a minimal Zarr store for SpectraZarrDataset."""
    zarr_path = os.path.join(tmp_dir, "spectra.zarr")
    root = zarr.open_group(zarr_path, mode="w")

    rng = np.random.default_rng(42)
    params = rng.random((n_rows, n_params)).astype(np.float32)
    flux = rng.random((n_rows, n_lambda)).astype(np.float32) + 0.5
    wavelength = np.linspace(5000, 5100, n_lambda, dtype=np.float32)
    param_names = [f"p{i}" for i in range(n_params)]

    root.array("params", params)
    root.array("flux", flux)
    root.array("wavelength", wavelength)
    root.array("param_names", np.array(param_names, dtype=object), dtype=object,
               object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, "VLenUTF8") else None)

    return zarr_path, params, flux, wavelength, param_names


# -- _split_indices ------------------------------------------------------------


def test_split_indices_sizes():
    splits = _split_indices(100, train_fraction=0.8, val_fraction=0.1, seed=0)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_split_indices_no_overlap():
    splits = _split_indices(100, train_fraction=0.7, val_fraction=0.15, seed=42)
    all_ids = np.concatenate([splits["train"], splits["val"], splits["test"]])
    assert len(np.unique(all_ids)) == 100


def test_split_indices_covers_all_rows():
    splits = _split_indices(50, train_fraction=0.6, val_fraction=0.2, seed=7)
    all_ids = sorted(np.concatenate([splits["train"], splits["val"], splits["test"]]).tolist())
    assert all_ids == list(range(50))


def test_split_indices_deterministic():
    a = _split_indices(100, train_fraction=0.8, val_fraction=0.1, seed=99)
    b = _split_indices(100, train_fraction=0.8, val_fraction=0.1, seed=99)
    np.testing.assert_array_equal(a["train"], b["train"])


def test_split_indices_rejects_invalid_fractions():
    with pytest.raises(ValueError, match="train_fraction"):
        _split_indices(100, train_fraction=0.0, val_fraction=0.1, seed=0)
    with pytest.raises(ValueError, match="train_fraction \\+ val_fraction"):
        _split_indices(100, train_fraction=0.9, val_fraction=0.2, seed=0)


def test_split_indices_zero_val():
    splits = _split_indices(100, train_fraction=0.8, val_fraction=0.0, seed=0)
    assert len(splits["val"]) == 0
    assert len(splits["train"]) + len(splits["test"]) == 100


# -- SpectraZarrDataset --------------------------------------------------------


def test_dataset_init():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, pnames = _make_dataset_zarr(tmp, n_rows=20)
        ds = SpectraZarrDataset(zarr_path)
        assert ds.row_count == 20
        assert ds.param_names == tuple(pnames)
        assert ds.has_array("flux")
        assert ds.has_array("params")


def test_dataset_shape():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _make_dataset_zarr(tmp, n_rows=15, n_lambda=8)
        ds = SpectraZarrDataset(zarr_path)
        assert ds.shape("flux") == (15, 8)
        assert ds.shape("params") == (15, 3)


def test_dataset_get_rows():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, flux, _, _ = _make_dataset_zarr(tmp, n_rows=10, n_lambda=5)
        ds = SpectraZarrDataset(zarr_path)
        rows = ds.get_rows("flux", [0, 3, 7])
        assert rows.shape == (3, 5)
        np.testing.assert_array_almost_equal(rows[0], flux[0])
        np.testing.assert_array_almost_equal(rows[1], flux[3])


def test_dataset_get_rows_with_column_ids():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, params, _, _, _ = _make_dataset_zarr(tmp, n_rows=10, n_params=4)
        ds = SpectraZarrDataset(zarr_path)
        rows = ds.get_rows("params", [0, 1], column_ids=[1, 3])
        assert rows.shape == (2, 2)
        np.testing.assert_array_almost_equal(rows[0, 0], params[0, 1])
        np.testing.assert_array_almost_equal(rows[0, 1], params[0, 3])


def test_dataset_resolve_param_columns():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _make_dataset_zarr(tmp, n_params=3)
        ds = SpectraZarrDataset(zarr_path)
        cols = ds.resolve_param_columns(["p0", "p2"])
        np.testing.assert_array_equal(cols, [0, 2])


def test_dataset_resolve_param_columns_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _make_dataset_zarr(tmp)
        ds = SpectraZarrDataset(zarr_path)
        with pytest.raises(KeyError, match="Unknown parameter"):
            ds.resolve_param_columns(["nonexistent"])


def test_dataset_rejects_empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "empty.zarr")
        zarr.open_group(zarr_path, mode="w")
        with pytest.raises(ValueError, match="No arrays"):
            SpectraZarrDataset(zarr_path)


# -- _infer_io_keys -----------------------------------------------------------


def test_infer_io_keys_auto():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _make_dataset_zarr(tmp)
        ds = SpectraZarrDataset(zarr_path)
        inp, tgt = _infer_io_keys(ds, None, "auto")
        assert inp == "params"
        assert tgt == "flux"


def test_infer_io_keys_explicit():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _make_dataset_zarr(tmp)
        ds = SpectraZarrDataset(zarr_path)
        inp, tgt = _infer_io_keys(ds, "flux", None)
        assert inp == "flux"
        assert tgt is None


# -- _compute_standardization --------------------------------------------------


def test_compute_standardization_basic():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, flux, _, _ = _make_dataset_zarr(tmp, n_rows=20, n_lambda=5)
        ds = SpectraZarrDataset(zarr_path)
        row_ids = np.arange(20, dtype=np.int64)
        stats = _compute_standardization(ds, "flux", row_ids, column_ids=None, block_rows=10)
        assert isinstance(stats, StandardizationStats)
        assert stats.mean.shape == (5,)
        assert stats.std.shape == (5,)
        np.testing.assert_allclose(stats.mean, flux.mean(axis=0), atol=1e-5)


def test_compute_standardization_small_blocks():
    """Multiple passes with small blocks should produce the same result."""
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, flux, _, _ = _make_dataset_zarr(tmp, n_rows=20, n_lambda=3)
        ds = SpectraZarrDataset(zarr_path)
        row_ids = np.arange(20, dtype=np.int64)
        stats_1 = _compute_standardization(ds, "flux", row_ids, column_ids=None, block_rows=20)
        stats_5 = _compute_standardization(ds, "flux", row_ids, column_ids=None, block_rows=5)
        np.testing.assert_allclose(stats_1.mean, stats_5.mean, atol=1e-5)
        np.testing.assert_allclose(stats_1.std, stats_5.std, atol=1e-5)
