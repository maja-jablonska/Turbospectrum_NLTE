"""Tests for the shared zarr v2/v3 compatibility module."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from zarr_compat import (
    ZARR_V3,
    compression_kwargs,
    create_array,
    create_root_group,
    open_root_group,
    write_fixed_string_scalar,
    write_string_array,
    write_string_scalar,
    zarr_store,
)


# -- version detection ---------------------------------------------------------


def test_zarr_v3_flag_matches_version():
    major = int(zarr.__version__.split(".")[0])
    assert ZARR_V3 == (major >= 3)


# -- zarr_store ----------------------------------------------------------------


def test_zarr_store_returns_usable_store():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        assert store is not None


# -- create_root_group / open_root_group ---------------------------------------


def test_create_and_open_root_group():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        assert root is not None

        root2 = open_root_group(path, mode="r")
        assert root2 is not None


# -- compression_kwargs --------------------------------------------------------


def test_compression_kwargs_empty():
    assert compression_kwargs(None) == {}
    assert compression_kwargs({}) == {}


def test_compression_kwargs_returns_dict():
    kw = compression_kwargs({"cname": "zstd", "clevel": 3, "shuffle": True})
    assert isinstance(kw, dict)
    # v2 uses "compressor", v3 uses "compressors"
    assert "compressor" in kw or "compressors" in kw


# -- create_array --------------------------------------------------------------


def test_create_array_with_data():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        create_array(root, "values", data=data, chunks=3)

        root2 = open_root_group(path)
        assert "values" in root2
        np.testing.assert_array_equal(root2["values"][:], data)


def test_create_array_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        create_array(root, "empty", shape=(5, 10), dtype=np.float32, chunks=(5, 10))

        root2 = open_root_group(path)
        assert root2["empty"].shape == (5, 10)


def test_create_array_2d():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        create_array(root, "matrix", data=data, chunks=(3, 4))

        root2 = open_root_group(path)
        np.testing.assert_array_equal(root2["matrix"][:], data)


# -- write_string_array --------------------------------------------------------


def test_write_string_array_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        write_string_array(root, "names", ["alpha", "beta", "gamma"])

        root2 = open_root_group(path)
        assert "names" in root2
        values = [str(v) for v in root2["names"][:].tolist()]
        assert values == ["alpha", "beta", "gamma"]


def test_write_string_array_with_compression():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        comp_kw = compression_kwargs({"cname": "zstd", "clevel": 1})
        write_string_array(root, "items", ["a", "b", "c"], compression_kw=comp_kw)

        root2 = open_root_group(path)
        values = [str(v) for v in root2["items"][:].tolist()]
        assert values == ["a", "b", "c"]


# -- write_string_scalar -------------------------------------------------------


def test_write_string_scalar_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        write_string_scalar(root, "version", "v1.0")

        root2 = open_root_group(path)
        assert "version" in root2
        raw = root2["version"][:]
        value = str(raw.item() if hasattr(raw, "item") else raw[0])
        assert "v1.0" in value


# -- write_fixed_string_scalar -------------------------------------------------


def test_write_fixed_string_scalar_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.zarr")
        store = zarr_store(path)
        root = create_root_group(store)
        write_fixed_string_scalar(root, "hash", "abc123def456", min_width=64)

        root2 = open_root_group(path)
        assert "hash" in root2
        raw = root2["hash"][:]
        value = str(raw.item() if hasattr(raw, "item") else raw[0]).strip()
        assert "abc123def456" in value


# -- Integration: full write/read cycle ----------------------------------------


def test_full_zarr_write_read_cycle():
    """Write a complete small dataset and verify all arrays are readable."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spectra.zarr")
        store = zarr_store(path)
        comp_kw = compression_kwargs({"cname": "zstd", "clevel": 3})
        root = create_root_group(store)

        # Numeric arrays
        flux = np.random.default_rng(42).random((5, 10)).astype(np.float32)
        wavelength = np.linspace(5000, 5100, 10, dtype=np.float32)
        params = np.random.default_rng(0).random((5, 3)).astype(np.float32)

        create_array(root, "flux", data=flux, chunks=(5, 10), **comp_kw)
        create_array(root, "wavelength", data=wavelength, chunks=10, **comp_kw)
        create_array(root, "params", data=params, chunks=(5, 3), **comp_kw)

        # String arrays
        write_string_array(root, "param_names", ["teff", "logg", "feh"],
                           compression_kw=comp_kw)
        write_string_scalar(root, "schema_version", "1.0.0", compression_kw=comp_kw)
        write_fixed_string_scalar(root, "physics_hash", "deadbeef" * 8,
                                  min_width=64, compression_kw=comp_kw)

        # Read back
        root2 = open_root_group(path)
        np.testing.assert_array_almost_equal(root2["flux"][:], flux)
        np.testing.assert_array_almost_equal(root2["wavelength"][:], wavelength)
        np.testing.assert_array_almost_equal(root2["params"][:], params)
        pnames = [str(v) for v in root2["param_names"][:].tolist()]
        assert pnames == ["teff", "logg", "feh"]
