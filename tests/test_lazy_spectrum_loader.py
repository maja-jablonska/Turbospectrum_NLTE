"""Tests for lazy spectrum lookup from Zarr stores."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from lazy_spectrum_loader import (
    LazySpectrumLoader,
    SpectrumMatch,
    _decode_strings,
    _parse_param_assignments,
    load_spectrum_for_params,
)


def _create_test_zarr(tmp_dir, n_rows=5, n_lambda=10, n_params=3):
    """Build a minimal Zarr store with params/flux/wavelength/param_names."""
    zarr_path = os.path.join(tmp_dir, "spectra.zarr")
    root = zarr.open_group(zarr_path, mode="w")

    param_names = ["teff", "logg", "feh"][:n_params]
    params = np.zeros((n_rows, n_params), dtype=np.float32)
    for i in range(n_rows):
        params[i, 0] = 4000 + i * 500  # teff
        if n_params > 1:
            params[i, 1] = 3.0 + i * 0.5  # logg
        if n_params > 2:
            params[i, 2] = -2.0 + i * 0.5  # feh
    root.array("params", params)

    flux = np.random.default_rng(42).random((n_rows, n_lambda)).astype(np.float32)
    root.array("flux", flux)

    wavelength = np.linspace(5000, 5100, n_lambda, dtype=np.float32)
    root.array("wavelength", wavelength)

    root.array("param_names", np.array(param_names, dtype=object), dtype=object,
               object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, "VLenUTF8") else None)

    return zarr_path, params, flux, wavelength, param_names


def _create_test_zarr_simple(tmp_dir, params_data, flux_data, param_names_list):
    """Build a Zarr store from explicit data arrays."""
    zarr_path = os.path.join(tmp_dir, "spectra.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    root.array("params", np.array(params_data, dtype=np.float32))
    root.array("flux", np.array(flux_data, dtype=np.float32))
    root.array("wavelength", np.linspace(5000, 5100, flux_data.shape[1], dtype=np.float32))
    root.array("param_names", np.array(param_names_list, dtype=object), dtype=object,
               object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, "VLenUTF8") else None)
    return zarr_path


# -- _parse_param_assignments -------------------------------------------------


def test_parse_param_assignments_valid():
    result = _parse_param_assignments(["teff=5000", "logg=4.5", "feh=-1.0"])
    assert result == {"teff": 5000.0, "logg": 4.5, "feh": -1.0}


def test_parse_param_assignments_rejects_missing_equals():
    with pytest.raises(ValueError, match="expected name=value"):
        _parse_param_assignments(["teff5000"])


def test_parse_param_assignments_rejects_empty():
    with pytest.raises(ValueError, match="No parameters"):
        _parse_param_assignments([])


def test_parse_param_assignments_rejects_bad_value():
    with pytest.raises(ValueError, match="Invalid value"):
        _parse_param_assignments(["teff=abc"])


def test_parse_param_assignments_rejects_missing_name():
    with pytest.raises(ValueError, match="missing name"):
        _parse_param_assignments(["=5000"])


# -- _decode_strings ----------------------------------------------------------


def test_decode_strings_from_strings():
    assert _decode_strings(["teff", "logg"]) == ("teff", "logg")


def test_decode_strings_from_bytes():
    assert _decode_strings([b"teff", b"logg"]) == ("teff", "logg")


def test_decode_strings_from_numpy():
    arr = np.array(["a", "b", "c"])
    assert _decode_strings(arr) == ("a", "b", "c")


# -- LazySpectrumLoader -------------------------------------------------------


def test_loader_init_and_row_count():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, params, flux, wl, pnames = _create_test_zarr(tmp, n_rows=5)
        loader = LazySpectrumLoader(zarr_path)
        assert loader.row_count == 5
        assert loader.param_names == tuple(pnames)


def test_loader_find_row_index_exact():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, params, _, _, _ = _create_test_zarr(tmp, n_rows=5)
        loader = LazySpectrumLoader(zarr_path)
        # Row 2: teff=5000, logg=4.0, feh=-1.0
        idx = loader.find_row_index({"teff": 5000.0, "logg": 4.0, "feh": -1.0})
        assert idx == 2


def test_loader_find_row_index_not_found():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _create_test_zarr(tmp, n_rows=5)
        loader = LazySpectrumLoader(zarr_path)
        with pytest.raises(KeyError, match="No matching spectrum"):
            loader.find_row_index({"teff": 9999.0, "logg": 9.0, "feh": 9.0})


def test_loader_find_row_index_unknown_param():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _create_test_zarr(tmp, n_rows=5)
        loader = LazySpectrumLoader(zarr_path)
        with pytest.raises(KeyError, match="Unknown parameter"):
            loader.find_row_index({"nonexistent": 1.0})


def test_loader_find_row_index_missing_param():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _create_test_zarr(tmp, n_rows=5)
        loader = LazySpectrumLoader(zarr_path)
        with pytest.raises(ValueError, match="Missing parameter"):
            loader.find_row_index({"teff": 5000.0})


def test_loader_find_row_index_partial():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _create_test_zarr(tmp, n_rows=5)
        loader = LazySpectrumLoader(zarr_path)
        idx = loader.find_row_index({"teff": 5000.0}, require_all_params=False)
        assert idx == 2


def test_loader_fetch_spectrum():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, params, flux, wl, _ = _create_test_zarr(tmp, n_rows=5, n_lambda=10)
        loader = LazySpectrumLoader(zarr_path)
        match = loader.fetch_spectrum({"teff": 4500.0, "logg": 3.5, "feh": -1.5})
        assert isinstance(match, SpectrumMatch)
        assert match.row_index == 1
        np.testing.assert_array_almost_equal(match.spectrum, flux[1])
        assert match.wavelength is not None
        np.testing.assert_array_almost_equal(match.wavelength, wl)


def test_loader_fetch_spectrum_no_wavelength():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _create_test_zarr(tmp, n_rows=3)
        loader = LazySpectrumLoader(zarr_path)
        match = loader.fetch_spectrum(
            {"teff": 4000.0, "logg": 3.0, "feh": -2.0},
            include_wavelength=False,
        )
        assert match.wavelength is None


def test_loader_scan_chunk_rows():
    """Verify small chunk sizes still find the correct row."""
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _create_test_zarr(tmp, n_rows=10)
        loader = LazySpectrumLoader(zarr_path, scan_chunk_rows=2)
        idx = loader.find_row_index({"teff": 6000.0, "logg": 5.0, "feh": 0.0})
        assert idx == 4


# -- load_spectrum_for_params (convenience) ------------------------------------


def test_load_spectrum_for_params_convenience():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, flux, _, _ = _create_test_zarr(tmp, n_rows=3, n_lambda=8)
        match = load_spectrum_for_params(
            zarr_path,
            {"teff": 4000.0, "logg": 3.0, "feh": -2.0},
        )
        assert match.row_index == 0
        np.testing.assert_array_almost_equal(match.spectrum, flux[0])


# -- Error conditions ----------------------------------------------------------


def test_loader_rejects_missing_arrays():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "empty.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        root.array("flux", np.zeros((2, 5)))
        with pytest.raises(KeyError, match="Missing required arrays"):
            LazySpectrumLoader(zarr_path)


def test_loader_rejects_row_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "bad.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        root.array("params", np.zeros((3, 2), dtype=np.float32))
        root.array("flux", np.zeros((5, 10), dtype=np.float32))
        root.array("param_names", np.array(["a", "b"], dtype=object), dtype=object,
                    object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, "VLenUTF8") else None)
        with pytest.raises(ValueError, match="Row mismatch"):
            LazySpectrumLoader(zarr_path)


def test_loader_rejects_invalid_scan_chunk():
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path, _, _, _, _ = _create_test_zarr(tmp, n_rows=3)
        with pytest.raises(ValueError, match="scan_chunk_rows must be positive"):
            LazySpectrumLoader(zarr_path, scan_chunk_rows=0)
