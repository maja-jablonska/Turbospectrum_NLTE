"""Tests for spectrum output: continuum reconstruction, flux extraction, metadata."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from spectrum_output import (
    DEFAULT_FLUX_DEFINITION,
    DEFAULT_FLUX_UNIT,
    extract_flux_and_continuum,
    infer_flux_metadata,
    reconstruct_continuum,
)


# -- reconstruct_continuum -----------------------------------------------------


def test_reconstruct_continuum_basic():
    """cont = abs / norm."""
    abs_flux = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    norm_flux = np.array([0.5, 0.8, 1.0], dtype=np.float32)
    cont = reconstruct_continuum(abs_flux, norm_flux)
    np.testing.assert_allclose(cont, [20.0, 25.0, 30.0], rtol=1e-5)


def test_reconstruct_continuum_zero_norm():
    """Zero normalised flux should produce NaN (not crash)."""
    abs_flux = np.array([10.0, 20.0], dtype=np.float32)
    norm_flux = np.array([0.0, 1.0], dtype=np.float32)
    cont = reconstruct_continuum(abs_flux, norm_flux)
    assert np.isnan(cont[0])
    np.testing.assert_allclose(cont[1], 20.0, rtol=1e-5)


def test_reconstruct_continuum_nan_input():
    abs_flux = np.array([np.nan, 20.0], dtype=np.float32)
    norm_flux = np.array([0.5, 0.8], dtype=np.float32)
    cont = reconstruct_continuum(abs_flux, norm_flux)
    assert np.isnan(cont[0])
    assert np.isfinite(cont[1])


def test_reconstruct_continuum_all_finite():
    abs_flux = np.random.default_rng(0).random(100).astype(np.float32) + 0.01
    norm_flux = np.random.default_rng(1).random(100).astype(np.float32) + 0.01
    cont = reconstruct_continuum(abs_flux, norm_flux)
    assert np.all(np.isfinite(cont))


# -- extract_flux_and_continuum (Flux mode) ------------------------------------


def test_extract_flux_mode_three_columns():
    """3 columns: wavelength(?), normalised, absolute -> returns (absolute, continuum)."""
    # col0=wavelength, col1=normalised, col2=absolute
    data = np.array([
        [5000.0, 0.9, 4.5],
        [5001.0, 0.8, 4.0],
        [5002.0, 1.0, 5.0],
    ], dtype=np.float32)
    flux, cont = extract_flux_and_continuum(data, is_intensity=False)
    np.testing.assert_allclose(flux, [4.5, 4.0, 5.0], rtol=1e-5)
    np.testing.assert_allclose(cont, [5.0, 5.0, 5.0], rtol=1e-5)


def test_extract_flux_mode_two_columns():
    """2 columns only: returns normalised flux, NaN continuum."""
    data = np.array([[5000.0, 0.9], [5001.0, 0.8]], dtype=np.float32)
    flux, cont = extract_flux_and_continuum(data, is_intensity=False)
    np.testing.assert_allclose(flux, [0.9, 0.8], rtol=1e-5)
    assert np.all(np.isnan(cont))


def test_extract_flux_mode_rejects_1d():
    data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unexpected spectrum table shape"):
        extract_flux_and_continuum(data, is_intensity=False)


def test_extract_flux_mode_rejects_single_column():
    data = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="Unexpected spectrum table shape"):
        extract_flux_and_continuum(data, is_intensity=False)


# -- extract_flux_and_continuum (Intensity mode) -------------------------------


def test_extract_intensity_mode_with_chosen_idx():
    """Intensity mode with chosen_idx selects specific mu columns."""
    # Build a table: wl | norm | abs | i_abs_0 | i_norm_0 | i_abs_1 | i_norm_1
    n = 3
    data = np.zeros((n, 7), dtype=np.float32)
    data[:, 0] = [5000, 5001, 5002]  # wavelength
    data[:, 1] = [0.9, 0.8, 1.0]     # norm
    data[:, 2] = [4.5, 4.0, 5.0]     # abs
    data[:, 3] = [2.0, 2.5, 3.0]     # i_abs_0
    data[:, 4] = [0.4, 0.5, 0.6]     # i_norm_0
    data[:, 5] = [1.0, 1.5, 2.0]     # i_abs_1
    data[:, 6] = [0.2, 0.3, 0.4]     # i_norm_1
    chosen_idx = np.array([0])
    flux, cont = extract_flux_and_continuum(data, is_intensity=True, chosen_idx=chosen_idx)
    np.testing.assert_allclose(flux, [2.0, 2.5, 3.0], rtol=1e-5)
    expected_cont = np.array([2.0 / 0.4, 2.5 / 0.5, 3.0 / 0.6], dtype=np.float32)
    np.testing.assert_allclose(cont, expected_cont, rtol=1e-4)


def test_extract_intensity_mode_without_chosen_idx_falls_back_to_flux():
    """Without chosen_idx, intensity mode falls back to flux-mode columns."""
    data = np.array([[5000, 0.9, 4.5], [5001, 0.8, 4.0]], dtype=np.float32)
    flux, cont = extract_flux_and_continuum(data, is_intensity=True, chosen_idx=None)
    np.testing.assert_allclose(flux, [4.5, 4.0], rtol=1e-5)


def test_extract_intensity_mean_reduction():
    """Mean reduction across multiple mu angles."""
    n = 2
    data = np.zeros((n, 7), dtype=np.float32)
    data[:, 3] = [2.0, 3.0]  # i_abs_0
    data[:, 4] = [0.5, 0.5]  # i_norm_0
    data[:, 5] = [4.0, 5.0]  # i_abs_1
    data[:, 6] = [0.5, 0.5]  # i_norm_1
    chosen_idx = np.array([0, 1])
    flux, cont = extract_flux_and_continuum(
        data, is_intensity=True, chosen_idx=chosen_idx, reduce_mode="mean",
    )
    np.testing.assert_allclose(flux, [3.0, 4.0], rtol=1e-5)


# -- infer_flux_metadata -------------------------------------------------------


def test_infer_flux_metadata_flux_mode():
    defn, unit = infer_flux_metadata(["Flux"])
    assert defn == "unnormalized_flux"
    assert unit == DEFAULT_FLUX_UNIT


def test_infer_flux_metadata_intensity_mode():
    defn, unit = infer_flux_metadata(["Intensity"])
    assert defn == "unnormalized_intensity"
    assert unit == DEFAULT_FLUX_UNIT


def test_infer_flux_metadata_explicit_override():
    defn, unit = infer_flux_metadata(
        ["Flux"],
        explicit_definition="continuum_normalized",
        explicit_unit="erg/cm2",
    )
    assert defn == "continuum_normalized"
    assert unit == "erg/cm2"


def test_infer_flux_metadata_mixed_modes():
    defn, unit = infer_flux_metadata(["Flux", "Intensity"])
    assert defn == "mode_dependent"


def test_infer_flux_metadata_no_modes_inherited():
    defn, unit = infer_flux_metadata(
        None,
        inherited_definition="unnormalized_flux",
        inherited_unit="relative",
    )
    assert defn == "unnormalized_flux"


def test_infer_flux_metadata_no_modes_default():
    defn, unit = infer_flux_metadata(None)
    assert defn == DEFAULT_FLUX_DEFINITION
    assert unit == DEFAULT_FLUX_UNIT
