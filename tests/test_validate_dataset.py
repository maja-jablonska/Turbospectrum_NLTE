"""Tests for dataset validation: shard structure, completeness, and provenance."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from validate_dataset import (
    _contract_fields_from_attrs,
    _missing_contract_fields,
    check_completeness,
    extract_shard_id,
    is_valid_zarr,
    validate_shard,
)


def _make_valid_shard(path, n_rows=3, n_lambda=10, *, with_provenance=True):
    """Create a valid shard Zarr at the given path."""
    root = zarr.open_group(str(path), mode="w")
    flux = np.random.default_rng(42).random((n_rows, n_lambda)).astype(np.float32) + 0.1
    root.array("flux", flux)
    root.array("wavelength", np.linspace(5000, 5100, n_lambda, dtype=np.float32))
    root.array("params", np.random.default_rng(0).random((n_rows, 3)).astype(np.float32))

    if with_provenance:
        root.attrs["config_hash"] = "abc123"
        root.attrs["grid_definition_hash"] = "def456"
        root.attrs["git_commit"] = "aaa111"
        root.attrs["turbospectrum_version"] = "v20.0"
        root.attrs["linelist_identifier"] = "nlte_ges"
        root.attrs["linelist_version"] = "v1.0"
        root.attrs["atmosphere_model_identifier"] = "MARCS"
        root.attrs["synthesis_timestamp"] = "2026-01-01T00:00:00"
        root.attrs["pipeline_version"] = "v2.0"

        prov = root.require_group("provenance")
        prov.array("canonical_config.yaml", np.array("config: true"))
        prov.array("synthesis_config.yaml", np.array("synth: true"))
        prov.array("linelist_manifest.json", np.array(json.dumps({"lines": 1})))
        prov.array("atmosphere_manifest.json", np.array(json.dumps({"atm": 1})))
        prov.array("software_manifest.json", np.array(json.dumps({"sw": 1})))
        prov.array("environment.txt", np.array("python==3.11"))

    return root


# -- extract_shard_id ---------------------------------------------------------


def test_extract_shard_id_valid():
    assert extract_shard_id(Path("spectra_shard_0.zarr")) == 0
    assert extract_shard_id(Path("spectra_shard_42.zarr")) == 42
    assert extract_shard_id(Path("spectra_shard_1000.zarr")) == 1000


def test_extract_shard_id_invalid():
    assert extract_shard_id(Path("not_a_shard.zarr")) is None
    assert extract_shard_id(Path("spectra_shard_.zarr")) is None
    assert extract_shard_id(Path("shard_0.zarr")) is None


# -- is_valid_zarr -------------------------------------------------------------


def test_is_valid_zarr_true():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "valid.zarr"
        zarr.open_group(str(path), mode="w")
        assert is_valid_zarr(path) is True


def test_is_valid_zarr_false():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "invalid.zarr"
        path.mkdir()
        (path / "garbage.txt").write_text("not zarr")
        assert is_valid_zarr(path) is False


# -- _contract_fields_from_attrs -----------------------------------------------


def test_contract_fields_from_attrs():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.zarr"
        _make_valid_shard(path, with_provenance=True)
        root = zarr.open_group(str(path), mode="r")
        fields = _contract_fields_from_attrs(root)
        assert fields["config_hash"] == "abc123"
        assert fields["git_commit"] == "aaa111"


def test_contract_fields_from_attrs_with_alternate_keys():
    """Test fallback attribute names (git_sha, created_utc, etc.)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs["git_sha"] = "bbb222"
        root.attrs["created_utc"] = "2026-01-01"
        fields = _contract_fields_from_attrs(root)
        assert fields["git_commit"] == "bbb222"
        assert fields["synthesis_timestamp"] == "2026-01-01"


# -- _missing_contract_fields -------------------------------------------------


def test_missing_contract_fields_none_missing():
    fields = {
        "config_hash": "a", "grid_definition_hash": "b", "git_commit": "c",
        "turbospectrum_version": "d", "linelist_identifier": "e", "linelist_version": "f",
        "atmosphere_model_identifier": "g", "synthesis_timestamp": "h", "pipeline_version": "i",
    }
    assert _missing_contract_fields(fields) == []


def test_missing_contract_fields_some_missing():
    fields = {"config_hash": "a"}
    missing = _missing_contract_fields(fields)
    assert len(missing) > 0
    assert "git_commit" in missing


# -- check_completeness -------------------------------------------------------


def test_check_completeness_all_present():
    shards = [Path(f"spectra_shard_{i}.zarr") for i in range(5)]
    result = check_completeness(shards, 5)
    assert result["found"] == 5
    assert result["missing"] == []
    assert result["unexpected"] == []
    assert not result["duplicates"]


def test_check_completeness_missing_shards():
    shards = [Path(f"spectra_shard_{i}.zarr") for i in [0, 1, 3]]
    result = check_completeness(shards, 5)
    assert result["found"] == 3
    assert 2 in result["missing"]
    assert 4 in result["missing"]


def test_check_completeness_unexpected_shards():
    shards = [Path(f"spectra_shard_{i}.zarr") for i in [0, 1, 2, 99]]
    result = check_completeness(shards, 3)
    assert 99 in result["unexpected"]


def test_check_completeness_duplicates():
    shards = [Path("spectra_shard_0.zarr"), Path("spectra_shard_0.zarr"), Path("spectra_shard_1.zarr")]
    result = check_completeness(shards, 2)
    assert result["duplicates"] is True


# -- validate_shard ------------------------------------------------------------


def test_validate_shard_valid():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        _make_valid_shard(path, with_provenance=True)
        ok, info = validate_shard(path, enforce_provenance=True)
        assert ok is True
        assert info["spectra"] == 3


def test_validate_shard_valid_no_provenance_enforcement():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        _make_valid_shard(path, with_provenance=False)
        ok, info = validate_shard(path, enforce_provenance=False)
        assert ok is True


def test_validate_shard_rejects_missing_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        _make_valid_shard(path, with_provenance=False)
        ok, info = validate_shard(path, enforce_provenance=True)
        assert ok is False


def test_validate_shard_rejects_nan_flux():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        root = zarr.open_group(str(path), mode="w")
        flux = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        root.array("flux", flux)
        root.array("wavelength", np.array([5000, 5050, 5100], dtype=np.float32))
        ok, info = validate_shard(path, enforce_provenance=False)
        assert ok is False
        assert info.get("nan") is True


def test_validate_shard_rejects_inf_flux():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        root = zarr.open_group(str(path), mode="w")
        flux = np.array([[1.0, np.inf, 3.0]], dtype=np.float32)
        root.array("flux", flux)
        root.array("wavelength", np.array([5000, 5050, 5100], dtype=np.float32))
        ok, info = validate_shard(path, enforce_provenance=False)
        assert ok is False
        assert info.get("inf") is True


def test_validate_shard_rejects_zero_flux():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        root = zarr.open_group(str(path), mode="w")
        flux = np.zeros((1, 5), dtype=np.float32)
        root.array("flux", flux)
        root.array("wavelength", np.linspace(5000, 5050, 5, dtype=np.float32))
        ok, info = validate_shard(path, enforce_provenance=False)
        assert ok is False
        assert info.get("empty") is True


def test_validate_shard_rejects_empty_flux():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.array("flux", np.zeros((0, 5), dtype=np.float32))
        root.array("wavelength", np.linspace(5000, 5050, 5, dtype=np.float32))
        ok, info = validate_shard(path, enforce_provenance=False)
        assert ok is False
        assert info.get("empty") is True


def test_validate_shard_wavelength_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        _make_valid_shard(path, n_lambda=10, with_provenance=False)
        ref_wl = np.linspace(6000, 6100, 10, dtype=np.float32)  # different range
        ok, info = validate_shard(path, reference_wavelength=ref_wl, enforce_provenance=False)
        assert ok is False


def test_validate_shard_corrupt_zarr():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        path.mkdir()
        (path / "garbage").write_text("not zarr")
        ok, info = validate_shard(path, enforce_provenance=False)
        assert ok is False
        assert info.get("corrupt_zarr") is True


def test_validate_shard_missing_arrays():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectra_shard_0.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.array("params", np.zeros((2, 3)))
        ok, info = validate_shard(path, enforce_provenance=False)
        assert ok is False
        assert info.get("missing_arrays") is True
