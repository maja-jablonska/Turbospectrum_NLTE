"""Tests for grid generation: parameter resolution, sampling, LHS, and output."""
from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from generate_grid import (
    _abundance_value,
    _choose_series,
    _compressed_csv_path,
    _format_abundance_value,
    _latin_hypercube,
    _ordered_abundance_keys,
    _resolve_ml_sampling,
    _resolve_values,
    _sample_parameter_space,
    _write_csv_from_columns,
    _write_zarr_from_columns,
)


# -- _resolve_values ----------------------------------------------------------


def test_resolve_values_min_max_step():
    values = _resolve_values("teff", {"min": 5000, "max": 6000, "step": 250}, None)
    assert values == [5000, 5250, 5500, 5750, 6000]


def test_resolve_values_list():
    values = _resolve_values("logg", [3.5, 4.0, 4.5], None)
    assert values == [3.5, 4.0, 4.5]


def test_resolve_values_single():
    values = _resolve_values("feh", -1.0, None)
    assert values == [-1.0]


def test_resolve_values_string():
    values = _resolve_values("a", "+0.00", None)
    assert values == ["+0.00"]


def test_resolve_values_distribution_gaussian():
    rng = np.random.default_rng(42)
    values = _resolve_values(
        "a",
        {"distribution": {"type": "gaussian", "mean": 0.0, "sigma": 0.1, "count": 10}},
        rng,
    )
    assert len(values) == 10
    assert all(isinstance(v, float) for v in values)


def test_resolve_values_distribution_uniform():
    rng = np.random.default_rng(42)
    values = _resolve_values(
        "feh",
        {"distribution": {"type": "uniform", "min": -2.0, "max": 0.5, "count": 20}},
        rng,
    )
    assert len(values) == 20
    assert all(-2.0 <= v <= 0.5 for v in values)


def test_resolve_values_abundance_formatting():
    rng = np.random.default_rng(42)
    values = _resolve_values(
        "a",
        {"distribution": {"type": "uniform", "min": -0.5, "max": 0.5, "count": 5}},
        rng,
        is_abundance=True,
    )
    assert len(values) == 5
    for v in values:
        assert isinstance(v, str)
        assert v[0] in ("+", "-")


def test_resolve_values_dict_with_values():
    values = _resolve_values("t_value", {"values": ["01", "02"]}, None)
    assert values == ["01", "02"]


# -- _sample_parameter_space --------------------------------------------------


def test_sample_parameter_space_full():
    params = {"a": [1, 2], "b": [3, 4]}
    combos = list(_sample_parameter_space(params, {"method": "full"}, None))
    assert len(combos) == 4
    assert (1, 3) in combos


def test_sample_parameter_space_full_with_max_rows():
    params = {"a": [1, 2, 3], "b": [4, 5, 6]}
    combos = list(_sample_parameter_space(params, {"method": "full", "max_rows": 5}, None))
    assert len(combos) == 5


def test_sample_parameter_space_random():
    rng = np.random.default_rng(0)
    params = {"a": [1, 2], "b": [3, 4]}
    combos = list(_sample_parameter_space(params, {"method": "random", "max_rows": 10}, rng))
    assert len(combos) == 10


def test_sample_parameter_space_latin_hypercube():
    rng = np.random.default_rng(0)
    params = {"a": [1, 2, 3], "b": [4, 5, 6]}
    combos = list(
        _sample_parameter_space(params, {"method": "latin_hypercube", "max_rows": 6}, rng)
    )
    assert len(combos) == 6


# -- _latin_hypercube ----------------------------------------------------------


def test_latin_hypercube_bounds_respected():
    rng = np.random.default_rng(42)
    lhs = _latin_hypercube([(0.0, 1.0), (10.0, 20.0)], 100, rng)
    assert lhs.shape == (100, 2)
    assert np.all(lhs[:, 0] >= 0.0) and np.all(lhs[:, 0] <= 1.0)
    assert np.all(lhs[:, 1] >= 10.0) and np.all(lhs[:, 1] <= 20.0)


def test_latin_hypercube_stratification():
    """Each stratum [i/N, (i+1)/N] should contain exactly one sample."""
    rng = np.random.default_rng(7)
    n = 50
    lhs = _latin_hypercube([(0.0, 1.0)], n, rng)
    bins = np.floor(lhs[:, 0] * n).astype(int)
    bins = np.clip(bins, 0, n - 1)
    assert len(np.unique(bins)) == n


def test_latin_hypercube_deterministic():
    lhs1 = _latin_hypercube([(0, 1), (0, 1)], 20, np.random.default_rng(99))
    lhs2 = _latin_hypercube([(0, 1), (0, 1)], 20, np.random.default_rng(99))
    np.testing.assert_array_equal(lhs1, lhs2)


# -- _abundance_value / _format_abundance_value --------------------------------


def test_abundance_value_float():
    assert _abundance_value(0.3) == "+0.30"
    assert _abundance_value(-1.0) == "-1.00"


def test_abundance_value_none():
    assert _abundance_value(None) == "+0.00"


def test_abundance_value_string_passthrough():
    assert _abundance_value("+0.40") == "+0.40"


def test_format_abundance_value_int():
    assert _format_abundance_value(0) == "+0.00"


# -- _choose_series ------------------------------------------------------------


def test_choose_series_scalar():
    rng = np.random.default_rng(0)
    result = _choose_series("01", rng, 5, "turbvel")
    assert len(result) == 5
    assert all(v == "01" for v in result)


def test_choose_series_list():
    rng = np.random.default_rng(0)
    result = _choose_series(["01", "02", "03"], rng, 10, "turbvel")
    assert len(result) == 10
    assert set(result).issubset({"01", "02", "03"})


# -- _ordered_abundance_keys --------------------------------------------------


def test_ordered_abundance_keys_legacy_order():
    cfg = {"c": "+0.00", "a": "+0.00", "o": "+0.00", "n": "+0.00", "r": "+0.00", "s": "+0.00"}
    keys = _ordered_abundance_keys(cfg)
    assert keys == ["a", "c", "n", "o", "r", "s"]


def test_ordered_abundance_keys_with_extra():
    cfg = {"a": "+0.00", "mg": "+0.10"}
    keys = _ordered_abundance_keys(cfg)
    assert keys[0] == "a"
    assert "mg" in keys


# -- _compressed_csv_path -----------------------------------------------------


def test_compressed_csv_path_none():
    assert _compressed_csv_path("grid.csv", None) is None
    assert _compressed_csv_path("grid.csv", "none") is None


def test_compressed_csv_path_gzip():
    assert _compressed_csv_path("grid.csv", "gzip") == "grid.csv.gz"
    assert _compressed_csv_path("grid.csv.gz", "gzip") == "grid.csv.gz"


def test_compressed_csv_path_zstd():
    assert _compressed_csv_path("grid.csv", "zstd") == "grid.csv.zst"


# -- _resolve_ml_sampling (end-to-end) ----------------------------------------


def test_resolve_ml_sampling_produces_columns():
    rng = np.random.default_rng(42)
    cfg = {
        "num_samples": 10,
        "bounds": {
            "teff": {"min": 4000, "max": 7000},
            "logg": {"min": 0.0, "max": 5.0},
            "feh": {"min": -2.5, "max": 0.5},
        },
        "abundances": {"a": "+0.00", "c": "+0.00", "n": "+0.00", "o": "+0.00", "r": "+0.00", "s": "+0.00"},
        "synthesis": {"lam_min": 5000, "lam_max": 5100, "lam_step": 0.01},
    }
    cols = _resolve_ml_sampling(cfg, rng)
    assert "teff" in cols and "logg" in cols and "feh" in cols
    assert len(cols["teff"]) == 10
    assert np.all(cols["teff"] >= 4000) and np.all(cols["teff"] <= 7000)
    assert np.all(cols["logg"] >= 0.0) and np.all(cols["logg"] <= 5.0)
    assert np.all(cols["feh"] >= -2.5) and np.all(cols["feh"] <= 0.5)


def test_resolve_ml_sampling_with_sampled_abundances():
    rng = np.random.default_rng(42)
    cfg = {
        "num_samples": 15,
        "bounds": {
            "teff": {"min": 4000, "max": 7000},
            "logg": {"min": 0.0, "max": 5.0},
            "feh": {"min": -2.5, "max": 0.5},
        },
        "abundances": {"a": {"min": -0.5, "max": 0.5}, "c": "+0.00", "n": "+0.00", "o": "+0.00", "r": "+0.00", "s": "+0.00"},
        "synthesis": {},
    }
    cols = _resolve_ml_sampling(cfg, rng)
    assert len(cols["a"]) == 15
    for val in cols["a"]:
        f = float(val)
        assert -0.5 <= f <= 0.5


def test_resolve_ml_sampling_continuous_vmicro_keeps_turbvel_and_derives_t_value():
    """Continuous `bounds.vmicro` must keep `turbvel` continuous (it flows to
    babsma line formation) while `t_value` — the discrete atmosphere label that
    will actually be read — is provably the nearest `t_value_options` entry to
    each row's sampled `turbvel`. Mirrors the snap synthesis re-applies against
    the on-disk MARCS labels, so the two stay faithful to each other.
    """
    from synthesize_regular_grid import nearest_t_value_for_turbvel

    options = ["01", "02"]
    cfg = {
        "num_samples": 200,
        "seed": 7,
        "bounds": {
            "teff": {"min": 4500, "max": 6500},
            "logg": {"min": 3.5, "max": 5.0},
            "feh": {"min": -1.0, "max": 0.5},
            "vmicro": {"min": 1.0, "max": 2.0},
        },
        "t_value_options": options,
        "abundances": {"a": "+0.00", "c": "+0.00", "n": "+0.00", "o": "+0.00", "r": "+0.00", "s": "+0.00"},
        "synthesis": {"lam_min": 5000, "lam_max": 5100, "lam_step": 0.01},
    }
    cols = _resolve_ml_sampling(cfg, np.random.default_rng(7))

    turbvel = np.asarray(cols["turbvel"], dtype=float)
    # Continuous: within bounds and NOT collapsed onto the discrete label set.
    assert np.all(turbvel >= 1.0) and np.all(turbvel <= 2.0)
    assert len(np.unique(turbvel)) > len(options)
    # t_value is a discrete atmosphere label drawn only from the options...
    assert set(cols["t_value"].tolist()).issubset(set(options))
    # ...and is provably the nearest option to turbvel for every single row.
    expected = nearest_t_value_for_turbvel(cols["turbvel"], options)
    np.testing.assert_array_equal(expected, cols["t_value"])


def test_resolve_ml_sampling_rejects_continuous_and_discrete_turbvel_together():
    """`bounds.vmicro` (continuous) and `sample_turbvel` (discrete) are mutually
    exclusive — choosing both is ambiguous about what drives the atmosphere."""
    cfg = {
        "num_samples": 5,
        "bounds": {
            "teff": {"min": 4500, "max": 6500},
            "logg": {"min": 3.5, "max": 5.0},
            "feh": {"min": -1.0, "max": 0.5},
            "vmicro": {"min": 1.0, "max": 2.0},
        },
        "sample_turbvel": True,
        "turbvel_options": ["01", "02"],
        "abundances": {"a": "+0.00", "c": "+0.00", "n": "+0.00", "o": "+0.00", "r": "+0.00", "s": "+0.00"},
        "synthesis": {},
    }
    with pytest.raises(ValueError, match="continuous"):
        _resolve_ml_sampling(cfg, np.random.default_rng(0))


def test_resolve_ml_sampling_deterministic():
    cfg = {
        "num_samples": 20,
        "bounds": {
            "teff": {"min": 4000, "max": 7000},
            "logg": {"min": 0.0, "max": 5.0},
            "feh": {"min": -2.5, "max": 0.5},
        },
        "abundances": {"a": "+0.00", "c": "+0.00", "n": "+0.00", "o": "+0.00", "r": "+0.00", "s": "+0.00"},
        "synthesis": {},
    }
    cols1 = _resolve_ml_sampling(cfg, np.random.default_rng(99))
    cols2 = _resolve_ml_sampling(cfg, np.random.default_rng(99))
    np.testing.assert_array_equal(cols1["teff"], cols2["teff"])
    np.testing.assert_array_equal(cols1["feh"], cols2["feh"])


# -- CSV and Zarr I/O ---------------------------------------------------------


def test_write_csv_from_columns_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.csv")
        cols = {
            "teff": np.array([5000, 5500, 6000]),
            "logg": np.array([4.0, 4.5, 3.5]),
        }
        _write_csv_from_columns(cols, path, compression=None, level=None)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["teff", "logg"]
            rows = list(reader)
            assert len(rows) == 3
            assert rows[0] == ["5000", "4.0"]


def test_write_zarr_from_columns():
    import zarr

    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "grid.zarr")
        cols = {
            "teff": np.array([5000, 5500], dtype=np.float32),
            "logg": np.array([4.0, 4.5], dtype=np.float32),
        }
        _write_zarr_from_columns(cols, zarr_path, chunks=64, compressor_cfg=None)
        root = zarr.open_group(zarr_path, mode="r")
        assert "teff" in root
        np.testing.assert_array_equal(root["teff"][:], [5000, 5500])
