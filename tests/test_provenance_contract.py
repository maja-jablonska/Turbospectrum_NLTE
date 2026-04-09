"""Tests for provenance contract: hashing, validation, and field checking."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from provenance_contract import (
    REQUIRED_PROVENANCE_FIELDS,
    assert_required_provenance_fields,
    canonical_json_sha256,
    compute_binary_manifest_hash,
    compute_grid_definition_hash,
    directory_manifest_sha256,
    file_sha256,
    is_meaningful_provenance_value,
    stable_numeric_digest,
    stable_string_digest,
)


# -- is_meaningful_provenance_value -------------------------------------------


def test_meaningful_value_accepts_real_strings():
    assert is_meaningful_provenance_value("abc123")
    assert is_meaningful_provenance_value("v1.0")
    assert is_meaningful_provenance_value(42)


def test_meaningful_value_rejects_empty_and_missing():
    assert not is_meaningful_provenance_value(None)
    assert not is_meaningful_provenance_value("")
    assert not is_meaningful_provenance_value("   ")


def test_meaningful_value_rejects_sentinel_tokens():
    for token in ("unknown", "not_recorded", "none", "null", "n/a", "na", "<unset>"):
        assert not is_meaningful_provenance_value(token), f"Should reject {token!r}"
        assert not is_meaningful_provenance_value(token.upper()), f"Should reject {token.upper()!r}"


# -- canonical_json_sha256 ----------------------------------------------------


def test_canonical_json_deterministic():
    a = canonical_json_sha256({"b": 2, "a": 1})
    b = canonical_json_sha256({"a": 1, "b": 2})
    assert a == b


def test_canonical_json_different_values():
    a = canonical_json_sha256({"x": 1})
    b = canonical_json_sha256({"x": 2})
    assert a != b


def test_canonical_json_is_hex_sha256():
    h = canonical_json_sha256([1, 2, 3])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# -- file_sha256 --------------------------------------------------------------


def test_file_sha256_deterministic():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".txt") as f:
        f.write(b"hello world")
        path = f.name
    try:
        h1 = file_sha256(path)
        h2 = file_sha256(path)
        assert h1 == h2
        assert len(h1) == 64
    finally:
        os.unlink(path)


def test_file_sha256_different_content():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "a.txt")
        p2 = os.path.join(tmp, "b.txt")
        with open(p1, "wb") as f:
            f.write(b"AAA")
        with open(p2, "wb") as f:
            f.write(b"BBB")
        assert file_sha256(p1) != file_sha256(p2)


# -- directory_manifest_sha256 ------------------------------------------------


def test_directory_manifest_sha256_with_files():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "a.dat"), "w") as f:
            f.write("data1")
        with open(os.path.join(tmp, "b.dat"), "w") as f:
            f.write("data2")
        h = directory_manifest_sha256(tmp)
        assert len(h) == 64


def test_directory_manifest_sha256_suffix_filter():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "a.dat"), "w") as f:
            f.write("data")
        with open(os.path.join(tmp, "b.txt"), "w") as f:
            f.write("text")
        h_dat = directory_manifest_sha256(tmp, suffixes=[".dat"])
        h_all = directory_manifest_sha256(tmp)
        assert h_dat != h_all


def test_directory_manifest_sha256_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        h = directory_manifest_sha256(tmp)
        assert h == ""


def test_directory_manifest_sha256_nonexistent():
    h = directory_manifest_sha256("/nonexistent/path/abc123")
    assert h == ""


def test_directory_manifest_sha256_with_content_hashing():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "a.dat"), "w") as f:
            f.write("data")
        h1 = directory_manifest_sha256(tmp, hash_contents=False)
        h2 = directory_manifest_sha256(tmp, hash_contents=True)
        assert h1 != h2


# -- stable_numeric_digest ----------------------------------------------------


def test_stable_numeric_digest_float():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    h1 = stable_numeric_digest(arr, "<f8")
    h2 = stable_numeric_digest(arr, "<f8")
    assert h1 == h2


def test_stable_numeric_digest_different_data():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 3.0])
    assert stable_numeric_digest(a, "<f8") != stable_numeric_digest(b, "<f8")


def test_stable_numeric_digest_handles_nan():
    arr = np.array([1.0, np.nan, 3.0])
    h = stable_numeric_digest(arr, "<f8")
    assert len(h) == 64


def test_stable_numeric_digest_int():
    arr = np.array([10, 20, 30], dtype=np.int64)
    h = stable_numeric_digest(arr, "<i8")
    assert len(h) == 64


# -- stable_string_digest -----------------------------------------------------


def test_stable_string_digest_deterministic():
    h1 = stable_string_digest(["alpha", "beta", "gamma"])
    h2 = stable_string_digest(["alpha", "beta", "gamma"])
    assert h1 == h2


def test_stable_string_digest_order_matters():
    a = stable_string_digest(["a", "b"])
    b = stable_string_digest(["b", "a"])
    assert a != b


# -- compute_grid_definition_hash ---------------------------------------------


def test_grid_definition_hash_deterministic():
    cols = {
        "teff": np.array([5000.0, 5500.0]),
        "logg": np.array([4.0, 4.5]),
    }
    h1 = compute_grid_definition_hash(cols)
    h2 = compute_grid_definition_hash(cols)
    assert h1 == h2


def test_grid_definition_hash_changes_with_data():
    cols_a = {"teff": np.array([5000.0, 5500.0])}
    cols_b = {"teff": np.array([5000.0, 6000.0])}
    assert compute_grid_definition_hash(cols_a) != compute_grid_definition_hash(cols_b)


def test_grid_definition_hash_handles_string_columns():
    cols = {
        "teff": np.array([5000.0]),
        "mode": np.array(["LTE"], dtype=object),
    }
    h = compute_grid_definition_hash(cols)
    assert len(h) == 64


def test_grid_definition_hash_handles_bool_columns():
    cols = {"flag": np.array([True, False, True])}
    h = compute_grid_definition_hash(cols)
    assert len(h) == 64


# -- compute_binary_manifest_hash ---------------------------------------------


def test_binary_manifest_hash_with_files():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "bsyn_lu")
        with open(p1, "wb") as f:
            f.write(b"\x7fELF_fake_binary")
        h = compute_binary_manifest_hash([p1])
        assert len(h) == 64


def test_binary_manifest_hash_empty():
    h = compute_binary_manifest_hash([])
    assert h == ""


def test_binary_manifest_hash_missing_files_skipped():
    h = compute_binary_manifest_hash(["/nonexistent/binary"])
    assert h == ""


def test_binary_manifest_hash_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "bin")
        with open(p, "wb") as f:
            f.write(b"content")
        h1 = compute_binary_manifest_hash([p])
        h2 = compute_binary_manifest_hash([p])
        assert h1 == h2


# -- assert_required_provenance_fields ----------------------------------------


def test_assert_provenance_passes_with_all_fields():
    fields = {key: f"value_{i}" for i, key in enumerate(REQUIRED_PROVENANCE_FIELDS)}
    assert_required_provenance_fields(fields, context="test")


def test_assert_provenance_fails_on_missing_field():
    fields = {key: f"value_{i}" for i, key in enumerate(REQUIRED_PROVENANCE_FIELDS)}
    fields["config_hash"] = ""
    with pytest.raises(ValueError, match="missing required provenance"):
        assert_required_provenance_fields(fields, context="test")


def test_assert_provenance_fails_on_sentinel():
    fields = {key: f"value_{i}" for i, key in enumerate(REQUIRED_PROVENANCE_FIELDS)}
    fields["git_commit"] = "unknown"
    with pytest.raises(ValueError, match="git_commit"):
        assert_required_provenance_fields(fields, context="test")
