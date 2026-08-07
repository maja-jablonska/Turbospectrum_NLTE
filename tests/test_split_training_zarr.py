"""Tests for split_training_zarr: global shuffle, split coverage, streaming copy."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from split_training_zarr import main as split_main, _split_sizes
from zarr_compat import (
    zarr_store,
    open_root_group,
    create_root_group,
    create_array,
    write_string_array,
)

N_ROWS = 100
N_LAMBDA = 40


@pytest.fixture()
def source_store(tmp_path):
    path = str(tmp_path / "train_test.zarr")
    root = create_root_group(zarr_store(path), overwrite=True)
    root.attrs["title"] = "test grid"

    rng = np.random.default_rng(42)
    create_array(root, "flux", data=rng.random((N_ROWS, N_LAMBDA)).astype(np.float32),
                 chunks=(16, N_LAMBDA))
    create_array(root, "params", data=rng.random((N_ROWS, 3)).astype(np.float32),
                 chunks=(16, 3))
    create_array(root, "teff", data=np.arange(N_ROWS, dtype=np.float32), chunks=(16,))
    create_array(root, "wavelength", data=np.linspace(4000, 5000, N_LAMBDA).astype(np.float32),
                 chunks=(N_LAMBDA,))
    write_string_array(root, "param_names", ["teff", "logg", "feh"])

    prov = root.create_group("provenance")
    prov.attrs["git_sha"] = "deadbeef"

    splits = root.create_group("splits")
    create_array(splits, "train_indices", data=np.arange(80, dtype=np.int64))
    return path


def _run(source, out_dir, *extra):
    return split_main(["-i", source, "-o", str(out_dir), "--rows-per-split", "30",
                       "--seed", "7", *extra])


def _open_splits(out_dir):
    names = sorted(p for p in os.listdir(out_dir) if p.endswith(".zarr"))
    return [open_root_group(os.path.join(str(out_dir), name)) for name in names]


def test_split_sizes_balanced():
    assert _split_sizes(100, 4) == [25, 25, 25, 25]
    assert _split_sizes(100, 3) == [34, 33, 33]
    assert sum(_split_sizes(7, 7)) == 7


def test_splits_cover_all_rows_exactly_once(source_store, tmp_path):
    out_dir = tmp_path / "out"
    assert _run(source_store, out_dir) == 0

    groups = _open_splits(out_dir)
    assert len(groups) == 4  # ceil(100 / 30)

    all_rows = np.concatenate([g["source_row_index"][...] for g in groups])
    assert sorted(all_rows.tolist()) == list(range(N_ROWS))
    # A fixed-seed global permutation must not be the identity order.
    assert not np.array_equal(all_rows, np.arange(N_ROWS))


def test_row_data_matches_source_rows(source_store, tmp_path):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir)
    src = open_root_group(source_store)

    for grp in _open_splits(out_dir):
        rows = grp["source_row_index"][...]
        np.testing.assert_array_equal(grp["flux"][...], src["flux"][...][rows])
        np.testing.assert_array_equal(grp["params"][...], src["params"][...][rows])
        np.testing.assert_array_equal(grp["teff"][...], src["teff"][...][rows])


def test_static_arrays_attrs_and_subgroups(source_store, tmp_path):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir)

    for grp in _open_splits(out_dir):
        assert grp.attrs["title"] == "test grid"
        np.testing.assert_array_equal(
            grp["wavelength"][...], open_root_group(source_store)["wavelength"][...])
        assert [str(v) for v in grp["param_names"][...]] == ["teff", "logg", "feh"]
        assert grp["provenance"].attrs["git_sha"] == "deadbeef"
        # Stale source-row split indices must not be carried over.
        assert "splits" not in set(grp.group_keys())
        assert grp["split"].attrs["seed"] == 7
        assert grp["split"].attrs["source_row_count"] == N_ROWS


def test_deterministic_across_runs(source_store, tmp_path):
    _run(source_store, tmp_path / "a")
    _run(source_store, tmp_path / "b")
    rows_a = [g["source_row_index"][...] for g in _open_splits(tmp_path / "a")]
    rows_b = [g["source_row_index"][...] for g in _open_splits(tmp_path / "b")]
    for a, b in zip(rows_a, rows_b):
        np.testing.assert_array_equal(a, b)


def test_split_index_writes_single_split(source_store, tmp_path):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir, "--split-index", "2")
    names = sorted(p for p in os.listdir(out_dir) if p.endswith(".zarr"))
    assert names == ["train_test_split_002.zarr"]

    # The lone split must match split 2 of a full run (same seed).
    full_dir = tmp_path / "full"
    _run(source_store, full_dir)
    single = open_root_group(os.path.join(str(out_dir), names[0]))
    full = open_root_group(os.path.join(str(full_dir), "train_test_split_002.zarr"))
    np.testing.assert_array_equal(single["source_row_index"][...],
                                  full["source_row_index"][...])


def test_existing_outputs_skipped(source_store, tmp_path, capsys):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir)
    before = {g["split"].attrs["split_index"] for g in _open_splits(out_dir)}
    _run(source_store, out_dir)  # rerun: all four must be skipped, not rewritten
    out = capsys.readouterr().out
    assert out.count("exists, skipping") == 4
    assert {g["split"].attrs["split_index"] for g in _open_splits(out_dir)} == before


def test_small_batch_streaming_matches(source_store, tmp_path):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir, "--batch-rows", "7")
    src = open_root_group(source_store)
    for grp in _open_splits(out_dir):
        rows = grp["source_row_index"][...]
        np.testing.assert_array_equal(grp["flux"][...], src["flux"][...][rows])


def test_never_deletes_source_or_foreign_files(source_store, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # A stale tmp dir from another (crashed or concurrent) job must survive.
    foreign_tmp = out_dir / "train_test_split_000.zarr.tmp-99999"
    foreign_tmp.mkdir()
    (foreign_tmp / "marker").write_text("in-progress work")

    src_files = sorted(
        os.path.join(dp, f) for dp, _dn, fn in os.walk(source_store) for f in fn)
    _run(source_store, out_dir)

    assert (foreign_tmp / "marker").read_text() == "in-progress work"
    after = sorted(
        os.path.join(dp, f) for dp, _dn, fn in os.walk(source_store) for f in fn)
    assert after == src_files  # source store untouched
    # No leftover tmp dirs from this run itself.
    ours = [p for p in os.listdir(out_dir) if f".tmp-{os.getpid()}" in p]
    assert ours == []


def test_dry_run_writes_nothing(source_store, tmp_path):
    out_dir = tmp_path / "out"
    assert _run(source_store, out_dir, "--dry-run") == 0
    assert not out_dir.exists()
