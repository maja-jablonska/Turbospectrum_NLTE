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


def test_split_stride_partitions_all_splits(source_store, tmp_path):
    # 4 splits over 3 strided "tasks" must together produce the same result
    # as one full run: task k writes splits k, k+3, ...
    out_dir = tmp_path / "out"
    for task in range(3):
        _run(source_store, out_dir, "--split-index", str(task), "--split-stride", "3")
    names = sorted(p for p in os.listdir(out_dir) if p.endswith(".zarr"))
    assert names == [f"train_test_split_{k:03d}.zarr" for k in range(4)]

    full_dir = tmp_path / "full"
    _run(source_store, full_dir)
    for name in names:
        a = open_root_group(os.path.join(str(out_dir), name))
        b = open_root_group(os.path.join(str(full_dir), name))
        np.testing.assert_array_equal(a["source_row_index"][...],
                                      b["source_row_index"][...])


def test_split_stride_requires_split_index(source_store, tmp_path):
    with pytest.raises(ValueError, match="--split-stride requires --split-index"):
        _run(source_store, tmp_path / "out", "--split-stride", "3")


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


def test_rerun_with_different_settings_refuses(source_store, tmp_path):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir)
    with pytest.raises(ValueError, match="different split settings"):
        _run(source_store, out_dir, "--seed", "99")


def test_spare_array_task_exits_cleanly(source_store, tmp_path):
    assert split_main(["-i", str(source_store), "-o", str(tmp_path / "o"),
                                     "--num-splits", "4", "--split-index", "7"]) == 0


def test_block_shuffle_covers_all_rows(source_store, tmp_path):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir, "--shuffle", "block")
    src = open_root_group(source_store)
    seen = []
    for grp in _open_splits(out_dir):
        rows = grp["source_row_index"][...]
        np.testing.assert_array_equal(grp["flux"][...], src["flux"][...][rows])
        seen.extend(rows.tolist())
    assert sorted(seen) == list(range(src["flux"].shape[0]))


def test_split_updates_lineage_and_counts(source_store, tmp_path):
    out_dir = tmp_path / "out"
    _run(source_store, out_dir)
    for grp in _open_splits(out_dir):
        assert grp.attrs["n_models"] == grp["flux"].shape[0]
        assert grp.attrs["lineage"][-1]["stage"] == "split"


def _store_contents(path):
    """Every array (values + dtype + chunks) and attrs of a store, recursively."""
    import zarr

    out = {}

    def walk(grp, prefix):
        out[prefix + "@attrs"] = {k: v for k, v in dict(grp.attrs).items()}
        for name in sorted(grp.array_keys()):
            arr = grp[name]
            out[prefix + name] = (np.asarray(arr[...]).tolist(), str(arr.dtype), tuple(arr.chunks))
        for name in sorted(grp.group_keys()):
            walk(grp[name], prefix + name + "/")

    walk(zarr.open_group(path, mode="r"), "")
    return out


@pytest.mark.parametrize("extra", [(), ("--max-buffer-gb", "1e-6"), ("--split-index", "1", "--split-stride", "2")])
def test_single_pass_identical_to_streaming(source_store, tmp_path, extra):
    a, b = tmp_path / "single", tmp_path / "stream"
    assert _run(source_store, a, *extra) == 0
    assert _run(source_store, b, "--no-single-pass", "--batch-rows", "7", *extra) == 0
    names = sorted(p for p in os.listdir(a) if p.endswith(".zarr"))
    assert names == sorted(p for p in os.listdir(b) if p.endswith(".zarr")) and names
    for name in names:
        assert _store_contents(str(a / name)) == _store_contents(str(b / name)), name


def test_gather_rows_matches_fancy_indexing(source_store):
    from zarr_rowstore import gather_rows

    src = open_root_group(source_store)
    flux = src["flux"][...]
    rng = np.random.default_rng(3)
    perm = rng.permutation(N_ROWS)
    lists = [perm[:30], perm[30:31], perm[31:31], perm[31:]]
    for slab in (1, 16, 17, 48, 1000):
        got = gather_rows(src["flux"], lists, slab_rows=slab)
        for rows, g in zip(lists, got):
            np.testing.assert_array_equal(g, flux[rows])
    with pytest.raises(IndexError):
        gather_rows(src["flux"], [np.array([N_ROWS])])
