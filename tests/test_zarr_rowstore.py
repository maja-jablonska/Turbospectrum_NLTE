import os
import sys

import numpy as np
import zarr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from subsample_spectra import subsample  # noqa: E402
from zarr_compat import create_array, write_string_array  # noqa: E402
from zarr_rowstore import write_row_subset  # noqa: E402


def _store(path, n=6, n_lambda=6):
    root = zarr.open_group(str(path), mode="w")
    create_array(root, "wavelength", data=np.linspace(5000, 5001, n_lambda))
    create_array(root, "flux", data=np.arange(n * n_lambda, dtype=np.float32).reshape(n, n_lambda))
    create_array(root, "teff", data=np.arange(n, dtype=np.float32) + 5000)
    write_string_array(root, "status", ["success"] * (n - 1) + ["failed"])
    root.attrs.update({"n_models": n, "physics_hash": "p" * 64, "status_counts": {"success": n - 1, "failed": 1}})
    prov = root.create_group("provenance")
    prov.attrs["config"] = "x"
    return root


def test_rows_equal_lambda_does_not_slice_wavelength(tmp_path):
    src = _store(tmp_path / "src.zarr", n=6, n_lambda=6)
    write_row_subset(src, str(tmp_path / "out.zarr"), np.array([4, 1]), stage="t", source_path="src")
    out = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert out["wavelength"].shape == (6,)
    np.testing.assert_array_equal(out["teff"][...], [5004, 5001])
    np.testing.assert_array_equal(out["flux"][...], src["flux"][...][[4, 1]])


def test_chained_subsets_compose_source_row_index_and_lineage(tmp_path):
    src = _store(tmp_path / "src.zarr")
    write_row_subset(src, str(tmp_path / "a.zarr"), np.array([1, 3, 5]), stage="snap_filter", source_path="src")
    a = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    # Chaining used to crash re-creating source_row_index.
    subsample(str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr"), 2, seed=0, head=True)
    b = zarr.open_group(str(tmp_path / "b.zarr"), mode="r")

    np.testing.assert_array_equal(b["source_row_index"][...], [1, 3])  # indices into the ORIGINAL store
    assert [e["stage"] for e in b.attrs["lineage"]] == ["snap_filter", "subsample"]
    assert b.attrs["n_models"] == 2
    assert a.attrs["status_counts"] == {"success": 2, "failed": 1}
    assert a.attrs["physics_hash"] != "p" * 64
    assert a.attrs["lineage"][0]["parent_physics_hash"] == "p" * 64
    assert "provenance" in b and "snap_filter" in b and "subsample" in b


def test_failure_leaves_no_output(tmp_path, monkeypatch):
    import zarr_rowstore

    src = _store(tmp_path / "src.zarr")

    def boom(*a, **k):
        raise RuntimeError("killed")

    monkeypatch.setattr(zarr_rowstore, "copy_rows", boom)
    try:
        write_row_subset(src, str(tmp_path / "out.zarr"), np.array([0]), stage="t", source_path="src")
    except RuntimeError:
        pass
    assert sorted(os.listdir(tmp_path)) == ["src.zarr"]
