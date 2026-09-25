"""Shared helpers for row-oriented Zarr stores (shards, merged, derived).

Keep store-level plumbing (atomic publishing, status decoding) here so the
merge / filter / split / subsample stages do not each re-implement it.
"""
from __future__ import annotations

import contextlib
import os
import shutil
from typing import Iterator, List, Mapping, Optional

import numpy as np

try:
    from synthesis_validation import SUCCESS_STATUSES
except ImportError:  # imported as scripts.zarr_rowstore
    from .synthesis_validation import SUCCESS_STATUSES


def decode_statuses(values) -> List[str]:
    """Normalise a zarr status array (bytes/str/object) to lowercase strings."""
    out: List[str] = []
    for item in np.asarray(values).tolist():
        if isinstance(item, (bytes, bytearray)):
            item = item.decode("utf-8", errors="ignore")
        out.append(str(item).strip().lower())
    return out


def success_mask(values) -> np.ndarray:
    return np.array([s in SUCCESS_STATUSES for s in decode_statuses(values)], dtype=bool)


@contextlib.contextmanager
def atomic_output(final_path: str) -> Iterator[str]:
    """Yield a sibling tmp path; publish it at ``final_path`` only on success.

    The tmp store lives next to the final path so the rename stays on one
    filesystem (SYSTEM_RULES: atomic writes). An existing final store is only
    removed after the new one is complete, so an interrupted run never
    destroys the previous good output.
    """
    final_path = os.path.abspath(final_path)
    parent = os.path.dirname(final_path)
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{final_path}.tmp-{os.getpid()}"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    try:
        yield tmp_path
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    old_path = None
    if os.path.exists(final_path):
        old_path = f"{final_path}.old-{os.getpid()}"
        os.rename(final_path, old_path)
    os.rename(tmp_path, final_path)
    if old_path is not None:
        shutil.rmtree(old_path, ignore_errors=True)


def expected_shard_indices(shard_index: int, shard_count: int, row_count: int) -> np.ndarray:
    """Global rows owned by one shard under the strided ``index::count`` layout."""
    return np.arange(shard_index, row_count, shard_count, dtype=np.int64)


def check_shard(
    path: str,
    *,
    expected_global_indices: Optional[np.ndarray] = None,
    expected_wavelengths: Optional[np.ndarray] = None,
    expected_attrs: Optional[Mapping[str, str]] = None,
    drop_failed_rows: bool = False,
) -> Optional[str]:
    """Return None if the shard store is complete and usable, else a reason.

    This is the single definition of "valid shard" used by resume (skip or
    recompute), the PBS missing-shard report, and stale-layout pruning.
    ``drop_failed_rows`` accepts non-success rows as long as success rows
    have finite flux (mirrors merge --drop-failed-rows).
    """
    import zarr

    try:
        root = zarr.open_group(path, mode="r")
    except Exception as exc:
        return f"unreadable: {exc}"
    missing = [n for n in ("wavelength", "global_index", "flux", "status", "message") if n not in root]
    if missing:
        return f"missing arrays {missing}"
    try:
        gidx = np.asarray(root["global_index"][:], dtype=np.int64)
        flux = np.asarray(root["flux"][:], dtype=np.float32)
        statuses = decode_statuses(root["status"][:])
    except Exception as exc:
        return f"unreadable arrays: {exc}"
    if gidx.ndim != 1 or gidx.size == 0:
        return f"bad global_index shape {gidx.shape}"
    if flux.ndim != 2 or flux.shape[0] != gidx.size:
        return f"flux shape {flux.shape} does not match rows={gidx.size}"
    if len(statuses) != gidx.size:
        return "status length does not match rows"
    if expected_global_indices is not None and not np.array_equal(gidx, expected_global_indices):
        return "layout_mismatch"
    if np.unique(gidx).size != gidx.size:
        return "duplicate global_index"
    if expected_wavelengths is not None:
        wl = np.asarray(root["wavelength"][:], dtype=np.float64)
        if wl.size != expected_wavelengths.size or not np.allclose(wl, expected_wavelengths, atol=1e-4):
            return "wavelength_mismatch"
    if expected_attrs:
        # A shard without the attr predates it (legacy) and is accepted; one
        # that records a different value was produced from different inputs.
        for key, want in expected_attrs.items():
            have = root.attrs.get(key)
            if have not in (None, "") and str(have) != str(want):
                return f"{key}_mismatch"

    ok = np.array([s in SUCCESS_STATUSES for s in statuses], dtype=bool)
    if drop_failed_rows:
        rows = ok
    else:
        if not ok.all():
            return f"non-success statuses {sorted(set(np.asarray(statuses)[~ok].tolist()))}"
        rows = slice(None)
    if not np.all(np.isfinite(flux[rows])):
        return "non-finite flux"
    if "continuum" in root:
        continuum = np.asarray(root["continuum"][:], dtype=np.float32)
        if continuum.shape != flux.shape:
            return "continuum shape mismatch"
        cont = continuum[rows]
        if cont.size and not np.all(np.any(np.isfinite(cont), axis=1)):
            return "rows without finite continuum"
    return None
