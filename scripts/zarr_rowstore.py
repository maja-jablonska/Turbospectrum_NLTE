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
    import zarr_compat as _zc
    from synthesis_validation import SUCCESS_STATUSES
except ImportError:  # imported as scripts.zarr_rowstore
    from . import zarr_compat as _zc
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


# ---------------------------------------------------------------------------
# Row-subset stores (filter / split / subsample)
# ---------------------------------------------------------------------------

# Never row-aligned even when their length happens to equal the row count.
NON_ROW_ARRAYS = frozenset({"wavelength", "param_names", "physics_hash", "schema_version", "dropped_global_index"})
_ROW_COUNT_HINTS = ("flux", "spectra", "labels", "teff", "global_index", "status")
_BATCH_TARGET_BYTES = 512 * 1024 ** 2


def row_count(group) -> int:
    """Number of rows of a row-oriented store."""
    for name in _ROW_COUNT_HINTS:
        if name in group:
            return int(group[name].shape[0])
    lengths: dict = {}
    for name in group.array_keys():
        shp = group[name].shape
        if shp and name not in NON_ROW_ARRAYS:
            lengths[shp[0]] = lengths.get(shp[0], 0) + 1
    if not lengths:
        raise ValueError("Could not infer row count: store has no arrays.")
    return max(lengths.items(), key=lambda kv: kv[1])[0]


def is_row_aligned(name: str, arr, n_rows: int) -> bool:
    return name not in NON_ROW_ARRAYS and bool(arr.shape) and arr.shape[0] == n_rows


def copy_attrs(src, dst) -> None:
    for key, value in dict(src.attrs).items():
        dst.attrs[key] = value


def _write_full(dst_group, name: str, data, chunks=None):
    compression_kwargs, create_array, write_string_array = _zc.compression_kwargs, _zc.create_array, _zc.write_string_array
    data = np.asarray(data)
    if data.dtype.kind in "USOT":
        if data.ndim > 1:
            raise ValueError(f"Unsupported multi-dimensional string array: {name}")
        return write_string_array(dst_group, name, [str(v) for v in data.ravel().tolist()],
                                  compression_kw=compression_kwargs())
    if chunks is not None and data.ndim >= 1:
        chunks = tuple(max(1, min(c, s)) for c, s in zip(chunks, data.shape))
    return create_array(dst_group, name, data=data, chunks=chunks, **compression_kwargs())


def copy_array(src_group, dst_group, name: str):
    arr = src_group[name]
    new = _write_full(dst_group, name, arr[...], chunks=arr.chunks)
    copy_attrs(arr, new)
    return new


def copy_subgroups(src, dst, *, drop=()) -> None:
    """Recursively copy subgroups (e.g. provenance) verbatim, except ``drop``."""
    for name in src.group_keys():
        if name in drop:
            continue
        sub_src = src[name]
        sub_dst = dst.create_group(name)
        copy_attrs(sub_src, sub_dst)
        for arr_name in sub_src.array_keys():
            copy_array(sub_src, sub_dst, arr_name)
        copy_subgroups(sub_src, sub_dst)


def copy_rows(src_arr, dst_arr, rows: np.ndarray, batch_rows: int) -> None:
    """Stream ``src_arr[rows]`` into ``dst_arr[:len(rows)]``, keeping ``rows`` order.

    Each batch is read with sorted indices (zarr decodes each touched chunk
    once per batch) and reordered before a contiguous destination write.
    """
    trailing = (slice(None),) * (src_arr.ndim - 1)
    for start in range(0, len(rows), batch_rows):
        idx = rows[start:start + batch_rows]
        order = np.argsort(idx, kind="stable")
        data = src_arr.oindex[(idx[order],) + trailing]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        dst_arr[start:start + len(idx)] = data[inv]


def default_batch_rows(arrays) -> int:
    widest = 1
    for arr in arrays:
        per_row = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
        widest = max(widest, per_row * int(np.dtype(arr.dtype).itemsize))
    return int(min(4096, max(64, _BATCH_TARGET_BYTES // widest)))


def gather_rows(src_arr, row_lists, slab_rows: Optional[int] = None) -> List[np.ndarray]:
    """Return ``[src_arr[rows] for rows in row_lists]`` reading ``src_arr`` once, in storage order.

    ``copy_rows`` decodes every source chunk touched by each batch, so a randomly scattered
    subset re-decodes most of the store once per batch (and again for every subset). Here the
    source is read in contiguous slabs of whole chunks (zarr decodes the chunks of a slab
    concurrently) and each slab is decoded once for all subsets; the requested rows are
    scattered into in-memory buffers, one per subset, in the requested order. Slabs holding
    no requested row are skipped. Memory: the buffers (sum of subset sizes) plus one slab.
    """
    n = int(src_arr.shape[0])
    chunk = int(src_arr.chunks[0]) if getattr(src_arr, "chunks", None) else n
    if slab_rows is None:
        slab_rows = default_slab_rows(src_arr)
    slab_rows = max(chunk, (int(slab_rows) // chunk) * chunk)
    tail = tuple(src_arr.shape[1:])
    out = [np.empty((len(r),) + tail, dtype=src_arr.dtype) for r in row_lists]
    if not row_lists:
        return out
    src_rows = np.concatenate([np.asarray(r, dtype=np.int64) for r in row_lists])
    owner = np.concatenate([np.full(len(r), k, dtype=np.int64) for k, r in enumerate(row_lists)])
    pos = np.concatenate([np.arange(len(r), dtype=np.int64) for r in row_lists])
    order = np.argsort(src_rows, kind="stable")
    src_rows, owner, pos = src_rows[order], owner[order], pos[order]
    if src_rows.size and (src_rows[0] < 0 or src_rows[-1] >= n):
        raise IndexError(f"row index out of range for {n} rows")
    bounds = np.searchsorted(src_rows, np.arange(0, n + slab_rows, slab_rows))
    for i, a in enumerate(range(0, n, slab_rows)):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if lo == hi:
            continue
        slab = np.asarray(src_arr[a:min(a + slab_rows, n)])
        for k in np.unique(owner[lo:hi]):
            sel = slice(lo, hi)
            m = owner[sel] == k
            out[k][pos[sel][m]] = slab[src_rows[sel][m] - a]
    return out


def default_slab_rows(arr) -> int:
    """Whole chunks per slab read, ~1 GiB, at least 8 chunks so zarr can decode them in parallel."""
    chunk = int(arr.chunks[0]) if getattr(arr, "chunks", None) else max(1, int(arr.shape[0]))
    per_row = int(np.prod(arr.shape[1:], dtype=np.int64)) * int(np.dtype(arr.dtype).itemsize) if arr.ndim > 1 else 8
    n_chunks = max(8, (1 << 30) // max(1, per_row * chunk))
    return chunk * n_chunks


def _derived_physics_hash(parent: str, rows: np.ndarray) -> str:
    import hashlib
    import json

    rows_sha = hashlib.sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest()
    payload = json.dumps({"parent_physics_hash": parent, "rows_sha256": rows_sha}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_row_subset(
    src,
    output_path: str,
    rows: np.ndarray,
    *,
    stage: str,
    source_path: str,
    stage_attrs: Optional[Mapping] = None,
    drop_subgroups=(),
    chunk_rows: Optional[int] = None,
    batch_rows: Optional[int] = None,
) -> dict:
    """Write ``src`` restricted to ``rows`` (in that order) as a new store.

    - atomic: built at a sibling tmp path and renamed into place;
    - row-aligned arrays are streamed, never loaded whole;
    - ``source_row_index`` maps each output row to the ORIGINAL store: an
      existing ``source_row_index`` is carried through, otherwise ``rows``;
    - root attrs are copied, then ``n_models`` / ``status_counts`` are
      recomputed, ``physics_hash`` becomes a hash of (parent, rows) and a
      ``lineage`` entry records this stage; ``stage_attrs`` also land in a
      ``<stage>`` subgroup.
    Returns {"sliced": [...], "copied": [...]} array names.
    """
    return write_row_subsets(src, [(output_path, rows, stage_attrs)], stage=stage, source_path=source_path,
                             drop_subgroups=drop_subgroups, chunk_rows=chunk_rows, batch_rows=batch_rows,
                             single_pass=False)


def write_row_subsets(
    src,
    outputs,
    *,
    stage: str,
    source_path: str,
    drop_subgroups=(),
    chunk_rows: Optional[int] = None,
    batch_rows: Optional[int] = None,
    single_pass: bool = True,
    slab_rows: Optional[int] = None,
) -> dict:
    """``write_row_subset`` for several ``(output_path, rows, stage_attrs)`` outputs at once.

    Each output is exactly what ``write_row_subset`` writes for it. With ``single_pass`` the
    numeric row-aligned arrays are gathered with ``gather_rows`` (one read of the source for all
    outputs; each output's rows are held in memory, one array at a time); otherwise each output
    is streamed separately with ``copy_rows`` (bounded memory, repeated source reads).
    """
    import json

    compression_kwargs, create_array, create_root_group = _zc.compression_kwargs, _zc.create_array, _zc.create_root_group
    write_fixed_string_scalar, zarr_store = _zc.write_fixed_string_scalar, _zc.zarr_store

    outputs = [(path, np.asarray(rows, dtype=np.int64), attrs) for path, rows, attrs in outputs]
    n_rows = row_count(src)
    row_names = [n for n in src.array_keys() if is_row_aligned(n, src[n], n_rows)]
    static_names = [n for n in src.array_keys() if n not in row_names]
    if batch_rows is None:
        batch_rows = default_batch_rows([src[n] for n in row_names])

    parent_attrs = dict(src.attrs)
    parent_hash = str(parent_attrs.get("physics_hash", "") or "")
    with contextlib.ExitStack() as stack:
        dsts = []
        for path, _rows, _attrs in outputs:
            tmp = stack.enter_context(atomic_output(path))
            dst = create_root_group(zarr_store(tmp), overwrite=True)
            copy_attrs(src, dst)
            dsts.append((tmp, dst))
        for name in row_names:
            arr = src[name]
            if name == "source_row_index":
                continue
            numeric = arr.dtype.kind not in "USOT"
            gathered = (gather_rows(arr, [rows for _p, rows, _a in outputs], slab_rows)
                        if numeric and single_pass else None)
            for k, ((_path, rows, _attrs), (_tmp, dst)) in enumerate(zip(outputs, dsts)):
                if not numeric:
                    new = _write_full(dst, name, np.asarray(arr[...])[rows])
                else:
                    row_chunk = chunk_rows or (arr.chunks[0] if arr.chunks else len(rows))
                    chunks = (max(1, min(len(rows), row_chunk)),) + tuple(arr.chunks[1:] if arr.chunks else arr.shape[1:])
                    new = create_array(dst, name, shape=(len(rows),) + tuple(arr.shape[1:]),
                                       dtype=arr.dtype, chunks=chunks, **compression_kwargs())
                    if gathered is not None:
                        if len(rows):
                            new[...] = gathered[k]
                        gathered[k] = None  # free each output's buffer as soon as it is written
                    else:
                        copy_rows(arr, new, rows, batch_rows)
                copy_attrs(arr, new)
            del gathered

        for (path, rows, stage_attrs), (tmp, dst) in zip(outputs, dsts):
            for name in static_names:
                if name == "physics_hash":
                    continue
                copy_array(src, dst, name)

            original = (np.asarray(src["source_row_index"][...], dtype=np.int64)[rows]
                        if "source_row_index" in row_names else rows)
            create_array(dst, "source_row_index", data=original, chunks=(max(1, min(len(rows), 1 << 20)),))
            copy_subgroups(src, dst, drop=set(drop_subgroups) | {stage})

            derived_hash = _derived_physics_hash(parent_hash, rows) if parent_hash else ""
            if "physics_hash" in static_names:
                write_fixed_string_scalar(dst, "physics_hash", derived_hash or parent_hash, min_width=64,
                                          compression_kw=compression_kwargs())
            entry = {"stage": stage, "source": os.path.abspath(source_path),
                     "source_rows": int(n_rows), "rows": int(len(rows)),
                     "parent_physics_hash": parent_hash, **dict(stage_attrs or {})}
            lineage = parent_attrs.get("lineage") or []
            if isinstance(lineage, str):
                lineage = json.loads(lineage)
            update = {"n_models": int(len(rows)), "lineage": list(lineage) + [entry]}
            if derived_hash:
                update["physics_hash"] = derived_hash
            if "status" in row_names:
                counts: dict = {}
                for s in decode_statuses(dst["status"][...]):
                    counts[s] = counts.get(s, 0) + 1
                update["status_counts"] = counts
            dst.attrs.update(update)
            grp = dst.create_group(stage)
            for key, value in (stage_attrs or {}).items():
                grp.attrs[key] = value
            try:
                import zarr

                zarr.consolidate_metadata(zarr_store(tmp))
            except Exception:
                pass
    return {"sliced": sorted(row_names), "copied": sorted(static_names)}
