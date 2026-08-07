#!/usr/bin/env python3
"""Split a training .zarr store into ~equal-size, fully shuffled .zarr chunks.

Ingests a full training (or merged spectra) zarr store and re-emits it as N
self-contained stores of roughly ``--target-gb`` each. A single seeded global
permutation of all rows is computed up front; consecutive slices of that
permutation become the splits. This guarantees:

* every source row appears in exactly one output split (no loss, no dupes)
* split membership is random across the whole dataset
* row order *within* each split is also shuffled
* the same ``--seed`` always reproduces the same assignment (safe to run
  splits as independent, restartable jobs via ``--split-index``)

Arrays whose first dimension matches the source row count (spectra/flux,
labels/params, teff, ...) are sliced to the split's rows; arrays that are not
row-aligned (e.g. ``wavelength``, ``param_names``) are copied verbatim into
every split, as are group attrs and subgroups such as ``provenance``. A
pre-existing ``splits/`` subgroup (train/val/test indices) is dropped because
its indices are meaningless after shuffling; each output instead records its
``source_row_index`` so rows can always be traced back.

Rows are streamed in batches — the store is never loaded into memory — and
each split is written to a unique ``<name>.zarr.tmp-<pid>`` dir then renamed
into place, so a final ``.zarr`` is never partial. The script never deletes
anything: the source is opened read-only, existing final outputs are skipped
on rerun, and stale tmp dirs from crashed jobs are left for manual cleanup.

Usage:
    # Plan only (row counts, split sizes) without writing anything
    python3 scripts/split_training_zarr.py -i train.zarr -o out_dir/ --dry-run

    # Write all splits sequentially, ~10 GiB each on disk
    python3 scripts/split_training_zarr.py -i train.zarr -o out_dir/

    # One split per PBS job (deterministic under a fixed seed)
    python3 scripts/split_training_zarr.py -i train.zarr -o out_dir/ --split-index $PBS_ARRAY_INDEX
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zarr_compat import (  # noqa: E402
    zarr_store,
    open_root_group,
    create_root_group,
    create_array,
    compression_kwargs,
    write_string_array,
)

# Subgroups whose contents index into the *source* row order and would be
# silently wrong after shuffling.
_DROPPED_SUBGROUPS = {"splits"}

# Target bytes held in memory per read batch (per widest row-aligned array).
_BATCH_TARGET_BYTES = 512 * 1024 ** 2


def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _row_count(group) -> int:
    """Infer the number of rows from common row-indexed arrays."""
    for name in ("spectra", "flux", "labels", "teff", "status"):
        if name in group:
            return int(group[name].shape[0])
    lengths: dict[int, int] = {}
    for name in group.array_keys():
        shp = group[name].shape
        if shp:
            lengths[shp[0]] = lengths.get(shp[0], 0) + 1
    if not lengths:
        raise ValueError("Could not infer row count: store has no arrays.")
    return max(lengths.items(), key=lambda kv: kv[1])[0]


def _logical_nbytes(arr) -> int:
    try:
        return int(np.prod(arr.shape, dtype=np.int64)) * int(np.dtype(arr.dtype).itemsize)
    except Exception:
        return 0


def _disk_nbytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fname))
            except OSError:
                pass
    return total


def _copy_group_attrs(src, dst) -> None:
    for key, value in dict(src.attrs).items():
        dst.attrs[key] = value


def _copy_subgroups(src, dst) -> None:
    """Recursively copy non-row-aligned subgroups (e.g. provenance) verbatim."""
    for name in src.group_keys():
        if name in _DROPPED_SUBGROUPS:
            print(f"  dropping subgroup '{name}/' (source-row indices are invalid after shuffling)")
            continue
        sub_src = src[name]
        sub_dst = dst.create_group(name)
        _copy_group_attrs(sub_src, sub_dst)
        for arr_name in sub_src.array_keys():
            _copy_array_verbatim(sub_src, sub_dst, arr_name)
        _copy_subgroups(sub_src, sub_dst)


def _copy_array_verbatim(src_group, dst_group, name: str):
    arr = src_group[name]
    data = arr[...]
    chunks = arr.chunks if arr.chunks and data.ndim >= 1 else None
    try:
        new = create_array(dst_group, name, data=data, chunks=chunks, **compression_kwargs())
    except Exception:
        if np.asarray(data).dtype.kind not in ("U", "S", "O", "T"):
            raise
        new = write_string_array(dst_group, name, [str(v) for v in np.asarray(data).ravel().tolist()])
    _copy_group_attrs(arr, new)
    return new


def _copy_rows_streamed(src_arr, dst_arr, rows: np.ndarray, batch_rows: int) -> None:
    """Copy ``src_arr[rows]`` into ``dst_arr[0:len(rows)]`` in batches.

    Each batch is read with sorted source indices (chunk-friendly on the
    source), then reordered back to the shuffled order before the contiguous
    destination write.
    """
    trailing = (slice(None),) * (src_arr.ndim - 1)
    for start in range(0, len(rows), batch_rows):
        idx = rows[start : start + batch_rows]
        order = np.argsort(idx, kind="stable")
        data = src_arr.oindex[(idx[order],) + trailing]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        dst_arr[start : start + len(idx)] = data[inv]


def _row_bytes(arr) -> int:
    try:
        per_row = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
        return per_row * int(np.dtype(arr.dtype).itemsize)
    except Exception:
        return 0


def _batch_rows_default(row_arrays) -> int:
    widest = max((_row_bytes(a) for _name, a in row_arrays), default=1) or 1
    return int(min(4096, max(64, _BATCH_TARGET_BYTES // widest)))


def _split_sizes(n_rows: int, num_splits: int) -> list[int]:
    base, rem = divmod(n_rows, num_splits)
    return [base + 1 if k < rem else base for k in range(num_splits)]


def _write_split(
    src,
    row_arrays,
    static_arrays,
    rows: np.ndarray,
    final_path: str,
    *,
    batch_rows: int,
    chunk_rows: int | None,
    meta: dict,
) -> None:
    # Unique per-process tmp name: never deletes or clobbers another job's
    # in-progress write. A crashed run leaves its tmp dir behind for manual
    # inspection/cleanup rather than being removed automatically.
    tmp_path = f"{final_path}.tmp-{os.getpid()}"
    if os.path.exists(tmp_path):
        raise FileExistsError(f"Refusing to overwrite existing tmp dir: {tmp_path}")

    dst = create_root_group(zarr_store(tmp_path), overwrite=False)
    _copy_group_attrs(src, dst)

    for name, arr in row_arrays:
        row_chunk = chunk_rows or (arr.chunks[0] if arr.chunks else len(rows))
        chunks = (min(len(rows), row_chunk),) + tuple(arr.chunks[1:] if arr.chunks else arr.shape[1:])
        new = create_array(
            dst,
            name,
            shape=(len(rows),) + tuple(arr.shape[1:]),
            dtype=arr.dtype,
            chunks=chunks,
            **compression_kwargs(),
        )
        _copy_group_attrs(arr, new)
        _copy_rows_streamed(arr, new, rows, batch_rows)

    for name, _arr in static_arrays:
        _copy_array_verbatim(src, dst, name)

    create_array(dst, "source_row_index", data=rows.astype(np.int64), chunks=(len(rows),))
    _copy_subgroups(src, dst)

    grp = dst.create_group("split")
    for key, value in meta.items():
        grp.attrs[key] = value

    os.rename(tmp_path, final_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="Source training .zarr store")
    ap.add_argument("-o", "--output-dir", required=True, help="Directory for the split .zarr stores")
    ap.add_argument("--prefix", default=None,
                    help="Output name prefix (default: <input basename>_split)")
    ap.add_argument("--target-gb", type=float, default=10.0,
                    help="Target size per split in GiB (default 10)")
    ap.add_argument("--size-basis", choices=("disk", "logical"), default="disk",
                    help="Interpret --target-gb as on-disk (compressed, default) or logical bytes")
    ap.add_argument("--rows-per-split", type=int, default=None,
                    help="Override: exact row budget per split (ignores --target-gb)")
    ap.add_argument("--num-splits", type=int, default=None,
                    help="Override: exact number of splits (ignores --target-gb)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the global shuffle (default 0); keep fixed across jobs")
    ap.add_argument("--split-index", type=int, default=None,
                    help="Write only this split (0-based); for one-split-per-job submission")
    ap.add_argument("--split-stride", type=int, default=None,
                    help="With --split-index K, write splits K, K+stride, K+2*stride, ... "
                         "so T array tasks (stride=T) can cover more than T splits")
    ap.add_argument("--batch-rows", type=int, default=None,
                    help="Rows per streamed read batch (default: sized to ~512 MiB)")
    ap.add_argument("--chunk-rows", type=int, default=None,
                    help="Row chunk for output arrays (default: keep source chunking)")
    ap.add_argument("--dry-run", action="store_true", help="Print the split plan and exit")
    args = ap.parse_args(argv)

    input_path = _norm(args.input)
    output_dir = _norm(args.output_dir)
    prefix = args.prefix or (os.path.basename(input_path.rstrip("/")).removesuffix(".zarr") + "_split")

    src = open_root_group(input_path, mode="r")
    n_rows = _row_count(src)

    row_arrays, static_arrays = [], []
    for name in sorted(src.array_keys()):
        arr = src[name]
        (row_arrays if arr.shape and arr.shape[0] == n_rows else static_arrays).append((name, arr))
    if not row_arrays:
        raise ValueError(f"No row-aligned arrays found (inferred {n_rows} rows).")

    logical_row_bytes = sum(_row_bytes(a) for _n, a in row_arrays)
    logical_total = sum(_logical_nbytes(a) for _n, a in row_arrays) or 1
    if args.size_basis == "disk":
        ratio = _disk_nbytes(input_path) / max(1, sum(_logical_nbytes(src[n]) for n in src.array_keys()))
        per_row_bytes = max(1, int(logical_row_bytes * ratio))
    else:
        per_row_bytes = max(1, logical_row_bytes)

    target_bytes = int(args.target_gb * 1024 ** 3)
    if args.num_splits:
        num_splits = args.num_splits
    elif args.rows_per_split:
        num_splits = math.ceil(n_rows / args.rows_per_split)
    else:
        num_splits = max(1, math.ceil(n_rows * per_row_bytes / target_bytes))
    num_splits = min(num_splits, n_rows)
    sizes = _split_sizes(n_rows, num_splits)
    batch_rows = args.batch_rows or _batch_rows_default(row_arrays)
    pad = max(3, len(str(num_splits - 1)))

    print(f"Source: {input_path}")
    print(f"  rows={n_rows}, row-aligned arrays={len(row_arrays)}, "
          f"verbatim arrays={len(static_arrays)}")
    print(f"  ~{per_row_bytes} B/row ({args.size_basis} basis) -> "
          f"{num_splits} splits of {sizes[0]}..{sizes[-1]} rows "
          f"(~{sizes[0] * per_row_bytes / 1024**3:.2f} GiB each)")
    print(f"  seed={args.seed}, batch_rows={batch_rows}")
    if args.dry_run:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_rows).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes)))

    if args.split_index is None:
        if args.split_stride is not None:
            raise ValueError("--split-stride requires --split-index")
        targets = range(num_splits)
    else:
        if not (0 <= args.split_index < num_splits):
            raise ValueError(f"--split-index {args.split_index} out of range [0, {num_splits})")
        if args.split_stride is not None and args.split_stride < 1:
            raise ValueError(f"--split-stride must be >= 1, got {args.split_stride}")
        targets = range(args.split_index, num_splits, args.split_stride or num_splits)
    for k in targets:
        final_path = os.path.join(output_dir, f"{prefix}_{k:0{pad}d}.zarr")
        if os.path.exists(final_path):
            print(f"[{k}] exists, skipping: {final_path}")
            continue
        rows = perm[offsets[k] : offsets[k + 1]]
        print(f"[{k}] writing {len(rows)} rows -> {final_path}")
        _write_split(
            src,
            row_arrays,
            static_arrays,
            rows,
            final_path,
            batch_rows=batch_rows,
            chunk_rows=args.chunk_rows,
            meta={
                "source_store": input_path,
                "source_row_count": int(n_rows),
                "rows_in_split": int(len(rows)),
                "split_index": int(k),
                "split_count": int(num_splits),
                "seed": int(args.seed),
                "size_basis": args.size_basis,
                "target_bytes": target_bytes,
                "shuffle": "global_permutation",
            },
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
