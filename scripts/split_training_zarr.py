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
each split is built in a ``<name>.zarr.tmp-<pid>`` dir then renamed into
place, so a final ``.zarr`` is never partial. The source is opened read-only.
On rerun an existing split is skipped only if its ``split`` attrs (seed,
split count/index, shuffle mode, source rows) match; otherwise the run stops.
``--split-index`` values beyond the split count exit 0 (spare array tasks).

``--shuffle global`` (default) permutes individual rows, so every split needs
rows from nearly every source chunk. By default (``--single-pass``) all splits
of one invocation (every split, or those of ``--split-index``/``--split-stride``)
are gathered together in one read of the source, decoding each chunk once, and
held in memory one array at a time (``--max-buffer-gb`` caps that: the splits
are then written in groups, one source read per group). ``--no-single-pass``
streams each split separately in ``--batch-rows`` batches with bounded memory,
which re-decodes most source chunks for every batch and every split. Both write
identical stores. ``--shuffle block`` permutes whole source chunks and then
shuffles rows within each split, at the cost of chunk-level (not row-level)
split membership.

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

from zarr_compat import open_root_group  # noqa: E402
from zarr_rowstore import default_batch_rows, is_row_aligned, row_count, write_row_subsets  # noqa: E402

# Subgroups whose contents index into the *source* row order and would be
# silently wrong after shuffling.
_DROPPED_SUBGROUPS = {"splits"}

# split attrs that must match for an existing output to count as "this split".
_SPLIT_IDENTITY_KEYS = ("source_row_count", "split_index", "split_count", "seed", "shuffle")


def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


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


def _row_bytes(arr) -> int:
    try:
        per_row = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
        return per_row * int(np.dtype(arr.dtype).itemsize)
    except Exception:
        return 0


def _split_sizes(n_rows: int, num_splits: int) -> list[int]:
    base, rem = divmod(n_rows, num_splits)
    return [base + 1 if k < rem else base for k in range(num_splits)]


def _permutation(n_rows: int, seed: int, shuffle: str, block_rows: int) -> np.ndarray:
    """Row order whose consecutive slices become the splits.

    ``global``: one permutation of all rows (row-level random membership, but
    each split touches nearly every source chunk). ``block``: permute whole
    ``block_rows`` blocks (= source row chunks), so each split reads each
    source chunk once; rows are still shuffled within each split.
    """
    rng = np.random.default_rng(seed)
    if shuffle == "global":
        return rng.permutation(n_rows).astype(np.int64)
    n_blocks = -(-n_rows // block_rows)
    order = rng.permutation(n_blocks)
    return np.concatenate([np.arange(b * block_rows, min((b + 1) * block_rows, n_rows)) for b in order]).astype(np.int64)


def _check_existing_split(path: str, meta: dict) -> None:
    grp = open_root_group(path, mode="r")
    have = dict(grp["split"].attrs) if "split" in grp else {}
    diffs = {k: (have.get(k), meta[k]) for k in _SPLIT_IDENTITY_KEYS if have.get(k) != meta[k]}
    if diffs:
        raise ValueError(
            f"{path} exists but was written with different split settings {diffs} "
            "(existing, requested); use a new --output-dir or remove the old splits."
        )


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
    ap.add_argument("--shuffle", choices=("global", "block"), default="global",
                    help="global: row-level permutation (default). block: permute whole source "
                         "chunks (much less read amplification; rows still shuffled within a split)")
    ap.add_argument("--split-index", type=int, default=None,
                    help="Write only this split (0-based); for one-split-per-job submission")
    ap.add_argument("--split-stride", type=int, default=None,
                    help="With --split-index K, write splits K, K+stride, K+2*stride, ... "
                         "so T array tasks (stride=T) can cover more than T splits")
    ap.add_argument("--batch-rows", type=int, default=None,
                    help="Rows per streamed read batch (default: sized to ~512 MiB)")
    ap.add_argument("--chunk-rows", type=int, default=None,
                    help="Row chunk for output arrays (default: keep source chunking)")
    ap.add_argument("--single-pass", action=argparse.BooleanOptionalAction, default=True,
                    help="Gather this invocation's splits in one source read (default) instead of "
                         "streaming each split separately")
    ap.add_argument("--max-buffer-gb", type=float, default=48.0,
                    help="With --single-pass: max GiB of split rows held in memory for one array "
                         "(the widest); more splits are written in several groups (default 48)")
    ap.add_argument("--dry-run", action="store_true", help="Print the split plan and exit")
    args = ap.parse_args(argv)

    input_path = _norm(args.input)
    output_dir = _norm(args.output_dir)
    prefix = args.prefix or (os.path.basename(input_path.rstrip("/")).removesuffix(".zarr") + "_split")

    src = open_root_group(input_path, mode="r")
    n_rows = row_count(src)

    row_arrays, static_arrays = [], []
    for name in sorted(src.array_keys()):
        arr = src[name]
        (row_arrays if is_row_aligned(name, arr, n_rows) else static_arrays).append((name, arr))
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
    batch_rows = args.batch_rows or default_batch_rows([a for _n, a in row_arrays])
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
    widest = max(row_arrays, key=lambda na: _row_bytes(na[1]))[1]
    block_rows = int(widest.chunks[0]) if widest.chunks else 128
    perm = _permutation(n_rows, args.seed, args.shuffle, block_rows)
    offsets = np.concatenate(([0], np.cumsum(sizes)))

    if args.split_index is None:
        if args.split_stride is not None:
            raise ValueError("--split-stride requires --split-index")
        targets = range(num_splits)
    else:
        if args.split_index < 0:
            raise ValueError(f"--split-index must be >= 0, got {args.split_index}")
        if args.split_index >= num_splits:
            # More array tasks than splits: nothing to do for this one.
            print(f"--split-index {args.split_index} >= {num_splits} splits; nothing to write.")
            return 0
        if args.split_stride is not None and args.split_stride < 1:
            raise ValueError(f"--split-stride must be >= 1, got {args.split_stride}")
        targets = range(args.split_index, num_splits, args.split_stride or num_splits)
    widest_row_bytes = max(_row_bytes(a) for _n, a in row_arrays)
    budget = int(args.max_buffer_gb * 1024 ** 3) if args.single_pass else 0
    group, group_bytes = [], 0

    def _flush():
        nonlocal group, group_bytes
        if group:
            for k, path, rows, _meta in group:
                print(f"[{k}] writing {len(rows)} rows -> {path}")
            write_row_subsets(
                src, [(path, rows, meta) for _k, path, rows, meta in group],
                stage="split", source_path=input_path,
                drop_subgroups=_DROPPED_SUBGROUPS,
                chunk_rows=args.chunk_rows, batch_rows=batch_rows, single_pass=args.single_pass,
            )
        group, group_bytes = [], 0

    for k in targets:
        final_path = os.path.join(output_dir, f"{prefix}_{k:0{pad}d}.zarr")
        rows = perm[offsets[k] : offsets[k + 1]]
        meta = {
            "source_store": input_path,
            "source_row_count": int(n_rows),
            "rows_in_split": int(len(rows)),
            "split_index": int(k),
            "split_count": int(num_splits),
            "seed": int(args.seed),
            "size_basis": args.size_basis,
            "target_bytes": target_bytes,
            "shuffle": "global_permutation" if args.shuffle == "global" else "block_permutation",
        }
        if os.path.exists(final_path):
            _check_existing_split(final_path, meta)
            print(f"[{k}] exists, skipping (settings match): {final_path}")
            continue
        if args.shuffle == "block":
            # Whole chunks were assigned to this split; shuffle rows within it.
            rows = np.random.default_rng([args.seed, k]).permutation(rows)
        need = len(rows) * widest_row_bytes
        if group and group_bytes + need > budget:
            _flush()
        group.append((k, final_path, rows, meta))
        group_bytes += need
    _flush()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
