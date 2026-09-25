#!/usr/bin/env python3
"""Rechunk a merged spectra store into the ML training layout, in place of
``rechunk_worker.py`` + ``merge_rechunked.py`` (one copy of the data, not two).

    rechunk_zarr.py all      -i merged.zarr -o ml.zarr            # single job
    rechunk_zarr.py init     -i merged.zarr -o ml.zarr --tasks N  # then, per task:
    rechunk_zarr.py task     -i merged.zarr -o ml.zarr --tasks N --task-index K
    rechunk_zarr.py finalize -i merged.zarr -o ml.zarr --tasks N

``init`` creates ``<output>.building`` with every array at its final shape and
chunking, and copies everything that is not row-aligned (wavelength,
param_names, provenance, string columns). Each ``task`` fills a disjoint,
chunk-aligned row range of the row-aligned numeric arrays — no two tasks touch
the same chunk — and drops a completion marker. ``finalize`` refuses to
publish unless every task's marker is present, then copies root attrs, adds a
``lineage`` entry and renames ``<output>.building`` to ``<output>``. A killed
task is simply rerun; nothing downstream sees a partial store.

Chunk constants (keep aligned with CLAUDE.md "Chunk constants"):
ROW_CHUNK=128 rows x LAMBDA_CHUNK=8192 wavelength points for 2D arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zarr_compat import compression_kwargs, create_array, create_root_group, open_root_group, zarr_store  # noqa: E402
from zarr_rowstore import copy_array, copy_attrs, is_row_aligned, row_count  # noqa: E402

ROW_CHUNK = 128
LAMBDA_CHUNK = 8192
ONE_D_CHUNK = 1024
# Task ranges are multiples of this, so every task owns whole chunks of every
# row-aligned array (2D: ROW_CHUNK, 1D: ONE_D_CHUNK).
_TASK_ALIGN = int(np.lcm(ROW_CHUNK, ONE_D_CHUNK))
_MARKER_DIR = ".rechunk_tasks"
_SKIP_GROUPS = {"provenance"}  # copied verbatim, never row-sliced


def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _building(output: str) -> str:
    return output + ".building"


def _row_arrays(group, n_rows: int, prefix: str = ""):
    """(path, array) for numeric row-aligned arrays, recursing into subgroups."""
    for name in sorted(group.array_keys()):
        arr = group[name]
        if is_row_aligned(name, arr, n_rows) and arr.dtype.kind not in "USOT":
            yield prefix + name, arr
    for name in sorted(group.group_keys()):
        if name not in _SKIP_GROUPS:
            yield from _row_arrays(group[name], n_rows, prefix + name + "/")


def _target_chunks(arr) -> tuple:
    if arr.ndim == 1:
        return (ONE_D_CHUNK,)
    return (ROW_CHUNK, min(LAMBDA_CHUNK, arr.shape[1])) + tuple(arr.shape[2:])


def task_ranges(n_rows: int, tasks: int) -> list[tuple[int, int]]:
    per = -(-n_rows // tasks)
    per = -(-per // _TASK_ALIGN) * _TASK_ALIGN
    return [(lo, min(lo + per, n_rows)) for lo in range(0, n_rows, per)] or [(0, 0)]


def init(src, src_path: str, output: str) -> None:
    building = _building(output)
    if os.path.exists(output):
        raise FileExistsError(f"Output already exists: {output}")
    if os.path.exists(building):
        shutil.rmtree(building)
    n_rows = row_count(src)
    dst = create_root_group(zarr_store(building), overwrite=True)

    row_paths = {p for p, _a in _row_arrays(src, n_rows)}

    def _mirror(src_group, dst_group, prefix=""):
        for name in sorted(src_group.array_keys()):
            arr = src_group[name]
            if prefix + name in row_paths:
                new = create_array(dst_group, name, shape=arr.shape, dtype=arr.dtype,
                                   chunks=_target_chunks(arr), **compression_kwargs())
                copy_attrs(arr, new)
            else:
                copy_array(src_group, dst_group, name)  # small / string / static
        for name in sorted(src_group.group_keys()):
            sub = dst_group.create_group(name)
            copy_attrs(src_group[name], sub)
            _mirror(src_group[name], sub, prefix + name + "/")

    _mirror(src, dst)
    os.makedirs(os.path.join(building, _MARKER_DIR), exist_ok=True)
    print(f"[rechunk] init: {building} rows={n_rows} row-aligned arrays={sorted(row_paths)}")


def run_task(src, output: str, tasks: int, task_index: int) -> None:
    building = _building(output)
    n_rows = row_count(src)
    ranges = task_ranges(n_rows, tasks)
    if task_index >= len(ranges):
        print(f"[rechunk] task {task_index}: no rows (only {len(ranges)} ranges); nothing to do.")
        return
    lo, hi = ranges[task_index]
    dst = zarr.open_group(building, mode="r+")
    for path, arr in _row_arrays(src, n_rows):
        out = dst[path]
        step = out.chunks[0]
        for i in range(lo, hi, step):
            j = min(i + step, hi)
            out[i:j] = arr[i:j]
    with open(os.path.join(building, _MARKER_DIR, f"task_{task_index}"), "w", encoding="utf-8") as fh:
        fh.write(f"{lo} {hi}\n")
    print(f"[rechunk] task {task_index}: rows [{lo}:{hi}] done")


def finalize(src, src_path: str, output: str, tasks: int) -> None:
    building = _building(output)
    n_rows = row_count(src)
    expected = len(task_ranges(n_rows, tasks))
    marker_dir = os.path.join(building, _MARKER_DIR)
    done = set(os.listdir(marker_dir)) if os.path.isdir(marker_dir) else set()
    missing = [k for k in range(expected) if f"task_{k}" not in done]
    if missing:
        raise SystemExit(f"[rechunk] finalize: {len(missing)} task(s) incomplete: {missing[:20]}; rerun them first.")

    dst = zarr.open_group(building, mode="r+")
    copy_attrs(src, dst)
    lineage = dict(src.attrs).get("lineage") or []
    if isinstance(lineage, str):
        lineage = json.loads(lineage)
    dst.attrs["lineage"] = list(lineage) + [{
        "stage": "rechunk", "source": src_path, "source_rows": int(n_rows), "rows": int(n_rows),
        "row_chunk": ROW_CHUNK, "lambda_chunk": LAMBDA_CHUNK, "tasks": int(expected),
    }]
    shutil.rmtree(marker_dir)
    zarr.consolidate_metadata(zarr_store(building))
    os.rename(building, output)
    print(f"[rechunk] finalize: wrote {output}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("plan", "init", "task", "finalize", "all"))
    ap.add_argument("-i", "--input", required=True, help="Merged spectra .zarr")
    ap.add_argument("-o", "--output", required=True, help="Destination ML-layout .zarr")
    ap.add_argument("--tasks", type=int, default=1, help="Number of parallel tasks (array size)")
    ap.add_argument("--task-index", type=int, default=None, help="0-based task (mode=task)")
    args = ap.parse_args(argv)

    src_path, output = _norm(args.input), _norm(args.output)
    src = open_root_group(src_path, mode="r")
    n_rows = row_count(src)
    if args.mode == "plan":
        for k, (lo, hi) in enumerate(task_ranges(n_rows, args.tasks)):
            print(f"task {k}: rows [{lo}:{hi}]")
        return 0
    if args.mode in ("init", "all"):
        init(src, src_path, output)
    if args.mode == "task":
        if args.task_index is None:
            ap.error("mode=task requires --task-index")
        run_task(src, output, args.tasks, args.task_index)
    if args.mode == "all":
        for k in range(len(task_ranges(n_rows, args.tasks))):
            run_task(src, output, args.tasks, k)
    if args.mode in ("finalize", "all"):
        finalize(src, src_path, output, args.tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
