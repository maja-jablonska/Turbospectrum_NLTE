"""Shared helpers for row-oriented Zarr stores (shards, merged, derived).

Keep store-level plumbing (atomic publishing, status decoding) here so the
merge / filter / split / subsample stages do not each re-implement it.
"""
from __future__ import annotations

import contextlib
import os
import shutil
from typing import Iterator, List

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
