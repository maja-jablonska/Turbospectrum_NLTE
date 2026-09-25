#!/usr/bin/env python3
"""Report which shards of a strided layout are missing or invalid.

Used by the PBS pipeline for the resume list and before merge. Validity is
``zarr_rowstore.check_shard`` — the same predicate the shard runner uses to
decide skip vs recompute — so the three can no longer disagree.

    shard_status.py --shard-dir D --shard-count N --row-count R \
        --missing-out missing.txt [--drop-failed-rows] [--prune-stale]

``--prune-stale`` deletes shard stores whose global_index does not match the
current (shard_count, row_count) layout, or whose index is out of range.
Never run it from concurrent array tasks.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from zarr_rowstore import check_shard, expected_shard_indices

SHARD_RE = re.compile(r"^spectra_shard_(\d+)\.zarr$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--shard-count", type=int, required=True)
    ap.add_argument("--row-count", type=int, required=True)
    ap.add_argument("--missing-out", required=True, help="Write missing/invalid shard indices here, one per line")
    ap.add_argument("--drop-failed-rows", action="store_true")
    ap.add_argument("--prune-stale", action="store_true")
    args = ap.parse_args(argv)

    shard_dir = Path(args.shard_dir)
    stores = sorted(shard_dir.glob("spectra_shard_*.zarr")) if shard_dir.is_dir() else []
    valid: set[int] = set()
    reasons: dict[str, int] = {}
    pruned: list[str] = []
    for path in stores:
        match = SHARD_RE.match(path.name)
        if match is None:
            continue
        idx = int(match.group(1))
        if idx >= args.shard_count:
            reason = "outside_expected_range"
        else:
            reason = check_shard(
                str(path),
                expected_global_indices=expected_shard_indices(idx, args.shard_count, args.row_count),
                drop_failed_rows=args.drop_failed_rows,
            )
        if reason is None:
            valid.add(idx)
            continue
        key = reason.split(":")[0].split(" [")[0]
        reasons[key] = reasons.get(key, 0) + 1
        if args.prune_stale and reason in ("layout_mismatch", "outside_expected_range"):
            shutil.rmtree(path, ignore_errors=True)
            pruned.append(path.name)

    missing = sorted(set(range(args.shard_count)) - valid)
    out = Path(args.missing_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(f"{i}\n" for i in missing), encoding="utf-8")
    tmp.replace(out)

    print(
        f"[shard-status] expected={args.shard_count} valid={len(valid)} missing={len(missing)} "
        f"stores={len(stores)} invalid_reasons={reasons}"
        + (f" pruned={len(pruned)}" if args.prune_stale else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
