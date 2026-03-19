#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SHARD_REGEX = re.compile(r"^spectra_shard_(\d+)\.zarr$")


def _extract_ids(shard_dir: Path) -> set[int]:
    ids: set[int] = set()
    for path in shard_dir.glob("spectra_shard_*.zarr"):
        match = SHARD_REGEX.match(path.name)
        if match is None:
            continue
        ids.add(int(match.group(1)))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List missing shard IDs given expected shard count."
    )
    parser.add_argument(
        "--shard-dir",
        required=True,
        help="Directory containing spectra_shard_<id>.zarr outputs.",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        required=True,
        help="Expected shard count (IDs are assumed 0..expected_shards-1).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path (one shard ID per line). Use '-' for stdout.",
    )
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir).expanduser().resolve()
    if not shard_dir.exists():
        print(f"ERROR: shard directory does not exist: {shard_dir}", file=sys.stderr)
        return 2
    if args.expected_shards <= 0:
        print("ERROR: --expected-shards must be > 0", file=sys.stderr)
        return 2

    found = _extract_ids(shard_dir)
    expected = set(range(args.expected_shards))
    missing = sorted(expected - found)

    text = "\n".join(str(x) for x in missing)
    if args.output == "-":
        if text:
            print(text)
    else:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"{text}\n" if text else "", encoding="utf-8")

    print(
        f"[find-missing] expected={args.expected_shards} found={len(found)} missing={len(missing)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
