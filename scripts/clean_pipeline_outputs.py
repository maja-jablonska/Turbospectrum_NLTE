#!/usr/bin/env python3
"""
Clean pipeline-generated outputs from a pipeline config.

Default behavior is DRY RUN. Use --apply to actually delete files/directories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, List


def _abs_from(base_dir: Path, maybe_relative: str | None) -> Path | None:
    if not maybe_relative:
        return None
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _is_glob_pattern(path_text: str) -> bool:
    return any(ch in path_text for ch in "*?[]")


def _template_to_glob(template: str) -> str:
    # Convert e.g. ".../spectra_shard_{shard_index}.zarr" -> ".../spectra_shard_*.zarr"
    return re.sub(r"\{[^{}]+\}", "*", template)


def _safe_to_delete(path: Path) -> bool:
    text = str(path).strip()
    if text in {"", "/", "."}:
        return False
    # Require at least two path components to avoid accidents.
    return len(path.resolve().parts) >= 3


@dataclass(frozen=True)
class Target:
    kind: str  # "path" | "glob"
    value: str
    reason: str


def _collect_targets(cfg: dict, cfg_dir: Path, run_root: Path | None, include_pipeline_tmp: bool) -> List[Target]:
    outputs = dict(cfg.get("outputs") or {})
    targets: List[Target] = []

    grid_csv = _abs_from(cfg_dir, outputs.get("grid_csv"))
    if grid_csv is not None:
        targets.append(Target("path", str(grid_csv), "grid_csv"))
        targets.append(Target("path", str(Path(f"{grid_csv}.gz")), "grid_csv gzip sibling"))
        targets.append(Target("path", str(Path(f"{grid_csv}.zst")), "grid_csv zstd sibling"))

    grid_index = _abs_from(cfg_dir, outputs.get("grid_index_parquet"))
    if grid_index is not None:
        targets.append(Target("path", str(grid_index), "grid_index_parquet"))

    grid_zarr = _abs_from(cfg_dir, outputs.get("grid_zarr"))
    if grid_zarr is not None:
        targets.append(Target("path", str(grid_zarr), "grid_zarr"))

    spectra_zarr = _abs_from(cfg_dir, outputs.get("spectra_zarr"))
    if spectra_zarr is not None:
        targets.append(Target("path", str(spectra_zarr), "spectra_zarr"))

    shard_template = outputs.get("spectra_shard_template")
    shard_template_abs = _abs_from(cfg_dir, shard_template)
    if shard_template_abs is not None:
        shard_glob = _template_to_glob(str(shard_template_abs))
        targets.append(Target("glob", shard_glob, "spectra_shard_template matches"))

    if include_pipeline_tmp:
        targets.append(Target("path", str((cfg_dir / ".pipeline_tmp").resolve()), ".pipeline_tmp"))

    if run_root is not None:
        rr = run_root.resolve()
        targets.extend(
            [
                Target("path", str(rr / "next_shard.txt"), "run_root shard counter"),
                Target("path", str(rr / "next_work_item.txt"), "run_root pending-work counter"),
                Target("path", str(rr / "pending_shards.txt"), "run_root pending-shards list"),
                Target("path", str(rr / "counter.lock"), "run_root lock file"),
                Target("path", str(rr / "missing_shards.txt"), "run_root missing-shards list"),
                Target("glob", str(rr / "tmp" / "pbs_*"), "run_root PBS temp dirs"),
            ]
        )

    # De-duplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    deduped: List[Target] = []
    for t in targets:
        key = (t.kind, t.value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def _existing_paths_for_target(target: Target) -> List[Path]:
    if target.kind == "path":
        p = Path(target.value)
        return [p] if p.exists() else []
    if target.kind == "glob":
        return [Path(p) for p in sorted(glob(target.value))]
    raise ValueError(f"Unsupported target kind: {target.kind}")


def _delete_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/pipeline/config_pipeline.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional run root to also clean scheduler state files (next_shard/next_work_item/pending_shards/counter.lock/missing_shards + tmp/pbs_*).",
    )
    parser.add_argument(
        "--no-pipeline-tmp",
        action="store_true",
        help="Do not include <config_dir>/.pipeline_tmp in cleanup targets.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched paths. Without this flag, performs a dry run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only summary lines.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg_dir = cfg_path.parent

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    run_root = Path(args.run_root).resolve() if args.run_root else None
    targets = _collect_targets(
        cfg=cfg,
        cfg_dir=cfg_dir,
        run_root=run_root,
        include_pipeline_tmp=not args.no_pipeline_tmp,
    )

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] Config: {cfg_path}")
    if run_root is not None:
        print(f"[{mode}] Run root: {run_root}")

    total_matches = 0
    deleted = 0
    skipped_unsafe = 0

    for target in targets:
        matches = _existing_paths_for_target(target)
        total_matches += len(matches)
        if not args.quiet:
            marker = f"{target.kind:>4}"
            print(f"[{mode}] {marker} {target.value} ({target.reason}) -> {len(matches)} match(es)")

        if not args.apply:
            continue

        for path in matches:
            if not _safe_to_delete(path):
                skipped_unsafe += 1
                print(f"[{mode}] SKIP unsafe path: {path}")
                continue
            if _delete_path(path):
                deleted += 1

    print(f"[{mode}] Targets: {len(targets)}")
    print(f"[{mode}] Existing matches: {total_matches}")
    if args.apply:
        print(f"[{mode}] Deleted: {deleted}")
        if skipped_unsafe:
            print(f"[{mode}] Skipped unsafe: {skipped_unsafe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
