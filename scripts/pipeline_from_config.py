#!/usr/bin/env python3
"""One-command pipeline: generate grid (CSV/Zarr) + synthesize spectra (Zarr).

This script is intentionally simple glue around:
- `scripts/generate_grid.py` (grid generation)
- `scripts/synthesize_spectra_from_zarr.py` (single-job multiprocessing synthesis)
- `scripts/synthesize_spectra_from_zarr_sharded.py` (sharded multiprocessing synthesis)

It exists to avoid managing separate config files for grid sampling vs synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def _abs_from(base_dir: str, maybe_relative: str | None) -> str | None:
    if not maybe_relative:
        return None
    return maybe_relative if os.path.isabs(maybe_relative) else os.path.abspath(os.path.join(base_dir, maybe_relative))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _strip_private_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private_keys(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip_private_keys(v) for v in obj]
    return obj


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _generate_grid(pipeline_cfg: Mapping[str, Any], config_dir: str) -> Dict[str, str]:
    """Generate grid outputs and return resolved paths."""
    grid_cfg = dict(pipeline_cfg.get("grid") or {})
    outputs = dict(pipeline_cfg.get("outputs") or {})

    grid_csv = _abs_from(config_dir, outputs.get("grid_csv"))
    grid_zarr = _abs_from(config_dir, outputs.get("grid_zarr"))
    if not grid_zarr:
        raise ValueError("outputs.grid_zarr is required")

    # Back-compat with generate_grid.py ML config format.
    if grid_csv:
        grid_cfg["output_csv"] = os.path.relpath(grid_csv, start=config_dir) if not os.path.isabs(outputs.get("grid_csv", "")) else grid_csv
    grid_cfg.setdefault("zarr", {})
    if isinstance(grid_cfg["zarr"], dict):
        grid_cfg["zarr"]["path"] = os.path.relpath(grid_zarr, start=config_dir) if not os.path.isabs(outputs.get("grid_zarr", "")) else grid_zarr

    # Import and run the existing implementation directly.
    sys.path.insert(0, SCRIPT_DIR)
    import generate_grid as gg  # type: ignore

    seed = grid_cfg.get("seed")
    rng = np.random.default_rng(seed)
    columns = gg._resolve_ml_sampling(grid_cfg, rng=rng)  # type: ignore[attr-defined]

    if grid_csv:
        csv_abs = _abs_from(config_dir, outputs.get("grid_csv"))  # recompute to be safe
        written = gg._write_csv_outputs(  # type: ignore[attr-defined]
            columns,
            csv_abs,
            compression=grid_cfg.get("csv_compression"),
            level=grid_cfg.get("csv_compression_level"),
        )
        # Keep stdout noise low but informative.
        print(f"Wrote grid CSV: {written[0]}")
        if len(written) > 1:
            print(f"Wrote compressed CSV: {written[1]}")

    zarr_cfg = grid_cfg.get("zarr") or {}
    chunks = int(zarr_cfg.get("chunks", 2048)) if isinstance(zarr_cfg, dict) else 2048
    compressor_cfg = (zarr_cfg.get("compressor") or {}) if isinstance(zarr_cfg, dict) else {}
    gg._write_zarr_from_columns(columns, grid_zarr, chunks=chunks, compressor_cfg=compressor_cfg)  # type: ignore[attr-defined]
    print(f"Wrote grid Zarr: {grid_zarr}")

    return {"grid_csv": grid_csv or "", "grid_zarr": grid_zarr}


def _write_temp_turbospectrum_config(pipeline_cfg: Mapping[str, Any], config_dir: str, scratch: Optional[str]) -> str:
    """Write a Turbospectrum config JSON file derived from pipeline_cfg."""
    ts_cfg = dict(pipeline_cfg.get("turbospectrum") or {})
    ts_cfg = _strip_private_keys(ts_cfg)

    # Ensure project_root is set to something usable on the current machine.
    project_root = ts_cfg.get("project_root")
    if not project_root or not os.path.isdir(str(project_root)):
        ts_cfg["project_root"] = REPO_ROOT

    # Place the temp config next to scratch if provided (good for HPC), else in config dir.
    base = scratch or os.path.join(config_dir, ".pipeline_tmp")
    os.makedirs(base, exist_ok=True)
    tmp_path = os.path.join(base, f"turbospectrum_config.{int(time.time())}.{os.getpid()}.json")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(ts_cfg, handle, indent=2, sort_keys=True)
    return tmp_path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to pipeline JSON config (see config_pipeline.example.json)")
    parser.add_argument("--skip-grid", action="store_true", help="Skip grid generation step")
    parser.add_argument("--skip-synthesis", action="store_true", help="Skip synthesis step")

    parser.add_argument(
        "--synthesis-mode",
        default="full",
        choices=["full", "sharded"],
        help="Run synthesis as one job (full) or as a shard (sharded).",
    )
    parser.add_argument("--shard-index", type=int, default=None, help="Shard index for sharded mode (0-based)")
    parser.add_argument("--shard-count", type=int, default=None, help="Total number of shards for sharded mode")

    parser.add_argument("--workers", type=int, default=None, help="Override worker process count")
    parser.add_argument("--scratch", default=None, help="Optional scratch dir (passed through to synthesis)")
    parser.add_argument(
        "--output",
        default=None,
        dest="output_tmp",
        metavar="TMP_PATH",
        help="Temp path for atomic write: write to TMP_PATH, then rename to final. Use for stability (avoids partial shards).",
    )
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg_dir = os.path.dirname(cfg_path)
    pipeline_cfg = _load_json(cfg_path)

    runtime = dict(pipeline_cfg.get("runtime") or {})
    outputs = dict(pipeline_cfg.get("outputs") or {})

    scratch = args.scratch or runtime.get("scratch")
    workers = args.workers or runtime.get("workers")
    chunk_rows = int(runtime.get("chunk_rows", 32))

    grid_zarr = _abs_from(cfg_dir, outputs.get("grid_zarr"))
    if not grid_zarr:
        raise ValueError("outputs.grid_zarr is required")

    if not args.skip_grid:
        _generate_grid(pipeline_cfg, config_dir=cfg_dir)
    else:
        print("Skipping grid generation.")

    if args.skip_synthesis:
        print("Skipping synthesis.")
        return

    ts_cfg_path = _write_temp_turbospectrum_config(pipeline_cfg, config_dir=cfg_dir, scratch=scratch)

    if args.synthesis_mode == "full":
        out_zarr = _abs_from(cfg_dir, outputs.get("spectra_zarr"))
        if not out_zarr:
            raise ValueError("outputs.spectra_zarr is required for synthesis-mode=full")
        _ensure_parent(out_zarr)
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "synthesize_spectra_from_zarr.py"),
            "--grid-zarr",
            grid_zarr,
            "--config",
            ts_cfg_path,
            "--output-zarr",
            out_zarr,
            "--chunk-rows",
            str(chunk_rows),
        ]
        if workers:
            cmd += ["--workers", str(workers)]
        if scratch:
            cmd += ["--scratch", str(scratch)]
        if args.output_tmp:
            cmd += ["--output-tmp", os.path.abspath(args.output_tmp)]
        _run(cmd)
        print(f"Wrote synthesized spectra Zarr: {out_zarr}")
        return

    # Sharded synthesis: user runs this script multiple times with different shard-index
    # (e.g. separate PBS jobs), writing separate outputs per shard.
    shard_index = args.shard_index
    shard_count = args.shard_count
    if shard_index is None or shard_count is None:
        raise ValueError("--shard-index and --shard-count are required for synthesis-mode=sharded")

    template = outputs.get("spectra_shard_template")
    if not template:
        raise ValueError("outputs.spectra_shard_template is required for synthesis-mode=sharded")
    out_zarr = _abs_from(cfg_dir, str(template).format(shard_index=shard_index))
    _ensure_parent(out_zarr)

    # Use the existing shard runner (available on HPC deployments).
    # Note: run_turbospectrum_shard.py uses strided sharding (index::count).
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_turbospectrum_shard.py"),
        "--grid-zarr",
        grid_zarr,
        "--config",
        ts_cfg_path,
        "--output-zarr",
        out_zarr,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]
    if workers:
        cmd += ["--workers", str(workers)]
    if scratch:
        cmd += ["--scratch", str(scratch)]
    if args.output_tmp:
        cmd += ["--output-tmp", os.path.abspath(args.output_tmp)]
    _run(cmd)
    print(f"Wrote shard spectra Zarr: {out_zarr}")


if __name__ == "__main__":
    main()

