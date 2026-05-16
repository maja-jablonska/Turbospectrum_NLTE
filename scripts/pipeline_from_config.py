#!/usr/bin/env python3
"""One-command pipeline: generate grid (CSV/Zarr) + synthesize spectra (Zarr).

This script is intentionally simple glue around:
- `scripts/generate_grid.py` (grid generation)
- `scripts/synthesize_spectra_from_zarr.py` (single-job multiprocessing synthesis)
- `scripts/run_turbospectrum_shard.py` (sharded synthesis)

It exists to avoid managing separate config files for grid sampling vs synthesis.
"""

from __future__ import annotations

import argparse
import copy
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


def _abs_output_from(run_root: str, cfg_dir: str, maybe_relative: str | None) -> str | None:
    """Resolve an output path.

    Absolute paths are returned as-is.  Dot-relative paths (../../...) resolve
    from *cfg_dir* (the directory that contains the config file).  Bare relative
    paths (outputs/...) resolve from *run_root* so that configs can use short
    names like ``outputs/grids/grid.zarr`` and have them land under the run root.
    """
    if not maybe_relative:
        return None
    if os.path.isabs(maybe_relative):
        return maybe_relative
    if not maybe_relative.startswith((".", "..")):
        return os.path.abspath(os.path.join(run_root, maybe_relative)) if run_root else os.path.abspath(os.path.join(cfg_dir, maybe_relative))
    return os.path.abspath(os.path.join(cfg_dir, maybe_relative))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _strip_private_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private_keys(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip_private_keys(v) for v in obj]
    return obj


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _normalize_synthesis_overrides(overrides: Mapping[str, Any], cfg_dir: str) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(overrides))

    project_root = normalized.get("project_root")
    if isinstance(project_root, str) and project_root.strip():
        normalized["project_root"] = _abs_from(cfg_dir, project_root)

    paths = normalized.get("paths")
    if isinstance(paths, dict):
        for key in ("model_atmosphere_path", "linelist_path", "output_dir", "log_dir", "tmp_dir"):
            value = paths.get(key)
            if isinstance(value, str) and value.strip():
                paths[key] = _abs_from(cfg_dir, value)
        paths_nlte = paths.get("nlte")
        if isinstance(paths_nlte, dict):
            for key in ("nlte_info_file", "model_atom_file", "departure_file"):
                value = paths_nlte.get(key)
                if isinstance(value, str) and value.strip():
                    paths_nlte[key] = _abs_from(cfg_dir, value)

    nlte = normalized.get("nlte")
    if isinstance(nlte, dict):
        for key in ("nlte_info_file", "model_atom_file", "departure_file"):
            value = nlte.get(key)
            if isinstance(value, str) and value.strip():
                nlte[key] = _abs_from(cfg_dir, value)

    return normalized


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _generate_grid(pipeline_cfg: Mapping[str, Any], config_dir: str) -> Dict[str, str]:
    """Generate grid outputs and return resolved paths."""
    grid_cfg = dict(pipeline_cfg.get("grid") or {})
    outputs = dict(pipeline_cfg.get("outputs") or {})

    top_nlte_ascii = pipeline_cfg.get("nlte_ascii_departures")
    if isinstance(top_nlte_ascii, Mapping) and "nlte_ascii_departures" not in grid_cfg:
        grid_cfg["nlte_ascii_departures"] = dict(top_nlte_ascii)
    grid_synthesis = dict(grid_cfg.get("synthesis") or {})
    ts_synthesis = dict(((pipeline_cfg.get("turbospectrum") or {}).get("synthesis_parameters")) or {})
    ts_mu_sampling = ts_synthesis.get("mu_sampling") or {}
    if (
        isinstance(ts_mu_sampling, Mapping)
        and str(grid_synthesis.get("output_mode", "Flux")).strip().lower() == "intensity"
        and str(ts_mu_sampling.get("mode", "none")).strip().lower() in {"nearest", "target"}
    ):
        grid_synthesis["mu_sampling"] = dict(ts_mu_sampling)
        grid_cfg["synthesis"] = grid_synthesis

    run_root = _abs_from(config_dir, outputs.get("run_root")) or ""
    grid_csv = _abs_output_from(run_root, config_dir, outputs.get("grid_csv"))
    grid_zarr = _abs_output_from(run_root, config_dir, outputs.get("grid_zarr"))
    grid_index_parquet = _abs_output_from(run_root, config_dir, outputs.get("grid_index_parquet"))
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
    if not grid_index_parquet:
        grid_index_parquet = gg._default_index_parquet_path(  # type: ignore[attr-defined]
            grid_zarr_path=grid_zarr,
            grid_csv_path=grid_csv,
        )

    seed = grid_cfg.get("seed")
    rng = np.random.default_rng(seed)
    columns = gg._resolve_ml_sampling(grid_cfg, rng=rng)  # type: ignore[attr-defined]

    if grid_csv:
        csv_abs = _abs_output_from(run_root, config_dir, outputs.get("grid_csv"))  # recompute to be safe
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
    gg._write_index_parquet_from_columns(columns, grid_index_parquet)  # type: ignore[attr-defined]
    print(f"Wrote grid Zarr: {grid_zarr}")
    print(f"Wrote grid parameter index: {grid_index_parquet}")

    return {
        "grid_csv": grid_csv or "",
        "grid_zarr": grid_zarr,
        "grid_index_parquet": grid_index_parquet,
    }


def _write_temp_turbospectrum_config(pipeline_cfg: Mapping[str, Any], config_dir: str, scratch: Optional[str]) -> str:
    """Write a Turbospectrum config JSON file derived from pipeline_cfg."""
    ts_cfg = dict(pipeline_cfg.get("turbospectrum") or {})
    ts_cfg = _strip_private_keys(ts_cfg)

    base_cfg_path_raw = ts_cfg.get("config")
    overrides = ts_cfg.pop("overrides", None)
    if base_cfg_path_raw:
        base_cfg_path = _abs_from(config_dir, str(base_cfg_path_raw))
        ts_cfg = _strip_private_keys(_load_json(base_cfg_path))
        if overrides:
            if not isinstance(overrides, Mapping):
                raise ValueError("turbospectrum.overrides must be a JSON object")
            ts_cfg = _deep_merge(ts_cfg, _normalize_synthesis_overrides(overrides, config_dir))

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


def _find_flag_value(cmd: list[str], flag: str) -> tuple[Optional[int], Optional[str]]:
    for idx, token in enumerate(cmd[:-1]):
        if token == flag:
            return idx + 1, cmd[idx + 1]
    return None, None


def _replace_flag_value(cmd: list[str], flag: str, value: str) -> list[str]:
    updated = list(cmd)
    idx, _ = _find_flag_value(updated, flag)
    if idx is None:
        updated.extend([flag, value])
    else:
        updated[idx] = value
    return updated


def _is_sigkill_returncode(returncode: int) -> bool:
    return int(returncode) in (-9, 137)


def _run(cmd: list[str]) -> None:
    current_cmd = list(cmd)

    while True:
        try:
            subprocess.run(current_cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            _, workers_raw = _find_flag_value(current_cmd, "--workers")
            try:
                workers = int(str(workers_raw)) if workers_raw is not None else None
            except ValueError:
                workers = None

            if _is_sigkill_returncode(exc.returncode) and workers and workers > 1:
                next_workers = max(1, workers // 2)
                if next_workers == workers:
                    raise
                print(
                    "Synthesis subprocess was killed with SIGKILL while using "
                    f"{workers} workers; retrying with {next_workers}.",
                    file=sys.stderr,
                )
                current_cmd = _replace_flag_value(current_cmd, "--workers", str(next_workers))
                continue

            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline JSON config (see configs/pipeline/config_pipeline.example.json)",
    )
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
    parser.add_argument("--no-scratch", action="store_true", help="Disable automatic scratch override (e.g. to use paths from config)")
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
    chunk_rows = int(runtime.get("chunk_rows", 128))

    run_root = _abs_from(cfg_dir, outputs.get("run_root")) or ""
    grid_zarr = _abs_output_from(run_root, cfg_dir, outputs.get("grid_zarr"))
    if not grid_zarr:
        raise ValueError("outputs.grid_zarr is required")

    if not args.skip_grid:
        _generate_grid(pipeline_cfg, config_dir=cfg_dir)
    else:
        print("Skipping grid generation.")

    if args.skip_synthesis:
        print("Synthesis deferred for this invocation (--skip-synthesis).")
        return

    ts_cfg_path = _write_temp_turbospectrum_config(pipeline_cfg, config_dir=cfg_dir, scratch=scratch)

    if args.synthesis_mode == "full":
        out_zarr = _abs_output_from(run_root, cfg_dir, outputs.get("spectra_zarr"))
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
        if args.no_scratch:
            cmd.append("--no-scratch")
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
    out_zarr = _abs_output_from(run_root, cfg_dir, str(template).format(shard_index=shard_index))
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
    if args.no_scratch:
        cmd.append("--no-scratch")
    if args.output_tmp:
        cmd += ["--output-tmp", os.path.abspath(args.output_tmp)]
    _run(cmd)
    print(f"Wrote shard spectra Zarr: {out_zarr}")


if __name__ == "__main__":
    main()
