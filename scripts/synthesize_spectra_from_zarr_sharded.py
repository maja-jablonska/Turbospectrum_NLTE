#!/usr/bin/env python3
"""Shard-aware batch synthesis from a Zarr parameter grid (PBS/SLURM friendly).

Why this exists:
- Writing to the *same* Zarr store from multiple array jobs is unsafe.
- Large grids + huge ProcessPool submissions can be inefficient.

This script:
- Reads an input grid Zarr (produced by the grid generator)
- Selects only a subset ("shard") of rows
- Synthesizes spectra for that shard using multiprocessing
- Writes one output Zarr *per shard* (safe for array jobs)

You can later merge shards (e.g. with a small post-processing step).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_turbospectrum import (  # noqa: E402
    TurbospectrumConfig,
    _normalize_config_dict,
    create_linelist_file,
    determine_worker_count,
    ensure_directories,
    get_model_filename,
    run_single_synthesis,
)


def _configure_logging(log_level: str) -> logging.Logger:
    logger = logging.getLogger("zarr_synthesis_sharded")
    if logger.handlers:
        return logger
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _zarr_store(path: str):
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
    from zarr import storage as zstorage  # type: ignore

    if hasattr(zstorage, "DirectoryStore"):
        return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
    if hasattr(zstorage, "LocalStore"):
        return zstorage.LocalStore(path)  # type: ignore[attr-defined]
    raise AttributeError("Unsupported Zarr version: cannot find DirectoryStore/LocalStore")


def _zarr_compression_kwargs(zarr_compressor_cfg: Mapping[str, Any]):
    if not zarr_compressor_cfg:
        return {}

    cname = zarr_compressor_cfg.get("cname", "zstd")
    clevel = int(zarr_compressor_cfg.get("clevel", 5))
    shuffle_enabled = bool(zarr_compressor_cfg.get("shuffle", True))

    try:
        import zarr.codecs as zc  # type: ignore

        if hasattr(zc, "BloscCodec") and hasattr(zc, "BloscShuffle"):
            shuffle = zc.BloscShuffle.bitshuffle if shuffle_enabled else None
            return {"compressors": [zc.BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)]}
    except Exception:
        pass

    from numcodecs import Blosc  # type: ignore

    return {
        "compressor": Blosc(
            cname=cname,
            clevel=clevel,
            shuffle=Blosc.BITSHUFFLE if shuffle_enabled else Blosc.NOSHUFFLE,
        )
    }


def _load_ts_config(config_path: str, project_root: str) -> TurbospectrumConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg_data = json.load(handle)
    cfg_data = _normalize_config_dict(cfg_data, default_project_root=project_root)
    accepted_fields = {fld.name for fld in dataclasses.fields(TurbospectrumConfig)}
    cfg_data = {k: v for k, v in cfg_data.items() if k in accepted_fields}
    if "project_root" not in cfg_data:
        cfg_data["project_root"] = project_root
    return TurbospectrumConfig(**cfg_data)


def _grid_row_count(grid_root) -> int:
    if "teff" not in grid_root:
        raise KeyError("Grid Zarr missing required column 'teff'")
    return int(grid_root["teff"].shape[0])


def _split_indices(row_count: int, shard_index: int, shard_count: int, mode: str) -> np.ndarray:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, shard_count); got {shard_index} of {shard_count}")
    if row_count <= 0:
        return np.asarray([], dtype=np.int64)

    mode = (mode or "block").lower().strip()
    if mode == "stride":
        return np.arange(shard_index, row_count, shard_count, dtype=np.int64)

    if mode == "block":
        base = row_count // shard_count
        rem = row_count % shard_count
        start = shard_index * base + min(shard_index, rem)
        size = base + (1 if shard_index < rem else 0)
        stop = start + size
        return np.arange(start, stop, dtype=np.int64)

    raise ValueError("Unsupported shard mode. Use 'block' or 'stride'.")


def _read_column(grid_root, name: str, indices: np.ndarray) -> np.ndarray:
    if name not in grid_root:
        raise KeyError(f"Grid Zarr missing required column '{name}'")
    arr = grid_root[name]
    if indices.size == 0:
        return np.asarray([], dtype=np.array(arr[:]).dtype)

    # Fast path: contiguous slice (block sharding).
    if indices.size > 0 and np.all(np.diff(indices) == 1):
        start = int(indices[0])
        stop = int(indices[-1]) + 1
        return np.asarray(arr[start:stop])

    # Fancy indexing fallback (stride sharding).
    try:
        return np.asarray(arr.oindex[indices])  # type: ignore[attr-defined]
    except Exception:
        full = np.asarray(arr[:])
        return full[indices]


def _expected_wavelengths(grid_root) -> Tuple[np.ndarray, int, float, float, float]:
    for key in ("lam_min", "lam_max", "lam_step"):
        if key not in grid_root:
            raise KeyError(f"Grid Zarr missing required column '{key}'")

    lam_min = float(np.asarray(grid_root["lam_min"][0]))
    lam_max = float(np.asarray(grid_root["lam_max"][0]))
    lam_step = float(np.asarray(grid_root["lam_step"][0]))
    if lam_step <= 0:
        raise ValueError("lam_step must be positive")
    count = int(round((lam_max - lam_min) / lam_step)) + 1
    wavelengths = lam_min + lam_step * np.arange(count, dtype=np.float64)
    return wavelengths, count, lam_min, lam_max, lam_step


def _choose_turb_column(grid_root) -> str:
    for col in ("turbvel", "t_value"):
        if col in grid_root:
            return col
    raise KeyError("Grid Zarr must include either 'turbvel' or 't_value'")


def _synthesis_task(args) -> Dict[str, Any]:
    global_index, row_values, base_config = args

    teff = int(row_values["teff"])
    logg = float(row_values["logg"])
    feh = float(row_values["feh"])
    turb_str = str(row_values["turb"]).strip()
    lam_min = float(row_values["lam_min"])
    lam_max = float(row_values["lam_max"])
    lam_step = float(row_values["lam_step"])
    output_mode = str(row_values.get("output_mode", "Flux"))
    calculation_mode = str(row_values.get("calculation_mode", "LTE"))

    cfg = base_config
    cfg.lambda_min = lam_min
    cfg.lambda_max = lam_max
    cfg.lambda_step = lam_step
    cfg.calculate_intensity = output_mode.lower() == "intensity"
    cfg.nlte = calculation_mode.lower() == "nlte"

    base_name = get_model_filename(teff, logg, feh, turb_str)
    start = time.perf_counter()
    result = run_single_synthesis(((teff, logg, feh, turb_str), cfg))
    duration = time.perf_counter() - start

    suffix = ".intensity.spec" if cfg.calculate_intensity else ".spec"
    spec_path = os.path.join(cfg.output_dir, f"{os.path.splitext(base_name)[0]}{suffix}")
    spectrum = None
    if os.path.exists(spec_path):
        try:
            data = np.loadtxt(spec_path)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(f"Unexpected spectrum shape {data.shape}")
            flux = data[:, 1].astype(np.float32)
            cont = (
                data[:, 2].astype(np.float32)
                if data.shape[1] > 2
                else np.full_like(flux, np.nan, dtype=np.float32)
            )
            spectrum = (flux, cont)
        except Exception as exc:  # noqa: BLE001
            return {
                "global_index": int(global_index),
                "base_name": base_name,
                "status": "error",
                "message": f"Failed to read spectrum {spec_path}: {exc}",
                "duration": duration,
                "spectrum": None,
            }

    return {
        "global_index": int(global_index),
        "base_name": base_name,
        "status": result["status"],
        "message": result["message"],
        "duration": duration,
        "spectrum": spectrum,
    }


def _write_shard_zarr(
    output_path: str,
    wavelengths: np.ndarray,
    global_indices: np.ndarray,
    fluxes: np.ndarray,
    continua: np.ndarray,
    row_columns: Mapping[str, np.ndarray],
    statuses: Sequence[str],
    messages: Sequence[str],
    compression_cfg: Mapping[str, Any],
    chunk_rows: int,
    shard_index: int,
    shard_count: int,
    shard_mode: str,
    grid_path: str,
    config_path: str,
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    store = _zarr_store(output_path)
    root = zarr.group(store=store, overwrite=True, zarr_format=3)

    compression_kwargs = _zarr_compression_kwargs(compression_cfg)
    import zarr.codecs as zc  # type: ignore
    from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

    chunk_shape = (min(chunk_rows, fluxes.shape[0]), fluxes.shape[1]) if fluxes.size else (1, wavelengths.size)
    root.create_array("wavelength", data=wavelengths, chunks=wavelengths.shape, **compression_kwargs)
    root.create_array("global_index", data=global_indices.astype(np.int64), chunks=min(chunk_rows, len(global_indices)), **compression_kwargs)
    root.create_array("flux", data=fluxes, chunks=chunk_shape, **compression_kwargs)
    root.create_array("continuum", data=continua, chunks=chunk_shape, **compression_kwargs)

    for name, values in row_columns.items():
        if values.dtype.kind in {"U", "S", "O"}:
            arr = root.create_array(
                name,
                shape=values.shape,
                dtype=VariableLengthUTF8(),
                serializer=zc.VLenUTF8Codec(),
                chunks=min(chunk_rows, len(values)),
                **compression_kwargs,
            )
            arr[:] = values.astype(str)
        else:
            root.create_array(name, data=values, chunks=min(chunk_rows, len(values)), **compression_kwargs)

    for field_name, values in {"status": statuses, "message": messages}.items():
        arr = root.create_array(
            field_name,
            shape=(len(values),),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            chunks=min(chunk_rows, len(values)),
            **compression_kwargs,
        )
        arr[:] = list(values)

    root.attrs.update(
        {
            "creation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wavelength_count": int(wavelengths.size),
            "chunk_rows": int(chunk_rows),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "shard_mode": str(shard_mode),
            "grid_zarr": os.path.abspath(grid_path),
            "config": os.path.abspath(config_path),
        }
    )
    logger.info("Wrote shard spectra to %s (rows=%d, wl=%d)", os.path.abspath(output_path), fluxes.shape[0], fluxes.shape[1] if fluxes.ndim == 2 else wavelengths.size)


def _resolve_array_env(default_index: Optional[int], default_count: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    # PBS
    pbs_idx = os.environ.get("PBS_ARRAY_INDEX")
    if pbs_idx is not None:
        try:
            idx = int(pbs_idx)
        except ValueError:
            idx = default_index
        return idx, default_count

    # SLURM
    slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_idx is not None:
        try:
            idx = int(slurm_idx)
        except ValueError:
            idx = default_index
        return idx, default_count

    return default_index, default_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-zarr", required=True, help="Input Zarr grid path")
    parser.add_argument("--config", required=True, help="Path to Turbospectrum JSON config")
    parser.add_argument("--output-zarr", required=True, help="Output Zarr path for this shard")
    parser.add_argument("--shard-index", type=int, default=None, help="Shard index (0-based). If omitted, read from PBS/SLURM env.")
    parser.add_argument("--shard-count", type=int, default=None, help="Total number of shards (array size)")
    parser.add_argument("--shard-mode", default="block", choices=["block", "stride"], help="How to split rows across shards")
    parser.add_argument("--workers", type=int, default=None, help="Override worker process count")
    parser.add_argument("--scratch", default=None, help="Optional node-local scratch dir to reduce shared FS I/O")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--chunk-rows", type=int, default=32, help="Zarr chunking along the sample dimension")
    parser.add_argument("--compressor", default=None, help="JSON string describing compressor options (cname, clevel, shuffle)")
    args = parser.parse_args()

    logger = _configure_logging(args.log_level)
    t0 = time.perf_counter()

    shard_index, shard_count = _resolve_array_env(args.shard_index, args.shard_count)
    if shard_index is None or shard_count is None:
        raise ValueError("Must provide --shard-index and --shard-count (or set PBS/SLURM array env vars)")

    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    config = _load_ts_config(args.config, project_root=project_root)

    # Optional: use node-local scratch for temp/log/opac/spec I/O to reduce contention.
    if args.scratch:
        scratch = os.path.abspath(args.scratch)
        os.makedirs(scratch, exist_ok=True)
        config.tmp_dir = os.path.join(scratch, "tmp")
        config.log_dir = os.path.join(scratch, "logs")
        config.output_dir = os.path.join(scratch, "spectra")
        config.model_opac_dir = os.path.join(scratch, "opac")

    ensure_directories(config)
    config.linelist_file_path = create_linelist_file(config)

    grid_store = _zarr_store(args.grid_zarr)
    grid_root = zarr.open_group(store=grid_store, mode="r")
    row_count = _grid_row_count(grid_root)
    indices = _split_indices(row_count, shard_index=shard_index, shard_count=shard_count, mode=args.shard_mode)

    wavelengths, expected_points, lam_min0, lam_max0, lam_step0 = _expected_wavelengths(grid_root)
    logger.info(
        "Grid rows=%d shard=%d/%d mode=%s shard_rows=%d wavelengths=%d (%.3f-%.3f step %.4f)",
        row_count,
        shard_index,
        shard_count,
        args.shard_mode,
        indices.size,
        expected_points,
        lam_min0,
        lam_max0,
        lam_step0,
    )

    if indices.size == 0:
        logger.warning("Shard has 0 rows; writing empty output.")

    turb_col = _choose_turb_column(grid_root)
    required_cols = ["teff", "logg", "feh", "lam_min", "lam_max", "lam_step", turb_col]
    optional_cols = [col for col in ("output_mode", "calculation_mode", "grid_version") if col in grid_root]

    columns: Dict[str, np.ndarray] = {col: _read_column(grid_root, col, indices) for col in required_cols + optional_cols}
    columns["turb"] = columns.pop(turb_col)

    # Worker count (prefer explicit --workers)
    worker_count = int(args.workers) if args.workers and args.workers > 0 else determine_worker_count(config)
    worker_count = max(1, worker_count)

    compressor_cfg: Dict[str, Any] = {}
    if args.compressor:
        compressor_cfg = json.loads(args.compressor)

    # Prepare output arrays for this shard.
    shard_rows = int(indices.size)
    fluxes = np.full((shard_rows, expected_points), np.nan, dtype=np.float32)
    continua = np.full_like(fluxes, np.nan)
    statuses: List[str] = ["pending"] * shard_rows
    messages: List[str] = [""] * shard_rows

    # Build tasks (global index preserved!)
    tasks = []
    for local_i, global_i in enumerate(indices.tolist()):
        row_values = {
            "teff": columns["teff"][local_i],
            "logg": columns["logg"][local_i],
            "feh": columns["feh"][local_i],
            "turb": columns["turb"][local_i],
            "lam_min": columns["lam_min"][local_i],
            "lam_max": columns["lam_max"][local_i],
            "lam_step": columns["lam_step"][local_i],
        }
        for opt in ("output_mode", "calculation_mode"):
            if opt in columns:
                row_values[opt] = columns[opt][local_i]
        tasks.append((int(global_i), row_values, config))

    logger.info("Starting synthesis with workers=%d", worker_count)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_synthesis_task, task): (local_idx, task[0]) for local_idx, task in enumerate(tasks)}
        done = 0
        for future in as_completed(futures):
            local_idx, global_idx = futures[future]
            done += 1
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                statuses[local_idx] = "exception"
                messages[local_idx] = str(exc)
                logger.exception("Row global=%d crashed: %s", global_idx, exc)
                continue

            statuses[local_idx] = result["status"]
            messages[local_idx] = result["message"]
            if result.get("spectrum"):
                fluxes[local_idx], continua[local_idx] = result["spectrum"]

            logger.info(
                "[%d/%d] global=%d %s %s (%.2fs) - %s",
                done,
                shard_rows,
                global_idx,
                str(result["status"]).upper(),
                result["base_name"],
                result["duration"],
                result["message"],
            )

    # Write shard output (include per-row metadata columns).
    row_columns = {k: v for k, v in columns.items() if k not in {"lam_min", "lam_max", "lam_step"}}
    row_columns["lam_min"] = columns["lam_min"]
    row_columns["lam_max"] = columns["lam_max"]
    row_columns["lam_step"] = columns["lam_step"]

    _write_shard_zarr(
        output_path=args.output_zarr,
        wavelengths=wavelengths,
        global_indices=indices,
        fluxes=fluxes,
        continua=continua,
        row_columns=row_columns,
        statuses=statuses,
        messages=messages,
        compression_cfg=compressor_cfg,
        chunk_rows=int(args.chunk_rows),
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        shard_mode=str(args.shard_mode),
        grid_path=args.grid_zarr,
        config_path=args.config,
        logger=logger,
    )
    logger.info("Done in %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()

