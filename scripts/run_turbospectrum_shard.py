#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

############################################
# Worker-global config
############################################

_WORKER_CONFIG = None


def _zarr_store(path: str):
    """Filesystem-backed Zarr store compatible with zarr v2/v3."""
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
    from zarr import storage as zstorage  # type: ignore
    if hasattr(zstorage, "DirectoryStore"):
        return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
    if hasattr(zstorage, "LocalStore"):
        return zstorage.LocalStore(path)  # type: ignore[attr-defined]
    raise AttributeError("Unsupported Zarr version: cannot find DirectoryStore/LocalStore")


def _open_root_for_write(path: str):
    """Create/overwrite a Zarr group (v2 or v3)."""
    store = _zarr_store(path)
    if hasattr(zarr, "group"):
        # zarr v3
        return zarr.group(store=store, overwrite=True, zarr_format=3)
    # zarr v2
    return zarr.open_group(store=store, mode="w")  # type: ignore[arg-type]


def _normalize_chunks(shape: tuple[int, ...], chunks: int | tuple[int, ...] | None) -> tuple[int, ...] | None:
    """Return a Zarr-friendly chunk tuple for a given shape."""
    if chunks is None:
        return None

    if isinstance(chunks, int):
        chunks_t = (int(chunks),) * len(shape)
    else:
        chunks_t = tuple(int(c) for c in chunks)
        if len(chunks_t) != len(shape):
            raise ValueError(f"chunks rank {len(chunks_t)} != shape rank {len(shape)}")

    out: list[int] = []
    for dim, ch in zip(shape, chunks_t, strict=True):
        ch = max(1, int(ch))
        if int(dim) > 0:
            ch = min(ch, int(dim))
        out.append(ch)
    return tuple(out)


def _write_array(root, name: str, data: Any, *, chunks: int | tuple[int, ...] | None = None) -> None:
    """Write a numeric/bytes array compatibly for Zarr v2/v3.

    Zarr v3's `create_dataset(..., data=...)` no longer infers shape, so we use
    `create_array(shape=..., dtype=...)` then assign when available.
    """
    arr = np.asarray(data)
    norm_chunks = _normalize_chunks(tuple(int(d) for d in arr.shape), chunks)

    if hasattr(root, "create_array"):
        # Provide a sensible default chunking if none is supplied (avoids one huge chunk).
        if norm_chunks is None:
            default = tuple(min(65536, max(1, int(d))) for d in arr.shape) if arr.ndim else ()
            norm_chunks = _normalize_chunks(tuple(int(d) for d in arr.shape), default if default else 1)
        za = root.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=norm_chunks)
        za[...] = arr
        return

    # zarr v2
    if norm_chunks is None:
        root.create_dataset(name, data=arr)
    else:
        root.create_dataset(name, data=arr, chunks=norm_chunks)


def _write_string_1d(root, name: str, values, chunks: int = 128):
    """Write 1D string array compatibly for zarr v2/v3."""
    vals = ["" if v is None else str(v) for v in values]
    if hasattr(root, "create_array"):
        import zarr.codecs as zc  # type: ignore
        from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

        arr = root.create_array(
            name,
            shape=(len(vals),),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            chunks=min(int(chunks), len(vals)) if len(vals) else 1,
        )
        arr[:] = vals
        return

    # zarr v2
    try:
        from numcodecs import VLenUTF8  # type: ignore

        root.array(name, vals, dtype=object, object_codec=VLenUTF8(), chunks=min(int(chunks), len(vals)) if len(vals) else 1)
    except Exception:
        arr = np.asarray(vals, dtype="U256")
        try:
            root.create_dataset(name, data=arr)
        except TypeError:
            root.create_dataset(name, shape=arr.shape, dtype=arr.dtype, data=arr)


def _init_worker(config):
    global _WORKER_CONFIG
    _WORKER_CONFIG = config

    # Prevent thread explosions
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


############################################
# Logging
############################################

def _configure_logging(level: str):
    logger = logging.getLogger("zarr_synthesis_sharded")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(processName)s] %(message)s"
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


############################################
# Adaptive batching
############################################

def _build_batches(indices: np.ndarray, batch_size: int):
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]


############################################
# Worker task
############################################

def _synthesis_task(batch):
    results = []

    for global_index, row_values in batch:
        cfg = copy.deepcopy(_WORKER_CONFIG)

        teff = int(row_values["teff"])
        logg = float(row_values["logg"])
        feh = float(row_values["feh"])
        turb = str(row_values["turb"])

        cfg.lambda_min = float(row_values["lam_min"])
        cfg.lambda_max = float(row_values["lam_max"])
        cfg.lambda_step = float(row_values["lam_step"])

        # If the grid provides per-row mode flags, honor them; otherwise, fall back
        # to whatever the Turbospectrum config requested.
        output_mode = row_values.get("output_mode")
        if output_mode is None:
            output_mode = "Intensity" if cfg.calculate_intensity else "Flux"
        calculation_mode = row_values.get("calculation_mode")
        if calculation_mode is None:
            calculation_mode = "NLTE" if cfg.nlte else "LTE"

        cfg.calculate_intensity = str(output_mode).lower() == "intensity"
        cfg.nlte = str(calculation_mode).lower() == "nlte"

        base_name = get_model_filename(teff, logg, feh, turb)

        start = time.perf_counter()
        result = run_single_synthesis(((teff, logg, feh, turb), cfg))
        duration = time.perf_counter() - start

        suffix = ".intensity.spec" if cfg.calculate_intensity else ".spec"
        spec_path = os.path.join(
            cfg.output_dir,
            f"{os.path.splitext(base_name)[0]}{suffix}"
        )

        spectrum = None
        if os.path.exists(spec_path):
            try:
                data = np.loadtxt(spec_path)

                flux = data[:, 1].astype(np.float32)
                cont = (
                    data[:, 2].astype(np.float32)
                    if data.shape[1] > 2
                    else np.full_like(flux, np.nan)
                )

                spectrum = (flux, cont)

            except Exception as exc:
                results.append({
                    "global_index": global_index,
                    "base_name": base_name,
                    "status": "error",
                    "message": str(exc),
                    "duration": duration,
                    "spectrum": None,
                })
                continue
        else:
            # If Turbospectrum reported success but did not produce the expected file,
            # promote this to an error so empty/NaN shards are diagnosable.
            if str(result.get("status", "")).lower() == "success":
                results.append({
                    "global_index": global_index,
                    "base_name": base_name,
                    "status": "error",
                    "message": f"Missing spectrum output: {spec_path}",
                    "duration": duration,
                    "spectrum": None,
                })
                continue

        results.append({
            "global_index": global_index,
            "base_name": base_name,
            "status": result["status"],
            "message": result["message"],
            "duration": duration,
            "spectrum": spectrum,
        })

    return results


############################################
# Main
############################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--grid-zarr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-zarr", required=True)

    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)

    parser.add_argument("--workers", type=int, default=None)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Initial adaptive batch size"
    )

    parser.add_argument("--scratch", default=None)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    ############################################
    # Fail fast
    ############################################

    if not os.path.exists(args.grid_zarr):
        raise FileNotFoundError(args.grid_zarr)

    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)

    ############################################
    # Thread safety in main
    ############################################

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    logger = _configure_logging(args.log_level)

    logger.info(
        "Running on host=%s cores=%s",
        os.uname().nodename,
        os.cpu_count()
    )

    ############################################
    # Idempotency
    ############################################

    if os.path.exists(args.output_zarr):
        logger.warning("Shard output already exists — skipping.")
        return

    ############################################
    # Auto scratch
    ############################################

    if args.scratch is None:
        jobfs = os.environ.get("PBS_JOBFS")
        if jobfs:
            args.scratch = jobfs
            logger.info("Using PBS_JOBFS scratch: %s", jobfs)

    ############################################
    # Load config
    ############################################

    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    with open(args.config) as f:
        cfg_data = json.load(f)

    # Guardrail: users sometimes accidentally pass the ML sampling config here.
    if isinstance(cfg_data, dict) and any(k in cfg_data for k in ("bounds", "num_samples", "output_csv")):
        raise ValueError(
            "The provided --config looks like an ML sampling config (e.g. config_ml_sampling.json), "
            "not a Turbospectrum synthesis config. Use config_sample_comprehensive.json (or another "
            "Turbospectrum config with paths/executables/synthesis_parameters)."
        )

    cfg_data = _normalize_config_dict(cfg_data, default_project_root=project_root)

    accepted = {fld.name for fld in dataclasses.fields(TurbospectrumConfig)}
    cfg_data = {k: v for k, v in cfg_data.items() if k in accepted}

    # Ensure project_root is usable on the current machine.
    cfg_project_root = cfg_data.get("project_root")
    if not cfg_project_root or not os.path.isdir(str(cfg_project_root)):
        logger.warning("Config project_root=%r is not a directory; using detected project_root=%s", cfg_project_root, project_root)
        cfg_data["project_root"] = project_root

    config = TurbospectrumConfig(**cfg_data)

    if args.scratch:
        scratch = os.path.abspath(args.scratch)

        config.tmp_dir = os.path.join(scratch, "tmp")
        config.log_dir = os.path.join(scratch, "logs")
        config.output_dir = os.path.join(scratch, "spectra")
        config.model_opac_dir = os.path.join(scratch, "opac")

    ensure_directories(config)
    config.linelist_file_path = create_linelist_file(config)

    ############################################
    # Load grid
    ############################################

    grid = zarr.open_group(args.grid_zarr, mode="r")

    row_count = grid["teff"].shape[0]

    indices = np.arange(row_count)[
        args.shard_index::args.shard_count
    ]

    logger.info(
        "Grid rows=%d shard=%d/%d rows_in_shard=%d",
        row_count,
        args.shard_index,
        args.shard_count,
        len(indices),
    )

    # If the shard is empty (common when shard_count > row_count), write an empty
    # but valid shard store and exit cleanly.
    if len(indices) == 0:
        logger.warning("Shard has 0 rows; writing empty shard and exiting.")

        wavelengths = np.asarray([], dtype=np.float64)
        if row_count > 0 and all(k in grid for k in ("lam_min", "lam_max", "lam_step")):
            lam_min0 = float(np.asarray(grid["lam_min"][0]))
            lam_max0 = float(np.asarray(grid["lam_max"][0]))
            lam_step0 = float(np.asarray(grid["lam_step"][0]))
            if lam_step0 > 0:
                npts = int(round((lam_max0 - lam_min0) / lam_step0)) + 1
                wavelengths = lam_min0 + lam_step0 * np.arange(npts, dtype=np.float64)

        root = _open_root_for_write(args.output_zarr)
        _write_array(root, "wavelength", wavelengths, chunks=min(65536, max(1, wavelengths.size)))
        _write_array(root, "global_index", np.asarray([], dtype=np.int64), chunks=1)
        _write_array(
            root,
            "flux",
            np.full((0, wavelengths.size), np.nan, np.float32),
            chunks=(1, min(65536, max(1, wavelengths.size))),
        )
        _write_array(
            root,
            "continuum",
            np.full((0, wavelengths.size), np.nan, np.float32),
            chunks=(1, min(65536, max(1, wavelengths.size))),
        )
        _write_string_1d(root, "status", [], chunks=1)
        _write_string_1d(root, "message", [], chunks=1)

        root.attrs.update({
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "grid": os.path.abspath(args.grid_zarr),
            "note": "empty shard (no rows assigned)",
        })
        logger.info("Empty shard written to %s", args.output_zarr)
        return

    ############################################
    # Columns
    ############################################

    turb_col = "turbvel" if "turbvel" in grid else "t_value"

    base_cols = ["teff", "logg", "feh", "lam_min", "lam_max", "lam_step", turb_col]
    # Optional columns can drive runtime behavior (Intensity/NLTE).
    optional_cols = [k for k in ("output_mode", "calculation_mode", "grid_version") if k in grid]

    columns = {k: np.asarray(grid[k][indices]) for k in base_cols + optional_cols}

    columns["turb"] = columns.pop(turb_col)

    ############################################
    # Workers
    ############################################

    worker_count = (
        args.workers
        if args.workers
        else determine_worker_count(config)
    )

    ############################################
    # Build adaptive batches
    ############################################

    tasks = []
    global_to_local = {int(g): i for i, g in enumerate(indices.tolist())}

    for batch_indices in _build_batches(indices, args.batch_size):
        batch = []

        for global_i in batch_indices:
            local_i = global_to_local[int(global_i)]

            row_values = {k: columns[k][local_i] for k in columns}

            batch.append((int(global_i), row_values))

        tasks.append(batch)

    ############################################
    # Run synthesis
    ############################################

    wavelengths = np.linspace(
        columns["lam_min"][0],
        columns["lam_max"][0],
        int(round(
            (columns["lam_max"][0] - columns["lam_min"][0])
            / columns["lam_step"][0]
        )) + 1,
    )

    fluxes = np.full((len(indices), len(wavelengths)), np.nan, np.float32)
    continua = np.full_like(fluxes, np.nan)

    statuses = ["pending"] * len(indices)
    messages = [""] * len(indices)

    logger.info("Starting synthesis with %d workers", worker_count)

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(config,),
    ) as executor:

        futures = {executor.submit(_synthesis_task, task): task for task in tasks}

        done = 0

        for future in as_completed(futures):
            task_batch = futures[future]
            try:
                batch_results = future.result()
            except Exception as exc:  # noqa: BLE001
                # If a worker crashes, don't abort the whole shard. Mark those rows
                # so the output shard isn't "empty" and we have diagnostics.
                err_msg = f"Worker crashed: {exc}"
                logger.exception(err_msg)
                for global_i, _row_values in task_batch:
                    idx = global_to_local.get(int(global_i))
                    if idx is None:
                        continue
                    statuses[idx] = "exception"
                    messages[idx] = err_msg
                continue

            for result in batch_results:
                idx = global_to_local[int(result["global_index"])]

                statuses[idx] = result["status"]
                messages[idx] = result["message"]

                if result["spectrum"]:
                    fluxes[idx], continua[idx] = result["spectrum"]

                done += 1

                logger.info(
                    "[%d/%d] global=%d %s (%.2fs)",
                    done,
                    len(indices),
                    result["global_index"],
                    result["status"],
                    result["duration"],
                )

    ############################################
    # Write shard
    ############################################

    logger.info(
        "Writing shard output: %s (rows=%d wl=%d)",
        args.output_zarr,
        len(indices),
        len(wavelengths),
    )
    root = _open_root_for_write(args.output_zarr)

    _write_array(root, "wavelength", wavelengths, chunks=min(65536, max(1, wavelengths.size)))
    _write_array(root, "global_index", indices.astype(np.int64), chunks=max(1, min(2048, len(indices))))
    _write_array(root, "flux", fluxes, chunks=(1, min(65536, max(1, wavelengths.size))))
    _write_array(root, "continuum", continua, chunks=(1, min(65536, max(1, wavelengths.size))))
    _write_string_1d(root, "status", statuses, chunks=max(1, min(256, len(statuses))))
    _write_string_1d(root, "message", messages, chunks=max(1, min(256, len(messages))))
    # Optional: base model filenames for debugging/traceability.
    if any(isinstance(m, str) and m for m in messages):
        try:
            base_names = np.asarray([str(get_model_filename(int(columns["teff"][i]), float(columns["logg"][i]), float(columns["feh"][i]), str(columns["turb"][i]))) for i in range(len(indices))], dtype="U256")
            _write_string_1d(root, "base_name", base_names.tolist(), chunks=max(1, min(256, len(base_names))))
        except Exception:
            pass

    # Write per-row metadata columns for later merging/QA (best-effort).
    for name, values in columns.items():
        if name in {"lam_min", "lam_max", "lam_step"}:
            continue
        try:
            _write_array(root, name, np.asarray(values), chunks=max(1, min(2048, len(values))))
        except Exception:
            pass
    for name in ("lam_min", "lam_max", "lam_step"):
        if name in columns:
            try:
                _write_array(root, name, np.asarray(columns[name]), chunks=max(1, min(2048, len(columns[name]))))
            except Exception:
                pass

    root.attrs.update({
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "grid": os.path.abspath(args.grid_zarr),
    })

    logger.info("Shard written to %s", args.output_zarr)


if __name__ == "__main__":
    main()