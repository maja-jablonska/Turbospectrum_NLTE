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

        cfg.calculate_intensity = (
            row_values.get("output_mode", "Flux").lower() == "intensity"
        )
        cfg.nlte = (
            row_values.get("calculation_mode", "LTE").lower() == "nlte"
        )

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
                    "status": "error",
                    "message": str(exc),
                    "duration": duration,
                    "spectrum": None,
                })
                continue

        results.append({
            "global_index": global_index,
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

    cfg_data = _normalize_config_dict(cfg_data, project_root)

    accepted = {fld.name for fld in dataclasses.fields(TurbospectrumConfig)}
    cfg_data = {k: v for k, v in cfg_data.items() if k in accepted}

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

    for batch_indices in _build_batches(indices, args.batch_size):
        batch = []

        for global_i in batch_indices:
            local_i = np.where(indices == global_i)[0][0]

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

        futures = {
            executor.submit(_synthesis_task, task): task
            for task in tasks
        }

        done = 0

        for future in as_completed(futures):
            batch_results = future.result()

            for result in batch_results:
                idx = np.where(indices == result["global_index"])[0][0]

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

    root = zarr.open_group(args.output_zarr, mode="w")

    root.create_dataset("wavelength", data=wavelengths)
    root.create_dataset("flux", data=fluxes)
    root.create_dataset("continuum", data=continua)
    root.create_dataset("status", data=np.array(statuses, dtype="U32"))

    root.attrs.update({
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "grid": os.path.abspath(args.grid_zarr),
    })

    logger.info("Shard written to %s", args.output_zarr)


if __name__ == "__main__":
    main()