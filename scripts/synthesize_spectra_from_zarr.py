#!/usr/bin/env python3
"""Batch synthesize spectra from a Zarr parameter grid.

This script reads a Zarr (v3) grid produced by ``sample_machine_learning_grid.py``,
uses Turbospectrum to synthesize spectra for each entry, and stores the fluxes
in a new Zarr v3 store. It is designed for HPC environments: worker counts are
auto-detected from scheduler variables, logging is verbose, and all I/O avoids
single-node bottlenecks where possible.
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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "config_sample_comprehensive.json"))
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "synthesized_spectra.zarr")


def _configure_logging(log_level: str, log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("zarr_synthesis")
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

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

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


def _zarr_compression_kwargs(zarr_compressor_cfg: Mapping):
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


def _load_config(config_path: str, project_root: str) -> TurbospectrumConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg_data = json.load(handle)
    cfg_data = _normalize_config_dict(cfg_data, default_project_root=project_root)
    accepted_fields = {fld.name for fld in dataclasses.fields(TurbospectrumConfig)}
    cfg_data = {k: v for k, v in cfg_data.items() if k in accepted_fields}
    if "project_root" not in cfg_data:
        cfg_data["project_root"] = project_root
    return TurbospectrumConfig(**cfg_data)


def _validate_grid(grid_root) -> Tuple[int, Dict[str, np.ndarray]]:
    required_columns = ["teff", "logg", "feh"]
    optional_turb = ["turbvel", "t_value"]
    wavelength_columns = ["lam_min", "lam_max", "lam_step"]

    available = set(grid_root.keys())
    missing = [col for col in required_columns + wavelength_columns if col not in available]
    if missing:
        raise KeyError(f"Grid Zarr is missing required columns: {missing}")

    column_data: Dict[str, np.ndarray] = {name: np.array(grid_root[name][:]) for name in required_columns + wavelength_columns}
    turb_column = next((col for col in optional_turb if col in available), None)
    if turb_column is None:
        raise KeyError("Grid Zarr must include either 'turbvel' or 't_value' for microturbulence selection")
    column_data["turb"] = np.array(grid_root[turb_column][:])

    optional_columns = ["output_mode", "calculation_mode", "grid_version"]
    for name in optional_columns:
        if name in available:
            column_data[name] = np.array(grid_root[name][:])

    row_count = len(column_data["teff"])
    for name, values in column_data.items():
        if len(values) != row_count:
            raise ValueError(f"Column {name} length {len(values)} does not match expected {row_count}")

    return row_count, column_data


def _expected_wavelengths(column_data: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, int]:
    lam_min_vals = np.unique(column_data["lam_min"])
    lam_max_vals = np.unique(column_data["lam_max"])
    lam_step_vals = np.unique(column_data["lam_step"])

    if len(lam_min_vals) != 1 or len(lam_max_vals) != 1 or len(lam_step_vals) != 1:
        raise ValueError(
            "Wavelength bounds/steps differ across grid rows; consistent grids are required "
            "for dense Zarr output."
        )

    lam_min = float(lam_min_vals[0])
    lam_max = float(lam_max_vals[0])
    lam_step = float(lam_step_vals[0])
    if lam_step <= 0:
        raise ValueError("lam_step must be positive")

    count = int(round((lam_max - lam_min) / lam_step)) + 1
    wavelengths = lam_min + lam_step * np.arange(count, dtype=np.float64)
    return wavelengths, count


def _synthesis_task(args) -> Dict:
    index, row_values, base_config = args
    teff = int(row_values["teff"])
    logg = float(row_values["logg"])
    feh = float(row_values["feh"])
    turb_str = str(row_values["turb"]).strip()
    lam_min = float(row_values["lam_min"])
    lam_max = float(row_values["lam_max"])
    lam_step = float(row_values["lam_step"])
    # If the grid provides per-row mode flags, honor them; otherwise fall back
    # to whatever the Turbospectrum config requested.
    output_mode = row_values.get("output_mode")
    if output_mode is None:
        output_mode = "Intensity" if base_config.calculate_intensity else "Flux"
    calculation_mode = row_values.get("calculation_mode")
    if calculation_mode is None:
        calculation_mode = "NLTE" if base_config.nlte else "LTE"
    output_mode = str(output_mode)
    calculation_mode = str(calculation_mode)

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
            cont = data[:, 2].astype(np.float32) if data.shape[1] > 2 else np.full_like(flux, np.nan, dtype=np.float32)
            spectrum = (flux, cont)
        except Exception as exc:  # noqa: BLE001
            return {
                "index": index,
                "base_name": base_name,
                "status": "error",
                "message": f"Failed to read spectrum {spec_path}: {exc}",
                "duration": duration,
                "spectrum": None,
            }

    return {
        "index": index,
        "base_name": base_name,
        "status": result["status"],
        "message": result["message"],
        "duration": duration,
        "spectrum": spectrum,
    }


def _write_zarr_output(
    output_path: str,
    wavelengths: np.ndarray,
    fluxes: np.ndarray,
    continua: np.ndarray,
    column_data: Mapping[str, np.ndarray],
    statuses: Sequence[str],
    messages: Sequence[str],
    compression_cfg: Mapping,
    chunk_rows: int,
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    store = _zarr_store(output_path)
    root = zarr.group(store=store, overwrite=True, zarr_format=3)

    compression_kwargs = _zarr_compression_kwargs(compression_cfg)
    import zarr.codecs as zc  # type: ignore
    from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

    chunk_shape = (min(chunk_rows, fluxes.shape[0]), fluxes.shape[1])
    root.create_array("wavelength", data=wavelengths, chunks=wavelengths.shape, **compression_kwargs)
    root.create_array("flux", data=fluxes, chunks=chunk_shape, **compression_kwargs)
    root.create_array("continuum", data=continua, chunks=chunk_shape, **compression_kwargs)

    for name, values in column_data.items():
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

    root.attrs["creation_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    root.attrs["wavelength_count"] = wavelengths.size
    root.attrs["chunk_rows"] = chunk_rows
    logger.info("Wrote spectra to %s (shape=%s)", os.path.abspath(output_path), fluxes.shape)


def _build_tasks(row_count: int, column_data: Mapping[str, np.ndarray], base_config: TurbospectrumConfig):
    tasks = []
    for idx in range(row_count):
        row_values = {
            "teff": column_data["teff"][idx],
            "logg": column_data["logg"][idx],
            "feh": column_data["feh"][idx],
            "turb": column_data["turb"][idx],
            "lam_min": column_data["lam_min"][idx],
            "lam_max": column_data["lam_max"][idx],
            "lam_step": column_data["lam_step"][idx],
        }
        for optional_key in ("output_mode", "calculation_mode"):
            if optional_key in column_data:
                row_values[optional_key] = column_data[optional_key][idx]
        tasks.append((idx, row_values, base_config))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-zarr", required=True, help="Input Zarr grid path")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to Turbospectrum JSON config")
    parser.add_argument("--output-zarr", default=DEFAULT_OUTPUT_PATH, help="Output Zarr path for synthesized spectra")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--chunk-rows", type=int, default=32, help="Zarr chunking along the sample dimension")
    parser.add_argument("--compressor", default=None, help="JSON string describing compressor options (cname, clevel, shuffle)")
    args = parser.parse_args()

    logger = _configure_logging(args.log_level, args.log_file)
    t0 = time.perf_counter()
    logger.info("Starting synthesis run: grid=%s config=%s", os.path.abspath(args.grid_zarr), os.path.abspath(args.config))
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    config = _load_config(args.config, project_root=project_root)
    ensure_directories(config)
    config.linelist_file_path = create_linelist_file(config)

    grid_store = _zarr_store(args.grid_zarr)
    grid_root = zarr.open_group(store=grid_store, mode="r")
    row_count, column_data = _validate_grid(grid_root)
    wavelengths, expected_points = _expected_wavelengths(column_data)
    logger.info("Grid rows=%d wavelength_points=%d", row_count, expected_points)
    if "output_mode" in column_data:
        unique_modes = sorted({str(x) for x in np.unique(column_data["output_mode"])})
        logger.info("Grid output_mode values: %s", unique_modes)
    if "calculation_mode" in column_data:
        unique_calc = sorted({str(x) for x in np.unique(column_data["calculation_mode"])})
        logger.info("Grid calculation_mode values: %s", unique_calc)

    fluxes = np.full((row_count, expected_points), np.nan, dtype=np.float32)
    continua = np.full_like(fluxes, np.nan)
    statuses: List[str] = ["pending"] * row_count
    messages: List[str] = [""] * row_count

    tasks = _build_tasks(row_count, column_data, config)
    worker_count = determine_worker_count(config)

    compressor_cfg: Dict[str, object] = {}
    if args.compressor:
        try:
            compressor_cfg = json.loads(args.compressor)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid compressor JSON: {exc}") from exc

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_synthesis_task, task): task[0] for task in tasks}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                statuses[idx] = "exception"
                messages[idx] = str(exc)
                logger.exception("Task %d crashed: %s", idx, exc)
                continue

            statuses[idx] = result["status"]
            messages[idx] = result["message"]

            if result.get("spectrum"):
                fluxes[idx], continua[idx] = result["spectrum"]
            logger.info(
                "[%d/%d] %s %s (%.2fs) - %s",
                idx + 1,
                row_count,
                result["status"].upper(),
                result["base_name"],
                result["duration"],
                result["message"],
            )

    _write_zarr_output(
        output_path=args.output_zarr,
        wavelengths=wavelengths,
        fluxes=fluxes,
        continua=continua,
        column_data=column_data,
        statuses=statuses,
        messages=messages,
        compression_cfg=compressor_cfg,
        chunk_rows=args.chunk_rows,
        logger=logger,
    )
    logger.info("Completed synthesis in %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
