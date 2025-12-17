#!/usr/bin/env python3
"""Generate machine-learning-friendly parameter samples via Latin Hypercube.

This helper reads a JSON configuration (defaults to ``config_ml_sampling.json``
at the repository root), samples stellar parameters within specified bounds,
and writes the results to a Zarr (v3) store that matches the grid format
expected by the synthesis scripts. The Polars + Zarr pipeline keeps I/O
efficient for HPC environments and downstream ML ingestion.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl
import zarr
from numcodecs import Blosc, VLenUTF8


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "config_ml_sampling.json"))
DEFAULT_ZARR_PATH = os.path.join(SCRIPT_DIR, "ml_parameter_grid.zarr")
ALLOWED_TURBVEL = {"01", "02", "03", "04", "05"}


def _ensure_polars_zarr_available() -> None:
    # Imports are already evaluated at module import time, but keep a helper for
    # clearer error messaging when dependencies are missing on HPC nodes.
    missing = []
    try:  # noqa: TRY300
        import polars  # type: ignore
    except Exception:
        missing.append("polars")
    try:  # noqa: TRY300
        import zarr  # type: ignore
    except Exception:
        missing.append("zarr")
    if missing:
        raise ImportError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install with `pip install polars zarr numcodecs` on the target node."
        )


def _load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _latin_hypercube(bounds: Iterable[Tuple[float, float]], samples: int, rng: np.random.Generator) -> np.ndarray:
    bounds = list(bounds)
    if samples <= 0:
        raise ValueError("Number of samples must be positive")
    if not bounds:
        raise ValueError("At least one dimension must be provided for sampling")

    dims = len(bounds)
    lhs = np.empty((samples, dims), dtype=float)
    for dim, (lower, upper) in enumerate(bounds):
        if upper <= lower:
            raise ValueError(f"Upper bound must exceed lower bound for dimension {dim}: {lower} >= {upper}")
        stratified = (rng.random(samples) + np.arange(samples)) / samples
        rng.shuffle(stratified)
        lhs[:, dim] = lower + stratified * (upper - lower)
    return lhs


def _choose_series(raw_value, rng: np.random.Generator, length: int, name: str) -> np.ndarray:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        if not raw_value:
            raise ValueError(f"Configuration for '{name}' lists no options")
        return rng.choice(np.asarray(raw_value, dtype=object), size=length)
    if raw_value is None:
        raise ValueError(f"Configuration for '{name}' is missing")
    return np.full(length, raw_value, dtype=object)


def _abundance_value(raw_value) -> str:
    if raw_value is None:
        return "+0.00"
    return f"{float(raw_value):+0.2f}" if isinstance(raw_value, (int, float, np.floating)) else str(raw_value)


def _validate_turbvel_options(raw_options) -> List[str]:
    if raw_options is None:
        return sorted(ALLOWED_TURBVEL)
    if isinstance(raw_options, (str, bytes)):
        raw_options = [raw_options]
    options = []
    for entry in raw_options:
        value = f"{int(entry):02d}" if isinstance(entry, (int, float, np.integer, np.floating)) else str(entry)
        if value not in ALLOWED_TURBVEL:
            raise ValueError(f"Turbvel option '{value}' is invalid. Allowed options: {sorted(ALLOWED_TURBVEL)}")
        if value not in options:
            options.append(value)
    if not options:
        raise ValueError("At least one turbvel option must be provided when sampling turbvel")
    return options


def _resolve_bounds(config: Dict) -> List[Tuple[float, float]]:
    bounds_cfg = config.get("bounds") or {}
    ordered_names = ["teff", "logg", "feh"]
    bounds: List[Tuple[float, float]] = []
    for name in ordered_names:
        spec = bounds_cfg.get(name)
        if not spec or "min" not in spec or "max" not in spec:
            raise ValueError(f"Bounds for '{name}' must include 'min' and 'max'")
        bounds.append((float(spec["min"]), float(spec["max"])))
    return bounds


def _resolve_sampling_dimensions(config: Dict) -> Tuple[List[Tuple[float, float]], List[str], Dict[str, str], List[str] | None]:
    bounds = _resolve_bounds(config)

    abundances_cfg = config.get("abundances", {})
    sampled_abundances: List[str] = []
    fixed_abundances: Dict[str, str] = {}
    for element in ["a", "c", "n", "o", "r", "s"]:
        raw_val = abundances_cfg.get(element)
        if isinstance(raw_val, dict):
            if "min" not in raw_val or "max" not in raw_val:
                raise ValueError(f"Abundance bounds for '{element}' must include 'min' and 'max'")
            bounds.append((float(raw_val["min"]), float(raw_val["max"])))
            sampled_abundances.append(element)
        else:
            fixed_abundances[element] = _abundance_value(raw_val)

    turbvel_cfg = config.get("turbvel", "01")
    sample_turbvel = bool(config.get("sample_turbvel", False))
    turbvel_options = None
    if sample_turbvel:
        options_raw = config.get("turbvel_options", turbvel_cfg if isinstance(turbvel_cfg, Sequence) and not isinstance(turbvel_cfg, (str, bytes)) else None)
        turbvel_options = _validate_turbvel_options(options_raw)
        bounds.append((0, float(len(turbvel_options))))

    return bounds, sampled_abundances, fixed_abundances, turbvel_options


def main() -> None:
    _ensure_polars_zarr_available()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to JSON config with sampling bounds and defaults")
    parser.add_argument(
        "--output",
        default=None,
        help="Deprecated alias for --zarr-output. CSV export is no longer supported; data are written to Zarr only.",
    )
    parser.add_argument("--zarr-output", default=None, help="Optional override for Zarr output path")
    parser.add_argument("--resume", action="store_true", help="Append new samples up to num_samples if outputs already exist")
    parser.add_argument("--samples", type=int, default=None, help="Override the number of samples without editing the config")
    args = parser.parse_args()

    config = _load_config(args.config)
    sample_count = args.samples or int(config.get("num_samples", 50))
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    bounds, sampled_abundances, fixed_abundances, sampled_turbvel_options = _resolve_sampling_dimensions(config)
    lhs = _latin_hypercube(bounds, sample_count, rng)

    synthesis_cfg = config.get("synthesis", {})
    lam_min = synthesis_cfg.get("lam_min", 6000)
    lam_max = synthesis_cfg.get("lam_max", 6100)
    lam_step = synthesis_cfg.get("lam_step", 0.01)
    output_mode = synthesis_cfg.get("output_mode", "Flux")

    abundances = config.get("abundances", {})

    grid_version = config.get("grid_version", "ml-sample")
    mode = config.get("mode", "1D")
    calculation_mode = config.get("calculation_mode", "LTE")
    turbvel_cfg = config.get("turbvel", "01")
    t_value_options = config.get("t_value_options", ["01"])

    zarr_cfg = config.get("zarr", {})
    zarr_path = args.zarr_output or args.output or zarr_cfg.get("path") or DEFAULT_ZARR_PATH
    zarr_chunks = int(zarr_cfg.get("chunks", 2048)) if zarr_cfg else 2048
    zarr_compressor_cfg = zarr_cfg.get("compressor", {}) if zarr_cfg else {}

    if not zarr_path:
        raise ValueError("Zarr output path must be provided via config, --zarr-output, or --output (deprecated alias).")
    os.makedirs(os.path.dirname(os.path.abspath(zarr_path)), exist_ok=True)

    existing_rows = 0
    if args.resume and os.path.exists(zarr_path):
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open_group(store=store, mode="a", zarr_format=3)
        array_keys = list(root.keys())
        if not array_keys:
            raise ValueError(f"Zarr path {zarr_path} exists but contains no arrays; remove it or disable --resume")
        existing_rows = root[array_keys[0]].shape[0]
        for name in array_keys[1:]:
            arr = root[name]
            if arr.shape[0] != existing_rows:
                raise ValueError(
                    f"Zarr column {name} has inconsistent length {arr.shape[0]} vs expected {existing_rows}"
                )
        if existing_rows >= sample_count:
            print(f"Resume requested, but output already has {existing_rows} rows (>= target {sample_count}). Nothing to do.")
            return

    lhs = lhs[existing_rows:]
    sample_count_new = lhs.shape[0]

    column_idx = 0
    int_teff = np.rint(lhs[:, column_idx]).astype(int)
    column_idx += 1
    logg_vals = np.round(lhs[:, column_idx], 3)
    column_idx += 1
    feh_vals = np.round(lhs[:, column_idx], 3)
    column_idx += 1

    abundance_samples: Dict[str, np.ndarray] = {}
    for element in sampled_abundances:
        abundance_values = np.round(lhs[:, column_idx], 2)
        abundance_samples[element] = np.array([f"{val:+0.2f}" for val in abundance_values], dtype=object)
        column_idx += 1

    if sampled_turbvel_options:
        turbvel_indices = np.floor(lhs[:, column_idx]).astype(int)
        turbvel_indices = np.clip(turbvel_indices, 0, len(sampled_turbvel_options) - 1)
        turbvel_series = np.asarray([sampled_turbvel_options[idx] for idx in turbvel_indices], dtype=object)
        column_idx += 1
    else:
        turbvel_series = _choose_series(turbvel_cfg, rng, sample_count_new, "turbvel")

    t_value_series = _choose_series(t_value_options, rng, sample_count_new, "t_value_options")

    df = pl.DataFrame(
        {
            "grid_version": np.full(sample_count_new, grid_version, dtype=object),
            "teff": int_teff,
            "logg": logg_vals,
            "feh": feh_vals,
            "lam_min": np.full(sample_count_new, lam_min, dtype=float),
            "lam_max": np.full(sample_count_new, lam_max, dtype=float),
            "lam_step": np.full(sample_count_new, lam_step, dtype=float),
            "turbvel": turbvel_series,
            "t_value": t_value_series,
            "a": abundance_samples.get("a", np.full(sample_count_new, fixed_abundances.get("a", "+0.00"), dtype=object)),
            "c": abundance_samples.get("c", np.full(sample_count_new, fixed_abundances.get("c", "+0.00"), dtype=object)),
            "n": abundance_samples.get("n", np.full(sample_count_new, fixed_abundances.get("n", "+0.00"), dtype=object)),
            "o": abundance_samples.get("o", np.full(sample_count_new, fixed_abundances.get("o", "+0.00"), dtype=object)),
            "r": abundance_samples.get("r", np.full(sample_count_new, fixed_abundances.get("r", "+0.00"), dtype=object)),
            "s": abundance_samples.get("s", np.full(sample_count_new, fixed_abundances.get("s", "+0.00"), dtype=object)),
            "output_mode": np.full(sample_count_new, output_mode, dtype=object),
            "mode": np.full(sample_count_new, mode, dtype=object),
            "calculation_mode": np.full(sample_count_new, calculation_mode, dtype=object),
        }
    )

    compressor = None
    if zarr_compressor_cfg:
        cname = zarr_compressor_cfg.get("cname", "zstd")
        clevel = int(zarr_compressor_cfg.get("clevel", 5))
        shuffle = zarr_compressor_cfg.get("shuffle", True)
        compressor = Blosc(cname=cname, clevel=clevel, shuffle=Blosc.BITSHUFFLE if shuffle else Blosc.NOSHUFFLE)

    strings_codec = VLenUTF8()
    store = zarr.DirectoryStore(zarr_path)

    if args.resume and os.path.exists(zarr_path):
        root = zarr.open_group(store=store, mode="a", zarr_format=3)
        array_keys = list(root.keys())
        if not array_keys:
            raise ValueError(f"Zarr path {zarr_path} exists but contains no arrays; remove it or disable --resume")
        existing_len = root[array_keys[0]].shape[0]
        new_len = existing_len + sample_count_new
        for column in df.columns:
            arr = root[column]
            if arr.shape[0] != existing_len:
                raise ValueError(f"Zarr column {column} has inconsistent length {arr.shape[0]} vs expected {existing_len}")
            data = df[column].to_list() if df[column].dtype == pl.Utf8 else df[column].to_numpy()
            arr.resize(new_len, axis=0)
            arr[existing_len:new_len] = data
        print(f"Appended {sample_count_new} samples to Zarr store at {zarr_path} (total rows now {new_len})")
    else:
        root = zarr.group(store=store, overwrite=True, zarr_format=3)
        for column in df.columns:
            series = df[column]
            if series.dtype == pl.Utf8:
                root.array(
                    column,
                    series.to_list(),
                    dtype=object,
                    object_codec=strings_codec,
                    compressor=compressor,
                    chunks=zarr_chunks,
                )
            else:
                root.array(column, series.to_numpy(), compressor=compressor, chunks=zarr_chunks)
        print(f"Wrote Zarr store to {zarr_path} with chunk size {zarr_chunks}")


if __name__ == "__main__":
    main()
