#!/usr/bin/env python3
"""Generate machine-learning-friendly parameter samples via Latin Hypercube.

This helper reads a JSON configuration (defaults to
``configs/sampling/config_ml_sampling.json``), samples stellar parameters within specified bounds,
and writes the results to a Zarr (v3) store that matches the grid format
expected by the synthesis scripts. The Polars + Zarr pipeline keeps I/O
efficient for HPC environments and downstream ML ingestion.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl
import zarr
from numcodecs import Blosc, VLenUTF8

try:
    from .nlte_ascii_departures import build_nlte_ascii_selector_columns, normalize_nlte_ascii_selector
except ImportError:
    from nlte_ascii_departures import build_nlte_ascii_selector_columns, normalize_nlte_ascii_selector


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "sampling", "config_ml_sampling.json")
DEFAULT_ZARR_PATH = os.path.join(REPO_ROOT, "runs", "local-dev", "outputs", "grids", "ml_parameter_grid.zarr")
ALLOWED_TURBVEL = {"01", "02", "03", "04", "05"}
LEGACY_ABUNDANCE_KEYS = ("a", "c", "n", "o", "r", "s")


def _configure_logging(log_level: str, log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("ml_grid_sampler")
    # Avoid duplicate handlers if re-imported/re-run in notebooks.
    if logger.handlers:
        return logger

    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
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


def _progress(iterable, total: int | None, desc: str, enabled: bool):
    """Progress wrapper: uses tqdm when installed, otherwise passthrough."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


try:
    from zarr_compat import (
        zarr_store as _zarr_store,
        compression_kwargs as _zarr_compression_kwargs_base,
        create_root_group as _zc_create_root_group,
        create_array as _zc_create_array,
        write_string_array as _zc_write_string_array,
        write_string_scalar as _zc_write_string_scalar,
        ZARR_V3 as _ZARR_V3,
    )
except ImportError:
    from .zarr_compat import (
        zarr_store as _zarr_store,
        compression_kwargs as _zarr_compression_kwargs_base,
        create_root_group as _zc_create_root_group,
        create_array as _zc_create_array,
        write_string_array as _zc_write_string_array,
        write_string_scalar as _zc_write_string_scalar,
        ZARR_V3 as _ZARR_V3,
    )


def _zarr_compression_kwargs(zarr_compressor_cfg: Dict):
    return _zarr_compression_kwargs_base(zarr_compressor_cfg)


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


def _ordered_abundance_keys(abundances_cfg: Dict[str, Any]) -> List[str]:
    configured = {str(key).strip() for key in (abundances_cfg or {}).keys() if str(key).strip()}
    ordered: List[str] = [key for key in LEGACY_ABUNDANCE_KEYS if key in configured]
    ordered.extend(sorted(configured.difference(ordered)))
    return ordered


def _resolve_sampling_dimensions(config: Dict) -> Tuple[List[Tuple[float, float]], List[str], Dict[str, str], List[str] | None]:
    bounds = _resolve_bounds(config)

    abundances_cfg = config.get("abundances", {})
    sampled_abundances: List[str] = []
    fixed_abundances: Dict[str, str] = {}
    for element in _ordered_abundance_keys(abundances_cfg):
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


def _default_index_parquet_path(zarr_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(zarr_path)), "index.parquet")


def _default_run_root() -> str:
    run_root = os.environ.get("RUN_ROOT")
    if run_root:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(run_root)))
    return os.path.join(REPO_ROOT, "runs", "local-dev")


def _resolve_output_path(run_root: str, cfg_dir: str, maybe_relative: str | None, default_path: str) -> str:
    if maybe_relative in (None, ""):
        return os.path.abspath(default_path)
    path = str(maybe_relative)
    if os.path.isabs(path):
        return os.path.abspath(path)
    if not path.startswith((".", "..")):
        return os.path.abspath(os.path.join(run_root, path))
    return os.path.abspath(os.path.join(cfg_dir, path))


def _index_df_from_zarr(root, column_names: Sequence[str]) -> pl.DataFrame:
    payload: Dict[str, Any] = {}
    row_count: int | None = None
    for name in column_names:
        values = np.asarray(root[name][:])
        if row_count is None:
            row_count = int(values.shape[0])
        elif int(values.shape[0]) != row_count:
            raise ValueError(f"Zarr column {name} has inconsistent length {values.shape[0]} vs expected {row_count}")
        payload[name] = values

    if row_count is None:
        row_count = 0
    return pl.DataFrame({"row_index": np.arange(row_count, dtype=np.int64), **payload})


def _write_index_parquet(
    *,
    df: pl.DataFrame,
    index_parquet_path: str,
    existing_rows: int,
    resume: bool,
    zarr_path: str,
    logger: logging.Logger,
) -> None:
    expected_columns = ["row_index", *df.columns]
    new_index_df = df.with_row_count("row_index", offset=existing_rows).select(expected_columns)

    final_index_df = new_index_df
    if resume and existing_rows > 0:
        if os.path.exists(index_parquet_path):
            existing_index_df = pl.read_parquet(index_parquet_path)
            existing_cols = existing_index_df.columns
            if "row_index" not in existing_cols and set(existing_cols) == set(df.columns):
                existing_index_df = existing_index_df.with_row_count("row_index", offset=0)
                existing_cols = existing_index_df.columns

            if existing_index_df.height == existing_rows and set(existing_cols) == set(expected_columns):
                existing_index_df = existing_index_df.select(expected_columns)
                final_index_df = pl.concat([existing_index_df, new_index_df], rechunk=True)
            else:
                logger.warning(
                    "Existing index parquet is out of sync (rows=%d expected=%d); rebuilding from Zarr.",
                    existing_index_df.height,
                    existing_rows,
                )
                store = _zarr_store(zarr_path)
                root = zarr.open_group(store=store, mode="r")
                final_index_df = _index_df_from_zarr(root, df.columns)
        else:
            logger.warning("Index parquet not found during resume; rebuilding from Zarr.")
            store = _zarr_store(zarr_path)
            root = zarr.open_group(store=store, mode="r")
            final_index_df = _index_df_from_zarr(root, df.columns)

    os.makedirs(os.path.dirname(os.path.abspath(index_parquet_path)), exist_ok=True)
    final_index_df.write_parquet(index_parquet_path, compression="zstd")
    logger.info("Wrote parameter index parquet: %s (%d rows)", os.path.abspath(index_parquet_path), final_index_df.height)


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
    parser.add_argument("--index-parquet-output", default=None, help="Optional override for parameter index parquet path")
    parser.add_argument("--resume", action="store_true", help="Append new samples up to num_samples if outputs already exist")
    parser.add_argument("--samples", type=int, default=None, help="Override the number of samples without editing the config")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument(
        "--nlte-ascii-dir",
        default=None,
        help="Directory containing per-abundance NLTE ASCII departure files",
    )
    parser.add_argument(
        "--nlte-ascii-species",
        default=None,
        help="Element symbol or atomic number for the NLTE ASCII departure files (default: Fe)",
    )
    parser.add_argument(
        "--nlte-ascii-abundance-column",
        default=None,
        help="Grid column that controls the departure-file abundance token (default: auto)",
    )
    parser.add_argument(
        "--nlte-ascii-abundance-scale",
        default=None,
        choices=("relative", "absolute"),
        help="Interpret the abundance column as [X/Fe] or absolute log abundance",
    )
    parser.add_argument(
        "--nlte-ascii-solar-abundance",
        type=float,
        default=None,
        help="Solar abundance used when converting relative [X/Fe] values into absolute abundance tokens",
    )
    parser.add_argument(
        "--nlte-ascii-match",
        default=None,
        choices=("nearest", "exact"),
        help="How to match available NLTE ASCII files in the departure directory",
    )
    args = parser.parse_args()

    logger = _configure_logging(args.log_level, args.log_file)
    t0 = time.perf_counter()

    config = _load_config(args.config)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    logger.debug("Config contents:\n%s", json.dumps(config, indent=2, sort_keys=True))
    sample_count = args.samples or int(config.get("num_samples", 50))
    seed = config.get("seed")
    rng = np.random.default_rng(seed)
    nlte_ascii_cfg = config.get("nlte_ascii_departures") or {}
    nlte_ascii_selector = normalize_nlte_ascii_selector(
        directory=args.nlte_ascii_dir if args.nlte_ascii_dir is not None else nlte_ascii_cfg.get("directory"),
        species=args.nlte_ascii_species if args.nlte_ascii_species is not None else nlte_ascii_cfg.get("species"),
        abundance_column=(
            args.nlte_ascii_abundance_column
            if args.nlte_ascii_abundance_column is not None
            else nlte_ascii_cfg.get("abundance_column")
        ),
        abundance_scale=(
            args.nlte_ascii_abundance_scale
            if args.nlte_ascii_abundance_scale is not None
            else nlte_ascii_cfg.get("abundance_scale")
        ),
        solar_abundance=(
            args.nlte_ascii_solar_abundance
            if args.nlte_ascii_solar_abundance is not None
            else nlte_ascii_cfg.get("solar_abundance")
        ),
        match=args.nlte_ascii_match if args.nlte_ascii_match is not None else nlte_ascii_cfg.get("match"),
    )

    logger.info("Loaded config: %s", os.path.abspath(args.config))
    logger.info("Sampling: target_rows=%d seed=%s resume=%s", sample_count, seed, args.resume)
    logger.info("Versions: polars=%s zarr=%s numpy=%s", getattr(pl, "__version__", "?"), getattr(zarr, "__version__", "?"), getattr(np, "__version__", "?"))
    if nlte_ascii_selector is not None:
        logger.info(
            "NLTE ASCII departures: dir=%s species=%s abundance_column=%s scale=%s match=%s",
            nlte_ascii_selector.directory,
            nlte_ascii_selector.species,
            nlte_ascii_selector.abundance_column,
            nlte_ascii_selector.abundance_scale,
            nlte_ascii_selector.match,
        )

    t_step = time.perf_counter()
    bounds, sampled_abundances, fixed_abundances, sampled_turbvel_options = _resolve_sampling_dimensions(config)
    logger.debug("Fixed abundances: %s", fixed_abundances)
    logger.debug("turbvel sampled options: %s", sampled_turbvel_options)
    logger.debug("t_value_options: %s", config.get("t_value_options", ["01"]))
    logger.info(
        "Sampling dims: teff/logg/feh + sampled_abundances=%s + sample_turbvel=%s",
        sampled_abundances,
        bool(sampled_turbvel_options),
    )
    logger.info("Prepared sampling dimensions in %.2fs", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    lhs = _latin_hypercube(bounds, sample_count, rng)
    logger.info("Generated Latin hypercube in %.2fs", time.perf_counter() - t_step)

    synthesis_cfg = config.get("synthesis", {})
    lam_min = synthesis_cfg.get("lam_min", 6000)
    lam_max = synthesis_cfg.get("lam_max", 6100)
    lam_step = synthesis_cfg.get("lam_step", 0.01)
    output_mode = synthesis_cfg.get("output_mode", "Flux")

    abundances = config.get("abundances", {})

    grid_version = config.get("grid_version", "ml-sample")
    mode = synthesis_cfg.get("mode", config.get("mode", "1D"))
    calculation_mode = synthesis_cfg.get("calculation_mode", config.get("calculation_mode", "LTE"))
    turbvel_cfg = config.get("turbvel", "01")
    t_value_options = config.get("t_value_options", ["01"])

    run_root_cfg = config.get("run_root")
    if run_root_cfg not in (None, ""):
        run_root = _resolve_output_path(REPO_ROOT, cfg_dir, run_root_cfg, _default_run_root())
    else:
        run_root = _default_run_root()

    zarr_cfg = config.get("zarr", {})
    if args.zarr_output or args.output:
        zarr_path = os.path.abspath(args.zarr_output or args.output)
    else:
        zarr_path = _resolve_output_path(run_root, cfg_dir, zarr_cfg.get("path"), DEFAULT_ZARR_PATH)
    if args.index_parquet_output:
        index_parquet_path = os.path.abspath(args.index_parquet_output)
    else:
        index_parquet_path = _resolve_output_path(
            run_root,
            cfg_dir,
            config.get("index_parquet"),
            _default_index_parquet_path(zarr_path),
        )
    zarr_chunks = int(zarr_cfg.get("chunks", 2048)) if zarr_cfg else 2048
    zarr_compressor_cfg = zarr_cfg.get("compressor", {}) if zarr_cfg else {}

    if not zarr_path:
        raise ValueError("Zarr output path must be provided via config, --zarr-output, or --output (deprecated alias).")
    os.makedirs(os.path.dirname(os.path.abspath(zarr_path)), exist_ok=True)
    logger.info("Run root: %s", os.path.abspath(run_root))
    logger.info("Zarr output: %s (chunks=%s)", os.path.abspath(zarr_path), zarr_chunks)
    logger.info("Index parquet: %s", os.path.abspath(index_parquet_path))

    existing_rows = 0
    if args.resume and os.path.exists(zarr_path):
        t_step = time.perf_counter()
        store = _zarr_store(zarr_path)
        root = zarr.open_group(store=store, mode="a")
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
            logger.info(
                "Resume requested, but output already has %d rows (>= target %d). Nothing to do.",
                existing_rows,
                sample_count,
            )
            return
        logger.info("Resuming: existing_rows=%d adding=%d", existing_rows, sample_count - existing_rows)
        logger.info("Resume preflight in %.2fs", time.perf_counter() - t_step)

    lhs = lhs[existing_rows:]
    sample_count_new = lhs.shape[0]
    logger.info("Generating %d new rows", sample_count_new)

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

    t_step = time.perf_counter()
    abundance_columns = {
        element: abundance_samples.get(
            element,
            np.full(sample_count_new, fixed_abundances.get(element, "+0.00"), dtype=object),
        )
        for element in _ordered_abundance_keys(abundances)
    }
    columns = {
        "grid_version": np.full(sample_count_new, grid_version, dtype=object),
        "teff": int_teff,
        "logg": logg_vals,
        "feh": feh_vals,
        "lam_min": np.full(sample_count_new, lam_min, dtype=float),
        "lam_max": np.full(sample_count_new, lam_max, dtype=float),
        "lam_step": np.full(sample_count_new, lam_step, dtype=float),
        "turbvel": turbvel_series,
        "t_value": t_value_series,
        **abundance_columns,
        "output_mode": np.full(sample_count_new, output_mode, dtype=object),
        "mode": np.full(sample_count_new, mode, dtype=object),
        "calculation_mode": np.full(sample_count_new, calculation_mode, dtype=object),
    }
    columns.update(build_nlte_ascii_selector_columns(sample_count_new, nlte_ascii_selector))
    df = pl.DataFrame(columns)
    logger.info("Built Polars DataFrame (%d rows, %d cols) in %.2fs", df.height, len(df.columns), time.perf_counter() - t_step)

    compression_kwargs = _zarr_compression_kwargs(zarr_compressor_cfg)
    logger.debug("Zarr compression kwargs: %s", compression_kwargs)
    strings_codec = VLenUTF8()
    store = _zarr_store(zarr_path)

    if args.resume and os.path.exists(zarr_path):
        t_step = time.perf_counter()
        root = zarr.open_group(store=store, mode="a")
        array_keys = list(root.keys())
        if not array_keys:
            raise ValueError(f"Zarr path {zarr_path} exists but contains no arrays; remove it or disable --resume")
        existing_len = root[array_keys[0]].shape[0]
        new_len = existing_len + sample_count_new
        for column in _progress(df.columns, total=len(df.columns), desc="Appending columns", enabled=not args.no_progress):
            arr = root[column]
            if arr.shape[0] != existing_len:
                raise ValueError(f"Zarr column {column} has inconsistent length {arr.shape[0]} vs expected {existing_len}")
            if df[column].dtype == pl.Utf8:
                data = df[column].to_list()
            else:
                data = df[column].to_numpy()
            arr.resize(new_len, axis=0)
            arr[existing_len:new_len] = data
        logger.info("Appended %d rows (total now %d)", sample_count_new, new_len)
        logger.info("Zarr append finished in %.2fs", time.perf_counter() - t_step)
    else:
        t_step = time.perf_counter()
        root = _zc_create_root_group(store, overwrite=True)
        for column in _progress(df.columns, total=len(df.columns), desc="Writing columns", enabled=not args.no_progress):
            series = df[column]
            if series.dtype == pl.Utf8:
                _zc_write_string_array(root, column, series.to_list(),
                                       chunks=zarr_chunks, compression_kw=compression_kwargs)
            else:
                _zc_create_array(root, column, data=series.to_numpy(),
                                 chunks=zarr_chunks, **compression_kwargs)
        logger.info("Wrote Zarr store to %s (chunk size %s)", zarr_path, zarr_chunks)
        logger.info("Zarr write finished in %.2fs", time.perf_counter() - t_step)

    _write_index_parquet(
        df=df,
        index_parquet_path=index_parquet_path,
        existing_rows=existing_rows,
        resume=args.resume,
        zarr_path=zarr_path,
        logger=logger,
    )

    logger.info("Done in %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
