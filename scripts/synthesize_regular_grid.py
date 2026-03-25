#!/usr/bin/env python3
"""Generate and synthesize a regular Cartesian stellar grid.

This script is intended for interpolation-ready datasets where parameter axes
must be uniformly spaced (linear interpolation assumptions).

Workflow:
1. Build a regular grid over (teff, logg, feh, turbvel).
2. Write grid CSV/Zarr in the same schema expected by synthesis scripts.
3. Optionally synthesize spectra from the generated grid.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

try:
    from .nlte_ascii_departures import build_nlte_ascii_selector_columns, normalize_nlte_ascii_selector
except ImportError:
    from nlte_ascii_departures import build_nlte_ascii_selector_columns, normalize_nlte_ascii_selector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def _default_run_root() -> str:
    run_root = os.environ.get("RUN_ROOT")
    if run_root:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(run_root)))
    scratch_base = (os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or "/tmp").rstrip("/")
    return os.path.join(scratch_base, "turbospectrum_nlte", os.environ.get("USER", "user"))


def _as_abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _abs_from(base_dir: str, path: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(base_dir, expanded))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _cfg_get(cfg: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _coalesce(cli_value: Any, cfg: Dict[str, Any], cfg_keys: Sequence[str], default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    cfg_value = _cfg_get(cfg, cfg_keys, None)
    if cfg_value is not None:
        return cfg_value
    return default


def _parse_numeric_axis(raw: str, name: str, *, integer: bool) -> np.ndarray:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{name} axis spec cannot be empty")

    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError(f"{name} axis range must be start:end:step, got {text!r}")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
        if step <= 0:
            raise ValueError(f"{name} axis step must be positive, got {step}")
        if stop < start:
            raise ValueError(f"{name} axis stop must be >= start, got {start}..{stop}")
        values = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    else:
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        if not tokens:
            raise ValueError(f"{name} axis list is empty")
        values = np.asarray([float(t) for t in tokens], dtype=np.float64)
        values = np.sort(np.unique(values))

    if values.size < 2:
        raise ValueError(f"{name} axis must contain at least two points for linear interpolation")

    diffs = np.diff(values)
    ref = float(diffs[0])
    tol = max(1e-10, abs(ref) * 1e-8)
    if not np.allclose(diffs, ref, rtol=0.0, atol=tol):
        raise ValueError(
            f"{name} axis is not uniformly spaced; got steps like {diffs[:5].tolist()}"
        )

    if integer:
        ints = np.rint(values).astype(np.int64)
        if not np.allclose(values, ints.astype(np.float64), rtol=0.0, atol=1e-8):
            raise ValueError(f"{name} axis requires integer values; got {values[:5].tolist()}")
        return ints
    return values.astype(np.float64)


def _parse_turbvel_values(raw: str) -> np.ndarray:
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        raise ValueError("turbvel axis cannot be empty")
    out: List[str] = []
    seen = set()
    for token in tokens:
        if token.lstrip("+-").isdigit():
            token = f"{int(token):02d}"
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return np.asarray(out, dtype=object)


def _build_regular_columns(
    *,
    teff_axis: np.ndarray,
    logg_axis: np.ndarray,
    feh_axis: np.ndarray,
    turbvel_axis: np.ndarray,
    mu_axis: np.ndarray | None,
    grid_version: str,
    lam_min: float,
    lam_max: float,
    lam_step: float,
    output_mode: str,
    mode: str,
    calculation_mode: str,
    abundances: Dict[str, str],
    max_rows: int,
    nlte_ascii_selector: Any = None,
) -> Dict[str, np.ndarray]:
    nt = int(teff_axis.size)
    ng = int(logg_axis.size)
    nf = int(feh_axis.size)
    nu = int(turbvel_axis.size)
    nm = int(mu_axis.size) if mu_axis is not None else 1
    base_count = nt * ng * nf * nu
    row_count = base_count * nm
    if row_count <= 0:
        raise ValueError("Computed row_count is zero")
    if row_count > max_rows:
        raise ValueError(
            f"Grid would create {row_count:,} rows (> max_rows={max_rows:,}). "
            "Lower axis resolutions or increase --max-rows."
        )

    teff_base = np.repeat(teff_axis.astype(np.int64), ng * nf * nu)
    logg_base = np.tile(np.repeat(logg_axis.astype(np.float64), nf * nu), nt)
    feh_base = np.tile(np.repeat(feh_axis.astype(np.float64), nu), nt * ng)
    turbvel_base = np.tile(turbvel_axis.astype(object), nt * ng * nf)

    if mu_axis is not None:
        teff = np.repeat(teff_base, nm)
        logg = np.repeat(logg_base, nm)
        feh = np.repeat(feh_base, nm)
        turbvel = np.repeat(turbvel_base, nm)
        mu = np.tile(mu_axis.astype(np.float64), base_count)
    else:
        teff = teff_base
        logg = logg_base
        feh = feh_base
        turbvel = turbvel_base
        mu = None

    columns: Dict[str, np.ndarray] = {
        "grid_version": np.full(row_count, str(grid_version), dtype=object),
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "lam_min": np.full(row_count, float(lam_min), dtype=np.float64),
        "lam_max": np.full(row_count, float(lam_max), dtype=np.float64),
        "lam_step": np.full(row_count, float(lam_step), dtype=np.float64),
        "turbvel": turbvel,
        # Keep t_value aligned with turbvel for compatibility with older readers.
        "t_value": turbvel.copy(),
        "a": np.full(row_count, abundances["a"], dtype=object),
        "c": np.full(row_count, abundances["c"], dtype=object),
        "n": np.full(row_count, abundances["n"], dtype=object),
        "o": np.full(row_count, abundances["o"], dtype=object),
        "r": np.full(row_count, abundances["r"], dtype=object),
        "s": np.full(row_count, abundances["s"], dtype=object),
        "output_mode": np.full(row_count, output_mode, dtype=object),
        "mode": np.full(row_count, mode, dtype=object),
        "calculation_mode": np.full(row_count, calculation_mode, dtype=object),
    }
    if mu is not None:
        columns["mu"] = mu
    columns.update(build_nlte_ascii_selector_columns(row_count, nlte_ascii_selector))
    return columns


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def _verify_continuum_saved(spectra_zarr: str) -> None:
    import zarr

    path = os.path.abspath(spectra_zarr)
    root = zarr.open_group(path, mode="r")
    missing = [name for name in ("flux", "continuum") if name not in root]
    if missing:
        raise KeyError(
            f"Synthesis output {path} is missing required arrays: {missing}. "
            "Expected both 'flux' and 'continuum'."
        )
    flux_shape = tuple(np.asarray(root["flux"]).shape)
    continuum_shape = tuple(np.asarray(root["continuum"]).shape)
    if flux_shape != continuum_shape:
        raise ValueError(
            f"Synthesis output {path} has mismatched shapes: "
            f"flux={flux_shape}, continuum={continuum_shape}"
        )
    status_counts = root.attrs.get("status_counts")
    if isinstance(status_counts, Mapping):
        bad = {}
        for key, value in status_counts.items():
            try:
                count = int(value)
            except Exception:
                continue
            if count <= 0:
                continue
            if str(key).strip().lower() not in {"success", "skipped"}:
                bad[str(key)] = count
        if bad:
            raise ValueError(
                f"Synthesis output {path} recorded failed rows in status_counts: {bad}"
            )


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

    return normalized


def _materialize_synthesis_config(
    *,
    base_config_path: str,
    run_root: str,
    overrides: Mapping[str, Any] | None = None,
    overrides_base_dir: str | None = None,
    mu_axis: np.ndarray | None = None,
) -> str:
    needs_write = bool(overrides) or mu_axis is not None
    if not needs_write:
        return base_config_path

    cfg = _load_json(base_config_path)
    if overrides:
        if not isinstance(overrides, Mapping):
            raise ValueError("turbospectrum.overrides must be a JSON object")
        override_cfg = dict(overrides)
        if overrides_base_dir:
            override_cfg = _normalize_synthesis_overrides(override_cfg, overrides_base_dir)
        cfg = _deep_merge(cfg, override_cfg)

    if mu_axis is not None:
        mu_min = float(np.min(mu_axis))
        mu_max = float(np.max(mu_axis))
        if mu_min < -1e-8 or mu_max > 1.0 + 1e-8:
            raise ValueError(f"mu range must stay within [0, 1], got {mu_min}..{mu_max}")

        synthesis = cfg.setdefault("synthesis_parameters", {})
        if not isinstance(synthesis, dict):
            raise ValueError("synthesis_parameters must be a JSON object in synthesis config")
        mu_sampling = synthesis.setdefault("mu_sampling", {})
        if not isinstance(mu_sampling, dict):
            raise ValueError("synthesis_parameters.mu_sampling must be a JSON object in synthesis config")
        mu_sampling["mode"] = "nearest"
        mu_sampling.setdefault("count", 1)
        mu_sampling["min"] = mu_min
        mu_sampling["max"] = mu_max

    cfg_dir = os.path.join(run_root, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    dst = os.path.join(cfg_dir, "synthesis_config.regular_grid.json")
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)
    return dst


def _default_shard_template(run_root: str) -> str:
    return os.path.join(run_root, "outputs", "shards", "spectra_shard_{shard_index}.zarr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None, help="Optional regular-grid JSON config file")
    parser.add_argument("--teff-axis", default=None, help="Teff axis (start:end:step or comma list)")
    parser.add_argument("--logg-axis", default=None, help="logg axis (start:end:step or comma list)")
    parser.add_argument("--feh-axis", default=None, help="feh axis (start:end:step or comma list)")
    parser.add_argument("--turbvel-axis", default=None, help="Comma-separated turbvel identifiers")
    parser.add_argument("--grid-version", default=None)

    parser.add_argument("--lam-min", type=float, default=None)
    parser.add_argument("--lam-max", type=float, default=None)
    parser.add_argument("--lam-step", type=float, default=None)
    parser.add_argument("--output-mode", default=None, choices=("Flux", "Intensity"))
    parser.add_argument("--mode", default=None)
    parser.add_argument("--calculation-mode", default=None)

    parser.add_argument("--a", default=None, help="Fixed [alpha/Fe] value stored in grid column 'a'")
    parser.add_argument("--c", default=None)
    parser.add_argument("--n", default=None)
    parser.add_argument("--o", default=None)
    parser.add_argument("--r", default=None)
    parser.add_argument("--s", default=None)
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

    parser.add_argument("--run-root", default=None, help="Runtime root (defaults to RUN_ROOT/scratch)")
    parser.add_argument("--grid-zarr", default=None, help="Output grid Zarr path")
    parser.add_argument("--grid-csv", default=None, help="Optional grid CSV path")
    parser.add_argument("--index-parquet", default=None, help="Optional parameter index parquet output path")
    parser.add_argument("--csv-compression", default=None, choices=("none", "gzip", "zstd"))
    parser.add_argument("--csv-compression-level", type=int, default=None)
    parser.add_argument("--zarr-chunks", type=int, default=None)
    parser.add_argument("--zarr-compressor", default=None, help="JSON compressor config")

    parser.add_argument("--max-rows", type=int, default=None, help="Safety cap for total grid rows")
    parser.add_argument("--skip-synthesis", action="store_true", default=None, help="Only generate the regular grid")

    parser.add_argument("--config", default=None, help="Turbospectrum synthesis config JSON path")
    parser.add_argument("--spectra-zarr", default=None, help="Output synthesized spectra Zarr path")
    parser.add_argument("--scratch", default=None, help="Optional scratch directory passed to synthesis")
    parser.add_argument("--workers", type=int, default=None, help="Worker count passed to synthesis")
    parser.add_argument("--chunk-rows", type=int, default=None, help="Output Zarr chunk_rows for synthesis output")
    parser.add_argument("--output-tmp", default=None, help="Atomic tmp output path passed as --output-tmp")
    parser.add_argument("--log-level", default=None, help="Logging level for synthesis script")
    parser.add_argument("--log-file", default=None, help="Optional synthesis log file path")
    parser.add_argument(
        "--print-runtime-json",
        action="store_true",
        help="Print resolved regular-grid runtime settings as JSON and exit.",
    )
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    cfg_dir = REPO_ROOT
    if args.config_json:
        cfg_path = _as_abspath(args.config_json)
        cfg_dir = os.path.dirname(cfg_path)
        cfg = _load_json(cfg_path)

    teff_axis_spec = str(_coalesce(args.teff_axis, cfg, ("grid", "axes", "teff"), "4000:7000:250"))
    logg_axis_spec = str(_coalesce(args.logg_axis, cfg, ("grid", "axes", "logg"), "0.0:5.0:0.5"))
    feh_axis_spec = str(_coalesce(args.feh_axis, cfg, ("grid", "axes", "feh"), "-2.5:0.5:0.25"))
    turbvel_axis_spec = str(_coalesce(args.turbvel_axis, cfg, ("grid", "axes", "turbvel"), "01,02,03"))
    mu_range_spec_raw = _cfg_get(cfg, ("grid", "synthesis", "mu_range"), None)
    mu_range_spec = None if mu_range_spec_raw in (None, "") else str(mu_range_spec_raw)
    grid_version = str(_coalesce(args.grid_version, cfg, ("grid", "grid_version"), "regular-linear-v1"))

    lam_min = float(_coalesce(args.lam_min, cfg, ("grid", "synthesis", "lam_min"), 8400.0))
    lam_max = float(_coalesce(args.lam_max, cfg, ("grid", "synthesis", "lam_max"), 8800.0))
    lam_step = float(_coalesce(args.lam_step, cfg, ("grid", "synthesis", "lam_step"), 0.01))
    output_mode = str(_coalesce(args.output_mode, cfg, ("grid", "synthesis", "output_mode"), "Flux"))
    mode = str(_coalesce(args.mode, cfg, ("grid", "synthesis", "mode"), "1D"))
    calculation_mode = str(_coalesce(args.calculation_mode, cfg, ("grid", "synthesis", "calculation_mode"), "LTE"))
    if output_mode not in {"Flux", "Intensity"}:
        raise ValueError(f"output_mode must be Flux or Intensity, got {output_mode!r}")

    csv_compression = str(_coalesce(args.csv_compression, cfg, ("grid", "io", "csv_compression"), "zstd")).lower()
    if csv_compression not in {"none", "gzip", "zstd"}:
        raise ValueError(f"csv_compression must be one of none/gzip/zstd, got {csv_compression!r}")
    csv_compression_level = int(_coalesce(args.csv_compression_level, cfg, ("grid", "io", "csv_compression_level"), 5))
    zarr_chunks = int(_coalesce(args.zarr_chunks, cfg, ("grid", "io", "zarr_chunks"), 2048))
    max_rows = int(_coalesce(args.max_rows, cfg, ("grid", "limits", "max_rows"), 2_000_000))

    zarr_compressor_raw = _coalesce(
        args.zarr_compressor,
        cfg,
        ("grid", "io", "zarr_compressor"),
        {"cname": "zstd", "clevel": 5, "shuffle": True},
    )
    if isinstance(zarr_compressor_raw, str):
        try:
            zarr_compressor = json.loads(zarr_compressor_raw) if zarr_compressor_raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --zarr-compressor JSON: {exc}") from exc
    elif isinstance(zarr_compressor_raw, dict):
        zarr_compressor = zarr_compressor_raw
    else:
        raise ValueError("zarr_compressor must be a JSON string or object mapping")

    run_root_cfg = _cfg_get(cfg, ("outputs", "run_root"), None)
    if args.run_root is not None:
        run_root = _as_abspath(args.run_root)
    elif run_root_cfg is not None:
        run_root = _abs_from(cfg_dir, str(run_root_cfg))
    else:
        run_root = _default_run_root()

    def _resolve_path(cli_value: Any, cfg_keys: Sequence[str], default_path: str) -> str:
        if cli_value is not None:
            return _as_abspath(str(cli_value))
        cfg_value = _cfg_get(cfg, cfg_keys, None)
        if cfg_value is not None:
            return _abs_from(cfg_dir, str(cfg_value))
        return os.path.abspath(default_path)

    grid_zarr = _resolve_path(
        args.grid_zarr,
        ("outputs", "grid_zarr"),
        os.path.join(run_root, "outputs", "grids", "regular_parameter_grid.zarr"),
    )
    grid_csv = _resolve_path(
        args.grid_csv,
        ("outputs", "grid_csv"),
        os.path.join(run_root, "outputs", "grids", "regular_parameter_grid.csv"),
    )
    index_parquet = _resolve_path(
        args.index_parquet,
        ("outputs", "grid_index_parquet"),
        os.path.join(os.path.dirname(grid_zarr), "index.parquet"),
    )
    spectra_zarr = _resolve_path(
        args.spectra_zarr,
        ("outputs", "spectra_zarr"),
        os.path.join(run_root, "outputs", "zarr", "regular_synthesized_spectra.zarr"),
    )
    shard_template_cfg = _cfg_get(cfg, ("outputs", "spectra_shard_template"), None)
    if shard_template_cfg is not None:
        spectra_shard_template = _abs_from(cfg_dir, str(shard_template_cfg))
    else:
        spectra_shard_template = os.path.abspath(_default_shard_template(run_root))
    spectra_shards_dir = os.path.dirname(spectra_shard_template)

    synthesis_cfg_default = os.path.join(REPO_ROOT, "configs", "synthesis", "config_sample_comprehensive.json")
    synthesis_cfg_from_template = _cfg_get(cfg, ("turbospectrum", "config"), None)
    if synthesis_cfg_from_template is None:
        synthesis_cfg_from_template = _cfg_get(cfg, ("runtime", "config"), None)
    synthesis_overrides = _cfg_get(cfg, ("turbospectrum", "overrides"), None)
    if args.config is not None:
        config_path = _as_abspath(args.config)
    elif synthesis_cfg_from_template is not None:
        config_path = _abs_from(cfg_dir, str(synthesis_cfg_from_template))
    else:
        config_path = synthesis_cfg_default

    workers_raw = _coalesce(args.workers, cfg, ("runtime", "workers"), None)
    workers = int(workers_raw) if workers_raw not in (None, "") else None
    chunk_rows = int(_coalesce(args.chunk_rows, cfg, ("runtime", "chunk_rows"), 32))
    scratch_default_base = os.environ.get("TMPDIR") or os.path.join(run_root, "tmp")
    scratch_default = os.path.join(scratch_default_base, "synthesis")
    scratch_raw = _coalesce(args.scratch, cfg, ("runtime", "scratch"), scratch_default)
    scratch = _abs_from(cfg_dir, str(scratch_raw)) if scratch_raw not in (None, "") else None
    output_tmp_raw = _coalesce(args.output_tmp, cfg, ("runtime", "output_tmp"), None)
    output_tmp = _abs_from(cfg_dir, str(output_tmp_raw)) if output_tmp_raw not in (None, "") else None
    log_level = str(_coalesce(args.log_level, cfg, ("runtime", "log_level"), "INFO"))
    log_file_raw = _coalesce(args.log_file, cfg, ("runtime", "log_file"), None)
    log_file = _abs_from(cfg_dir, str(log_file_raw)) if log_file_raw not in (None, "") else None

    skip_synthesis_cfg = _cfg_get(cfg, ("runtime", "skip_synthesis"), False)
    skip_synthesis = bool(args.skip_synthesis) if args.skip_synthesis is not None else bool(skip_synthesis_cfg)
    nlte_ascii_cfg = cfg.get("nlte_ascii_departures") if isinstance(cfg.get("nlte_ascii_departures"), dict) else {}
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

    teff_axis = _parse_numeric_axis(teff_axis_spec, "teff", integer=True)
    logg_axis = _parse_numeric_axis(logg_axis_spec, "logg", integer=False)
    feh_axis = _parse_numeric_axis(feh_axis_spec, "feh", integer=False)
    turbvel_axis = _parse_turbvel_values(turbvel_axis_spec)
    mu_axis = _parse_numeric_axis(mu_range_spec, "mu", integer=False) if mu_range_spec is not None else None
    if mu_axis is not None and output_mode != "Intensity":
        raise ValueError("mu_range requires output_mode='Intensity'")

    if lam_step <= 0:
        raise ValueError("lam-step must be positive")
    if lam_max <= lam_min:
        raise ValueError("lam-max must be greater than lam-min")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Synthesis config not found: {config_path}")
    config_path = _materialize_synthesis_config(
        base_config_path=config_path,
        run_root=run_root,
        overrides=synthesis_overrides,
        overrides_base_dir=cfg_dir,
        mu_axis=mu_axis,
    )

    resolved_runtime = {
        "run_root": run_root,
        "grid_zarr": grid_zarr,
        "grid_csv": grid_csv,
        "index_parquet": index_parquet,
        "spectra_zarr": spectra_zarr,
        "spectra_shard_template": spectra_shard_template,
        "spectra_shards_dir": spectra_shards_dir,
        "synthesis_config_path": config_path,
        "scratch": scratch,
        "output_tmp": output_tmp,
        "workers": workers,
        "chunk_rows": chunk_rows,
        "log_level": log_level,
        "log_file": log_file,
        "skip_synthesis": bool(skip_synthesis),
        "output_mode": output_mode,
        "calculation_mode": calculation_mode,
    }

    if args.print_runtime_json:
        print(json.dumps(resolved_runtime, sort_keys=True))
        return

    abundances = {
        "a": str(_coalesce(args.a, cfg, ("grid", "abundances", "a"), "+0.00")),
        "c": str(_coalesce(args.c, cfg, ("grid", "abundances", "c"), "+0.00")),
        "n": str(_coalesce(args.n, cfg, ("grid", "abundances", "n"), "+0.00")),
        "o": str(_coalesce(args.o, cfg, ("grid", "abundances", "o"), "+0.00")),
        "r": str(_coalesce(args.r, cfg, ("grid", "abundances", "r"), "+0.00")),
        "s": str(_coalesce(args.s, cfg, ("grid", "abundances", "s"), "+0.00")),
    }
    columns = _build_regular_columns(
        teff_axis=teff_axis,
        logg_axis=logg_axis,
        feh_axis=feh_axis,
        turbvel_axis=turbvel_axis,
        mu_axis=mu_axis,
        grid_version=grid_version,
        lam_min=lam_min,
        lam_max=lam_max,
        lam_step=lam_step,
        output_mode=output_mode,
        mode=mode,
        calculation_mode=calculation_mode,
        abundances=abundances,
        nlte_ascii_selector=nlte_ascii_selector,
        max_rows=max_rows,
    )

    print(
        "[regular-grid] axes: "
        f"teff={teff_axis.size} logg={logg_axis.size} feh={feh_axis.size} turbvel={turbvel_axis.size} "
        f"rows={len(columns['teff']):,}"
    )
    if mu_axis is not None:
        print(
            "[regular-grid] mu axis: "
            f"{float(np.min(mu_axis)):.6f}..{float(np.max(mu_axis)):.6f} "
            f"({mu_axis.size} uniformly spaced samples)"
        )
    if nlte_ascii_selector is not None:
        print(
            "[regular-grid] NLTE ASCII departures: "
            f"dir={nlte_ascii_selector.directory} species={nlte_ascii_selector.species} "
            f"abundance_column={nlte_ascii_selector.abundance_column} "
            f"scale={nlte_ascii_selector.abundance_scale} match={nlte_ascii_selector.match}"
        )
    if scratch:
        print(f"[regular-grid] synthesis scratch: {scratch}")
    print(f"[regular-grid] writing CSV: {grid_csv}")
    print(f"[regular-grid] writing Zarr: {grid_zarr}")
    print(f"[regular-grid] writing index parquet: {index_parquet}")

    sys.path.insert(0, SCRIPT_DIR)
    import generate_grid as gg  # type: ignore

    csv_compression_opt = None if csv_compression == "none" else csv_compression
    gg._write_csv_outputs(columns, grid_csv, compression=csv_compression_opt, level=csv_compression_level)  # type: ignore[attr-defined]
    gg._write_zarr_from_columns(columns, grid_zarr, chunks=zarr_chunks, compressor_cfg=zarr_compressor)  # type: ignore[attr-defined]
    gg._write_index_parquet_from_columns(columns, index_parquet)  # type: ignore[attr-defined]

    if skip_synthesis:
        print("[regular-grid] skip-synthesis requested; done.")
        return

    cmd: List[str] = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "synthesize_spectra_from_zarr.py"),
        "--grid-zarr",
        grid_zarr,
        "--config",
        config_path,
        "--output-zarr",
        spectra_zarr,
        "--chunk-rows",
        str(chunk_rows),
        "--log-level",
        log_level,
        "--generator",
        "turbospectrum_nlte.regular_grid",
    ]
    if workers and workers > 0:
        cmd.extend(["--workers", str(workers)])
    if scratch:
        cmd.extend(["--scratch", scratch])
    if output_tmp:
        cmd.extend(["--output-tmp", output_tmp])
    if log_file:
        cmd.extend(["--log-file", log_file])

    print("[regular-grid] launching synthesis...")
    _run(cmd)
    _verify_continuum_saved(spectra_zarr)
    print("[regular-grid] verified continuum array in synthesized output")
    print(f"[regular-grid] wrote synthesized spectra Zarr: {spectra_zarr}")


if __name__ == "__main__":
    main()
