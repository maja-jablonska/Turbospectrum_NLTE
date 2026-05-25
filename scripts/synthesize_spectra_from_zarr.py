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
import copy
import dataclasses
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_turbospectrum import (  # noqa: E402
    LinelistValidationError,
    TurbospectrumConfig,
    _normalize_config_dict,
    _with_turbospectrum_log_context,
    create_linelist_file,
    determine_worker_count,
    ensure_directories,
    get_synthesis_output_stem_from_params,
    get_synthesis_stem,
    resolve_linelist_paths,
    run_single_synthesis,
    validate_runtime_environment,
)
try:
    from .nlte_ascii_departures import NLTE_ASCII_CONTROL_KEYS
except ImportError:
    from nlte_ascii_departures import NLTE_ASCII_CONTROL_KEYS
from provenance_contract import (  # noqa: E402
    assert_required_provenance_fields,
    canonical_json_sha256,
    compute_binary_manifest_hash,
    compute_grid_definition_hash,
    directory_manifest_sha256,
    file_sha256,
    is_meaningful_provenance_value,
)
from synthesis_validation import SUCCESS_STATUSES, FailFastTracker, validate_synthesis_results
from spectrum_output import extract_flux_and_continuum, infer_flux_metadata


def _read_mu_points(spec_path: str) -> np.ndarray:
    """Parse '# mu-points ...' header from an Intensity .spec file (best-effort)."""
    try:
        header = None
        with open(spec_path, "r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(50):
                line = handle.readline()
                if not line:
                    break
                if not line.lstrip().startswith("#"):
                    break
                if "mu-points" in line.lower():
                    header = line
        if not header:
            return np.asarray([], dtype=np.float32)
        m = re.search(r"mu\s*-\s*points|mu\s*points|mu-points", header, flags=re.IGNORECASE)
        if not m:
            return np.asarray([], dtype=np.float32)
        tail = header[m.end() :]
        num_re = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
        toks = re.findall(num_re, tail)
        if not toks:
            return np.asarray([], dtype=np.float32)
        vals: list[float] = []
        for t in toks:
            try:
                vals.append(float(t.replace("D", "E").replace("d", "e")))
            except ValueError:
                continue
        return np.asarray(vals, dtype=np.float32)
    except Exception:
        return np.asarray([], dtype=np.float32)


def _mu_candidates(mu_points: np.ndarray, cfg: TurbospectrumConfig) -> np.ndarray:
    ms: Mapping[str, Any] = getattr(cfg, "mu_sampling", {}) or {}
    mu_min = float(ms.get("min", 0.0))
    mu_max = float(ms.get("max", 1.0))
    candidates = np.where((mu_points >= mu_min) & (mu_points <= mu_max))[0]
    if candidates.size == 0:
        candidates = np.arange(mu_points.size, dtype=np.int64)
    return np.asarray(candidates, dtype=np.int64)


def _choose_mu_indices(
    mu_points: np.ndarray,
    *,
    row_index: int,
    cfg: TurbospectrumConfig,
    target_mu: float | None = None,
    reduce_mode: str | None = None,
) -> tuple[np.ndarray, float]:
    ms: Mapping[str, Any] = getattr(cfg, "mu_sampling", {}) or {}
    mode = str(ms.get("mode", "none")).lower()
    if mu_points.size == 0:
        return np.asarray([], dtype=np.int64), float("nan")

    # When no explicit mu_sampling mode is configured but a target_mu is
    # available from the grid, default to "nearest" so the correct mu column
    # is selected and mu_selected is populated (instead of NaN).
    if mode == "none" and target_mu is not None:
        mode = "nearest"

    count = int(ms.get("count", 1) or 1)
    candidates = _mu_candidates(mu_points, cfg)
    if mode in {"nearest", "target"} and target_mu is not None:
        distances = np.abs(mu_points[candidates] - float(target_mu))
        order = np.lexsort((candidates, distances))
        ranked = candidates[order]
        chosen = ranked[:count] if count <= ranked.size else np.resize(ranked, count)
    elif mode == "random":
        seed = ms.get("seed")
        base_seed = 0 if seed in (None, "") else int(seed)
        rng = np.random.default_rng((base_seed + int(row_index)) % (2**32))
        replace = bool(count > candidates.size)
        chosen = rng.choice(candidates, size=count, replace=replace)
    else:
        return np.asarray([], dtype=np.int64), float("nan")

    chosen = np.asarray(chosen, dtype=np.int64)
    mu_sel = mu_points[chosen]
    # mu_summary must match the mu actually used by extract_flux_and_continuum:
    # reduce="mean" averages all chosen angles; "first" (default) uses chosen[0].
    effective_reduce = str(reduce_mode if reduce_mode is not None else ms.get("reduce", "first")).lower()
    if mu_sel.size == 1:
        mu_summary = float(mu_sel[0])
    elif effective_reduce == "mean":
        mu_summary = float(np.mean(mu_sel))
    else:
        mu_summary = float(mu_sel[0])
    return chosen, mu_summary


def _estimate_mu_from_index(
    chosen_idx: np.ndarray,
    n_mu: int,
    ms_cfg: Mapping[str, Any],
    target_mu: float | None,
) -> float:
    """Estimate the mu value when the spectrum header is missing.

    Uses *target_mu* when available (nearest/target mode).  Otherwise
    linearly interpolates from the column index across [mu_min, mu_max].
    """
    if target_mu is not None:
        return float(target_mu)
    if chosen_idx.size == 0 or n_mu <= 0:
        return float("nan")
    mu_min = float(ms_cfg.get("min", 0.0))
    mu_max = float(ms_cfg.get("max", 1.0))
    idx = int(chosen_idx[0])
    if n_mu == 1:
        return float((mu_min + mu_max) / 2.0)
    return float(mu_min + (mu_max - mu_min) * idx / (n_mu - 1))


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "configs", "synthesis", "config_sample_comprehensive.json")
)
DEFAULT_OUTPUT_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "runs", "local-dev", "outputs", "zarr", "synthesized_spectra.zarr")
)

_SUCCESS_STATUSES = {"success", "skipped"}

_WORKER_CONFIG: TurbospectrumConfig | None = None


def _init_worker(config: TurbospectrumConfig) -> None:
    """Initialize worker-local config and constrain thread oversubscription."""
    global _WORKER_CONFIG
    _WORKER_CONFIG = config

    # Prevent BLAS/OpenMP oversubscription inside each worker process.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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


try:
    from zarr_compat import (
        zarr_store as _zarr_store,
        compression_kwargs as _zarr_compression_kwargs_base,
        create_root_group as _open_root_group,
        create_array as _create_array_compat,
        write_string_scalar as _zc_write_string_scalar,
        write_fixed_string_scalar as _zc_write_fixed_string_scalar,
        write_string_array as _zc_write_string_array,
    )
except ImportError:
    from .zarr_compat import (
        zarr_store as _zarr_store,
        compression_kwargs as _zarr_compression_kwargs_base,
        create_root_group as _open_root_group,
        create_array as _create_array_compat,
        write_string_scalar as _zc_write_string_scalar,
        write_fixed_string_scalar as _zc_write_fixed_string_scalar,
        write_string_array as _zc_write_string_array,
    )


def _zarr_compression_kwargs(zarr_compressor_cfg: Mapping):
    return _zarr_compression_kwargs_base(zarr_compressor_cfg)


def _to_float32(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    for i, v in enumerate(values.tolist()):
        try:
            out[i] = np.float32(float(v))
        except Exception:
            s = str(v).strip().lower()
            if s.startswith("t"):
                s = s[1:]
            try:
                out[i] = np.float32(float(s))
            except Exception:
                out[i] = np.nan
    return out


def _ordered_param_names(column_data: Mapping[str, np.ndarray]) -> List[str]:
    reserved = {
        "grid_version",
        "lam_min",
        "lam_max",
        "lam_step",
        "output_mode",
        "mode",
        "calculation_mode",
        "turb",
        *NLTE_ASCII_CONTROL_KEYS,
    }
    candidate_order = ["teff", "logg", "feh", "vmicro", "turbvel", "t_value", "a", "c", "n", "o", "r", "s"]
    extras = sorted(
        name
        for name in column_data.keys()
        if name not in reserved and name not in {"teff", "logg", "feh", "turbvel", "t_value", "a", "c", "n", "o", "r", "s"}
    )
    return candidate_order + extras


def _build_params_matrix(column_data: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Fixed ordering keeps schema stable across runs.
    candidate_order = _ordered_param_names(column_data)
    params_by_name: Dict[str, np.ndarray] = {}

    for name in ("teff", "logg", "feh"):
        if name in column_data:
            params_by_name[name] = _to_float32(np.asarray(column_data[name]))

    if "turb" in column_data:
        params_by_name["vmicro"] = _to_float32(np.asarray(column_data["turb"]))
    elif "turbvel" in column_data:
        params_by_name["vmicro"] = _to_float32(np.asarray(column_data["turbvel"]))
    elif "t_value" in column_data:
        params_by_name["vmicro"] = _to_float32(np.asarray(column_data["t_value"]))

    if "turbvel" in column_data:
        params_by_name["turbvel"] = _to_float32(np.asarray(column_data["turbvel"]))
    if "t_value" in column_data:
        params_by_name["t_value"] = _to_float32(np.asarray(column_data["t_value"]))

    for name in candidate_order:
        if name in {"teff", "logg", "feh", "vmicro", "turbvel", "t_value"}:
            continue
        if name in column_data:
            params_by_name[name] = _to_float32(np.asarray(column_data[name]))

    param_names = [name for name in candidate_order if name in params_by_name]
    if not param_names:
        raise ValueError("Unable to build params matrix: no parameter columns available")

    params = np.column_stack([params_by_name[name] for name in param_names]).astype(np.float32, copy=False)
    return params, np.asarray(param_names, dtype=object)


def _compute_model_ids(params: np.ndarray) -> np.ndarray:
    ids = np.zeros(params.shape[0], dtype=np.uint64)
    for i in range(params.shape[0]):
        row = np.nan_to_num(params[i].astype(np.float32, copy=False), nan=9.96921e36, posinf=3.4e38, neginf=-3.4e38)
        digest = hashlib.sha256(row.astype("<f4", copy=False).tobytes()).digest()
        ids[i] = np.uint64(int.from_bytes(digest[:8], "big", signed=False))
    return ids


def _git_commit(project_root: str) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL, timeout=3)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _resolve_linelist_paths(config: TurbospectrumConfig) -> List[str]:
    return resolve_linelist_paths(str(config.linelist_path), config.linelist_files)


def _resolve_synthesis_binary_paths(config: TurbospectrumConfig, project_root: str) -> List[str]:
    candidates = [config.babsma_exec, config.bsyn_exec, config.interpol_exec]
    exec_root = os.path.join(project_root, f"exec-{config.compiler}")
    out: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        raw = str(item or "").strip()
        if not raw:
            continue
        path = raw if os.path.isabs(raw) else os.path.abspath(os.path.join(exec_root, raw))
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        out.append(path)
    return out


def _capture_environment_text(mode: str) -> str:
    mode_norm = str(mode or "pip_freeze").strip().lower()
    attempts: List[Tuple[List[str], str]] = []
    if mode_norm in {"pip_freeze", "auto"}:
        attempts.append(([sys.executable, "-m", "pip", "freeze"], "pip_freeze"))
    if mode_norm in {"conda_env_export", "auto"}:
        attempts.append((["conda", "env", "export"], "conda_env_export"))
    if not attempts:
        attempts.append(([sys.executable, "-m", "pip", "freeze"], "pip_freeze"))

    errors: List[str] = []
    for cmd, label in attempts:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30)
            txt = out.decode("utf-8", errors="replace").strip()
            if txt:
                return txt
            errors.append(f"{label}: command succeeded but returned empty output")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")

    return "\n".join(
        [
            "# Environment capture fallback (failed to run pip/conda export)",
            *[f"# {err}" for err in errors],
            f"python={sys.version}",
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}",
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '')}",
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '')}",
            f"VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS', '')}",
        ]
    )


def _count_matching_files(root_path: str, suffix: str) -> int:
    try:
        total = 0
        for _dirpath, _dirnames, filenames in os.walk(root_path):
            total += sum(1 for fname in filenames if fname.lower().endswith(suffix.lower()))
        return int(total)
    except Exception:
        return 0


def _compute_physics_hash(config: TurbospectrumConfig, column_data: Mapping[str, np.ndarray]) -> str:
    def _uniq(name: str) -> List[str]:
        if name not in column_data:
            return []
        return sorted({str(x) for x in np.asarray(column_data[name]).tolist()})

    payload = {
        "compiler": str(config.compiler),
        "nlte_default": bool(config.nlte),
        "linelist_path": str(config.linelist_path),
        "linelist_files": [str(x) for x in (config.linelist_files or [])],
        "linelist_version": str(getattr(config, "linelist_version", "")),
        "linelist_sha256": str(getattr(config, "linelist_sha256", "")),
        "linelist_preprocessing": str(getattr(config, "linelist_preprocessing", "")),
        "model_atmosphere_path": str(config.model_atmosphere_path),
        "atmosphere_geometry": str(getattr(config, "atmosphere_geometry", "")),
        "atmosphere_version": str(getattr(config, "atmosphere_version", "")),
        "atmosphere_sha256": str(getattr(config, "atmosphere_sha256", "")),
        "model_opac_dir": str(config.model_opac_dir),
        "synthesis_code_version": str(getattr(config, "synthesis_code_version", "")),
        "spice_version": str(getattr(config, "spice_version", "")),
        "mu_sampling": getattr(config, "mu_sampling", {}) or {},
        "wavelength": {
            "lam_min": _uniq("lam_min"),
            "lam_max": _uniq("lam_max"),
            "lam_step": _uniq("lam_step"),
        },
        "output_mode": _uniq("output_mode"),
        "calculation_mode": _uniq("calculation_mode"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_string_scalar(root, name: str, value: str, compression_kwargs: Mapping[str, Any]) -> None:
    _zc_write_string_scalar(root, name, value, compression_kw=compression_kwargs)


def _write_fixed_string_scalar(root, name: str, value: str, min_width: int, compression_kwargs: Mapping[str, Any]) -> None:
    _zc_write_fixed_string_scalar(root, name, value, min_width=min_width, compression_kw=compression_kwargs)


def _write_parameter_columns(
    root,
    *,
    params: np.ndarray,
    param_names: Sequence[str],
    chunk_rows: int,
    compression_kwargs: Mapping[str, Any],
) -> None:
    """Expose packed params as named 1D arrays for easier downstream lookup."""
    group = root.create_group("parameter_columns")
    for col_idx, name in enumerate(param_names):
        values = params[:, col_idx].astype(np.float32, copy=False)
        _create_array_compat(
            group,
            str(name),
            data=values,
            chunks=(min(chunk_rows, len(values)) if len(values) else 1,),
            **compression_kwargs,
        )


def _finalize_zarr_store(write_path: str, final_path: str, *, logger: logging.Logger, label: str) -> None:
    """Move a completed Zarr directory into place."""
    if write_path == final_path:
        return

    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    try:
        os.rename(write_path, final_path)
        logger.info("%s written to %s (atomic rename)", label, final_path)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    logger.warning(
        "Cross-filesystem move for %s detected; falling back to copy+delete from %s to %s",
        label.lower(),
        write_path,
        final_path,
    )
    shutil.move(write_path, final_path)
    logger.info("%s written to %s (copy+delete fallback)", label, final_path)


def _to_u32_param_names(values: Sequence[str]) -> np.ndarray:
    names = [str(v) for v in values]
    too_long = [n for n in names if len(n) > 32]
    if too_long:
        raise ValueError(
            "param_names entries must be <= 32 characters for DATA_SCHEMA.md U32 storage; "
            f"offending values: {too_long[:3]}"
        )
    return np.asarray(names, dtype="<U32")


def _load_config(config_path: str, project_root: str) -> TurbospectrumConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg_data = json.load(handle)
    cfg_data = _normalize_config_dict(cfg_data, default_project_root=project_root)
    accepted_fields = {fld.name for fld in dataclasses.fields(TurbospectrumConfig)}
    cfg_data = {k: v for k, v in cfg_data.items() if k in accepted_fields}
    cfg_project_root = cfg_data.get("project_root")
    cfg_project_root_abs = ""
    if cfg_project_root not in (None, ""):
        cfg_project_root_abs = str(cfg_project_root).strip()
        if cfg_project_root_abs and not os.path.isabs(cfg_project_root_abs):
            cfg_project_root_abs = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), cfg_project_root_abs))
    if not cfg_project_root_abs or not os.path.isdir(cfg_project_root_abs):
        cfg_data["project_root"] = project_root
    else:
        cfg_data["project_root"] = cfg_project_root_abs
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
    present_turb_columns = [col for col in optional_turb if col in available]
    if not present_turb_columns:
        raise KeyError("Grid Zarr must include either 'turbvel' or 't_value' for microturbulence selection")
    for name in present_turb_columns:
        column_data[name] = np.array(grid_root[name][:])

    passthrough_columns = sorted(
        name
        for name in available
        if name not in set(required_columns + wavelength_columns + optional_turb)
    )
    for name in passthrough_columns:
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


_ROW_NON_ABUNDANCE_KEYS = {
    "teff",
    "logg",
    "feh",
    "turbvel",
    "t_value",
    "lam_min",
    "lam_max",
    "lam_step",
    "mu",
    "output_mode",
    "calculation_mode",
    "mode",
    "grid_version",
}


def _row_abundance_values(row_values: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = _ROW_NON_ABUNDANCE_KEYS | set(NLTE_ASCII_CONTROL_KEYS)
    return {key: value for key, value in row_values.items() if key not in excluded}


def _prepare_cfg_for_row(row_values: Mapping[str, Any]):
    """Build a per-row config copy; resolve intensity + target mu + mu mode."""
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker config not initialized")
    cfg = copy.deepcopy(_WORKER_CONFIG)
    cfg.lambda_min = float(row_values["lam_min"])
    cfg.lambda_max = float(row_values["lam_max"])
    cfg.lambda_step = float(row_values["lam_step"])
    # If the grid provides per-row mode flags, honor them; otherwise fall back
    # to whatever the Turbospectrum config requested.
    output_mode = row_values.get("output_mode")
    if output_mode is None:
        output_mode = getattr(cfg, "output_mode", "Flux")
    calculation_mode = row_values.get("calculation_mode")
    if calculation_mode is None:
        calculation_mode = "NLTE" if cfg.nlte else "LTE"
    output_mode = str(output_mode)
    calculation_mode = str(calculation_mode)

    cfg.output_mode = output_mode
    cfg.nlte = calculation_mode.lower() == "nlte"
    is_intensity = output_mode.lower() == "intensity"
    target_mu_raw = row_values.get("mu")
    target_mu = None if target_mu_raw in (None, "") else float(target_mu_raw)
    mu_sampling = getattr(cfg, "mu_sampling", {}) or {}
    if not isinstance(mu_sampling, dict):
        mu_sampling = {}
    if is_intensity and str(mu_sampling.get("mode", "none")).strip().lower() in {"", "none"}:
        mu_sampling["mode"] = "nearest" if target_mu is not None else "random"
    cfg.mu_sampling = mu_sampling
    return cfg, is_intensity, target_mu


def _resolve_spec_path(cfg, row_values, result, is_intensity):
    log_path = str(result.get("log_path", "") or "")
    spec_path = str(result.get("output_path", "") or "")
    if not spec_path:
        base_name = str(result.get("base_name", "") or get_synthesis_stem(
            int(row_values["teff"]),
            float(row_values["logg"]),
            float(row_values["feh"]),
            str(row_values.get("t_value") or row_values.get("turbvel") or "01").strip(),
            _row_abundance_values(row_values),
        ))
        suffix = ".intensity.spec" if is_intensity else ".spec"
        spec_path = os.path.join(cfg.output_dir, f"{os.path.splitext(base_name)[0]}{suffix}")
    base_name = str(result.get("base_name", "") or os.path.splitext(os.path.basename(spec_path))[0])
    return spec_path, base_name, log_path


def _extract_row_result(
    index,
    cfg,
    is_intensity,
    target_mu,
    result,
    duration,
    spec_path,
    base_name,
    log_path,
) -> Dict:
    """Read the (already-synthesised) spectrum file and build this row's result.

    The synthesis itself is performed once per output stem; for Intensity grids
    that share a mu axis this is called once per mu-row against the same
    ``spec_path``, selecting that row's own target mu.
    """
    spectrum = None
    mu_selected = float("nan")
    mu_selected_index = -1
    if os.path.exists(spec_path):
        try:
            try:
                if os.path.getsize(spec_path) == 0:
                    raise ValueError("Spectrum file is empty (0 bytes)")
            except OSError:
                pass

            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=r"loadtxt: input contained no data.*")
                data = np.loadtxt(spec_path)

            if data.size == 0:
                raise ValueError("Spectrum file contains no numeric data")
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError(f"Unexpected spectrum shape {getattr(data, 'shape', None)}")

            expected_n = int(round((cfg.lambda_max - cfg.lambda_min) / cfg.lambda_step)) + 1
            if data.shape[0] != expected_n:
                raise ValueError(f"Unexpected wavelength count {data.shape[0]} (expected {expected_n})")

            if is_intensity:
                mu_points = _read_mu_points(spec_path)
                ms_cfg: Mapping[str, Any] = getattr(cfg, "mu_sampling", {}) or {}
                mu_mode = str(ms_cfg.get("mode", "none")).lower()
                reduce_mode = str(ms_cfg.get("reduce", "first")).lower()
                chosen_idx, mu_selected = _choose_mu_indices(
                    mu_points,
                    row_index=int(index),
                    cfg=cfg,
                    target_mu=target_mu,
                    reduce_mode=reduce_mode,
                )

                # Fallback when mu-points header is missing / unparseable.
                if chosen_idx.size == 0:
                    n_mu = max(0, int((data.shape[1] - 3) // 2))
                    if n_mu > 0:
                        mu_min = float(ms_cfg.get("min", 0.0))
                        mu_max = float(ms_cfg.get("max", 1.0))
                        count = int(ms_cfg.get("count", 1) or 1)
                        if mu_mode in {"nearest", "target"} and target_mu is not None:
                            denom = mu_max - mu_min
                            frac = 0.0 if denom <= 0 else float(np.clip((target_mu - mu_min) / denom, 0.0, 1.0))
                            ranked = np.argsort(np.abs(np.arange(n_mu, dtype=np.float64) - frac * max(0, n_mu - 1)))
                            chosen_idx = np.asarray(ranked[:count] if count <= ranked.size else np.resize(ranked, count), dtype=np.int64)
                        else:
                            seed = ms_cfg.get("seed")
                            base_seed = 0 if seed in (None, "") else int(seed)
                            rng = np.random.default_rng((base_seed + int(index)) % (2**32))
                            replace = bool(count > n_mu)
                            chosen_idx = np.asarray(rng.choice(np.arange(n_mu), size=count, replace=replace), dtype=np.int64)
                        # Estimate mu from column position when header is absent.
                        mu_selected = _estimate_mu_from_index(chosen_idx, n_mu, ms_cfg, target_mu)
                        logging.getLogger("zarr_synthesis").warning(
                            "mu-points header missing from %s; estimated mu_selected=%.6g",
                            spec_path,
                            mu_selected,
                        )

                if chosen_idx.size > 0:
                    mu_selected_index = int(chosen_idx[0])
                flux, cont = extract_flux_and_continuum(
                    data,
                    is_intensity=is_intensity,
                    chosen_idx=chosen_idx,
                    reduce_mode=reduce_mode,
                )
            else:
                flux, cont = extract_flux_and_continuum(
                    data,
                    is_intensity=is_intensity,
                )
            spectrum = (flux, cont)
        except Exception as exc:  # noqa: BLE001
            return {
                "index": index,
                "base_name": base_name,
                "status": "error",
                "message": _with_turbospectrum_log_context(
                    f"Failed to read spectrum {spec_path}: {exc}",
                    log_path,
                ),
                "duration": duration,
                "spectrum": None,
                "mu_selected": float("nan"),
                "mu_selected_index": -1,
            }
    elif str(result.get("status", "")).lower() in _SUCCESS_STATUSES:
        return {
            "index": index,
            "base_name": base_name,
            "status": "error",
            "message": _with_turbospectrum_log_context(
                f"Missing spectrum output: {spec_path}",
                log_path,
            ),
            "duration": duration,
            "spectrum": None,
            "mu_selected": float("nan"),
            "mu_selected_index": -1,
        }

    return {
        "index": index,
        "base_name": base_name,
        "status": result["status"],
        "message": result["message"],
        "duration": duration,
        "spectrum": spectrum,
        "mu_selected": float(mu_selected),
        "mu_selected_index": int(mu_selected_index),
    }


def _synthesis_group_task(group) -> List[Dict]:
    """Synthesise one output stem once, then extract every row in the group.

    A group is a list of ``(index, row_values)`` tasks that all map to the same
    Turbospectrum output stem (identical physics; for Intensity grids that share
    a mu axis the stem is mu-independent so the rows differ only in target mu).
    bsyn is run a single time (computing all mu columns); each row then selects
    its own mu from that shared spectrum. This avoids the redundant re-synthesis
    and the concurrent writes to the same .opac/.spec that one-task-per-row caused.
    """
    rep_index, rep_row = group[0]
    cfg, is_intensity, target_mu = _prepare_cfg_for_row(rep_row)
    start = time.perf_counter()
    result = run_single_synthesis((rep_row, cfg))
    duration = time.perf_counter() - start
    spec_path, base_name, log_path = _resolve_spec_path(cfg, rep_row, result, is_intensity)

    results = [
        _extract_row_result(
            rep_index, cfg, is_intensity, target_mu, result, duration, spec_path, base_name, log_path
        )
    ]
    if len(group) == 1:
        return results

    synth_ok = str(result.get("status", "")).lower() in _SUCCESS_STATUSES
    for index, row_values in group[1:]:
        if not synth_ok:
            # The shared synthesis failed; report the same failure per row rather
            # than re-attempting (which would just fail again).
            results.append(
                {
                    "index": index,
                    "base_name": base_name,
                    "status": result.get("status", "error"),
                    "message": result.get("message", "shared synthesis failed"),
                    "duration": 0.0,
                    "spectrum": None,
                    "mu_selected": float("nan"),
                    "mu_selected_index": -1,
                }
            )
            continue
        member_cfg, member_is_intensity, member_target_mu = _prepare_cfg_for_row(row_values)
        # Reuse the representative synthesis result (status/output_path); only the
        # per-row mu selection differs. No re-synthesis -> no redundancy, no race.
        results.append(
            _extract_row_result(
                index,
                member_cfg,
                member_is_intensity,
                member_target_mu,
                result,
                0.0,
                spec_path,
                base_name,
                log_path,
            )
        )
    return results


def _synthesis_task(args) -> Dict:
    """Synthesise and extract a single row (thin wrapper over the group path)."""
    return _synthesis_group_task([args])[0]


def _write_zarr_output(
    output_path: str,
    wavelengths: np.ndarray,
    fluxes: np.ndarray,
    continua: np.ndarray,
    mu_selected: np.ndarray,
    mu_selected_index: np.ndarray,
    params: np.ndarray,
    param_names: np.ndarray,
    model_id: np.ndarray,
    physics_hash: str,
    schema_version: str,
    created_utc: str,
    git_commit: str,
    contact: str,
    generator: str,
    flux_definition: str,
    flux_unit: str,
    provenance_payload: Mapping[str, str],
    contract_provenance: Mapping[str, str],
    status_counts: Mapping[str, int],
    compression_cfg: Mapping,
    chunk_rows: int,
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    store = _zarr_store(output_path)
    root = _open_root_group(store)

    compression_kwargs = _zarr_compression_kwargs(compression_cfg)
    chunk_shape = (min(chunk_rows, fluxes.shape[0]), fluxes.shape[1]) if fluxes.shape[0] else (1, fluxes.shape[1])
    param_chunk_shape = (min(chunk_rows, params.shape[0]), params.shape[1]) if params.shape[0] else (1, params.shape[1])
    wl = wavelengths.astype(np.float32, copy=False)
    param_name_list = [str(x) for x in param_names.tolist()]
    param_names_u32 = _to_u32_param_names(param_name_list)

    # DATA_SCHEMA.md synthesis layout
    _create_array_compat(root, "wavelength", data=wl, chunks=wl.shape if wl.size else (1,), **compression_kwargs)
    _create_array_compat(root, "flux", data=fluxes, chunks=chunk_shape, **compression_kwargs)
    _create_array_compat(root, "continuum", data=continua, chunks=chunk_shape, **compression_kwargs)
    _create_array_compat(
        root,
        "mu_selected",
        data=mu_selected.astype(np.float32, copy=False),
        chunks=(min(chunk_rows, len(mu_selected)) if len(mu_selected) else 1,),
        **compression_kwargs,
    )
    _create_array_compat(
        root,
        "mu_selected_index",
        data=mu_selected_index.astype(np.int16, copy=False),
        chunks=(min(chunk_rows, len(mu_selected_index)) if len(mu_selected_index) else 1,),
        **compression_kwargs,
    )
    _create_array_compat(root, "params", data=params.astype(np.float32, copy=False), chunks=param_chunk_shape, **compression_kwargs)
    _create_array_compat(
        root,
        "param_names",
        data=param_names_u32,
        chunks=(min(max(1, params.shape[1]), len(param_names_u32)) if len(param_names_u32) else 1,),
        **compression_kwargs,
    )
    _create_array_compat(
        root,
        "model_id",
        data=model_id.astype(np.uint64, copy=False),
        chunks=(min(chunk_rows, len(model_id)) if len(model_id) else 1,),
        **compression_kwargs,
    )
    _write_parameter_columns(
        root,
        params=params,
        param_names=param_name_list,
        chunk_rows=chunk_rows,
        compression_kwargs=compression_kwargs,
    )
    _write_fixed_string_scalar(root, "physics_hash", physics_hash, min_width=64, compression_kwargs=compression_kwargs)
    _write_fixed_string_scalar(root, "schema_version", schema_version, min_width=16, compression_kwargs=compression_kwargs)

    # Minimal provenance group requested by DATA_SCHEMA.md
    prov = root.create_group("provenance")
    for name, value in provenance_payload.items():
        _write_string_scalar(prov, name, value, compression_kwargs=compression_kwargs)

    param_units = {}
    for name in param_name_list:
        if name == "teff":
            param_units[name] = "K"
        elif name in {"logg", "feh", "a", "c", "n", "o", "r", "s"} or name.isalpha():
            param_units[name] = "dex"
        elif name == "vmicro":
            param_units[name] = "km/s"
        else:
            param_units[name] = ""

    attrs_payload: Dict[str, Any] = {
        "title": "SPICE Synthetic Spectral Grid",
        "generator": generator,
        "created_utc": created_utc,
        "synthesis_timestamp": created_utc,
        "flux_definition": flux_definition,
        "wavelength_unit": "angstrom",
        "flux_unit": flux_unit,
        "parameter_units": param_units,
        "parameter_columns_group": "parameter_columns",
        "physics_hash": physics_hash,
        "git_commit": git_commit,
        "git_sha": git_commit,
        "contact": contact,
        "schema_version": schema_version,
        "n_models": int(fluxes.shape[0]),
        "n_lambda": int(fluxes.shape[1]) if fluxes.ndim == 2 else 0,
        "n_params": int(params.shape[1]) if params.ndim == 2 else 0,
        # Keep diagnostics in attrs (not top-level arrays) to preserve schema shape.
        "status_counts": dict(status_counts),
    }
    attrs_payload.update({str(k): str(v) for k, v in contract_provenance.items()})
    root.attrs.update(attrs_payload)
    logger.info(
        "Wrote spectra to %s (flux_shape=%s continuum_shape=%s)",
        os.path.abspath(output_path),
        fluxes.shape,
        continua.shape,
    )


def _build_tasks(row_count: int, column_data: Mapping[str, np.ndarray], base_config: TurbospectrumConfig):
    tasks = []
    base_name_counts: Dict[str, int] = {}
    for idx in range(row_count):
        row_values = {
            "teff": column_data["teff"][idx],
            "logg": column_data["logg"][idx],
            "feh": column_data["feh"][idx],
            "lam_min": column_data["lam_min"][idx],
            "lam_max": column_data["lam_max"][idx],
            "lam_step": column_data["lam_step"][idx],
        }
        for optional_key in ("turbvel", "t_value", "output_mode", "calculation_mode", "mode"):
            if optional_key in column_data:
                row_values[optional_key] = column_data[optional_key][idx]
        for control_key in NLTE_ASCII_CONTROL_KEYS:
            if control_key in column_data:
                row_values[control_key] = column_data[control_key][idx]
        for passthrough_key, values in column_data.items():
            if passthrough_key in {
                "teff",
                "logg",
                "feh",
                "turbvel",
                "t_value",
                "lam_min",
                "lam_max",
                "lam_step",
                "output_mode",
                "calculation_mode",
                "mode",
                "grid_version",
                *NLTE_ASCII_CONTROL_KEYS,
            }:
                continue
            row_values[passthrough_key] = values[idx]
        abundance_values = {
            key: value
            for key, value in row_values.items()
            if key
            not in {
                "teff",
                "logg",
                "feh",
                "turbvel",
                "t_value",
                "lam_min",
                "lam_max",
                "lam_step",
                "output_mode",
                "calculation_mode",
                "mode",
                "grid_version",
                *NLTE_ASCII_CONTROL_KEYS,
            }
        }
        base_name = get_synthesis_output_stem_from_params(
            {**row_values, **abundance_values},
            default_output_mode=getattr(base_config, "output_mode", "Flux"),
            default_calculation_mode="NLTE" if base_config.nlte else "LTE",
        )
        base_name_counts[base_name] = base_name_counts.get(base_name, 0) + 1
        tasks.append((idx, row_values))
    duplicate_base_names = [name for name, count in base_name_counts.items() if count > 1]
    if duplicate_base_names:
        sample = ", ".join(sorted(duplicate_base_names)[:3])
        raise ValueError(
            "Grid rows collapse to duplicate Turbospectrum base filenames, which can overwrite/reuse spectra "
            f"(examples: {sample}). Coarsen grid precision (logg/feh) or adjust naming."
        )
    return tasks


def _group_tasks(tasks, base_config: TurbospectrumConfig):
    """Group tasks that map to the same Turbospectrum output stem.

    Each group is synthesised exactly once. Only Intensity rows driven by a
    shared mu axis (``mu_sampling.points``) collapse together — they share one
    mu-independent ``.intensity.spec`` and differ only in target mu. Every other
    row (Flux, or per-row LHS mu, whose stem already encodes mu) is its own
    group, preserving the previous one-task-per-row behaviour.
    """
    ms = getattr(base_config, "mu_sampling", {}) or {}
    shared_mu_axis = isinstance(ms, Mapping) and bool(ms.get("points"))
    if not shared_mu_axis:
        return [[task] for task in tasks]

    groups: "Dict[str, List]" = {}
    singletons: List = []
    for task in tasks:
        _, row_values = task
        output_mode = str(row_values.get("output_mode") or getattr(base_config, "output_mode", "Flux"))
        if output_mode.lower() != "intensity":
            singletons.append([task])
            continue
        # The output stem run_single_synthesis writes for the shared-mu case is
        # mu-independent, so dropping "mu" yields the same key for a point's rows.
        key_row = {k: v for k, v in row_values.items() if k != "mu"}
        stem = get_synthesis_output_stem_from_params(
            key_row,
            default_output_mode=getattr(base_config, "output_mode", "Flux"),
            default_calculation_mode="NLTE" if base_config.nlte else "LTE",
        )
        groups.setdefault(stem, []).append(task)
    return list(groups.values()) + singletons


def _validate_synthesis_results(
    *,
    statuses: Sequence[str],
    messages: Sequence[str],
    fluxes: np.ndarray,
    continua: np.ndarray,
    max_failures: int = 0,
) -> Dict[str, int]:
    return validate_synthesis_results(
        statuses=statuses,
        messages=messages,
        fluxes=fluxes,
        continua=continua,
        max_failures=max_failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-zarr", required=True, help="Input Zarr grid path")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to Turbospectrum JSON config")
    parser.add_argument("--output-zarr", default=DEFAULT_OUTPUT_PATH, help="Final output Zarr path (after atomic rename if --output-tmp used)")
    parser.add_argument(
        "--output-tmp",
        default=None,
        metavar="TMP_PATH",
        help="Temp path for atomic write: write to TMP_PATH, then rename to --output-zarr. Prevents partial outputs.",
    )
    parser.add_argument("--scratch", default=None, help="Optional node-local scratch dir to reduce shared FS I/O")
    parser.add_argument("--no-scratch", action="store_true", help="Disable automatic scratch override (e.g. to use paths from config)")
    parser.add_argument("--workers", type=int, default=None, help="Override worker process count")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--chunk-rows", type=int, default=128, help="Zarr chunking along the sample dimension")
    parser.add_argument("--compressor", default=None, help="JSON string describing compressor options (cname, clevel, shuffle)")
    parser.add_argument("--schema-version", default="1.0.0", help="DATA_SCHEMA.md schema version")
    parser.add_argument("--physics-hash", default=None, help="Optional override for physics hash")
    parser.add_argument("--contact", default=os.environ.get("SPICE_CONTACT", "unknown"), help="Contact metadata")
    parser.add_argument("--generator", default="turbospectrum_nlte", help="Generator string for metadata")
    parser.add_argument(
        "--flux-definition",
        default="auto",
        help="Flux definition metadata. Use 'auto' to infer from output_mode.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Maximum number of failed rows to tolerate before aborting the write (default: 0 = abort on any failure).",
    )
    parser.add_argument(
        "--fail-fast-after",
        type=int,
        default=10,
        help=(
            "Exit the processing loop early after this many consecutive failures "
            "(default: 10). Set to 0 to disable early exit."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-provenance",
        action="store_true",
        help="Allow writing datasets even when required provenance contract fields are missing.",
    )
    args = parser.parse_args()

    logger = _configure_logging(args.log_level, args.log_file)
    t0 = time.perf_counter()
    logger.info("Starting synthesis run: grid=%s config=%s", os.path.abspath(args.grid_zarr), os.path.abspath(args.config))
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    config = _load_config(args.config, project_root=project_root)
    # Verbatim total input config exactly as supplied via --config, before
    # normalization or filtering to TurbospectrumConfig fields. Recorded in
    # provenance so the dataset stays reproducible from the original input.
    with open(args.config, "r", encoding="utf-8") as _cfg_handle:
        raw_config_text = _cfg_handle.read()
    logger.info("mu_sampling=%s", json.dumps(getattr(config, "mu_sampling", {}) or {}, sort_keys=True))
    if args.scratch and not args.no_scratch:
        scratch = os.path.abspath(args.scratch)
        os.makedirs(scratch, exist_ok=True)
        # Redirect I/O-heavy directories to scratch to reduce shared FS contention.
        config.tmp_dir = os.path.join(scratch, "tmp")
        config.log_dir = os.path.join(scratch, "logs")
        config.output_dir = os.path.join(scratch, "spectra")
        config.model_opac_dir = os.path.join(scratch, "opac")
        logger.info(
            "Scratch override — effective paths: output_dir=%s  tmp_dir=%s  log_dir=%s",
            config.output_dir, config.tmp_dir, config.log_dir,
        )
    ensure_directories(config)
    validate_runtime_environment(config)
    try:
        config.linelist_file_path = create_linelist_file(config)
    except LinelistValidationError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    grid_store = _zarr_store(args.grid_zarr)
    grid_root = zarr.open_group(store=grid_store, mode="r")
    row_count, column_data = _validate_grid(grid_root)
    wavelengths, expected_points = _expected_wavelengths(column_data)
    logger.info("Grid rows=%d wavelength_points=%d", row_count, expected_points)
    output_mode_values: List[str] = []
    if "output_mode" in column_data:
        output_mode_values = sorted({str(x) for x in np.unique(column_data["output_mode"])})
        logger.info("Grid output_mode values: %s", output_mode_values)
    if "calculation_mode" in column_data:
        unique_calc = sorted({str(x) for x in np.unique(column_data["calculation_mode"])})
        logger.info("Grid calculation_mode values: %s", unique_calc)

    fluxes = np.full((row_count, expected_points), np.nan, dtype=np.float32)
    continua = np.full_like(fluxes, np.nan)
    statuses: List[str] = ["pending"] * row_count
    messages: List[str] = [""] * row_count
    mu_selected = np.full(row_count, np.nan, dtype=np.float32)
    mu_selected_index = np.full(row_count, -1, dtype=np.int16)

    tasks = _build_tasks(row_count, column_data, config)
    groups = _group_tasks(tasks, config)
    if len(groups) < len(tasks):
        logger.info(
            "Deduplicated synthesis: %d rows -> %d unique output stems (shared mu axis)",
            len(tasks),
            len(groups),
        )
    worker_count = determine_worker_count(config, requested_workers=args.workers)

    compressor_cfg: Dict[str, object] = {}
    if args.compressor:
        try:
            compressor_cfg = json.loads(args.compressor)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid compressor JSON: {exc}") from exc

    tracker = FailFastTracker(threshold=args.fail_fast_after)
    early_exit = False

    def _record_result(result: Dict) -> None:
        idx = int(result["index"])
        statuses[idx] = result["status"]
        messages[idx] = result["message"]
        try:
            mu_selected[idx] = float(result.get("mu_selected", np.nan))
        except Exception:
            mu_selected[idx] = np.nan
        try:
            mu_selected_index[idx] = int(result.get("mu_selected_index", -1))
        except Exception:
            mu_selected_index[idx] = -1
        if result.get("spectrum"):
            fluxes[idx] = result["spectrum"][0]
            continua[idx] = result["spectrum"][1]
        is_success = str(result["status"]).lower() in _SUCCESS_STATUSES
        if is_success:
            tracker.record_success()
        else:
            tracker.record_failure(result["message"])
        log_method = logger.info if is_success else logger.error
        log_method(
            "[%d/%d] %s %s (%.2fs) - %s",
            idx + 1,
            row_count,
            str(result["status"]).upper(),
            result["base_name"],
            result["duration"],
            result["message"],
        )

    def _trigger_early_exit() -> None:
        logger.error(
            "Early exit after %d consecutive failures — cancelling remaining tasks. %s",
            tracker.consecutive_failures,
            tracker.summary(),
        )
        for f in futures:
            f.cancel()

    with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker, initargs=(config,)) as executor:
        futures = {executor.submit(_synthesis_group_task, group): group for group in groups}
        for future in as_completed(futures):
            group = futures[future]
            try:
                group_results = future.result()
            except Exception as exc:  # noqa: BLE001
                for gidx, _row in group:
                    statuses[gidx] = "exception"
                    messages[gidx] = str(exc)
                    tracker.record_failure(str(exc))
                logger.exception("Group task crashed (%d rows): %s", len(group), exc)
                if tracker.should_exit:
                    early_exit = True
                    _trigger_early_exit()
                    break
                continue

            for result in group_results:
                _record_result(result)
                if tracker.should_exit:
                    break

            if tracker.should_exit:
                early_exit = True
                _trigger_early_exit()
                break

    if early_exit:
        logger.warning(
            "Processing stopped early: %d/%d tasks completed (%d failures). "
            "Partial results will still be written.",
            tracker.total_done,
            row_count,
            tracker.total_failures,
        )

    final_path = os.path.abspath(args.output_zarr)
    write_path = os.path.abspath(args.output_tmp) if args.output_tmp else final_path
    if args.output_tmp and os.path.dirname(write_path) != os.path.dirname(final_path):
        logger.warning(
            "output-tmp and output-zarr should be on same filesystem for atomic rename; "
            "cross-FS rename will copy+delete (not atomic)"
        )

    params, param_names = _build_params_matrix(column_data)
    model_id = _compute_model_ids(params)
    physics_hash = args.physics_hash or _compute_physics_hash(config, column_data)
    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    git_commit = _git_commit(project_root)
    resolved_flux_definition, resolved_flux_unit = infer_flux_metadata(
        output_mode_values,
        explicit_definition=args.flux_definition,
    )
    config_payload = dataclasses.asdict(config)
    config_hash = canonical_json_sha256(config_payload)
    grid_definition_hash = compute_grid_definition_hash(column_data)
    status_counts = _validate_synthesis_results(
        statuses=statuses,
        messages=messages,
        fluxes=fluxes,
        continua=continua,
        max_failures=args.max_failures,
    )

    n_failures = sum(v for k, v in status_counts.items() if k.lower() not in SUCCESS_STATUSES)
    validation_failed = n_failures > args.max_failures

    linelist_files_abs = _resolve_linelist_paths(config)
    linelist_files_manifest: List[Dict[str, Any]] = []
    linelist_digest_tokens: List[str] = []
    for path in linelist_files_abs:
        entry: Dict[str, Any] = {"path": path, "exists": bool(os.path.isfile(path))}
        if entry["exists"]:
            try:
                sha = file_sha256(path)
                entry["sha256"] = sha
                entry["size_bytes"] = int(os.path.getsize(path))
                linelist_digest_tokens.append(f"{path}:{sha}")
            except Exception as exc:  # noqa: BLE001
                entry["sha256_error"] = str(exc)
        linelist_files_manifest.append(entry)
    computed_linelist_sha = ""
    if linelist_digest_tokens:
        canonical_tokens = json.dumps(sorted(linelist_digest_tokens), separators=(",", ":"))
        computed_linelist_sha = hashlib.sha256(canonical_tokens.encode("utf-8")).hexdigest()

    atmosphere_path = os.path.abspath(str(config.model_atmosphere_path))
    atmosphere_model_count = _count_matching_files(atmosphere_path, ".mod")
    configured_atmosphere_sha = str(getattr(config, "atmosphere_sha256", "") or "").strip()
    atmosphere_sha = configured_atmosphere_sha
    if not atmosphere_sha and os.path.isfile(atmosphere_path):
        try:
            atmosphere_sha = file_sha256(atmosphere_path)
        except Exception:
            atmosphere_sha = ""
    if not atmosphere_sha and os.path.isdir(atmosphere_path):
        atmosphere_sha = directory_manifest_sha256(atmosphere_path, suffixes=(".mod",), hash_contents=False)

    linelist_version = str(getattr(config, "linelist_version", "") or "").strip()
    linelist_sha = str(getattr(config, "linelist_sha256", "") or "").strip() or computed_linelist_sha
    linelist_preprocessing = str(getattr(config, "linelist_preprocessing", "") or "").strip()
    atmosphere_geometry = str(getattr(config, "atmosphere_geometry", "") or "").strip()
    atmosphere_version = str(getattr(config, "atmosphere_version", "") or "").strip()
    synthesis_code_version = str(getattr(config, "synthesis_code_version", "") or "").strip()
    spice_version = str(getattr(config, "spice_version", "") or "").strip()
    environment_capture = str(getattr(config, "environment_capture", "pip_freeze") or "pip_freeze")
    binary_manifest_hash = compute_binary_manifest_hash(_resolve_synthesis_binary_paths(config, project_root))

    linelist_identifier = str(config.linelist_path).strip() or ",".join(str(x) for x in (config.linelist_files or []))
    if not linelist_identifier and linelist_files_abs:
        linelist_identifier = ",".join(linelist_files_abs)
    atmosphere_model_identifier = atmosphere_path

    if (not is_meaningful_provenance_value(linelist_version)) and is_meaningful_provenance_value(linelist_sha):
        linelist_version = f"sha256:{linelist_sha[:16]}"
    if (not is_meaningful_provenance_value(atmosphere_version)) and is_meaningful_provenance_value(atmosphere_sha):
        atmosphere_version = f"sha256:{atmosphere_sha[:16]}"

    turbospectrum_version = synthesis_code_version
    if (not is_meaningful_provenance_value(turbospectrum_version)) and is_meaningful_provenance_value(binary_manifest_hash):
        turbospectrum_version = f"binary_sha256:{binary_manifest_hash}"

    pipeline_version = spice_version
    if (not is_meaningful_provenance_value(pipeline_version)) and is_meaningful_provenance_value(git_commit):
        pipeline_version = f"git:{git_commit[:12]}"

    contract_provenance: Dict[str, str] = {
        "config_hash": config_hash,
        "grid_definition_hash": grid_definition_hash,
        "git_commit": git_commit,
        "turbospectrum_version": turbospectrum_version,
        "linelist_identifier": linelist_identifier,
        "linelist_version": linelist_version,
        "atmosphere_model_identifier": atmosphere_model_identifier,
        "synthesis_timestamp": created_utc,
        "pipeline_version": pipeline_version,
    }
    if not args.allow_incomplete_provenance:
        assert_required_provenance_fields(contract_provenance, context="synthesize_spectra_from_zarr.py")

    canonical_config_payload = {
        "contract_fields": contract_provenance,
        "physics_hash": physics_hash,
        "linelist": {
            "path": linelist_identifier,
            "files": [str(x) for x in (config.linelist_files or [])],
            "version": linelist_version,
            "sha256": linelist_sha,
            "preprocessing": linelist_preprocessing,
        },
        "atmospheres": {
            "path": atmosphere_path,
            "geometry": atmosphere_geometry,
            "version": atmosphere_version,
            "sha256": atmosphere_sha,
            "model_file_count": atmosphere_model_count,
        },
        "physics": {
            "nlte": bool(config.nlte),
            "mu_sampling": getattr(config, "mu_sampling", {}) or {},
            "output_mode_values": sorted({str(x) for x in np.asarray(column_data.get("output_mode", [])).tolist()}) if "output_mode" in column_data else [],
            "calculation_mode_values": sorted({str(x) for x in np.asarray(column_data.get("calculation_mode", [])).tolist()}) if "calculation_mode" in column_data else [],
            "compiler": str(config.compiler),
        },
        "synthesis": {
            "code": str(args.generator),
            "version": turbospectrum_version,
            "spice_version": pipeline_version,
            "git_commit": git_commit,
            "binaries_hash": binary_manifest_hash,
        },
        "wavelength": {
            "lam_min": sorted({str(x) for x in np.asarray(column_data["lam_min"]).tolist()}),
            "lam_max": sorted({str(x) for x in np.asarray(column_data["lam_max"]).tolist()}),
            "lam_step": sorted({str(x) for x in np.asarray(column_data["lam_step"]).tolist()}),
        },
    }

    provenance_payload = {
        "canonical_config.yaml": json.dumps(canonical_config_payload, sort_keys=True, indent=2),
        "synthesis_config.yaml": json.dumps(config_payload, sort_keys=True, indent=2, default=str),
        "input_config.json": raw_config_text,
        "linelist_manifest.json": json.dumps(
            {
                "source": linelist_identifier,
                "version": linelist_version,
                "sha256": linelist_sha,
                "preprocessing": linelist_preprocessing,
                "files": linelist_files_manifest,
            },
            sort_keys=True,
            indent=2,
        ),
        "atmosphere_manifest.json": json.dumps(
            {
                "path": atmosphere_path,
                "geometry": atmosphere_geometry,
                "version": atmosphere_version,
                "sha256": atmosphere_sha,
                "model_file_count": atmosphere_model_count,
            },
            sort_keys=True,
            indent=2,
        ),
        "software_manifest.json": json.dumps(
            {
                "generator": str(args.generator),
                "git_commit": git_commit,
                "python": sys.version.split()[0],
                "compiler": str(config.compiler),
                "synthesis_code_version": turbospectrum_version,
                "spice_version": pipeline_version,
                "binaries_hash": binary_manifest_hash,
            },
            sort_keys=True,
            indent=2,
        ),
        "environment.txt": _capture_environment_text(environment_capture),
    }

    _write_zarr_output(
        output_path=write_path,
        wavelengths=wavelengths,
        fluxes=fluxes,
        continua=continua,
        mu_selected=mu_selected,
        mu_selected_index=mu_selected_index,
        params=params,
        param_names=param_names,
        model_id=model_id,
        physics_hash=physics_hash,
        schema_version=args.schema_version,
        created_utc=created_utc,
        git_commit=git_commit,
        contact=args.contact,
        generator=args.generator,
        flux_definition=resolved_flux_definition,
        flux_unit=resolved_flux_unit,
        provenance_payload=provenance_payload,
        contract_provenance=contract_provenance,
        status_counts=status_counts,
        compression_cfg=compressor_cfg,
        chunk_rows=args.chunk_rows,
        logger=logger,
    )
    if write_path != final_path:
        _finalize_zarr_store(write_path, final_path, logger=logger, label="Synthesis Zarr")
    logger.info("Completed synthesis in %.2fs", time.perf_counter() - t0)

    if validation_failed:
        logger.error(
            "Output saved to %s but exiting with error: %d failure(s) exceed --max-failures %d",
            final_path,
            n_failures,
            args.max_failures,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
