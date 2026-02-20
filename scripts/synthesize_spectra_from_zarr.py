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
import copy
import re
import warnings
import hashlib
import subprocess
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


def _choose_mu_indices(mu_points: np.ndarray, *, row_index: int, cfg: TurbospectrumConfig) -> tuple[np.ndarray, float]:
    ms: Mapping[str, Any] = getattr(cfg, "mu_sampling", {}) or {}
    mode = str(ms.get("mode", "none")).lower()
    if mode != "random" or mu_points.size == 0:
        return np.asarray([], dtype=np.int64), float("nan")

    count = int(ms.get("count", 1) or 1)
    mu_min = float(ms.get("min", 0.0))
    mu_max = float(ms.get("max", 1.0))
    seed = ms.get("seed")
    base_seed = 0 if seed in (None, "") else int(seed)

    candidates = np.where((mu_points >= mu_min) & (mu_points <= mu_max))[0]
    if candidates.size == 0:
        candidates = np.arange(mu_points.size, dtype=np.int64)

    rng = np.random.default_rng((base_seed + int(row_index)) % (2**32))
    replace = bool(count > candidates.size)
    chosen = rng.choice(candidates, size=count, replace=replace)
    chosen = np.asarray(chosen, dtype=np.int64)
    mu_sel = mu_points[chosen]
    mu_summary = float(mu_sel[0]) if mu_sel.size == 1 else float(np.mean(mu_sel))
    return chosen, mu_summary


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "configs", "synthesis", "config_sample_comprehensive.json")
)
DEFAULT_OUTPUT_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "runs", "local-dev", "outputs", "zarr", "synthesized_spectra.zarr")
)

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


def _build_params_matrix(column_data: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Fixed ordering keeps schema stable across runs.
    candidate_order = ["teff", "logg", "feh", "vmicro", "a", "c", "n", "o", "r", "s"]
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

    for name in ("a", "c", "n", "o", "r", "s"):
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
        return "unknown"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_linelist_paths(config: TurbospectrumConfig) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in (config.linelist_files or []):
        raw = str(item).strip()
        if not raw:
            continue
        path = raw if os.path.isabs(raw) else os.path.abspath(os.path.join(str(config.linelist_path), raw))
        if path in seen:
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
    import zarr.codecs as zc  # type: ignore
    from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

    try:
        arr = root.create_array(
            name,
            shape=(),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            **compression_kwargs,
        )
        arr[...] = str(value)
    except Exception:
        arr = root.create_array(
            name,
            shape=(1,),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            chunks=1,
            **compression_kwargs,
        )
        arr[0] = str(value)


def _write_fixed_string_scalar(root, name: str, value: str, min_width: int, compression_kwargs: Mapping[str, Any]) -> None:
    sval = str(value)
    width = max(int(min_width), len(sval), 1)
    try:
        arr = root.create_array(
            name,
            shape=(),
            dtype=f"<U{width}",
            **compression_kwargs,
        )
        arr[...] = sval
    except Exception:
        _write_string_scalar(root, name, sval, compression_kwargs=compression_kwargs)


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

    optional_columns = ["output_mode", "calculation_mode", "grid_version", "a", "c", "n", "o", "r", "s"]
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
    index, row_values = args
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker config not initialized")
    cfg = copy.deepcopy(_WORKER_CONFIG)
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
        output_mode = "Intensity" if cfg.calculate_intensity else "Flux"
    calculation_mode = row_values.get("calculation_mode")
    if calculation_mode is None:
        calculation_mode = "NLTE" if cfg.nlte else "LTE"
    output_mode = str(output_mode)
    calculation_mode = str(calculation_mode)

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

            if cfg.calculate_intensity:
                mu_points = _read_mu_points(spec_path)
                chosen_idx, mu_selected = _choose_mu_indices(mu_points, row_index=int(index), cfg=cfg)
                reduce_mode = str(getattr(cfg, "mu_sampling", {}).get("reduce", "first")).lower()
                if chosen_idx.size == 0 and str(getattr(cfg, "mu_sampling", {}).get("mode", "none")).lower() == "random":
                    n_mu = max(0, int((data.shape[1] - 3) // 2))
                    if n_mu > 0:
                        seed = getattr(cfg, "mu_sampling", {}).get("seed")
                        base_seed = 0 if seed in (None, "") else int(seed)
                        rng = np.random.default_rng((base_seed + int(index)) % (2**32))
                        count = int(getattr(cfg, "mu_sampling", {}).get("count", 1) or 1)
                        replace = bool(count > n_mu)
                        chosen_idx = np.asarray(rng.choice(np.arange(n_mu), size=count, replace=replace), dtype=np.int64)
                        mu_selected = float("nan")

                if chosen_idx.size > 0:
                    mu_selected_index = int(chosen_idx[0])
                    abs_cols = [int(3 + 2 * i) for i in chosen_idx.tolist()]
                    norm_cols = [int(4 + 2 * i) for i in chosen_idx.tolist()]
                    if data.shape[1] <= max(abs_cols + norm_cols):
                        raise ValueError(f"Intensity spectrum has too few columns: {data.shape}")
                    i_abs = data[:, abs_cols].astype(np.float32)
                    i_norm = data[:, norm_cols].astype(np.float32)
                    if i_abs.ndim == 1:
                        i_abs = i_abs[:, None]
                        i_norm = i_norm[:, None]
                    if reduce_mode == "mean" and i_abs.shape[1] > 1:
                        flux = i_abs.mean(axis=1)
                        cont = i_norm.mean(axis=1)
                    else:
                        flux = i_abs[:, 0]
                        cont = i_norm[:, 0]
                else:
                    flux = data[:, 1].astype(np.float32)
                    cont = data[:, 2].astype(np.float32) if data.shape[1] > 2 else np.full_like(flux, np.nan, dtype=np.float32)
            else:
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


def _write_zarr_output(
    output_path: str,
    wavelengths: np.ndarray,
    fluxes: np.ndarray,
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
    provenance_payload: Mapping[str, str],
    status_counts: Mapping[str, int],
    compression_cfg: Mapping,
    chunk_rows: int,
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    store = _zarr_store(output_path)
    root = zarr.group(store=store, overwrite=True, zarr_format=3)

    compression_kwargs = _zarr_compression_kwargs(compression_cfg)
    chunk_shape = (min(chunk_rows, fluxes.shape[0]), fluxes.shape[1]) if fluxes.shape[0] else (1, fluxes.shape[1])
    param_chunk_shape = (min(chunk_rows, params.shape[0]), params.shape[1]) if params.shape[0] else (1, params.shape[1])
    wl = wavelengths.astype(np.float32, copy=False)
    param_name_list = [str(x) for x in param_names.tolist()]
    param_names_u32 = _to_u32_param_names(param_name_list)

    # DATA_SCHEMA.md synthesis layout
    root.create_array("wavelength", data=wl, chunks=wl.shape if wl.size else (1,), **compression_kwargs)
    root.create_array("flux", data=fluxes, chunks=chunk_shape, **compression_kwargs)
    root.create_array("params", data=params.astype(np.float32, copy=False), chunks=param_chunk_shape, **compression_kwargs)
    root.create_array(
        "param_names",
        data=param_names_u32,
        chunks=(min(max(1, params.shape[1]), len(param_names_u32)) if len(param_names_u32) else 1,),
        **compression_kwargs,
    )
    root.create_array(
        "model_id",
        data=model_id.astype(np.uint64, copy=False),
        chunks=(min(chunk_rows, len(model_id)) if len(model_id) else 1,),
        **compression_kwargs,
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
        elif name in {"logg", "feh", "a", "c", "n", "o", "r", "s"}:
            param_units[name] = "dex"
        elif name == "vmicro":
            param_units[name] = "km/s"
        else:
            param_units[name] = ""

    root.attrs.update(
        {
            "title": "SPICE Synthetic Spectral Grid",
            "generator": generator,
            "created_utc": created_utc,
            "flux_definition": flux_definition,
            "wavelength_unit": "angstrom",
            "flux_unit": "relative",
            "parameter_units": param_units,
            "physics_hash": physics_hash,
            "git_commit": git_commit,
            "contact": contact,
            "schema_version": schema_version,
            "n_models": int(fluxes.shape[0]),
            "n_lambda": int(fluxes.shape[1]) if fluxes.ndim == 2 else 0,
            "n_params": int(params.shape[1]) if params.ndim == 2 else 0,
            # Keep diagnostics in attrs (not top-level arrays) to preserve schema shape.
            "status_counts": dict(status_counts),
        }
    )
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
        tasks.append((idx, row_values))
    return tasks


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
    parser.add_argument("--workers", type=int, default=None, help="Override worker process count")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--chunk-rows", type=int, default=32, help="Zarr chunking along the sample dimension")
    parser.add_argument("--compressor", default=None, help="JSON string describing compressor options (cname, clevel, shuffle)")
    parser.add_argument("--schema-version", default="1.0.0", help="DATA_SCHEMA.md schema version")
    parser.add_argument("--physics-hash", default=None, help="Optional override for physics hash")
    parser.add_argument("--contact", default=os.environ.get("SPICE_CONTACT", "unknown"), help="Contact metadata")
    parser.add_argument("--generator", default="turbospectrum_nlte", help="Generator string for metadata")
    parser.add_argument("--flux-definition", default="continuum_normalized", help="Flux definition metadata")
    args = parser.parse_args()

    logger = _configure_logging(args.log_level, args.log_file)
    t0 = time.perf_counter()
    logger.info("Starting synthesis run: grid=%s config=%s", os.path.abspath(args.grid_zarr), os.path.abspath(args.config))
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    config = _load_config(args.config, project_root=project_root)
    logger.info("mu_sampling=%s", json.dumps(getattr(config, "mu_sampling", {}) or {}, sort_keys=True))
    if args.scratch:
        scratch = os.path.abspath(args.scratch)
        os.makedirs(scratch, exist_ok=True)
        # Redirect I/O-heavy directories to scratch to reduce shared FS contention.
        config.tmp_dir = os.path.join(scratch, "tmp")
        config.log_dir = os.path.join(scratch, "logs")
        config.output_dir = os.path.join(scratch, "spectra")
        config.model_opac_dir = os.path.join(scratch, "opac")
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
    statuses: List[str] = ["pending"] * row_count
    messages: List[str] = [""] * row_count

    tasks = _build_tasks(row_count, column_data, config)
    worker_count = int(args.workers) if args.workers and args.workers > 0 else determine_worker_count(config)

    compressor_cfg: Dict[str, object] = {}
    if args.compressor:
        try:
            compressor_cfg = json.loads(args.compressor)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid compressor JSON: {exc}") from exc

    with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker, initargs=(config,)) as executor:
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
                fluxes[idx] = result["spectrum"][0]
            logger.info(
                "[%d/%d] %s %s (%.2fs) - %s",
                idx + 1,
                row_count,
                result["status"].upper(),
                result["base_name"],
                result["duration"],
                result["message"],
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
    status_counts: Dict[str, int] = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1

    linelist_files_abs = _resolve_linelist_paths(config)
    linelist_files_manifest: List[Dict[str, Any]] = []
    linelist_digest_tokens: List[str] = []
    for path in linelist_files_abs:
        entry: Dict[str, Any] = {"path": path, "exists": bool(os.path.isfile(path))}
        if entry["exists"]:
            try:
                sha = _sha256_file(path)
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
            atmosphere_sha = _sha256_file(atmosphere_path)
        except Exception:
            atmosphere_sha = ""

    linelist_version = str(getattr(config, "linelist_version", "") or "").strip()
    linelist_sha = str(getattr(config, "linelist_sha256", "") or "").strip() or computed_linelist_sha or "not_recorded"
    linelist_preprocessing = str(getattr(config, "linelist_preprocessing", "") or "").strip()
    atmosphere_geometry = str(getattr(config, "atmosphere_geometry", "") or "").strip()
    atmosphere_version = str(getattr(config, "atmosphere_version", "") or "").strip()
    synthesis_code_version = str(getattr(config, "synthesis_code_version", "") or "").strip()
    spice_version = str(getattr(config, "spice_version", "") or "").strip()
    environment_capture = str(getattr(config, "environment_capture", "pip_freeze") or "pip_freeze")

    canonical_config_payload = {
        "physics_hash": physics_hash,
        "linelist": {
            "path": str(config.linelist_path),
            "files": [str(x) for x in (config.linelist_files or [])],
            "version": linelist_version or "not_recorded",
            "sha256": linelist_sha,
            "preprocessing": linelist_preprocessing or "not_recorded",
        },
        "atmospheres": {
            "path": atmosphere_path,
            "geometry": atmosphere_geometry or "not_recorded",
            "version": atmosphere_version or "not_recorded",
            "sha256": atmosphere_sha or "not_recorded",
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
            "version": synthesis_code_version or "not_recorded",
            "spice_version": spice_version or "not_recorded",
            "git_commit": git_commit,
        },
        "wavelength": {
            "lam_min": sorted({str(x) for x in np.asarray(column_data["lam_min"]).tolist()}),
            "lam_max": sorted({str(x) for x in np.asarray(column_data["lam_max"]).tolist()}),
            "lam_step": sorted({str(x) for x in np.asarray(column_data["lam_step"]).tolist()}),
        },
    }

    provenance_payload = {
        "canonical_config.yaml": json.dumps(canonical_config_payload, sort_keys=True, indent=2),
        "synthesis_config.yaml": json.dumps(dataclasses.asdict(config), sort_keys=True, indent=2, default=str),
        "linelist_manifest.json": json.dumps(
            {
                "source": str(config.linelist_path),
                "version": linelist_version or "not_recorded",
                "sha256": linelist_sha,
                "preprocessing": linelist_preprocessing or "not_recorded",
                "files": linelist_files_manifest,
            },
            sort_keys=True,
            indent=2,
        ),
        "atmosphere_manifest.json": json.dumps(
            {
                "path": atmosphere_path,
                "geometry": atmosphere_geometry or "not_recorded",
                "version": atmosphere_version or "not_recorded",
                "sha256": atmosphere_sha or "not_recorded",
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
                "synthesis_code_version": synthesis_code_version or "not_recorded",
                "spice_version": spice_version or "not_recorded",
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
        params=params,
        param_names=param_names,
        model_id=model_id,
        physics_hash=physics_hash,
        schema_version=args.schema_version,
        created_utc=created_utc,
        git_commit=git_commit,
        contact=args.contact,
        generator=args.generator,
        flux_definition=args.flux_definition,
        provenance_payload=provenance_payload,
        status_counts=status_counts,
        compression_cfg=compressor_cfg,
        chunk_rows=args.chunk_rows,
        logger=logger,
    )
    if write_path != final_path:
        os.rename(write_path, final_path)
        logger.info("Atomic rename: %s -> %s", write_path, final_path)
    logger.info("Completed synthesis in %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
