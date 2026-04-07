#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import dataclasses
import errno
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    resolve_linelist_paths,
    run_single_synthesis,
    validate_runtime_environment,
)
from provenance_contract import (  # noqa: E402
    assert_required_provenance_fields,
    canonical_json_sha256,
    compute_binary_manifest_hash,
    compute_grid_definition_hash,
    directory_manifest_sha256,
    file_sha256,
    is_meaningful_provenance_value,
)
from spectrum_output import extract_flux_and_continuum, infer_flux_metadata
from synthesis_validation import SUCCESS_STATUSES, validate_synthesis_results


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
                    # stop at first data line
                    break
                if "mu-points" in line.lower():
                    header = line
        if not header:
            return np.asarray([], dtype=np.float32)
        # Be permissive: header can be e.g. "mu-points:" or have odd spacing.
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
    ms = getattr(cfg, "mu_sampling", {}) or {}
    mu_min = float(ms.get("min", 0.0))
    mu_max = float(ms.get("max", 1.0))
    candidates = np.where((mu_points >= mu_min) & (mu_points <= mu_max))[0]
    if candidates.size == 0:
        candidates = np.arange(mu_points.size, dtype=np.int64)
    return np.asarray(candidates, dtype=np.int64)


def _choose_mu_indices(
    mu_points: np.ndarray,
    *,
    global_index: int,
    cfg: TurbospectrumConfig,
    target_mu: float | None = None,
) -> tuple[np.ndarray, float]:
    """Choose mu indices for one spectrum row, deterministically if seeded.

    Returns (indices, mu_summary) where mu_summary is the selected mu (count==1)
    or mean(mu_selected) otherwise.
    """
    ms = getattr(cfg, "mu_sampling", {}) or {}
    mode = str(ms.get("mode", "none")).lower()
    if mu_points.size == 0:
        return np.asarray([], dtype=np.int64), float("nan")

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
        # Deterministic per-row RNG: independent of multiprocessing ordering.
        rng = np.random.default_rng((base_seed + int(global_index)) % (2**32))
        replace = bool(count > candidates.size)
        chosen = rng.choice(candidates, size=count, replace=replace)
    else:
        return np.asarray([], dtype=np.int64), float("nan")

    chosen = np.asarray(chosen, dtype=np.int64)
    mu_sel = mu_points[chosen]
    mu_summary = float(mu_sel[0]) if mu_sel.size == 1 else float(np.mean(mu_sel))
    return chosen, mu_summary


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


def _write_string_scalar(root, name: str, value: str) -> None:
    if hasattr(root, "create_array"):
        import zarr.codecs as zc  # type: ignore
        from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

        try:
            arr = root.create_array(
                name,
                shape=(),
                dtype=VariableLengthUTF8(),
                serializer=zc.VLenUTF8Codec(),
            )
            arr[...] = str(value)
            return
        except Exception:
            arr = root.create_array(
                name,
                shape=(1,),
                dtype=VariableLengthUTF8(),
                serializer=zc.VLenUTF8Codec(),
                chunks=1,
            )
            arr[0] = str(value)
            return
    _write_string_1d(root, name, [str(value)], chunks=1)


def _existing_shard_is_usable(
    path: str,
    expected_global_indices: np.ndarray,
    *,
    expected_wavelengths: np.ndarray | None = None,
) -> bool:
    try:
        root = _open_shard(path)
    except Exception:
        return False

    required = ("wavelength", "global_index", "flux", "status", "message")
    if any(name not in root for name in required):
        return False

    try:
        gidx = np.asarray(root["global_index"][:], dtype=np.int64)
        flux = np.asarray(root["flux"][:], dtype=np.float32)
    except Exception:
        return False

    if gidx.ndim != 1 or flux.ndim != 2 or flux.shape[0] != gidx.size:
        return False

    if not np.array_equal(gidx, expected_global_indices.astype(np.int64, copy=False)):
        return False

    # Validate that the stored wavelength grid matches the current request.
    # Without this check, shards from a previous run with a different wavelength
    # range at the same file path would be silently reused.
    if expected_wavelengths is not None:
        try:
            wl = np.asarray(root["wavelength"][:], dtype=np.float64)
        except Exception:
            return False
        if wl.size != expected_wavelengths.size:
            return False
        if not np.allclose(wl, expected_wavelengths, atol=1e-4):
            return False

    try:
        statuses = [str(x).strip().lower() for x in np.asarray(root["status"][:]).tolist()]
    except Exception:
        return False

    if len(statuses) != gidx.size:
        return False
    if any(status not in SUCCESS_STATUSES for status in statuses):
        return False

    if not np.all(np.isfinite(flux)):
        return False

    if "continuum" in root:
        try:
            continuum = np.asarray(root["continuum"][:], dtype=np.float32)
        except Exception:
            return False
        if continuum.shape != flux.shape:
            return False
        if not np.all(np.any(np.isfinite(continuum), axis=1)):
            return False

    return True


def _finalize_zarr_store(write_path: str, final_path: str, *, logger: logging.Logger, label: str) -> None:
    """Move a completed Zarr directory into place.

    On the same filesystem we keep the atomic rename. Across filesystems, fall
    back to copy+delete so scratch-backed temp outputs still complete.
    """
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


def _git_commit(project_root: str) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL, timeout=3)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _resolve_linelist_paths(config: TurbospectrumConfig) -> List[str]:
    return resolve_linelist_paths(str(config.linelist_path), config.linelist_files)


def _resolve_synthesis_binary_paths(config: TurbospectrumConfig, project_root: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in (
        getattr(config, "babsma_path", ""),
        getattr(config, "bsyn_path", ""),
        getattr(config, "interpol_path", ""),
    ):
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(item or "").strip())))
        if not path:
            continue
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        out.append(path)
    return out


def _capture_environment_text() -> str:
    return "\n".join(
        [
            f"python={sys.version}",
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}",
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '')}",
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '')}",
            f"VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS', '')}",
        ]
    )


def _grid_columns_for_hash(grid_root) -> Dict[str, np.ndarray]:
    keys = (
        "teff",
        "logg",
        "feh",
        "turbvel",
        "t_value",
        "lam_min",
        "lam_max",
        "lam_step",
        "a",
        "c",
        "n",
        "o",
        "r",
        "s",
        "output_mode",
        "calculation_mode",
        "grid_version",
    )
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        if key in grid_root:
            out[key] = np.asarray(grid_root[key][:])
    return out


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

        cfg.lambda_min = float(row_values["lam_min"])
        cfg.lambda_max = float(row_values["lam_max"])
        cfg.lambda_step = float(row_values["lam_step"])

        # If the grid provides per-row mode flags, honor them; otherwise, fall back
        # to whatever the Turbospectrum config requested.
        output_mode = row_values.get("output_mode")
        if output_mode is None:
            output_mode = getattr(cfg, "output_mode", "Flux")
        calculation_mode = row_values.get("calculation_mode")
        if calculation_mode is None:
            calculation_mode = "NLTE" if cfg.nlte else "LTE"

        cfg.output_mode = str(output_mode)
        cfg.nlte = str(calculation_mode).lower() == "nlte"
        is_intensity = str(output_mode).lower() == "intensity"
        mu_sampling = getattr(cfg, "mu_sampling", {}) or {}
        if not isinstance(mu_sampling, dict):
            mu_sampling = {}
        if is_intensity and str(mu_sampling.get("mode", "none")).strip().lower() in {"", "none"}:
            mu_sampling["mode"] = "random"
        cfg.mu_sampling = mu_sampling
        target_mu_raw = row_values.get("mu")
        target_mu = None if target_mu_raw in (None, "") else float(target_mu_raw)

        start = time.perf_counter()
        result = run_single_synthesis((row_values, cfg))
        duration = time.perf_counter() - start
        log_path = str(result.get("log_path", "") or "")

        spec_path = str(result.get("output_path", "") or "")
        base_name = str(result.get("base_name", "") or os.path.splitext(os.path.basename(spec_path))[0])
        if not spec_path:
            suffix = ".intensity.spec" if is_intensity else ".spec"
            spec_path = os.path.join(cfg.output_dir, f"{base_name}{suffix}")

        spectrum = None
        mu_selected = float("nan")
        mu_selected_index = -1
        if os.path.exists(spec_path):
            try:
                # Fail fast on empty files (Turbospectrum can produce a 0-byte output on failure).
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

                # Validate wavelength point count matches what the grid requested.
                expected_n = int(round((cfg.lambda_max - cfg.lambda_min) / cfg.lambda_step)) + 1
                if data.shape[0] != expected_n:
                    raise ValueError(f"Unexpected wavelength count {data.shape[0]} (expected {expected_n})")

                if is_intensity:
                    # Pick the requested mu from the Intensity file, or fall back
                    # to the historical per-row random selection mode.
                    mu_points = _read_mu_points(spec_path)
                    mu_mode = str(getattr(cfg, "mu_sampling", {}).get("mode", "none")).lower()
                    chosen_idx, mu_selected = _choose_mu_indices(
                        mu_points,
                        global_index=int(global_index),
                        cfg=cfg,
                        target_mu=target_mu,
                    )
                    reduce_mode = str(getattr(cfg, "mu_sampling", {}).get("reduce", "first")).lower()
                    if chosen_idx.size == 0 and mu_mode == "random":
                        # Fallback: if header parsing failed, infer mu column count from file columns.
                        # Column layout: wl, flux_norm, flux_abs, (Iabs, Inorm)*n_mu
                        n_mu = max(0, int((data.shape[1] - 3) // 2))
                        if n_mu > 0:
                            seed = getattr(cfg, "mu_sampling", {}).get("seed")
                            base_seed = 0 if seed in (None, "") else int(seed)
                            rng = np.random.default_rng((base_seed + int(global_index)) % (2**32))
                            count = int(getattr(cfg, "mu_sampling", {}).get("count", 1) or 1)
                            replace = bool(count > n_mu)
                            chosen_idx = np.asarray(rng.choice(np.arange(n_mu), size=count, replace=replace), dtype=np.int64)
                            # mu values unknown without header.
                            mu_selected = float("nan")
                    elif chosen_idx.size == 0 and target_mu is not None and mu_mode in {"nearest", "target"}:
                        n_mu = max(0, int((data.shape[1] - 3) // 2))
                        if n_mu > 0:
                            mu_min = float(getattr(cfg, "mu_sampling", {}).get("min", 0.0))
                            mu_max = float(getattr(cfg, "mu_sampling", {}).get("max", 1.0))
                            denom = mu_max - mu_min
                            frac = 0.0 if denom <= 0 else float(np.clip((target_mu - mu_min) / denom, 0.0, 1.0))
                            ranked = np.argsort(np.abs(np.arange(n_mu, dtype=np.float64) - frac * max(0, n_mu - 1)))
                            count = int(getattr(cfg, "mu_sampling", {}).get("count", 1) or 1)
                            chosen_idx = ranked[:count] if count <= ranked.size else np.resize(ranked, count)
                            chosen_idx = np.asarray(chosen_idx, dtype=np.int64)
                            logging.getLogger("zarr_synthesis_sharded").warning(
                                "mu-points header missing from %s; using target_mu=%.6g as mu_selected estimate",
                                spec_path,
                                target_mu,
                            )
                            mu_selected = float(target_mu)

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

            except Exception as exc:
                results.append({
                    "global_index": global_index,
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
                    "message": _with_turbospectrum_log_context(
                        f"Missing spectrum output: {spec_path}",
                        log_path,
                    ),
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
            "mu_selected": float(mu_selected),
            "mu_selected_index": int(mu_selected_index),
        })

    return results


def _build_task_batches(
    indices: np.ndarray,
    columns: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    config: TurbospectrumConfig,
) -> tuple[list[list[tuple[int, dict[str, Any]]]], dict[int, int]]:
    tasks: list[list[tuple[int, dict[str, Any]]]] = []
    global_to_local = {int(g): i for i, g in enumerate(indices.tolist())}
    output_name_counts: Dict[str, int] = {}

    for batch_indices in _build_batches(indices, batch_size):
        batch: list[tuple[int, dict[str, Any]]] = []
        for global_i in batch_indices:
            local_i = global_to_local[int(global_i)]
            row_values = {k: columns[k][local_i] for k in columns}
            output_stem = get_synthesis_output_stem_from_params(
                row_values,
                default_output_mode=getattr(config, "output_mode", "Flux"),
                default_calculation_mode="NLTE" if config.nlte else "LTE",
            )
            output_name_counts[output_stem] = output_name_counts.get(output_stem, 0) + 1
            batch.append((int(global_i), row_values))
        tasks.append(batch)

    duplicate_output_names = [name for name, count in output_name_counts.items() if count > 1]
    if duplicate_output_names:
        sample = ", ".join(sorted(duplicate_output_names)[:3])
        raise ValueError(
            "Grid rows collapse to duplicate Turbospectrum output filenames inside this shard, "
            f"which can overwrite spectra/opacity files (examples: {sample}). "
            "Preserve all distinguishing columns or adjust naming."
        )

    return tasks, global_to_local


############################################
# Main
############################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--grid-zarr", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-zarr", required=True, help="Final output Zarr path (after atomic rename if --output-tmp used)")
    parser.add_argument(
        "--output-tmp",
        default=None,
        metavar="TMP_PATH",
        help="Temp path for atomic write: write to TMP_PATH, then rename to --output-zarr. Prevents partial shards.",
    )

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
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Maximum number of failed rows to tolerate before aborting the shard write (default: 0 = abort on any failure).",
    )
    parser.add_argument(
        "--allow-incomplete-provenance",
        action="store_true",
        help="Allow writing shards even when required provenance contract fields are missing.",
    )

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
    # Output paths (atomic write via temp + rename)
    ############################################

    final_path = os.path.abspath(args.output_zarr)
    if args.output_tmp:
        write_path = os.path.abspath(args.output_tmp)
        if os.path.dirname(write_path) != os.path.dirname(final_path):
            logger.warning(
                "output-tmp and output-zarr should be on same filesystem for atomic rename; "
                "cross-FS rename will copy+delete (not atomic)"
            )
    else:
        write_path = final_path

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
            "The provided --config looks like an ML sampling config (e.g. configs/sampling/config_ml_sampling.json), "
            "not a Turbospectrum synthesis config. Use configs/synthesis/config_sample_comprehensive.json (or another "
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
    logger.info(
        "mu_sampling=%s",
        json.dumps(getattr(config, "mu_sampling", {}) or {}, sort_keys=True),
    )

    if args.scratch:
        scratch = os.path.abspath(args.scratch)

        config.tmp_dir = os.path.join(scratch, "tmp")
        config.log_dir = os.path.join(scratch, "logs")
        config.output_dir = os.path.join(scratch, "spectra")
        config.model_opac_dir = os.path.join(scratch, "opac")

    ensure_directories(config)
    validate_runtime_environment(config)
    try:
        config.linelist_file_path = create_linelist_file(config)
    except LinelistValidationError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    ############################################
    # Load grid
    ############################################

    grid = zarr.open_group(args.grid_zarr, mode="r")

    row_count = grid["teff"].shape[0]
    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    git_commit = _git_commit(project_root)
    config_payload = dataclasses.asdict(config)
    config_hash = canonical_json_sha256(config_payload)
    grid_definition_hash = compute_grid_definition_hash(_grid_columns_for_hash(grid))

    linelist_files_abs = _resolve_linelist_paths(config)
    linelist_manifest_files: List[Dict[str, Any]] = []
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
        linelist_manifest_files.append(entry)
    computed_linelist_sha = ""
    if linelist_digest_tokens:
        computed_linelist_sha = canonical_json_sha256(sorted(linelist_digest_tokens))

    atmosphere_path = os.path.abspath(str(config.model_atmosphere_path))
    atmosphere_sha = str(getattr(config, "atmosphere_sha256", "") or "").strip()
    if not atmosphere_sha and os.path.isfile(atmosphere_path):
        try:
            atmosphere_sha = file_sha256(atmosphere_path)
        except Exception:
            atmosphere_sha = ""
    if not atmosphere_sha and os.path.isdir(atmosphere_path):
        atmosphere_sha = directory_manifest_sha256(atmosphere_path, suffixes=(".mod",), hash_contents=False)

    linelist_identifier = str(config.linelist_path).strip() or ",".join(str(x) for x in (config.linelist_files or []))
    if not linelist_identifier and linelist_files_abs:
        linelist_identifier = ",".join(linelist_files_abs)
    linelist_sha = str(getattr(config, "linelist_sha256", "") or "").strip() or computed_linelist_sha
    linelist_version = str(getattr(config, "linelist_version", "") or "").strip()
    if (not is_meaningful_provenance_value(linelist_version)) and is_meaningful_provenance_value(linelist_sha):
        linelist_version = f"sha256:{linelist_sha[:16]}"

    atmosphere_geometry = str(getattr(config, "atmosphere_geometry", "") or "").strip()
    atmosphere_version = str(getattr(config, "atmosphere_version", "") or "").strip()
    if (not is_meaningful_provenance_value(atmosphere_version)) and is_meaningful_provenance_value(atmosphere_sha):
        atmosphere_version = f"sha256:{atmosphere_sha[:16]}"
    atmosphere_model_identifier = atmosphere_path

    synthesis_code_version = str(getattr(config, "synthesis_code_version", "") or "").strip()
    binary_manifest_hash = compute_binary_manifest_hash(_resolve_synthesis_binary_paths(config, project_root))
    turbospectrum_version = synthesis_code_version
    if (not is_meaningful_provenance_value(turbospectrum_version)) and is_meaningful_provenance_value(binary_manifest_hash):
        turbospectrum_version = f"binary_sha256:{binary_manifest_hash}"

    pipeline_version = str(getattr(config, "spice_version", "") or "").strip()
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
        assert_required_provenance_fields(contract_provenance, context="run_turbospectrum_shard.py")

    provenance_payload = {
        "canonical_config.yaml": json.dumps(
            {
                "contract_fields": contract_provenance,
                "physics": {
                    "nlte": bool(getattr(config, "nlte", False)),
                    "mu_sampling": getattr(config, "mu_sampling", {}) or {},
                },
                "wavelength": {
                    "row_count": int(row_count),
                },
            },
            sort_keys=True,
            indent=2,
        ),
        "synthesis_config.yaml": json.dumps(config_payload, sort_keys=True, indent=2, default=str),
        "linelist_manifest.json": json.dumps(
            {
                "source": linelist_identifier,
                "version": linelist_version,
                "sha256": linelist_sha,
                "files": linelist_manifest_files,
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
            },
            sort_keys=True,
            indent=2,
        ),
        "software_manifest.json": json.dumps(
            {
                "git_commit": git_commit,
                "compiler": str(config.compiler),
                "synthesis_code_version": turbospectrum_version,
                "pipeline_version": pipeline_version,
                "binaries_hash": binary_manifest_hash,
            },
            sort_keys=True,
            indent=2,
        ),
        "environment.txt": _capture_environment_text(),
    }

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
    # Idempotency
    ############################################

    grid_output_mode_values: List[str] = []
    if "output_mode" in grid:
        grid_output_mode_values = sorted(
            {
                str(value).strip()
                for value in np.asarray(grid["output_mode"][:]).tolist()
                if str(value).strip()
            }
        )
    elif str(getattr(config, "output_mode", "")).strip():
        grid_output_mode_values = [str(getattr(config, "output_mode")).strip()]
    resolved_flux_definition, resolved_flux_unit = infer_flux_metadata(grid_output_mode_values)

    # Compute the expected wavelength grid early so it can be used for
    # idempotency validation (detecting stale shards from a different run).
    _expected_wl: np.ndarray | None = None
    if all(k in grid for k in ("lam_min", "lam_max", "lam_step")):
        _lam_min0 = float(np.asarray(grid["lam_min"][0]))
        _lam_max0 = float(np.asarray(grid["lam_max"][0]))
        _lam_step0 = float(np.asarray(grid["lam_step"][0]))
        if _lam_step0 > 0:
            _npts = int(round((_lam_max0 - _lam_min0) / _lam_step0)) + 1
            _expected_wl = np.linspace(_lam_min0, _lam_max0, _npts)

    if os.path.exists(final_path):
        if _existing_shard_is_usable(final_path, indices, expected_wavelengths=_expected_wl):
            logger.warning("Shard output already exists and is valid — skipping.")
            return
        logger.warning("Shard output exists but is incomplete/invalid; removing and recomputing: %s", final_path)
        shutil.rmtree(final_path, ignore_errors=True)
    if write_path != final_path and os.path.exists(write_path):
        logger.warning("Removing stale shard tmp output before recompute: %s", write_path)
        shutil.rmtree(write_path, ignore_errors=True)

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

        root = _open_root_for_write(write_path)
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
        _write_array(root, "mu_selected", np.asarray([], dtype=np.float32), chunks=1)
        _write_array(root, "mu_selected_index", np.asarray([], dtype=np.int16), chunks=1)
        _write_string_1d(root, "status", [], chunks=1)
        _write_string_1d(root, "message", [], chunks=1)

        attrs_payload: Dict[str, Any] = {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "grid": os.path.abspath(args.grid_zarr),
            "note": "empty shard (no rows assigned)",
            "flux_definition": resolved_flux_definition,
            "flux_unit": resolved_flux_unit,
            "output_mode_values": grid_output_mode_values,
            "created_utc": created_utc,
            "git_sha": git_commit,
            "synthesis_timestamp": created_utc,
            "linelist_path": str(getattr(config, "linelist_path", "")),
            "linelist_files": json.dumps([str(x) for x in (getattr(config, "linelist_files", []) or [])], sort_keys=True),
            "linelist_version": linelist_version,
            "linelist_sha256": linelist_sha,
            "linelist_preprocessing": str(getattr(config, "linelist_preprocessing", "")),
            "model_atmosphere_path": str(getattr(config, "model_atmosphere_path", "")),
            "atmosphere_geometry": atmosphere_geometry,
            "atmosphere_version": atmosphere_version,
            "atmosphere_sha256": atmosphere_sha,
            "synthesis_code_version": turbospectrum_version,
            "spice_version": pipeline_version,
            "compiler": str(getattr(config, "compiler", "")),
            "nlte": bool(getattr(config, "nlte", False)),
        }
        attrs_payload.update({str(k): str(v) for k, v in contract_provenance.items()})
        root.attrs.update(attrs_payload)
        prov = root.create_group("provenance")
        for name, value in provenance_payload.items():
            _write_string_scalar(prov, name, str(value))
        if write_path != final_path:
            _finalize_zarr_store(write_path, final_path, logger=logger, label="Empty shard")
        else:
            logger.info("Empty shard written to %s", final_path)
        return

    ############################################
    # Columns
    ############################################

    required_cols = ["teff", "logg", "feh", "lam_min", "lam_max", "lam_step"]
    optional_cols = [k for k in ("turbvel", "t_value", "output_mode", "calculation_mode", "mode", "grid_version") if k in grid]
    if "turbvel" not in optional_cols and "t_value" not in optional_cols:
        raise KeyError("Grid Zarr must include either 'turbvel' or 't_value' for synthesis")

    passthrough_cols = sorted(
        name for name in grid.keys() if name not in set(required_cols + optional_cols)
    )
    columns = {k: np.asarray(grid[k][indices]) for k in required_cols + optional_cols + passthrough_cols}

    ############################################
    # Workers
    ############################################

    worker_count = determine_worker_count(config, requested_workers=args.workers)

    ############################################
    # Build adaptive batches
    ############################################

    tasks, global_to_local = _build_task_batches(
        indices,
        columns,
        batch_size=args.batch_size,
        config=config,
    )

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
    mu_selected = np.full(len(indices), np.nan, dtype=np.float32)
    mu_selected_index = np.full(len(indices), -1, dtype=np.int16)
    base_names = [""] * len(indices)

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
                base_names[idx] = str(result.get("base_name", "") or "")
                try:
                    mu_selected[idx] = float(result.get("mu_selected", np.nan))
                except Exception:
                    mu_selected[idx] = np.nan
                try:
                    mu_selected_index[idx] = int(result.get("mu_selected_index", -1))
                except Exception:
                    mu_selected_index[idx] = -1

                if result["spectrum"]:
                    fluxes[idx], continua[idx] = result["spectrum"]

                done += 1

                status_text = str(result["status"]).upper()
                if str(result["status"]).lower() in SUCCESS_STATUSES:
                    logger.info(
                        "[%d/%d] global=%d %s (%.2fs)",
                        done,
                        len(indices),
                        result["global_index"],
                        status_text,
                        result["duration"],
                    )
                else:
                    logger.error(
                        "[%d/%d] global=%d %s (%.2fs) - %s",
                        done,
                        len(indices),
                        result["global_index"],
                        status_text,
                        result["duration"],
                        result["message"],
                    )

    ############################################
    # Validate & write shard (always write to preserve HPC work)
    ############################################

    status_counts = validate_synthesis_results(
        statuses=statuses,
        messages=messages,
        fluxes=fluxes,
        continua=continua,
        max_failures=args.max_failures,
    )

    n_failures = sum(v for k, v in status_counts.items() if k.lower() not in SUCCESS_STATUSES)
    validation_failed = n_failures > args.max_failures

    logger.info(
        "Writing shard output: %s (rows=%d wl=%d)",
        write_path if write_path != final_path else final_path,
        len(indices),
        len(wavelengths),
    )
    root = _open_root_for_write(write_path)

    _write_array(root, "wavelength", wavelengths, chunks=min(65536, max(1, wavelengths.size)))
    _write_array(root, "global_index", indices.astype(np.int64), chunks=max(1, min(2048, len(indices))))
    _write_array(root, "flux", fluxes, chunks=(1, min(65536, max(1, wavelengths.size))))
    _write_array(root, "continuum", continua, chunks=(1, min(65536, max(1, wavelengths.size))))
    _write_array(root, "mu_selected", mu_selected, chunks=max(1, min(2048, len(mu_selected))))
    _write_array(root, "mu_selected_index", mu_selected_index, chunks=max(1, min(2048, len(mu_selected_index))))
    _write_string_1d(root, "status", statuses, chunks=max(1, min(256, len(statuses))))
    _write_string_1d(root, "message", messages, chunks=max(1, min(256, len(messages))))
    # Optional: base model filenames for debugging/traceability.
    if any(isinstance(name, str) and name for name in base_names):
        try:
            _write_string_1d(root, "base_name", base_names, chunks=max(1, min(256, len(base_names))))
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

    shard_output_mode_values = sorted(
        {
            str(value).strip()
            for value in np.asarray(columns.get("output_mode", np.asarray([], dtype=object))).tolist()
            if str(value).strip()
        }
    )
    if not shard_output_mode_values:
        shard_output_mode_values = grid_output_mode_values
    resolved_flux_definition, resolved_flux_unit = infer_flux_metadata(shard_output_mode_values)

    attrs_payload: Dict[str, Any] = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "grid": os.path.abspath(args.grid_zarr),
        "flux_definition": resolved_flux_definition,
        "flux_unit": resolved_flux_unit,
        "output_mode_values": shard_output_mode_values,
        "mu_sampling": json.dumps(getattr(config, "mu_sampling", {}) or {}, sort_keys=True),
        "created_utc": created_utc,
        "git_sha": git_commit,
        "synthesis_timestamp": created_utc,
        "linelist_path": str(getattr(config, "linelist_path", "")),
        "linelist_files": json.dumps([str(x) for x in (getattr(config, "linelist_files", []) or [])], sort_keys=True),
        "linelist_version": linelist_version,
        "linelist_sha256": linelist_sha,
        "linelist_preprocessing": str(getattr(config, "linelist_preprocessing", "")),
        "model_atmosphere_path": str(getattr(config, "model_atmosphere_path", "")),
        "atmosphere_geometry": atmosphere_geometry,
        "atmosphere_version": atmosphere_version,
        "atmosphere_sha256": atmosphere_sha,
        "synthesis_code_version": turbospectrum_version,
        "spice_version": pipeline_version,
        "compiler": str(getattr(config, "compiler", "")),
        "nlte": bool(getattr(config, "nlte", False)),
        "status_counts": dict(status_counts),
    }
    attrs_payload.update({str(k): str(v) for k, v in contract_provenance.items()})
    root.attrs.update(attrs_payload)
    prov = root.create_group("provenance")
    for name, value in provenance_payload.items():
        _write_string_scalar(prov, name, str(value))

    if write_path != final_path:
        _finalize_zarr_store(write_path, final_path, logger=logger, label="Shard")
    else:
        logger.info("Shard written to %s", final_path)

    if validation_failed:
        logger.error(
            "Shard saved to %s but exiting with error: %d failure(s) exceed --max-failures %d",
            final_path,
            n_failures,
            args.max_failures,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
