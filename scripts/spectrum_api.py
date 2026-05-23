#!/usr/bin/env python3
"""Lightweight Python API for synthesizing a single spectrum on the fly.

Wraps `TurbospectrumConfig` + `run_single_synthesis` so callers (notebooks,
validation scripts) get back numpy arrays without touching the CLI, PBS,
or Zarr machinery. This is for ad-hoc validation, not production grids —
the scalable path is still `pipeline_from_config.py` per CLAUDE.md.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from run_turbospectrum import (  # type: ignore  # noqa: E402
    LinelistValidationError,
    TurbospectrumConfig,
    create_linelist_file,
    ensure_directories,
    run_single_synthesis,
)
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)


@dataclass
class SpectrumResult:
    wavelength: np.ndarray
    flux: np.ndarray
    flux_normalized: np.ndarray
    continuum: np.ndarray
    mu_angles: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    intensity_normalized: Optional[np.ndarray] = None
    params: dict = field(default_factory=dict)
    output_mode: str = "Flux"
    nlte: bool = False
    output_path: str = ""
    log_path: str = ""
    base_name: str = ""
    raw_columns: Optional[np.ndarray] = None

    @property
    def n_points(self) -> int:
        return int(self.wavelength.size)

    @property
    def n_mu(self) -> int:
        return int(self.mu_angles.size) if self.mu_angles is not None else 0


def _parse_mu_points(spec_path: str) -> np.ndarray:
    """Read the '# mu-points ...' header from an Intensity .spec file.

    Returns a 1D float array (possibly empty) ordered as the file writes them
    — small mu first (limb), 1.0 last (disk center), in the standard
    plane-parallel set."""
    try:
        with open(spec_path, encoding="utf-8", errors="ignore") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                if not line.lstrip().startswith("#"):
                    break
                if re.search(r"mu\s*-?\s*points", line, flags=re.IGNORECASE):
                    tail = re.split(r"mu\s*-?\s*points\s*:?", line, flags=re.IGNORECASE)[-1]
                    nums = re.findall(
                        r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?",
                        tail,
                    )
                    return np.asarray(
                        [float(n.replace("D", "E").replace("d", "e")) for n in nums],
                        dtype=np.float64,
                    )
    except OSError:
        pass
    return np.empty(0, dtype=np.float64)


_MAX_MU_POINTS = 30  # bsyn allocates muoutp(30) — see source/input.f


def _validate_mu_angles(mu_angles: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(mu_angles), dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("mu_angles must be non-empty when supplied")
    if arr.size > _MAX_MU_POINTS:
        raise ValueError(
            f"mu_angles has {arr.size} entries; bsyn supports at most {_MAX_MU_POINTS}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("mu_angles must all be finite")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(f"mu_angles must lie in [0, 1]; got [{arr.min()}, {arr.max()}]")
    return arr


def _write_mu_points_file(mus: np.ndarray, out_dir: str) -> str:
    """Write a bsyn-format mu-points file (line 1 = count, line 2 = values)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mupoints.dat")
    # bsyn's file reader accepts comma- or whitespace-separated values.
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{mus.size}\n")
        fh.write(", ".join(f"{m:.6f}" for m in mus) + "\n")
    return path


def _vmicro_to_label(vmicro: Any) -> str:
    """Map a microturbulence value (km/s) to the MARCS turbulence label.

    MARCS files are tagged `_t01`, `_t02`, `_t05`, … — integer km/s. We round
    to the nearest available integer label; the synthesis-time turbulence is
    passed through as ``turbvel`` so the line formation still uses the float."""
    if isinstance(vmicro, str) and vmicro.strip():
        return vmicro.strip().zfill(2)
    try:
        v = float(vmicro)
    except (TypeError, ValueError):
        return "01"
    return f"{max(0, int(round(v))):02d}"


def _default_model_atmosphere_path(project_root: str) -> str:
    env = os.environ.get("MODEL_PATH")
    if env:
        return env
    return os.path.join(
        project_root, "input_files", "model_atmospheres", "1D", "marcs_standard_comp"
    )


def _default_linelist_path(project_root: str) -> str:
    env = os.environ.get("LINELIST_PATH")
    if env:
        return env
    return os.path.join(project_root, "input_files", "linelists")


def synthesize_spectrum(
    teff: float,
    logg: float,
    feh: float,
    *,
    vmicro: Any = 1.0,
    lambda_min: float = 5000.0,
    lambda_max: float = 5500.0,
    lambda_step: float = 0.05,
    abundances: Optional[Mapping[str, float]] = None,
    output_mode: str = "Flux",
    mu_angles: Optional[Sequence[float]] = None,
    nlte: bool = False,
    nlte_info_file: Optional[str] = None,
    model_atmosphere_path: Optional[str] = None,
    linelist_path: Optional[str] = None,
    linelist_files: Optional[Sequence[str]] = None,
    compiler: str = "gf",
    project_root: Optional[str] = None,
    work_dir: Optional[str] = None,
    keep_files: bool = False,
    force: bool = True,
) -> SpectrumResult:
    """Run Turbospectrum once and return a `SpectrumResult` with numpy arrays.

    Parameters mirror what a single grid point in the pipeline would carry.
    Defaults pull from the same env vars as ``scripts/env.sh`` so an existing
    repo layout works without configuration.

    `abundances` accepts symbols or short names: ``{"alpha": 0.0, "C": -0.2}``.
    `vmicro` may be a float (km/s) or a string label like ``"01"``.

    `mu_angles`: ignored unless ``output_mode='Intensity'``. When ``None``
    (default), bsyn computes the 12 Gauss-Radau mu points it ships with. When a
    sequence is supplied, the API writes a temporary mu-points file and bsyn
    computes intensities at exactly those mus. All values must lie in [0, 1];
    bsyn supports up to 30 mu points.

    With ``keep_files=False`` (default), all opacity/spec/log files are written
    to a temp dir that's removed before returning. Set to ``True`` (and pass
    ``work_dir`` if you want a stable location) to keep them for inspection.
    """
    project_root = os.path.abspath(project_root or _PROJECT_ROOT)
    model_atmosphere_path = os.path.abspath(
        model_atmosphere_path or _default_model_atmosphere_path(project_root)
    )
    linelist_path = os.path.abspath(linelist_path or _default_linelist_path(project_root))
    linelist_files = list(linelist_files or ["nlte_ges_linelist_jmg6may2025_I_II"])

    if work_dir is None:
        cleanup_dir: Optional[str] = tempfile.mkdtemp(prefix="turbo_api_")
        run_root = cleanup_dir
    else:
        cleanup_dir = None
        run_root = os.path.abspath(work_dir)
        os.makedirs(run_root, exist_ok=True)

    output_dir = os.path.join(run_root, "spectra")
    log_dir = os.path.join(run_root, "logs")
    tmp_dir = os.path.join(run_root, "tmp")

    output_mode_norm = str(output_mode).strip().capitalize()
    if output_mode_norm not in {"Flux", "Intensity"}:
        raise ValueError(f"output_mode must be 'Flux' or 'Intensity', got {output_mode!r}")

    config_kwargs: dict[str, Any] = dict(
        project_root=project_root,
        compiler=compiler,
        model_atmosphere_path=model_atmosphere_path,
        linelist_path=linelist_path,
        linelist_files=list(linelist_files),
        output_dir=output_dir,
        log_dir=log_dir,
        tmp_dir=tmp_dir,
        output_mode=output_mode_norm,
        nlte=bool(nlte),
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        lambda_step=float(lambda_step),
        force=bool(force),
        max_workers=1,
    )
    if mu_angles is not None:
        if output_mode_norm != "Intensity":
            raise ValueError(
                "mu_angles is only meaningful with output_mode='Intensity'"
            )
        validated_mus = _validate_mu_angles(mu_angles)
        os.makedirs(tmp_dir, exist_ok=True)
        mu_file_path = _write_mu_points_file(validated_mus, tmp_dir)
        config_kwargs["mu_angles"] = validated_mus.tolist()
        config_kwargs["mu_points_file"] = mu_file_path
    if nlte_info_file:
        config_kwargs["nlte_info_file"] = os.path.abspath(nlte_info_file)

    config = TurbospectrumConfig(**config_kwargs)

    grid_point: dict[str, Any] = {
        "teff": int(round(float(teff))),
        "logg": float(logg),
        "feh": float(feh),
        "turbvel": vmicro,
        "t_value": _vmicro_to_label(vmicro),
        "lam_min": float(lambda_min),
        "lam_max": float(lambda_max),
        "lam_step": float(lambda_step),
        "output_mode": output_mode_norm,
        "calculation_mode": "NLTE" if nlte else "LTE",
    }
    if abundances:
        for key, value in abundances.items():
            grid_point[str(key)] = float(value)

    succeeded = False
    try:
        ensure_directories(config)
        try:
            config.linelist_file_path = create_linelist_file(config)
        except LinelistValidationError as exc:
            raise RuntimeError(f"Linelist validation failed: {exc}") from exc

        result = run_single_synthesis((grid_point, config))

        if result["status"] not in {"success", "skipped"}:
            log_path = result.get("log_path", "")
            log_hint = f" (see {log_path})" if log_path else ""
            kept_hint = f" Logs preserved under {run_root}." if cleanup_dir is not None else ""
            raise RuntimeError(
                f"Turbospectrum synthesis failed: {result.get('message', '')}{log_hint}.{kept_hint}"
            )

        spec_path = result.get("output_path") or ""
        if not spec_path or not os.path.exists(spec_path):
            suffix = ".intensity.spec" if output_mode_norm == "Intensity" else ".spec"
            spec_path = os.path.join(output_dir, f"{result['base_name']}{suffix}")
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spectrum file not found: {spec_path}")

        data = np.loadtxt(spec_path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Unexpected spectrum table shape: {data.shape}")

        wavelength = data[:, 0].astype(np.float64)
        # Cols 1, 2 are disk-integrated normalized + absolute flux in BOTH
        # output modes — bsyn writes them whether or not per-mu intensities
        # follow.
        flux_norm = data[:, 1].astype(np.float32)
        if data.shape[1] > 2:
            flux_abs = data[:, 2].astype(np.float32)
            continuum = np.full_like(flux_abs, np.nan, dtype=np.float32)
            valid = (
                np.isfinite(flux_abs)
                & np.isfinite(flux_norm)
                & (np.abs(flux_norm) > np.float32(1e-8))
            )
            np.divide(flux_abs, flux_norm, out=continuum, where=valid)
        else:
            flux_abs = flux_norm
            continuum = np.full_like(flux_norm, np.nan, dtype=np.float32)

        mu_array: Optional[np.ndarray] = None
        intensity_abs: Optional[np.ndarray] = None
        intensity_norm: Optional[np.ndarray] = None
        if output_mode_norm == "Intensity":
            n_mu_cols = max(0, (data.shape[1] - 3) // 2)
            parsed = _parse_mu_points(spec_path)
            if parsed.size == n_mu_cols and n_mu_cols > 0:
                mus = parsed
            elif n_mu_cols > 0:
                # Header missing/malformed — fall back to a uniform (0,1] grid.
                mus = np.linspace(1.0, 1.0 / n_mu_cols, n_mu_cols)[::-1]
            else:
                mus = np.empty(0, dtype=np.float64)

            if n_mu_cols > 0:
                mu_array = mus
                intensity_abs = np.stack(
                    [data[:, 3 + 2 * i] for i in range(n_mu_cols)], axis=0
                ).astype(np.float32)
                intensity_norm = np.stack(
                    [data[:, 4 + 2 * i] for i in range(n_mu_cols)], axis=0
                ).astype(np.float32)

        succeeded = True
        return SpectrumResult(
            wavelength=wavelength,
            flux=flux_abs,
            flux_normalized=flux_norm,
            continuum=continuum,
            mu_angles=mu_array,
            intensity=intensity_abs,
            intensity_normalized=intensity_norm,
            params=result.get("params", {}),
            output_mode=output_mode_norm,
            nlte=bool(nlte),
            output_path=spec_path if keep_files else "",
            log_path=result.get("log_path", "") if keep_files else "",
            base_name=result.get("base_name", ""),
            raw_columns=data if output_mode_norm == "Intensity" else None,
        )
    finally:
        # Only remove an auto-created temp dir on success. On failure, keep it
        # so the log path referenced in the exception message is still readable.
        if cleanup_dir is not None and succeeded and not keep_files:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


__all__ = ["SpectrumResult", "synthesize_spectrum"]
