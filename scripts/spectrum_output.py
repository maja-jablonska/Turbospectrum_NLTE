#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_FLUX_DEFINITION = "continuum_normalized"
DEFAULT_FLUX_UNIT = "relative"


def reconstruct_continuum(abs_flux: np.ndarray, norm_flux: np.ndarray) -> np.ndarray:
    """Reconstruct continuum from absolute and normalized quantities: cont = abs / norm."""
    abs_arr = np.asarray(abs_flux, dtype=np.float32)
    norm_arr = np.asarray(norm_flux, dtype=np.float32)
    cont = np.full_like(abs_arr, np.nan, dtype=np.float32)
    valid = np.isfinite(abs_arr) & np.isfinite(norm_arr) & (np.abs(norm_arr) > np.float32(1e-8))
    np.divide(abs_arr, norm_arr, out=cont, where=valid)
    return cont


def extract_flux_and_continuum(
    data: np.ndarray,
    *,
    is_intensity: bool,
    chosen_idx: np.ndarray | None = None,
    reduce_mode: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the stored spectrum and continuum from a Turbospectrum text output."""
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected spectrum table shape: {arr.shape}")

    if is_intensity and chosen_idx is not None and chosen_idx.size > 0:
        selected = np.asarray(chosen_idx, dtype=np.int64)
        abs_cols = [int(3 + 2 * i) for i in selected.tolist()]
        norm_cols = [int(4 + 2 * i) for i in selected.tolist()]
        if arr.shape[1] <= max(abs_cols + norm_cols):
            raise ValueError(f"Intensity spectrum has too few columns: {arr.shape}")
        i_abs = arr[:, abs_cols].astype(np.float32)
        i_norm = arr[:, norm_cols].astype(np.float32)
        if i_abs.ndim == 1:
            i_abs = i_abs[:, None]
            i_norm = i_norm[:, None]
        if str(reduce_mode).lower() == "mean" and i_abs.shape[1] > 1:
            flux = i_abs.mean(axis=1)
            norm = i_norm.mean(axis=1)
        else:
            flux = i_abs[:, 0]
            norm = i_norm[:, 0]
        return flux, reconstruct_continuum(flux, norm)

    norm_flux = arr[:, 1].astype(np.float32)
    if arr.shape[1] > 2:
        abs_flux = arr[:, 2].astype(np.float32)
        return abs_flux, reconstruct_continuum(abs_flux, norm_flux)

    return norm_flux, np.full_like(norm_flux, np.nan, dtype=np.float32)


def infer_flux_metadata(
    output_mode_values: Iterable[object] | None,
    *,
    explicit_definition: str | None = None,
    explicit_unit: str | None = None,
    inherited_definition: str | None = None,
    inherited_unit: str | None = None,
) -> tuple[str, str]:
    """Resolve dataset-level flux metadata from CLI overrides and output mode."""
    explicit = str(explicit_definition or "").strip()
    explicit_unit_value = str(explicit_unit or "").strip()
    if explicit and explicit.lower() != "auto":
        return explicit, explicit_unit_value or DEFAULT_FLUX_UNIT

    modes = sorted(
        {
            str(value).strip().lower()
            for value in (output_mode_values or [])
            if str(value).strip()
        }
    )
    if len(modes) == 1:
        mode = modes[0]
        if mode == "flux":
            return "unnormalized_flux", DEFAULT_FLUX_UNIT
        if mode == "intensity":
            return "unnormalized_intensity", DEFAULT_FLUX_UNIT
        return f"unnormalized_{mode}", DEFAULT_FLUX_UNIT
    if len(modes) > 1:
        return "mode_dependent", DEFAULT_FLUX_UNIT

    inherited = str(inherited_definition or "").strip()
    inherited_unit_value = str(inherited_unit or "").strip()
    if inherited and inherited.lower() != "auto":
        return inherited, inherited_unit_value or explicit_unit_value or DEFAULT_FLUX_UNIT

    return DEFAULT_FLUX_DEFINITION, inherited_unit_value or explicit_unit_value or DEFAULT_FLUX_UNIT
