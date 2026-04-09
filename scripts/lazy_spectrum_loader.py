#!/usr/bin/env python3
"""Lazy spectrum lookup from a Zarr store by parameter combination.

This utility is designed for stores that expose at least:
- `params`: shape (n_rows, n_params)
- `param_names`: shape (n_params,)
- `flux`: shape (n_rows, n_lambda) or (n_rows, ...)

It scans only chunks of `params` to find matching row(s), then loads only the
requested row from `flux`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import zarr


try:
    from zarr_compat import zarr_store as _zarr_store
except ImportError:
    try:
        from .zarr_compat import zarr_store as _zarr_store
    except ImportError:
        def _zarr_store(path: str):
            if hasattr(zarr, "DirectoryStore"):
                return zarr.DirectoryStore(path)
            from zarr import storage as zstorage
            if hasattr(zstorage, "LocalStore"):
                return zstorage.LocalStore(path)
            return zstorage.DirectoryStore(path)


def _decode_strings(values: Sequence[object] | np.ndarray) -> tuple[str, ...]:
    out: list[str] = []
    for item in np.asarray(values).tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return tuple(out)


def _parse_param_assignments(items: Iterable[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for raw in items:
        text = str(raw).strip()
        if "=" not in text:
            raise ValueError(f"Invalid parameter '{text}': expected name=value")
        name, value = text.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"Invalid parameter '{text}': missing name")
        try:
            parsed[key] = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid value for parameter '{key}': {value}") from exc
    if not parsed:
        raise ValueError("No parameters provided")
    return parsed


def _take_flux_row(array, row_index: int) -> np.ndarray:
    ridx = int(row_index)
    try:
        return np.asarray(array.oindex[ridx, ...])  # type: ignore[attr-defined]
    except Exception:
        pass
    return np.asarray(array[ridx, ...])


@dataclass(frozen=True)
class SpectrumMatch:
    row_index: int
    spectrum: np.ndarray
    wavelength: np.ndarray | None


class LazySpectrumLoader:
    """Find and fetch a single spectrum by parameter values with chunked reads."""

    def __init__(
        self,
        zarr_path: str,
        *,
        params_key: str = "params",
        param_names_key: str = "param_names",
        flux_key: str = "flux",
        wavelength_key: str = "wavelength",
        scan_chunk_rows: int = 4096,
    ):
        self.zarr_path = zarr_path
        self.params_key = params_key
        self.param_names_key = param_names_key
        self.flux_key = flux_key
        self.wavelength_key = wavelength_key
        self.scan_chunk_rows = int(scan_chunk_rows)

        if self.scan_chunk_rows <= 0:
            raise ValueError("scan_chunk_rows must be positive")

        store = _zarr_store(zarr_path)
        self.root = zarr.open_group(store=store, mode="r")

        required = [params_key, param_names_key, flux_key]
        missing = [name for name in required if name not in self.root]
        if missing:
            raise KeyError(f"Missing required arrays in {zarr_path}: {missing}")

        self.params = self.root[params_key]
        self.param_names = _decode_strings(np.asarray(self.root[param_names_key][:]))
        self._col_by_name = {name: i for i, name in enumerate(self.param_names)}
        self.flux = self.root[flux_key]

        if self.params.ndim != 2:
            raise ValueError(f"Expected '{params_key}' to be 2D, got shape={self.params.shape}")
        if self.flux.ndim < 2:
            raise ValueError(f"Expected '{flux_key}' to be at least 2D, got shape={self.flux.shape}")
        if int(self.params.shape[0]) != int(self.flux.shape[0]):
            raise ValueError(
                f"Row mismatch: {params_key}.shape[0]={self.params.shape[0]} vs "
                f"{flux_key}.shape[0]={self.flux.shape[0]}"
            )
        if int(self.params.shape[1]) != len(self.param_names):
            raise ValueError(
                f"Column mismatch: {params_key}.shape[1]={self.params.shape[1]} vs "
                f"len({param_names_key})={len(self.param_names)}"
            )

    @property
    def row_count(self) -> int:
        return int(self.params.shape[0])

    def find_row_index(
        self,
        param_values: Mapping[str, float],
        *,
        atol: float = 1.0e-6,
        rtol: float = 0.0,
        require_all_params: bool = True,
        allow_multiple: bool = False,
    ) -> int:
        if not param_values:
            raise ValueError("param_values cannot be empty")

        requested = tuple(param_values.keys())
        unknown = [name for name in requested if name not in self._col_by_name]
        if unknown:
            raise KeyError(f"Unknown parameter name(s): {unknown}. Available: {list(self.param_names)}")

        if require_all_params:
            missing = [name for name in self.param_names if name not in param_values]
            if missing:
                raise ValueError(f"Missing parameter(s) for full-row match: {missing}")

        col_ids = np.asarray([self._col_by_name[name] for name in requested], dtype=np.int64)
        query = np.asarray([float(param_values[name]) for name in requested], dtype=np.float64)

        matches: list[int] = []
        for start in range(0, self.row_count, self.scan_chunk_rows):
            stop = min(start + self.scan_chunk_rows, self.row_count)
            block = np.asarray(self.params[start:stop, :], dtype=np.float64)
            block = block[:, col_ids]
            is_match = np.isclose(block, query[None, :], atol=atol, rtol=rtol, equal_nan=True).all(axis=1)
            if not np.any(is_match):
                continue
            local_ids = np.flatnonzero(is_match)
            matches.extend(int(start + lid) for lid in local_ids.tolist())
            if matches and not allow_multiple:
                break

        if not matches:
            raise KeyError(
                "No matching spectrum found for parameters: "
                + ", ".join(f"{k}={v}" for k, v in param_values.items())
            )
        if len(matches) > 1 and not allow_multiple:
            raise ValueError(
                f"Multiple matching rows found ({len(matches)}). "
                "Provide all params or tighten atol/rtol."
            )
        return matches[0]

    def fetch_spectrum(
        self,
        param_values: Mapping[str, float],
        *,
        atol: float = 1.0e-6,
        rtol: float = 0.0,
        require_all_params: bool = True,
        include_wavelength: bool = True,
    ) -> SpectrumMatch:
        row_index = self.find_row_index(
            param_values,
            atol=atol,
            rtol=rtol,
            require_all_params=require_all_params,
            allow_multiple=False,
        )
        spectrum = _take_flux_row(self.flux, row_index)
        wavelength = None
        if include_wavelength and self.wavelength_key in self.root:
            wavelength = np.asarray(self.root[self.wavelength_key][:])
        return SpectrumMatch(row_index=row_index, spectrum=spectrum, wavelength=wavelength)


def load_spectrum_for_params(
    zarr_path: str,
    param_values: Mapping[str, float],
    *,
    atol: float = 1.0e-6,
    rtol: float = 0.0,
    scan_chunk_rows: int = 4096,
) -> SpectrumMatch:
    """Convenience one-shot helper."""
    loader = LazySpectrumLoader(zarr_path=zarr_path, scan_chunk_rows=scan_chunk_rows)
    return loader.fetch_spectrum(param_values, atol=atol, rtol=rtol)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lazy Zarr spectrum lookup by parameter combination.")
    parser.add_argument("zarr_path", help="Path to spectra Zarr store")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Parameter assignment in name=value form (repeat for multiple params)",
    )
    parser.add_argument("--atol", type=float, default=1.0e-6, help="Absolute tolerance for float matching")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for float matching")
    parser.add_argument(
        "--scan-chunk-rows",
        type=int,
        default=4096,
        help="Rows per chunk when scanning params for matches",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow subset parameter queries (may be ambiguous if non-unique)",
    )
    parser.add_argument(
        "--no-wavelength",
        action="store_true",
        help="Do not load wavelength array in output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    params = _parse_param_assignments(args.param)
    loader = LazySpectrumLoader(zarr_path=args.zarr_path, scan_chunk_rows=args.scan_chunk_rows)
    result = loader.fetch_spectrum(
        params,
        atol=float(args.atol),
        rtol=float(args.rtol),
        require_all_params=not bool(args.allow_partial),
        include_wavelength=not bool(args.no_wavelength),
    )

    print(f"row_index={result.row_index}")
    print(f"spectrum_shape={tuple(result.spectrum.shape)}")
    if result.wavelength is not None:
        print(f"wavelength_shape={tuple(result.wavelength.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
