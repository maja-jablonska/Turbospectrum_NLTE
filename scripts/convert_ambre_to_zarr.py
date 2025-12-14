#!/usr/bin/env python3
"""Convert AMBRE FITS spectra into a consolidated Zarr store.

This script reads every AMBRE FITS spectrum in an input directory and writes
all spectra to a single Zarr v3 store with flux, normalized flux, wavelength,
and per-spectrum metadata derived from the filename.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import zarr
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from zarr.codecs import ZstdCodec

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
FILENAME_RE = re.compile(
    r"""
    ^
    (?P<model_family>[A-Za-z]+)_
    p(?P<teff>\d+)
    g(?P<logg>-?\d+(?:\.\d+)?)
    z(?P<feh>-?\d+(?:\.\d+)?)
    t(?P<vturb>-?\d+(?:\.\d+)?)
    _
    a(?P<alpha>-?\d+(?:\.\d+)?)
    c(?P<cfe>-?\d+(?:\.\d+)?)
    n(?P<nfe>-?\d+(?:\.\d+)?)
    o(?P<ofe>-?\d+(?:\.\d+)?)
    r(?P<rfe>-?\d+(?:\.\d+)?)
    s(?P<sfe>-?\d+(?:\.\d+)?)
    _
    (?P<domain>[A-Z]+)
    (?:\.spec)?     # Optional '.spec' extension
    $
    """,
    re.VERBOSE,
)


def parse_spectrum_filename(filename: str) -> Dict[str, object]:
    """Parse AMBRE spectrum filename into metadata fields."""

    base = filename
    for ext in [".fits", ".fit", ".gz", ".bz2", ".txt", ".dat"]:
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    m = FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Unrecognized filename format: {filename}")

    d: Dict[str, object] = m.groupdict()
    for k in [
        "teff",
        "logg",
        "feh",
        "vturb",
        "alpha",
        "cfe",
        "nfe",
        "ofe",
        "rfe",
        "sfe",
    ]:
        d[k] = float(d[k]) if "." in d[k] or "-" in d[k] else int(d[k])

    return d


# ---------------------------------------------------------------------------
# FITS loading
# ---------------------------------------------------------------------------
def read_ambre_fits(
    fits_path: Path,
    *,
    hdu_index: int = 1,
    flux_col: str = "flux",
    norm_flux_col: str = "normalized flux",
    wave_col: str | None = "wavelength",
    cast_float32: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Read a single AMBRE FITS file and return flux, normalized flux, wavelength.

    Returns
    -------
    flux : np.ndarray
        Shape (N_spec, N_lambda).
    flux_norm : np.ndarray
        Shape (N_spec, N_lambda).
    wavelength : np.ndarray
        Shape (N_lambda,).
    metadata : dict
        Parsed filename parameters plus a ``source_file`` field.
    """

    with fits.open(fits_path, memmap=True) as hdul:
        hdu = hdul[hdu_index]

        if not isinstance(hdu, fits.BinTableHDU):
            raise ValueError(
                f"HDU {hdu_index} in {fits_path} is not a BinTableHDU (required)."
            )

        table = Table(hdu.data)
        flux = np.asarray(table[flux_col])
        flux_norm = np.asarray(table[norm_flux_col])

        if wave_col is not None:
            wavelength = np.asarray(table[wave_col])
        else:
            wcs = WCS(hdu.header)
            n_lambda = flux.shape[1]
            wavelength = wcs.pixel_to_world(np.arange(n_lambda), 0)[0].value

        header = dict(hdu.header)

    # Normalize shapes
    if flux.shape != flux_norm.shape:
        raise ValueError(
            f"FLUX and FLUX_NORM shapes do not match in {fits_path}: {flux.shape} vs {flux_norm.shape}"
        )

    if flux.ndim == 1:
        n_lambda = flux.shape[0]
        flux = flux.reshape(1, n_lambda)
        flux_norm = flux_norm.reshape(1, n_lambda)
    elif flux.ndim == 2:
        _, n_lambda = flux.shape
    else:
        raise ValueError(f"Flux array in {fits_path} must be 1D or 2D, got {flux.shape}.")

    if wavelength.ndim == 2 and wavelength.shape[0] == 1:
        wavelength = wavelength[0]
    elif wavelength.ndim == 2 and wavelength.shape[1] == 1:
        wavelength = wavelength[:, 0]

    if wavelength.shape == (flux.shape[0], n_lambda):
        wavelength = wavelength[0]
    elif wavelength.shape != (n_lambda,):
        raise ValueError(
            f"Wavelength shape mismatch in {fits_path}: expected ({n_lambda},), got {wavelength.shape}"
        )

    if cast_float32:
        flux = flux.astype("f4", copy=False)
        flux_norm = flux_norm.astype("f4", copy=False)
        wavelength = wavelength.astype("f8", copy=False)

    metadata = parse_spectrum_filename(fits_path.stem)
    metadata.update(
        {
            "source_file": fits_path.name,
            "header": json.dumps(header),
        }
    )
    return flux, flux_norm, wavelength, metadata


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert_directory_to_zarr(
    input_dir: Path,
    output: Path,
    *,
    batch_chunk: int = 128,
    compression_level: int = 3,
    cast_float32: bool = True,
    hdu_index: int = 1,
    flux_col: str = "flux",
    norm_flux_col: str = "normalized flux",
    wave_col: str | None = "wavelength",
) -> None:
    """Convert every AMBRE FITS file in ``input_dir`` to a Zarr store."""

    valid_suffixes = (".fits", ".fit", ".fits.gz", ".fit.gz")
    fits_files = sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.name.lower().endswith(valid_suffixes)
    )
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {input_dir}")

    flux_list: List[np.ndarray] = []
    flux_norm_list: List[np.ndarray] = []
    meta_accum: Dict[str, List[object]] = {}
    reference_wavelength: np.ndarray | None = None

    for fits_path in fits_files:
        flux, flux_norm, wavelength, metadata = read_ambre_fits(
            fits_path,
            hdu_index=hdu_index,
            flux_col=flux_col,
            norm_flux_col=norm_flux_col,
            wave_col=wave_col,
            cast_float32=cast_float32,
        )

        if reference_wavelength is None:
            reference_wavelength = wavelength
        else:
            if wavelength.shape != reference_wavelength.shape or not np.allclose(
                wavelength, reference_wavelength
            ):
                raise ValueError(
                    f"Wavelength grid mismatch in {fits_path}; expected shape {reference_wavelength.shape}"
                )

        n_spec = flux.shape[0]
        flux_list.append(flux)
        flux_norm_list.append(flux_norm)

        for key, value in metadata.items():
            meta_accum.setdefault(key, []).extend([value] * n_spec)

    all_flux = np.concatenate(flux_list, axis=0)
    all_flux_norm = np.concatenate(flux_norm_list, axis=0)
    wavelength = reference_wavelength

    n_spec, n_lambda = all_flux.shape
    chunk_shape_2d = (min(batch_chunk, n_spec), n_lambda)
    chunk_shape_1d = (n_lambda,)

    codec = ZstdCodec(level=compression_level)
    compressors = [codec]

    root = zarr.open_group(output, mode="w")
    root.create_array("flux", data=all_flux, chunks=chunk_shape_2d, compressors=compressors)
    root.create_array(
        "flux_norm", data=all_flux_norm, chunks=chunk_shape_2d, compressors=compressors
    )
    root.create_array("wavelength", data=wavelength, chunks=chunk_shape_1d, compressors=compressors)

    meta_chunk = (min(batch_chunk, n_spec),)
    for key, values in meta_accum.items():
        dtype = "f4" if isinstance(values[0], (float, int)) and key != "header" else object
        data = np.array(values, dtype=dtype)
        root.create_array(key, data=data, chunks=meta_chunk, compressors=compressors)

    root.attrs.update(
        {
            "n_spectra": int(n_spec),
            "n_wavelength": int(n_lambda),
            "compression": "zstd",
            "compression_level": compression_level,
            "cast_float32": cast_float32,
            "zarr_format": 3,
            "source": str(input_dir),
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing AMBRE FITS spectra")
    parser.add_argument("output", type=Path, help="Path to output Zarr store")
    parser.add_argument("--batch-chunk", type=int, default=128, help="Chunk size along spectrum axis")
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Zstd compression level (1-22, typical 3)",
    )
    parser.add_argument(
        "--no-cast-float32",
        action="store_false",
        dest="cast_float32",
        help="Keep original floating dtype instead of casting to float32",
    )
    parser.add_argument("--hdu-index", type=int, default=1, help="HDU index containing spectra table")
    parser.add_argument("--flux-col", default="flux", help="Column name for raw flux")
    parser.add_argument(
        "--norm-flux-col", default="normalized flux", help="Column name for normalized flux"
    )
    parser.add_argument(
        "--wave-col",
        default="wavelength",
        help="Column name for wavelength (set to 'None' to derive from WCS)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    wave_col = None if args.wave_col.lower() == "none" else args.wave_col
    convert_directory_to_zarr(
        args.input_dir,
        args.output,
        batch_chunk=args.batch_chunk,
        compression_level=args.compression_level,
        cast_float32=args.cast_float32,
        hdu_index=args.hdu_index,
        flux_col=args.flux_col,
        norm_flux_col=args.norm_flux_col,
        wave_col=wave_col,
    )


if __name__ == "__main__":
    main()
