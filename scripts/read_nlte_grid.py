# Read TurboSpectrum NLTE binary grids (departure coefficients) with stream I/O.

from __future__ import annotations

import os
import struct
from typing import BinaryIO, Iterator, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore

# matplotlib is only needed for plot_departure_coefficients; import it lazily there
# so that reading/exporting grids does not require a plotting backend.

# Reasonable upper bounds to catch bad pointers / corrupt headers before huge allocations
_MAX_NDEP = 4096
_MAX_NLEV = 100_000


def read_grid_header(grid_file: str, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Read the fixed 1000-byte NLTE grid header (TurboSpectrum / Fortran stream format)."""
    with open(grid_file, "rb") as f:
        raw = f.read(1000)
    if len(raw) < 1000:
        raise ValueError(f"File shorter than 1000-byte header: {grid_file!r}")
    return raw.decode(encoding, errors=errors)


def _read_record_at(
    f: BinaryIO,
    pointer: int,
    *,
    show_progress: bool,
    desc: str,
) -> Tuple[int, int, np.ndarray, np.ndarray, str]:
    """
    pointer: 1-based byte offset of the record (first model byte = 1), as in Fortran aux files.
    """
    if pointer < 1:
        raise ValueError("pointer must be >= 1 (1-based byte position in the file)")

    f.seek(pointer - 1)
    id_raw = f.read(500)
    if len(id_raw) < 500:
        raise EOFError("Unexpected EOF while reading model id (500 bytes)")

    atmos_str = id_raw.decode("utf-8", "ignore").strip()

    hdr = f.read(8)
    if len(hdr) < 8:
        raise EOFError("Unexpected EOF while reading ndep / nlev")
    ndep, nk = struct.unpack("<ii", hdr)

    if not (1 <= ndep <= _MAX_NDEP and 1 <= nk <= _MAX_NLEV):
        raise ValueError(
            f"Unreasonable ndep={ndep} nk={nk} (limits {_MAX_NDEP}, {_MAX_NLEV}). "
            "Check pointer / auxiliary file."
        )

    tau_lin = np.fromfile(f, dtype="<f8", count=ndep)
    if tau_lin.size != ndep:
        raise EOFError("Unexpected EOF while reading tau grid")

    # Fortran nlte_data(ndep, nlev): depth index varies fastest on disk.
    # Read level-by-level (each level = ndep doubles) for progress and true streaming.
    depart = np.empty((nk, ndep), dtype=np.float64)
    iterator: Iterator[int] = range(nk)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, total=nk, desc=desc, unit="lvl")  # type: ignore[assignment]

    for lev in iterator:
        slab = np.fromfile(f, dtype="<f8", count=ndep)
        if slab.size != ndep:
            raise EOFError(f"Unexpected EOF while reading departures (level {lev})")
        depart[lev, :] = slab

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.log10(tau_lin)

    return ndep, nk, depart, tau, atmos_str


def read_binary_grid(
    grid_file: str,
    pointer: Optional[int] = None,
    *,
    show_progress: bool = True,
    progress_desc: str = "NLTE departures",
) -> Tuple[int, int, np.ndarray, np.ndarray, str]:
    """
    Read one NLTE record from a binary TurboSpectrum grid (Fortran stream layout).

    The file begins with a 1000-byte text header. Each record starts at a byte position
    given in the auxiliary model list (same convention as the Fortran interpolators).

    Parameters
    ----------
    grid_file : str
        Path to the binary NLTE grid.
    pointer : int, optional
        1-based byte offset where this record begins (as in the auxiliary file). If omitted,
        the first record is read immediately after the 1000-byte header (pointer = 1001).
    show_progress : bool
        If True and `tqdm` is installed, show a per-level progress bar while reading
        departure coefficients.
    progress_desc : str
        Description shown on the progress bar.

    Returns
    -------
    ndep : int
        Number of depth points.
    nk : int
        Number of energy levels.
    depart : ndarray
        Departure coefficients, shape (nk, ndep).
    tau : ndarray
        log10 of the depth scale (e.g. tau500), shape (ndep), for plotting and TS ascii output.
    atmosStr : str
        Model identifier string from the binary record.
    """
    if pointer is None:
        pointer = 1001  # first byte after 1000-byte grid header

    file_size = os.path.getsize(grid_file)
    if pointer > file_size:
        raise ValueError(f"pointer {pointer} is past end of file ({file_size} bytes)")

    with open(grid_file, "rb") as f:
        return _read_record_at(
            f,
            pointer,
            show_progress=show_progress,
            desc=progress_desc,
        )


def write_departures_for_ts(fileName, tau, depart, abund):
    """
    Writes NLTE departure coefficients into the file compatible
    with TurboSpectrum

    Parameters
    ----------
    fileName : str
        name of the file in which to write the departure coefficients
    tau : np.array
        depth scale in the model atmosphere used to solve for NLTE RT
        (e.g. TAU500nm)
    depart : np.ndarray
        departure coefficients
    abund : float
        chemical element abundance on log 12 scale
    """

    ndep = len(tau)
    nk = len(depart)
    with open(fileName, "w") as f:
        """  Comment lines below are requested by TS """
        for i in range(8):
            f.write('# parameter 1.0 1.0\n')

        f.write(f"{abund:.3f}\n")
        f.write(f"{ndep:.0f}\n")
        f.write(f"{nk:.0f}\n")
        for t in tau:
            f.write(F"{t:15.8E}\n")

        for i in range(ndep):
            f.write( f"{'  '.join(str(depart[j,i]) for j in range(nk))} \n" )


def plot_departure_coefficients(depart, tau, atmosStr, levels_to_plot):
    # plot departure coefficients
    import matplotlib.pyplot as plt

    # if levels_to_plot is int, convert to list
    if isinstance(levels_to_plot, int):
        levels_to_plot = [levels_to_plot]

    fig, ax = plt.subplots()
    for level in levels_to_plot:
        ax.plot(tau, depart[level - 1], label=f"level {level}")
    ax.set_xlabel(r"$\log_{10}(\tau)$")
    ax.set_ylabel(r"$b_{\rm NLTE}$")
    ax.set_title(f"Departure coefficients for {atmosStr}")
    ax.legend()
    plt.show()


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Read one record from a TurboSpectrum NLTE binary grid.")
    p.add_argument("grid_file", help="Path to NLTE .bin grid")
    p.add_argument(
        "--pointer",
        type=int,
        default=None,
        help="1-based byte position of record (default: first record after 1000-byte header)",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    args = p.parse_args()

    ndep, nk, depart, tau, name = read_binary_grid(
        args.grid_file,
        pointer=args.pointer,
        show_progress=not args.no_progress,
    )
    print(f"model: {name}")
    print(f"ndep={ndep}, n_levels={nk}, depart shape={depart.shape}")


if __name__ == "__main__":
    _main()
