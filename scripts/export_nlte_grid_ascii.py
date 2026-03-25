#!/usr/bin/env python3
"""
Export every NLTE departure record from a TurboSpectrum binary grid to individual
ASCII departure files (format expected by TurboSpectrum / read_departure).

Requires the auxiliary model list that accompanies the binary (same file the Fortran
interpolator reads: nlte_model_list). Each data line has:
  id_model  Teff  logg  [M/H]  alpha  mass  vturb  abundance(log ε, H=12)  byte_pointer

The model id may be wrapped in single quotes (e.g. auxData_*.dat). Comment lines start
with '#'. The first line is often a header comment.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

# Same directory as read_nlte_grid.py when run as `python scripts/export_nlte_grid_ascii.py`
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from read_nlte_grid import read_binary_grid, write_departures_for_ts

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


def _strip_optional_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


@dataclass(frozen=True)
class ModelListEntry:
    id_model: str
    teff: float
    logg: float
    metal: float
    alpha: float
    mass: float
    vturb: float
    abundance: float
    pointer: int
    line_index: int


def parse_nlte_model_list(path: str) -> List[ModelListEntry]:
    """
    Parse the NLTE auxiliary file (see interpol_multi_nlte*.f: read from unit 199).
    Each non-comment line: id_model and 7 floats and one integer pointer.
    The id may be CSV-style quoted (e.g. 'p2500_...') as in auxData_*.dat.
    """
    entries: List[ModelListEntry] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_index, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(
                    f"{path}:{line_index}: expected at least 9 whitespace-separated "
                    f"fields (id_model + 7 floats + pointer), got {len(parts)}: {line!r}"
                )
            pointer = int(parts[8])
            entries.append(
                ModelListEntry(
                    id_model=_strip_optional_quotes(parts[0]),
                    teff=float(parts[1]),
                    logg=float(parts[2]),
                    metal=float(parts[3]),
                    alpha=float(parts[4]),
                    mass=float(parts[5]),
                    vturb=float(parts[6]),
                    abundance=float(parts[7]),
                    pointer=pointer,
                    line_index=line_index,
                )
            )
    return entries


def _sanitize_filename(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r"[^\w.\-+]+", "_", s)
    s = s.strip("_") or "model"
    return s[:max_len]


def default_out_name(index: int, entry: ModelListEntry) -> str:
    """Default output basename: index + model id + abundance tag."""
    safe_id = _sanitize_filename(entry.id_model)
    return f"{index:06d}_{safe_id}_abu{entry.abundance:+.3f}.dat"


def filter_unique_abundance(entries: Iterable[ModelListEntry]) -> List[ModelListEntry]:
    """Keep first entry for each distinct abundance value (order preserved)."""
    seen: set[float] = set()
    out: List[ModelListEntry] = []
    for e in entries:
        key = round(e.abundance, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def export_all(
    grid_file: str,
    aux_file: str,
    out_dir: str,
    *,
    unique_abundance: bool = False,
    limit: Optional[int] = None,
    skip_existing: bool = False,
    per_record_progress: bool = False,
    name_fn=default_out_name,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    entries = parse_nlte_model_list(aux_file)
    if unique_abundance:
        entries = filter_unique_abundance(entries)
    if limit is not None:
        entries = entries[:limit]

    if not entries:
        print("No data lines in auxiliary file.", file=sys.stderr)
        return 1

    iterator: Iterable[ModelListEntry] = entries
    if tqdm is not None:
        iterator = tqdm(entries, desc="Models", unit="model")

    n_written = 0
    for i, entry in enumerate(iterator):
        basename = name_fn(i, entry)
        out_path = os.path.join(out_dir, basename)
        if skip_existing and os.path.isfile(out_path):
            continue

        ndep, nk, depart, tau, atmos_str = read_binary_grid(
            grid_file,
            pointer=entry.pointer,
            show_progress=per_record_progress,
            progress_desc=f"lvl {entry.id_model[:20]}",
        )
        # Sanity: aux id should match binary (warn only)
        if atmos_str.strip() != entry.id_model.strip():
            print(
                f"Warning: aux id {entry.id_model!r} != binary id {atmos_str!r} "
                f"(pointer {entry.pointer})",
                file=sys.stderr,
            )

        write_departures_for_ts(out_path, tau, depart, entry.abundance)
        n_written += 1

    print(f"Wrote {n_written} departure file(s) under {out_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export NLTE binary grid records to TurboSpectrum ASCII departure files."
    )
    p.add_argument("grid_file", help="Path to NLTE .bin grid")
    p.add_argument(
        "aux_file",
        help="NLTE model list (auxiliary file with id, params, abundance, byte pointer)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default="nlte_departures_ascii",
        help="Output directory (created if missing)",
    )
    p.add_argument(
        "--unique-abundance",
        action="store_true",
        help="Only export one model per distinct abundance (first occurrence kept)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many models (for testing)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file already exists",
    )
    p.add_argument(
        "--per-record-level-progress",
        action="store_true",
        help="Show tqdm per level inside each model (verbose; many bars)",
    )
    args = p.parse_args()

    return export_all(
        args.grid_file,
        args.aux_file,
        args.out_dir,
        unique_abundance=args.unique_abundance,
        limit=args.limit,
        skip_existing=args.skip_existing,
        per_record_progress=args.per_record_level_progress,
    )


if __name__ == "__main__":
    raise SystemExit(main())
