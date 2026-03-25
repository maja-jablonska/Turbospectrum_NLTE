from __future__ import annotations

import dataclasses
import functools
import hashlib
import os
import re
import shlex
import shutil
from typing import Any, Dict, Mapping, Sequence

import numpy as np

NLTE_ASCII_CONTROL_KEYS = (
    "nlte_ascii_departure_dir",
    "nlte_ascii_departure_species",
    "nlte_ascii_abundance_column",
    "nlte_ascii_abundance_scale",
    "nlte_ascii_solar_abundance",
    "nlte_ascii_match",
)

_ELEMENT_SYMBOLS = (
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
)

PERIODIC_TABLE = {symbol.lower(): atomic_number for atomic_number, symbol in enumerate(_ELEMENT_SYMBOLS) if symbol}
ATOMIC_SYMBOL_BY_NUMBER = {atomic_number: symbol for atomic_number, symbol in enumerate(_ELEMENT_SYMBOLS) if symbol}

# Default solar abundances on the usual log epsilon(H)=12 scale.
# These defaults are only used to convert relative [X/Fe] values into the
# absolute abundance token encoded in NLTE ASCII departure filenames.
DEFAULT_SOLAR_ABUNDANCE = {
    "h": 12.00,
    "he": 10.93,
    "li": 1.05,
    "be": 1.38,
    "b": 2.70,
    "c": 8.46,
    "n": 7.83,
    "o": 8.69,
    "f": 4.40,
    "ne": 7.93,
    "na": 6.22,
    "mg": 7.55,
    "al": 6.43,
    "si": 7.51,
    "p": 5.41,
    "s": 7.12,
    "cl": 5.31,
    "ar": 6.38,
    "k": 5.07,
    "ca": 6.30,
    "sc": 3.14,
    "ti": 4.97,
    "v": 3.90,
    "cr": 5.62,
    "mn": 5.42,
    "fe": 7.50,
    "co": 4.94,
    "ni": 6.20,
    "cu": 4.18,
    "zn": 4.56,
    "sr": 2.83,
    "y": 2.21,
    "zr": 2.59,
    "ba": 2.25,
    "eu": 0.52,
}

_FILENAME_RE = re.compile(
    r"^(?:(?P<prefix>\d+)_)?(?P<stem>.+?)_abu(?P<abundance>[+-]?\d+(?:\.\d+)?)\.(?P<ext>[^.]+)$",
    flags=re.IGNORECASE,
)
_RELATIVE_FLOAT_TOL = 5e-4


def _normalize_key(raw_value: Any) -> str:
    return "".join(ch for ch in str(raw_value if raw_value is not None else "").strip().lower() if ch.isalnum())


def _as_abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))


def _coerce_float(raw_value: Any, *, label: str) -> float:
    text = str(raw_value if raw_value is not None else "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    try:
        return float(text)
    except Exception as exc:
        raise ValueError(f"{label} must be numeric, got {raw_value!r}") from exc


def _find_row_value(row_values: Mapping[str, Any], requested_key: str) -> Any:
    if requested_key in row_values:
        return row_values[requested_key]

    normalized_requested = _normalize_key(requested_key)
    for key, value in row_values.items():
        if _normalize_key(key) == normalized_requested:
            return value
    raise KeyError(requested_key)


def _canonical_species_symbol(raw_value: Any) -> str:
    text = str(raw_value if raw_value is not None else "").strip()
    if not text:
        return "Fe"

    if text.isdigit():
        atomic_number = int(text)
        if atomic_number in ATOMIC_SYMBOL_BY_NUMBER:
            return ATOMIC_SYMBOL_BY_NUMBER[atomic_number]
        raise ValueError(f"Unsupported atomic number for NLTE ASCII departures: {text}")

    normalized = _normalize_key(text)
    if normalized not in PERIODIC_TABLE:
        raise ValueError(f"Unsupported NLTE ASCII departure species: {raw_value!r}")
    symbol = ATOMIC_SYMBOL_BY_NUMBER[PERIODIC_TABLE[normalized]]
    return symbol


@dataclasses.dataclass(frozen=True)
class NlteAsciiSelector:
    directory: str
    species: str
    abundance_column: str
    abundance_scale: str
    solar_abundance: float | None
    match: str


@dataclasses.dataclass(frozen=True)
class DepartureCandidate:
    model_stem: str
    abundance: float
    path: str


@dataclasses.dataclass(frozen=True)
class NlteInfoEntry:
    atomic_number: int
    element: str
    nlte_flag: str
    model_atom_file: str
    departure_file: str
    bin_flag: str


def normalize_nlte_ascii_selector(
    *,
    directory: Any,
    species: Any = None,
    abundance_column: Any = None,
    abundance_scale: Any = None,
    solar_abundance: Any = None,
    match: Any = None,
) -> NlteAsciiSelector | None:
    raw_directory = str(directory if directory is not None else "").strip()
    if not raw_directory:
        return None

    directory_path = _as_abspath(raw_directory)
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"NLTE ASCII departure directory not found: {directory_path}")

    species_symbol = _canonical_species_symbol(species)
    abundance_column_text = str(abundance_column if abundance_column is not None else "").strip() or "auto"
    abundance_scale_text = str(abundance_scale if abundance_scale is not None else "relative").strip().lower() or "relative"
    if abundance_scale_text not in {"relative", "absolute"}:
        raise ValueError(
            f"nlte_ascii abundance_scale must be 'relative' or 'absolute', got {abundance_scale!r}"
        )

    match_text = str(match if match is not None else "nearest").strip().lower() or "nearest"
    if match_text not in {"nearest", "exact"}:
        raise ValueError(f"nlte_ascii match must be 'nearest' or 'exact', got {match!r}")

    solar_value: float | None
    if solar_abundance in (None, "", "auto"):
        solar_value = None
    else:
        solar_value = _coerce_float(solar_abundance, label="nlte_ascii solar_abundance")

    if abundance_scale_text == "relative" and solar_value is None:
        species_key = _normalize_key(species_symbol)
        if species_key not in DEFAULT_SOLAR_ABUNDANCE:
            raise ValueError(
                f"No default solar abundance is available for species {species_symbol}; "
                "set --nlte-ascii-solar-abundance explicitly."
            )

    return NlteAsciiSelector(
        directory=directory_path,
        species=species_symbol,
        abundance_column=abundance_column_text,
        abundance_scale=abundance_scale_text,
        solar_abundance=solar_value,
        match=match_text,
    )


def build_nlte_ascii_selector_columns(
    row_count: int,
    selector: NlteAsciiSelector | None,
) -> Dict[str, np.ndarray]:
    if selector is None:
        return {}

    columns: Dict[str, np.ndarray] = {
        "nlte_ascii_departure_dir": np.full(row_count, selector.directory, dtype=object),
        "nlte_ascii_departure_species": np.full(row_count, selector.species, dtype=object),
        "nlte_ascii_abundance_column": np.full(row_count, selector.abundance_column, dtype=object),
        "nlte_ascii_abundance_scale": np.full(row_count, selector.abundance_scale, dtype=object),
        "nlte_ascii_match": np.full(row_count, selector.match, dtype=object),
    }
    if selector.solar_abundance is not None:
        columns["nlte_ascii_solar_abundance"] = np.full(
            row_count,
            selector.solar_abundance,
            dtype=np.float64,
        )
    return columns


def selector_from_row(row_values: Mapping[str, Any]) -> NlteAsciiSelector | None:
    return normalize_nlte_ascii_selector(
        directory=row_values.get("nlte_ascii_departure_dir"),
        species=row_values.get("nlte_ascii_departure_species"),
        abundance_column=row_values.get("nlte_ascii_abundance_column"),
        abundance_scale=row_values.get("nlte_ascii_abundance_scale"),
        solar_abundance=row_values.get("nlte_ascii_solar_abundance"),
        match=row_values.get("nlte_ascii_match"),
    )


def resolve_absolute_abundance(
    row_values: Mapping[str, Any],
    selector: NlteAsciiSelector,
) -> tuple[float, str]:
    abundance_column = selector.abundance_column
    if _normalize_key(abundance_column) in {"", "auto"}:
        abundance_column = "feh" if _normalize_key(selector.species) == "fe" else _normalize_key(selector.species)

    if selector.abundance_scale == "absolute":
        raw_value = _find_row_value(row_values, abundance_column)
        return _coerce_float(raw_value, label=f"{abundance_column} absolute abundance"), abundance_column

    solar_abundance = selector.solar_abundance
    if solar_abundance is None:
        solar_abundance = DEFAULT_SOLAR_ABUNDANCE[_normalize_key(selector.species)]

    feh = _coerce_float(_find_row_value(row_values, "feh"), label="feh")
    if _normalize_key(abundance_column) == "feh":
        return solar_abundance + feh, abundance_column

    relative_value = _coerce_float(
        _find_row_value(row_values, abundance_column),
        label=f"{abundance_column} relative abundance",
    )
    return solar_abundance + feh + relative_value, abundance_column


@functools.lru_cache(maxsize=16)
def _build_departure_index(directory: str) -> dict[str, tuple[DepartureCandidate, ...]]:
    index: dict[str, list[DepartureCandidate]] = {}
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if not os.path.isfile(path):
            continue
        match = _FILENAME_RE.match(entry)
        if not match:
            continue
        model_stem = match.group("stem")
        abundance = float(match.group("abundance"))
        index.setdefault(model_stem, []).append(
            DepartureCandidate(model_stem=model_stem, abundance=abundance, path=path)
        )

    return {
        model_stem: tuple(sorted(candidates, key=lambda item: (item.abundance, item.path)))
        for model_stem, candidates in index.items()
    }


def select_departure_file(
    *,
    directory: str,
    model_stem: str,
    abundance: float,
    match: str,
) -> DepartureCandidate:
    index = _build_departure_index(directory)
    candidates = index.get(model_stem)
    if not candidates:
        available = ", ".join(sorted(index.keys())[:3])
        raise FileNotFoundError(
            f"No NLTE ASCII departure files were found for model stem {model_stem!r} in {directory}. "
            f"Example stems present: {available or '<none>'}"
        )

    ranked = sorted(candidates, key=lambda item: (abs(item.abundance - abundance), item.abundance, item.path))
    best = ranked[0]
    if match == "exact" and abs(best.abundance - abundance) > _RELATIVE_FLOAT_TOL:
        raise FileNotFoundError(
            f"No exact NLTE ASCII departure file found for model stem {model_stem!r} at abundance {abundance:+0.3f} "
            f"in {directory}"
        )
    return best


def _ensure_trailing_sep(path: str) -> str:
    if not path:
        return ""
    return path if path.endswith(os.sep) else f"{path}{os.sep}"


def parse_nlte_info_file(path: str) -> tuple[str, str, list[NlteInfoEntry]]:
    resolved_path = _as_abspath(path)
    resolved_dir = os.path.dirname(resolved_path)

    def _resolve_info_path(raw_value: str) -> str:
        expanded = os.path.expanduser(os.path.expandvars(str(raw_value).strip()))
        if not expanded:
            return ""
        if os.path.isabs(expanded):
            return _ensure_trailing_sep(os.path.abspath(expanded))
        return _ensure_trailing_sep(os.path.abspath(os.path.join(resolved_dir, expanded)))

    with open(resolved_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    path_model = ""
    path_depart = ""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# path for model atom files") and idx + 1 < len(lines):
            path_model = _resolve_info_path(lines[idx + 1].strip())
        if stripped.startswith("# path for departure files") and idx + 1 < len(lines):
            path_depart = _resolve_info_path(lines[idx + 1].strip())

    entries: list[NlteInfoEntry] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = shlex.split(stripped, comments=False, posix=True)
        if len(tokens) < 6:
            continue
        try:
            atomic_number = int(tokens[0])
        except ValueError:
            continue
        entries.append(
            NlteInfoEntry(
                atomic_number=atomic_number,
                element=tokens[1],
                nlte_flag=tokens[2],
                model_atom_file=tokens[3],
                departure_file=tokens[4],
                bin_flag=tokens[5],
            )
        )

    if not path_model:
        raise ValueError(f"Could not find model-atom path header in NLTE info file: {resolved_path}")
    if not path_depart:
        raise ValueError(f"Could not find departure-file path header in NLTE info file: {resolved_path}")
    return path_model, path_depart, entries


def _symlink_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_nlte_info_with_departure_override(
    *,
    base_info_path: str,
    selector: NlteAsciiSelector,
    departure_file_path: str,
    output_root: str,
) -> str:
    resolved_base = _as_abspath(base_info_path)
    resolved_departure = _as_abspath(departure_file_path)
    digest = hashlib.sha256(
        f"{resolved_base}|{selector.species}|{resolved_departure}".encode("utf-8")
    ).hexdigest()[:16]
    target_root = os.path.join(_as_abspath(output_root), "nlte_ascii_info_cache", digest)
    info_path = os.path.join(target_root, "SPECIES_LTE_NLTE.dat")
    if os.path.isfile(info_path):
        return info_path

    model_path, departure_path, entries = parse_nlte_info_file(resolved_base)
    target_atomic_number = PERIODIC_TABLE[_normalize_key(selector.species)]
    staged_departure_dir = os.path.join(target_root, "departures")
    os.makedirs(staged_departure_dir, exist_ok=True)

    rewritten_entries: list[NlteInfoEntry] = []
    found_target_species = False
    for entry in entries:
        original_departure_name = str(entry.departure_file).strip()
        rewritten_departure_name = original_departure_name
        rewritten_bin_flag = entry.bin_flag
        source_path = ""

        if original_departure_name:
            if entry.atomic_number == target_atomic_number:
                found_target_species = True
                source_path = resolved_departure
                rewritten_departure_name = os.path.basename(resolved_departure)
                rewritten_bin_flag = "ascii"
            else:
                source_path = os.path.join(departure_path, original_departure_name)

            if source_path and os.path.isfile(source_path):
                _symlink_or_copy(source_path, os.path.join(staged_departure_dir, rewritten_departure_name))
            elif str(entry.nlte_flag).strip().lower().startswith("nlte") or entry.atomic_number == target_atomic_number:
                raise FileNotFoundError(
                    f"Referenced NLTE departure file does not exist: {source_path or original_departure_name}"
                )

        rewritten_entries.append(
            dataclasses.replace(
                entry,
                departure_file=rewritten_departure_name,
                bin_flag=rewritten_bin_flag,
            )
        )

    if not found_target_species:
        raise KeyError(
            f"Species {selector.species} (Z={target_atomic_number}) was not found in base NLTE info file {resolved_base}"
        )

    os.makedirs(target_root, exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as handle:
        handle.write("# Auto-generated by nlte_ascii_departures.py\n")
        handle.write("# path for model atom files     ! don't change this line !\n")
        handle.write(f"{_ensure_trailing_sep(model_path)}\n")
        handle.write("#\n")
        handle.write("# path for departure files      ! don't change this line !\n")
        handle.write(f"{_ensure_trailing_sep(staged_departure_dir)}\n")
        handle.write("#\n")
        handle.write("# atomic (N)LTE setup\n")
        for entry in rewritten_entries:
            handle.write(
                f"{entry.atomic_number}\t'{entry.element}'\t'{entry.nlte_flag}'\t"
                f"'{entry.model_atom_file}'\t'{entry.departure_file}' '{entry.bin_flag}'\n"
            )

    return info_path
