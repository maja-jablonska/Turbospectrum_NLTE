#!/usr/bin/env python3
"""Bootstrap an NLTE setup end to end: download the data, then generate a
matching ``SPECIES_LTE_NLTE.dat`` (the NLTE info file).

The species map that ships with the repo points at HPC paths, so it never works
out of the box on another machine. This script instead **downloads** the model
atoms + departure grids (via scripts/download_data.sh) and **writes a fresh
species map** whose two path headers point at your downloaded data, with one
entry per discovered species.

    # download everything and write DATA/SPECIES_LTE_NLTE.dat (backs up any existing one)
    python3 scripts/setup_nlte.py

    # restrict NLTE to specific elements (the rest are written as LTE)
    python3 scripts/setup_nlte.py --nlte Fe Mg Ca

    # use already-downloaded data; just (re)generate the species map
    python3 scripts/setup_nlte.py --no-download

    # see what would be written without downloading or writing
    python3 scripts/setup_nlte.py --no-download --dry-run

Entry pairing (atom <-> departure grid, atomic number, binary/ascii) is
heuristic from filenames; the script prints what it generated and runs a
preflight so you can review before synthesizing.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from nlte_config import preflight_nlte  # noqa: E402

# Atomic symbols 1..92 -> Z is index+1.
_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
]
_SYMBOL_TO_Z = {sym.lower(): idx + 1 for idx, sym in enumerate(_ELEMENTS)}
_CANONICAL = {sym.lower(): sym for sym in _ELEMENTS}


def _element_from_atom(filename: str) -> Optional[str]:
    """atom.fe607a -> 'Fe' (longest leading element symbol after the 'atom.' prefix)."""
    stem = filename[len("atom."):] if filename.lower().startswith("atom.") else filename
    leading = re.match(r"[A-Za-z]+", stem)
    if not leading:
        return None
    token = leading.group(0).lower()
    for length in (2, 1):  # prefer a 2-letter symbol (Fe) over a 1-letter one (F)
        cand = token[:length]
        if cand in _SYMBOL_TO_Z:
            return _CANONICAL[cand]
    return None


def _element_from_grid(filename: str) -> Optional[str]:
    """NLTEgrid_Fe_Sun.ascii -> 'Fe' (the field that is exactly an element symbol)."""
    for token in re.split(r"[._\-]+", filename):
        if token.lower() in _SYMBOL_TO_Z:
            return _CANONICAL[token.lower()]
    return None


def _scan(model_dir: str, dep_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map element -> atom filename, and element -> departure filename."""
    atoms: Dict[str, str] = {}
    if os.path.isdir(model_dir):
        for name in sorted(os.listdir(model_dir)):
            if name.startswith(".") or not name.lower().startswith("atom."):
                continue
            element = _element_from_atom(name)
            if element and element not in atoms:
                atoms[element] = name

    deps: Dict[str, str] = {}
    if os.path.isdir(dep_dir):
        for name in sorted(os.listdir(dep_dir)):
            if name.startswith(".") or name.lower().startswith("auxdata") or not name.lower().startswith("nltegrid"):
                continue
            element = _element_from_grid(name)
            if element and element not in deps:
                deps[element] = name
    return atoms, deps


def _pair_grids_with_aux(dep_dir: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Pair each binary NLTEgrid_* file with its auxData_* companion.

    Returns (pairs, unpaired_grids). Pairing is by the shared stem after dropping
    the ``NLTEgrid``/``auxData`` prefix and extension (e.g. NLTEgrid_Fe_Sun.bin
    <-> auxData_Fe_Sun.dat). ``.ascii`` grids are skipped — those are not the
    binary format the exporter reads.
    """
    if not os.path.isdir(dep_dir):
        return [], []
    names = [f for f in os.listdir(dep_dir) if not f.startswith(".")]

    def _key(name: str, prefix: str) -> str:
        stem = os.path.splitext(name)[0].lower()
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
        return stem.strip("._-")

    aux_by_key: Dict[str, str] = {}
    for name in names:
        if name.lower().startswith("auxdata"):
            aux_by_key.setdefault(_key(name, "auxdata"), name)

    grids = sorted(
        f for f in names
        if f.lower().startswith("nltegrid") and not f.lower().endswith(".ascii")
    )
    pairs: List[Tuple[str, str]] = []
    unpaired: List[str] = []
    for grid in grids:
        aux = aux_by_key.get(_key(grid, "nltegrid"))
        if aux:
            pairs.append((grid, aux))
        else:
            unpaired.append(grid)
    return pairs, unpaired


def _ascii_export(dep_dir: str, out_dir: str, *, dry_run: bool) -> int:
    """Export each binary grid to per-species ASCII departure files under out_dir/<El>.

    Per-species subdirectories matter: the ASCII matcher keys on model stem +
    abundance and does NOT filter by element, so mixing species in one directory
    would let it pick the wrong element's departures.
    """
    pairs, unpaired = _pair_grids_with_aux(dep_dir)
    if not pairs:
        print(f"No binary NLTEgrid/auxData pairs found in {dep_dir}.", file=sys.stderr)
        return 1

    print(f"\nASCII export: {len(pairs)} grid/aux pair(s) -> {out_dir}/<element>/")
    plan: List[Tuple[str, str, str]] = []  # grid, aux, element
    for grid, aux in pairs:
        element = _element_from_grid(grid) or "unknown"
        plan.append((grid, aux, element))
        print(f"  {grid}  +  {aux}   ->  {os.path.join(out_dir, element)}")
    for grid in unpaired:
        print(f"  (no auxData match for {grid}; skipped)")

    if dry_run:
        print("--dry-run: not exporting.")
        return 0

    try:
        from export_nlte_grid_ascii import export_all
    except ImportError as exc:  # pragma: no cover - depends on local deps
        print(f"Could not import the ASCII exporter: {exc}", file=sys.stderr)
        return 1

    rc = 0
    species_dirs: List[Tuple[str, str]] = []
    for grid, aux, element in plan:
        species_dir = os.path.join(out_dir, element)
        print(f"\nExporting {grid} ({element}) -> {species_dir}")
        rc |= export_all(os.path.join(dep_dir, grid), os.path.join(dep_dir, aux), species_dir)
        species_dirs.append((element, species_dir))

    print("\nASCII departures ready. Point your config at the per-species directory, e.g.:")
    for element, species_dir in species_dirs:
        print(f"  nlte_ascii_departures: {{ \"directory\": \"{species_dir}\", \"species\": \"{element}\" }}")
    return rc


def _build_species_map(
    model_dir: str,
    dep_dir: str,
    atoms: Dict[str, str],
    deps: Dict[str, str],
    nlte_only: Optional[List[str]],
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Return (file text, list of (Z, element, flag)) for elements that have an atom."""
    nlte_set = {e.lower() for e in nlte_only} if nlte_only else None
    rows: List[Tuple[int, str, str, str, str, str]] = []  # Z, el, flag, atom, dep, bin/ascii
    summary: List[Tuple[int, str, str]] = []
    for element in sorted(atoms, key=lambda e: _SYMBOL_TO_Z[e.lower()]):
        z = _SYMBOL_TO_Z[element.lower()]
        atom_file = atoms[element]
        dep_file = deps.get(element, "")
        has_dep = bool(dep_file)
        want_nlte = has_dep and (nlte_set is None or element.lower() in nlte_set)
        flag = "nlte" if want_nlte else "lte"
        fmt = "ascii" if dep_file.lower().endswith(".ascii") else "binary"
        rows.append((z, element, flag, atom_file, dep_file or "-", fmt))
        summary.append((z, element, flag))

    lines = [
        "# This file controls which species are treated in LTE/NLTE",
        "# Generated by scripts/setup_nlte.py — review the entries below.",
        "# each line: atomic number / name / (n)lte / model atom / departure file / binary|ascii",
        "#",
        "# path for model atom files     ! don't change this line !",
        model_dir,
        "#",
        "# path for departure files      ! don't change this line !",
        dep_dir,
        "#",
        "# atomic (N)LTE setup",
    ]
    for z, element, flag, atom_file, dep_file, fmt in rows:
        prefix = "" if flag == "nlte" else "#"  # comment out LTE rows (LTE is the default anyway)
        lines.append(f"{prefix}{z}\t'{element}'\t'{flag}'\t'{atom_file}'\t'{dep_file}'\t'{fmt}'")
    return "\n".join(lines) + "\n", summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nlte-data-dir", default=os.path.join(REPO_ROOT, "input_files", "nlte_data"),
                        help="Where the NLTE data lives / will be downloaded")
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "DATA", "SPECIES_LTE_NLTE.dat"),
                        help="Species-map (nlte_info_file) path to write")
    parser.add_argument("--nlte", nargs="*", default=None, metavar="EL",
                        help="Mark only these elements NLTE (rest LTE). Default: every species that has a departure grid.")
    parser.add_argument("--download", dest="download", action="store_true", default=True,
                        help="Download NLTE data first (default).")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="Skip downloading; use data already in --nlte-data-dir.")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if files exist.")
    parser.add_argument("--species-map", dest="species_map", action="store_true", default=True,
                        help="Write the binary-workflow species map (default).")
    parser.add_argument("--no-species-map", dest="species_map", action="store_false",
                        help="Skip writing SPECIES_LTE_NLTE.dat (e.g. when you only want the ASCII export).")
    parser.add_argument("--ascii-export", action="store_true",
                        help="Also export the binary grids to per-species ASCII departure files.")
    parser.add_argument("--ascii-out", default=os.path.join(REPO_ROOT, "DATA", "DEP", "nlte_departures_ascii"),
                        help="Output root for --ascii-export (per-element subdirs are created under it).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written; don't download or write.")
    args = parser.parse_args()

    model_dir = os.path.join(args.nlte_data_dir, "model_atoms")
    dep_dir = os.path.join(args.nlte_data_dir, "departure_grids")

    if args.download and not args.dry_run:
        downloader = os.path.join(SCRIPT_DIR, "download_data.sh")
        # Invoke via bash so it works even if the script lost its +x bit on checkout.
        cmd = ["bash", downloader, "--nlte-atoms", "all"]
        env = dict(os.environ)
        if args.force_download:
            env["FORCE_DOWNLOAD"] = "true"
        print(f"Downloading NLTE data -> {args.nlte_data_dir} ...")
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print("download_data.sh failed; aborting.", file=sys.stderr)
            return proc.returncode

    rc = 0

    # --- Binary workflow: write the species map -------------------------------
    if args.species_map:
        atoms, deps = _scan(model_dir, dep_dir)
        if not atoms:
            print(f"No model atoms (atom.*) found in {model_dir}.", file=sys.stderr)
            print("Run with --download (the default) or point --nlte-data-dir at your data.", file=sys.stderr)
            return 1

        text, summary = _build_species_map(model_dir, dep_dir, atoms, deps, args.nlte)
        print(f"\nDiscovered {len(atoms)} model atom(s), {len(deps)} departure grid(s).")
        print("Species map entries:")
        for z, element, flag in summary:
            mark = "NLTE" if flag == "nlte" else "lte "
            dep = deps.get(element, "(no departure grid — LTE only)")
            print(f"  {z:>3}  {element:<3} {mark}  {dep}")

        if args.dry_run:
            print("\n--dry-run: not writing. File that would be written:\n")
            print(text)
        else:
            output = os.path.abspath(args.output)
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            if os.path.isfile(output):
                backup = output + ".bak"
                shutil.copy2(output, backup)
                print(f"\nBacked up existing species map -> {backup}")
            with open(output, "w", encoding="utf-8") as handle:
                handle.write(text)
            print(f"Wrote species map: {output}")

            problems = preflight_nlte(project_root=REPO_ROOT, enabled=True, nlte_info_file=output)
            print("\nPreflight (binary):")
            if problems:
                for line in problems:
                    print(f"  {line}" if line else "")
                rc = 1
            else:
                print("  OK: NLTE inputs look runnable.")

    # --- ASCII workflow: export per-species departure files --------------------
    if args.ascii_export:
        rc |= _ascii_export(dep_dir, os.path.abspath(args.ascii_out), dry_run=args.dry_run)

    if not args.species_map and not args.ascii_export:
        print("Nothing to do: both --no-species-map and (no) --ascii-export.", file=sys.stderr)
        return 1

    if rc == 0 and not args.dry_run:
        print("\nNext:  python3 scripts/init_nlte_config.py --output configs/pipeline/config_nlte.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
