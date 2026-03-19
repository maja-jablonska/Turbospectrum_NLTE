#!/usr/bin/env python3
"""
Clean interpolated atmosphere models and their contopac products.

Definition used here:
- "Native" models are those listed in:
  input_files/model_atmospheres/1D/marcs_standard_comp/model_list
- Any *.mod file present in the models directory but NOT listed in model_list
  is assumed to be an interpolated/generated artifact and will be removed.
- For each removed model, matching contopac files are removed from COM/contopac:
  <model_base>opac*  (covers files like "...opac", "...opac.mod", etc.)

Default behavior is DRY RUN. Use --apply to actually delete.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set, List


def _repo_root_from_script() -> Path:
    # scripts/clean_interpolated_atmospheres.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def _read_model_list(model_list_path: Path) -> Set[str]:
    keep: Set[str] = set()
    with model_list_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            keep.add(Path(line).name)
    return keep


def _gather_model_files(models_dir: Path) -> List[Path]:
    return sorted([p for p in models_dir.glob("*.mod") if p.is_file()])


def _gather_contopac_files_for_model(contopac_dir: Path, model_file: Path) -> List[Path]:
    base = model_file.name.removesuffix(".mod")
    pattern = f"{base}opac*"
    return sorted([p for p in contopac_dir.glob(pattern) if p.is_file()])


def _ensure_under_root(path: Path, root: Path) -> None:
    rp = path.resolve()
    rr = root.resolve()
    if rr not in rp.parents and rp != rr:
        raise RuntimeError(f"Refusing to operate outside repo root. Path={rp} Root={rr}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove interpolated atmosphere *.mod files and matching COM/contopac outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo root (default: auto-detected from this script location).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing model atmospheres (*.mod).",
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="File listing native models to keep (one filename per line).",
    )
    parser.add_argument(
        "--contopac-dir",
        type=Path,
        default=None,
        help="Directory containing contopac outputs to remove for deleted models.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, performs a dry run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only a short summary.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.root or _repo_root_from_script()).resolve()
    models_dir = (args.models_dir or (root / "input_files/model_atmospheres/1D/marcs_standard_comp")).resolve()
    model_list_path = (args.model_list or (models_dir / "model_list")).resolve()
    contopac_dir = (args.contopac_dir or (root / "COM/contopac")).resolve()

    # Safety rails: ensure targets are inside the repo
    _ensure_under_root(models_dir, root)
    _ensure_under_root(model_list_path, root)
    _ensure_under_root(contopac_dir, root)

    if not model_list_path.exists():
        raise FileNotFoundError(f"model_list not found: {model_list_path}")

    keep = _read_model_list(model_list_path)
    model_files = _gather_model_files(models_dir)

    to_delete_models = [p for p in model_files if p.name not in keep]

    to_delete_contopac: List[Path] = []
    for m in to_delete_models:
        to_delete_contopac.extend(_gather_contopac_files_for_model(contopac_dir, m))

    # De-dup while preserving stable order
    seen = set()
    to_delete_contopac = [p for p in to_delete_contopac if not (p in seen or seen.add(p))]

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] Repo root:      {root}")
    print(f"[{mode}] Models dir:     {models_dir}")
    print(f"[{mode}] Model list:     {model_list_path} (keep={len(keep)})")
    print(f"[{mode}] Contopac dir:   {contopac_dir}")
    print()

    print(f"Models found: {len(model_files)}")
    print(f"Models to delete (not in model_list): {len(to_delete_models)}")
    print(f"Contopac files to delete: {len(to_delete_contopac)}")
    print()

    if not args.quiet and to_delete_models:
        print("=== Models to delete ===")
        for p in to_delete_models:
            print(str(p))
        print()

    if not args.quiet and to_delete_contopac:
        print("=== Contopac files to delete ===")
        for p in to_delete_contopac:
            print(str(p))
        print()

    if not args.apply:
        return 0

    deleted_models = 0
    deleted_contopac = 0

    for p in to_delete_models:
        try:
            p.unlink()
            deleted_models += 1
        except FileNotFoundError:
            pass

    for p in to_delete_contopac:
        try:
            p.unlink()
            deleted_contopac += 1
        except FileNotFoundError:
            pass

    print(f"Deleted models: {deleted_models}")
    print(f"Deleted contopac files: {deleted_contopac}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

