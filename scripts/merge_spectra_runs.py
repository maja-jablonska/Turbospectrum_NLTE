#!/usr/bin/env python3
"""Merge final per-run spectra Zarr stores into one consolidated store.

This is a *cross-run* combiner. Its inputs are the schema-compliant merged
outputs produced by ``scripts/merge_spectra_shards.py`` (one per run), i.e.
stores that already contain:

- ``flux``            (float32) [N_models, N_lambda]
- ``wavelength``      (float32) [N_lambda]
- ``params``          (float32) [N_models, N_params]
- ``param_names``     (U32)     [N_params]
- ``model_id``        (uint64)  [N_models]
- ``physics_hash`` / ``schema_version`` scalars
- optional ``continuum`` (float32) [N_models, N_lambda]

Unlike merging shards (which stitches disjoint row *ranges* of a single grid),
this concatenates independent runs along the model axis and writes a new
schema-compliant store matching ``DATA_SCHEMA.md``.

Design decisions (see ``--help`` for overrides):

- **Wavelength must be identical** across all runs. ``DATA_SCHEMA.md`` mandates a
  single wavelength grid per dataset ("If grids diverge in wavelength -> create a
  new dataset, NOT a new axis"). Divergent grids abort the merge.
- **param_names are unioned** in first-seen order. A run missing a column stores
  ``NaN`` for it. Use ``--strict-params`` to instead require every run to expose
  the exact same param columns.
- **continuum** is written whenever *any* input has it; runs without it store
  ``NaN``.
- **model_id is recomputed** from the aligned params matrix so ids stay
  deterministic and comparable across the combined store (the per-run ids were
  hashed against per-run column orderings).
- **Per-row lineage** is preserved: ``run_index`` (uint16) and
  ``source_physics_hash`` (str) arrays record which input each row came from; the
  run_index -> path/physics_hash mapping is stored in attrs.
- Optional ``--dedup-by-model-id`` drops rows whose recomputed model_id was
  already seen (keeps the first occurrence, in input order).
- Optional ``--filter-bad-snaps`` (with ``--atmosphere-path``) drops rows whose
  recorded ``marcs_*`` atmosphere is not the correct nearest snap for the
  requested parameters, using the same classifier as
  ``scripts/filter_bad_snap_rows.py``. Applied before dedup; per-run drop
  counts land in the manifest and attrs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import zarr

# Reuse the shard-merger's building blocks so the two mergers stay in lock-step
# on hashing, string handling and provenance conventions.
try:
    from merge_spectra_shards import (
        PROVENANCE_FILENAMES,
        _compute_model_ids,
        _git_commit,
        _read_string_scalar_optional,
        _to_u32_param_names,
    )
    from provenance_contract import (
        assert_required_provenance_fields,
        is_meaningful_provenance_value,
    )
    from spectrum_output import infer_flux_metadata
    from zarr_compat import (
        zarr_store as _zarr_store,
        compression_kwargs as _zarr_compression_kwargs,
        create_root_group as _create_root_group,
        create_array as _zc_create_array,
        write_string_scalar as _zc_write_string_scalar,
        write_fixed_string_scalar as _zc_write_fixed_string_scalar,
        write_string_array as _zc_write_string_array,
    )
except ImportError:  # pragma: no cover - package-relative fallback
    from .merge_spectra_shards import (  # type: ignore
        PROVENANCE_FILENAMES,
        _compute_model_ids,
        _git_commit,
        _read_string_scalar_optional,
        _to_u32_param_names,
    )
    from .provenance_contract import (  # type: ignore
        assert_required_provenance_fields,
        is_meaningful_provenance_value,
    )
    from .spectrum_output import infer_flux_metadata  # type: ignore
    from .zarr_compat import (  # type: ignore
        zarr_store as _zarr_store,
        compression_kwargs as _zarr_compression_kwargs,
        create_root_group as _create_root_group,
        create_array as _zc_create_array,
        write_string_scalar as _zc_write_string_scalar,
        write_fixed_string_scalar as _zc_write_fixed_string_scalar,
        write_string_array as _zc_write_string_array,
    )


def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _list_runs(run_paths: Sequence[str] | None, run_dir: str | None) -> List[str]:
    if run_paths:
        paths = [_norm(p) for p in run_paths]
    elif run_dir:
        d = _norm(run_dir)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Run directory does not exist: {d}")
        paths = sorted(str(p) for p in Path(d).glob("*.zarr"))
        if not paths:
            raise FileNotFoundError(f"No *.zarr stores found in {d}")
    else:
        raise ValueError("Provide either --run (repeatable) or --run-dir")

    if len(paths) < 1:
        raise ValueError("At least one input run is required")
    # Preserve order but drop accidental duplicates.
    seen: set[str] = set()
    unique: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _open_run(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run store does not exist: {path}")
    return zarr.open_group(store=_zarr_store(path), mode="r")


def _require_arrays(root, names: Sequence[str], path: str) -> None:
    missing = [n for n in names if n not in root]
    if missing:
        raise KeyError(f"Run {path} is missing required arrays: {missing}")


def _read_param_names(root) -> List[str]:
    raw = np.asarray(root["param_names"][:]).tolist()
    names: List[str] = []
    for v in raw:
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="ignore")
        names.append(str(v).strip())
    return names


def _collect_attrs(root) -> Dict[str, Any]:
    try:
        return {str(k): root.attrs[k] for k in root.attrs.keys()}
    except Exception:
        try:
            return {str(k): v for k, v in dict(root.attrs).items()}
        except Exception:
            return {}


class RunInfo:
    """Per-run metadata gathered in the first (cheap) pass."""

    __slots__ = (
        "path",
        "n_rows",
        "n_lambda",
        "param_names",
        "has_continuum",
        "physics_hash",
        "attrs",
    )

    def __init__(self, path: str, root) -> None:
        self.path = path
        _require_arrays(root, ["flux", "wavelength", "params", "param_names"], path)
        flux = root["flux"]
        params = root["params"]
        if flux.ndim != 2:
            raise ValueError(f"Run {path} flux must be 2D, got shape {flux.shape}")
        self.n_rows = int(flux.shape[0])
        self.n_lambda = int(flux.shape[1])
        self.param_names = _read_param_names(root)
        if params.shape != (self.n_rows, len(self.param_names)):
            raise ValueError(
                f"Run {path} params shape {tuple(params.shape)} inconsistent with "
                f"flux rows={self.n_rows} and param_names={len(self.param_names)}"
            )
        self.has_continuum = "continuum" in root
        self.physics_hash = _read_string_scalar_optional(root, "physics_hash") or ""
        self.attrs = _collect_attrs(root)


def _union_param_names(runs: Sequence[RunInfo], *, strict: bool) -> List[str]:
    if strict:
        reference = runs[0].param_names
        for r in runs[1:]:
            if r.param_names != reference:
                raise ValueError(
                    "--strict-params: param_names differ between runs.\n"
                    f"  {runs[0].path}: {reference}\n"
                    f"  {r.path}: {r.param_names}"
                )
        return list(reference)

    union: List[str] = []
    seen: set[str] = set()
    for r in runs:
        for name in r.param_names:
            if name not in seen:
                seen.add(name)
                union.append(name)
    return union


def _validate_wavelengths(runs: Sequence[RunInfo], *, atol: float) -> np.ndarray:
    ref_root = _open_run(runs[0].path)
    reference = np.asarray(ref_root["wavelength"][:], dtype=np.float64)
    if reference.size == 0:
        raise ValueError(f"Run {runs[0].path} has an empty wavelength array")
    for r in runs[1:]:
        root = _open_run(r.path)
        wl = np.asarray(root["wavelength"][:], dtype=np.float64)
        if wl.shape != reference.shape or not np.allclose(wl, reference, atol=atol):
            raise ValueError(
                "Wavelength grids diverge between runs -- DATA_SCHEMA.md requires a "
                "single wavelength grid per dataset. Build separate datasets or "
                "resample first.\n"
                f"  reference {runs[0].path}: n={reference.size}, "
                f"{reference[0]:.4f}..{reference[-1]:.4f}\n"
                f"  offending {r.path}: n={wl.size}, {wl[0]:.4f}..{wl[-1]:.4f}"
            )
    return reference.astype(np.float32, copy=False)


def _aligned_params(root, run_names: Sequence[str], union_names: Sequence[str]) -> np.ndarray:
    """Return this run's params re-projected onto the union column order.

    Columns the run does not have are filled with NaN.
    """
    src = np.asarray(root["params"][:], dtype=np.float32)
    n_rows = src.shape[0]
    out = np.full((n_rows, len(union_names)), np.nan, dtype=np.float32)
    src_index = {name: i for i, name in enumerate(run_names)}
    for j, name in enumerate(union_names):
        i = src_index.get(name)
        if i is not None:
            out[:, j] = src[:, i]
    return out


def _compute_keep_masks(
    runs: Sequence[RunInfo],
    union_names: Sequence[str],
    *,
    dedup: bool,
    pre_masks: Sequence[np.ndarray] | None = None,
) -> Tuple[List[np.ndarray], int]:
    """Decide which rows to keep per run.

    ``pre_masks`` (e.g. the bad-snap filter) excludes rows up front; excluded
    rows never claim a model_id, so a filtered row cannot shadow a kept
    duplicate. Without dedup the pre-mask (or everything) is kept. With dedup,
    rows whose recomputed model_id was already emitted by an earlier run (or
    earlier in the same run) are dropped, keeping the first occurrence.
    """
    masks: List[np.ndarray] = []
    total = 0
    seen: set[int] = set()
    for idx, r in enumerate(runs):
        base = (
            np.asarray(pre_masks[idx], dtype=bool)
            if pre_masks is not None
            else np.ones(r.n_rows, dtype=bool)
        )
        if not dedup:
            mask = base.copy()
        else:
            root = _open_run(r.path)
            params = _aligned_params(root, r.param_names, union_names)
            ids = _compute_model_ids(params)
            mask = np.zeros(r.n_rows, dtype=bool)
            for i, mid in enumerate(ids.tolist()):
                if base[i] and mid not in seen:
                    seen.add(mid)
                    mask[i] = True
        masks.append(mask)
        total += int(mask.sum())
    return masks, total


def _first_meaningful_attr(runs: Sequence[RunInfo], keys: Sequence[str]) -> str:
    for r in runs:
        for key in keys:
            if key in r.attrs:
                val = str(r.attrs[key]).strip()
                if val and val.lower() not in ("", "unknown", "none"):
                    return val
    return ""


def _build_provenance_group(
    root_out,
    runs: Sequence[RunInfo],
    *,
    run_masks: Sequence[np.ndarray],
    compression_kwargs: Mapping[str, Any],
    snap_stats: Sequence[Mapping[str, Any] | None] | None = None,
) -> None:
    """Write a provenance group carrying per-run manifests plus contract files.

    Combining physics-divergent runs has no single canonical config, so the
    manifest records each source store verbatim (path, physics_hash, kept rows,
    attrs) and the carry-forward files hold first-meaningful values so downstream
    validators still find the required keys.
    """
    prov = (
        root_out.create_group("provenance")
        if hasattr(root_out, "create_group")
        else root_out.require_group("provenance")
    )

    per_run = []
    for i, (r, mask) in enumerate(zip(runs, run_masks)):
        entry = {
            "path": r.path,
            "physics_hash": r.physics_hash,
            "n_rows_input": int(r.n_rows),
            "n_rows_kept": int(mask.sum()),
            "param_names": list(r.param_names),
            "has_continuum": bool(r.has_continuum),
            "attrs": {k: str(v) for k, v in r.attrs.items()},
        }
        if snap_stats is not None and snap_stats[i] is not None:
            entry["snap_filter"] = dict(snap_stats[i])
        per_run.append(entry)
    manifest = {
        "merge_tool": "scripts/merge_spectra_runs.py",
        "n_runs": len(runs),
        "runs": per_run,
    }
    _zc_write_string_scalar(
        prov, "merge_runs_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
        compression_kw=compression_kwargs,
    )

    # Carry-forward the standard provenance filenames so validate_dataset finds
    # them. Prefer verbatim source provenance when a single run supplies it;
    # otherwise emit a placeholder that records the cross-run nature.
    for name in PROVENANCE_FILENAMES:
        carried = None
        for r in runs:
            src_root = _open_run(r.path)
            if "provenance" in src_root:
                value = _read_string_scalar_optional(src_root["provenance"], name)
                if value:
                    carried = value
                    break
        if carried is None:
            carried = json.dumps(
                {
                    "note": f"{name} combined across {len(runs)} runs by merge_spectra_runs.py",
                    "source_runs": [r.path for r in runs],
                },
                indent=2,
                sort_keys=True,
            )
        _zc_write_string_scalar(prov, name, carried, compression_kw=compression_kwargs)


def _copy_block(
    dst,
    src,
    dst_start: int,
    mask: np.ndarray,
    *,
    chunk_rows: int,
) -> None:
    """Copy the masked source rows into dst starting at dst_start.

    Streams in row blocks either way, so a large run's flux array is never
    materialised in memory.
    """
    n = src.shape[0]
    cursor = dst_start
    for a in range(0, n, chunk_rows):
        b = min(a + chunk_rows, n)
        block_mask = mask[a:b]
        kept = int(block_mask.sum())
        if not kept:
            continue
        block = np.asarray(src[a:b], dtype=np.float32)
        if kept != b - a:
            block = block[block_mask]
        dst[cursor : cursor + kept] = block
        cursor += kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-zarr", required=True, help="Consolidated output Zarr path")
    parser.add_argument("--run", action="append", default=None, help="Input run spectra.zarr (repeatable)")
    parser.add_argument("--run-dir", default=None, help="Directory containing input *.zarr run stores")
    parser.add_argument("--chunk-rows", type=int, default=128, help="Row chunk of output arrays (matches merge default)")
    parser.add_argument("--compressor", default=None, help="JSON compressor options (cname, clevel, shuffle)")
    parser.add_argument("--schema-version", default="1.0.0", help="DATA_SCHEMA.md schema version")
    parser.add_argument("--generator", default="turbospectrum_nlte.merge_runs", help="Generator metadata string")
    parser.add_argument("--contact", default=os.environ.get("SPICE_CONTACT", "unknown"), help="Contact metadata")
    parser.add_argument("--flux-definition", default="auto", help="Flux definition metadata ('auto' infers from inputs)")
    parser.add_argument("--wavelength-atol", type=float, default=1e-2, help="Tolerance for wavelength-grid equality")
    parser.add_argument(
        "--strict-params",
        action="store_true",
        help="Require every run to expose identical param_names (otherwise columns are unioned, missing filled NaN).",
    )
    parser.add_argument(
        "--dedup-by-model-id",
        action="store_true",
        help="Drop rows whose recomputed model_id was already emitted (keep first occurrence in input order).",
    )
    parser.add_argument(
        "--allow-incomplete-provenance",
        action="store_true",
        help="Write output even when required provenance contract fields cannot be assembled.",
    )
    parser.add_argument(
        "--filter-bad-snaps",
        action="store_true",
        help="Drop rows whose recorded marcs_* atmosphere is not the correct nearest snap "
             "(see scripts/filter_bad_snap_rows.py). Requires --atmosphere-path.",
    )
    parser.add_argument(
        "--atmosphere-path",
        default=None,
        help="MARCS model directory the runs were synthesised against (for --filter-bad-snaps)",
    )
    parser.add_argument("--max-dteff", type=float, default=None, help="With --filter-bad-snaps: also drop rows with |marcs_teff - teff| above this (K)")
    parser.add_argument("--max-dlogg", type=float, default=None, help="With --filter-bad-snaps: also drop rows with |marcs_logg - logg| above this (dex)")
    parser.add_argument("--max-dfeh", type=float, default=None, help="With --filter-bad-snaps: also drop rows with |marcs_fe_h - feh| above this (dex)")
    parser.add_argument(
        "--rejects-csv",
        default=None,
        help="With --filter-bad-snaps: write all rejected rows (across runs) to this CSV, "
             "plus a *_summary.json of the rejected parameter space next to it.",
    )
    args = parser.parse_args()

    if args.filter_bad_snaps and not args.atmosphere_path:
        parser.error("--filter-bad-snaps requires --atmosphere-path")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_paths = _list_runs(args.run, args.run_dir)
    print(f"Merging {len(run_paths)} run(s):")
    for p in run_paths:
        print(f"  - {p}")

    compressor_cfg: Dict[str, Any] = json.loads(args.compressor) if args.compressor else {}
    compression_kwargs = _zarr_compression_kwargs(compressor_cfg)

    # ---- Pass 1: gather metadata ------------------------------------------
    runs: List[RunInfo] = []
    for p in run_paths:
        runs.append(RunInfo(p, _open_run(p)))

    wavelengths = _validate_wavelengths(runs, atol=args.wavelength_atol)
    wl_count = int(wavelengths.size)
    for r in runs:
        if r.n_lambda != wl_count:
            raise ValueError(
                f"Run {r.path} has N_lambda={r.n_lambda}, expected {wl_count}"
            )

    union_names = _union_param_names(runs, strict=args.strict_params)
    if not union_names:
        raise ValueError("Could not determine any param columns across the inputs")
    if not args.strict_params:
        divergent = [r.path for r in runs if r.param_names != union_names]
        if divergent:
            print(
                f"NOTE: param_names differ across runs; unioned to {len(union_names)} "
                f"columns (missing columns filled with NaN). Union order: {union_names}",
                file=sys.stderr,
            )

    any_continuum = any(r.has_continuum for r in runs)

    # ---- Decide kept rows (snap filter, then dedup) -----------------------
    snap_masks: List[np.ndarray] | None = None
    snap_stats: List[Dict[str, Any] | None] = [None] * len(runs)
    if args.filter_bad_snaps:
        try:
            from filter_bad_snap_rows import classify_rows, collect_rejects, write_rejects
        except ImportError:  # pragma: no cover - package-relative fallback
            from .filter_bad_snap_rows import classify_rows, collect_rejects, write_rejects  # type: ignore
        atmosphere_path = _norm(args.atmosphere_path)
        snap_masks = []
        all_rejects: List[Dict[str, Any]] = []
        for i, r in enumerate(runs):
            src_root = _open_run(r.path)
            keep, stats, details = classify_rows(
                src_root, atmosphere_path,
                max_dteff=args.max_dteff, max_dlogg=args.max_dlogg, max_dfeh=args.max_dfeh,
            )
            snap_masks.append(keep)
            snap_stats[i] = stats
            if args.rejects_csv:
                all_rejects.extend(collect_rejects(src_root, r.path, keep, details))
            floor_note = ""
            if stats.get("n_below_pool_logg_floor"):
                floor_note = (
                    f"; {stats['n_below_pool_logg_floor']} rows below the pool logg floor "
                    f"{stats['pool_logg_floor']:g} are NOT flaggable as wrong-snap -- "
                    f"see the WARNING above"
                )
            print(
                f"Snap filter {r.path}: kept {stats['n_kept']}/{stats['n_rows']} "
                f"(wrong snap {stats['n_wrong_snap']}, out of tolerance "
                f"{stats['n_out_of_tolerance']}, NaN {stats['n_nan']}{floor_note})"
            )
        if args.rejects_csv:
            rejects_path = _norm(args.rejects_csv)
            summary_path = write_rejects(rejects_path, all_rejects, sum(r.n_rows for r in runs))
            print(f"Rejects written to {rejects_path} (summary: {summary_path})")

    run_masks, out_row_count = _compute_keep_masks(
        runs, union_names, dedup=bool(args.dedup_by_model_id), pre_masks=snap_masks
    )
    if out_row_count <= 0:
        raise ValueError("No rows remain to write (empty inputs, everything filtered, or everything deduplicated)")
    total_input_rows = sum(r.n_rows for r in runs)
    if args.filter_bad_snaps or args.dedup_by_model_id:
        print(f"Keeping {out_row_count}/{total_input_rows} rows after filtering/dedup")

    out_path = _norm(args.output_zarr)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ---- Create output store ----------------------------------------------
    store = _zarr_store(out_path)
    root_out = _create_root_group(store, overwrite=True)
    chunk_rows = max(1, min(int(args.chunk_rows), out_row_count))
    chunk_shape = (chunk_rows, wl_count)

    _zc_create_array(root_out, "wavelength", data=wavelengths, chunks=wavelengths.shape, **compression_kwargs)
    flux_out = _zc_create_array(root_out, "flux", shape=(out_row_count, wl_count), dtype=np.float32, chunks=chunk_shape, **compression_kwargs)
    continuum_out = None
    if any_continuum:
        continuum_out = _zc_create_array(root_out, "continuum", shape=(out_row_count, wl_count), dtype=np.float32, chunks=chunk_shape, **compression_kwargs)
        continuum_out[:] = np.nan
    params_out = _zc_create_array(root_out, "params", shape=(out_row_count, len(union_names)), dtype=np.float32, chunks=(chunk_rows, len(union_names)), **compression_kwargs)
    model_id_out = _zc_create_array(root_out, "model_id", shape=(out_row_count,), dtype=np.uint64, chunks=(chunk_rows,), **compression_kwargs)
    run_index_out = _zc_create_array(root_out, "run_index", shape=(out_row_count,), dtype=np.uint16, chunks=(chunk_rows,), **compression_kwargs)

    source_hashes = np.empty(out_row_count, dtype=object)

    # ---- Pass 2: stream rows into the output ------------------------------
    offset = 0
    for run_idx, (r, mask) in enumerate(zip(runs, run_masks)):
        n_kept = int(mask.sum())
        if n_kept == 0:
            continue
        root_in = _open_run(r.path)

        _copy_block(flux_out, root_in["flux"], offset, mask, chunk_rows=chunk_rows)
        if continuum_out is not None:
            if r.has_continuum:
                _copy_block(continuum_out, root_in["continuum"], offset, mask, chunk_rows=chunk_rows)
            # else: leave the NaN-initialised block as-is.

        params_aligned = _aligned_params(root_in, r.param_names, union_names)
        if not mask.all():
            params_aligned = params_aligned[mask]
        params_out[offset : offset + n_kept] = params_aligned
        model_id_out[offset : offset + n_kept] = _compute_model_ids(params_aligned)
        run_index_out[offset : offset + n_kept] = np.uint16(run_idx)
        source_hashes[offset : offset + n_kept] = r.physics_hash or ""

        offset += n_kept

    assert offset == out_row_count, f"row bookkeeping mismatch: wrote {offset}, expected {out_row_count}"

    # ---- Names / ids / scalars --------------------------------------------
    param_names_u32 = _to_u32_param_names(union_names)
    _zc_create_array(
        root_out, "param_names", data=param_names_u32,
        chunks=(len(param_names_u32) if len(param_names_u32) else 1,),
        **compression_kwargs,
    )
    _zc_write_string_array(
        root_out, "source_physics_hash", [str(x) for x in source_hashes.tolist()],
        chunks=chunk_rows, compression_kw=compression_kwargs,
    )

    # Combined physics hash: deterministic over the multiset of source hashes,
    # the wavelength grid and the param columns. Distinct from any single run.
    combined_basis = {
        "hash_basis": "merge-runs-v1",
        "source_physics_hashes": sorted(r.physics_hash for r in runs),
        "param_names": list(union_names),
        "n_lambda": wl_count,
        "wavelength_endpoints": [float(wavelengths[0]), float(wavelengths[-1])],
    }
    combined_physics_hash = hashlib.sha256(
        json.dumps(combined_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    _zc_write_fixed_string_scalar(root_out, "physics_hash", combined_physics_hash, min_width=64, compression_kw=compression_kwargs)
    _zc_write_fixed_string_scalar(root_out, "schema_version", args.schema_version, min_width=16, compression_kw=compression_kwargs)

    _build_provenance_group(
        root_out, runs, run_masks=run_masks, compression_kwargs=compression_kwargs,
        snap_stats=snap_stats if args.filter_bad_snaps else None,
    )

    # ---- Attributes / provenance contract ---------------------------------
    git_commit = _git_commit(project_root)
    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    output_mode_values = sorted(
        {
            str(v)
            for r in runs
            for v in (r.attrs.get("output_mode_values") or [])
        }
    )
    resolved_flux_definition, resolved_flux_unit = infer_flux_metadata(
        output_mode_values,
        explicit_definition=args.flux_definition,
        inherited_definition=_first_meaningful_attr(runs, ("flux_definition",)) or None,
        inherited_unit=_first_meaningful_attr(runs, ("flux_unit",)) or None,
    )

    contract_provenance = {
        "config_hash": _first_meaningful_attr(runs, ("config_hash",)) or combined_physics_hash,
        "grid_definition_hash": _first_meaningful_attr(runs, ("grid_definition_hash",)) or combined_physics_hash,
        "git_commit": git_commit or _first_meaningful_attr(runs, ("git_commit", "git_sha")),
        "turbospectrum_version": _first_meaningful_attr(runs, ("turbospectrum_version", "synthesis_code_version")),
        "linelist_identifier": _first_meaningful_attr(runs, ("linelist_identifier", "linelist_path", "linelist")),
        "linelist_version": _first_meaningful_attr(runs, ("linelist_version",)),
        "atmosphere_model_identifier": _first_meaningful_attr(runs, ("atmosphere_model_identifier", "model_atmosphere_path", "atmosphere")),
        "synthesis_timestamp": created_utc,
        "pipeline_version": _first_meaningful_attr(runs, ("pipeline_version",)) or (f"git:{git_commit[:12]}" if git_commit else ""),
    }
    if not args.allow_incomplete_provenance:
        assert_required_provenance_fields(contract_provenance, context="merge_spectra_runs.py")

    attrs_payload: Dict[str, Any] = {
        "title": "SPICE Synthetic Spectral Grid (multi-run merge)",
        "generator": args.generator,
        "created_utc": created_utc,
        "synthesis_timestamp": created_utc,
        "flux_definition": resolved_flux_definition,
        "wavelength_unit": "angstrom",
        "flux_unit": resolved_flux_unit,
        "physics_hash": combined_physics_hash,
        "git_commit": git_commit,
        "git_sha": git_commit,
        "contact": args.contact,
        "schema_version": args.schema_version,
        "n_models": int(out_row_count),
        "n_lambda": int(wl_count),
        "n_params": int(len(union_names)),
        "continuum_present": bool(any_continuum),
        "runs_merged": len(runs),
        "source_runs": [r.path for r in runs],
        "source_physics_hashes": [r.physics_hash for r in runs],
        "run_rows": [int(r.n_rows) for r in runs],
        "run_rows_kept": [int(m.sum()) for m in run_masks],
        "param_names": list(union_names),
        "strict_params": bool(args.strict_params),
        "dedup_by_model_id": bool(args.dedup_by_model_id),
        "total_input_rows": int(total_input_rows),
        "output_mode_values": output_mode_values,
        "snap_filter_applied": bool(args.filter_bad_snaps),
    }
    if args.filter_bad_snaps:
        attrs_payload["snap_filter_atmosphere_path"] = _norm(args.atmosphere_path)
        attrs_payload["snap_filter_thresholds"] = {
            k: v for k, v in (
                ("max_dteff", args.max_dteff),
                ("max_dlogg", args.max_dlogg),
                ("max_dfeh", args.max_dfeh),
            ) if v is not None
        }
        attrs_payload["run_rows_wrong_snap"] = [
            int(s["n_wrong_snap"]) if s else 0 for s in snap_stats
        ]
    attrs_payload.update({str(k): str(v) for k, v in contract_provenance.items()})
    root_out.attrs.update(attrs_payload)

    try:
        zarr.consolidate_metadata(store)
        print(f"Consolidated Zarr metadata: {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"WARN: zarr.consolidate_metadata failed (non-fatal): {exc}")

    print(
        f"Wrote combined spectra Zarr: {out_path}\n"
        f"  models={out_row_count} lambda={wl_count} params={len(union_names)} "
        f"continuum={'yes' if any_continuum else 'no'} physics_hash={combined_physics_hash[:12]}"
    )


if __name__ == "__main__":
    main()
