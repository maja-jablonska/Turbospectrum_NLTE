#!/usr/bin/env python3
"""Merge sharded synthesized-spectra Zarr stores into one consolidated store.

This is intended to merge outputs produced by:
- `scripts/run_turbospectrum_shard.py`

Each shard must contain:
- `wavelength` (1D)
- `global_index` (1D int, mapping shard rows -> original grid rows)
- `flux` (2D: shard_rows x wavelength_points)
- `status` (1D)
- `message` (1D)
Optional in newer shards:
- `continuum` (2D: shard_rows x wavelength_points)

The merger writes a schema-compliant synthesis Zarr matching DATA_SCHEMA.md:
- flux: (row_count, wavelength_points)
- continuum: (row_count, wavelength_points)
- wavelength: (wavelength_points,)
- params: (row_count, n_params)
- param_names: (n_params,)
- model_id: (row_count,)
- physics_hash/schema_version scalars
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import zarr
from provenance_contract import (
    assert_required_provenance_fields,
    canonical_json_sha256,
    compute_grid_definition_hash,
    is_meaningful_provenance_value,
)

PROVENANCE_FILENAMES: Tuple[str, ...] = (
    "canonical_config.yaml",
    "synthesis_config.yaml",
    "linelist_manifest.json",
    "atmosphere_manifest.json",
    "software_manifest.json",
    "environment.txt",
)


def _zarr_store(path: str):
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(path)  # type: ignore[attr-defined]
    from zarr import storage as zstorage  # type: ignore

    if hasattr(zstorage, "DirectoryStore"):
        return zstorage.DirectoryStore(path)  # type: ignore[attr-defined]
    if hasattr(zstorage, "LocalStore"):
        return zstorage.LocalStore(path)  # type: ignore[attr-defined]
    raise AttributeError("Unsupported Zarr version: cannot find DirectoryStore/LocalStore")


def _zarr_compression_kwargs(zarr_compressor_cfg: Mapping[str, Any]):
    if not zarr_compressor_cfg:
        return {}

    cname = zarr_compressor_cfg.get("cname", "zstd")
    clevel = int(zarr_compressor_cfg.get("clevel", 5))
    shuffle_enabled = bool(zarr_compressor_cfg.get("shuffle", True))

    try:
        import zarr.codecs as zc  # type: ignore

        if hasattr(zc, "BloscCodec") and hasattr(zc, "BloscShuffle"):
            shuffle = zc.BloscShuffle.bitshuffle if shuffle_enabled else None
            return {"compressors": [zc.BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)]}
    except Exception:
        pass

    from numcodecs import Blosc  # type: ignore

    return {
        "compressor": Blosc(
            cname=cname,
            clevel=clevel,
            shuffle=Blosc.BITSHUFFLE if shuffle_enabled else Blosc.NOSHUFFLE,
        )
    }


def _list_shards(shard_paths: Sequence[str] | None, shard_dir: str | None) -> List[str]:
    def _norm(path: str) -> str:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    if shard_paths:
        return [_norm(p) for p in shard_paths]
    if not shard_dir:
        raise ValueError("Provide either --shard or --shard-dir")
    raw_shard_dir = shard_dir
    shard_dir = _norm(shard_dir)
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(
            f"Shard directory does not exist: {shard_dir} "
            f"(input='{raw_shard_dir}', cwd='{os.getcwd()}')."
        )
    paths = sorted(str(p) for p in Path(shard_dir).glob("*.zarr"))
    if not paths:
        raise FileNotFoundError(
            f"No *.zarr shards found in {shard_dir}. "
            "If this came from $RUN_ROOT, ensure RUN_ROOT is an absolute path under /scratch."
        )
    return paths


def _open_shard(path: str):
    store = _zarr_store(path)
    return zarr.open_group(store=store, mode="r")


def _require_arrays(root, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in root]
    if missing:
        raise KeyError(f"Shard {getattr(root.store, 'path', '?')} missing arrays: {missing}")


def _validate_shard_structure(path: str) -> np.ndarray:
    root = _open_shard(path)
    _require_arrays(root, ["wavelength", "global_index", "flux", "status", "message"])
    wavelengths = np.asarray(root["wavelength"][:], dtype=np.float32)
    if wavelengths.size <= 0:
        raise ValueError(f"Shard {path} wavelength array is empty")
    return wavelengths


def _infer_row_count_from_grid(grid_zarr: str) -> int:
    store = _zarr_store(grid_zarr)
    root = zarr.open_group(store=store, mode="r")
    if "teff" not in root:
        raise KeyError("Grid Zarr missing required column 'teff'")
    return int(root["teff"].shape[0])


def _infer_row_count_from_shards(shards: Sequence[str]) -> int:
    max_idx = -1
    for p in shards:
        root = _open_shard(p)
        _require_arrays(root, ["global_index"])
        idx = np.asarray(root["global_index"][:], dtype=np.int64)
        if idx.size:
            max_idx = max(max_idx, int(idx.max()))
    return max_idx + 1


def _filter_nonexistent_shards(shards: Sequence[str], *, skip_nonexistent: bool) -> tuple[List[str], List[str]]:
    if not skip_nonexistent:
        return list(shards), []

    kept: List[str] = []
    skipped: List[str] = []
    for path in shards:
        try:
            # Validate store existence/readability once up front so later passes don't fail fast.
            _open_shard(path)
        except FileNotFoundError:
            skipped.append(path)
            continue
        kept.append(path)

    if skipped:
        preview = skipped[:10]
        suffix = " (truncated)" if len(skipped) > 10 else ""
        print(
            f"WARNING: skipping {len(skipped)} non-existent shard(s): {preview}{suffix}",
            file=sys.stderr,
        )
    return kept, skipped


def _filter_invalid_shards(
    shards: Sequence[str], *, skip_invalid: bool
) -> tuple[List[str], List[str], Optional[np.ndarray], Optional[str]]:
    if not skip_invalid:
        return list(shards), [], None, None

    kept: List[str] = []
    skipped_messages: List[str] = []
    reference_wavelengths: Optional[np.ndarray] = None
    reference_path: Optional[str] = None

    for path in shards:
        try:
            wavelengths = _validate_shard_structure(path)
            if reference_wavelengths is None:
                reference_wavelengths = wavelengths
                reference_path = path
            elif (
                wavelengths.shape != reference_wavelengths.shape
                or not np.allclose(wavelengths, reference_wavelengths)
            ):
                raise ValueError(f"Wavelength mismatch vs {reference_path}")
        except Exception as exc:
            skipped_messages.append(f"{path}: {exc}")
            continue
        kept.append(path)

    if skipped_messages:
        preview = skipped_messages[:10]
        suffix = " (truncated)" if len(skipped_messages) > 10 else ""
        print(
            f"WARNING: skipping {len(skipped_messages)} invalid shard(s): {preview}{suffix}",
            file=sys.stderr,
        )
    return kept, skipped_messages, reference_wavelengths, reference_path


def _write_string_scalar(root, name: str, value: str, compression_kwargs: Mapping[str, Any]) -> None:
    import zarr.codecs as zc  # type: ignore
    from zarr.core.dtype.npy.string import VariableLengthUTF8  # type: ignore

    try:
        arr = root.create_array(
            name,
            shape=(),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            **compression_kwargs,
        )
        arr[...] = str(value)
    except Exception:
        arr = root.create_array(
            name,
            shape=(1,),
            dtype=VariableLengthUTF8(),
            serializer=zc.VLenUTF8Codec(),
            chunks=1,
            **compression_kwargs,
        )
        arr[0] = str(value)


def _write_fixed_string_scalar(root, name: str, value: str, min_width: int, compression_kwargs: Mapping[str, Any]) -> None:
    sval = str(value)
    width = max(int(min_width), len(sval), 1)
    try:
        arr = root.create_array(
            name,
            shape=(),
            dtype=f"<U{width}",
            **compression_kwargs,
        )
        arr[...] = sval
    except Exception:
        _write_string_scalar(root, name, sval, compression_kwargs=compression_kwargs)


def _to_u32_param_names(values: Sequence[str]) -> np.ndarray:
    names = [str(v) for v in values]
    too_long = [n for n in names if len(n) > 32]
    if too_long:
        raise ValueError(
            "param_names entries must be <= 32 characters for DATA_SCHEMA.md U32 storage; "
            f"offending values: {too_long[:3]}"
        )
    return np.asarray(names, dtype="<U32")


def _to_float32(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    for i, v in enumerate(values.tolist()):
        try:
            out[i] = np.float32(float(v))
        except Exception:
            s = str(v).strip().lower()
            if s.startswith("t"):
                s = s[1:]
            try:
                out[i] = np.float32(float(s))
            except Exception:
                out[i] = np.nan
    return out


def _ordered_param_names(columns: Mapping[str, np.ndarray]) -> List[str]:
    reserved = {
        "grid_version",
        "lam_min",
        "lam_max",
        "lam_step",
        "output_mode",
        "mode",
        "calculation_mode",
        "turb",
        "turbvel",
        "t_value",
    }
    candidate_order = ["teff", "logg", "feh", "vmicro", "a", "c", "n", "o", "r", "s"]
    extras = sorted(
        name
        for name in columns.keys()
        if name not in reserved and name not in {"teff", "logg", "feh", "a", "c", "n", "o", "r", "s"}
    )
    return candidate_order + extras


def _build_params_matrix(columns: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    candidate_order = _ordered_param_names(columns)
    params_by_name: Dict[str, np.ndarray] = {}

    for name in ("teff", "logg", "feh"):
        if name in columns:
            params_by_name[name] = _to_float32(np.asarray(columns[name]))

    if "turb" in columns:
        params_by_name["vmicro"] = _to_float32(np.asarray(columns["turb"]))
    elif "turbvel" in columns:
        params_by_name["vmicro"] = _to_float32(np.asarray(columns["turbvel"]))
    elif "t_value" in columns:
        params_by_name["vmicro"] = _to_float32(np.asarray(columns["t_value"]))

    for name in candidate_order:
        if name in {"teff", "logg", "feh", "vmicro"}:
            continue
        if name in columns:
            params_by_name[name] = _to_float32(np.asarray(columns[name]))

    param_names = [n for n in candidate_order if n in params_by_name]
    if not param_names:
        raise ValueError("Could not construct params matrix from merged metadata")
    params = np.column_stack([params_by_name[n] for n in param_names]).astype(np.float32, copy=False)
    return params, np.asarray(param_names, dtype=object)


def _compute_model_ids(params: np.ndarray) -> np.ndarray:
    ids = np.zeros(params.shape[0], dtype=np.uint64)
    for i in range(params.shape[0]):
        row = np.nan_to_num(params[i].astype(np.float32, copy=False), nan=9.96921e36, posinf=3.4e38, neginf=-3.4e38)
        digest = hashlib.sha256(row.astype("<f4", copy=False).tobytes()).digest()
        ids[i] = np.uint64(int.from_bytes(digest[:8], "big", signed=False))
    return ids


def _git_commit(project_root: str) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL, timeout=3)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _stable_numeric_digest(values: np.ndarray, dtype: str) -> str:
    arr = np.asarray(values).astype(dtype, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=9.96921e36, posinf=3.4e38, neginf=-3.4e38)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _collect_unique_strings(values: np.ndarray) -> List[str]:
    unique = {str(v).strip() for v in np.asarray(values).tolist()}
    return sorted(v for v in unique if v)


def _collect_attrs(root) -> Dict[str, str]:
    try:
        return {str(k): str(root.attrs[k]) for k in root.attrs.keys()}
    except Exception:
        try:
            return {str(k): str(v) for k, v in dict(root.attrs).items()}
        except Exception:
            return {}


def _read_string_scalar_optional(root, name: str) -> str | None:
    if name not in root:
        return None
    try:
        arr = root[name]
    except Exception:
        return None

    raw: Any = None
    for reader in (
        lambda a: a[...],
        lambda a: a[:],
        lambda a: a[0],
    ):
        try:
            raw = reader(arr)
            break
        except Exception:
            continue
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            raw = raw.item()
        elif raw.size:
            raw = raw.reshape(-1)[0]
        else:
            return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    text = str(raw).strip()
    return text or None


def _meaningful_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "unknown":
        return None
    return text


def _first_meaningful(values: Sequence[Any]) -> str | None:
    for value in values:
        text = _meaningful_text(value)
        if text is not None:
            return text
    return None


def _first_attr_value(attr_sources: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str | None:
    wanted = {str(k).lower() for k in keys}
    for attrs in attr_sources:
        for key in keys:
            if key in attrs:
                text = _meaningful_text(attrs[key])
                if text is not None:
                    return text
        for key, value in attrs.items():
            if str(key).lower() in wanted:
                text = _meaningful_text(value)
                if text is not None:
                    return text
    return None


def _attrs_with_tokens(attr_sources: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> Dict[str, str]:
    wanted = [str(t).lower() for t in tokens]
    out: Dict[str, str] = {}
    for attrs in attr_sources:
        for key, value in attrs.items():
            key_s = str(key)
            if not any(tok in key_s.lower() for tok in wanted):
                continue
            text = _meaningful_text(value)
            if text is None:
                continue
            if key_s not in out:
                out[key_s] = text
    return out


def _physics_relevant_grid_attrs(attrs: Mapping[str, Any]) -> Dict[str, str]:
    tokens = (
        "physics",
        "hash",
        "linelist",
        "line",
        "atmos",
        "opacity",
        "nlte",
        "lte",
        "abund",
        "config",
        "commit",
        "version",
        "turb",
    )
    out: Dict[str, str] = {}
    for key, value in attrs.items():
        key_s = str(key)
        if any(t in key_s.lower() for t in tokens):
            out[key_s] = str(value)
    return out


def _grid_columns_for_hash(grid_root) -> Dict[str, np.ndarray]:
    keys = (
        "teff",
        "logg",
        "feh",
        "turbvel",
        "t_value",
        "lam_min",
        "lam_max",
        "lam_step",
        "a",
        "c",
        "n",
        "o",
        "r",
        "s",
        "output_mode",
        "calculation_mode",
        "grid_version",
    )
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        if key in grid_root:
            out[key] = np.asarray(grid_root[key][:])
    return out


def _compute_merge_physics_hash(
    *,
    wavelengths: np.ndarray,
    params: np.ndarray,
    param_names: Sequence[str],
    output_mode_values: Sequence[str],
    calculation_mode_values: Sequence[str],
    mu_sampling_values: Sequence[str],
    grid_attrs: Mapping[str, Any],
) -> tuple[str, Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "hash_basis": "merge-physics-v1",
        "wavelength_hash": _stable_numeric_digest(wavelengths, dtype="<f8"),
        "params_hash": _stable_numeric_digest(params, dtype="<f4"),
        "param_names": [str(v) for v in param_names],
        "output_mode_values": [str(v) for v in output_mode_values],
        "calculation_mode_values": [str(v) for v in calculation_mode_values],
        "mu_sampling_values": [str(v) for v in mu_sampling_values],
        "grid_attrs": _physics_relevant_grid_attrs(grid_attrs),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def _build_merge_provenance_payload(
    *,
    shard_paths: Sequence[str],
    grid_zarr: str | None,
    args_generator: str,
    chunk_rows: int,
    compressor_cfg: Mapping[str, Any],
    allow_missing: bool,
    expected_models: int,
    merged_models: int,
    physics_hash: str,
    physics_payload: Mapping[str, Any],
    output_mode_values: Sequence[str],
    calculation_mode_values: Sequence[str],
    mu_sampling_values: Sequence[str],
    grid_attrs: Mapping[str, Any],
    grid_provenance: Mapping[str, str],
    shard_attrs: Sequence[Mapping[str, str]],
    shard_provenance: Mapping[str, Sequence[str]],
    git_commit: str,
) -> Dict[str, str]:
    provenance: Dict[str, str] = {}

    def _carry_forward(name: str) -> str | None:
        values = []
        if name in grid_provenance:
            values.append(grid_provenance[name])
        values.extend(shard_provenance.get(name, []))
        return _first_meaningful(values)

    attr_sources: List[Mapping[str, Any]] = [grid_attrs]
    attr_sources.extend(shard_attrs)
    grid_reference = os.path.abspath(grid_zarr) if grid_zarr else "not_provided"

    canonical_payload = {
        "physics_hash": physics_hash,
        "physics_hash_payload": dict(physics_payload),
        "grid_reference": grid_reference,
        "source_shards": [os.path.abspath(p) for p in shard_paths],
        "output_mode_values": [str(v) for v in output_mode_values],
        "calculation_mode_values": [str(v) for v in calculation_mode_values],
        "mu_sampling_values": [str(v) for v in mu_sampling_values],
        "grid_physics_attrs": _physics_relevant_grid_attrs(grid_attrs),
    }
    synthesis_payload = {
        "merge_tool": "scripts/merge_spectra_shards.py",
        "generator": args_generator,
        "grid_reference": grid_reference,
        "source_shards": [os.path.abspath(p) for p in shard_paths],
        "chunk_rows": int(chunk_rows),
        "compressor": dict(compressor_cfg),
        "allow_missing": bool(allow_missing),
        "expected_models": int(expected_models),
        "merged_models": int(merged_models),
        "mu_sampling_values": [str(v) for v in mu_sampling_values],
    }

    linelist_source = _first_attr_value(
        attr_sources,
        keys=("linelist_path", "linelist_files", "linelist_file_path", "linelist"),
    )
    linelist_version = _first_attr_value(
        attr_sources,
        keys=("linelist_version", "line_version", "linelist_rev"),
    )
    linelist_manifest = {
        "source": linelist_source or "",
        "version": linelist_version or "",
        "attrs": _attrs_with_tokens(attr_sources, tokens=("linelist", "line")),
    }

    atmosphere_source = _first_attr_value(
        attr_sources,
        keys=("model_atmosphere_path", "atmosphere_path", "atmosphere_grid", "atmosphere"),
    )
    atmosphere_geometry = _first_attr_value(attr_sources, keys=("geometry", "atmosphere_geometry"))
    atmosphere_version = _first_attr_value(attr_sources, keys=("atmosphere_version", "model_version"))
    atmosphere_manifest = {
        "source": atmosphere_source or "",
        "geometry": atmosphere_geometry or "",
        "version": atmosphere_version or "",
        "attrs": _attrs_with_tokens(attr_sources, tokens=("atmos", "model")),
    }

    software_manifest = {
        "generator": args_generator,
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "compiler": _first_attr_value(attr_sources, keys=("compiler",)) or "",
        "nlte": _first_attr_value(attr_sources, keys=("nlte", "nlte_enabled")) or "",
        "attrs": _attrs_with_tokens(attr_sources, tokens=("version", "commit", "compiler", "software")),
    }

    environment_text = "\n".join(
        [
            f"python={sys.version}",
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}",
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '')}",
            f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '')}",
            f"VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS', '')}",
        ]
    )

    provenance["canonical_config.yaml"] = _carry_forward("canonical_config.yaml") or json.dumps(
        canonical_payload,
        indent=2,
        sort_keys=True,
    )
    provenance["synthesis_config.yaml"] = _carry_forward("synthesis_config.yaml") or json.dumps(
        synthesis_payload,
        indent=2,
        sort_keys=True,
    )
    provenance["linelist_manifest.json"] = _carry_forward("linelist_manifest.json") or json.dumps(
        linelist_manifest,
        indent=2,
        sort_keys=True,
    )
    provenance["atmosphere_manifest.json"] = _carry_forward("atmosphere_manifest.json") or json.dumps(
        atmosphere_manifest,
        indent=2,
        sort_keys=True,
    )
    provenance["software_manifest.json"] = _carry_forward("software_manifest.json") or json.dumps(
        software_manifest,
        indent=2,
        sort_keys=True,
    )
    provenance["environment.txt"] = _carry_forward("environment.txt") or environment_text
    return provenance


def main() -> None:
    def _norm(path: str) -> str:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-zarr", required=True, help="Consolidated output Zarr path")
    parser.add_argument("--grid-zarr", default=None, help="Optional original grid Zarr to determine total row_count")

    parser.add_argument("--shard", action="append", default=None, help="Shard Zarr path (repeatable)")
    parser.add_argument("--shard-dir", default=None, help="Directory containing shard *.zarr outputs")

    parser.add_argument("--chunk-rows", type=int, default=32, help="Chunking along the sample dimension")
    parser.add_argument("--compressor", default=None, help="JSON string describing compressor options (cname, clevel, shuffle)")
    parser.add_argument(
        "--tmp-dir",
        default=os.environ.get("TMPDIR"),
        help="Optional tmp directory to force mkstemp/tempfile writes (also sets TMPDIR/TMP/TEMP).",
    )
    parser.add_argument("--schema-version", default="1.0.0", help="DATA_SCHEMA.md schema version")
    parser.add_argument("--physics-hash", default=None, help="Optional override for physics hash")
    parser.add_argument("--contact", default=os.environ.get("SPICE_CONTACT", "unknown"), help="Contact metadata")
    parser.add_argument("--generator", default="turbospectrum_nlte.merge", help="Generator string for metadata")
    parser.add_argument("--flux-definition", default="continuum_normalized", help="Flux definition metadata")
    parser.add_argument(
        "--allow-incomplete-provenance",
        action="store_true",
        help="Allow writing merged outputs even when required provenance contract fields are missing.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Allow partial merges when some shards are missing. "
            "Output will contain only rows present in available shards."
        ),
    )
    parser.add_argument(
        "--skip-nonexistent-shards",
        action="store_true",
        help=(
            "Skip shard paths that do not exist at merge time (e.g. stale shard lists "
            "or interrupted reruns)."
        ),
    )
    parser.add_argument(
        "--skip-invalid-shards",
        action="store_true",
        help=(
            "Skip shard stores that exist but fail structural validation "
            "(for example missing required arrays or mismatched wavelength grids)."
        ),
    )
    args = parser.parse_args()

    if args.tmp_dir:
        tmp_dir = _norm(args.tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        os.environ["TMPDIR"] = tmp_dir
        os.environ["TMP"] = tmp_dir
        os.environ["TEMP"] = tmp_dir
        tempfile.tempdir = tmp_dir

    shards = _list_shards(args.shard, args.shard_dir)
    shards, skipped_nonexistent_shards = _filter_nonexistent_shards(
        shards,
        skip_nonexistent=bool(args.skip_nonexistent_shards),
    )
    shards, skipped_invalid_shards, validated_wavelengths, _validated_wavelength_path = _filter_invalid_shards(
        shards,
        skip_invalid=bool(args.skip_invalid_shards),
    )
    if not shards:
        raise FileNotFoundError(
            "No readable shard stores remain after filtering. "
            "Either point --shard-dir to existing shard outputs or disable shard skipping flags."
        )
    effective_allow_missing = bool(
        args.allow_missing or args.skip_nonexistent_shards or args.skip_invalid_shards
    )
    if (args.skip_nonexistent_shards or args.skip_invalid_shards) and not args.allow_missing:
        print(
            "WARNING: shard skipping enables partial merge semantics "
            "(missing rows will be omitted).",
            file=sys.stderr,
        )
    out_path = _norm(args.output_zarr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    compressor_cfg: Dict[str, Any] = {}
    if args.compressor:
        compressor_cfg = json.loads(args.compressor)
    compression_kwargs = _zarr_compression_kwargs(compressor_cfg)

    # Determine row_count.
    if args.grid_zarr:
        row_count = _infer_row_count_from_grid(_norm(args.grid_zarr))
    else:
        row_count = _infer_row_count_from_shards(shards)
    if row_count <= 0:
        raise ValueError("Could not infer a positive row_count; provide --grid-zarr or non-empty shards")

    # Read wavelength from first shard and validate all shards match.
    if validated_wavelengths is not None:
        wavelengths = validated_wavelengths
    else:
        first = _open_shard(shards[0])
        _require_arrays(first, ["wavelength", "flux", "global_index", "status", "message"])
        wavelengths = np.asarray(first["wavelength"][:], dtype=np.float32)
    wl_count = int(wavelengths.size)
    if wl_count <= 0:
        raise ValueError("Shard wavelength array is empty")

    if validated_wavelengths is None:
        for p in shards[1:]:
            root = _open_shard(p)
            _require_arrays(root, ["wavelength"])
            other = np.asarray(root["wavelength"][:], dtype=np.float32)
            if other.shape != wavelengths.shape or not np.allclose(other, wavelengths):
                raise ValueError(f"Wavelength mismatch between shards: {shards[0]} vs {p}")

    # Open grid once for metadata/params if provided.
    grid_root = None
    grid_attrs: Dict[str, Any] = {}
    grid_provenance: Dict[str, str] = {}
    if args.grid_zarr:
        grid_root = zarr.open_group(store=_zarr_store(_norm(args.grid_zarr)), mode="r")
        try:
            grid_attrs = {str(k): grid_root.attrs[k] for k in grid_root.attrs.keys()}
        except Exception:
            try:
                grid_attrs = {str(k): v for k, v in dict(grid_root.attrs).items()}
            except Exception:
                grid_attrs = {}
        if "provenance" in grid_root:
            grid_prov_group = grid_root["provenance"]
            for name in PROVENANCE_FILENAMES:
                value = _read_string_scalar_optional(grid_prov_group, name)
                if value is not None:
                    grid_provenance[name] = value

    # Preflight integrity checks: every row must be written exactly once.
    seen_rows = np.zeros(row_count, dtype=bool)
    mu_sampling_values: Set[str] = set()
    shard_attr_snapshots: List[Dict[str, str]] = []
    shard_provenance_candidates: Dict[str, List[str]] = {name: [] for name in PROVENANCE_FILENAMES}
    for p in shards:
        try:
            shard = _open_shard(p)
            _require_arrays(shard, ["global_index", "flux", "status", "message"])
        except Exception as exc:
            if effective_allow_missing:
                print(f"WARNING: skipping shard during preflight: {p}: {exc}", file=sys.stderr)
                continue
            raise
        shard_attrs = _collect_attrs(shard)
        if shard_attrs:
            shard_attr_snapshots.append(shard_attrs)
        if "provenance" in shard:
            shard_prov_group = shard["provenance"]
            for name in PROVENANCE_FILENAMES:
                value = _read_string_scalar_optional(shard_prov_group, name)
                if value is not None:
                    shard_provenance_candidates[name].append(value)
        gidx = np.asarray(shard["global_index"][:], dtype=np.int64)
        if gidx.ndim != 1:
            raise ValueError(f"Shard {p} global_index must be 1D, got shape={gidx.shape}")
        if gidx.size == 0:
            continue
        if np.any(gidx < 0):
            bad = gidx[gidx < 0][:10].tolist()
            raise ValueError(f"Shard {p} contains negative global_index values: {bad}")
        if int(gidx.max()) >= row_count:
            raise ValueError(f"Shard {p} contains global_index beyond row_count={row_count}")

        uniq, counts = np.unique(gidx, return_counts=True)
        dup_within = uniq[counts > 1]
        if dup_within.size:
            raise ValueError(
                f"Shard {p} contains duplicate global_index values within shard: "
                f"{dup_within[:20].tolist()}"
            )

        overlap = uniq[seen_rows[uniq]]
        if overlap.size:
            raise ValueError(
                f"Duplicate global_index encountered across shards while reading {p}: "
                f"{overlap[:20].tolist()}"
            )
        seen_rows[uniq] = True

        mu_sampling_attr = shard.attrs.get("mu_sampling")
        if mu_sampling_attr is not None:
            sval = str(mu_sampling_attr).strip()
            if sval:
                mu_sampling_values.add(sval)

    missing_rows = np.flatnonzero(~seen_rows)
    if missing_rows.size:
        if not effective_allow_missing:
            preview = missing_rows[:20].tolist()
            suffix = " (truncated)" if missing_rows.size > 20 else ""
            raise ValueError(
                f"Merged dataset would be incomplete: missing {missing_rows.size} row(s), "
                f"examples={preview}{suffix}"
            )
        print(
            f"WARNING: partial merge enabled; omitting {missing_rows.size} missing row(s) "
            f"from output."
        )

    selected_global_indices = (
        np.flatnonzero(seen_rows).astype(np.int64)
        if effective_allow_missing
        else np.arange(row_count, dtype=np.int64)
    )
    out_row_count = int(selected_global_indices.size)
    if out_row_count <= 0:
        raise ValueError("No valid shard rows available to merge")

    global_to_out = np.full(row_count, -1, dtype=np.int64)
    global_to_out[selected_global_indices] = np.arange(out_row_count, dtype=np.int64)

    # Create output store and base datasets.
    store = _zarr_store(out_path)
    root_out = zarr.group(store=store, overwrite=True, zarr_format=3)
    chunk_shape = (min(int(args.chunk_rows), out_row_count), wl_count)
    root_out.create_array("wavelength", data=wavelengths, chunks=wavelengths.shape, **compression_kwargs)
    root_out.create_array(
        "global_index",
        data=selected_global_indices.astype(np.int64, copy=False),
        chunks=(min(int(args.chunk_rows), out_row_count),),
        **compression_kwargs,
    )
    flux_out = root_out.create_array("flux", shape=(out_row_count, wl_count), dtype=np.float32, chunks=chunk_shape, **compression_kwargs)
    continuum_out = root_out.create_array("continuum", shape=(out_row_count, wl_count), dtype=np.float32, chunks=chunk_shape, **compression_kwargs)
    mu_chunk = (min(int(args.chunk_rows), out_row_count) if out_row_count else 1,)
    mu_selected_out = root_out.create_array("mu_selected", shape=(out_row_count,), dtype=np.float32, chunks=mu_chunk, **compression_kwargs)
    mu_selected_index_out = root_out.create_array("mu_selected_index", shape=(out_row_count,), dtype=np.int16, chunks=mu_chunk, **compression_kwargs)

    # Initialize to NaNs for missing rows.
    flux_out[:] = np.nan
    continuum_out[:] = np.nan
    mu_selected_out[:] = np.nan
    mu_selected_index_out[:] = np.int16(-1)

    statuses = ["missing"] * out_row_count
    messages = [""] * out_row_count
    saw_continuum = False
    saw_mu_selected = False
    saw_mu_selected_index = False

    # Collect parameter metadata columns for schema-compliant params matrix.
    param_candidate_cols = [
        "teff",
        "logg",
        "feh",
        "turb",
        "turbvel",
        "t_value",
        "a",
        "c",
        "n",
        "o",
        "r",
        "s",
        "output_mode",
        "calculation_mode",
    ]
    if grid_root is not None:
        extra_grid_cols = sorted(
            name
            for name in grid_root.keys()
            if name not in {
                "teff",
                "logg",
                "feh",
                "turb",
                "turbvel",
                "t_value",
                "a",
                "c",
                "n",
                "o",
                "r",
                "s",
                "lam_min",
                "lam_max",
                "lam_step",
                "output_mode",
                "mode",
                "calculation_mode",
                "grid_version",
            }
        )
        param_candidate_cols.extend(extra_grid_cols)
    merged_meta: Dict[str, np.ndarray] = {}
    merged_meta_is_str: Dict[str, bool] = {}
    for name in param_candidate_cols:
        if name in {"teff", "logg", "feh", "a", "c", "n", "o", "r", "s"} or name.isalpha():
            merged_meta[name] = np.full(out_row_count, np.nan, dtype=np.float32)
            merged_meta_is_str[name] = False
        else:
            merged_meta[name] = np.array([""] * out_row_count, dtype=object)
            merged_meta_is_str[name] = True

    for p in shards:
        try:
            shard = _open_shard(p)
            _require_arrays(shard, ["global_index", "flux", "status", "message"])
        except Exception as exc:
            if effective_allow_missing:
                print(f"WARNING: skipping shard during write: {p}: {exc}", file=sys.stderr)
                continue
            raise
        gidx = np.asarray(shard["global_index"][:], dtype=np.int64)
        if gidx.size == 0:
            continue

        flux = np.asarray(shard["flux"][:], dtype=np.float32)
        if flux.shape != (gidx.size, wl_count):
            raise ValueError(f"Shard {p} has unexpected flux shape: {flux.shape}")

        local_idx = global_to_out[gidx]
        if np.any(local_idx < 0):
            bad = gidx[local_idx < 0][:20].tolist()
            raise ValueError(f"Shard {p} maps to rows excluded from partial merge: {bad}")

        # Write shard rows into the consolidated arrays.
        # Use oindex when available for correct fancy indexing.
        try:
            flux_out.oindex[local_idx, :] = flux  # type: ignore[attr-defined]
        except Exception:
            for i, li in enumerate(local_idx.tolist()):
                flux_out[int(li), :] = flux[i]

        if "continuum" in shard:
            continuum = np.asarray(shard["continuum"][:], dtype=np.float32)
            if continuum.shape != (gidx.size, wl_count):
                raise ValueError(f"Shard {p} has unexpected continuum shape: {continuum.shape}")
            try:
                continuum_out.oindex[local_idx, :] = continuum  # type: ignore[attr-defined]
            except Exception:
                for i, li in enumerate(local_idx.tolist()):
                    continuum_out[int(li), :] = continuum[i]
            saw_continuum = True

        shard_status = [str(x) for x in np.asarray(shard["status"][:]).tolist()]
        shard_msg = [str(x) for x in np.asarray(shard["message"][:]).tolist()]
        if len(shard_status) != gidx.size or len(shard_msg) != gidx.size:
            raise ValueError(
                f"Shard {p} has inconsistent status/message lengths: "
                f"status={len(shard_status)} message={len(shard_msg)} rows={gidx.size}"
            )
        for i, li in enumerate(local_idx.tolist()):
            statuses[int(li)] = shard_status[i]
            messages[int(li)] = shard_msg[i]

        if "mu_selected" in shard:
            shard_mu = np.asarray(shard["mu_selected"][:], dtype=np.float32)
            if shard_mu.ndim != 1 or shard_mu.shape[0] != gidx.size:
                raise ValueError(
                    f"Shard {p} mu_selected shape mismatch: expected ({gidx.size},), got {shard_mu.shape}"
                )
            try:
                mu_selected_out.oindex[local_idx] = shard_mu  # type: ignore[attr-defined]
            except Exception:
                for i, li in enumerate(local_idx.tolist()):
                    mu_selected_out[int(li)] = shard_mu[i]
            saw_mu_selected = True

        if "mu_selected_index" in shard:
            shard_mu_idx = np.asarray(shard["mu_selected_index"][:], dtype=np.int16)
            if shard_mu_idx.ndim != 1 or shard_mu_idx.shape[0] != gidx.size:
                raise ValueError(
                    f"Shard {p} mu_selected_index shape mismatch: expected ({gidx.size},), got {shard_mu_idx.shape}"
                )
            try:
                mu_selected_index_out.oindex[local_idx] = shard_mu_idx  # type: ignore[attr-defined]
            except Exception:
                for i, li in enumerate(local_idx.tolist()):
                    mu_selected_index_out[int(li)] = shard_mu_idx[i]
            saw_mu_selected_index = True

        # Merge metadata columns if present.
        for name in param_candidate_cols:
            if name not in shard:
                continue
            arr = np.asarray(shard[name][:])
            if arr.ndim != 1 or arr.shape[0] != gidx.size:
                raise ValueError(
                    f"Shard {p} metadata column '{name}' shape mismatch: "
                    f"expected ({gidx.size},), got {arr.shape}"
                )

            if merged_meta_is_str[name]:
                # NumPy 2 may use StringDType, which can fail on astype(str).
                vals = [str(x) for x in np.asarray(arr).tolist()]
                for i, li in enumerate(local_idx.tolist()):
                    merged_meta[name][int(li)] = vals[i]
            else:
                vals = _to_float32(np.asarray(arr))
                for i, li in enumerate(local_idx.tolist()):
                    merged_meta[name][int(li)] = vals[i]

    missing_status_rows = [i for i, s in enumerate(statuses) if s == "missing"]
    if missing_status_rows:
        preview = missing_status_rows[:20]
        suffix = " (truncated)" if len(missing_status_rows) > 20 else ""
        if effective_allow_missing:
            print(
                f"WARNING: leaving {len(missing_status_rows)} merged row(s) with missing status "
                f"after shard skips, examples={preview}{suffix}",
                file=sys.stderr,
            )
        else:
            raise ValueError(
                f"Merged status vector is incomplete: {len(missing_status_rows)} row(s) still missing, "
                f"examples={preview}{suffix}"
            )

    # Build params matrix from original grid when available; otherwise fall back to merged shard metadata.
    if grid_root is not None:
        grid_cols: Dict[str, np.ndarray] = {}
        for name in grid_root.keys():
            if name in grid_root:
                if name in {"lam_min", "lam_max", "lam_step", "output_mode", "mode", "calculation_mode", "grid_version"}:
                    continue
                values = np.asarray(grid_root[name][:])
                if effective_allow_missing:
                    values = values[selected_global_indices]
                grid_cols[name] = values
        if "turbvel" in grid_root:
            turb_vals = np.asarray(grid_root["turbvel"][:])
            if effective_allow_missing:
                turb_vals = turb_vals[selected_global_indices]
            grid_cols["turb"] = turb_vals
        elif "t_value" in grid_root:
            turb_vals = np.asarray(grid_root["t_value"][:])
            if effective_allow_missing:
                turb_vals = turb_vals[selected_global_indices]
            grid_cols["turb"] = turb_vals
        params, param_names = _build_params_matrix(grid_cols)
    else:
        params_source: Dict[str, np.ndarray] = {}
        for name in merged_meta.keys():
            if name in merged_meta:
                params_source[name] = np.asarray(merged_meta[name])
        if "turb" in merged_meta and any(str(x).strip() for x in merged_meta["turb"].tolist()):
            params_source["turb"] = np.asarray(merged_meta["turb"])
        elif "turbvel" in merged_meta and any(str(x).strip() for x in merged_meta["turbvel"].tolist()):
            params_source["turb"] = np.asarray(merged_meta["turbvel"])
        elif "t_value" in merged_meta and any(str(x).strip() for x in merged_meta["t_value"].tolist()):
            params_source["turb"] = np.asarray(merged_meta["t_value"])
        params, param_names = _build_params_matrix(params_source)

    model_id = _compute_model_ids(params)
    param_name_list = [str(x) for x in param_names.tolist()]

    if grid_root is not None and "output_mode" in grid_root:
        output_mode_data = np.asarray(grid_root["output_mode"][:])
        if effective_allow_missing:
            output_mode_data = output_mode_data[selected_global_indices]
        output_mode_values = _collect_unique_strings(output_mode_data)
    else:
        output_mode_values = _collect_unique_strings(np.asarray(merged_meta["output_mode"]))

    if grid_root is not None and "calculation_mode" in grid_root:
        calculation_mode_data = np.asarray(grid_root["calculation_mode"][:])
        if effective_allow_missing:
            calculation_mode_data = calculation_mode_data[selected_global_indices]
        calculation_mode_values = _collect_unique_strings(calculation_mode_data)
    else:
        calculation_mode_values = _collect_unique_strings(np.asarray(merged_meta["calculation_mode"]))

    if args.physics_hash:
        physics_hash = args.physics_hash
        physics_payload: Dict[str, Any] = {"hash_basis": "cli-override", "physics_hash": physics_hash}
    else:
        physics_hash, physics_payload = _compute_merge_physics_hash(
            wavelengths=wavelengths,
            params=params,
            param_names=param_name_list,
            output_mode_values=output_mode_values,
            calculation_mode_values=calculation_mode_values,
            mu_sampling_values=sorted(mu_sampling_values),
            grid_attrs=grid_attrs,
        )
    git_commit = _git_commit(project_root)
    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    param_names_u32 = _to_u32_param_names(param_name_list)
    root_out.create_array(
        "param_names",
        data=param_names_u32,
        chunks=(min(max(1, params.shape[1]), len(param_names_u32)) if len(param_names_u32) else 1,),
        **compression_kwargs,
    )
    root_out.create_array(
        "params",
        data=params.astype(np.float32, copy=False),
        chunks=(min(int(args.chunk_rows), out_row_count), params.shape[1]),
        **compression_kwargs,
    )
    root_out.create_array(
        "model_id",
        data=model_id.astype(np.uint64, copy=False),
        chunks=(min(int(args.chunk_rows), len(model_id)) if len(model_id) else 1,),
        **compression_kwargs,
    )
    _write_fixed_string_scalar(root_out, "physics_hash", physics_hash, min_width=64, compression_kwargs=compression_kwargs)
    _write_fixed_string_scalar(root_out, "schema_version", args.schema_version, min_width=16, compression_kwargs=compression_kwargs)

    provenance_payload = _build_merge_provenance_payload(
        shard_paths=shards,
        grid_zarr=args.grid_zarr,
        args_generator=args.generator,
        chunk_rows=int(args.chunk_rows),
        compressor_cfg=compressor_cfg,
        allow_missing=bool(effective_allow_missing),
        expected_models=int(row_count),
        merged_models=int(out_row_count),
        physics_hash=physics_hash,
        physics_payload=physics_payload,
        output_mode_values=output_mode_values,
        calculation_mode_values=calculation_mode_values,
        mu_sampling_values=sorted(mu_sampling_values),
        grid_attrs=grid_attrs,
        grid_provenance=grid_provenance,
        shard_attrs=shard_attr_snapshots,
        shard_provenance=shard_provenance_candidates,
        git_commit=git_commit,
    )
    prov = root_out.create_group("provenance")
    for name in PROVENANCE_FILENAMES:
        _write_string_scalar(
            prov,
            name,
            provenance_payload[name],
            compression_kwargs=compression_kwargs,
        )

    attr_sources: List[Mapping[str, Any]] = [grid_attrs]
    attr_sources.extend(shard_attr_snapshots)
    try:
        linelist_manifest = json.loads(provenance_payload.get("linelist_manifest.json", "{}"))
    except Exception:
        linelist_manifest = {}
    try:
        atmosphere_manifest = json.loads(provenance_payload.get("atmosphere_manifest.json", "{}"))
    except Exception:
        atmosphere_manifest = {}
    try:
        software_manifest = json.loads(provenance_payload.get("software_manifest.json", "{}"))
    except Exception:
        software_manifest = {}

    config_hash = _first_attr_value(attr_sources, keys=("config_hash",))
    if not is_meaningful_provenance_value(config_hash):
        config_hash = canonical_json_sha256({"synthesis_config.yaml": provenance_payload.get("synthesis_config.yaml", "")})

    grid_definition_hash = _first_attr_value(attr_sources, keys=("grid_definition_hash",))
    if (not is_meaningful_provenance_value(grid_definition_hash)) and grid_root is not None:
        grid_definition_hash = compute_grid_definition_hash(_grid_columns_for_hash(grid_root))
    if not is_meaningful_provenance_value(grid_definition_hash):
        grid_definition_hash = canonical_json_sha256(
            {
                "grid_reference": args.grid_zarr or "not_provided",
                "row_count": int(row_count),
                "physics_hash": physics_hash,
            }
        )

    turbospectrum_version = _first_attr_value(attr_sources, keys=("turbospectrum_version", "synthesis_code_version"))
    if not is_meaningful_provenance_value(turbospectrum_version):
        turbospectrum_version = str(software_manifest.get("synthesis_code_version", "")).strip()

    linelist_identifier = _first_attr_value(
        attr_sources,
        keys=("linelist_identifier", "linelist_path", "linelist_files", "linelist"),
    )
    if not is_meaningful_provenance_value(linelist_identifier):
        linelist_identifier = str(linelist_manifest.get("source", "")).strip()

    linelist_version = _first_attr_value(attr_sources, keys=("linelist_version", "line_version", "linelist_rev"))
    if not is_meaningful_provenance_value(linelist_version):
        linelist_version = str(linelist_manifest.get("version", "")).strip()

    atmosphere_model_identifier = _first_attr_value(
        attr_sources,
        keys=("atmosphere_model_identifier", "model_atmosphere_path", "atmosphere_path", "atmosphere_grid", "atmosphere"),
    )
    if not is_meaningful_provenance_value(atmosphere_model_identifier):
        atmosphere_model_identifier = str(
            atmosphere_manifest.get("path") or atmosphere_manifest.get("source") or ""
        ).strip()

    pipeline_version = _first_attr_value(attr_sources, keys=("pipeline_version", "spice_version"))
    if not is_meaningful_provenance_value(pipeline_version):
        pipeline_version = str(software_manifest.get("pipeline_version") or software_manifest.get("spice_version") or "").strip()
    if (not is_meaningful_provenance_value(pipeline_version)) and is_meaningful_provenance_value(git_commit):
        pipeline_version = f"git:{git_commit[:12]}"

    contract_provenance: Dict[str, str] = {
        "config_hash": str(config_hash or ""),
        "grid_definition_hash": str(grid_definition_hash or ""),
        "git_commit": str(git_commit or ""),
        "turbospectrum_version": str(turbospectrum_version or ""),
        "linelist_identifier": str(linelist_identifier or ""),
        "linelist_version": str(linelist_version or ""),
        "atmosphere_model_identifier": str(atmosphere_model_identifier or ""),
        "synthesis_timestamp": created_utc,
        "pipeline_version": str(pipeline_version or ""),
    }
    if not args.allow_incomplete_provenance:
        assert_required_provenance_fields(contract_provenance, context="merge_spectra_shards.py")

    status_counts: Dict[str, int] = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1

    param_units: Dict[str, str] = {}
    for name in param_name_list:
        if name == "teff":
            param_units[name] = "K"
        elif name in {"logg", "feh", "a", "c", "n", "o", "r", "s"}:
            param_units[name] = "dex"
        elif name == "vmicro":
            param_units[name] = "km/s"
        else:
            param_units[name] = ""

    attrs_payload: Dict[str, Any] = {
        "title": "SPICE Synthetic Spectral Grid",
        "generator": args.generator,
        "created_utc": created_utc,
        "synthesis_timestamp": created_utc,
        "flux_definition": args.flux_definition,
        "wavelength_unit": "angstrom",
        "flux_unit": "relative",
        "parameter_units": param_units,
        "physics_hash": physics_hash,
        "git_commit": git_commit,
        "git_sha": git_commit,
        "contact": args.contact,
        "schema_version": args.schema_version,
        "n_models": int(out_row_count),
        "n_lambda": int(wl_count),
        "n_params": int(params.shape[1]),
        "shards_merged": len(shards),
        "status_counts": status_counts,
        "continuum_present": bool(saw_continuum),
        "output_mode_values": output_mode_values,
        "calculation_mode_values": calculation_mode_values,
        "mu_sampling_values": sorted(mu_sampling_values),
        "mu_selected_present": bool(saw_mu_selected),
        "mu_selected_index_present": bool(saw_mu_selected_index),
        "allow_missing": bool(effective_allow_missing),
        "allow_missing_requested": bool(args.allow_missing),
        "skip_nonexistent_shards": bool(args.skip_nonexistent_shards),
        "skip_invalid_shards": bool(args.skip_invalid_shards),
        "skipped_nonexistent_shards": int(len(skipped_nonexistent_shards)),
        "skipped_invalid_shards": int(len(skipped_invalid_shards)),
        "expected_models": int(row_count),
        "missing_models": int(row_count - out_row_count),
    }
    attrs_payload.update({str(k): str(v) for k, v in contract_provenance.items()})
    root_out.attrs.update(attrs_payload)

    print(f"Wrote merged spectra Zarr: {out_path}")


if __name__ == "__main__":
    main()
