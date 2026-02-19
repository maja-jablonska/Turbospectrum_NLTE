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

The merger writes a schema-compliant synthesis Zarr matching DATA_SCHEMA.md:
- flux: (row_count, wavelength_points)
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import zarr


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
    if shard_paths:
        return [os.path.abspath(p) for p in shard_paths]
    if not shard_dir:
        raise ValueError("Provide either --shard or --shard-dir")
    shard_dir = os.path.abspath(shard_dir)
    paths = sorted(str(p) for p in Path(shard_dir).glob("*.zarr"))
    if not paths:
        raise FileNotFoundError(f"No *.zarr shards found in {shard_dir}")
    return paths


def _open_shard(path: str):
    store = _zarr_store(path)
    return zarr.open_group(store=store, mode="r")


def _require_arrays(root, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in root]
    if missing:
        raise KeyError(f"Shard {getattr(root.store, 'path', '?')} missing arrays: {missing}")


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


def _build_params_matrix(columns: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    candidate_order = ["teff", "logg", "feh", "vmicro", "a", "c", "n", "o", "r", "s"]
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

    for name in ("a", "c", "n", "o", "r", "s"):
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
        return "unknown"


def _stable_numeric_digest(values: np.ndarray, dtype: str) -> str:
    arr = np.asarray(values).astype(dtype, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=9.96921e36, posinf=3.4e38, neginf=-3.4e38)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _collect_unique_strings(values: np.ndarray) -> List[str]:
    unique = {str(v).strip() for v in np.asarray(values).tolist()}
    return sorted(v for v in unique if v)


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


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-zarr", required=True, help="Consolidated output Zarr path")
    parser.add_argument("--grid-zarr", default=None, help="Optional original grid Zarr to determine total row_count")

    parser.add_argument("--shard", action="append", default=None, help="Shard Zarr path (repeatable)")
    parser.add_argument("--shard-dir", default=None, help="Directory containing shard *.zarr outputs")

    parser.add_argument("--chunk-rows", type=int, default=32, help="Chunking along the sample dimension")
    parser.add_argument("--compressor", default=None, help="JSON string describing compressor options (cname, clevel, shuffle)")
    parser.add_argument("--schema-version", default="1.0.0", help="DATA_SCHEMA.md schema version")
    parser.add_argument("--physics-hash", default=None, help="Optional override for physics hash")
    parser.add_argument("--contact", default=os.environ.get("SPICE_CONTACT", "unknown"), help="Contact metadata")
    parser.add_argument("--generator", default="turbospectrum_nlte.merge", help="Generator string for metadata")
    parser.add_argument("--flux-definition", default="continuum_normalized", help="Flux definition metadata")
    args = parser.parse_args()

    shards = _list_shards(args.shard, args.shard_dir)
    out_path = os.path.abspath(args.output_zarr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    compressor_cfg: Dict[str, Any] = {}
    if args.compressor:
        compressor_cfg = json.loads(args.compressor)
    compression_kwargs = _zarr_compression_kwargs(compressor_cfg)

    # Determine row_count.
    if args.grid_zarr:
        row_count = _infer_row_count_from_grid(os.path.abspath(args.grid_zarr))
    else:
        row_count = _infer_row_count_from_shards(shards)
    if row_count <= 0:
        raise ValueError("Could not infer a positive row_count; provide --grid-zarr or non-empty shards")

    # Read wavelength from first shard and validate all shards match.
    first = _open_shard(shards[0])
    _require_arrays(first, ["wavelength", "flux", "global_index", "status", "message"])
    wavelengths = np.asarray(first["wavelength"][:], dtype=np.float32)
    wl_count = int(wavelengths.size)
    if wl_count <= 0:
        raise ValueError("Shard wavelength array is empty")

    for p in shards[1:]:
        root = _open_shard(p)
        _require_arrays(root, ["wavelength"])
        other = np.asarray(root["wavelength"][:], dtype=np.float32)
        if other.shape != wavelengths.shape or not np.allclose(other, wavelengths):
            raise ValueError(f"Wavelength mismatch between shards: {shards[0]} vs {p}")

    # Open grid once for metadata/params if provided.
    grid_root = None
    grid_attrs: Dict[str, Any] = {}
    if args.grid_zarr:
        grid_root = zarr.open_group(store=_zarr_store(os.path.abspath(args.grid_zarr)), mode="r")
        try:
            grid_attrs = {str(k): grid_root.attrs[k] for k in grid_root.attrs.keys()}
        except Exception:
            try:
                grid_attrs = {str(k): v for k, v in dict(grid_root.attrs).items()}
            except Exception:
                grid_attrs = {}

    # Preflight integrity checks: every row must be written exactly once.
    seen_rows = np.zeros(row_count, dtype=bool)
    mu_sampling_values: Set[str] = set()
    for p in shards:
        shard = _open_shard(p)
        _require_arrays(shard, ["global_index", "flux", "status", "message"])
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
        preview = missing_rows[:20].tolist()
        suffix = " (truncated)" if missing_rows.size > 20 else ""
        raise ValueError(
            f"Merged dataset would be incomplete: missing {missing_rows.size} row(s), "
            f"examples={preview}{suffix}"
        )

    # Create output store and base datasets.
    store = _zarr_store(out_path)
    root_out = zarr.group(store=store, overwrite=True, zarr_format=3)
    chunk_shape = (min(int(args.chunk_rows), row_count), wl_count)
    root_out.create_array("wavelength", data=wavelengths, chunks=wavelengths.shape, **compression_kwargs)
    flux_out = root_out.create_array("flux", shape=(row_count, wl_count), dtype=np.float32, chunks=chunk_shape, **compression_kwargs)

    # Initialize to NaNs for missing rows.
    flux_out[:] = np.nan

    statuses = ["missing"] * row_count
    messages = [""] * row_count

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
    merged_meta: Dict[str, np.ndarray] = {}
    merged_meta_is_str: Dict[str, bool] = {}
    for name in param_candidate_cols:
        if name in {"teff", "logg", "feh", "a", "c", "n", "o", "r", "s"}:
            merged_meta[name] = np.full(row_count, np.nan, dtype=np.float32)
            merged_meta_is_str[name] = False
        else:
            merged_meta[name] = np.array([""] * row_count, dtype=object)
            merged_meta_is_str[name] = True

    for p in shards:
        shard = _open_shard(p)
        _require_arrays(shard, ["global_index", "flux", "status", "message"])
        gidx = np.asarray(shard["global_index"][:], dtype=np.int64)
        if gidx.size == 0:
            continue

        flux = np.asarray(shard["flux"][:], dtype=np.float32)
        if flux.shape != (gidx.size, wl_count):
            raise ValueError(f"Shard {p} has unexpected flux shape: {flux.shape}")

        # Write shard rows into the consolidated arrays.
        # Use oindex when available for correct fancy indexing.
        try:
            flux_out.oindex[gidx, :] = flux  # type: ignore[attr-defined]
        except Exception:
            for i, gi in enumerate(gidx.tolist()):
                flux_out[int(gi), :] = flux[i]

        shard_status = [str(x) for x in np.asarray(shard["status"][:]).tolist()]
        shard_msg = [str(x) for x in np.asarray(shard["message"][:]).tolist()]
        if len(shard_status) != gidx.size or len(shard_msg) != gidx.size:
            raise ValueError(
                f"Shard {p} has inconsistent status/message lengths: "
                f"status={len(shard_status)} message={len(shard_msg)} rows={gidx.size}"
            )
        for i, gi in enumerate(gidx.tolist()):
            statuses[int(gi)] = shard_status[i]
            messages[int(gi)] = shard_msg[i]

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
                for i, gi in enumerate(gidx.tolist()):
                    merged_meta[name][int(gi)] = vals[i]
            else:
                vals = _to_float32(np.asarray(arr))
                for i, gi in enumerate(gidx.tolist()):
                    merged_meta[name][int(gi)] = vals[i]

    missing_status_rows = [i for i, s in enumerate(statuses) if s == "missing"]
    if missing_status_rows:
        preview = missing_status_rows[:20]
        suffix = " (truncated)" if len(missing_status_rows) > 20 else ""
        raise ValueError(
            f"Merged status vector is incomplete: {len(missing_status_rows)} row(s) still missing, "
            f"examples={preview}{suffix}"
        )

    # Build params matrix from original grid when available; otherwise fall back to merged shard metadata.
    if grid_root is not None:
        grid_cols: Dict[str, np.ndarray] = {}
        for name in ("teff", "logg", "feh", "a", "c", "n", "o", "r", "s"):
            if name in grid_root:
                grid_cols[name] = np.asarray(grid_root[name][:])
        if "turbvel" in grid_root:
            grid_cols["turb"] = np.asarray(grid_root["turbvel"][:])
        elif "t_value" in grid_root:
            grid_cols["turb"] = np.asarray(grid_root["t_value"][:])
        params, param_names = _build_params_matrix(grid_cols)
    else:
        params_source: Dict[str, np.ndarray] = {}
        for name in ("teff", "logg", "feh", "a", "c", "n", "o", "r", "s"):
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
        output_mode_values = _collect_unique_strings(np.asarray(grid_root["output_mode"][:]))
    else:
        output_mode_values = _collect_unique_strings(np.asarray(merged_meta["output_mode"]))

    if grid_root is not None and "calculation_mode" in grid_root:
        calculation_mode_values = _collect_unique_strings(np.asarray(grid_root["calculation_mode"][:]))
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
        chunks=(min(int(args.chunk_rows), row_count), params.shape[1]),
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

    prov = root_out.create_group("provenance")
    _write_string_scalar(
        prov,
        "canonical_config.yaml",
        json.dumps(
            {
                "physics_hash": physics_hash,
                "physics_hash_payload": physics_payload,
                "grid_reference": os.path.abspath(args.grid_zarr) if args.grid_zarr else "unknown",
            },
            indent=2,
            sort_keys=True,
        ),
        compression_kwargs=compression_kwargs,
    )
    _write_string_scalar(prov, "synthesis_config.yaml", "unknown", compression_kwargs=compression_kwargs)
    _write_string_scalar(
        prov,
        "linelist_manifest.json",
        json.dumps({"source": "unknown", "version": "unknown"}, indent=2, sort_keys=True),
        compression_kwargs=compression_kwargs,
    )
    _write_string_scalar(
        prov,
        "atmosphere_manifest.json",
        json.dumps({"source": "unknown", "version": "unknown"}, indent=2, sort_keys=True),
        compression_kwargs=compression_kwargs,
    )
    _write_string_scalar(
        prov,
        "software_manifest.json",
        json.dumps({"generator": args.generator, "git_commit": _git_commit(project_root), "python": sys.version.split()[0]}, indent=2, sort_keys=True),
        compression_kwargs=compression_kwargs,
    )
    _write_string_scalar(
        prov,
        "environment.txt",
        f"python={sys.version}\nOMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}\n",
        compression_kwargs=compression_kwargs,
    )

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

    root_out.attrs.update(
        {
            "title": "SPICE Synthetic Spectral Grid",
            "generator": args.generator,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "flux_definition": args.flux_definition,
            "wavelength_unit": "angstrom",
            "flux_unit": "relative",
            "parameter_units": param_units,
            "physics_hash": physics_hash,
            "git_commit": _git_commit(project_root),
            "contact": args.contact,
            "schema_version": args.schema_version,
            "n_models": int(row_count),
            "n_lambda": int(wl_count),
            "n_params": int(params.shape[1]),
            "shards_merged": len(shards),
            "status_counts": status_counts,
            "output_mode_values": output_mode_values,
            "calculation_mode_values": calculation_mode_values,
            "mu_sampling_values": sorted(mu_sampling_values),
        }
    )

    print(f"Wrote merged spectra Zarr: {out_path}")


if __name__ == "__main__":
    main()
